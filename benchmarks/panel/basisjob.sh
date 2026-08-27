#!/usr/bin/env bash
# a1_basis_k30.npz -- the ancestry basis Layer 1's k-sweep consumes.
#
# Y = G U, plus the per-variant integer moments s_v and q_v, where U is an
# orthonormal basis of [1 | PCs] via QR.
#
# THE INTERCEPT IS DELIBERATELY IN U. This is the host float64 post-hoc path,
# where the projection does the centring: r_adj = (S - Y_i.Y_j) / sqrt(q_adj_i
# q_adj_j) with q_adj = q_v - ||Y||^2, and removing the mean is part of what the
# projection must do. The FUSED GPU path is the opposite -- there the epilogue
# already centres from s_v, and including the intercept produced catastrophic
# fp32 cancellation (max|dr| = 1.6e-3 at K=1). Same maths, different arithmetic;
# do not "harmonise" these.
#
# Recipe held identical to the original run so the old -> new comparison
# isolates the panel.
#
# NO `set -x`: R2 credentials are in this environment.
set -uo pipefail
D=/root/data; mkdir -p $D
exec > >(tee -a /root/basis.log) 2>&1
say(){ echo "===== $(date -u +%H:%M:%S) basis $* ====="; }
fail(){ echo "BASIS_FAIL: $*"; exit 1; }
for v in r2_endpoint r2_access_key r2_secret; do
    if [[ -n "${!v:-}" ]]; then echo "  $v present? YES"; else fail "no $v"; fi
done

say "deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq curl git >/dev/null 2>&1
command -v rclone >/dev/null || curl -sSL https://rclone.org/install.sh | bash >/dev/null 2>&1
pip install -q --no-input --break-system-packages numpy "pandas>=2.0" "scipy>=1.10" \
    >/dev/null 2>&1 || fail "pip"
[ -d /root/cugen_ld ] || git clone -q --branch ld-rowblock --depth 3 \
    https://github.com/bschilder/cugen.git /root/cugen_ld || fail "clone"

R2=( --s3-provider Cloudflare --s3-endpoint "$r2_endpoint"
     --s3-access-key-id "$r2_access_key" --s3-secret-access-key "$r2_secret"
     --s3-no-check-bucket )
SRC=":s3:smb-data-prod-scratch/cugen/1kg-30x-grch38"

say "pull PCs and per-chromosome .cugen"
rclone "${R2[@]}" copyto "$SRC/derived/pcs_no_lrld_center.npy" \
    "$D/pcs_no_lrld_center.npy" --stats-one-line >/dev/null 2>&1 \
    || fail "PCs not on R2 yet -- run pcjob first"
for c in $(seq 1 22); do
    rclone "${R2[@]}" copyto "$SRC/perchrom/chr$c.cugen" "$D/chr$c.cugen" \
        --stats-one-line >/dev/null 2>&1 || fail "pull chr$c"
done

say "merge -> genome-wide panel"
python3 - <<'PYX' || fail "merge"
import sys; sys.path.insert(0, "/root/cugen_ld")
from cugen.convert import merge_cugen
from cugen.io import CugenReader
merge_cugen([f"/root/data/chr{c}.cugen" for c in range(1, 23)],
            "/root/data/chrALL.cugen", verbose=False)
r = CugenReader("/root/data/chrALL.cugen")
print(f"  panel n={r.n_samples:,} p={r.n_variants:,}"); r.close()
PYX
for c in $(seq 1 22); do rm -f "$D/chr$c.cugen"; done

say "Y = G U, s_v, q_v"
python3 - <<'PYX' || fail "basis"
import sys, time; sys.path.insert(0, "/root/cugen_ld")
import numpy as np
from cugen.io import read_cugen
from cugen.popstruct import _unpack_tile

K = 30
pcs = np.load("/root/data/pcs_no_lrld_center.npy")[:, :K]
n = pcs.shape[0]
U, _ = np.linalg.qr(np.concatenate([np.ones((n, 1)), pcs], axis=1))
orth = np.abs(U.T @ U - np.eye(U.shape[1])).max()
print(f"  U {U.shape}  orthonormality {orth:.2e}", flush=True)
assert orth < 1e-10, "QR did not produce an orthonormal basis"

r = read_cugen("/root/data/chrALL.cugen")
p, bpv = int(r.n_variants), int(r.bytes_per_variant)
Y = np.empty((p, U.shape[1]), np.float64)
s_v = np.empty(p); q_v = np.empty(p)
T, t0 = 8192, time.time()
for s0 in range(0, p, T):
    s1 = min(s0 + T, p)
    g = _unpack_tile(np.frombuffer(r.read_packed_bytes(s0, s1), np.uint8
                                   ).reshape(s1 - s0, bpv), n).astype(np.float64)
    g[g == 3] = 0.0                       # 2-bit missing code; none in this panel
    Y[s0:s1] = g @ U
    s_v[s0:s1] = g.sum(1)
    q_v[s0:s1] = (g * g).sum(1)
    if (s0 // T) % 200 == 0:
        el = time.time() - t0
        print(f"    {s1:,}/{p:,} ({100*s1/p:.1f}%) {el/60:.1f} min "
              f"eta {el/max(s1,1)*(p-s1)/60:.0f} min", flush=True)
gidx = np.asarray(r.gidx, np.int64); r.close()
np.savez("/root/data/a1_basis_k30.npz", Y=Y, q_v=q_v, s_v=s_v, gidx=gidx,
         U=U, pcs=pcs, n=n)
print(f"  Y {Y.shape} in {(time.time()-t0)/60:.1f} min", flush=True)
# q_adj must stay positive for a variant that is not an exact function of the
# basis; a non-positive value means the epilogue's vA>0 guard will drop it.
q_adj = q_v - (Y * Y).sum(1)
print(f"  q_adj: min {q_adj.min():.4g}, {int((q_adj <= 0).sum()):,} variants "
      f"<= 0 (annihilated by the basis, correctly dropped downstream)")
PYX

say "upload"
rclone "${R2[@]}" copyto "$D/a1_basis_k30.npz" "$SRC/derived/a1_basis_k30.npz" \
    --stats-one-line || fail "upload"
echo "  uploaded a1_basis_k30.npz"
say "BASIS_OK"
