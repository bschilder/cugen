#!/usr/bin/env bash
# PCs for the corrected panel: LD-prune -> GRM -> eigendecomposition.
#
# The pruned marker set changes with the panel, so the PCs change, and every
# Layer 1 number depends on them. Recipe held identical to the original so the
# old -> new comparison isolates the panel:
#   ld_prune(window=1000, r2=0.1) per chromosome
#   the 24 Price et al. long-range-LD regions excluded (data/lrld_grch38.txt)
#   grm(standardize="center")  -- plink2 --make-rel cov convention
#   pcs_from_grm, k=40 retained (analyses use the top 10)
#
# NO `set -x`: R2 credentials are in this environment.
set -uo pipefail
D=/root/data; mkdir -p $D
exec > >(tee -a /root/pc.log) 2>&1
say(){ echo "===== $(date -u +%H:%M:%S) pc $* ====="; }
fail(){ echo "PC_FAIL: $*"; exit 1; }
for v in r2_endpoint r2_access_key r2_secret; do
    if [[ -n "${!v:-}" ]]; then echo "  $v present? YES"; else fail "no $v"; fi
done

say "deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq curl git >/dev/null 2>&1
command -v rclone >/dev/null || curl -sSL https://rclone.org/install.sh | bash >/dev/null 2>&1
pip install -q --no-input --break-system-packages numpy "pandas>=2.0" \
    "scipy>=1.10" cupy-cuda12x >/dev/null 2>&1 || fail "pip"
[ -d /root/cugen_ld ] || git clone -q --branch ld-rowblock --depth 3 \
    https://github.com/bschilder/cugen.git /root/cugen_ld || fail "clone"
cp /root/cugen_ld/data/lrld_grch38.txt $D/ || fail "lrld regions"

R2=( --s3-provider Cloudflare --s3-endpoint "$r2_endpoint"
     --s3-access-key-id "$r2_access_key" --s3-secret-access-key "$r2_secret"
     --s3-no-check-bucket )
SRC=":s3:smb-data-prod-scratch/cugen/1kg-30x-grch38"

say "pull per-chromosome .cugen and ann.tsv"
for c in $(seq 1 22); do
    rclone "${R2[@]}" copyto "$SRC/perchrom/chr$c.cugen" "$D/chr$c.cugen" \
        --stats-one-line >/dev/null 2>&1 || fail "pull chr$c"
done
rclone "${R2[@]}" copyto "$SRC/derived/ann.tsv" "$D/ann.tsv" \
    --stats-one-line >/dev/null 2>&1 || fail "pull ann.tsv"

say "LD-prune per chromosome, excluding the 24 Price LRLD regions"
python3 - <<'PYX' || fail "prune"
import sys; sys.path.insert(0, "/root/cugen_ld")
import numpy as np, pandas as pd, time
from cugen.ld import ld_prune
from cugen.io import read_cugen_header

ann = pd.read_csv("/root/data/ann.tsv", sep="\t")
reg = pd.read_csv("/root/data/lrld_grch38.txt", sep="\t", header=None,
                  names=["chrom", "start", "end", "name"], comment="#")
reg["c"] = reg.chrom.astype(str).str.replace("chr", "", regex=False).astype(int)

keep_all, off = [], 0
for c in range(1, 23):
    p = read_cugen_header(f"/root/data/chr{c}.cugen")["n_variants"]
    sub = ann.iloc[off:off + p]
    assert len(sub) == p and (sub.CHROM == c).all(), f"chr{c} ann slice misaligned"
    pos = sub.POS.to_numpy()
    bad = np.zeros(p, dtype=bool)
    for _, r in reg[reg.c == c].iterrows():
        bad |= (pos >= r.start) & (pos <= r.end)
    cand = np.nonzero(~bad)[0]
    t0 = time.time()
    # max_pairs must be lifted explicitly. ld_prune forwards it to ld_matrix,
    # which defaults to 100,000,000 and REFUSES a larger plan -- and a 1.6M
    # variant chromosome at window=1000 plans ~978M pairs. The cap exists to
    # catch runaway all-pairs requests, not a bounded sliding window, and it is
    # the third place in this project it has had to be raised deliberately.
    kept, _ = ld_prune(f"/root/data/chr{c}.cugen", window=1000, r2=0.1,
                       variants=cand, maf_min=0.01, backend="gpu",
                       max_pairs=10**18, verbose=False)
    # ld_prune returns (keep, drop) as DataFrames with gidx/ID columns, NOT
    # index arrays. np.asarray(kept) yields a 2-column object array, and adding
    # the chromosome offset to it fails on the ID column with
    # "can only concatenate str (not int) to str". Assert the shape so a future
    # change surfaces here rather than as an arithmetic error.
    assert hasattr(kept, "columns") and "gidx" in kept.columns, (
        f"ld_prune returned {type(kept)}; expected a frame with a gidx column")
    # For a per-chromosome .cugen the stored gidx IS the row position (0..p-1),
    # and merge_cugen renumbers continuously in path order, so adding the
    # running offset gives the gidx in the merged panel -- which is what
    # grm(variants=) matches on (stored gidx, not row position).
    keep_all.append(kept.gidx.to_numpy().astype(np.int64) + off)
    print(f"  chr{c}: {p:,} variants, {bad.sum():,} in LRLD regions, "
          f"{len(kept):,} pruned-in  ({time.time()-t0:.0f}s)", flush=True)
    assert keep_all[-1].max() < off + p, "pruned gidx exceeds this chromosome"
    off += p
sel = np.concatenate(keep_all)
np.save("/root/data/pruned_no_lrld.npy", sel)
print(f"  total pruned-in: {len(sel):,} of {off:,}")
PYX

say "merge -> genome-wide panel"
python3 - <<'PYX' || fail "merge"
import sys; sys.path.insert(0, "/root/cugen_ld")
from cugen.convert import merge_cugen
merge_cugen([f"/root/data/chr{c}.cugen" for c in range(1, 23)],
            "/root/data/chrALL.cugen", verbose=False)
from cugen.io import CugenReader
r = CugenReader("/root/data/chrALL.cugen")
print(f"  panel p={r.n_variants:,} n={r.n_samples:,}"); r.close()
PYX
for c in $(seq 1 22); do rm -f "$D/chr$c.cugen"; done

say "GRM (centred) on the pruned set, then PCs"
python3 - <<'PYX' || fail "grm"
import sys; sys.path.insert(0, "/root/cugen_ld")
import numpy as np
from cugen.popstruct import grm, pcs_from_grm
sel = np.load("/root/data/pruned_no_lrld.npy")
print(f"  GRM over {len(sel):,} pruned markers", flush=True)
A = grm("/root/data/chrALL.cugen", variants=sel, maf_min=0.01,
        standardize="center", verbose=True)
pcs, ev = pcs_from_grm(A, 40, return_eigenvalues=True)
np.save("/root/data/pcs_no_lrld_center.npy", pcs)
np.save("/root/data/eig_no_lrld_center.npy", ev)
print(f"  PCs {pcs.shape}; top-10 eigenvalue share "
      f"{100*ev[:10].sum()/ev.sum():.2f}%")
print(f"  eigenvalues 1-10: {np.round(ev[:10], 3)}")
PYX

say "upload"
for f in pcs_no_lrld_center.npy eig_no_lrld_center.npy pruned_no_lrld.npy; do
    rclone "${R2[@]}" copyto "$D/$f" "$SRC/derived/$f" --stats-one-line \
        || fail "upload $f"
    echo "  uploaded $f"
done
say "PC_OK"
