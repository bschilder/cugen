#!/usr/bin/env bash
# Phase C prerequisites: the per-variant artifacts that the layer scripts index.
#
# All three are invalid against the new panel because the variant set changed by
# k = 1.6185, so every one must be rebuilt before any layer runs:
#
#   ann.tsv           per-variant chrom/pos keyed by gidx
#   layer2_annot.npz  per-variant mappability / segdup / satellite flags
#   pcs_*.npy         PCs from an LD-pruned GRM (new pruned set -> new PCs)
#
# GIDX ALIGNMENT IS THE WHOLE GAME. merge_cugen numbers gidx continuously in the
# order paths are given and keeps source order within each file, so ann.tsv must
# be chr1.bim..chr22.bim concatenated in that exact order. A misalignment here
# does not error -- it silently attributes one variant's annotation to another,
# and Layer 2's entire enrichment table is computed against it. Asserted below.
#
# NO `set -x`: R2 credentials are in this environment.
set -uo pipefail
D=/root/data; mkdir -p $D
exec > >(tee -a /root/prep.log) 2>&1
say(){ echo "===== $(date -u +%H:%M:%S) prep $* ====="; }
fail(){ echo "PREP_FAIL: $*"; exit 1; }
for v in r2_endpoint r2_access_key r2_secret; do
    if [[ -n "${!v:-}" ]]; then echo "  $v present? YES"; else fail "no $v"; fi
done

say "deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq curl git unzip >/dev/null 2>&1
command -v rclone >/dev/null || curl -sSL https://rclone.org/install.sh | bash >/dev/null 2>&1
pip install -q --no-input --break-system-packages numpy "pandas>=2.0" \
    "scipy>=1.10" "pyarrow>=13" cupy-cuda12x >/dev/null 2>&1 || fail "pip"
[ -d /root/cugen_ld ] || git clone -q --branch ld-rowblock --depth 3 \
    https://github.com/bschilder/cugen.git /root/cugen_ld || fail "clone"
[ -x /usr/local/bin/plink2 ] || {
    curl -sSL -o /tmp/p2.zip \
      https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250129.zip \
      && unzip -oq /tmp/p2.zip -d /usr/local/bin/ || fail "plink2"; }
chmod +x /usr/local/bin/plink2

R2=( --s3-provider Cloudflare --s3-endpoint "$r2_endpoint"
     --s3-access-key-id "$r2_access_key" --s3-secret-access-key "$r2_secret"
     --s3-no-check-bucket )
SRC=":s3:smb-data-prod-scratch/cugen/1kg-30x-grch38"

say "pull per-chromosome .bim, .bed, .fam and .cugen"
for c in $(seq 1 22); do
    for e in bim bed fam cugen; do
        rclone "${R2[@]}" copyto "$SRC/perchrom/chr$c.$e" "$D/chr$c.$e" \
            --stats-one-line >/dev/null 2>&1 || fail "pull chr$c.$e"
    done
done
echo "  pulled: $(du -sh $D | cut -f1)"

say "build ann.tsv in gidx order, and ASSERT alignment against the panel"
python3 - <<'PYX' || fail "ann"
import sys; sys.path.insert(0, "/root/cugen_ld")
from cugen.io import read_cugen_header
# Schema is what the consumers actually read: CHROM (NUMERIC -- l12fix casts it
# to int8, so "chr1" must be 1), POS, gidx. Writing CHR/ID instead would raise
# an AttributeError deep inside layer2_annot rather than here.
per = []
gidx = 0
out = open("/root/data/ann.tsv", "w")
out.write("CHROM\tPOS\tID\tgidx\n")
for c in range(1, 23):
    n = 0
    with open(f"/root/data/chr{c}.bim") as f:
        for line in f:
            fld = line.split()
            # .bim col0 may be "1" or "chr1" depending on plink2 version
            ch = fld[0][3:] if fld[0].lower().startswith("chr") else fld[0]
            out.write(f"{ch}\t{fld[3]}\t{fld[1]}\t{gidx}\n")
            gidx += 1
            n += 1
    hp = read_cugen_header(f"/root/data/chr{c}.cugen")["n_variants"]
    if hp != n:
        raise SystemExit(f"chr{c}: .bim has {n:,} rows but .cugen has {hp:,} "
                         f"variants -- gidx would misalign")
    per.append(n)
out.close()
import pandas as pd
chk = pd.read_csv("/root/data/ann.tsv", sep="\t", nrows=5)
assert list(chk.columns) == ["CHROM", "POS", "ID", "gidx"], chk.columns
assert str(chk.CHROM.dtype).startswith("int"), f"CHROM not numeric: {chk.CHROM.dtype}"
print(f"  ann.tsv: {gidx:,} variants, columns {list(chk.columns)}, "
      f"CHROM dtype {chk.CHROM.dtype}")
print(f"  per-chrom: {per}")
PYX

say "merge -> genome-wide panel (unphased) for GRM/PCA"
python3 - <<'PYX' || fail "merge"
import sys; sys.path.insert(0, "/root/cugen_ld")
from cugen.convert import merge_cugen
from cugen.io import CugenReader
paths = [f"/root/data/chr{c}.cugen" for c in range(1, 23)]
merge_cugen(paths, "/root/data/chrALL.cugen", verbose=False)
r = CugenReader("/root/data/chrALL.cugen")
P = r.n_variants; r.close()
n_ann = sum(1 for _ in open("/root/data/ann.tsv")) - 1
if P != n_ann:
    raise SystemExit(f"PANEL/ANN MISALIGNED: panel p={P:,} ann rows={n_ann:,}")
print(f"  merged panel p={P:,} == ann.tsv rows -- gidx aligned")
PYX

say "layer2_annot.npz (downloads its own UCSC/GIAB tracks)"
rclone "${R2[@]}" copyto "$SRC/build/layer2_annot.py" /root/layer2_annot.py \
    --stats-one-line >/dev/null 2>&1 || fail "pull layer2_annot.py"
python3 /root/layer2_annot.py || fail "layer2_annot"

say "upload"
for f in ann.tsv layer2_annot.npz; do
    [ -f "$D/$f" ] || { echo "  MISSING $f"; continue; }
    rclone "${R2[@]}" copyto "$D/$f" "$SRC/derived/$f" --stats-one-line \
        || fail "upload $f"
    echo "  uploaded $f"
done
say "PREP_STAGE1_OK"
