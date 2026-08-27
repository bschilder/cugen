#!/usr/bin/env bash
# Genome-wide LD scan for one (population, arm) on a GPU pod.
#
# Pulls the per-chromosome .cugen files for POP from R2, concatenates them into
# a genome-wide panel, scans, and streams shards back to R2 as they land.
#
# NO `set -x`: R2 credentials are in this environment.
set -uo pipefail
POP="${POP:?POP not set}"          # ALL | AFR | AMR | EAS | EUR | SAS
ARM="${ARM:?ARM not set}"          # un | ph
MODE="${MODE:-gw}"                 # gw = genome-wide (cis+trans) | cis = per-chromosome
#
# Both are needed and they are not redundant. The genome-wide scan concatenates
# all 22 chromosomes and emits every pair, so it CONTAINS the within-chromosome
# pairs -- but recovering them means filtering across ~2,500 shards, since shards
# are row-budgeted and each spans a narrow slab of both variant axes rather than
# aligning to chromosome boundaries. A per-chromosome scan writes one
# independently-queryable dataset per chromosome, which is what a viz layer wants
# to load. Cis is also cheap: cross-chromosome pairs were 93.7% of the old row
# count, so the cis arm is ~6% of the genome-wide output.
D=/root/data; mkdir -p $D /root/ld_out
exec > >(tee -a /root/ldjob.log) 2>&1
say(){ echo "===== $(date -u +%H:%M:%S) $POP/$ARM $* ====="; }
fail(){ echo "LDJOB_FAIL $POP/$ARM: $*"; exit 1; }

for v in r2_endpoint r2_access_key r2_secret; do
    if [[ -n "${!v:-}" ]]; then echo "  $v present? YES"; else fail "no $v"; fi
done

say "deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq curl git >/dev/null 2>&1
command -v rclone >/dev/null || curl -sSL https://rclone.org/install.sh | bash >/dev/null 2>&1
pip install -q --no-input --break-system-packages numpy "pandas>=2.0" \
    "scipy>=1.10" "pyarrow>=13" cupy-cuda12x >/dev/null 2>&1 || fail "pip"
[ -d /root/cugen_ld ] || git clone -q --branch ld-rowblock --depth 3 \
    https://github.com/bschilder/cugen.git /root/cugen_ld || fail "clone"
python3 -c "import cupy; d=cupy.cuda.Device(0); print('  GPU cc', d.compute_capability, 'free', round(d.mem_info[0]/1e9,1), 'GB')" || fail "cupy"

R2=( --s3-provider Cloudflare --s3-endpoint "$r2_endpoint"
     --s3-access-key-id "$r2_access_key" --s3-secret-access-key "$r2_secret"
     --s3-no-check-bucket )
SRC=":s3:smb-data-prod-scratch/cugen/1kg-30x-grch38/perchrom"
[ "$POP" = "ALL" ] && PRE="" || PRE="superpop/$POP/"
SUF=""; [ "$ARM" = "ph" ] && SUF="_ph"

say "pull per-chromosome .cugen for $POP"
for c in $(seq 1 22); do
    rclone "${R2[@]}" copyto "$SRC/${PRE}chr${c}${SUF}.cugen" "$D/chr${c}.cugen" \
        --stats-one-line >/dev/null 2>&1 || fail "pull chr$c"
done
echo "  pulled $(ls $D/chr*.cugen | wc -l) files, $(du -sh $D | cut -f1)"

if [ "$MODE" = "cis" ]; then
    say "per-chromosome scans (within-chromosome pairs only)"
    for c in $(seq 1 22); do
        [ -f "$D/chr${c}.cugen" ] || fail "missing chr$c"
        say "scan chr$c"
        CHROM="$c" ARM="$ARM" POP="$POP" MODE=cis \
            python3 /root/ldwrite_r2.py || fail "scan chr$c"
        rm -f "$D/chr${c}.cugen"
    done
    say "LDJOB_OK"
    exit 0
fi

say "concatenate -> genome-wide panel"
python3 - <<'PYX' || fail "merge"
import sys; sys.path.insert(0, "/root/cugen_ld")
import os
from cugen.convert import merge_cugen
from cugen.io import CugenReader
suf = "_ph" if os.environ["ARM"] == "ph" else ""
paths = [f"/root/data/chr{c}.cugen" for c in range(1, 23)]
out = f"/root/data/chrALL{suf}.cugen"
merge_cugen(paths, out, verbose=False)
r = CugenReader(out)
print(f"  merged: n={r.n_samples:,} p={r.n_variants:,} phased={r.phased} "
      f"has_missing={r.has_missing}")
if os.environ["ARM"] == "ph":
    assert r.phased, "PHASE LOST IN MERGE"
r.close()
for p in paths:
    os.remove(p)          # free disk before the scan writes ~1.6 TB
PYX

say "scan"
ARM="$ARM" POP="$POP" python3 /root/ldwrite_r2.py || fail "scan"
say "LDJOB_OK"
