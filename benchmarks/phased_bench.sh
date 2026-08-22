#!/usr/bin/env bash
set -uo pipefail
D=/root/data; C=/root/cugen_phased
say(){ echo "===== $* ====="; }
cd $C && git fetch -q origin && git reset -q --hard origin/phased-ld && git log --oneline -1

say "B1 build a PHASED hap2bit .cugen from the phased VCF"
pip install -q cyvcf2 2>&1 | tail -1
if [ ! -s $D/chr22_maf01_ph.cugen ]; then
python3 - <<'PY'
from cugen.convert import vcf2cugenh
vcf2cugenh("/root/data/chr22_maf01_ph.vcf.gz", "/root/data/chr22_maf01_ph.cugen",
           verbose=True)
PY
fi
python3 -c "
import cugen as cg, json
h = cg.io.read_cugen_header('/root/data/chr22_maf01_ph.cugen')
print('  header:', {k: str(v) for k, v in h.items()})
print('  is_phased:', cg.is_phased('/root/data/chr22_maf01_ph.cugen'))"

say "B2 CORRECTNESS: cugen r2_phased vs plink2 --r2-phased (phased PGEN), p=1000"
LO=$(awk 'NR==1{print $4}' $D/chr22_maf01.bim); HI=$(awk 'NR==1000{print $4}' $D/chr22_maf01.bim)
python3 - <<PY
from cugen.ld import ld_matrix
df = ld_matrix("/root/data/chr22_maf01_ph.cugen", variant_range=(0,1000),
               min_r2=0.2, stats=("r_phased","r2_phased"), sign_reference="major",
               output="/tmp/ph_cg.tsv", max_pairs=10**15, verbose=True)
print("  cugen phased rows:", len(df))
PY
rm -f /tmp/ph_pl.*
plink2 --pfile $D/chr22_ph --chr 22 --from-bp $LO --to-bp $HI \
  --r2-phased allow-ambiguous-allele cols=chrom,pos,id --ld-window 999999 \
  --ld-window-kb 999999 --ld-window-r2 0.2 --threads 13 --out /tmp/ph_pl --silent
echo "  plink2 --r2-phased rows: $(( $(wc -l < $(ls /tmp/ph_pl.vcor* | head -1)) - 1 ))"
python3 /root/phased_cmp.py

say "B3 SPEED: cugen phased vs plink2 --r2-phased across the grid"
printf "%9s %15s %10s %12s %10s %12s %9s\n" p pairs cugen_s cugen_rows plink_s plink_rows speedup
for P in 1000 5000 20000 50000 170949; do
  HI=$(awk -v n=$P 'NR==n{print $4}' $D/chr22_maf01.bim)
  CG=$(python3 - <<PY
import time
from cugen.ld import ld_matrix
ts=[]
for _ in range(3):
    t0=time.perf_counter()
    df=ld_matrix("/root/data/chr22_maf01_ph.cugen", variant_range=(0,$P), min_r2=0.2,
                 stats=("r_phased","r2_phased"), sign_reference="major",
                 output="/tmp/pb_cg.tsv", max_pairs=10**15, verbose=False)
    ts.append(time.perf_counter()-t0); n=len(df)
print(f"{sorted(ts)[1]:.4f} {n}")
PY
)
  rm -f /tmp/pb_pl.*
  S=$(date +%s%N)
  plink2 --pfile $D/chr22_ph --chr 22 --from-bp $LO --to-bp $HI \
    --r2-phased allow-ambiguous-allele cols=chrom,pos,id --ld-window 999999 \
    --ld-window-kb 999999 --ld-window-r2 0.2 --threads 13 --out /tmp/pb_pl --silent >/dev/null 2>&1
  E=$(date +%s%N)
  PF=$(ls /tmp/pb_pl.vcor* 2>/dev/null | head -1)
  PR=$([ -n "$PF" ] && echo $(( $(wc -l < "$PF") - 1 )) || echo NOFILE)
  awk -v p=$P -v cg="$CG" -v pl="$(awk "BEGIN{printf \"%.4f\", ($E-$S)/1e9}")" -v pr="$PR" \
    'BEGIN{split(cg,a," "); printf "%9d %15d %10s %12s %10s %12s %8.1fx\n",
     p, p*(p-1)/2, a[1], a[2], pl, pr, pl/a[1]}'
done
say "PHASED BENCH DONE"
