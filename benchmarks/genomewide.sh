#!/usr/bin/env bash
set -uo pipefail
D=/root/data; C=/root/cugen_phased
say(){ echo "===== $* ====="; }
cd $C && git fetch -q origin && git reset -q --hard origin/phased-ld && git log --oneline -1

say "G1 fetch chr21 and build a matched fixture"
if [ ! -s $D/chr21_maf01.bed ]; then
  curl -fsSL --retry 3 -o $D/chr21.vcf.gz \
    "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr21.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
  plink2 --vcf $D/chr21.vcf.gz --max-alleles 2 --snps-only --maf 0.01 \
         --make-bed --out $D/chr21_maf01 --threads 13 --silent
fi
echo "  chr21 variants: $(wc -l < $D/chr21_maf01.bim)"
[ -s $D/chr21_maf01.cugen ] || python3 -c "
from cugen.convert import bed2cugen
bed2cugen('$D/chr21_maf01.bed','$D/chr21_maf01.cugen',
          bim='$D/chr21_maf01.bim',fam='$D/chr21_maf01.fam',verbose=False)"
echo "  chr21 cugen: $(python3 -c "
import cugen as cg; print(cg.io.read_cugen_header('$D/chr21_maf01.cugen')['n_variants'])")"

say "G2 merge chr21 + chr22 into ONE genome-scale .cugen"
python3 -c "
from cugen.convert import merge_cugen
merge_cugen(['$D/chr21_maf01.cugen','$D/chr22_maf01.cugen'],
            '$D/chr21_22.cugen', verbose=True)"
python3 -c "
import cugen as cg
h=cg.io.read_cugen_header('$D/chr21_22.cugen')
print('  merged:', {k:str(v) for k,v in h.items() if k in
      ('n_variants','n_samples','encoding','file_size_gb')})"

say "G3 VALIDATE: within-chromosome pairs unchanged by the merge"
python3 - <<'PY'
import numpy as np
from cugen.ld import ld_matrix
import cugen as cg
n21 = int(cg.io.read_cugen_header('/root/data/chr21_maf01.cugen')['n_variants'])
# same 2,000 variants, once inside chr21 alone and once inside the merged file
solo = ld_matrix('/root/data/chr21_maf01.cugen', variant_range=(0,2000),
                 min_r2=0.2, stats=('r','r2'), sign_reference='major',
                 output='/tmp/gw_solo.tsv', max_pairs=10**15, verbose=False)
both = ld_matrix('/root/data/chr21_22.cugen', variant_range=(0,2000),
                 min_r2=0.2, stats=('r','r2'), sign_reference='major',
                 output='/tmp/gw_both.tsv', max_pairs=10**15, verbose=False)
f = lambda d: {(int(a),int(b)): float(v) for a,b,v in
               zip(d['gidx_a'].to_numpy() if hasattr(d['gidx_a'],'to_numpy') else d['gidx_a'],
                   d['gidx_b'].to_numpy() if hasattr(d['gidx_b'],'to_numpy') else d['gidx_b'],
                   d['R'].to_numpy() if hasattr(d['R'],'to_numpy') else d['R'])}
S, B = f(solo), f(both)
bad = [k for k in S if k not in B or abs(S[k]-B[k])>1e-6]
print(f"  chr21-alone rows {len(S):,}  merged-prefix rows {len(B):,}  differing {len(bad):,}")
print(f"  VALIDATED" if not bad and len(S)==len(B) else f"  MISMATCH")
PY

say "G4 CROSS-CHROMOSOME all-pairs over the merged file"
python3 - <<'PY'
import time
import cugen as cg
from cugen.ld import ld_matrix
h = cg.io.read_cugen_header('/root/data/chr21_22.cugen')
P = int(h['n_variants'])
print(f"  p = {P:,}  pairs = {P*(P-1)//2:,}")
ts = []
for _ in range(2):
    t0 = time.perf_counter()
    df = ld_matrix('/root/data/chr21_22.cugen', min_r2=0.2, stats=('r','r2'),
                   sign_reference='major', output='/tmp/gw_all.tsv',
                   max_pairs=10**15, verbose=False)
    ts.append(time.perf_counter()-t0); n = len(df)
w = min(ts)
print(f"  wall {w:.3f}s   rows {n:,}   {P*(P-1)//2/w:.3e} pairs/s")
PY
say "GENOME-WIDE DONE"
