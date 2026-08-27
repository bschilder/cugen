#!/usr/bin/env bash
# One chromosome of the 1kGP 30x panel, start to finish, on a throwaway CPU pod.
# Downloads its own VCF, filters, converts, uploads artifacts to HF, deletes the
# VCF. Nothing but the outputs ever leaves the pod.
#
# NO `set -x` anywhere: HF_TOKEN is in this environment.
#
# ASCERTAINMENT (the reason this file was rewritten)
# --------------------------------------------------
# The previous version filtered `--maf 0.01` on the POOLED 2,504 samples, and
# superpop.sh then built each superpopulation panel from that already-filtered
# file. So every panel was (pooled MAF>=1%) AND (within-pop MAF>=1%), and a
# variant common inside one superpopulation but rare pooled was never available
# to ANY analysis. A variant private to a group holding fraction f of the cohort
# has pooled AF ~= AF_group*f, so the pooled filter discarded everything with
# AF_group < 0.01/f: AF < 3.8% for AFR, < 7.2% for AMR. That is the highest-Fst
# slice of the panel -- precisely the variants that generate two-locus Wahlund
# LD, and precisely what a multi-ancestry LD reference is for.
#
# Now: the ALL panel keeps a variant reaching MAF>=0.01 in ANY superpopulation
# (the ACAF rule, cugen.freq.union_maf_pass), and each superpopulation panel is
# filtered from the UNFILTERED pgen rather than from a pooled-filtered ancestor.
#
# One VCF download now yields all six panels. The old flow downloaded the VCF
# once for ALL, then re-downloaded the pooled .bed from HF five more times.
set -uo pipefail
N="${CHROM:?CHROM not set}"
MAF="${MAF_MIN:-0.01}"
POPS="${POPS:-AFR AMR EAS EUR SAS}"
D=/root/w; mkdir -p $D
L=/root/job.log; exec > >(tee -a "$L") 2>&1
say(){ echo "===== $(date -u +%H:%M:%S) chr$N $* ====="; }
fail(){ echo "CHROMJOB_FAIL chr$N: $*"; exit 1; }

if [[ -n "${HF_TOKEN:-}" ]]; then echo "  HF_TOKEN present? YES"; else fail "no HF_TOKEN"; fi

say "deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq bcftools tabix curl git unzip >/dev/null 2>&1 || fail "apt"
pip install -q --no-input --break-system-packages --root-user-action=ignore \
    numpy "pandas>=2.0" "scipy>=1.10" "pyarrow>=13" cyvcf2 huggingface_hub \
    >/dev/null 2>&1 || fail "pip"
[ -d /root/cugen_ld ] || git clone -q --branch ld-rowblock --depth 3 \
    https://github.com/bschilder/cugen.git /root/cugen_ld || fail "clone"
[ -x /usr/local/bin/plink2 ] || {
    curl -sSL -o /tmp/p2.zip \
      https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250129.zip \
      && unzip -oq /tmp/p2.zip -d /usr/local/bin/ || fail "plink2 fetch"; }
chmod +x /usr/local/bin/plink2
plink2 --version | head -1

say "download (chunked ranges; EBI drops long transfers)"
U=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV
F=1kGP_high_coverage_Illumina.chr$N.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
M5=$(grep -F "$F " /root/manifest.txt | awk '{print $2}')
[ -z "$M5" ] && fail "no md5 in manifest for $F"
/root/chunkget.sh "$U/$F" "$D/chr$N.vcf.gz" "$M5" 16777216 8 || fail "download/md5"

# --- 1. UNFILTERED pgen: 2504 unrelated, biallelic SNV, phase kept, NO --maf --
say "plink2: 2504 unrelated, biallelic SNV, NO frequency filter yet"
plink2 --vcf "$D/chr$N.vcf.gz" --keep /root/keep_ALL.txt \
       --max-alleles 2 --snps-only \
       --make-pgen --out "$D/raw$N" --threads "$(nproc)" --silent || fail "pgen raw"
NRAW=$(grep -vc '^#' "$D/raw$N.pvar")
echo "  unfiltered variants=$NRAW"

# --- 2. per-superpopulation allele frequencies -------------------------------
say "plink2 --freq per superpopulation ($POPS)"
for P in $POPS; do
    [ -s "/root/keep_$P.txt" ] || fail "missing /root/keep_$P.txt"
    plink2 --pfile "$D/raw$N" --keep "/root/keep_$P.txt" --freq \
           --out "$D/f$N.$P" --threads "$(nproc)" --silent || fail "freq $P"
    echo "  $P: $(grep -vc '^#' "$D/f$N.$P.afreq") variants scored"
done

# --- 3. union: MAF >= MAF_MIN in ANY superpopulation (the ACAF rule) ---------
say "union of per-superpop MAF>=$MAF"
python3 - <<PY || fail "union"
import sys; sys.path.insert(0, "/root/cugen_ld")
from cugen.freq import read_afreq, union_maf_pass
pops = "$POPS".split()
freqs = {p: read_afreq(f"$D/f$N.{p}.afreq") for p in pops}
keep = union_maf_pass(freqs, $MAF)
with open("$D/union$N.txt", "w") as fh:
    fh.write("\n".join(sorted(keep)) + "\n")
allv = set().union(*(set(t) for t in freqs.values()))
print(f"  union keeps {len(keep):,} of {len(allv):,} variants "
      f"({100.0*len(keep)/max(len(allv),1):.1f}%)")
# What a POOLED filter would have kept. This difference IS the defect being
# corrected and the report needs the number, so the arithmetic lives in
# cugen.freq under test rather than inline here.
from cugen.freq import pooled_from_groups
sizes = {p: sum(1 for _ in open(f"/root/keep_{p}.txt")) for p in pops}
paf = pooled_from_groups(freqs, sizes)
pooled = {v for v, a in paf.items() if min(a, 1.0 - a) >= $MAF}
assert pooled <= keep, "pooled set must be a subset of the union -- check folding"
print(f"  a POOLED filter would keep {len(pooled):,} "
      f"-- union adds {len(keep - pooled):,} variants "
      f"({100.0*len(keep - pooled)/max(len(keep),1):.1f}% of the panel)")
open("$D/ascertainment$N.txt","w").write(
    "chrom\tunfiltered\tunion\tpooled\tunion_only\n"
    f"chr$N\t{len(allv)}\t{len(keep)}\t{len(pooled)}\t{len(keep-pooled)}\n")
PY

# --- 4. the ALL panel --------------------------------------------------------
say "ALL panel: --extract union"
plink2 --pfile "$D/raw$N" --extract "$D/union$N.txt" \
       --make-pgen --out "$D/c${N}ph" --threads "$(nproc)" --silent || fail "pgen ALL"
NV=$(grep -vc '^#' "$D/c${N}ph.pvar")
NS=$(( $(wc -l < "$D/c${N}ph.psam") - 1 ))
echo "  ALL variants=$NV samples=$NS"

say "export phased VCF, then vcf2cugenh (require_phased=True is the gate)"
plink2 --pfile "$D/c${N}ph" --export vcf bgz --out "$D/c${N}px" \
       --threads "$(nproc)" --silent || fail "export"
python3 - <<PY || fail "vcf2cugenh"
import sys; sys.path.insert(0,"/root/cugen_ld")
from cugen.convert import vcf2cugenh
vcf2cugenh("$D/c${N}px.vcf.gz", "$D/chr${N}_ph.cugen", verbose=False)
from cugen.io import CugenReader
r=CugenReader("$D/chr${N}_ph.cugen")
print(f"  phased .cugen: n={r.n_samples} p={r.n_variants} phased={r.phased}")
assert r.phased, "PHASE LOST"
r.close()
PY

say "unphased .bed (plink2 comparator) + unphased .cugen"
plink2 --pfile "$D/c${N}ph" --make-bed --out "$D/chr$N" \
       --threads "$(nproc)" --silent || fail "bed"
python3 - <<PY || fail "bed2cugen"
import sys; sys.path.insert(0,"/root/cugen_ld")
from cugen.convert import bed2cugen
bed2cugen("$D/chr$N.bed","$D/chr$N.cugen",bim="$D/chr$N.bim",fam="$D/chr$N.fam",verbose=False)
from cugen.io import CugenReader
r=CugenReader("$D/chr$N.cugen"); print(f"  unphased .cugen: n={r.n_samples} p={r.n_variants}"); r.close()
PY

# --- 5. per-superpopulation panels, from the UNFILTERED pgen -----------------
# NOT from the ALL panel. Subsetting a union-filtered file would be closer to
# correct than the old code was, but it still cannot contain a variant the
# union dropped, and the point of a per-pop panel is its own ascertainment.
say "per-superpopulation panels, each from the unfiltered pgen"
for P in $POPS; do
    plink2 --pfile "$D/raw$N" --keep "/root/keep_$P.txt" --maf "$MAF" \
           --make-bed --out "$D/sp$N.$P" --threads "$(nproc)" --silent || fail "bed $P"
    echo "  $P: $(grep -c . "$D/sp$N.$P.bim") variants"
    python3 - <<PY || fail "bed2cugen $P"
import sys; sys.path.insert(0,"/root/cugen_ld")
from cugen.convert import bed2cugen
bed2cugen("$D/sp$N.$P.bed","$D/sp$N.$P.cugen",
          bim="$D/sp$N.$P.bim",fam="$D/sp$N.$P.fam",verbose=False)
PY
done

say "free the VCF and intermediates before upload"
rm -f "$D/chr$N.vcf.gz" "$D/c${N}px.vcf.gz" \
      "$D/c${N}ph.pgen" "$D/c${N}ph.pvar" "$D/c${N}ph.psam" \
      "$D/raw$N.pgen" "$D/raw$N.pvar" "$D/raw$N.psam" \
      "$D"/f$N.*.afreq "$D"/sp$N.*.bed
df -h /root | tail -1

say "upload artifacts to HF staging"
python3 - <<PY || fail "hf upload"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "standardmodelbio/cugen-ld"
base = "1kg-30x-grch38/_staging_v2/perchrom"
items = [
    ("chr${N}_ph.cugen", f"{base}/chr${N}_ph.cugen"),
    ("chr${N}.cugen",    f"{base}/chr${N}.cugen"),
    ("chr${N}.bed",      f"{base}/chr${N}.bed"),
    ("chr${N}.bim",      f"{base}/chr${N}.bim"),
    ("chr${N}.fam",      f"{base}/chr${N}.fam"),
    ("union${N}.txt",    f"{base}/union/chr${N}.union.txt"),
    ("ascertainment${N}.txt", f"{base}/union/chr${N}.ascertainment.tsv"),
]
for p in "$POPS".split():
    items += [(f"sp${N}.{p}.cugen", f"{base}/superpop/{p}/chr${N}.cugen"),
              (f"sp${N}.{p}.bim",   f"{base}/superpop/{p}/chr${N}.bim"),
              (f"sp${N}.{p}.fam",   f"{base}/superpop/{p}/chr${N}.fam")]
for f, dest in items:
    p = f"/root/w/{f}"
    if not os.path.exists(p):
        print("  MISSING", f); continue
    api.upload_file(path_or_fileobj=p, path_in_repo=dest, repo_id=REPO,
                    repo_type="dataset")
    print(f"  uploaded {f} ({os.path.getsize(p)/1e6:.0f} MB)")
PY
say "CHROMJOB_OK unfiltered=$NRAW all=$NV"
