# LD test fixtures

`ld_fixture.npy` -- (12 variants, 60 samples) uint8 dosages, 3 = missing.
Variant 11 is monomorphic; variants 4 and 9 carry missing calls; ALT frequency
straddles 0.5 so the "alt" and "major" sign conventions genuinely diverge.

Golden files produced with **PLINK v2.0.0-a.7.1 M1 (4 May 2026)**:

```
plink2 --vcf ld_fixture.vcf --make-pgen --out t
plink2 --pfile t --r-unphased allow-ambiguous-allele \
  cols=chrom,pos,id,ref,alt,maj,nonmaj,freq \
  --ld-window 999999 --ld-window-kb 99999 --ld-window-r2 0 --out u
plink2 --pfile t --r-phased allow-ambiguous-allele \
  cols=chrom,pos,id,ref,alt,maj,nonmaj,freq,d,dprime \
  --ld-window 999999 --ld-window-kb 99999 --ld-window-r2 0 --out p
```

`--ld-window-r2 0` is mandatory: plink2 defaults to 0.2 and would silently drop
most pairs. `--ld-window` has NO default in plink2 (it was 10 in plink 1.9), so
it is set explicitly. `allow-ambiguous-allele` is a MODIFIER on --r*, not a
standalone flag.

The .vcor files carry ~6 significant figures, so compare with rtol=1e-6.

## Clumping fixtures

`clump_fixture.npz` -- `G` (150 variants, 250 samples) uint8 dosages laid out
in LD blocks of 4-9 variants, plus `POS` (sorted, spread over 1.5 Mb so the
250 kb window bites) and `P`. `clump_sumstats.tsv` is the same p-values in
plink report form.

`clump_gold.clumps` and `clump_gold_overlap.clumps` are plink2 v2.0.0-a.7.1
output, the second with `--clump-allow-overlap`. Regenerate with:

```bash
# rebuild the VCF from the npz first (see tests/test_ld_clump.py::_clump_fixture)
plink2 --vcf cl.vcf --make-pgen --out cl
plink2 --pfile cl --clump clump_sumstats.tsv --clump-unphased \
    --clump-p1 1e-4 --clump-p2 0.01 --clump-r2 0.5 --clump-kb 250 --out gold
plink2 --pfile cl --clump clump_sumstats.tsv --clump-unphased \
    --clump-allow-overlap --clump-p1 1e-4 --clump-p2 0.01 --clump-r2 0.5 \
    --clump-kb 250 --out gold_overlap
```

`--clump-unphased` is mandatory: plink2's default `--clump` uses PHASED r^2 and
.cugen discards phase, so the unphased form is the only one we can match. Same
caveat as D/D'.

The fixture is chosen so the golden actually exercises the rules -- it contains
clumps whose `TOTAL` exceeds their `SP2` count (members above p2) and a
non-zero `NONSIG` bin. `test_golden_exercises_the_asymmetry_it_is_meant_to`
asserts that, because a parity test that cannot fail is worthless: an earlier
implementation gated membership on p2, produced the right index variants with
systematically low `TOTAL`, and would have passed a golden without them.

## Real 1000 Genomes fixture

`ld_1kg_chr22_eur.cugen` -- hap2bit (phased), **150 variants x 503 samples
(1,006 haplotypes)**, 22 KB. Real data, not simulated: the other fixtures here
are constructed, and constructed frequencies do not reproduce the rare-variant
tail that the exact conditional test exists for.

Spectrum, asserted by
`test_the_real_1kg_fixture_still_has_the_maf_spectrum_these_tests_need`:
9 singletons (AC=1), 123 variants at AC<=5, 14 at MAF>=0.05, and some variants
monomorphic within EUR. **Do not regenerate this with a MAF filter** -- the
significance tests that use it would still pass and would stop testing anything.

Built from the 1kGP high-coverage phased panel (`20220422_3202_phased_SNV_INDEL_SV`,
3,202 samples), streamed rather than downloaded:

```
U=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
P=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel

# EUR sample list, intersected with the samples actually in the VCF
curl -s $P | awk 'NR>1 && $3=="EUR" {print $1}' | sort > eur.txt

bcftools view -r chr22:20000000-21000000 -v snps -m2 -M2 -Oz -o raw.vcf.gz $U
bcftools index -t raw.vcf.gz
bcftools view -S eur.txt --force-samples raw.vcf.gz -Oz -o eur.vcf.gz   # NO -q/-Q
bcftools index -t eur.vcf.gz
bcftools view eur.vcf.gz -Oz -o fix.vcf.gz \
  --regions-file <(bcftools query -f '%CHROM\t%POS\n' eur.vcf.gz | head -150)
bcftools index -t fix.vcf.gz

python -c "from cugen.convert import vcf2cugenh; \
  vcf2cugenh('fix.vcf.gz','ld_1kg_chr22_eur.cugen')"     # needs cyvcf2 or pysam
```

There is deliberately **no plink2 golden alongside this one**. plink2 emits no LD
p-values, so it cannot check the quantity under test, and a golden for the r^2 it
does emit would be ~130 KB for a fixture whose point is the p-value. Real-data
plink2 parity for the chi-square lives in `benchmarks/significance_1kg.py`
instead, alongside the other real-data comparisons (`compare_plink.py`,
`phased_cmp.py`), and its measured result is in
`benchmarks/results/SIGNIFICANCE.md`.
