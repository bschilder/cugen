# Coevolution between physically unlinked loci

Applies `cugen.coevo` and the `ga` statistic to the design of

> Rohlfs RV, Swanson WJ, Weir BS (2010). Detecting Coevolution through Allelic
> Association between Physically Unlinked Loci. *Am J Hum Genet* 86:674-685.

## What the test is

Selection for allele **matching** between two interacting proteins can maintain
allelic association with no linkage, because a mismatched gamete pair is less
fit. The evidence is relative: a candidate gene pair on different chromosomes
should carry more association than a background of random unlinked pairs drawn
from the same individuals.

Two statistics, both on unphased genotypes:

- **CLD** — Weir's composite LD, `chi2 = N * r^2`, 1 df. Already in cugen.
- **GA** — the 3x3 genotype table test, up to 4 df. Added alongside, because CLD
  is a one-df additive summary and cannot see association whose dosage
  covariance cancels.

## Why structure does not sink it

Population structure creates association between unlinked loci (the two-locus
Wahlund effect) and this design never removes it. It does not have to: structure
inflates the candidate and the background alike, so it cancels in the comparison.

That cancellation has a precondition, which is why `rohlfs_replication.py` runs
**per superpopulation and never pooled**. In a pooled panel a candidate pair
whose alleles are continentally divergent would outrun a background that is
mostly not divergent, and the Wahlund term would stop cancelling.

Reference-mapping artifact does **not** cancel, because it is specific to
particular region pairs rather than shared across them. Background windows
overlapping the ENCODE blacklist, segmental duplications, satellite arrays or
centromeres are therefore dropped. Rohlfs et al. blastn-checked their two
candidate genes for probe cross-hybridisation, which was the right control for
the candidates; the background they compared against was not filtered, which
makes their test conservative by an unknown amount rather than anti-conservative.

## A note on the candidate genes

Their primary pair was **ZP3 x ZP3R**. `ZP3R` now returns zero hits in both HGNC
and NCBI Gene for human; the locus is annotated as the pseudogenes `C4BPAP1` and
`C4BPAP2`. A unitary pseudogene makes no protein, so the protein-protein
coevolution premise does not survive re-annotation. The script runs that locus
labelled as a pseudogene proxy, and runs their secondary pair **GHR x GH2**,
both of which are still valid genes, as the interpretable comparison.

## Running it

```bash
HF_TOKEN=... python benchmarks/coevo/rohlfs_replication.py
```

Reads per-superpopulation `.cugen` / `.bim` panels from the private
`standardmodelbio/cugen` dataset. Writes `rohlfs_replication.json`.

## Results

1000 Genomes 30x, GRCh38, 8 tag SNPs per locus, 60 artifact-masked background
window pairs, 100 random gene pairs for calibration, 1000 permutations.

| pop | pair | cand median GA | bg median GA | KS p (naive) | **KS p (empirical)** |
|---|---|---|---|---|---|
| AFR | GHR x GH2 | 3.38 | 3.12 | 0.481 | 0.475 |
| AMR | GHR x GH2 | 3.30 | 3.60 | 0.149 | 0.337 |
| EAS | GHR x GH2 | 6.53 | 3.22 | 3.2e-10 | **0.010** |
| EUR | GHR x GH2 | 4.51 | 3.23 | 0.00225 | **0.040** |
| SAS | GHR x GH2 | 2.56 | 3.30 | 0.850 | 0.723 |
| AFR | ZP3 x C4BPAP1 | 3.40 | 3.13 | 0.192 | 0.267 |
| AMR | ZP3 x C4BPAP1 | 3.70 | 3.63 | 0.756 | 0.733 |
| EAS | ZP3 x C4BPAP1 | 3.41 | 3.45 | 0.709 | 0.515 |
| EUR | ZP3 x C4BPAP1 | 3.19 | 3.29 | 0.741 | 0.703 |
| SAS | ZP3 x C4BPAP1 | 3.57 | 3.35 | 0.731 | 0.624 |

**Read the empirical column, not the naive one.** In EAS the naive KS p-value is
3.2e-10 and the empirical one is 0.010 — eight orders of magnitude apart. The
naive value is wrong because the 64 candidate SNP pairs share within-gene LD and
are nowhere near 64 independent observations; the empirical value asks where the
candidate's KS statistic falls among 100 random gene pairs carrying the same
dependence. Any pipeline that reports the naive number will manufacture findings.

**The background is well calibrated.** Its median GA across the ten runs averages
**3.30** against the theoretical median of chi-square with 4 df, **3.357**. So
within a superpopulation the unlinked-pair null behaves as theory says it should,
and GA is not being inflated by residual structure at this scale.

**Nothing survives multiple testing.** Ten tests were run. The strongest,
GHR x GH2 in EAS, is p = 0.010 uncorrected and 0.10 after Bonferroni; EUR is
0.040 and 0.40. Treat both as suggestive, not established.

**The ZP3 locus does not replicate in any population** (empirical p 0.267-0.733).
Given that ZP3R is now a pseudogene, that is the expected outcome rather than a
surprising one.
