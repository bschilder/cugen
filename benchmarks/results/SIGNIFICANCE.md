# LD significance testing — correctness, cost, and what it changes

All numbers measured on the CPU reference path (`backend="numpy"`,
Apple M-series, Python 3.13, scipy 1.16) on this branch. Every figure below was
produced by a run; nothing is modelled.

**Two kinds of data appear below and they are labelled.** The unit tests and the
timings use **simulated** panels, which is the right instrument for pinning
arithmetic and for constructing a specific confound on demand. The section
"Real 1000 Genomes chr22" uses **real** data, because a simulated frequency
spectrum cannot tell you how the layer behaves on the rare-variant tail that
dominates real cohorts — and that turns out to be where the interesting
behaviour is.

There is **no plink2 comparison in this document, and that is not an omission**.
plink2 emits no LD p-values. Christopher Chang planned `{chi-square, df,
p-value}` columns for `--r2` and never shipped them
([plink2-users](https://groups.google.com/g/plink2-users/c/lME6ld4i4cQ)), and the
[LD documentation](https://www.cog-genomics.org/plink/2.0/ld) lists no
significance option. The closest published GPU work, *cheeta* (Genome Biology
2026, ~10^14 pairs on an RTX 4090), reports 1,195 discoveries at `r2 >= 0.8`
with no p-values and no multiple-testing correction at all. The oracle here is
scipy.

## What is computed

    chi2 = N_OBS * r^2,  1 df

Park (2019) eq. 1 writes it as `2n D^2 / (pA qA pB qB)`, the same quantity.
`N_OBS` counts **gametes** for the phased statistics and **individuals** for the
composite ones — the factor of two between gametic and composite LD — and cugen
already carried the right one on every path.

`NEG_LOG10_P` carries -log10(p), never p. Measured underflow points:

    p underflows float64 at chi2 ~ 1450        (-log10 p ~ 316)
    p underflows float32 at chi2 ~  170        (-log10 p ~  38)

Since `chi2 = N_OBS * r^2`, at 1000 Genomes size (N_hap = 5008) a float32 `P`
column dies at **r^2 = 0.034** and float64 at **r^2 = 0.29**. A `P` column would
read as a flat zero for essentially every linked pair genome-wide.

## Correctness

| quantity | oracle | agreement |
|---|---|---|
| `-log10(p)` | `log(2) + scipy.special.log_ndtr(-sqrt(chi2))` | < 1e-6 over chi2 in [0.5, 1e7] |
| `NEG_LOG10_P_EXACT` | `scipy.stats.fisher_exact` | < 1e-4, 40+ pairs |
| exact test | enumerated fixed-margin permutation null | < 1e-12 |
| BH-FDR | textbook rank-walk in log space | exact |
| `lambda_gc` | 1e5 true chi2(1) draws | 1.0022 |
| nAB from float32 `r` | direct counting, 250+ pairs at N=500 | 0 wrong |

`scipy.stats.chi2.logsf` is **not** a usable oracle: it computes `log(sf())` and
returns `inf` above chi2 ~ 1450 — the exact regime the helper exists to serve.
Placement of the erfc/asymptotic branch cut is a real accuracy knob: at chi2=30
the truncated expansion is good to 2e-4, at 400 it is 1e-7, and erfc is still
far from its 1450 underflow there.

## Cost — the asymptotic layer is free

*Simulated panel.* 400 variants x 2,000 samples, 79,800 pairs, median of 5 runs:

    r2 only                       352.4 ms   (baseline)
    + chi2, p                     350.3 ms    0.99x
    + max_p filter                354.2 ms    1.01x
    + BH-FDR                      356.1 ms    1.01x
    + lambda_gc                   350.7 ms    1.00x

Within noise, and by construction rather than luck. With no missingness every
pair shares one N, so chi2 is strictly monotone in r^2 and a p-value cut **is**
an r^2 cut: `max_p` is converted to `min_r2` and handed to the filter the kernel
already applies. The number of tests needs no pass over the data either — it is
`_count_pairs`, closed form from the row count and the window — which is what
makes Bonferroni and BH affordable at 10^14 tests. BH runs in -log10(p) space
because at that scale the thresholds are ~1e-16 and the p-values are
unrepresentable, so the textbook form compares two zeros.

## Cost — the exact conditional test

*Simulated panel.* 200 variants x 500 samples (1,000 haplotypes), 18,915 pairs, 20% of variants
rare (f = 0.03), median of 3:

    asymptotic only                24.7 ms   (baseline)
    exact='auto'                   70.7 ms    2.86x   fired on 6,032 pairs (32%)
    exact='always'                221.4 ms    8.95x   fired on all

`auto` fires where the minimum expected cell count is under 5. The gate is
self-limiting: `min(nA,nB)^2 <= nA*nB < 5N`, so the hypergeometric tail sum is
bounded by `sqrt(5N)` terms — about 158 at 1000 Genomes size, fewer for rarer
variants. The pairs that need the exact test are the pairs where it is cheap.

## What the exact test changes (simulated)

Over the 6,032 pairs where `auto` fired, comparing the two p-values:

    log10(p_exact / p_asymptotic)   median +0.272   p95 +1.369   max +215.7
    asymptotic OVERSTATES significance on 97.2% of them
    median ratio                     p_exact is 1.87x larger than p_asymptotic

    called significant at p < 5e-8:  asymptotic 111    exact 0

All 111 are false positives. The worst single pair is the whole argument:

    r^2 = 1.0000,  N_OBS = 1000
    asymptotic  -log10 p = 218.7      (chi2 = N * r^2 = 1000)
    exact       -log10 p =   3.0

One haplotype carrying both alt alleles gives r^2 = 1 exactly. The asymptotic
test reads that as p = 1e-219; the honest answer is p = 1e-3, because with a
single copy of each allele a perfect table has probability 1/N. This is the
`r^2 >= 0.8` trap in one line — a hard r^2 threshold cannot distinguish real
disequilibrium from a singleton coincidence, and neither can chi2.

## What inflation control changes (simulated)

*A confound built on purpose.* Two subpopulations differing by dAF = 0.60, pooled, n = 2,000, 120 variants,
each variant drawn **independently within each subpopulation** — so there is no
gametic LD anywhere in the fixture and every significant call is a false
positive:

    lambda_gc                        920.6
    median -log10 p, raw              92.4
    median -log10 p, adjusted          0.30
    pairs at p < 5e-8, raw         7,140 / 7,140   (100%)
    pairs at p < 5e-8, adjusted        0 / 7,140

Calibration on data that really is null: `lambda = 0.967` on unlinked genotypes
at n = 4,000, and 1.0022 on 1e5 true chi2(1) draws.

`lambda_gc` is **off by default**. It is estimated over the more distant half of
pairs by index separation, and always before any significance filtering —
filtering selects the tail, and the median of a selected tail says nothing about
the null.

## A threshold worth seeing

From a real run, 60 variants x 500 samples, m = 1,770 tests:

    cugen ld chr22.cugen --correction bonferroni
    -> max_p = 2.82e-05  ->  min_r2 = 0.0351

A *rigorous* family-wise threshold at this scale corresponds to **r^2 = 0.035**,
two orders of magnitude below the r^2 >= 0.8 heuristic the field reaches for.
The conventional cut is not conservative; it is answering a different question.
Emitted row counts on that run: 1,770 unfiltered, 1,347 at BH-FDR 0.05, 967 at
Bonferroni 0.05 — the expected ordering.

## Real 1000 Genomes chr22

1kGP high-coverage phased panel (`20220422_3202_phased_SNV_INDEL_SV`),
chr22:20–21 Mb, biallelic SNVs, streamed with `bcftools` rather than downloaded.
Reproduce with `uv run --with cyvcf2 python benchmarks/significance_1kg.py --dir DIR`;
the fixture commands are in `tests/data/README.md`.

    EUR       503 samples, 800 variants, MAF >= 0.01 within EUR
    EAS       504 samples, 800 variants, MAF >= 0.01 within EAS
    EUR+EAS  1007 samples, 800 variants, MAF >= 0.01 in the pooled sample
    EURrare   503 samples, 800 variants, NO frequency filter

### chi2 against plink2's own r^2, 319,600 real pairs

plink2 emits no p-value, so this validates the **statistic** against an
independent LD implementation and leaves the tail to scipy.

    pairs joined                    319,600 of 319,600 / 319,600   (identical sets)
    |chi2_cugen - N_hap * r2_plink|  max 9.997e-04   median 7.255e-06
    relative                         max 9.977e-06
    -log10(p) from plink2's r2       max 2.145e-04

9.98e-06 relative is plink2's six-significant-figure text floor, the same limit
`PHASED.md` reports for r^2 itself — not a difference in arithmetic.

Join on **variant ID, not position**. Split multi-allelic sites share a POS, so a
`(POS_A, POS_B)` key is not unique, the merge silently becomes many-to-many
(322,796 rows out of 319,600 × 319,600) and the max error comes out as 954
instead of 1e-3. `compare_plink.py` documents the same trap from the other
direction, where every 1KG chr22 ID was `.`.

### The exact test on a real frequency spectrum — EUR, no MAF filter

    pairs 32,640      auto fired on 26,844  (82.2%)
    vs scipy.stats.fisher_exact, 400 real pairs:   max |diff| 1.161e-06

**`auto` fires on 82% of real pairs, against 32% on the simulated panel.** Real
1KG is dominated by rare variants, so the regime where the asymptotic test is
untrustworthy is the common case, not a corner.

    asymptotic OVERSTATES significance on 97.9% of them
    log10(p_exact / p_asym)   median +0.110   p95 +0.539   max +217.0
    called significant at p < 5e-8:   asymptotic 644     exact 387

**257 of 644 genome-wide-significant calls — 40% — are false positives the exact
test removes, on real data.** The worst pair is the singleton trap in the wild:

    real pair, r^2 = 1.0000, N = 1,006 haplotypes
    asymptotic  -log10 p = 220.0
    exact       -log10 p =   3.0

r^2 = 1 with a single copy of each allele. `chi2 = N * r^2` hits its maximum
regardless of how few copies produced it, so the asymptotic test reads p = 1e-220
where the honest answer is p = 1e-3. This pair exists in 1000 Genomes chr22.

### lambda_gc across real populations

    population   n      pairs      lambda    p<5e-8 raw -> adjusted
    EUR          503    319,600     13.97      30.5%  ->   5.3%
    EAS          504    319,600     15.11      32.4%  ->   5.1%
    EUR+EAS     1007    319,600     29.24      43.2%  ->   4.3%

Read this carefully, because it does **not** say what the simulated
lambda = 920 said. Within a single population lambda is already ~14 — and that is
mostly **real LD**, not confounding: every pair here sits inside a 1 Mb window,
so the "most tests are null" assumption behind genomic control is violated by
construction. The structure signal is the **ratio**: pooling two populations
takes lambda from ~14 to 29.2, a 2.1x inflation on top of whatever the
within-population baseline is.

So on real windowed data lambda is a **diagnostic, not a correction to apply
blindly** — it cannot separate pervasive true LD from stratification. It is
opt-in for exactly this reason. The clean use is an unwindowed or
trans-chromosome scan, where the null assumption holds.

### Multiple testing on real chr22 LD, m = 319,600

    none              319,600 pairs (100.0%)    min r2 retained  0.00000
    Bonferroni 0.05   101,380 pairs ( 31.7%)    min r2 retained  0.02735
    BH-FDR 0.05       208,415 pairs ( 65.2%)    min r2 retained  0.00454
    r2 >= 0.8 (convention)  8,523 pairs (2.667%)

The conventional cut keeps **2.7%** of pairs; a family-wise-controlled threshold
keeps **31.7%**, and its actual r^2 boundary is **0.027**. So `r2 >= 0.8` is not a
conservative version of a significance test — it is roughly 12x stricter in what
it retains while providing no error control at all, and it simultaneously admits
the singleton pairs above that no error rate would tolerate. It is strict in the
wrong place and permissive in the wrong place.

## Bias-corrected r^2 — estimators, deliberately without p-values

`r2_s`, `r2_v`, `r2_vs` implement Mangin et al. (2012) *Heredity* 108:285–291 —
r^2 corrected for population structure, for relatedness, and for both. They are
the other half of the robustness story: `lambda_gc` adjusts the *test* for
inflation, these correct the *estimator*.

All three are ratios of entries of a covariance-like matrix that is the Gram
matrix of a linearly transformed genotype vector, so each reduces to **one n×n
map applied once**, after which an ordinary uncentered r^2 is the answer:

    r2_s   P = (I - H_S)(I - 11'/n)                centre, then residualise on S
    r2_v   P = W (I - F),  F = 1 1' V^- / (1' V^- 1)   GLS-centre, then whiten
    r2_vs  P = (I - H_Z) W (I - F)                 the same, in the V^- metric

Validated against the authors' own R implementation (LDcorSV 1.3.3,
`Measure.R2V` / `R2S` / `R2VS`) transcribed literally into numpy — max |diff|
< 1e-6 over 190 pairs each. With V = I, `r2_v` collapses onto ordinary `r2`
(atol 1e-6); with a structure matrix uncorrelated with the loci, `r2_s` leaves
`r2` alone.

Two independent checks of the correction actually correcting:

    two populations, dAF = 0.70, no true LD within either (20 variants, n = 200)
      mean r2    0.2719      (pure structure artefact)
      mean r2_S  0.0074      (corrected)

    Mangin Table 2 clone scenario, true r2 = 0.05, 80 independent + 20 clones
      r2, independent samples only   0.0587
      r2, with the clones            0.0769
      r2_v, same sample + kinship    0.0587      (paper reports 0.060 / 0.063)

**They carry no p-value, and asking for one raises.** Main text and
Supplementary Information both establish only unbiasedness for unlinked loci
(Appendix A) and a power result — Appendix B shows the *association* t-test
carrying S as a covariate is asymptotically Gaussian with variance 1 and
expectation `sqrt(r2_S)·Esp(t_causal)`, which is a statement about that test, not
about the LD measure. No null sampling law for the measures themselves.

A route exists and was rejected deliberately, which is worth stating so nobody
re-derives it. `r2_s` is an ordinary squared **partial** correlation, so
classical normal theory gives `r²/(1−r²)·(N−K−2) ~ F(1, N−K−2)`. But that needs
joint normality, which genotypes do not have — whereas `chi2 = N·r²` rests on the
multinomial structure of the 2×2 table and needs no such assumption. And the
real-data section above measures that asymptotic test **overstating significance
on 97.9% of rare-variant pairs**, with the exact conditional test as the thing
that rescues it. Residualizing destroys the contingency table, so the exact test
is unavailable and the remaining option is *weaker* than an approximation already
known to fail on this data. It would look like added rigour and be the opposite.
For `r2_v` the effective degrees of freedom after whitening by a rank-deficient
`V⁻` is not even clear.

Note that the paper's Table 1 numbers are **not** reproduced here, and the reason
is informative. Pooling two populations induces Hardy–Weinberg disequilibrium
(Wahlund), so the composite genotypic r^2 exceeds the gametic one — a
two-population sample with true r^2 = 0.01 and frequencies 0.9/0.1 at both loci
gives 0.65 on dosages against the paper's 0.460, which is the gametic value. The
paper does not say which its simulation produced, so matching that table would be
testing a reconstruction of their simulation rather than an implementation of
their formula. The structural claim is tested instead.

Practical limits: `r2_v` needs an n×n eigendecomposition, which is fine at the
hundreds-to-thousands scale the paper worked at (183 grapevine accessions; the
authors note inverting V "drastically slowed down the computation") but not at
biobank n. `r2_s` is rank-K and has no such limit. Both force the reference path.
Dosage input only — the correction is defined between individuals, and a hap2bit
file is indexed by haplotype. Which V to use is, in the authors' words, "an open
question".

## Caveats that travel with these numbers

- All timings are the **CPU reference path**. The asymptotic statistics also run
  on the fused GPU path (they are a few elementwise ops on `r` and `N_OBS` in
  `_assemble_device`); `p_exact` and `lambda_gc` deliberately force the
  reference path, as `d`/`dp` already do.
- GPU paths are untested on this branch — no CUDA device was available. The
  `@requires_cudf` tests skip rather than fail.
- `lambda_gc` assumes most tested pairs are null. That holds for an unwindowed
  chromosome scan; inside a tight window it is false, which is why the estimate
  uses the distant half and why the flag is opt-in. The real-data numbers above
  show this concretely: lambda ~ 14 within one population at 1 Mb is mostly true
  LD, so only the pooled/single ratio is interpretable as structure.
- The real-data section is one 1 Mb window of chr22 in two populations. It is
  enough to show the rare-variant behaviour and the pooling effect; it is not a
  genome-wide or multi-cohort characterisation.
- The exact test is hap2bit only. Dosage data carries no 2x2 haplotype table, so
  there is nothing to condition on, and asking for it there raises.
- Missing calls make N vary per pair, which breaks the monotone p<->r^2
  shortcut. The conversion is then used as a conservative pre-filter and the
  exact per-pair cut is applied after the scan.

## Reproducing

    uv run pytest tests/test_ld_significance.py tests/test_popstruct.py -q
    uv run python tests/mutation_sweep.py        # 28 mutations, 0 missed

    # real data (streams ~1 Mb of chr22 from EBI; needs bcftools + plink2)
    uv run --with cyvcf2 python benchmarks/significance_1kg.py --dir DIR
