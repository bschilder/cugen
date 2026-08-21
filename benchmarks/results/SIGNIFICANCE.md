# LD significance testing — correctness, cost, and what it changes

All numbers measured on the CPU reference path (`backend="numpy"`,
Apple M-series, Python 3.13, scipy 1.16) on this branch. Every figure below was
produced by a run; nothing is modelled.

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

400 variants x 2,000 samples, 79,800 pairs, median of 5 runs:

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

200 variants x 500 samples (1,000 haplotypes), 18,915 pairs, 20% of variants
rare (f = 0.03), median of 3:

    asymptotic only                24.7 ms   (baseline)
    exact='auto'                   70.7 ms    2.86x   fired on 6,032 pairs (32%)
    exact='always'                221.4 ms    8.95x   fired on all

`auto` fires where the minimum expected cell count is under 5. The gate is
self-limiting: `min(nA,nB)^2 <= nA*nB < 5N`, so the hypergeometric tail sum is
bounded by `sqrt(5N)` terms — about 158 at 1000 Genomes size, fewer for rarer
variants. The pairs that need the exact test are the pairs where it is cheap.

## What the exact test changes

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

## What inflation control changes

Two subpopulations differing by dAF = 0.60, pooled, n = 2,000, 120 variants,
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

## Caveats that travel with these numbers

- All timings are the **CPU reference path**. The asymptotic statistics also run
  on the fused GPU path (they are a few elementwise ops on `r` and `N_OBS` in
  `_assemble_device`); `p_exact` and `lambda_gc` deliberately force the
  reference path, as `d`/`dp` already do.
- GPU paths are untested on this branch — no CUDA device was available. The
  `@requires_cudf` tests skip rather than fail.
- `lambda_gc` assumes most tested pairs are null. That holds for an unwindowed
  chromosome scan; inside a tight window it is false, which is why the estimate
  uses the distant half and why the flag is opt-in.
- The exact test is hap2bit only. Dosage data carries no 2x2 haplotype table, so
  there is nothing to condition on, and asking for it there raises.
- Missing calls make N vary per pair, which breaks the monotone p<->r^2
  shortcut. The conversion is then used as a conservative pre-filter and the
  exact per-pair cut is applied after the scan.

## Reproducing

    uv run pytest tests/test_ld_significance.py tests/test_popstruct.py -q
    uv run python tests/mutation_sweep.py        # 28 mutations, 0 missed
