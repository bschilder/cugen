# Phased LD statistics — correctness and speed

All numbers from one NVIDIA A100-SXM4-80GB (driver 580.159.04, CuPy 14.2.0,
cuDF present so the fused path is live) against plink2 v2.0.0-a.7.0 AVX2 on the
same host. Fixture: 1000 Genomes phase 3 chr22, `--max-alleles 2 --snps-only
--maf 0.01` -> **170,949 variants x 2,504 samples**, reproduced independently on
three separate machines.

cugen reads a **hap2bit** `.cugen` (observed phase, built by `vcf2cugenh` from
the phased VCF). plink2 reads a **phased PGEN** -- not a `.bed`. That distinction
is the whole experiment: `.bed` cannot store phase, so `--r2-phased` on bed input
EM-estimates haplotype frequencies and answers a different question. Comparing
against bed produced a 0.616 max deviation; comparing against a phased PGEN
produces 5.7e-07.

## Correctness, p = 1,000

    cugen rows        19,888
    plink2 rows       19,888
    shared            19,888      cugen_only 0   plink_only 0
    |dr2| > 1e-4      0  (0.000%)
    max |dr2|         5.710e-07
    median |dr2|      2.510e-07
    worst pair        (16505854, 16582763)  cugen=0.78501457  plink=0.785014

Identical pair sets, zero disagreements. 5.71e-07 is the six-significant-figure
floor of plink2's TEXT output (the same limit issue #2 documents for unphased r),
not a difference in arithmetic -- the worst pair differs in the seventh digit.

## Speed

| p | pairs | cugen s | plink2 s | speedup | rows (both) |
|---|---|---|---|---|---|
| 1,000 | 4.995e5 | 0.0175 | 0.0867 | 5.0x | 19,888 |
| 5,000 | 1.250e7 | 0.0780 | 0.4525 | 5.8x | 122,389 |
| 20,000 | 2.000e8 | 0.1677 | 12.4072 | 74.0x | 495,015 |
| 50,000 | 1.250e9 | 0.4044 | 121.9037 | 301.4x | 1,413,182 |
| **170,949** | **1.461e10** | **1.8038** | **1013.8759** | **562.1x** | **5,864,576** |

Row counts match exactly at every point.

## Why phased scales BETTER than unphased

Unphased on the same grid tops out near 45x against plink2; phased reaches 562x.
The costs are asymmetric:

* cugen's phased path is the SAME fused GEMM on a plane twice as wide (5,008
  haplotype columns against 2,504 dosage columns), so it costs about 2x. The
  measured 1.804 s against 1.734 s unphased is exactly that.
* plink2's `--r2-phased` must solve a haplotype-frequency problem per pair,
  which is far more than a dosage correlation. 1013.9 s against 42.2 s.

cugen pays a factor of two; plink2 pays a factor of twenty-four.

The algebra is why the GPU side is nearly free. For 0/1 indicators
`sum(x*x) == sum(x)`, so the Hill & Robertson allele-count correlation the
existing epilogue already computes collapses to the haplotypic identity exactly:

    r = (H*nAB - nA*nB) / sqrt(nA(H-nA) * nB(H-nB))
      = D / sqrt(pA qA pB qB)

so no new kernel was needed -- only a haplotype plane builder and `q_v = s_v`.

## Caveats

* `d_phased`/`dp_phased` are NOT on the fused path. The fused epilogue returns
  one correlation per surviving pair and no D, so those fall back to the
  reference path. They are correct, not fast; the table above is r/r2 only.
* N_OBS is 2*n_samples unconditionally on the phased path. hap2bit has no spare
  code for missing (all four codes are meaningful), so a phased file cannot
  carry missingness and every pair is complete by construction.
* Single runs for plink2; cugen figures are medians of three.
* n = 2,504 is cugen's worst regime -- the GEMM contraction dimension is small.
  The sample axis is a separate sweep.

## Reproducing

    bash benchmarks/phased_bench.sh        # needs a phased VCF and a phased PGEN
