# Sample axis to 1,000,000 — cugen (phased + unphased) vs plink2

p is FIXED at 4,000 so the axis is genuinely n_samples. Synthetic phased
haplotypes from a latent-factor model, so real LD exists and the emitted row
count (which drives write cost) stays comparable across n -- roughly 0.5-1.0 M
rows at every point.

cugen: 1x NVIDIA A100-SXM4-80GB (Runpod, $1.59/hr).
plink2: AWS c7a.32xlarge, **128 physical cores** (SMT off, no cgroup quota,
125.2 effective cores measured), $6.56896/hr from the AWS Pricing API.
plink2 run with `--threads 128`.

| n_samples | haplotypes | cugen 2bit s | cugen hap2bit s | plink2 s | cugen speedup |
|---|---|---|---|---|---|
| 2,504 | 5,008 | 0.5633 (*) | 0.0643 | 0.092 | - |
| 50,000 | 100,000 | 0.1891 | 0.1838 | 0.391 | 2.1x |
| 100,000 | 200,000 | 0.3369 | 0.2567 | 0.766 | 2.3x |
| 250,000 | 500,000 | 0.4201 | 0.4821 | 2.194 | 5.2x |
| **500,000** | 1,000,000 | **0.6808** | 0.8268 | **4.310** | **6.3x** |
| 1,000,000 | 2,000,000 | 1.1037 | 1.3238 | 8.510 | 7.7x |

(*) The n=2,504 unphased point includes CUDA kernel compilation, because it is
the first `ld_matrix` call in the process. It is not a measurement. The same
artifact makes `psweep.json`'s p=1,000 row slower than its p=5,000 row.

## Cost at biobank scale (n = 500,000)

    cugen    0.681 s x $1.59/hr / 3600 = $3.01e-04
    plink2   4.310 s x $6.57/hr / 3600 = $7.86e-03      26.1x

Break-even: a 128-core host would have to cost **$0.251/hr** to match cugen.

## The sample axis is cugen's WEAKEST axis, and that is the point

6.3x at n=500,000 against 24x (unphased) and 562x (phased) on the variant axis
at full chr22. The asymmetry is structural:

* growing n lengthens the GEMM's contraction dimension -- the dimension a GPU
  is already efficient in, and the one plink2 also handles well. plink2 scales
  nearly linearly here (20x the time for 20x the samples, 0.391 s -> 8.510 s),
  so neither tool degrades.
* growing VARIANTS grows pairs quadratically. That is where tiling, the fused
  epilogue and single-pass atomic compaction win, and where plink2's per-pair
  costs and p^2 memory growth hurt.

A single "cugen is Nx faster than plink2" figure is a point on a surface, not a
property of the tools.

## Phased vs unphased on this axis

The phased path operates on a plane TWICE as wide (2n haplotype columns against
n dosage columns) yet costs 0.76x-1.21x, not 2x. At p=4,000 the GEMM is not the
bottleneck: the per-variant moments pass is, and `hap_moments` is a pure
popcount while `_variant_moments` computes both sum(x) and sum(x*x). Below
n=100,000 phased is actually FASTER; above n=250,000 it settles at ~1.2x.

At n=1,000,000 the plane is 2,000,000 haplotype columns wide and `_tile_size_for`
sizes tiles for it with no intervention; peak GPU memory stayed at 441 MiB.

## Caveats

* Synthetic genotypes, not a real biobank cohort. LD structure comes from one
  latent factor plus per-variant noise, which is not the LD structure of a real
  population -- it is designed to keep emitted row counts stable across n.
* plink2 numbers are single runs; cugen figures are medians of two.
* The two tools ran on different hosts (A100 pod, AWS instance). They do not
  contend, and the fixture is generated identically by the same simulation, but
  this is not a same-host comparison the way the chr22 sweeps are.

## Reproducing

    python benchmarks/nsamples_phased.py --p 4000      # cugen, needs a GPU
    python benchmarks/biobank_plink2.py --p 4000       # plink2, writes .bed directly
    python benchmarks/cost_analysis.py                 # $/job from measured times
