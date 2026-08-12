# cugen.impute — measured results

Everything here was measured on an **NVIDIA A100 80GB PCIe** (RunPod, CA-MTL-3),
CUDA 12.8, CuPy 14.1.1, against **Beagle 5.5 (27Feb25.75f)** on Java 21 with
`nthreads=8` on a 252-core host. Raw JSON alongside this file.

Nothing below is projected. Where a number has not been measured it says so.

## Accuracy: matches Beagle

Browning et al. (2018)'s own fixture, rebuilt: 1000 Genomes phase 3 v5a
chromosome 20, two individuals from each of the 26 populations as targets (52),
the remaining 2,452 as the reference panel, restricted to diallelic SNVs with at
least one minor-allele copy in the reference, targets masked to the Illumina
Omni2.5 sites.

| | this build | paper |
|---|---|---|
| reference markers | 1,733,485 | 1,718,742 (0.86% apart) |
| target markers | 55,230 | 54,885 (0.63% apart) |

Reproducing those counts is treated as a checksum on the fixture, and it earned
that status: an earlier build using the 1000 Genomes panel hosted on Beagle's
own site — the obvious input — came out 64% short, because that copy is already
filtered. The paper's Web Resources cite the release FTP directly.

| aggregate dosage r2, imputed markers | |
|---|---|
| **cugen** | **0.9789** |
| Beagle 5.5 | 0.9791 |

Dose correlation between the two tools: **0.9990**.

By minor allele count in the reference panel — the paper's own metric, which is
r2 between the true non-major allele **on a haplotype** and the posterior allele
probability, not the per-genotype dosage r2 usually quoted:

| MAC | markers | r2 |
|---|---|---|
| 1 | 759,745 | 0.126 |
| 2 | 194,298 | 0.316 |
| 3–5 | 189,265 | 0.505 |
| 6–10 | 114,150 | 0.649 |
| 11–20 | 94,971 | 0.759 |
| 21–50 | 100,781 | 0.853 |
| 51–100 | 59,229 | 0.916 |
| 101–200 | 43,714 | 0.946 |
| 201–500 | 45,698 | 0.968 |
| 501–1000 | 40,174 | 0.978 |
| 1001–2000 | 53,250 | 0.982 |
| 2001+ | 21,874 | 0.978 |

## Speed: cugen 28.5s against Beagle's 32.6s — 1.14x

| phase | seconds | scales with |
|---|---|---|
| forward-backward (GPU) | 12.12 | targets x panel |
| write | 8.46 | output size |
| carriers (GPU) | 1.73 | reference panel only |
| summary | 1.39 | output size |
| read | 0.93 | reference panel only |
| aggregate | 0.61 | target markers |
| dose (GPU) | 0.11 | targets x carriers |
| **total** | **28.31** | |

This started at 76.7s — a **2.4x loss** — and the two phases responsible were
both reference-panel work that no target sample ever touches:

| phase | was | now | |
|---|---|---|---|
| carriers | 30.7s | 1.73s | 17.7x |
| read | 18.5s | 0.93s | 20x |

Neither was made cleverer. The carrier lists are now built on the GPU straight
from the packed bytes, so the host never expands a window to a (K, M) byte
matrix — 2.4 GiB per window that was read, written and scanned purely to
support a step that has since moved off the host entirely. Only the genotyped
columns are unpacked now: tens of thousands against hundreds of thousands.

The direction came from asking whether the reference panel could be encoded
once to make reading it cheaper. It already was a `.cugen`; the cost was
everything *derived* from it on every run.

Beagle solves the same problem the other way, with bref3 — a prepared binary
panel — and the 2018 paper attributes its sublinear scaling in panel size to
exactly that.

## Scaling: the fixed cost is the whole story

Compute only (no file I/O), 40,000 markers, 10% genotyped, K = 4,904:

| target haplotypes | seconds | ms per haplotype |
|---|---|---|
| 32 | 1.68 | 52.6 |
| 128 | 1.45 | 11.3 |
| 512 | 1.69 | 3.3 |
| 2,048 | 2.41 | 1.2 |
| 8,192 | 5.28 | **0.64** |

**256x the targets for 3.1x the time** — 82x better per haplotype. `carriers`
stays at ~1.0s throughout; only `dose` grows (0.29s to 2.97s).

Reference panel axis, T = 256:

| K | seconds |
|---|---|
| 1,000 | 0.52 |
| 2,500 | 1.00 |
| 4,904 | 1.41 |

Linear in panel size, as the brute-force state space implies.

**The crossover against Beagle has NOT been measured.** The curve above says
cugen's per-target cost falls steeply while Beagle's is roughly linear in target
count, so a crossover should exist — but "should" is not a measurement, and the
only honest statement today is that cugen loses at 52 targets and that its cost
per target falls 82x between 32 and 8,192 haplotypes. Measuring Beagle across
the same target axis is the next thing to do.

## Two parameters, settled by running Beagle rather than reading about it

Both disagreed between the 2018 paper and the 5.5 manual.

**`overlap`** — paper 4 cM, manual 2.0. Beagle's own chr20 windows convert to
[-0.00, 39.99], [37.99, 77.99] and [75.99, 108.29] cM: **overlap 2.00 cM,
window 40.00**, so the manual is right. cugen produced [-0.0, 40.0],
[38.0, 78.0], [76.0, 116.0] on the same map — an independent check on the
windowing as well as on the default.

**`err`** — paper a flat 1e-4, manual `theta/(2(theta+H))` with
`theta = 1/(0.5 + ln H)`, which is 1.133e-05 at H = 4,904. Imputing under both
and comparing against Beagle's output over 1,733,484 shared markers:

| err | corr with Beagle | mean abs diff |
|---|---|---|
| 1.133e-05 (manual formula) | **0.999014** | **0.000596** |
| 1.0e-04 (paper constant) | 0.998637 | 0.000739 |

The manual's formula agrees more closely on both metrics and is the default.
The margin is small, so this is evidence rather than proof.

## What optimisation actually achieved, and what it cost

| | seconds | note |
|---|---|---|
| first working version | 148 | |
| unpackbits reads, bulk float writes | 142 | read 47.6 -> 17.6, write 31.9 -> 7.8 |
| bit-packed allele codes | — | aggregate 53.4 -> 1.9 (28x) |
| chunked carrier build | **175** | **regression**, reverted |
| transposed carrier walk | **76.7** | carriers 46 -> 30.7 |

Two lessons, both paid for:

**Instrument first.** Every phase that mattered was a host phase. The GPU
kernels were never the bottleneck and optimising them further would have bought
nothing — the same mistake this project made in its previous round, where three
optimisation passes went into a GEMM that was 0.9% of runtime.

**A measurement that contradicts the plan is still a measurement.** The chunked
carrier build was benchmarked on synthetic data, came out 5x slower, and was
adopted anyway on the reasoning that real data at 2.6% density would behave
differently from synthetic at ~50%. It did not: 2.9x slower in situ, 4.7x in
isolation. The first measurement was right and overriding it cost a full
benchmark cycle.
