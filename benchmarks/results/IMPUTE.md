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

## Speed

There is **one** comparison here where both tools do the same job, and it is not
a wall-clock comparison of the two runs.

### The comparable number: imputation

Beagle reports its own imputation time per window (5 + 4 + 2 s), separately from
its I/O. cugen's corresponding phases are `forward_backward` 12.12 +
`carriers` 1.73 + `aggregate` 0.61 + `dose` 0.11.

| | imputation |
|---|---|
| Beagle 5.5 | **11.0 s** |
| cugen | 14.6 s |

**Beagle is about 1.3x faster**, consistent with it running
`imp-states=1600` PBWT-selected states where this runs all 4,904 reference
haplotypes: roughly 3x the state space, most but not all of it absorbed by the
GPU.

Caveat on this number too: Beagle's documentation does not say what its
per-window imputation counter starts and stops on, in particular whether it
includes reading that window's reference markers -- the work cugen is charged
1.73 + 0.93 s for above. So this is approximately like-for-like, not exactly.

### The two wall-clock numbers are NOT a comparison

cugen's run took 28.5 s and Beagle's 32.6 s, but these measure two different
pipelines with no shared start or finish:

| | cugen 28.5 s | Beagle 32.6 s |
|---|---|---|
| reads | `reference.cugen`, 1.1 GB uncompressed binary, mmap | `reference.vcf.gz`, 108 MB bgzipped text |
| writes | `cugen_out.cugen`, ~205 MB float16 dosages | `beagle_out.vcf.gz`, bgzipped text with GT, DS, INFO |
| process | warm Python, CuPy context already initialised | cold JVM, `-Xmx60g` |
| excluded before the timer | VCF to `.cugen` conversion: **598 s** reference, 3.8 s target | nothing |

One decompresses text and recompresses text; the other mmaps a binary blob and
writes raw float16. And **bref3 -- Beagle's own prepared binary format, the
direct analogue of a `.cugen` -- was not used**, though the 2018 paper credits
it for Beagle's sublinear scaling in panel size.

Quoting 28.5 against 32.6 as though it were a ratio, which an earlier version of
this file did, compares cugen's fast format against Beagle's slow one and
charges only Beagle for its I/O.

### Full cugen profile at 28.31 s

| phase | seconds | scales with |
|---|---|---|
| forward-backward (GPU) | 12.12 | targets x panel |
| write | 8.46 | output size |
| carriers (GPU) | 1.73 | reference panel only |
| summary | 1.39 | output size |
| read | 0.93 | reference panel only |
| aggregate | 0.61 | target markers |
| dose (GPU) | 0.11 | targets x carriers |

This began at 76.7 s. The two phases responsible were reference-panel work no
target sample touches:

| phase | was | now | |
|---|---|---|---|
| carriers | 30.7 s | 1.73 s | 17.7x |
| read | 18.5 s | 0.93 s | 20x |

Those are cugen measured against its own prior state on identical inputs, so
they are unaffected by everything above. Carrier lists are now built on the GPU
straight from the packed bytes; the host never expands a window to a `(K, M)`
byte matrix, 2.4 GiB per window read and written and scanned to feed a step that
has since moved off the host.

### Not measured

- **Beagle with a bref3 panel.** Required before any two-pipeline wall-clock
  number here is worth stating. Expect Beagle to get faster.
- **A pipeline comparison from a common starting artifact.**
- **Either scaling axis, for Beagle.** `benchmarks/target_axis.py` and
  `benchmarks/panel_axis.py` exist to fix this by running Beagle 5.5 directly.
  Until they have, nothing here says how Beagle scales.

  Note on what is NOT usable as a substitute: Browning et al. (2018) publish
  Beagle scaling sublinearly in panel size (1,000x the reference samples for 11x
  the time), and an earlier version of this file leaned on it. That figure is
  **Beagle 5.0**, seven years and several releases before the 5.5 benchmarked
  here, including the 2021 phasing rewrite. It is a claim about a different
  program and is not cited as evidence about this one.

## Not measured

- **Beagle with a bref3 panel.** Required before any two-pipeline wall-clock
  number here is worth stating. Expect Beagle to get faster.
- **A pipeline comparison from a common starting artifact** (say, both from
  `.vcf.gz`, or both from their own prepared format, with conversion counted).
- **The crossover on the target axis.**

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
