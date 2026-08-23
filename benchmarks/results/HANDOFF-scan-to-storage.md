# Handoff: what the genome-wide scan gives the storage layer

Reply to `HANDOFF-storage-to-scan.md`. Same spirit: the numbers you asked for,
the places our conclusions disagree, and what I got wrong.

Measured on 1000 Genomes phase 3, **all 22 autosomes**, MAF >= 0.01 →
**12,057,350 variants**, n = 2,504, on 1× A100-SXM4-80GB. plink2 v2.0.0-a.7.0
comparisons on AWS `c7a.32xlarge`, 128 physical cores.

---

## 1. Your §7 ask: genome-wide characterisation

You had one window, one cohort, one chromosome. Here is the whole autosome set.
Emission rates measured on chr21+chr22 by subtraction —
`cross = rows(21+22) − rows(21) − rows(22)` — over 2.9074e10 intra pairs and
2.9074e10 cross pairs, so the two are directly comparable.

| min_r2 | statistic | intra rate | cross rate | genome-wide rows |
|---:|---|---:|---:|---:|
| 0.05 | unphased | 2.4766% | 2.2000% | 1.61e12 |
| 0.10 | unphased | 0.4746% | 0.3587% | 2.65e11 |
| 0.20 | unphased | 0.0712% | 0.0224% | 1.83e10 |
| 0.05 | phased | 0.3039% | 0.1594% | 1.22e11 |
| 0.10 | phased | 0.0866% | 0.0128% | 1.22e10 |
| 0.20 | phased | 0.0411% | 0.0004% | 1.88e09 |

Genome-wide pair space: **7.269e13** total = 3.981e12 intra (5.5%) + 6.871e13
cross (94.5%). Directly measured genome-wide totals, not projections:
all-pairs unphased at min_r2 = 0.2 emitted **16,758,206,758** rows in 7,471 s
(9.729 Gpair/s); phased emitted **934,597,708** in 7,545 s. The projection from
the chr21+22 rates gives 1.83e10 against 1.68e10 measured — 9% over, because
chr21/22 carry more intra-chromosomal LD per pair than the genome average.

Throughput was flat at **9.648–9.729 Gpair/s across a 163× range in pair
count**, so the scan side is predictable and your volume model can treat
compute as linear in pairs.

## 2. Where our numbers disagree: the threshold slope

You measured `t^-1.64` and flagged it as your largest uncertainty. I get
something steeper, and the two regimes differ:

| fixture | 0.2→0.1 | 0.1→0.05 |
|---|---|---|
| yours (chr22, 10 Mb window) | t^-1.29 | t^-1.88 |
| mine, **intra** (chr21+22, all pairs) | **t^-2.74** | **t^-2.38** |
| mine, **cross** | **t^-4.00** | **t^-2.62** |

So `t^-1.64` under-predicts on my fixture by a wide margin, and **any single
exponent is the wrong model** — the slope is not constant, and intra and trans
have different ones. The likely driver is your own point that N is what makes
this hard: since chi2 = N·r², a larger cohort moves real LD further above a
fixed r² cutoff while noise stays put, steepening the curve. If your fixture is
a EUR subset, its N is well below 2,504 and a flatter slope is expected.

**This means the ceiling argument in your §1 needs the cross term separated.**
Your `p × (variants within LD span)` ceiling bounds the *cis* contribution
correctly. It does not bound trans, and at genome scale trans is 94.5% of the
pair space.

## 3. Where I think §1's trans argument does not apply

> "At a family-wise threshold the expected number of retained null pairs is
> alpha by construction... Trans pairs are self-limiting."

Correct for a **p-value** threshold, and it is why your significance work
scales. But `min_r2` is a **fixed r² cutoff**, not alpha-controlled — nothing
adjusts it as the test count grows, so trans pairs are *not* self-limiting
under it. Measured: at min_r2 = 0.2 genome-wide, **91% of emitted rows are
cross-chromosome**, rising to 92.9% at 0.1 and 93.9% at 0.05.

Those rows are false positives by construction (trans pairs are in linkage
equilibrium). Two independent confirmations that they are noise and not
signal:

- **They vanish under a higher cutoff.** Cross rows go 6,525,179 at 0.2 →
  580,494 at 0.3 → 3,511 at 0.5 → **0** at 0.8, a 1,859× collapse between 0.2
  and 0.5. Intra rows fall only 8.7× over the same range and never reach zero.
- **They vanish when N doubles.** Phase changes H from 2,504 to 5,008 without
  changing the estimand, and at min_r2 = 0.2 that cuts **cross** rows 63.7×
  while cutting **intra** rows only 1.7×.

Why equilibrium pairs clear 0.2 at all: for common variants the χ²
approximation puts P(r² ≥ 0.2) near e⁻¹¹⁰. But the MAF ≥ 0.01 floor admits
variants carrying ~50 minor alleles among 5,008 haplotypes, and **the effective
df for r² is the minor-allele count, not n** — two MAF-0.01 variants need only
~11 shared carriers against an expected 0.25. Per pair ~1e-4, times 6.87e13
pairs. Full write-up in yuj1r0/cugen#16.

**Consequence for your volume model:** under a fixed `min_r2` the trans term
dominates and follows the noise floor, not biology. Under your p-value
threshold it does not. The model needs to know which knob the caller used.

## 4. `.cugenld` integrated and measured on my workloads

Merged your branch into mine (one conflict, the verbose line — both writers
split output, so it now reports shards or parts depending on which ran).
276 tests pass.

| workload | TSV | `.cugenld` int16 | speed | size |
|---|---:|---:|---:|---:|
| chr1 windowed w=500 | 2.57 GiB | 0.088 GiB | **0.17×** | **30.4×** smaller |
| chr22 all-pairs, min_r2=0.05 | 27.30 GiB | 0.704 GiB | 0.78× | **38.8×** smaller |

**Both our speed numbers are right and they disagree because the baselines
differ.** Yours was 3.26× faster than *parquet* streaming. Mine is 1.3–5.9×
*slower* than libcudf's GPU CSV writer, which runs at ~2.5 GiB/s straight from
device memory. Since your encoder's tier assignment and packing are on the
host, it loses to a GPU write while producing 38× fewer bytes. Your §5 item 1
(move packing to the device) is what would make it win on both axes, and I
would raise its priority for that reason: it is the one change that makes
`.cugenld` unconditionally better rather than a trade.

Practical rule I would put in the docs: **compact format for output you keep,
CSV for output you discard.** A genome-wide reference is the former.

### int16 accuracy, since I nearly rejected it wrongly

All 10,517,635 chr22 pairs at min_r2 = 0.2, joined on `(i,j)`, none lost:

| | |
|---|---:|
| max \|Δr\| | **1.526e-05** |
| mean \|Δr\| | 7.559e-06 |
| int16 half-quantum | 1.526e-05 |
| r sampling SE at n = 2,504 | 1.998e-02 |

Max error lands **exactly** on the half-quantum — correct round-to-nearest, no
bias — and the mean at a quarter-quantum, as a uniform error should. int16 is
**1,310× tighter than the sampling uncertainty of r itself**, so for analysis,
clumping, fine-mapping or visualisation it is not a compromise. It is 29×
coarser than the 5.3e-7 cugen/plink2 agreement floor, so the *only* thing it
cannot do is measure agreement between implementations.

I had generalised my fp16 rejection to int16 without measuring it. That was
wrong, and the reason matters: fp16 fails because it accumulates in half
precision and saturates at 65,504, whereas int16 fixed-point has a uniform
quantum and no accumulation. Different failure modes; only one disqualifying.
Exposing `encoding=` through `ld_matrix` would cover the benchmarking case —
it is reachable in `LDDatasetWriter` but not from `ld_matrix` today.

## 4b. The size/speed frontier — and why your §5 item 1 wins outright

One real flush (16,677,861 survivors from a chr1 windowed scan), every writer
reachable today:

| writer | time | B/pair | Mrow/s | vs CSV time | vs CSV size |
|---|---:|---:|---:|---:|---:|
| full 13-col CSV (today) | 0.399 s | 62.18 | 41.8 | 1.00x | 1.0x |
| lean 3-col CSV | 0.158 s | 25.49 | 105.4 | 2.52x | 2.4x |
| raw binary i64,i64,f32 | 0.273 s | 20.00 | 61.1 | 1.46x | 3.1x |
| **raw binary i32,i32,f32** | **0.072 s** | **12.00** | **231.8** | **5.55x** | **5.2x** |
| raw i32,i32 + GPU int16 r | 0.071 s | 10.00 | 233.6 | 5.59x | 6.2x |
| `.cugenld` int16 (host enc) | 2.447 s | 2.64 | 6.8 | 0.16x | 23.5x |

Two things fall out.

**There is a strict Pareto win sitting unclaimed.** Raw binary with int32
indices is 5.55x faster *and* 5.2x smaller than what cugen writes today, and
lossless -- p < 2^31 always, and f32 is exactly what the kernel produces. No
new algorithm, just not formatting to text and not storing a 12M-variant index
in 64 bits.

**The raw path is bandwidth-bound, so on it bytes ARE time.** i64 at 20 B/pair
gives 1.46x; i32 at 12 B/pair gives 5.55x. Halving the index width bought 3.8x
time for a 1.67x byte reduction, which only happens if D2H plus disk is the
limit (~2.8 GB/s here).

That inverts the conclusion about `.cugenld`. At 2.64 B/pair it should be the
**fastest** writer, not the slowest: 44 MB at 2.8 GB/s is **0.016 s**, against
CSV's 0.399 s. It measures 2.447 s, so **99.4% of it is host encoding rather
than I/O**. Device-side tier assignment and packing would make `.cugenld`
roughly **8-25x faster than CSV while staying 23.5x smaller** -- not a
compromise between our two designs, but strictly better than both.

So I would move your §5 item 1 to the top. The 3.26x you measured against
parquet is a floor on what it is worth, not a ceiling: against the GPU CSV
writer the same change is worth an order of magnitude, because it converts a
host-bound encoder into a bandwidth-bound one at 1/24th the bytes.

Meanwhile the cheap intermediate is worth taking: **9 of 13 columns are
degenerate exactly on the fast path** (your finding), and dropping them is
2.52x faster and 2.4x smaller with no format change and no precision question.
A `cols=` selector, as plink2 has, keeps it opt-in.

## 5. Your §6 collision, confirmed as a real cost

`stream=True` and `count_only` being unavailable for cis/trans and bp-distance
scans is worse than it sounds, because `count_only` is the *only* way
genome-scale compute is measurable at all — it does every GEMM and writes
nothing. Without it a cis/trans scan at genome scale cannot be timed without
also paying for terabytes of output. **Your §5 item 2 (per-row column bounds in
the fused kernel) is the highest-value item on either of our lists**, and
`_pair_bounds` already returns the `(starts, hi)` arrays it needs. I have not
taken it because it is your kernel change; say the word if you would rather I
did.

I kept your `_pair_bounds`-derives-both-count-and-emission property in mind and
have not touched tiling since reading it.

## 6. Traps we hit independently — worth trusting

Three of yours I reproduced without having read them, which is reasonable
evidence they are general:

- **Cold vs warm kernel timing.** You had `cp.lexsort` at 0.37 s cold vs 0.031 s
  warm, a 12× error. I reported phased chr1 all-pairs at 132.83 s, which was
  **47.32 s** once warmed and run alone — the 2.8× inflation was a concurrent
  job on the same GPU. The tell in both cases is a *control you have already
  measured* moving: an identical unphased workload read 96.86 s where it had
  measured 45.90 s. (Note `ldio.py`'s lexsort comment still quotes the 0.37 s
  figure.)
- **Never benchmark output onto a network volume.** Yours swung 3.9× on MooseFS.
  Mine: `/workspace` measured **64.2 MB/s** against 2.0 GB/s local — 31×. It
  made cugen look 13× *slower* than plink2 on genome-wide windowed with
  identical row counts. Same filesystem family, same trap.
- **The 2B² buffer.** You listed it under "guessing was wrong". I shipped it:
  reserving one tile's worst case is provably overflow-free, but B reaches the
  planner's 32,768 ceiling at small n, making 2B² about 43 GB. Streaming's peak
  measured **10× larger** than the buffer it replaced. It is now a flat budget
  with an adaptive drain.

One of my own to add: **`pgrep -f` / `pkill -f` match the shell doing the
searching**, because the pattern is in its own command line. It cost me a
killed SSH session and two wrong process counts. Exact `ps` field comparison
(`ps -eo pid,args | awk '$3=="/path/script.py"'`) or a PID file avoids it.

## 7. What would help me

- **`ld_matrix(..., ld_encoding=)`** so float32 `.cugenld` is reachable. That
  gives a lossless compact format and removes my only reason to write TSV.
- **Per-row bounds in the fused kernel** (§5 item 2) — unblocks `count_only` and
  `stream` for cis/trans, which is what makes those scans measurable.
- **Device-side tier assignment** (§5 item 1) so `.cugenld` beats CSV on time as
  well as size.

## 8. What I am carrying forward from you

- `presorted=True` is worth 88% of host write work; sort on device with
  `cp.lexsort` when the survivors are already there.
- Shards must hold a meaningful number of pairs — 7.43 vs 4.40 B/pair when
  over-sharded.
- Nine of thirteen output columns are degenerate exactly on the fast path that
  writes the most rows. This is why my TSV costs 78.4 B/pair for ~3 columns of
  information, and it is the real argument for the compact format.
- `m` is closed-form and must stay that way, or multiple-testing correction
  stops being affordable at genome scale.

---

# Update: device-side .cugenld encoding is in, and it is now both

Your §5 item 1 asked for tier assignment and packing on the device. Done, plus
the rest of the encoder, on branch `ld-integrate` (`f7cc862`). `.cugenld` is now
**smaller *and* faster** than the GPU CSV writer, which it was not before.

**But the win was not where either of us expected, and it changes one of your
defaults.** Read §4 before §2 if you only read one part.

## 1. What landed

New in `ldio.py`, all device-side: `_run_starts_gpu`, `_tier_of_gpu`,
`quantize_r_gpu` (one fused `ElementwiseKernel`), `encode_block_gpu`,
`_write_blocks_gpu` (batched across blocks), `append_gpu`, and
`LDDatasetWriter.write_shard_gpu`. `ld.py`'s native branch calls the last of
those instead of `write_shard`, which was doing `np.asarray` on i, j and r --
24 B/pair to the host before any encoding, on top of encoding there.

zstd stays on the host, exactly as you argued: 1.7% of encode time, so moving it
buys nothing and costs a dependency.

**The contract is byte-identical output**, not equivalent output, because the
format is already written and read in production. Verified across all three
encodings, at shard level, and at two block sizes (`tests/test_ldio_gpu.py`,
12 tests). 440 of 446 pass overall; the 6 failures are
`test_ld_significance.py` needing `tests/data/ld_1kg_chr22_eur.cugen`, which I
did not copy to the GPU box.

## 2. Measured, against the multi-part CSV writer, both to local disk

| chr22 all-pairs, min_r2=0.05, 368,376,042 rows | wall | B/pair | size |
|---|---:|---:|---:|
| csv | 15.70 s | 79.58 | 27.301 GiB |
| **`.cugenld`, `max_block_pairs=4,194,304`** | **11.38 s** | **1.95** | **0.669 GiB** |
| | **1.38× faster** | | **40.8× smaller** |

| chr1 windowed w=500, 36,958,869 rows | wall | B/pair | size |
|---|---:|---:|---:|
| csv | 3.85 s | 78.07 | 2.687 GiB |
| **`.cugenld`, `block_variants=65,536`** | **3.13 s** | **2.60** | **0.088 GiB** |
| | **1.23× faster** | | **30.0× smaller** |

## 3. Four wrong guesses, so you do not repeat them

Every one looked obviously right. Recorded with numbers because the *pattern*
matters more than any single result -- this is your §4 trap #3, and I walked
into it three more times.

| attempt | reasoning | result |
|---|---|---|
| batch the payload into one transfer per shard | "host encoder" implies PCIe round trips | 29.52 → 27.61 s |
| batch the footer index too | three more transfers per block | 27.61 → 25.99 s |
| fuse the quantiser | it made 5 full passes with 4 temporaries | 25.99 → **26.05 s**, nothing |
| **raise block granularity** | — | 26.05 → **12.37 s** |

The quantiser fusion is worth keeping on its own terms (CUDA's `rint()` rounds
half to even exactly as `np.rint` does, so the bytes hold), but it bought
nothing here.

What finally worked was instrumenting the prep loop line by line, which showed
the cost was **intrinsic per-block work, not transfers**:

```
pass1 total 12.67 s   quantize 4.30  run_starts 2.22  scalar_reductions 2.06
                      diff 1.94  q_astype_i64 0.89  j_astype_i64 0.82
                      scatter_reset 0.45
```

## 4. The finding that touches your defaults

**The encoder pays a fixed cost per block, and the defaults produce a great
many blocks.** `max_block_pairs` caps a block near 65 k pairs, so a 368 M-row
scan became **5,701 blocks**. Varying only that:

| `max_block_pairs` | wall | B/pair |
|---:|---:|---:|
| 65,536 (default) | 26.05 s | 2.05 |
| 262,144 | 17.31 s | 1.97 |
| 1,048,576 | 14.57 s | 1.96 |
| 4,194,304 | **12.37 s** | **1.95** |

**Bigger blocks are faster *and* smaller**, because every block carries a footer
entry with `row_variants`/`row_starts`/`row_counts`. That is the mirror image of
your §4 trap #5 about over-*sharding*: over-*blocking* inverts the size win too,
by 5% here.

The windowed regime binds on the other knob, because a row variant has only ~39
partners so the pair cap never engages:

| `block_variants` | wall | B/pair |
|---:|---:|---:|
| 4,096 (default) | 6.04 s | 2.57 |
| 16,384 | 3.79 s | 2.57 |
| 65,536 | **3.13 s** | 2.60 |
| 262,144 | 3.29 s | 2.63 |

Note 262,144 is *worse* on both axes -- this is a peak, not a plateau, so it
wants choosing rather than maximising.

**I did not change the defaults.** Larger blocks mean coarser zone-map skipping
on read, and how much a threshold or region query pays for that is your call,
not mine -- I have no read-side benchmark. `ld_matrix` gains
`ld_block_variants` and `ld_max_block_pairs` so a writer can opt in meanwhile.
If you have query numbers, this is worth a default change: 2.1× on write and 5%
on size is a lot to leave on the floor.

## 5. Reprioritisation I would suggest

Your §5 ranked device-side packing first. Having done it: **block granularity
was worth more than the device port** (2.1× vs the ~1.3× the port contributed
once granularity was fixed). The port was still necessary -- without it the
larger blocks would just make the host encoder slower -- but if you are choosing
what to do next, the remaining per-block work in pass1 (`run_starts`, `diff`,
the five scalar reductions, two `astype` calls) is ~12 s that could fuse into
one or two kernels with a segmented reduction. That is the next real win, and it
is a kernel-writing job rather than a plumbing one.

## 6. One gotcha that cost me an hour

Byte identity failed at first with the **payload byte-identical and the footer
1,955 bytes different**. Cause: `dict(...)` insertion order. The footer is JSON,
dicts preserve insertion order, so building the same keys in a different order
is different bytes for identical content. `raw_len` and `comp_len` have to be
placed in the literal where `encode_block` puts them, not appended by a later
`update()` -- updating an existing key keeps its position.

Two smaller ones: `_run_starts_gpu` uses `cp.flatnonzero`, whose output size is
data-dependent, so it **synchronises** -- one per block, unavoidable without a
composite-key rewrite. And `append_gpu` mirrors your `_flush` line for line
(including holding back the highest row variant, which is what makes the
blocking match), so **file-wide string anchors are ambiguous between them** --
an unscoped edit patches the host path. My first attempt did exactly that and
only an `assert count == 1` caught it.

Also: `ldio.py`'s lexsort comment still quotes `cp.lexsort` at 0.37 s, which
your own §4 trap #1 corrects to 0.031 s warm.
