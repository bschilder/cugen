# LD result storage — formats, size, and query latency

All numbers measured on **one NVIDIA A100-SXM4-80GB** (driver 580.126.16, CuPy
14.2.0, cuDF 25.12, so the fused device path is live), against **real 1000
Genomes chr22:20–30 Mb, MAF ≥ 0.01 → 51,100 variants × 3,202 samples**, streamed
from EBI rather than downloaded. Reproduce with
`python benchmarks/ld_storage.py --cugen chr22_dos.cugen`; the fixture commands
are in that script's docstring.

This is the storage analysis `GENOMEWIDE.md` promised and pointed at.

## Why it matters

Writing already costs as much as computing — the headline benchmark is 47% scan
and 43% serialisation, 61.5 ns/row on the cuDF path. And of the thirteen columns
in `_empty_pairs`, only `gidx_a`, `gidx_b` and one float carry information. `CHR`
is constant per file, `POS` is literal zeros and `ID` is `"."` on the device
path, `MAF` is in the `.cugen` header, `N_OBS` is constant without missingness,
`R2 = R*R`, and `NEG_LOG10_P` is a closed-form function of `N_OBS * r2`. The
fused path — the only one a large run takes — is gated on `annotation is None`,
which is exactly the condition that makes those columns degenerate.

## Formats, windowed scan (window=500, min_r2=0.2, 1,081,550 pairs)

| format | size | B/pair | vs TSV | scan+write |
|---|---:|---:|---:|---:|
| `.tsv` | 83.9 MB | 77.57 | 1.0× | 2.13 s |
| `.feather` | 18.2 MB | 16.85 | 4.6× | 2.17 s |
| `.parquet` (zstd, tuned) | 8.2 MB | 7.57 | 10.2× | 2.52 s |
| `.npz` | 6.7 MB | 6.17 | 12.6× | 2.97 s |
| **`.cugenld`** | **3.5 MB** | **3.21** | **24.1×** | 2.22 s |

Every format round-trips the same pair set; `tests/test_ldio.py` asserts it.
`.cugenld` reaches **2.10 B/pair** on the larger all-pairs run below, where the
fixed header and footer stop mattering.

Two bugs this measurement caught. `.feather` first came out at *exactly* TSV's
77.57 B/pair — the legacy cuDF branch has no feather case, so it fell through and
wrote CSV into a `.feather` filename. And untuned Parquet defaults to snappy with
no statistics; with zstd and row-group statistics it is 7.57 B/pair and a
bp-range predicate can actually skip row groups.

## All-pairs, and the `min_r2` lever

51,100 variants → 1.31e9 pairs, no window. This is the shape a genome-wide run
takes.

| min_r2 | rows | rows/variant | % of all pairs | vs r²≥0.2 | wall | size | B/pair |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 1,415,371 | 27.7 | 0.108% | 1× | 7.5 s | 4.0 MB | 2.84 |
| 0.1 | 3,448,914 | 67.5 | 0.264% | 2.4× | 7.1 s | 8.2 MB | 2.39 |
| 0.05 | 12,730,642 | 249.1 | 0.975% | 9.0× | 10.2 s | 25.6 MB | 2.01 |
| 0.02 | 66,765,884 | 1,306.6 | 5.114% | 47× | 31.0 s | 135.9 MB | 2.04 |
| **0.01** | **164,079,591** | **3,211.0** | **12.568%** | **116×** | 323 s | 345.1 MB | 2.10 |

The log-log slope is **t^-1.64**, with local slopes of −1.29 (0.2→0.1), −1.88
(0.1→0.05), −1.81 (0.05→0.02) and −1.30 (0.02→0.01). An earlier version of this
analysis modelled it as 1/t, which is wrong in the measured range — tightening
the threshold buys more than 1/t suggests.

But the power law cannot continue, and the "% of all pairs" column is why: at
r² ≥ 0.01 the retained set is already **12.6% of every pair in the file**. It
saturates toward the full within-LD-span pair space rather than growing without
bound, so extrapolating t^-1.64 into a biobank regime overshoots. The real
ceiling is `p × (variants within the LD span)` — roughly 7e10 rows genome-wide
for a ~1 Mb span — plus a trans contribution that stays α-limited by
construction. The flattening at both ends of the measured range is that ceiling
becoming visible.

**This is the measurement the repo did not have.** Every previously recorded
output volume used `min_r2 = 0.2`, and `GENOMEWIDE.md` flagged the threshold as
"the dominant lever on output size" without quantifying it. It is **116×**
between 0.2 and 0.01 on real data — and `SIGNIFICANCE.md` shows a defensible
family-wise threshold on this cohort is r² ≈ 0.013, i.e. squarely in the
expensive regime. A format that only works at 0.2 does not survive its own
significance layer.

Note the wall time: at 164 M rows the run is write-bound, not scan-bound.

## Query latency

On the `min_r2 = 0.01` file above: 164,079,591 pairs, 2,399 blocks, 2.10 B/pair,
mean 68,395 pairs/block.

| query | rows | latency | blocks read |
|---|---:|---:|---:|
| `above(r²≥0.05)` | 12,733,891 | 1203 ms | 1504/2399 (63%) |
| `above(r²≥0.2)` | 1,415,522 | 73.8 ms | 95/2399 (4%) |
| `above(r²≥0.5)` | 619,257 | 26.8 ms | 34/2399 (1%) |
| `above(r²≥0.8)` | 306,674 | 13.1 ms | 18/2399 (1%) |
| `variant(v)` | — | **2.94 ms** | ≤ tiers |

For scale, LDmat reports a 1 Mb region query in under 2 s on chr21.

### Two design decisions that only measurement could settle

**A zone map keyed on position alone is decorative.** Blocks were first cut only
by row-variant range. On a sparse trans-like fixture — 4% strong pairs, the
all-by-all regime — a max-|r| map read **94 of 95 blocks**, because strong LD is
spread across the variant axis rather than clustered into contiguous row ranges,
so every block contained something strong. Cutting each block by **r² tier** as
well as by position makes it homogeneous in |r|, and the skip becomes
proportional to the query: 4% of blocks at r²≥0.2, 1% at r²≥0.8. It costs 0–16%
in bytes/pair.

**zstd frames are not seekable, so block size bounds single-variant lookup.** At
`block_variants=4096` a block held 2.5 M pairs and `variant()` cost **360 ms** —
one lookup decompressing 2.5 M pairs to return a few hundred. Capping a block by
pair count as well (`MAX_BLOCK_PAIRS = 65536`) took it to **2.94 ms, a 122×
improvement**, and halved `above(r²≥0.2)` from 172.8 ms to 73.8 ms, for +7% in
bytes/pair (1.96 → 2.10) and 321 → 345 MB. Indexing alone did not help; the cost
was never the scan.

## Significance-layer overhead on the fused GPU path

20,000 variants, window=500. `SIGNIFICANCE.md` measured this on the CPU reference
path and found it free; on the GPU it is cheap but not free.

| case | wall | ratio | rows |
|---|---:|---:|---:|
| r² only | 1080 ms | 1.00× | 445,604 |
| + `chi2`, `p` | 1161 ms | **1.08×** | 445,604 |
| + `max_p=1e-8` | 1874 ms | 1.74× | 2,339,363 |
| + Bonferroni 0.05 | 1903 ms | 1.76× | 2,282,750 |
| + BH-FDR 0.05 | 3630 ms | 3.36× | 5,781,297 |

**Read the rows column before the ratio.** Computing the statistics costs 8%. The
filters look expensive because a statistically defensible threshold retains
**5–13× more pairs** than `min_r2 = 0.2` — that is output volume, not overhead,
and it is the same finding as the 116× above.

## Sharded datasets and resumability

A shard is one scan tile's output, keyed by the `(A, B)` variant-block pair the
scan already walks, so shards are written independently — concurrently, across
GPUs, and across a resumed run — with no cross-shard coordination and no global
sort. Real chr22, `min_r2 = 0.1`, tile = 8192 variants → 28 shards:

    interrupted after 11/28 shards, 1,206,999 pairs, 27.8 s
    resume saw 11 completed shards and recomputed 17
    resumed: 28 shards, 3,448,696 pairs, 26.2 s, 2.55 B/pair
    resume skipped 52% of the total work

**The resumed dataset is identical to a single unsharded scan** — 3,448,696 rows
on both sides, symmetric difference **0**. That is the property that makes
resumability trustworthy rather than merely convenient: on spot or preemptible
capacity a format that forces all-or-nothing writes makes an hours-long
genome-wide job impractical.

Shards land by **atomic rename**, so a process killed mid-write leaves a temp
file no reader ever sees, and the manifest — not the directory listing — is the
record of what exists. Resuming with different parameters is refused outright:
the test space sets the number of tests and therefore every corrected threshold,
so half a dataset at one setting and half at another is not a dataset.

### Query latency across 28 shards, 386 blocks, 3.45 M pairs

| query | rows | latency | shards opened | blocks read |
|---|---:|---:|---:|---:|
| `above(r²≥0.2)` | 1,415,522 | 68.6 ms | 28/28 | 152/386 (39%) |
| `above(r²≥0.5)` | 619,257 | 31.3 ms | 27/28 | 84/386 (22%) |
| `above(r²≥0.8)` | 306,674 | 15.3 ms | 27/28 | 41/386 (11%) |
| `region(10k,12k)` | 60,455 | 9.4 ms | **1/28 (4%)** | — |
| `variant(v)` | — | **2.38 ms** | ~10 | — |

**Footer re-parsing was dominating every cross-shard query.** Each shard open
parses its footer and rebuilds its row-variant index; doing that per lookup cost
more than the decompression it was there to avoid. Caching the readers — the
files are immutable, so it is always safe — gave:

    variant()          21.74 ms -> 2.38 ms    9.1x
    above(r2>=0.2)   1209.9 ms  -> 68.6 ms   17.6x
    above(r2>=0.5)   1162.9 ms  -> 31.3 ms   37.2x
    above(r2>=0.8)    635.6 ms  -> 15.3 ms   41.5x
    region(10k,12k)     22.5 ms ->  9.4 ms    2.4x

Note that `above()` opens nearly every shard here and that is correct: at
tile = 8192 variants (~1.6 Mb) every shard in a 10 Mb window is within LD range,
so none can be excluded on its max |r|. The block-level tier skip is what does
the work (39% → 11% of blocks as the cut tightens).

### Where each skip actually earns its keep

Two levels, and they are not interchangeable:

- **Index-based skips** (`min_i`/`max_i`, `min_j`/`max_j`) are the reliable ones.
  `region()` opens 1 of 28 shards; `variant()` about 10 of 55 on a synthetic
  fixture. These work regardless of how signal is distributed.
- **Value-based skips** (per-shard `max_abs_r`, per-block tier) depend on the data
  being clustered in r. On a fixture with strong pairs spread *uniformly* the
  shard-level r skip opened **all 55 shards** — correct, but unexercised, because
  every shard genuinely contained something strong. On realistic near-diagonal LD
  it opens 21.8% for r²≥0.8. The test suite carries both fixtures for exactly
  this reason.

## Streaming straight into `.cugenld`

#4's `stream=True` calls an `on_flush(i, j, r)` callback whose flushes land on
tile boundaries, and a shard *is* one tile's output — so the two compose: each
flush becomes one self-contained shard and nothing is accumulated. The native
path skips `_assemble_device` entirely, since building a 76 B/row frame per
flush would put ~4x more bytes across PCIe than the encoded shard occupies on
disk.

Real chr22, all pairs, p = 51,100:

| min_r2 | target | rows | wall | size | B/pair | vs parquet |
|---:|---|---:|---:|---:|---:|---:|
| 0.2 | parquet (#4) | 1,930,958 | 1.8 s | 22.3 MB | 11.56 | 1.00x |
| 0.2 | **`.cugenld`** | 1,930,958 | **0.6 s** | **5.1 MB** | **2.65** | **3.24x** |
| 0.2 | `count_only` (scan floor) | — | 0.3 s | — | — | — |
| 0.05 | parquet (#4) | 27,541,325 | 5.7 s | 338.2 MB | 12.28 | 1.00x |
| 0.05 | **`.cugenld`** | 27,541,325 | **2.4 s** | **55.4 MB** | **2.01** | **2.36x** |
| 0.05 | `count_only` (scan floor) | — | 0.3 s | — | — | — |

**6.1x smaller and 2.4x faster** at 27.5 M rows. Against the 0.3 s scan floor,
parquet spends 5.4 s serialising (95% of wall clock) and `.cugenld` 2.1 s (87%) —
still the dominant term, but 2.6x less of it.

### The sort, not the compression, was the cost

The first native version was **slower** than parquet at 27.5 M rows — 7.9 s
against 5.7 s — while being 6.1x smaller. Profiling the host path at 8 M rows
found where the time actually went, and it was not where I assumed:

    D2H copy (int64, int64, float64 = 24 B/row)    0.04 s
    host np.lexsort                                2.03 s   <- 88%
    quantise r -> int16                            0.03 s
    per-block delta + zstd encode                  0.21 s   <- 9%
    ------------------------------------------------------
    total host work                                2.31 s

I had assumed the Python encode loop dominated. It is 9%. **The host sort is
88%**, and `cp.lexsort` does the same 8 M rows in 0.37 s with a 0.01 s D2H
afterwards because the payload is narrow. Sorting on the device and handing the
writer `presorted=True` replaced 2.10 s of host work with 0.39 s — and turned a
0.71x loss into a 2.36x win.

The lesson generalises past this format: at these row counts any per-row host
work is the bottleneck, and the useful question is always which one.

## Caveats

- One 10 Mb window of chr22 in one cohort. Enough to size the formats and the
  threshold lever; not a genome-wide characterisation.
- The `.zarr` backend is implemented but not measured here — `zarr` is an
  optional dependency and was not installed on the benchmark host.
- Query latencies are single-threaded host-side decode. Nothing in the reader is
  on the GPU yet.
- `variant()` builds a row-variant index at open, which is O(row variants) in
  Python. At 84 M variants that becomes the cost to beat.
- The delta + zstd encode is still a host-side Python loop. It is only 9% of
  host work now that the sort moved to the device, but it is the next term, and
  zstd on the GPU would need nvCOMP.
- Shard-level `max_abs_r` skipping is weak whenever every shard is within LD
  range of the diagonal, which is the common case for a windowed cis scan. It
  pays for trans and long-range work.
- `bytes_per_pair` rises with shard count: each shard pays a 256-byte header and
  its own footer, so a small dataset cut into many shards is worse than one file
  (7.43 vs 4.4 B/pair on a 115 k-pair fixture). Shards must be sized to hold a
  meaningful number of pairs; `tests/test_ldio.py` pins the direction.
