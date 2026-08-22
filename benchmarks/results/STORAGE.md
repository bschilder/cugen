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

| min_r2 | rows | vs r²≥0.2 | wall | size | B/pair |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 1,415,371 | 1× | 6.7 s | 4.0 MB | 2.83 |
| 0.05 | 12,730,642 | 9.0× | 10.5 s | 24.7 MB | 1.94 |
| **0.01** | **164,079,591** | **116×** | 70.4 s | 345 MB | 2.10 |

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

## Caveats

- One 10 Mb window of chr22 in one cohort. Enough to size the formats and the
  threshold lever; not a genome-wide characterisation.
- The `.zarr` backend is implemented but not measured here — `zarr` is an
  optional dependency and was not installed on the benchmark host.
- Query latencies are single-threaded host-side decode. Nothing in the reader is
  on the GPU yet.
- `variant()` builds a row-variant index at open, which is O(row variants) in
  Python. At 84 M variants that becomes the cost to beat.
- Sharding by variant-block pair, the manifest, and resumability are designed
  (see `cugen/ldio.py`) but not yet implemented: this is one shard per file.
