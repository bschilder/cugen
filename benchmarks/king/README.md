# KING at biobank scale

`king()` builds three (n, n) accumulators. `king_pairs()` walks blocks of the
sample-pair space and keeps only pairs over a threshold, so peak memory is the
packed panel plus one (B, B) block.

## Measured, A100-80GB, block=8192

| n | markers | pairs evaluated | elapsed | throughput | peak RSS | dense would need |
|---|---|---|---|---|---|---|
| 500,000 | 5,000 | 124,999,750,000 | 106.7 s | 1.17 G pair/s | 7.83 GB | 6 TB |
| 1,000,000 | 5,000 | 499,999,500,000 | 377.2 s | 1.33 G pair/s | 16.62 GB | 24 TB |
| 500,000 | 20,000 | 124,999,750,000 | 339.2 s | 0.37 G pair/s | 13.84 GB | 6 TB |

All three recovered 50/50 planted duplicates at exactly 0.500000.

Half a trillion pairs in 6.3 minutes at n=1,000,000, in 16.6 GB — about 1/1400th
of what the dense form would allocate. Scaling is slightly sub-quadratic in n
(4x the pairs cost 3.5x the time) because larger n amortises per-block overhead.

## Marker count is what controls false positives

The threshold is applied to n(n-1)/2 draws from the null, so the per-pair error
rate must fall below ~1/n^2. SE(phi) ~ 1/sqrt(markers), so the tail collapses
quickly in marker count:

| markers | emitted at phi >= 0.0442 | false positives |
|---|---|---|
| 5,000 | 7,574,975 | 7,574,925 |
| 20,000 | 50 | **0** |

Same cohort, same threshold, same 50 real relatives. Use 50,000-100,000 pruned
markers for a real scan; the throughput column above shows the cost is
sub-linear in markers.

## Reproducing

```bash
python benchmarks/king/scale.py 500000 5000  --block 8192 --backend gpu
python benchmarks/king/scale.py 1000000 5000 --block 8192 --backend gpu
python benchmarks/king/scale.py 500000 20000 --block 8192 --backend gpu
```

## GPU vs CPU, on the shipped one-GEMM code

RTX 4090, n=2,504, p=100,000, min of 3 reps, both backends asserted bit-identical:

| | CPU | GPU | speedup |
|---|---|---|---|
| `king()` dense | 3.812 s (26,231 markers/s) | **0.048 s** (2,096,974 markers/s) | **79.9x** |
| `king_pairs()` at phi >= 0.0442 | 8.496 s | 0.877 s | 9.7x |

An earlier 11.7x figure for `king()` was measured on the two-product version and
should not be quoted: it predates both the algebraic change and the device-side
unpack. Re-measuring is what exposed the second one. Cancelling a matrix product
moved the GPU wall-clock from 0.81 s to 0.868 s -- i.e. not at all -- while the
CPU side improved 2.4x, because `king()` was still unpacking 2-bit codes on the
host and was never GEMM-bound on GPU. Fixing that took it to 0.048 s.

**`king_pairs` is not the fast path, it is the possible path.** Its 9.7x reflects
a deliberate trade: bounding memory at (B, B) means each block pair re-walks the
markers, so at small n that redundancy is pure overhead. Use `king()` while the
(n, n) matrix fits and `king_pairs` when it does not -- the point of the latter
is n=1,000,000 in 16.6 GB, not throughput at n=2,504.

End to end on a 4090 at this shape, original CPU implementation to current GPU:
9.41 s -> 0.048 s, about **196x** (the two CPU timings come from different 4090
instances, same GPU model and same benchmark).

## The dense matrix on disk, queryable by person

`king_matrix()` writes what `king()` would hold in memory, so n is bounded by
disk. `open_king_matrix()` memory-maps it and every accessor takes a row index
**or a sample ID**:

```python
km = open_king_matrix("cohort.king")
km["NA12878", "NA12891"]              # one cell
km.row("NA12878")                     # that person against everyone
km.related("NA12878")                 # just their relatives, kinship descending
km.submatrix(cases)                   # k x k block for a set of people
km.submatrix(cases, cols=controls)    # rectangular, cases against controls
km.submatrix(cases, as_frame=True)    # labelled by sample ID
```

`submatrix` returns rows in the order you passed, not sorted, so they line up
with your own list. Measured at n=602 (square layout, float32):

| people | block | time |
|---|---|---|
| 10 | 10x10 | 0.01 ms |
| 100 | 100x100 | 0.05 ms |
| 300 | 300x300 | 0.36 ms |

Cost is k contiguous row gathers under `square`. Under `triangle` it is one
vectorised gather over up to k^2 scattered offsets, which is fine for a few
hundred people and another reason `square` is the default.

### Layout is about access, not size

In a lower triangle, row i is one contiguous run of i+1 entries **plus n−i−1
entries that each live in a different row** — so a per-person query touches up
to n scattered pages across the whole file.

| layout | n=1,000,000 int16 | row query |
|---|---|---|
| `square` (default) | 2.0 TB | ~1 ms (one contiguous 2 MB read) |
| `triangle` | 1.0 TB | ~10 s (up to 1M scattered page touches) |

Four orders of magnitude on latency for twice the bytes. `square` is the default
because the reason to materialise a dense matrix at all is random access;
`triangle` is for archival, where halving 1 TB matters and nobody queries a row.

### Format

```
magic "CUKING02" (8) | n int64 (8) | encoding int32 (4) | layout int32 (4)
                     | id_bytes int64 (8) | ids (newline-joined utf8) | data
```

Sample IDs are stored in the file because `.cugen` does not carry them — they
live in a `<file>.cugen.samples.txt` sidecar, which `king_matrix` picks up
automatically when `sample_ids=` is omitted. A count mismatch is refused rather
than truncated, since mislabelled IDs would silently corrupt every query.
