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
