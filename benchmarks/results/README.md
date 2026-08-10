# Benchmark results

Raw output from every run reported in issues #1 and #2. Nothing here is
hand-edited; the plots and the issue text are generated from these files by
`../make_plots.py` and `../gen_subissue.py`, so any number in the write-up can
be traced back to a file in this directory.

## GPU portability matrix

| file | what it is |
|---|---|
| `gpu_matrix_results.json` | **the merged matrix** — 14 devices attempted, 12 measured. Source for the plots and the table. |
| `matrix_main.json` | first sweep (A2000 → H100), before the Blackwell retry |
| `gpu_matrix_blackwell.json` | Blackwell retry: RTX 5090 and RTX PRO 4500 succeeded; PRO 5000 and PRO 6000 never returned a measurement |
| `a100.json` | A100 80GB probe, run separately on the chr22 pod |
| `gpu_matrix.png`, `gpu_scaling.png` | figures |
| `gpu_matrix_table.md` | the same data as markdown |

Probe workload: p = 20,000 variants, 2,504 samples, seeded synthetic genotypes
(`tests/conftest.py::simulate_haplotypes`), `stats=(r, r2)`, `min_r2=0.5`.
That fixture is deliberately LD-rich so D/D' have signal, so ~4.3% of pairs
clear the threshold versus 0.034% on real chr22 — these timings are dominated
by output volume and are **not** a throughput headline. See `bench_*.json` for
that.

## Real data — 1000 Genomes chr22

| file | what it is |
|---|---|
| `bench_maf01_fixed.json` | scaling sweep, 170,949 variants (MAF ≥ 0.01), all-by-all |
| `bench_all.json` | unfiltered chr22: all 1,055,454 variants, 5.57e11 pairs |
| `vs_plink2.json` | head-to-head against plink2 at matched workloads |
| `parity.json` | plink2 agreement summary (r, D, D') |
| `plink_real.txt` | raw console output of the parity run |

`_fixed` in the filename is load-bearing: an earlier sweep measured peak memory
as `pool.total_bytes()` after the run, which is not a high-water mark because
`_scan_gpu` frees blocks mid-run. It reported peak at p=50,000 as *lower* than
at p=1,000. Peak memory is now sampled on a thread (`../_peak.py`); the bad
sweep is not published here to avoid it being quoted by accident.

## Caveats that travel with these numbers

1. **n = 1 pod per GPU**, so host quality is confounded with GPU model. The
   RTX A4000 row (239 s vs 14 s for the architecturally similar A5000) is
   almost certainly a slow host draw, not the chip.
2. **Correctness is not affected by either caveat.** All 12 measured devices
   returned results bit-identical to the CPU reference (`max|ΔR| = 0.0`).
3. **D' diverges from plink2 on ~0.005% of real pairs** — tables where the
   likelihood has multiple admissible roots. See issue #2 and
   `test_cubic_picks_the_global_maximum_likelihood_root`.
4. **AMD is untested.** `cupy-cuda12x` is CUDA-only.
