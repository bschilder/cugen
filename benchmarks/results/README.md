# Benchmark results

Raw output from every run reported in issues #1 and #2. Nothing here is
hand-edited; the plots and the issue text are generated from these files by
`../make_plots.py` and `../gen_subissue.py`, so any number in the write-up can
be traced back to a file in this directory.

## GPU portability matrix

| file | what it is |
|---|---|
| `gpu_matrix_all.json` | **the merged matrix** — every device ever attempted, across all passes. Source for the plots and the table. |
| `gpu_matrix_results.json` | pass A: the original sweep on the **pre-optimisation** code — 14 attempted, 12 measured |
| `gpu_matrix_opt.json` | pass B: re-run on the optimised code — 11 attempted, 7 measured |
| `gpu_matrix_fanout_pass1.json` | pass C: wider fan-out, 17 devices, run 6-at-a-time |
| `gpu_matrix_fanout_pass2.json` | pass D: retries + 7 additional devices, run concurrently with C |
| `matrix_main.json` | first sweep (A2000 → H100), before the Blackwell retry |
| `gpu_matrix_blackwell.json` | Blackwell retry: RTX 5090 and RTX PRO 4500 succeeded; PRO 5000 and PRO 6000 never returned a measurement |
| `a100.json` | A100 80GB probe, run separately on the chr22 pod |
| `gpu_matrix.png`, `gpu_scaling.png` | figures |
| `gpu_matrix_table.md` | the same data as markdown |

Passes C and D keep **every draw**, not just the kept one, so a re-draw is
auditable. The merge rule is that a later pass overwrites an earlier record
only if it actually measured the device — otherwise a retry that hit "no
capacity" would erase a good measurement, which is the most destructive thing
the merge could do and would leave a well-formed matrix behind while doing it.

Fleet availability, not code, set the yield: many devices were lost to "no
instances currently available", several of which the capacity API had listed as
available minutes earlier. Those are recorded with their reason rather than
dropped, because "no capacity at 11:20 on a Tuesday" and "the kernel does not
compile" are different facts and a dash conflates them. Two devices (RTX 3070,
RTX 5000 Ada — different architectures) failed CUDA device init before any
cugen code ran, with `nvidia-smi` also returning nothing: broken
driver/container pairings on those hosts, not results about this module.

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
   harness now re-draws automatically when throughput fails to grow with tile
   size (see below), but a single draw is still a single draw.
2. **Every pre-optimisation timing is an upper bound, not a capability.**
   Throughput growth from p=2k to p=20k separates the two code versions with no
   overlap: **12/12 pre-optimisation devices fall between 0.5x and 1.8x**, and
   **0/19 optimised devices do** (they run 7.3x-38.2x). The old code was
   host-bound on every device it ever ran on, which is what an SM utilisation
   of 1-4% before the fused kernel implies. The RTX A4000's 239 s is the
   extreme of that, not a uniquely bad chip.
3. **Correctness is not affected by any of this.** All measured devices, on
   both code versions, returned results bit-identical to the CPU reference
   (`max|ΔR| = 0.0`). Bit-exactness does not care how slow the host was.
4. **D' diverges from plink2 on ~0.005% of real pairs** — tables where the
   likelihood has multiple admissible roots. See issue #2 and
   `test_cubic_picks_the_global_maximum_likelihood_root`.
5. **AMD is untested.** `cupy-cuda12x` is CUDA-only.
6. **Turing (cc 7.5) is untested** — there is no T4, RTX 20-series or Quadro
   RTX in RunPod's catalogue, so no Turing part can be obtained at any price.
   It is bracketed by Volta (7.0) and Ampere (8.0), both of which work, but
   that is an inference rather than a measurement.

## Optimisation series — cugen vs plink2

Added after the first write-up, when a head-to-head showed cugen *losing* the
windowed case. Files are in the order they were produced, so the sequence is
auditable rather than just the final number.

| file | what it measures |
|---|---|
| `vs_plink2.json` | windowed, before any optimisation — plink2 wins 12-20x |
| `vs_plink2_allpairs.json` | all-pairs, before optimisation — crossover at p=50k |
| `vs_plink2_big.json` | all-pairs at chr22 scale, pre-cuDF: 7.56x |
| `vs_win_opt.json` | after r-only path + window-aware tiles (~16% better) |
| `vs_win_opt2.json` | after replacing pandas.to_csv with pyarrow (4x better) |
| `vs_win_cudf.json` | after the cuDF device path (8.3x better than the start) |
| `vs_all_cudf.json` | all-pairs with cuDF: **14x faster, 14x less memory** |
| `nscale_tf32.log` | sample-count sweep to 1,000,000, fp32 vs TF32 |

### What each optimisation was actually worth

Recorded because the ordering is the interesting part -- the changes reasoned
out from first principles were correct and nearly worthless, and the ones that
mattered came from profiling:

1. r-only path (skip the 3x3 table) + window-aware tiling — **16%**, despite
   the analysis being right that a narrow band wastes ~94% of GEMM work. It
   was 2% of runtime.
2. `pandas.to_csv` -> pyarrow — **4x**. Writing 1.4M rows cost 7.3 s against
   0.22 s for the entire GPU scan.
3. cuDF device path (zero-copy from CuPy, GPU CSV writer) — a further **1.6x**
   windowed and **~3x** all-pairs. The survivors were already in device
   memory; the old path copied them to the host to hand to a serialiser.
4. Fused epilogue kernel with atomic compaction — SM utilisation **1-4% ->
   100%**. Roughly 15 CuPy launches, six B x B temporaries and a blocking
   `cp.nonzero` per tile, replaced by one kernel.
5. TF32 — **6.3x on the isolated GEMM and bit-exact**, but only ~1.0-1.25x
   end to end, because the pipeline is not GEMM-bound at large n.

### Numbers here are single or triplicate runs on a shared cloud host

Two sweeps of identical code and data disagreed by 1.9x at n=200,000. Treat
the direction as solid and the specific multipliers as approximate; the
replicated sweep reports medians of three with the observed range.

## Scaling sweeps (final, optimised code)

| file | what it measures |
|---|---|
| `psweep.json` | variants axis, 1KG chr22, n=2,504 fixed, p to whole chromosome |
| `nscale.json` | samples axis, p=4,000 fixed, n to 1,000,000 (medians of 3) |
| `scaling_variants.png`, `scaling_samples.png` | the two line plots |

Headline: whole chr22 all-pairs (170,949 variants, 1.46e10 pairs) in **1.51 s**
against plink2's 771.78 s on 128 cores -- **513x**, with GPU memory flat at
8.4 GiB while plink2's RSS grew to 109 GiB.

Crossovers: p ~= 5,000 on the variant axis, n ~= 10,000 on the sample axis.
Below those the GPU's fixed costs dominate and plink2 wins; the variant sweep
also runs at n=2,504, which is cugen's worst regime.

## Corrections applied 2026-08-11

`nscale.json` was regenerated. The version first published came from a sweep
run at 23:13 the previous night -- three optimisation commits BEFORE the ones
that landed at 23:16, 23:29 and 23:38 -- while `psweep.json` was measured at
00:26 with all of them. One comment carried a current variant table beside a
stale sample table. Both now come from the same code.

That stale table *understated* cugen (52x peak where the corrected sweep shows
70x), which is why nothing looked wrong: an inconsistency that flatters you
gets caught, one that under-reports does not.

Re-measuring also exposed a real bug: `_tile_size_for` bounded tile size by
MEMORY but not by p, so at p=4,000 with n=100,000 it chose B=31,744 and
allocated a 31,744 x 100,000 plane buffer to hold 4,000 rows. Fixed by
`B = min(B, p)`; GPU memory at n=100,000 fell 24.12 -> 3.44 GiB.

`concordance.parquet` holds 854,850 matched cugen/plink2 pairs behind
`concordance.png` and `concordance_dprime.png`.
