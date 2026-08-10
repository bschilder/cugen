"""Benchmark cugen.ld.ld_matrix -- wall time and PEAK DEVICE MEMORY vs p.

The claim under test is not "it is fast" but "peak memory is flat in the
variant count while time grows as O(p^2)". A single fast number does not
demonstrate that; the scaling sweep does.

    python benchmarks/bench_ld.py --cugen chr22.cugen --scaling --out scale.json
    python benchmarks/bench_ld.py --cugen chr22.cugen --p 170000 --min-r2 0.2
"""
import os
import sys

# Make `python benchmarks/<script>.py` work from a source checkout: running a
# script puts ITS directory on sys.path, not the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "tests") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "tests"))

import argparse
import json
import threading
import time

import numpy as np

import cugen as cg
from cugen.ld import ld_matrix

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def _peak_reset():
    if HAS_CUPY:
        cp.get_default_memory_pool().free_all_blocks()
        try:
            cp.cuda.runtime.deviceReset
        except Exception:
            pass


def _free_bytes():
    return cp.cuda.Device().mem_info[0] if HAS_CUPY else 0


class _PeakSampler:
    """Sample live device allocations during a run.

    NB pool.total_bytes() at the end is NOT the peak: cugen.ld calls
    free_all_blocks() inside its tile loop, which shrinks the pool mid-run and
    destroys the monotonic high-water property. (Symptom: 'peak' at p=50k
    reading lower than at p=1k.) So sample used_bytes() on a thread instead.
    """

    def __init__(self, interval=0.002):
        self.interval, self.peak, self._stop = interval, 0, False

    def __enter__(self):
        if HAS_CUPY:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        pool = cp.get_default_memory_pool()
        while not self._stop:
            self.peak = max(self.peak, pool.used_bytes())
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop = True
        if HAS_CUPY:
            self._t.join(timeout=1.0)
            self.peak = max(self.peak, cp.get_default_memory_pool().used_bytes())


def run_once(path, p, *, stats, min_r2, window, tile_size, backend,
             max_pairs=1e15):
    """One timed run, with a sampled peak-memory measurement."""
    _peak_reset()
    pool = cp.get_default_memory_pool() if HAS_CUPY else None
    if pool is not None:
        pool.free_all_blocks()
    free_before = _free_bytes()
    with _PeakSampler() as sampler:
        t0 = time.perf_counter()
        df = ld_matrix(path, variant_range=(0, p), stats=stats, min_r2=min_r2,
                       window=window, tile_size=tile_size, backend=backend,
                       max_pairs=int(max_pairs), verbose=False)
        dt = time.perf_counter() - t0
    peak = sampler.peak
    n_pairs = p * (p - 1) // 2 if window is None else None
    return {
        "p": p,
        "wall_s": round(dt, 4),
        "peak_pool_bytes": int(peak),
        "peak_pool_gib": round(peak / 2**30, 3),
        "free_before_gib": round(free_before / 2**30, 3),
        "pairs_considered": n_pairs,
        "pairs_emitted": int(len(df)),
        "pairs_per_s": round((n_pairs or len(df)) / dt, 1),
        "stats": list(stats),
        "min_r2": min_r2,
        "window": window,
        "tile_size": tile_size,
        "backend": backend,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cugen", required=True)
    ap.add_argument("--p", type=int, default=None)
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--min-r2", type=float, default=0.0)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--tile-size", type=int, default=None)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--stats", default="r,r2")
    ap.add_argument("--max-pairs", type=float, default=1e15)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    stats = tuple(s.strip() for s in a.stats.split(",") if s.strip())
    hdr = cg.io.read_cugen_header(a.cugen)
    print(f"file      : {a.cugen}")
    print(f"samples   : {hdr['n_samples']:,}   variants: {hdr['n_variants']:,}")
    print(f"encoding  : {hdr['encoding']}   size: {hdr['file_size_gb']:.2f} GB")
    if HAS_CUPY:
        dev = cp.cuda.Device()
        free, total = dev.mem_info
        print(f"gpu free  : {free/2**30:.1f} / {total/2**30:.1f} GiB")
    print()

    results = []
    if a.scaling:
        pmax = int(hdr["n_variants"])
        grid = [g for g in (1000, 2000, 5000, 10000, 20000, 50000,
                            100000, 200000, 500000, 1000000) if g <= pmax]
        if pmax not in grid:
            grid.append(pmax)
    else:
        grid = [a.p or int(hdr["n_variants"])]

    print(f"{'p':>10s} {'wall_s':>10s} {'peak_GiB':>10s} {'pairs':>16s} "
          f"{'emitted':>12s} {'pairs/s':>14s}")
    print("-" * 78)
    for p in grid:
        # warm the kernel/cuBLAS handles once so the first row isn't an outlier
        if p == grid[0]:
            ld_matrix(a.cugen, variant_range=(0, min(256, p)), stats=stats,
                      backend=a.backend, max_pairs=int(a.max_pairs),
                      verbose=False)
        r = run_once(a.cugen, p, stats=stats, min_r2=a.min_r2,
                     window=a.window, tile_size=a.tile_size, backend=a.backend,
                     max_pairs=a.max_pairs)
        results.append(r)
        print(f"{r['p']:>10,} {r['wall_s']:>10.3f} {r['peak_pool_gib']:>10.3f} "
              f"{(r['pairs_considered'] or 0):>16,} {r['pairs_emitted']:>12,} "
              f"{r['pairs_per_s']:>14,.0f}")

    if len(results) > 2:
        ps = np.array([r["p"] for r in results], float)
        ts = np.array([r["wall_s"] for r in results], float)
        mem = np.array([r["peak_pool_gib"] for r in results], float)
        ok = ts > 0
        slope = np.polyfit(np.log(ps[ok]), np.log(ts[ok]), 1)[0]
        # The full-range slope is dragged down by fixed overhead at small p,
        # where the device is nowhere near saturated. The asymptotic slope over
        # the largest points is the honest number to quote.
        tail = slice(-3, None)
        slope_tail = (np.polyfit(np.log(ps[ok][tail]), np.log(ts[ok][tail]), 1)[0]
                      if ok.sum() >= 3 else float("nan"))
        print()
        print(f"time scaling exponent, full range     : {slope:.2f}")
        print(f"time scaling exponent, largest 3 p    : {slope_tail:.2f}   "
              f"(2.0 == O(p^2); < 2 means the GPU is still filling up)")
        # Memory is NOT flat across the whole sweep and saying so would be
        # wrong: while p < tile size the tile IS the whole matrix, so cost is
        # O(p^2). Above the tile size it plateaus. Report the plateau, which is
        # the actual claim, and let the small-p region speak for itself.
        big = ps >= 20000
        if big.sum() >= 2:
            mb = mem[big]
            print(f"peak memory, p >= 20k                 : "
                  f"{mb.min():.2f} - {mb.max():.2f} GiB over a "
                  f"{ps[big].max()/ps[big].min():.0f}x range in p "
                  f"({(ps[big].max()/ps[big].min())**2:.0f}x in pairs)")
        print(f"peak memory, full sweep               : "
              f"{mem.min():.3f} - {mem.max():.3f} GiB "
              f"(grows until p reaches the tile size, then plateaus; the "
              f"plateau is set by DEVICE memory, not by p)")

    payload = {"file": a.cugen, "header": {k: str(v) for k, v in hdr.items()},
               "results": results}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
