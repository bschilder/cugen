"""Scaling behaviour of cugen.impute, and where it stops being cheap.

Two axes, because they stress different phases:

  target cohort size (T)   the axis a biobank grows along. Both the
                           forward-backward and the dose step are linear in T,
                           so this should be a straight line -- and if it is
                           not, something is serialising.

  reference panel size (K) the axis that decides whether brute-force states
                           remain viable. The forward-backward is linear in K;
                           the sparse dose step is NOT, since it depends on
                           minor-allele counts rather than on K directly. This
                           is the measurement that says when PBWT selection
                           stops being optional.

Per-phase timings are reported, not just totals. The previous round of this
project spent three optimisation passes on a phase that turned out to be 0.9%
of runtime, and only phase instrumentation broke the loop.

    python benchmarks/bench_impute.py --out impute_scaling.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def simulate_mosaic(n_hap, n_var, n_founders=12, seed=0, min_seg=200,
                    max_seg=900):
    """Haplotypes as founder mosaics -- the structure Li-Stephens models.

    Independent Bernoulli draws would carry no information from typed to
    untyped markers, so a benchmark built on them would measure the machinery
    while imputing nothing.
    """
    rng = np.random.default_rng(seed)
    founders = (rng.random((n_founders, n_var)) < 0.35).astype(np.uint8)
    out = np.empty((n_hap, n_var), dtype=np.uint8)
    for i in range(n_hap):
        who = int(rng.integers(0, n_founders))
        j = 0
        while j < n_var:
            seg = int(rng.integers(min_seg, max_seg))
            out[i, j:j + seg] = founders[who, j:j + seg]
            who = int(rng.integers(0, n_founders))
            j += seg
    return out


def scaled_ne(n_ref_hap, base_ne=100_000, base_hap=4904):
    """Beagle's ne rescaled to the panel; see cugen.impute's tau warning."""
    return base_ne * n_ref_hap / base_hap


def one(K, T, M, frac_typed, backend, rng_seed=0, warm=False):
    from cugen._impute_core import impute_haplotypes
    rng = np.random.default_rng(rng_seed)
    ref = simulate_mosaic(K, M, seed=rng_seed)
    tgt_full = simulate_mosaic(T, M, seed=rng_seed + 1)
    tgt_idx = np.sort(rng.choice(M, size=max(2, int(M * frac_typed)),
                                 replace=False))
    cm = np.linspace(0.0, M / 30000.0, M)      # ~1 cM/Mb at 30 markers/kb
    kw = dict(ne=scaled_ne(K), cluster=0.005)
    timers = {}
    if backend == "gpu":
        from cugen._impute_gpu import impute_haplotypes_gpu
        fn = impute_haplotypes_gpu
    else:
        fn = impute_haplotypes
    t0 = time.perf_counter()
    out = fn(ref, tgt_full[:, tgt_idx], tgt_idx, cm, timers=timers, **kw)
    dt = time.perf_counter() - t0
    peak = None
    if backend == "gpu":
        import cupy as cp
        peak = cp.get_default_memory_pool().used_bytes() / 2 ** 30
    return {"K": K, "T": T, "M": M, "seconds": round(dt, 3),
            "phases": {k: round(v, 4) for k, v in timers.items()
                       if isinstance(v, float)},
            "peak_gib": peak, "shape": list(out.shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="impute_scaling.json")
    ap.add_argument("--markers", type=int, default=60_000)
    ap.add_argument("--frac-typed", type=float, default=0.1)
    ap.add_argument("--backend", default="gpu", choices=("gpu", "numpy"))
    ap.add_argument("--t-grid", default="16,64,256,1024,4096")
    ap.add_argument("--k-grid", default="500,1000,2000,4904,10000")
    a = ap.parse_args()

    results = {"markers": a.markers, "frac_typed": a.frac_typed,
               "backend": a.backend, "target_axis": [], "panel_axis": []}

    if a.backend == "gpu":
        import cupy as cp
        results["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        print(f"device: {results['gpu']}", flush=True)
        # Warm up NVRTC and the CUDA context before ANY timing. A cold first
        # configuration absorbs seconds of compilation and silently inflates
        # whichever row happens to run first -- which is how a benchmark
        # "discovers" that small inputs are slow.
        print("warming up (not counted)...", flush=True)
        t0 = time.perf_counter()
        one(200, 8, 2000, 0.1, "gpu", rng_seed=99)
        print(f"  warm-up {time.perf_counter() - t0:.2f}s", flush=True)

    print("\n=== target cohort axis (K fixed at 4904 = 1000G panel) ===",
          flush=True)
    for T in [int(x) for x in a.t_grid.split(",")]:
        r = one(4904, T, a.markers, a.frac_typed, a.backend, rng_seed=1)
        results["target_axis"].append(r)
        ph = "  ".join(f"{k} {v:.2f}s" for k, v in
                       sorted(r["phases"].items(), key=lambda kv: -kv[1])[:3])
        print(f"  T={T:6,d}  {r['seconds']:8.3f}s   {ph}", flush=True)
        json.dump(results, open(a.out, "w"), indent=2)

    print("\n=== reference panel axis (T fixed at 256) ===", flush=True)
    for K in [int(x) for x in a.k_grid.split(",")]:
        r = one(K, 256, a.markers, a.frac_typed, a.backend, rng_seed=2)
        results["panel_axis"].append(r)
        ph = "  ".join(f"{k} {v:.2f}s" for k, v in
                       sorted(r["phases"].items(), key=lambda kv: -kv[1])[:3])
        print(f"  K={K:6,d}  {r['seconds']:8.3f}s   {ph}", flush=True)
        json.dump(results, open(a.out, "w"), indent=2)

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
