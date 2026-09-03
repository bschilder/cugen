"""GPU vs CPU for the one-GEMM KING, same box, same data, same code.

An earlier 11.7x figure for this pair was measured on the two-product version of
`king`, before HetHet was cancelled out of the estimator. That number therefore
does not describe the shipped code: the algebraic change removed one of the two
matrix products, and it removed the product that was NOT symmetric, so it
shifted the CPU and GPU sides by different amounts. This re-measures both on the
current implementation rather than carrying the stale ratio forward.

Reports the min of `--reps` runs per backend, and asserts the two agree, because
a speedup between two different answers is not a speedup.

    python benchmarks/king/gpu_vs_cpu.py --n 2504 --p 100000
"""
import argparse
import os
import tempfile
import time

import numpy as np

from cugen.popstruct import king, king_pairs
from cugen.write import write_cugen


def build(n, p, seed=0):
    rng = np.random.default_rng(seed)
    G = ((rng.random((n, p)) < 0.3).astype(np.uint8)
         + (rng.random((n, p)) < 0.3).astype(np.uint8))
    G[1] = G[0]                                  # one duplicate as a landmark
    path = os.path.join(tempfile.mkdtemp(), "k.cugen")
    write_cugen(path, G)
    return path


def timeit(fn, reps):
    out, ts = None, []
    for _ in range(reps):
        t = time.time()
        out = fn()
        ts.append(time.time() - t)
    return min(ts), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2504)
    ap.add_argument("--p", type=int, default=100_000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--block", type=int, default=1024)
    a = ap.parse_args()

    print(f"=== one-GEMM KING, n={a.n:,}  p={a.p:,}  reps={a.reps} ===")
    path = build(a.n, a.p)

    print("\n  king()  -- dense (n,n)")
    tc, Kc = timeit(lambda: king(path, backend="numpy", verbose=False), a.reps)
    tg, Kg = timeit(lambda: king(path, backend="gpu", verbose=False), a.reps)
    same = np.abs(Kc - Kg).max()
    print(f"    numpy {tc:8.3f} s   {a.p/tc:>12,.0f} markers/s")
    print(f"    gpu   {tg:8.3f} s   {a.p/tg:>12,.0f} markers/s")
    print(f"    SPEEDUP {tc/tg:.2f}x     max|gpu-cpu| = {same:.3e}"
          f"   duplicate={Kg[0,1]:.6f}")
    assert same == 0.0, f"backends disagree by {same}"

    print("\n  king_pairs()  -- blocked, thresholded")
    kw = dict(min_kinship=-np.inf, sample_block=a.block, verbose=False)
    tc2, Dc = timeit(lambda: king_pairs(path, backend="numpy", **kw), a.reps)
    tg2, Dg = timeit(lambda: king_pairs(path, backend="gpu", **kw), a.reps)
    Dc = Dc.sort_values(["i", "j"]).reset_index(drop=True)
    Dg = Dg.sort_values(["i", "j"]).reset_index(drop=True)
    d = float(np.abs(Dc.kinship.to_numpy() - Dg.kinship.to_numpy()).max())
    print(f"    numpy {tc2:8.3f} s   {len(Dc):>12,} pairs")
    print(f"    gpu   {tg2:8.3f} s   {len(Dg):>12,} pairs")
    print(f"    SPEEDUP {tc2/tg2:.2f}x     max|gpu-cpu| = {d:.3e}")
    assert d == 0.0, f"backends disagree by {d}"
    print("\n  both backends bit-identical")


if __name__ == "__main__":
    main()
