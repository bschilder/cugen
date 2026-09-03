"""Does king_pairs actually reach biobank n? Generate, run, and check.

`king()` cannot: its three (n, n) float64 accumulators are 6 TB at n=500,000 and
24 TB at n=1,000,000, and the dense output alone is 2 TB / 8 TB. `king_pairs`
walks blocks of the sample-pair space and keeps only pairs over a threshold, so
peak memory is the packed panel plus one (B, B) accumulator.

Writes the panel with CugenWriter one variant at a time, so generation is O(n)
rather than materialising an n x p array (5 GB at n=1e6, p=5000).

    python benchmarks/king/scale.py 500000 5000
    python benchmarks/king/scale.py 1000000 5000 --block 8192
"""
import argparse
import os
import resource
import sys
import time

import numpy as np

from cugen.popstruct import king_pairs
from cugen.write import CugenWriter


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)


def build(path, n, p, n_rel, seed=0):
    """Panel of unrelated samples plus `n_rel` planted duplicate pairs.

    Duplicates are the unambiguous positive control: phi is exactly 0.5, so a
    recovered count below n_rel is a miss and nothing else.
    """
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.05, 0.95, p).astype(np.float32)
    # partner[k] is a copy of sample k, parked at the end of the cohort
    dup_src = rng.choice(n - n_rel, n_rel, replace=False)
    t0 = time.time()
    with CugenWriter(str(path), n_samples=n, n_variants=p) as w:
        for v in range(p):
            f = freq[v]
            g = ((rng.random(n, dtype=np.float32) < f).astype(np.uint8)
                 + (rng.random(n, dtype=np.float32) < f).astype(np.uint8))
            g[n - n_rel:] = g[dup_src]          # exact duplicates
            w.add_variant(v, g)
            if v and v % 1000 == 0:
                print(f"    {v:,}/{p:,} variants  [{time.time()-t0:.0f}s]",
                      flush=True)
    print(f"  built {path} ({os.path.getsize(path)/1e9:.2f} GB) "
          f"in {time.time()-t0:.0f}s", flush=True)
    return {(int(s), int(n - n_rel + k)) for k, s in enumerate(dup_src)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("p", type=int)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--n-rel", type=int, default=50)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--min-kinship", type=float, default=0.0442)
    ap.add_argument("--keep", default="/tmp/king_scale.cugen")
    a = ap.parse_args()

    print(f"=== n={a.n:,}  p={a.p:,}  block={a.block}  backend={a.backend} ===")
    dense_gb = a.n * a.n * 8 / 1e9
    print(f"  dense king() would need {3*dense_gb/1000:,.1f} TB of accumulators "
          f"and a {dense_gb/1000:,.1f} TB result -- not attempted")
    if not os.path.exists(a.keep):
        planted = build(a.keep, a.n, a.p, a.n_rel)
        np.save(a.keep + ".planted.npy", np.array(sorted(planted)))
    else:
        planted = {tuple(x) for x in np.load(a.keep + ".planted.npy")}
        print(f"  reusing {a.keep}")

    t = time.time()
    df = king_pairs(a.keep, min_kinship=a.min_kinship, sample_block=a.block,
                    backend=a.backend, verbose=True)
    el = time.time() - t
    npairs = a.n * (a.n - 1) // 2
    found = {(int(i), int(j)) for i, j in zip(df.i, df.j)}
    hit = len(planted & found)
    print(f"\n  elapsed          {el:,.1f} s")
    print(f"  pairs evaluated  {npairs:,}  ({npairs/el/1e9:,.2f} G pair/s)")
    print(f"  emitted          {len(df):,} at kinship >= {a.min_kinship}")
    print(f"  planted recovered{hit:>6,} / {len(planted):,}"
          f"   {'OK' if hit == len(planted) else 'MISSING'}")
    if len(df):
        print(f"  top kinship      {df.kinship.iloc[0]:.6f} (duplicates = 0.5)")
    print(f"  peak RSS         {rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
