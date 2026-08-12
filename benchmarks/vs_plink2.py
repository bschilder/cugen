"""Head-to-head: cugen.ld on one GPU vs plink2 on CPU.

Fairness rules, because a speedup number is only as good as its controls:
  * Same variants, same window, same r^2 threshold, same .bed source.
  * Both WRITE their output to disk. cugen returning a DataFrame in memory
    while plink2 serialises a .vcor would be an unfair comparison at high
    output volume, which is exactly the regime that matters.
  * plink2 gets every core on the box, and the thread count is reported. A
    "GPU beats CPU" claim against a single-threaded baseline is not a claim.
  * Memory is peak RSS for plink2, and peak GPU + peak host RSS for cugen --
    cugen uses both, so reporting only GPU bytes would flatter it.
  * Only r^2 is compared. plink2 --r2-phased is a different statistic, and
    cugen's D' is known to diverge from it on ~0.005% of pairs (see #2).

    python benchmarks/vs_plink2.py --bfile chr22_maf01 --cugen chr22_maf01.cugen
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import resource
import subprocess
import time

import pandas as pd

from _peak import PeakSampler
from cugen.ld import ld_matrix


def peak_rss_gib():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 2) if sys.platform == "darwin" else r / (1024 ** 2)


HAVE_GNU_TIME = os.path.exists("/usr/bin/time")


def run_plink(bfile, keep, out, window, min_r2, threads):
    """Returns (wall_s, peak_rss_gib, n_rows, err).

    Peak RSS comes from GNU time when present, else from RUSAGE_CHILDREN.
    NB ru_maxrss over children is a running maximum across all reaped
    children, so it is only a per-run peak because p increases monotonically
    here and plink's footprint grows with p. Don't reuse this helper for a
    descending sweep without fixing that.
    """
    base = ["plink2", "--bfile", bfile,
            "--extract", keep, "--r2-unphased", "allow-ambiguous-allele",
            "cols=chrom,pos,id", "--ld-window", str(window + 1),
            "--ld-window-kb", "999999", "--ld-window-r2", str(min_r2),
            "--threads", str(threads), "--out", out, "--silent"]
    cmd = (["/usr/bin/time", "-v"] + base) if HAVE_GNU_TIME else base
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode:
        return None, None, None, r.stdout[-800:] + r.stderr[-800:]
    peak = None
    for line in r.stderr.splitlines():
        if "Maximum resident set size" in line:
            peak = int(line.split()[-1]) / (1024 ** 2)      # kB -> GiB
    if peak is None:
        ru = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak = ru / (1024 ** 2) if sys.platform != "darwin" else ru / (1024 ** 3)
    n = 0
    p = out + ".vcor"
    if os.path.exists(p):
        with open(p) as f:
            n = sum(1 for _ in f) - 1
    return dt, peak, n, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--cugen", required=True)
    ap.add_argument("--grid", default="2000,5000,10000,20000,50000")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--min-r2", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="vs_plink2.json")
    a = ap.parse_args()

    bim = pd.read_csv(a.bfile + ".bim", sep="\t", header=None)
    print(f"plink2 threads: {a.threads}   window: {a.window} variants   "
          f"min_r2: {a.min_r2}")
    print(f"{'p':>8s} {'pairs':>14s} | {'plink2 s':>9s} {'RSS GiB':>8s} | "
          f"{'cugen s':>9s} {'GPU GiB':>8s} {'RSS GiB':>8s} | {'speedup':>8s}")
    print("-" * 96)

    rows = []
    for p in [int(x) for x in a.grid.split(",")]:
        keep = "/tmp/_keep.txt"
        bim.iloc[:p, 1].to_csv(keep, index=False, header=False)

        pl_s, pl_rss, pl_n, err = run_plink(a.bfile, keep, "/tmp/_pl", a.window,
                                            a.min_r2, a.threads)
        if err:
            print(f"{p:>8,}  plink2 FAILED: {err[:120]}")
            continue

        with PeakSampler() as smp:
            t0 = time.perf_counter()
            df = ld_matrix(a.cugen, variant_range=(0, p), window=a.window,
                           min_r2=a.min_r2, stats=("r", "r2"),
                           sign_reference="major", max_pairs=10**15,
                           output="/tmp/_cg.tsv", verbose=False)
            cg_s = time.perf_counter() - t0
        cg_rss = peak_rss_gib()
        npairs = sum(min(a.window, p - 1 - i) for i in range(p))
        spd = pl_s / cg_s if cg_s else float("nan")
        rows.append(dict(p=p, pairs=npairs, plink_s=pl_s, plink_rss_gib=pl_rss,
                         plink_rows=pl_n, cugen_s=cg_s,
                         cugen_gpu_gib=smp.peak_gib, cugen_rss_gib=cg_rss,
                         cugen_rows=len(df), speedup=spd,
                         threads=a.threads, window=a.window, min_r2=a.min_r2))
        print(f"{p:>8,} {npairs:>14,} | {pl_s:>9.2f} {pl_rss:>8.2f} | "
              f"{cg_s:>9.2f} {smp.peak_gib:>8.2f} {cg_rss:>8.2f} | "
              f"{spd:>7.2f}x")
        if pl_n is not None and abs(pl_n - len(df)) > max(10, 0.001 * pl_n):
            print(f"          ^ WARNING row counts differ: plink2 {pl_n:,} vs "
                  f"cugen {len(df):,} — workloads may not match")

    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    if rows:
        best = max(rows, key=lambda r: r["speedup"])
        cross = [r for r in rows if r["speedup"] >= 1.0]
        print(f"best speedup: {best['speedup']:.2f}x at p={best['p']:,}")
        print("crossover: " + (f"cugen first wins at p={cross[0]['p']:,}"
                               if cross else "plink2 wins at every size tested"))


if __name__ == "__main__":
    main()
