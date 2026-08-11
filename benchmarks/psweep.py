"""Scaling along the VARIANT axis, up to a whole chromosome.

Real 1000 Genomes chr22 (n = 2,504 samples), all-pairs LD, cugen vs plink2.
This is the axis a genomics reader cares about most: p grows to a whole
chromosome while n stays at cohort size.

Note n = 2,504 is cugen's WORST regime -- the GEMM's contraction dimension is
tiny, so the device is least well fed. The sample-count sweep covers the other
axis. Reporting both keeps the picture honest.

Writes JSON for plotting.
"""
import json
import os
import resource
import subprocess
import sys
import time

_ROOT = "/root/cugen"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))

import pandas as pd

from _peak import PeakSampler
from cugen.ld import ld_matrix

BF = "/root/data/chr22_maf01"
CG = "/root/data/chr22_maf01.cugen"
GRID = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else
                         "1000,2000,5000,10000,20000,50000,100000,170949"
                         ).split(",")]
PLINK_MAX_P = int(os.environ.get("PLINK_MAX_P", "170949"))
OUT = "/root/psweep.json"

bim = pd.read_csv(BF + ".bim", sep="\t", header=None)
print(f"1000 Genomes chr22, n = 2,504 samples, all-pairs, min_r2 = 0.2, "
      f"plink2 threads = {os.cpu_count()}")
print(f"{'variants':>9s} {'pairs':>16s} | {'plink2 s':>9s} {'RSS GiB':>8s} | "
      f"{'cugen s':>8s} {'GPU GiB':>8s} | {'speedup':>8s} | {'emitted':>11s}")
print("-" * 104)

rows = []
for p in GRID:
    bim.iloc[:p, 1].to_csv("/tmp/_pk.txt", index=False, header=False)
    npairs = p * (p - 1) // 2

    pl = pl_rss = None
    if p <= PLINK_MAX_P:
        t0 = time.perf_counter()
        r = subprocess.run(
            ["plink2", "--bfile", BF, "--extract", "/tmp/_pk.txt",
             "--r2-unphased", "allow-ambiguous-allele", "cols=chrom,pos,id",
             "--ld-window", "999999", "--ld-window-kb", "999999",
             "--ld-window-r2", "0.2", "--threads", str(os.cpu_count()),
             "--out", "/tmp/_pk", "--silent"], capture_output=True, text=True)
        pl = time.perf_counter() - t0
        if r.returncode:
            print(f"{p:>9,}  plink2 FAILED: {(r.stdout + r.stderr)[-120:]}")
            pl = None
        pl_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 ** 2

    with PeakSampler() as smp:
        t0 = time.perf_counter()
        df = ld_matrix(CG, variant_range=(0, p), min_r2=0.2, stats=("r", "r2"),
                       sign_reference="major", output="/tmp/_pk_cg.tsv",
                       max_pairs=10 ** 15, verbose=False)
        cg = time.perf_counter() - t0

    spd = (pl / cg) if pl else float("nan")
    rows.append(dict(p=p, pairs=npairs, plink_s=pl, plink_rss_gib=pl_rss,
                     cugen_s=cg, cugen_gpu_gib=smp.peak_gib,
                     emitted=int(len(df)), speedup=spd))
    json.dump(rows, open(OUT, "w"), indent=2)
    ps = f"{pl:>9.2f}" if pl else "        -"
    pr = f"{pl_rss:>8.2f}" if pl_rss else "       -"
    sp = f"{spd:>7.2f}x" if pl else "       -"
    print(f"{p:>9,} {npairs:>16,} | {ps} {pr} | {cg:>8.2f} "
          f"{smp.peak_gib:>8.2f} | {sp} | {len(df):>11,}")

print(f"\nwrote {OUT}")
