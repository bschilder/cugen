"""Compare cugen.ld.ld_matrix against plink2 at scale, on real data.

Both tools are fed the SAME PLINK1 .bed. That matters: .bed is 2-bit hard
calls, so phase is already gone for both, which makes the --r2-phased
comparison apples-to-apples rather than the phased/unphased mismatch you get
from a stock 1000 Genomes VCF.

    python benchmarks/compare_plink.py --bfile data/chr22_maf01 \
        --cugen data/chr22_maf01.cugen --p 5000 --window 500
"""
import argparse
import subprocess
import time

import numpy as np
import pandas as pd

from cugen.ld import ld_matrix


def run_plink(bfile, out, window, extra):
    cmd = ["plink2", "--bfile", bfile, "--out", out, "--silent",
           "--ld-window", str(window + 1), "--ld-window-kb", "999999",
           "--ld-window-r2", "0"] + extra
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode:
        raise SystemExit(f"plink2 failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return dt


def read_vcor(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.lstrip("#") for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--cugen", required=True)
    ap.add_argument("--p", type=int, default=5000)
    ap.add_argument("--window", type=int, default=500)
    a = ap.parse_args()

    # restrict plink to the same leading p variants
    bim = pd.read_csv(a.bfile + ".bim", sep="\t", header=None)
    keep = bim.iloc[:a.p, 1]
    keep_file = "/tmp/_keep.txt"
    keep.to_csv(keep_file, index=False, header=False)

    print(f"comparing first {a.p:,} variants, window={a.window}")
    t_pl = run_plink(a.bfile, "/tmp/_gold_u", a.window,
                     ["--extract", keep_file, "--r-unphased",
                      "allow-ambiguous-allele", "cols=chrom,pos,id,maj,nonmaj"])
    t_pl += run_plink(a.bfile, "/tmp/_gold_p", a.window,
                      ["--extract", keep_file, "--r-phased",
                       "allow-ambiguous-allele",
                       "cols=chrom,pos,id,maj,nonmaj,d,dprime"])
    print(f"plink2 wall (both runs) : {t_pl:8.2f} s")

    t0 = time.perf_counter()
    df = ld_matrix(a.cugen, variant_range=(0, a.p), window=a.window,
                   sign_reference="major", verbose=False)
    t_cg = time.perf_counter() - t0
    print(f"cugen  wall (all stats) : {t_cg:8.2f} s   -> {len(df):,} pairs")
    print(f"speedup                 : {t_pl / max(t_cg, 1e-9):8.2f}x")

    gu = read_vcor("/tmp/_gold_u.vcor")
    gp = read_vcor("/tmp/_gold_p.vcor")
    ids = bim.iloc[:a.p, 1].to_numpy()
    df = df.assign(_a=ids[df["gidx_a"].to_numpy()],
                   _b=ids[df["gidx_b"].to_numpy()])
    m = df.merge(gu[["ID_A", "ID_B", "UNPHASED_R"]],
                 left_on=["_a", "_b"], right_on=["ID_A", "ID_B"], how="inner")
    m = m.merge(gp[["ID_A", "ID_B", "D", "DPRIME"]].rename(
        columns={"D": "D_PL", "DPRIME": "DP_PL"}),
        left_on=["_a", "_b"], right_on=["ID_A", "ID_B"], how="inner")
    print(f"\npairs matched to plink2 : {len(m):,}")
    if not len(m):
        raise SystemExit("no overlap -- check --window and --extract")

    print(f"\n{'quantity':22s} {'max |err|':>12s} {'rms':>12s}  verdict")
    print("-" * 62)
    worst = {}
    for label, ours, theirs, tol in (
            ("r  vs UNPHASED_R", m["R"], m["UNPHASED_R"], 2e-6),
            ("D  vs D",          m["D"], m["D_PL"],       2e-6),
            ("D' vs DPRIME",     m["DP"], m["DP_PL"],     2e-6)):
        e = np.abs(ours.to_numpy() - theirs.to_numpy())
        worst[label] = float(e.max())
        print(f"{label:22s} {e.max():12.3e} {np.sqrt((e**2).mean()):12.3e}"
              f"  {'PASS' if e.max() < tol else 'FAIL'}")
    print("\nplink2 text .vcor carries ~6 significant figures, so ~1e-6 is the")
    print("format floor, not our error.")


if __name__ == "__main__":
    main()
