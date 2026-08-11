"""Dump paired cugen / plink2 values for r, r2, D and D' on real data.

Produces a parquet of matched pairs so the agreement can be *seen* rather than
summarised as a max-error. r and r2 should sit on the identity line; D and D'
agree for all but a small tail where the likelihood has multiple admissible
roots and the two implementations pick different ones.

    python benchmarks/concordance.py --bfile chr22_maf01 \
        --cugen chr22_maf01.cugen --p 3000 --window 300
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import subprocess

import numpy as np
import pandas as pd

from cugen.ld import ld_matrix


def read_vcor(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.lstrip("#") for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--cugen", required=True)
    ap.add_argument("--p", type=int, default=3000)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--out", default="concordance.parquet")
    a = ap.parse_args()

    bim = pd.read_csv(a.bfile + ".bim", sep="\t", header=None)
    bim.iloc[:a.p, 1].to_csv("/tmp/_ck.txt", index=False, header=False)
    win = ["--ld-window", str(a.window + 1), "--ld-window-kb", "999999",
           "--ld-window-r2", "0"]

    # unphased r for the r/r2 comparison; phased for D/D'
    for flag, cols, out in (
            ("--r-unphased", "cols=chrom,pos,id,maj,nonmaj", "/tmp/_cu"),
            ("--r-phased", "cols=chrom,pos,id,maj,nonmaj,d,dprime", "/tmp/_cp")):
        r = subprocess.run(
            ["plink2", "--bfile", a.bfile, "--extract", "/tmp/_ck.txt", flag,
             "allow-ambiguous-allele", cols, *win, "--out", out, "--silent"],
            capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"plink2 failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    gu, gp = read_vcor("/tmp/_cu.vcor"), read_vcor("/tmp/_cp.vcor")

    # cugen with every statistic, oriented to the major allele like plink2
    df = ld_matrix(a.cugen, variant_range=(0, a.p), window=a.window,
                   min_r2=0.0, sign_reference="major", max_pairs=10 ** 15,
                   verbose=False)
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    ids = bim.iloc[:a.p, 1].to_numpy()
    df = df.assign(_a=ids[df.gidx_a.to_numpy()], _b=ids[df.gidx_b.to_numpy()])

    m = df.merge(gu[["ID_A", "ID_B", "UNPHASED_R"]],
                 left_on=["_a", "_b"], right_on=["ID_A", "ID_B"], how="inner")
    m = m.merge(gp[["ID_A", "ID_B", "D", "DPRIME"]].rename(
        columns={"D": "D_PLINK", "DPRIME": "DP_PLINK"}),
        left_on=["_a", "_b"], right_on=["ID_A", "ID_B"], how="inner")

    out = pd.DataFrame({
        "r_cugen": m["R"].astype(float),
        "r_plink": m["UNPHASED_R"].astype(float),
        "r2_cugen": m["R2"].astype(float),
        "r2_plink": (m["UNPHASED_R"].astype(float) ** 2),
        "d_cugen": m["D"].astype(float),
        "d_plink": m["D_PLINK"].astype(float),
        "dp_cugen": m["DP"].astype(float),
        "dp_plink": m["DP_PLINK"].astype(float),
        "maf_a": m["MAF_A"].astype(float),
        "maf_b": m["MAF_B"].astype(float),
        "n_obs": m["N_OBS"].astype(int),
    })
    out.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(out):,} matched pairs")
    for lbl, c1, c2 in (("r", "r_cugen", "r_plink"),
                        ("r2", "r2_cugen", "r2_plink"),
                        ("D", "d_cugen", "d_plink"),
                        ("D'", "dp_cugen", "dp_plink")):
        e = (out[c1] - out[c2]).abs()
        print(f"  {lbl:3s} max|err| {e.max():.3e}   rms {np.sqrt((e**2).mean()):.3e}"
              f"   >1e-4: {(e > 1e-4).sum():,} ({100*(e > 1e-4).mean():.4f}%)")


if __name__ == "__main__":
    main()
