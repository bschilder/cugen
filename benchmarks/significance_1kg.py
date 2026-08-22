"""LD significance testing validated on real 1000 Genomes data.

The unit tests use simulated panels, which is right for pinning arithmetic but
says nothing about how the significance layer behaves on a real minor-allele
frequency spectrum, real LD structure, or real population structure. This script
runs the same claims against 1000 Genomes chr22.

Fixture (built by the commands in the docstring of build_fixture below):
  1kGP high-coverage phased panel, chr22:20-21Mb, biallelic SNVs
  EUR      503 samples, 800 variants, MAF >= 0.01 within EUR
  EAS      504 samples, 800 variants, MAF >= 0.01 within EAS
  EUR_EAS 1007 samples, 800 variants, MAF >= 0.01 in the pooled sample
  EURrare  503 samples, 800 variants, NO frequency filter

Usage:  uv run --with cyvcf2 python benchmarks/significance_1kg.py --dir DIR
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cugen import ld as L  # noqa: E402

CPU = dict(backend="numpy", verbose=False)
LN10 = math.log(10.0)


def say(msg):
    print(f"\n===== {msg} =====")


def oracle_neglog10p(chi2):
    from scipy.special import log_ndtr
    return -(math.log(2.0) + log_ndtr(-math.sqrt(chi2))) / LN10


# --------------------------------------------------------------- 1. plink2
def plink2_parity(vcf, cugen_path, out_prefix, n_hap):
    """chi2 = N_hap * r2 checked against plink2's own r2 on real data.

    plink2 emits no p-value, so this validates the STATISTIC against an
    independent LD implementation and leaves the tail to scipy.
    """
    say("1. chi2 vs plink2 --r-phased on real chr22")
    if not shutil_which("plink2"):
        print("  plink2 not on PATH -- skipped")
        return
    # --make-pgen from a PHASED VCF, never from a .bed. A .bed cannot store
    # phase, so plink2 --r-phased on bed input EM-estimates haplotype
    # frequencies and answers a different question -- measured deviation 0.616
    # against 5.71e-07 for a phased PGEN.
    subprocess.run(["plink2", "--vcf", vcf, "--make-pgen", "--out", out_prefix,
                    "--silent"], check=True)
    subprocess.run(["plink2", "--pfile", out_prefix, "--r-phased",
                    "allow-ambiguous-allele", "cols=chrom,pos,id",
                    "--ld-window", "999999", "--ld-window-kb", "999999",
                    "--ld-window-r2", "0", "--out", out_prefix, "--silent"],
                   check=True)
    import pandas as pd
    pl = pd.read_csv(out_prefix + ".vcor", sep="\t")
    pl.columns = [c.lstrip("#") for c in pl.columns]
    rcol = "PHASED_R" if "PHASED_R" in pl.columns else "R"
    # Join on variant ID, NOT position. Split multi-allelic sites share a POS,
    # so a (POS_A, POS_B) key is not unique and the merge silently becomes
    # many-to-many -- it produced 322,796 rows from 319,600 x 319,600 and a
    # bogus max error of 954. benchmarks/compare_plink.py documents the same
    # trap from the other direction (there every ID was ".").
    pl["key"] = list(zip(pl["ID_A"], pl["ID_B"]))
    assert pl["key"].is_unique, "plink2 keys are not unique"
    # 1000G phase3 IDs are almost all "." -- joining on them there would match
    # everything and report a spurious agreement rather than an error. The
    # high-coverage panel used here carries real chr:pos:ref:alt IDs, but that
    # is a property of the fixture, not of the code, so check it.
    placeholder = (pl["ID_A"] == ".").mean()
    assert placeholder < 0.01, (
        f"{placeholder:.1%} of plink2 variant IDs are '.'; this join would "
        f"match indiscriminately. Rebuild the PGEN with "
        f"--set-all-var-ids '@:#:$r:$a'.")

    df = L.ld_matrix(cugen_path, stats=("r2_phased", "chi2", "p"), **CPU)
    vid = variant_ids(vcf)
    df["key"] = list(zip(vid[df["gidx_a"].to_numpy()],
                         vid[df["gidx_b"].to_numpy()]))
    assert df["key"].is_unique, "cugen keys are not unique"
    m = df.merge(pl[["key", rcol]], on="key", how="inner")
    print(f"  pairs joined: {len(m):,} of {len(df):,} cugen / {len(pl):,} plink2")

    chi2_plink = n_hap * m[rcol].to_numpy(np.float64) ** 2
    e = np.abs(m["CHI2"].to_numpy(np.float64) - chi2_plink)
    rel = e / np.maximum(chi2_plink, 1e-30)
    print(f"  |chi2_cugen - N_hap*r2_plink|   max {e.max():.3e}   "
          f"median {np.median(e):.3e}")
    print(f"  relative                        max {rel.max():.3e}")
    print(f"  (plink2 .vcor carries ~6 significant figures, so the floor here "
          f"is ~1e-6 * chi2)")

    nl = np.array([oracle_neglog10p(c) for c in chi2_plink])
    e2 = np.abs(m["NEG_LOG10_P"].to_numpy(np.float64) - nl)
    print(f"  -log10(p) from plink2's r2 vs cugen's   max {e2.max():.3e}")


def variant_ids(vcf):
    out = subprocess.run(["bcftools", "query", "-f", "%ID\n", vcf],
                         capture_output=True, text=True, check=True)
    return np.array(out.stdout.split(), dtype=object)


def shutil_which(x):
    from shutil import which
    return which(x)


# ----------------------------------------------- 2. exact test, real rares
def exact_on_real_rare_variants(cugen_path, vcf):
    say("2. exact conditional test on real rare variants (EUR, no MAF filter)")
    from scipy.stats import fisher_exact

    df = L.ld_matrix(cugen_path, stats=("r2_phased", "p"), exact="auto", **CPU)
    fired = df["NEG_LOG10_P_EXACT"].notna()
    print(f"  pairs {len(df):,}   auto fired on {int(fired.sum()):,} "
          f"({fired.mean():.1%})")

    hap = haplotypes(cugen_path)
    N = hap.shape[0]
    a = df.loc[fired]
    worst = 0.0
    checked = 0
    for ga, gb, got in zip(a["gidx_a"].to_numpy(), a["gidx_b"].to_numpy(),
                           a["NEG_LOG10_P_EXACT"].to_numpy(np.float64)):
        if checked >= 400:
            break
        x, y = hap[:, int(ga)].astype(np.int64), hap[:, int(gb)].astype(np.int64)
        nAB, nA, nB = int((x & y).sum()), int(x.sum()), int(y.sum())
        tbl = [[nAB, nA - nAB], [nB - nAB, N - nA - nB + nAB]]
        want = -math.log10(fisher_exact(tbl)[1])
        worst = max(worst, abs(got - want))
        checked += 1
    print(f"  vs scipy.stats.fisher_exact on {checked} real pairs: "
          f"max |diff| {worst:.3e}")

    nl_as = a["NEG_LOG10_P"].to_numpy(np.float64)
    nl_ex = a["NEG_LOG10_P_EXACT"].to_numpy(np.float64)
    d = nl_as - nl_ex
    print(f"  asymptotic OVERSTATES significance on {(d > 0).mean():.1%} of them")
    print(f"  log10(p_exact/p_asym)  median {np.median(d):+.3f}  "
          f"p95 {np.percentile(d, 95):+.3f}  max {d.max():+.1f}")
    gws = 7.3
    print(f"  called significant at p<5e-8:  asymptotic {int((nl_as > gws).sum()):,}"
          f"   exact {int((nl_ex > gws).sum()):,}")
    if d.size:
        w = int(np.argmax(d))
        print(f"  worst pair: r2={a['R2_PHASED'].to_numpy()[w]:.4f} "
              f"N={int(a['N_OBS'].to_numpy()[w])}  "
              f"asym -log10p={nl_as[w]:.1f}  exact={nl_ex[w]:.1f}")


def haplotypes(cugen_path):
    from cugen.io import read_cugen
    return L._haplotypes_numpy(read_cugen(cugen_path)).T   # (H, n_variants)


# --------------------------------------------------- 3. real structure
def lambda_on_real_populations(paths):
    say("3. lambda_gc on real populations vs a real pooled sample")
    for name, p in paths:
        df = L.ld_matrix(p, stats=("r2_phased", "chi2", "p"), lambda_gc=True,
                         **CPU)
        lam = df.attrs["lambda_gc"]
        nl = df["NEG_LOG10_P"].to_numpy(np.float64)
        adj = df["NEG_LOG10_P_ADJ"].to_numpy(np.float64)
        gws = 7.3
        print(f"  {name:10s} n={df['N_OBS'].iloc[0] // 2:>5}  pairs={len(df):>8,}"
              f"  lambda={lam:8.2f}"
              f"  p<5e-8 raw {(nl > gws).mean():6.1%} -> adj {(adj > gws).mean():6.1%}")


# --------------------------------------------------- 4. multiple testing
def corrections_on_real_data(cugen_path):
    say("4. multiple-testing thresholds on real chr22 LD")
    every = L.ld_matrix(cugen_path, stats=("r2_phased", "p"), **CPU)
    m = len(every)
    for label, kw in (("none", {}),
                      ("Bonferroni 0.05", dict(correction="bonferroni")),
                      ("BH-FDR 0.05", dict(correction="fdr"))):
        df = L.ld_matrix(cugen_path, stats=("r2_phased", "p"), alpha=0.05,
                         **kw, **CPU) if kw else every
        r2 = df["R2_PHASED"].to_numpy(np.float64)
        print(f"  {label:16s} emitted {len(df):>8,} of {m:,} ({len(df)/m:6.1%})"
              f"   min r2 retained {r2.min() if len(r2) else float('nan'):.5f}")
    print(f"  for reference, pairs with r2 >= 0.8: "
          f"{int((every['R2_PHASED'] >= 0.8).sum()):,} ({(every['R2_PHASED'] >= 0.8).mean():.3%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory with the fixtures")
    a = ap.parse_args()
    d = a.dir
    print(__doc__.split("Usage:")[0].rstrip())
    plink2_parity(f"{d}/s_EUR.vcf.gz", f"{d}/EUR.cugen", f"{d}/pl_EUR", 2 * 503)
    exact_on_real_rare_variants(f"{d}/EURrare.cugen", f"{d}/s_EURrare.vcf.gz")
    lambda_on_real_populations([("EUR", f"{d}/EUR.cugen"),
                                ("EAS", f"{d}/EAS.cugen"),
                                ("EUR+EAS", f"{d}/EUR_EAS.cugen")])
    corrections_on_real_data(f"{d}/EUR.cugen")


if __name__ == "__main__":
    main()
