"""Compare cugen.impute against Beagle 5.5 on the paper's own fixture.

Reports the metric the 2018 paper reports, which is NOT the aggregate dosage r2
usually quoted:

    "We report the squared correlation (r2) between the true number of
     non-major alleles on a haplotype (0 or 1) and the posterior imputed allele
     probability. ... we binned imputed minor alleles according to the minor
     allele count in the reference sample, and report r2 for the imputed minor
     alleles in each minor allele count bin."

Per HAPLOTYPE, against the NON-MAJOR allele, binned by minor allele COUNT in
the reference panel. Computing the more familiar per-genotype dosage r2 binned
by population MAF gives numbers that look reasonable and are not comparable to
the paper's, so both are reported here and labelled.

    python benchmarks/validate_beagle.py --workdir /root/fixture --out cmp.json
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

BEAGLE_JAR_URL = ("https://faculty.washington.edu/browning/beagle/"
                  "beagle.27Feb25.75f.jar")
MAC_BINS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100),
            (101, 200), (201, 500), (501, 1000), (1001, 2000), (2001, 10**9)]


def read_gt_dosage(vcf, n_expected_samples=None):
    """(n_samples, n_markers) dosages and (n_markers,) positions from a VCF."""
    import subprocess as sp
    q = sp.run(["bcftools", "query", "-f", "%POS[\\t%GT]\\n", vcf],
               capture_output=True, text=True, check=True)
    pos, rows = [], []
    for line in q.stdout.splitlines():
        f = line.split("\t")
        pos.append(int(f[0]))
        rows.append([_gt(g) for g in f[1:]])
    return np.asarray(rows, dtype=np.float32).T, np.asarray(pos, dtype=np.int64)


def _gt(g):
    if g in (".", "./.", ".|."):
        return np.nan
    a = g.replace("|", "/").split("/")
    return float(sum(int(x) for x in a if x != "."))


def read_ds(vcf):
    """Beagle's DS field: (n_samples, n_markers) allele dosages, plus POS."""
    q = subprocess.run(["bcftools", "query", "-f", "%POS[\\t%DS]\\n", vcf],
                       capture_output=True, text=True, check=True)
    pos, rows = [], []
    for line in q.stdout.splitlines():
        f = line.split("\t")
        pos.append(int(f[0]))
        rows.append([float(x) if x not in (".", "") else np.nan for x in f[1:]])
    return np.asarray(rows, dtype=np.float32).T, np.asarray(pos, dtype=np.int64)


def paper_r2_by_mac(hap_prob, truth_hap, mac, minor_is_alt):
    """The paper's metric: per-haplotype r2 against the non-major allele.

    hap_prob    : (T, M) P(allele 1) per target haplotype
    truth_hap   : (T, M) observed allele (0/1)
    mac         : (M,) minor allele count in the REFERENCE panel
    minor_is_alt: (M,) whether allele 1 is the minor allele
    """
    p = np.where(minor_is_alt[None, :], hap_prob, 1.0 - hap_prob)
    t = np.where(minor_is_alt[None, :], truth_hap, 1.0 - truth_hap)
    out = []
    for lo, hi in MAC_BINS:
        sel = (mac >= lo) & (mac <= hi)
        n = int(sel.sum())
        if n < 2:
            out.append({"mac_lo": lo, "mac_hi": hi, "n_markers": n, "r2": None})
            continue
        a = p[:, sel].ravel()
        b = t[:, sel].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        r2 = None
        if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0:
            r2 = float(np.corrcoef(a[ok], b[ok])[0, 1] ** 2)
        out.append({"mac_lo": lo, "mac_hi": hi, "n_markers": n, "r2": r2})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/root/fixture")
    ap.add_argument("--chrom", type=int, default=20)
    ap.add_argument("--out", default="beagle_compare.json")
    ap.add_argument("--ne", type=int, default=100_000)
    ap.add_argument("--skip-beagle", action="store_true")
    a = ap.parse_args()
    W = a.workdir
    res = {"chrom": a.chrom, "ne": a.ne}

    import pandas as pd
    from cugen.impute import impute

    ref_pos = np.load(f"{W}/ref_pos.npy")
    tgt_gidx = np.load(f"{W}/tgt_gidx.npy")
    ann = pd.DataFrame({"gidx": np.arange(ref_pos.size), "POS": ref_pos,
                        "CHR": str(a.chrom),
                        "ID": [f"{a.chrom}:{p}" for p in ref_pos]})
    gmap = f"{W}/plink.chr{a.chrom}.GRCh37.map"

    print("=== cugen ===", flush=True)
    t0 = time.perf_counter()
    frame = impute(f"{W}/target.cugen", ref=f"{W}/reference.cugen",
                   annotation=ann, map=gmap, chrom=a.chrom, ne=a.ne,
                   out=f"{W}/cugen_out.cugen", verbose=True)
    res["cugen_seconds"] = round(time.perf_counter() - t0, 2)
    frame.to_parquet(f"{W}/cugen_info.parquet")

    # allele probabilities per haplotype, recomputed for the paper's metric
    from cugen.io import CugenReader
    with CugenReader(f"{W}/cugen_out.cugen") as r:
        n_s, n_v = r.n_samples, r.n_variants
        dose = np.frombuffer(r.read_packed_bytes(), dtype=np.float16
                             ).reshape(n_v, n_s).T.astype(np.float32)

    print("\n=== truth ===", flush=True)
    truth, tpos = read_gt_dosage(f"{W}/target.truth.vcf.gz")
    assert np.array_equal(tpos, ref_pos), "truth positions != reference markers"

    is_typed = np.zeros(ref_pos.size, dtype=bool)
    is_typed[tgt_gidx] = True
    imp = ~is_typed
    af = frame["AF"].to_numpy()
    # minor allele count in the reference panel, from the panel itself
    with CugenReader(f"{W}/reference.cugen") as rr:
        n_ref_hap = 2 * rr.n_samples
        ac = np.rint(rr.mu_x * rr.n_samples).astype(np.int64)   # ALT count
    mac = np.minimum(ac, n_ref_hap - ac)
    minor_is_alt = ac <= (n_ref_hap - ac)

    ok = np.isfinite(dose[:, imp]) & np.isfinite(truth[:, imp])
    res["cugen_dosage_r2_all_imputed"] = float(
        np.corrcoef(dose[:, imp][ok], truth[:, imp][ok])[0, 1] ** 2)
    print(f"  cugen aggregate dosage r2 (imputed markers): "
          f"{res['cugen_dosage_r2_all_imputed']:.4f}")

    # paper metric needs per-haplotype probabilities; dose/2 is the per-hap mean
    res["cugen_paper_r2_by_mac"] = paper_r2_by_mac(
        dose / 2.0, truth / 2.0, mac, minor_is_alt)

    if not a.skip_beagle:
        print("\n=== beagle 5.5 ===", flush=True)
        jar = f"{W}/beagle.jar"
        if not os.path.exists(jar):
            subprocess.run(["curl", "-fsSL", "-o", jar, BEAGLE_JAR_URL],
                           check=True)
        t0 = time.perf_counter()
        cmd = ["java", "-Xmx60g", "-jar", jar,
               f"gt={W}/target.masked.vcf.gz",
               f"ref={W}/reference.vcf.gz",
               f"map={gmap}", f"ne={a.ne}",
               f"out={W}/beagle_out", "impute=true", "nthreads=8"]
        print("  $ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        res["beagle_seconds"] = round(time.perf_counter() - t0, 2)
        bdose, bpos = read_ds(f"{W}/beagle_out.vcf.gz")
        common = np.intersect1d(bpos, ref_pos)
        bi = np.searchsorted(bpos, common)
        ri = np.searchsorted(ref_pos, common)
        m = imp[ri]
        okb = np.isfinite(bdose[:, bi][:, m]) & np.isfinite(truth[:, ri][:, m])
        res["beagle_dosage_r2_all_imputed"] = float(np.corrcoef(
            bdose[:, bi][:, m][okb], truth[:, ri][:, m][okb])[0, 1] ** 2)
        res["cugen_vs_beagle_dose_corr"] = float(np.corrcoef(
            dose[:, ri][:, m].ravel(), bdose[:, bi][:, m].ravel())[0, 1])
        res["cugen_vs_beagle_max_abs_diff"] = float(np.nanmax(
            np.abs(dose[:, ri][:, m] - bdose[:, bi][:, m])))
        res["n_common_markers"] = int(common.size)
        res["speedup_vs_beagle"] = round(
            res["beagle_seconds"] / max(res["cugen_seconds"], 1e-9), 2)
        print(f"  beagle aggregate dosage r2: "
              f"{res['beagle_dosage_r2_all_imputed']:.4f}")
        print(f"  cugen {res['cugen_seconds']:.1f}s vs beagle "
              f"{res['beagle_seconds']:.1f}s -> {res['speedup_vs_beagle']}x")

    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("\n  MAC bin      n markers      paper r2")
    for b in res["cugen_paper_r2_by_mac"]:
        r = "n/a" if b["r2"] is None else f"{b['r2']:.4f}"
        print(f"  {b['mac_lo']:>5}-{b['mac_hi']:<7} {b['n_markers']:>9,}   {r:>8}")


if __name__ == "__main__":
    main()
