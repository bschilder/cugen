"""cugen.ld.ld_clump against plink2 --clump, and device vs host.

Two questions, deliberately separated:

  1. CORRECTNESS at scale -- does the device path still reproduce plink2
     exactly on real data, not just on a 150-variant fixture?
  2. Does the device path actually hold memory flat where the host path
     cannot? The interesting regime is clumping-and-thresholding for
     polygenic scores (p1=1, r2=0.1), where every variant is an index
     candidate and the edge set is orders of magnitude larger than at the
     --clump-r2 0.5 default.

    python benchmarks/bench_clump.py --cugen chr22.cugen --bfile chr22 \
        --annotation chr22_ann.tsv --out clump_bench.json
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd

from _peak import PeakSampler

PARITY_COLS = ["POS", "ID", "P", "TOTAL", "NONSIG", "SP2"]


def run_plink(bfile, ss, p1, p2, r2, kb, out):
    t0 = time.perf_counter()
    r = subprocess.run(
        ["plink2", "--bfile", bfile, "--clump", ss, "--clump-unphased",
         "--clump-p1", str(p1), "--clump-p2", str(p2), "--clump-r2", str(r2),
         "--clump-kb", str(kb), "--out", out, "--silent"],
        capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode:
        return None, dt, r.stdout[-800:] + r.stderr[-800:]
    return f"{out}.clumps", dt, None


def compare(cg, pk_path):
    """Field-by-field, not just clump counts: an implementation can get the
    right index variants with wrong TOTALs, which is exactly what happened
    once already."""
    pk = pd.read_csv(pk_path, sep="\t").rename(columns=lambda c: c.lstrip("#"))
    out = {"plink_clumps": len(pk), "cugen_clumps": len(cg)}
    if set(pk["ID"]) != set(cg["ID"]):
        out["index_sets_equal"] = False
        out["only_plink"] = sorted(set(pk.ID) - set(cg.ID))[:10]
        out["only_cugen"] = sorted(set(cg.ID) - set(pk.ID))[:10]
        return out
    out["index_sets_equal"] = True
    a = pk.sort_values("ID").reset_index(drop=True)
    b = cg.sort_values("ID").reset_index(drop=True)
    diff = {}
    for c in PARITY_COLS:
        if c not in a.columns or c not in b.columns:
            continue
        if c == "P":
            d = ~np.isclose(a[c].to_numpy(float), b[c].to_numpy(float),
                            rtol=1e-6)
        else:
            d = a[c].astype(str).to_numpy() != b[c].astype(str).to_numpy()
        if d.any():
            diff[c] = int(d.sum())
    out["columns_differing"] = diff
    out["identical"] = not diff
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cugen", required=True)
    ap.add_argument("--bfile", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--sumstats", default=None,
                    help="default: synthesise from the annotation")
    ap.add_argument("--out", default="clump_bench.json")
    ap.add_argument("--cpu-subset", type=int, default=20000,
                    help="variants for the host-path datapoint (0 to skip). "
                         "The numpy reference is O(p*window*n) and will not "
                         "finish at chromosome scale, so it is measured on a "
                         "capped prefix and reported as such.")
    a = ap.parse_args()

    from cugen.ld import ld_clump

    ann = pd.read_csv(a.annotation, sep=None, engine="python")
    ann = ann.rename(columns={c: c.lstrip("#") for c in ann.columns})

    if a.sumstats:
        ss_path = a.sumstats
    else:
        # Synthetic p-values, seeded, shaped like a REAL GWAS.
        #
        # The first version drew the exponent uniformly on [0.05, 14], which
        # puts 72% of variants below 5e-8 -- a genome-wide-significant rate
        # about a thousand times too high. Every configuration then behaved
        # like the most extreme one, and the membership count it produced OOM
        # killed the benchmark on chr22. The clump ALGORITHM does not care
        # where p came from, but the SHAPE of the distribution sets the edge
        # count, so it has to be plausible.
        #
        # Null-dominated with a small significant tail: ~0.1% of variants
        # carry real signal, which is what a well-powered GWAS looks like.
        rng = np.random.default_rng(20260811)
        pv = rng.uniform(0.0, 1.0, size=len(ann))
        n_hit = max(1, len(ann) // 1000)
        hits = rng.choice(len(ann), size=n_hit, replace=False)
        pv[hits] = 10.0 ** (-rng.uniform(5.0, 20.0, size=n_hit))
        print(f"synthetic sumstats: {len(ann):,} variants, "
              f"{int((pv <= 5e-8).sum()):,} genome-wide significant "
              f"({100 * (pv <= 5e-8).mean():.3f}%)")
        ss_path = "/tmp/bench_ss.tsv"
        pd.DataFrame({"ID": ann["ID"], "P": pv}).to_csv(
            ss_path, sep="\t", index=False)

    results = []
    # (p1, p2, r2, kb, label)
    grid = [
        (1e-4, 0.01, 0.50, 250, "standard GWAS clumping"),
        (1e-2, 0.01, 0.50, 250, "loose p1"),
        (1.0,  0.01, 0.20, 250, "C+T, r2=0.2"),
        (1.0,  0.01, 0.10, 250, "C+T for PRS (p1=1, r2=0.1)"),
    ]
    for p1, p2, r2, kb, label in grid:
        rec = {"label": label, "p1": p1, "p2": p2, "r2": r2, "kb": kb}
        print(f"\n=== {label}: p1={p1:g} p2={p2:g} r2={r2:g} kb={kb} ===",
              flush=True)
        # GPU only at full scale. The numpy reference is O(p * window * n) --
        # about 1e12 operations on chr22 at a 250 kb window -- so running it
        # here would not produce a slow number, it would produce no number at
        # all and burn the whole session. Backend agreement is established on
        # the fixture by the unit tests; the meaningful comparison at scale is
        # against plink2. See --cpu-subset for a tractable host-path datapoint.
        gpu_df = None
        try:
            with PeakSampler() as s:
                t0 = time.perf_counter()
                gpu_df = ld_clump(a.cugen, ss_path, annotation=ann, p1=p1,
                                  p2=p2, r2=r2, kb=kb, backend="gpu",
                                  verbose=True)
                dt = time.perf_counter() - t0
            rec["gpu_s"] = round(dt, 3)
            rec["gpu_peak_gib"] = s.peak_gib
            rec["gpu_clumps"] = int(len(gpu_df))
            print(f"  cugen[gpu]  {dt:8.2f} s  peak {s.peak_gib:6.2f} GiB  "
                  f"{len(gpu_df):,} clumps")
        except Exception as e:                            # noqa: BLE001
            rec["gpu_error"] = f"{type(e).__name__}: {e}"
            print(f"  cugen[gpu]  FAILED {type(e).__name__}: {e}")

        pk_path, pk_s, err = run_plink(a.bfile, ss_path, p1, p2, r2, kb,
                                       f"/tmp/pk_{abs(hash(label)) % 9999}")
        rec["plink_s"] = round(pk_s, 3)
        # One tractable host-path datapoint, on a prefix, so the device-vs-host
        # memory claim rests on a measurement rather than an extrapolation.
        if a.cpu_subset and gpu_df is not None:
            sub = ann.head(a.cpu_subset)
            try:
                with PeakSampler() as s2:
                    t2 = time.perf_counter()
                    c = ld_clump(a.cugen, ss_path, annotation=sub, p1=p1,
                                 p2=p2, r2=r2, kb=kb, backend="numpy",
                                 verbose=False)
                    rec["cpu_subset_s"] = round(time.perf_counter() - t2, 3)
                rec["cpu_subset_n"] = int(len(sub))
                rec["cpu_subset_peak_gib"] = s2.peak_gib
                rec["cpu_subset_clumps"] = int(len(c))
                print(f"  cugen[numpy, first {len(sub):,} variants]  "
                      f"{rec['cpu_subset_s']:8.2f} s  {len(c):,} clumps")
            except Exception as e:                        # noqa: BLE001
                rec["cpu_subset_error"] = f"{type(e).__name__}: {e}"
                print(f"  cugen[numpy subset] FAILED {type(e).__name__}: {e}")
        if err:
            rec["plink_error"] = err[:400]
            print(f"  plink2         FAILED: {err[:160]}")
        else:
            print(f"  plink2         {pk_s:8.2f} s")
            if "gpu_s" in rec:
                rec["speedup_vs_plink"] = round(pk_s / rec["gpu_s"], 1)
                rec["parity"] = compare(gpu_df, pk_path)
                print(f"  speedup {rec['speedup_vs_plink']}x   "
                      f"parity: {rec['parity'].get('identical')}")
        results.append(rec)
        json.dump(results, open(a.out, "w"), indent=2)

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
