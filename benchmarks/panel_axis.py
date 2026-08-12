"""cugen vs Beagle 5.5 across the REFERENCE PANEL axis, at a fixed cohort.

The companion to target_axis.py, and the axis where brute-force states are
expected to lose: cugen uses every reference haplotype as an HMM state, so its
cost is linear in panel size, while Beagle selects `imp-states=1600` composite
reference haplotypes regardless of how large the panel is.

WHY THIS IS MEASURED RATHER THAN CITED
--------------------------------------
Browning et al. (2018) report Beagle scaling sublinearly in panel size -- 1,000x
the reference samples for 11x the time. That is **Beagle 5.0**, published in
2018. This project benchmarks **Beagle 5.5 (27Feb25.75f)**, which is seven years
and several releases later, including the 2021 two-stage phasing rewrite. Using
5.0's published curve to predict 5.5's behaviour is a secondhand claim about a
different program, and this file exists so no such claim is needed.

THE CONFOUNDS THIS AVOIDS
-------------------------
K and the marker set move together if you let them: "diallelic SNVs with at
least one minor-allele copy in the reference" admits more markers as the panel
grows, so a larger panel would also mean more markers and the curve would
conflate two effects. The marker set is therefore computed ONCE from the LARGEST
panel and reused at every K.

The target cohort is fixed and drawn from a pool disjoint from every reference
panel, so no sample is ever in both, and the panels are nested (each smaller one
is a prefix of the largest) so the comparison is not confounded by which
individuals were drawn.

Range note: 1000 Genomes caps this at about 12x, from ~200 to ~2,400 reference
samples. That does not reach HRC or TOPMed, and a curve fitted here should not
be extrapolated to them -- it establishes the SHAPE for each tool, measured, on
the version actually in use.

    python benchmarks/panel_axis.py --workdir /root/paxis --chrom 20
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from build_1kg_fixture import (MAP_URL, OMNI_URL, OMNI_URL_FALLBACK,  # noqa: E402
                               SAMPLE_PANEL, SAMPLE_PANEL_FALLBACK, _safe_workdir,
                               fetch, fetch_panel, sh)
from target_axis import BEAGLE_JAR_URL, beagle_imputation_seconds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", type=int, default=20)
    ap.add_argument("--workdir", default="/root/paxis")
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--k-grid", default="200,600,1200,2400")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--out", default="panel_axis.json")
    a = ap.parse_args()
    W = _safe_workdir(a.workdir)
    os.makedirs(W, exist_ok=True)
    c = a.chrom
    grid = sorted(int(x) for x in a.k_grid.split(","))

    import pandas as pd
    from cugen.convert import vcf2cugenh
    from cugen.impute import impute

    panel = fetch_panel(c, W)
    omni = fetch(OMNI_URL, f"{W}/omni.vcf.gz", fallback=OMNI_URL_FALLBACK)
    pan = fetch(SAMPLE_PANEL, f"{W}/samples.panel", verify_gzip=False,
                fallback=SAMPLE_PANEL_FALLBACK)
    fetch(MAP_URL, f"{W}/plink.GRCh37.map.zip", verify_gzip=False)
    if not os.path.exists(f"{W}/plink.chr{c}.GRCh37.map"):
        sh(f"cd {W} && unzip -o -q plink.GRCh37.map.zip")
    gmap = f"{W}/plink.chr{c}.GRCh37.map"
    if not os.path.exists(panel + ".tbi"):
        sh(f"bcftools index -f -t {panel}")

    samples = sorted(pd.read_csv(pan, sep="\t")["sample"].dropna().tolist())
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(samples))
    tids = sorted(samples[i] for i in perm[:a.n_targets])
    pool = [samples[i] for i in perm[a.n_targets:]]
    if max(grid) > len(pool):
        raise ValueError(f"reference pool has {len(pool)}, need {max(grid)}")
    open(f"{W}/targets.txt", "w").write("\n".join(tids) + "\n")
    # Nested panels: each smaller one is a prefix of the largest, so moving
    # along the axis changes only HOW MANY reference samples there are, never
    # WHICH ones.
    for k in grid:
        open(f"{W}/ref{k}.txt", "w").write(
            "\n".join(sorted(pool[:k])) + "\n")
    print(f"targets FIXED at {len(tids)} samples ({2*len(tids)} haplotypes)",
          flush=True)

    # Marker set from the LARGEST panel, reused everywhere.
    big = max(grid)
    if not os.path.exists(f"{W}/sites.txt"):
        sh(f"bcftools view -S {W}/ref{big}.txt --force-samples --no-update -Ou "
           f"{panel} | bcftools view -m2 -M2 -v snps -Ou "
           f"| bcftools view -c 1:minor -Oz -o {W}/sites.vcf.gz")
        sh(f"bcftools index -f -t {W}/sites.vcf.gz")
        sh(f"bcftools query -f '%CHROM\\t%POS\\n' {W}/sites.vcf.gz "
           f"> {W}/sites.txt")
        sh(f"bcftools query -f '%CHROM\\t%POS\\n' {omni} "
           f"| awk -v c={c} '$1==c || $1==\"chr\"c' | sort -k2,2n -u "
           f"> {W}/omni.pos")
        sh(f"awk 'NR==FNR{{x[$1\"_\"$2]=1;next}} ($1\"_\"$2) in x' "
           f"{W}/omni.pos {W}/sites.txt > {W}/target_sites.txt")
    n_ref_markers = int(subprocess.run(f"wc -l < {W}/sites.txt", shell=True,
                                       capture_output=True, text=True).stdout)
    print(f"markers FIXED at {n_ref_markers:,} (from the {big}-sample panel)",
          flush=True)

    ref_pos = np.asarray(subprocess.run(
        ["bcftools", "query", "-f", "%POS\n", f"{W}/sites.vcf.gz"],
        capture_output=True, text=True, check=True).stdout.split(),
        dtype=np.int64)
    ann = pd.DataFrame({"gidx": np.arange(ref_pos.size), "POS": ref_pos,
                        "CHR": str(c), "ID": [f"{c}:{p}" for p in ref_pos]})

    if not os.path.exists(f"{W}/target.cugen"):
        sh(f"bcftools view -S {W}/targets.txt --force-samples --no-update -Ou "
           f"{panel} | bcftools view -T {W}/target_sites.txt -Oz "
           f"-o {W}/target.vcf.gz")
        sh(f"bcftools index -f -t {W}/target.vcf.gz")
        tpos = np.asarray(subprocess.run(
            ["bcftools", "query", "-f", "%POS\n", f"{W}/target.vcf.gz"],
            capture_output=True, text=True, check=True).stdout.split(),
            dtype=np.int64)
        gidx = np.searchsorted(ref_pos, tpos)
        assert np.array_equal(ref_pos[gidx], tpos), "target not a subset"
        vcf2cugenh(f"{W}/target.vcf.gz", f"{W}/target.cugen", gidx=gidx,
                   verbose=False)

    jar = f"{W}/beagle.jar"
    if not os.path.exists(jar):
        subprocess.run(["curl", "-fsSL", "-o", jar, BEAGLE_JAR_URL], check=True)

    results = {"chrom": c, "n_target_samples": len(tids),
               "n_target_haplotypes": 2 * len(tids),
               "n_reference_markers": n_ref_markers, "points": []}
    warm = False
    for k in grid:
        print(f"\n=== K = {k} reference samples ({2*k} haplotypes) ===",
              flush=True)
        if not os.path.exists(f"{W}/ref{k}.cugen"):
            sh(f"bcftools view -S {W}/ref{k}.txt --force-samples --no-update "
               f"-Oz -o {W}/ref{k}.vcf.gz {W}/sites.vcf.gz")
            sh(f"bcftools index -f -t {W}/ref{k}.vcf.gz")
            vcf2cugenh(f"{W}/ref{k}.vcf.gz", f"{W}/ref{k}.cugen", verbose=False)

        if not warm:
            impute(f"{W}/target.cugen", ref=f"{W}/ref{k}.cugen",
                   annotation=ann, map=gmap, chrom=c, verbose=False)
            warm = True
            print("  warm-up done (not counted)", flush=True)

        rec = {"n_reference_samples": k, "n_reference_haplotypes": 2 * k}
        timers = {}
        t0 = time.perf_counter()
        impute(f"{W}/target.cugen", ref=f"{W}/ref{k}.cugen", annotation=ann,
               map=gmap, chrom=c, out=f"{W}/out{k}.cugen", verbose=False,
               _timers_out=timers)
        rec["cugen_wall_s"] = round(time.perf_counter() - t0, 2)
        rec["cugen_impute_s"] = round(sum(
            timers.get(x, 0.0) for x in
            ("forward_backward", "carriers", "aggregate", "dose")), 2)
        rec["cugen_phases"] = {x: round(v, 3) for x, v in timers.items()
                               if isinstance(v, float)}

        t0 = time.perf_counter()
        subprocess.run(["java", "-Xmx60g", "-jar", jar,
                        f"gt={W}/target.vcf.gz", f"ref={W}/ref{k}.vcf.gz",
                        f"map={gmap}", f"out={W}/b{k}", "impute=true",
                        "nthreads=8"], check=True, capture_output=True)
        rec["beagle_wall_s"] = round(time.perf_counter() - t0, 2)
        rec["beagle_impute_s"] = beagle_imputation_seconds(f"{W}/b{k}.log")
        if rec["beagle_impute_s"]:
            rec["impute_ratio_beagle_over_cugen"] = round(
                rec["beagle_impute_s"] / max(rec["cugen_impute_s"], 1e-9), 2)
        print(f"  cugen impute {rec['cugen_impute_s']:7.2f}s   "
              f"beagle impute {rec['beagle_impute_s']}s   "
              f"ratio {rec.get('impute_ratio_beagle_over_cugen')}", flush=True)
        results["points"].append(rec)
        json.dump(results, open(a.out, "w"), indent=2)

    print(f"\nwrote {a.out}")
    print(f"\n{'K haps':>8} {'cugen impute':>13} {'beagle impute':>14} {'ratio':>7}")
    for r in results["points"]:
        print(f"{r['n_reference_haplotypes']:8,d} {r['cugen_impute_s']:13.2f} "
              f"{str(r['beagle_impute_s']):>14} "
              f"{r.get('impute_ratio_beagle_over_cugen', '-'):>7}")


if __name__ == "__main__":
    main()
