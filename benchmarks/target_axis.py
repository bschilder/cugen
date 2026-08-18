"""cugen vs Beagle 5.5 across the TARGET COHORT axis, at a fixed panel.

The question this answers: cugen's cost per target haplotype falls steeply with
cohort size (52.6 ms at T=32 to 0.64 ms at T=8,192, measured) while Beagle's
per-target work is CPU-bound and closer to linear. So a crossover should exist.
"Should" is not a measurement, and this measures it.

THE CONFOUND THIS AVOIDS
------------------------
The obvious design -- vary the target/reference split of 1000 Genomes' 2,504
samples -- moves BOTH axes at once: more targets necessarily means fewer
reference samples, so K shrinks exactly as T grows. Any curve from that design
conflates the two effects and would be worthless for the question asked.

So the reference panel is FIXED at `--n-reference` samples, and targets are
drawn from a disjoint pool. K is constant across every point; only T moves.

The marker set is also computed ONCE, from the fixed reference panel, and reused
at every T. Recomputing "diallelic SNVs with MAC >= 1 in the reference" per
point would leave each T with a slightly different number of markers, which is a
second confound in a timing comparison.

Beagle's per-window "Imputation time" is parsed out of its log rather than
timing the java process, because the two are not the same thing: on this fixture
Beagle reported 11 s of imputation inside a 31 s process, the rest being JVM
startup and VCF I/O. cugen's comparable figure is the sum of its compute phases.
Both tools' wall clock is recorded too, and labelled as what it is.

    python benchmarks/target_axis.py --workdir /root/taxis --chrom 20
"""
import argparse
import json
import os
import re
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

BEAGLE_JAR_URL = ("https://faculty.washington.edu/browning/beagle/"
                  "beagle.27Feb25.75f.jar")


def beagle_imputation_seconds(log_path):
    """Beagle's own cumulative imputation time, in seconds.

    Its log prints per-window imputation times and then a cumulative block. The
    cumulative one is taken; summing the per-window lines would double count it.
    """
    txt = open(log_path).read()
    tail = txt[txt.rindex("Cumulative Statistics"):] if \
        "Cumulative Statistics" in txt else txt
    m = re.search(r"Imputation time:\s+(\d+)\s+(second|minute|hour)", tail)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    return n * {"second": 1, "minute": 60, "hour": 3600}[unit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", type=int, default=20)
    ap.add_argument("--workdir", default="/root/taxis")
    ap.add_argument("--n-reference", type=int, default=1504)
    ap.add_argument("--t-grid", default="50,200,500,1000")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--out", default="target_axis.json")
    a = ap.parse_args()
    W = _safe_workdir(a.workdir)
    os.makedirs(W, exist_ok=True)
    c = a.chrom
    grid = [int(x) for x in a.t_grid.split(",")]

    import pandas as pd
    from cugen.convert import vcf2cugenh
    from cugen.impute import impute
    from cugen.io import CugenReader

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
    ref_ids = sorted(samples[i] for i in perm[:a.n_reference])
    pool = [samples[i] for i in perm[a.n_reference:]]
    if max(grid) > len(pool):
        raise ValueError(f"target pool has {len(pool)} samples, need {max(grid)}")
    open(f"{W}/reference.txt", "w").write("\n".join(ref_ids) + "\n")
    print(f"panel FIXED at {len(ref_ids):,} samples "
          f"({2*len(ref_ids):,} haplotypes); target pool {len(pool):,}",
          flush=True)

    # --- reference and marker set, built ONCE ------------------------------
    if not os.path.exists(f"{W}/reference.cugen"):
        sh(f"bcftools view -S {W}/reference.txt --force-samples --no-update -Ou "
           f"{panel} | bcftools view -m2 -M2 -v snps -Ou "
           f"| bcftools view -c 1:minor -Oz -o {W}/reference.vcf.gz")
        sh(f"bcftools index -f -t {W}/reference.vcf.gz")
        sh(f"bcftools query -f '%CHROM\\t%POS\\n' {W}/reference.vcf.gz "
           f"> {W}/sites.txt")
        sh(f"bcftools query -f '%CHROM\\t%POS\\n' {omni} "
           f"| awk -v c={c} '$1==c || $1==\"chr\"c' | sort -k2,2n -u "
           f"> {W}/omni.pos")
        sh(f"awk 'NR==FNR{{x[$1\"_\"$2]=1;next}} ($1\"_\"$2) in x' "
           f"{W}/omni.pos {W}/sites.txt > {W}/target_sites.txt")
        vcf2cugenh(f"{W}/reference.vcf.gz", f"{W}/reference.cugen", verbose=True)
    n_ref_markers = int(subprocess.run(f"wc -l < {W}/sites.txt", shell=True,
                                       capture_output=True, text=True).stdout)
    n_tgt_markers = int(subprocess.run(f"wc -l < {W}/target_sites.txt",
                                       shell=True, capture_output=True,
                                       text=True).stdout)
    print(f"markers: {n_ref_markers:,} reference, {n_tgt_markers:,} genotyped "
          f"(both FIXED across every T)", flush=True)

    ref_pos = np.asarray(subprocess.run(
        ["bcftools", "query", "-f", "%POS\n", f"{W}/reference.vcf.gz"],
        capture_output=True, text=True, check=True).stdout.split(),
        dtype=np.int64)
    ann = pd.DataFrame({"gidx": np.arange(ref_pos.size), "POS": ref_pos,
                        "CHR": str(c), "ID": [f"{c}:{p}" for p in ref_pos]})

    jar = f"{W}/beagle.jar"
    if not os.path.exists(jar):
        subprocess.run(["curl", "-fsSL", "-o", jar, BEAGLE_JAR_URL], check=True)

    results = {"chrom": c, "n_reference_samples": len(ref_ids),
               "n_reference_haplotypes": 2 * len(ref_ids),
               "n_reference_markers": n_ref_markers,
               "n_target_markers": n_tgt_markers, "points": []}

    warm = False
    for n_t in grid:
        tag = f"t{n_t}"
        tids = sorted(pool[:n_t])
        open(f"{W}/{tag}.txt", "w").write("\n".join(tids) + "\n")
        print(f"\n=== T = {n_t} samples ({2*n_t} haplotypes) ===", flush=True)

        if not os.path.exists(f"{W}/{tag}.cugen"):
            # -T matches on POSITION, and a multi-allelic site contributes
            # several rows at one position in the full panel, so the same
            # biallelic-SNV filter has to be reapplied. Without it the target
            # carries duplicate positions, searchsorted returns a non-increasing
            # gidx, and impute() refuses the pair.
            sh(f"bcftools view -S {W}/{tag}.txt --force-samples --no-update -Ou "
               f"{panel} | bcftools view -m2 -M2 -v snps -Ou "
               f"| bcftools view -T {W}/target_sites.txt -Oz "
               f"-o {W}/{tag}.vcf.gz")
            sh(f"bcftools index -f -t {W}/{tag}.vcf.gz")
            tpos = np.asarray(subprocess.run(
                ["bcftools", "query", "-f", "%POS\n", f"{W}/{tag}.vcf.gz"],
                capture_output=True, text=True, check=True).stdout.split(),
                dtype=np.int64)
            gidx = np.searchsorted(ref_pos, tpos)
            assert np.array_equal(ref_pos[gidx], tpos), "target not a subset"
            assert np.all(np.diff(gidx) > 0), (
                "target positions are not strictly increasing -- duplicate "
                "positions from multi-allelic sites survive an equality check "
                "because both sides are duplicated")
            vcf2cugenh(f"{W}/{tag}.vcf.gz", f"{W}/{tag}.cugen", gidx=gidx,
                       verbose=False)

        if not warm:
            # NVRTC and the CUDA context, absorbed before any timing. A cold
            # first point would otherwise carry seconds of compilation and make
            # the smallest cohort look slow -- the exact artifact that inverted
            # a published conclusion in the previous round of this project.
            print("  warm-up (not counted)", flush=True)
            impute(f"{W}/{tag}.cugen", ref=f"{W}/reference.cugen",
                   annotation=ann, map=gmap, chrom=c, verbose=False)
            warm = True

        rec = {"n_target_samples": n_t, "n_target_haplotypes": 2 * n_t}
        timers = {}
        t0 = time.perf_counter()
        impute(f"{W}/{tag}.cugen", ref=f"{W}/reference.cugen", annotation=ann,
               map=gmap, chrom=c, out=f"{W}/{tag}_out.cugen", verbose=False,
               _timers_out=timers)
        rec["cugen_wall_s"] = round(time.perf_counter() - t0, 2)
        comp = sum(timers.get(k, 0.0) for k in
                   ("forward_backward", "carriers", "aggregate", "dose"))
        rec["cugen_impute_s"] = round(comp, 2)
        rec["cugen_phases"] = {k: round(v, 3) for k, v in timers.items()
                               if isinstance(v, float)}

        t0 = time.perf_counter()
        subprocess.run(["java", "-Xmx60g", "-jar", jar,
                        f"gt={W}/{tag}.vcf.gz", f"ref={W}/reference.vcf.gz",
                        f"map={gmap}", f"out={W}/{tag}_beagle",
                        "impute=true", "nthreads=8"], check=True,
                       capture_output=True)
        rec["beagle_wall_s"] = round(time.perf_counter() - t0, 2)
        rec["beagle_impute_s"] = beagle_imputation_seconds(f"{W}/{tag}_beagle.log")
        if rec["beagle_impute_s"]:
            rec["impute_ratio_beagle_over_cugen"] = round(
                rec["beagle_impute_s"] / max(rec["cugen_impute_s"], 1e-9), 2)
        print(f"  cugen impute {rec['cugen_impute_s']:7.2f}s "
              f"(wall {rec['cugen_wall_s']:.1f}s)   "
              f"beagle impute {rec['beagle_impute_s']}s "
              f"(wall {rec['beagle_wall_s']:.1f}s)   "
              f"ratio {rec.get('impute_ratio_beagle_over_cugen')}", flush=True)
        results["points"].append(rec)
        json.dump(results, open(a.out, "w"), indent=2)

    print(f"\nwrote {a.out}")
    print(f"\n{'T haps':>8} {'cugen impute':>13} {'beagle impute':>14} {'ratio':>7}")
    for r in results["points"]:
        print(f"{r['n_target_haplotypes']:8,d} {r['cugen_impute_s']:13.2f} "
              f"{str(r['beagle_impute_s']):>14} "
              f"{r.get('impute_ratio_beagle_over_cugen', '-'):>7}")


if __name__ == "__main__":
    main()
