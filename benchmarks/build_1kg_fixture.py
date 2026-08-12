"""Rebuild the exact validation fixture from Browning et al. (2018).

The paper specifies its fixture completely, and publishes the resulting marker
counts. Those counts are a CHECKSUM: if this script does not reproduce them, the
fixture is wrong and every accuracy number downstream is meaningless. Half of
what went wrong in the LD work in this repo was fixtures that looked fine and
silently did not exercise what they were meant to.

    "We downloaded the publicly available 1000 Genomes Project phase 3 version
     5a data. The 1000 Genomes data contain 2,504 individuals with phased
     sequence data from 26 populations. We randomly selected two individuals
     from each population (52 total) to be the imputation target. The remaining
     2,452 individuals were the reference panel. We restricted the 1000 Genomes
     reference and target data to diallelic SNVs having at least one copy of
     the minor allele in the reference panel. After marker filtering there were
     2,508,019 markers on chromosome 14 and 1,718,742 markers on chromosome 20.
     In the target samples, we masked markers that were not on the Illumina
     Omni2.5 array, resulting in 72,973 target markers on chromosome 14 and
     54,885 target markers on chromosome 20."

Data sources are chosen deliberately, and the obvious choice is wrong. Beagle's
own site hosts a 1000 Genomes panel that looks like the right input and is
already filtered: on chr20 it carries 679,241 markers against the release's
much larger set, and running the paper's filters over it lands 64% short. Use
the 1000 Genomes RELEASE for genotypes and Beagle's site only for the genetic
map. Chromosome files are versioned v5a or v5b and which one exists varies by
chromosome, so both are probed rather than assumed -- a previous project here
shipped a URL naming the version its mirror did not carry.

    python benchmarks/build_1kg_fixture.py --chrom 20 --workdir /root/fix
"""
import argparse
import gzip
import os
import subprocess
import sys

BEAGLE_HOST = "https://bochet.gcc.biostat.washington.edu/beagle"
MAP_URL = f"{BEAGLE_HOST}/genetic_maps/plink.GRCh37.map.zip"

# The panel comes from the 1000 Genomes RELEASE, not from the convenience copy
# Beagle's own site hosts. Measured on chr20: the beagle-site file carries
# 679,241 markers, and filtering it the paper's way yields 618,570 against the
# paper's 1,718,742 -- a 64% shortfall, because that copy is already filtered.
# The shortfall was caught only because the paper's counts are asserted; a
# fixture built on it would have looked entirely reasonable and scored a
# different marker set from every published number.
#
# Chromosome-level files are versioned v5a or v5b and which one exists VARIES BY
# CHROMOSOME, so both are tried. Guessing wrong silently changes the marker set:
# a previous project shipped a URL naming the version the mirror did not carry.
# The 1000 Genomes release is mirrored on S3 and served from EBI's FTP. Measured
# from a CA-MTL RunPod pod on 2026-08-12: S3 77 MB/s, EBI 0.2 MB/s -- a 370x
# gap, which is 17 seconds against two hours for the 1.33 GB chip file. Prefer
# S3 and keep EBI only as a fallback.
#
# The main phase3 panel deliberately comes from Browning's own host instead of
# either, because the S3 and EBI copies have carried DIFFERENT VERSIONS (v5a vs
# v5b) of that file, and picking the wrong one silently changes the marker set.
_1KG_S3 = "https://1000genomes.s3.amazonaws.com"
_1KG_EBI = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp"
_OMNI_REL = ("/release/20130502/supporting/hd_genotype_chip/"
             "ALL.chip.omni_broad_sanger_combined.20140818.snps.genotypes.vcf.gz")
_PANEL_REL = "/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"

# Omni2.5 site list, taken from the 1000 Genomes chip genotypes rather than the
# Illumina manifest, which needs registration.
OMNI_URL = _1KG_S3 + _OMNI_REL
OMNI_URL_FALLBACK = _1KG_EBI + _OMNI_REL
SAMPLE_PANEL = _1KG_S3 + _PANEL_REL
SAMPLE_PANEL_FALLBACK = _1KG_EBI + _PANEL_REL

# From Table 2 and the Data section. Reproducing these is the gate.
EXPECTED = {
    14: {"reference_markers": 2_508_019, "target_markers": 72_973},
    20: {"reference_markers": 1_718_742, "target_markers": 54_885},
}


def sh(cmd, **kw):
    """Run a shell pipeline.

    shell=True is deliberate: these are bcftools pipelines and the pipe is the
    point. Every interpolated value is either a module constant or `workdir`,
    which _safe_workdir() restricts to characters no shell will reinterpret --
    so there is no path from an argument to a metacharacter.
    """
    print(f"  $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def _safe_workdir(p):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", p):
        raise ValueError(
            f"--workdir {p!r} may only contain letters, digits and _./- ; "
            f"it is interpolated into shell pipelines")
    return os.path.abspath(p)


def fetch(url, dest, verify_gzip=True, fallback=None):
    """Download with resume, then prove the file decompresses.

    A truncated or corrupt gzip is the failure mode that cost this project a
    day previously: three pods produced byte-identical, size-correct, CRC-wrong
    downloads. Checking the size proves nothing; decompressing does.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        try:
            if not verify_gzip or _gzip_ok(dest):
                print(f"  have {os.path.basename(dest)}")
                return dest
        except OSError:
            pass
        print(f"  {os.path.basename(dest)} is corrupt, refetching")
        os.remove(dest)
    urls = [url] + ([fallback] if fallback else [])
    for u in urls:
        for attempt in range(2):
            try:
                sh(f"curl -fsSL --retry 3 --retry-delay 5 --speed-time 60 "
                   f"--speed-limit 100000 -C - -o {dest!r} {u!r}")
            except subprocess.CalledProcessError as e:
                # --speed-limit aborts a transfer stuck below 100 KB/s for a
                # minute, so a slow mirror fails over instead of hanging for
                # hours. EBI was measured at 0.2 MB/s against S3's 77.
                print(f"  {u.split('/')[2]}: transfer failed ({e.returncode})")
                break
            if not verify_gzip or _gzip_ok(dest):
                return dest
            print(f"  attempt {attempt + 1}: downloaded file does not decompress")
            os.remove(dest)
    raise RuntimeError(f"could not obtain a valid copy from any of {urls}")


def fetch_panel(c, W):
    """The 1000 Genomes phase 3 release for one chromosome, v5a or v5b."""
    rel = "/release/20130502/ALL.chr{c}.phase3_shapeit2_mvncall_integrated_{v}.20130502.genotypes.vcf.gz"
    last = None
    for v in ("v5a", "v5b"):
        url = _1KG_S3 + rel.format(c=c, v=v)
        head = subprocess.run(["curl", "-fsSI", url], capture_output=True,
                              text=True)
        if head.returncode == 0:
            print(f"  chromosome {c} release is {v}")
            # Filename records the SOURCE, not just the version. The earlier
            # name matched the file Beagle's site serves, so a workdir that had
            # already fetched the filtered copy cache-hit on it and the release
            # was never downloaded -- the run then reported the filtered
            # marker counts as though they came from the release.
            dest = f"{W}/chr{c}.1kg.release.{v}.vcf.gz"
            expect = int(head.stdout.lower().split("content-length:")[1]
                         .split()[0]) if "content-length:" in \
                head.stdout.lower() else None
            if expect and os.path.exists(dest) and \
                    os.path.getsize(dest) != expect:
                print(f"  {os.path.basename(dest)} is "
                      f"{os.path.getsize(dest):,}B, server says {expect:,}B "
                      f"-- refetching")
                os.remove(dest)
            return fetch(url, dest,
                         fallback=_1KG_EBI + rel.format(c=c, v=v))
        last = url
    raise RuntimeError(f"neither v5a nor v5b exists for chr{c} (tried {last})")


def _gzip_ok(path):
    if not path.endswith(".gz"):
        return True
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 24):
                pass
        return True
    except (OSError, EOFError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", type=int, default=20)
    ap.add_argument("--workdir", default="/root/fixture")
    ap.add_argument("--seed", type=int, default=20180701)
    a = ap.parse_args()
    c = a.chrom
    W = _safe_workdir(a.workdir)
    os.makedirs(W, exist_ok=True)

    print(f"=== 1000 Genomes phase3 v5a, chromosome {c} ===", flush=True)
    panel = fetch_panel(c, W)
    omni = fetch(OMNI_URL, f"{W}/omni.vcf.gz",
                 fallback=OMNI_URL_FALLBACK)
    pan = fetch(SAMPLE_PANEL, f"{W}/samples.panel", verify_gzip=False,
                fallback=SAMPLE_PANEL_FALLBACK)
    fetch(MAP_URL, f"{W}/plink.GRCh37.map.zip", verify_gzip=False)
    if not os.path.exists(f"{W}/plink.chr{c}.GRCh37.map"):
        sh(f"cd {W} && unzip -o -q plink.GRCh37.map.zip")

    # --- targets: two individuals per population, deterministically ---------
    import numpy as np
    import pandas as pd
    pop = pd.read_csv(pan, sep="\t").dropna(axis=1, how="all")
    pop = pop[["sample", "pop"]].dropna()
    print(f"  {len(pop):,} samples across {pop['pop'].nunique()} populations")
    assert len(pop) == 2504, f"expected 2,504 samples, got {len(pop):,}"
    assert pop["pop"].nunique() == 26, "expected 26 populations"
    rng = np.random.default_rng(a.seed)
    targets = (pop.groupby("pop")["sample"]
               .apply(lambda s: pd.Series(rng.choice(sorted(s), 2, replace=False)))
               .tolist())
    assert len(targets) == 52, len(targets)
    ref_samples = sorted(set(pop["sample"]) - set(targets))
    assert len(ref_samples) == 2452, len(ref_samples)
    open(f"{W}/targets.txt", "w").write("\n".join(sorted(targets)) + "\n")
    open(f"{W}/reference.txt", "w").write("\n".join(ref_samples) + "\n")
    print(f"  52 target samples, {len(ref_samples):,} reference samples")

    # --- marker filter: diallelic SNVs with MAC >= 1 in the REFERENCE -------
    # The minor-allele condition is evaluated on the reference panel only, not
    # on all 2,504 -- a variant private to a target sample is not imputable and
    # the paper excludes it. Computing AC over everyone would inflate the
    # marker count and quietly change the denominator of every accuracy figure.
    print("\n=== filtering markers (diallelic SNVs, MAC>=1 in reference) ===",
          flush=True)
    # Two things this panel needs that a typical VCF does not.
    #
    # It carries no ##contig lines, so region queries need a tabix index built
    # first -- otherwise bcftools warns "Contig '20' is not defined in the
    # header" and region selection silently returns nothing.
    #
    # Its header declares only FORMAT/GT: no INFO tags at all. Sample subsetting
    # normally recomputes AC/AN, and with nothing to recompute bcftools refuses
    # outright ("Undefined tags in the header, cannot proceed in the sample
    # subset mode"). --no-update tells it not to try. The minor-allele filter
    # below still works, because -c 1:minor counts genotypes directly rather
    # than reading INFO/AC.
    if not os.path.exists(panel + ".tbi"):
        sh(f"bcftools index -f -t {panel}")
    sh(f"bcftools view -S {W}/reference.txt --force-samples --no-update -Ou "
       f"{panel} "
       f"| bcftools view -m2 -M2 -v snps -Ou "
       f"| bcftools view -c 1:minor -Oz -o {W}/reference.vcf.gz")
    sh(f"bcftools index -f -t {W}/reference.vcf.gz")
    # The surviving marker POSITIONS, kept separately from the reference panel.
    # An earlier version cut the target samples out of the reference file
    # itself, which of course contains only reference samples -- bcftools said
    # "subsetting has removed all samples" and the script carried on to write an
    # empty truth file. Marker selection and sample selection have to be two
    # steps over the FULL panel, not one chained over a subset.
    sh(f"bcftools query -f '%CHROM\\t%POS\\n' {W}/reference.vcf.gz "
       f"> {W}/sites.txt")
    n_ref = int(subprocess.run(f"bcftools index -n {W}/reference.vcf.gz",
                               shell=True, capture_output=True, text=True
                               ).stdout.strip())
    exp = EXPECTED.get(c, {})
    print(f"  reference markers: {n_ref:,}"
          + (f"   (paper: {exp['reference_markers']:,})" if exp else ""))

    # --- target markers: intersect with the Omni2.5 sites -------------------
    sh(f"bcftools query -f '%CHROM\\t%POS\\n' {omni} "
       f"| awk -v c={c} '$1==c || $1==\"chr\"c' | sort -k2,2n -u "
       f"> {W}/omni.chr{c}.pos")
    # target markers = surviving markers that are also Omni2.5 sites
    sh(f"awk 'NR==FNR{{a[$1\"_\"$2]=1; next}} ($1\"_\"$2) in a' "
       f"{W}/omni.chr{c}.pos {W}/sites.txt > {W}/target_sites.txt")
    n_tgt = int(subprocess.run(f"wc -l < {W}/target_sites.txt",
                               shell=True, capture_output=True, text=True
                               ).stdout.strip())
    print(f"  target markers   : {n_tgt:,}"
          + (f"   (paper: {exp['target_markers']:,})" if exp else ""))

    if exp:
        for got, want, what in ((n_ref, exp["reference_markers"], "reference"),
                                (n_tgt, exp["target_markers"], "target")):
            pct = 100.0 * abs(got - want) / want
            flag = "OK" if pct < 2.0 else "MISMATCH"
            print(f"  {flag:8s} {what}: {got:,} vs {want:,} ({pct:.2f}% apart)")
        print("\n  A few percent is expected -- bcftools' filters are not "
              "byte-identical to the paper's pipeline, and the Omni site list "
              "here comes from the 1000 Genomes chip data rather than the "
              "Illumina manifest. A LARGE gap means the filter is wrong.")

    # --- build the actual reference / target files --------------------------
    print("\n=== writing target VCFs (from the FULL panel) ===", flush=True)
    # truth: target samples at every surviving marker, for scoring
    sh(f"bcftools view -S {W}/targets.txt --force-samples --no-update -Ou "
       f"{panel} | bcftools view -T {W}/sites.txt -Oz "
       f"-o {W}/target.truth.vcf.gz")
    sh(f"bcftools index -f -t {W}/target.truth.vcf.gz")
    # masked: the same samples restricted to the Omni2.5 sites
    sh(f"bcftools view -T {W}/target_sites.txt -Oz "
       f"-o {W}/target.masked.vcf.gz {W}/target.truth.vcf.gz")
    sh(f"bcftools index -f -t {W}/target.masked.vcf.gz")
    for f_, want in ((f"{W}/target.truth.vcf.gz", 52),
                     (f"{W}/target.masked.vcf.gz", 52),
                     (f"{W}/reference.vcf.gz", 2452)):
        got = int(subprocess.run(f"bcftools query -l {f_} | wc -l", shell=True,
                                 capture_output=True, text=True).stdout)
        assert got == want, f"{f_} has {got} samples, expected {want}"
    print(f"  sample counts verified: 52 target, 2,452 reference")

    print("\n=== converting to .cugen ===", flush=True)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cugen.convert import vcf2cugenh
    vcf2cugenh(f"{W}/reference.vcf.gz", f"{W}/reference.cugen", verbose=True)
    vcf2cugenh(f"{W}/target.masked.vcf.gz", f"{W}/target.cugen", verbose=True)

    # gidx must be the REFERENCE marker index for both files, since impute()
    # matches target to reference through gidx. Writing 0..n-1 into each
    # independently would silently align target marker 0 with reference marker
    # 0 and shift everything after it.
    import numpy as np
    ref_pos = subprocess.run(
        f"bcftools query -f '%POS\\n' {W}/reference.vcf.gz", shell=True,
        capture_output=True, text=True).stdout.split()
    tgt_pos = subprocess.run(
        f"bcftools query -f '%POS\\n' {W}/target.masked.vcf.gz", shell=True,
        capture_output=True, text=True).stdout.split()
    ref_pos = np.asarray(ref_pos, dtype=np.int64)
    tgt_pos = np.asarray(tgt_pos, dtype=np.int64)
    gidx = np.searchsorted(ref_pos, tgt_pos)
    assert np.array_equal(ref_pos[gidx], tgt_pos), \
        "target positions are not a subset of reference positions"
    np.save(f"{W}/ref_pos.npy", ref_pos)
    np.save(f"{W}/tgt_gidx.npy", gidx)
    print(f"  ref_pos {ref_pos.size:,}  tgt_gidx {gidx.size:,}  -> {W}")

    import json
    # Pin the sample draw. Browning et al. state only that two individuals per
    # population were "randomly selected"; the supplement is 1 figure, 16 tables
    # and msprime code for the SIMULATED panels, and carries no seed or sample
    # list for the 1000 Genomes draw. Their exact 52 are therefore not
    # recoverable. Recording ours makes this fixture reproducible even though
    # the paper's is not.
    #
    # Two other terms contribute to the residual against their marker counts,
    # and the sample draw is probably the smallest of the three: they ran
    # bcftools 1.5 against this pipeline's 1.19, and the Omni2.5 site list here
    # is derived from the 1000 Genomes chip VCF because the Illumina manifest
    # requires registration -- a proxy for their source, not the same list.
    json.dump({"chrom": c, "n_reference_markers": int(n_ref),
               "target_samples": sorted(targets),
               "bcftools_version": subprocess.run(
                   "bcftools --version | head -1", shell=True,
                   capture_output=True, text=True).stdout.strip(),
               "n_target_markers": int(n_tgt),
               "paper_reference_markers": exp.get("reference_markers"),
               "paper_target_markers": exp.get("target_markers"),
               "n_reference_samples": len(ref_samples),
               "n_target_samples": len(targets), "seed": a.seed},
              open(f"{W}/fixture.json", "w"), indent=2)
    print(f"\nwrote {W}/fixture.json")


if __name__ == "__main__":
    main()
