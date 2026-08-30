"""Rohlfs, Swanson & Weir (2010) on 1000 Genomes 30x, per superpopulation.

Their design in one line: a candidate gene pair on different chromosomes should
carry MORE allelic association than a background of random unlinked pairs drawn
from the same individuals. Population structure inflates candidate and
background alike, so it cancels -- which is why the test works without ever
removing structure, and why the background must come from the same samples.

Three deliberate departures from the 2010 paper, each forced by what has changed
since:

1. **Per superpopulation, never pooled.** They used one homogeneous cohort (1958
   British Birth Cohort, n=1,480). Pooling 1000 Genomes would reintroduce the
   two-locus Wahlund term their background cannot absorb, because a candidate
   pair with strong continental frequency divergence would outrun a background
   that is mostly not divergent. Each superpopulation is analysed separately.

2. **The background is artifact-masked; theirs was not.** They checked their two
   candidate genes for probe cross-hybridisation by blastn, which was the right
   control for their candidates. But mapping artifact does not cancel the way
   structure does -- it is specific to particular region pairs -- so an
   unmasked background is inflated by a term the candidate lacks, making the
   test conservative by an unknown amount. Background windows overlapping the
   ENCODE blacklist, segmental duplications, satellite or centromeric regions
   are dropped.

3. **ZP3R has been reclassified as a pseudogene.** Their primary candidate pair
   was ZP3 x ZP3R. The symbol ZP3R is retired; the locus is now HGNC:1326
   **ZP3RP**, "zona pellucida 3 receptor, pseudogene", locus_type `pseudogene`,
   at 1q32.2 -- and C4BPAP1 is one of its PREVIOUS SYMBOLS, so the coordinates
   used here are the same locus they studied, not a substitute for it. A
   pseudogene makes no protein, so the protein-protein coevolution premise does
   not survive re-annotation. We run it anyway, labelled, and run their
   secondary pair GHR x GH2 -- both still protein-coding -- as the
   interpretable replication.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cugen import coevo                       # noqa: E402
from cugen.io import read_cugen               # noqa: E402
from cugen.ld import contingency_tables, ld_from_counts   # noqa: E402
from cugen.popstruct import _unpack_tile      # noqa: E402

REPO = "standardmodelbio/cugen"
TOK = os.environ["HF_TOKEN"]
POPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
FLANK = 100_000          # Rohlfs: "no further than 100 kb up- and downstream"
MAF_MIN = 0.05           # Rohlfs excluded MAF < 0.05 (genotyping-error sensitivity)
MAX_SNPS = 8             # tag-SNP cap; they used 7 and 9
N_BACKGROUND = 60
N_NULL = 100          # random gene pairs used to calibrate the KS statistic
N_PERM = 1000
SEED = 20260830

# GRCh38, from Ensembl. ZP3R is retired: C4BPAP1 is the successor locus.
GENES = {
    "ZP3":     ("chr7", 76_397_497, 76_442_071),
    "C4BPAP1": ("chr1", 207_165_496, 207_183_910),   # was ZP3R; now a pseudogene
    "GHR":     ("chr5", 42_423_439, 42_721_878),
    "GH2":     ("chr17", 63_880_205, 63_881_970),
}
PAIRS = [("GHR", "GH2"), ("ZP3", "C4BPAP1")]

SCR = ("/private/tmp/claude-501/-Users-bschilder-code-cugen/"
       "81b0ecb9-0cdc-49a7-9e62-34fce01a7a7c/scratchpad/.regions_cache")
ARTIFACT = [("hg38-blacklist.v2.bed.gz", (0, 1, 2)),
            ("GRCh38_satellites_slop5.bed.gz", (0, 1, 2)),
            ("genomicSuperDups.txt.gz", (1, 2, 3)),
            ("centromeres.txt.gz", (1, 2, 3))]


def load_artifact():
    iv = defaultdict(list)
    for fn, c in ARTIFACT:
        with gzip.open(f"{SCR}/{fn}", "rt") as fh:
            for ln in fh:
                if ln.startswith(("#", "track", "browser")):
                    continue
                f = ln.rstrip("\n").split("\t")
                if len(f) <= max(c):
                    continue
                ch = f[c[0]]
                ch = ch if ch.startswith("chr") else "chr" + ch
                try:
                    iv[ch].append((int(f[c[1]]), int(f[c[2]])))
                except ValueError:
                    pass
    return {k: sorted(v) for k, v in iv.items()}


def overlaps(art, chrom, lo, hi):
    for a, b in art.get(chrom, ()):
        if a < hi and lo < b:
            return True
    return False


def paths(pop, chrom):
    base = f"1kg-30x-grch38/perchrom/superpop/{pop}/{chrom}"
    return (hf_hub_download(REPO, base + ".cugen", repo_type="dataset", token=TOK),
            hf_hub_download(REPO, base + ".bim", repo_type="dataset", token=TOK))


_BIM = {}


def bim_pos(path):
    if path not in _BIM:
        _BIM[path] = np.array([int(l.split("\t")[3]) for l in open(path)],
                              dtype=np.int64)
    return _BIM[path]


def dosages_in(cg_path, pos, lo, hi):
    """(n_variants_in_range, n_samples) dosages. Rows are contiguous in a .bim."""
    a, b = int(np.searchsorted(pos, lo)), int(np.searchsorted(pos, hi, "right"))
    if b <= a:
        return np.zeros((0, 0), np.uint8), a, b
    r = read_cugen(cg_path)
    bpv = int(r.bytes_per_variant)
    raw = np.frombuffer(r.read_packed_bytes(a, b), np.uint8).reshape(b - a, bpv)
    g = _unpack_tile(raw, r.n_samples)
    r.close()
    return g, a, b


def thin(g, rng, cap=MAX_SNPS):
    """MAF filter, then an evenly spaced subset -- a cheap stand-in for tagging."""
    n = g.shape[1]
    af = g.sum(1) / (2.0 * n)
    keep = np.nonzero(np.minimum(af, 1 - af) >= MAF_MIN)[0]
    if keep.size <= cap:
        return g[keep]
    return g[keep[np.linspace(0, keep.size - 1, cap).astype(int)]]


def ga_of(A, B):
    """Observed GA for every cross pair, restricted to full-rank (4 df) tables."""
    nA = A.shape[0]
    G = np.vstack([A, B])
    pr = np.array([(i, nA + j) for i in range(nA) for j in range(B.shape[0])],
                  dtype=np.int64)
    res = ld_from_counts(contingency_tables(G, pr))
    ga, df = np.asarray(res["ga"]), np.asarray(res["ga_df"])
    return ga[df == 4]


def main():
    art = load_artifact()
    rng = np.random.default_rng(SEED)
    out = []
    for pop in POPS:
        for gA, gB in PAIRS:
            cA, loA, hiA = GENES[gA]
            cB, loB, hiB = GENES[gB]
            pA, bA = paths(pop, cA)
            pB, bB = paths(pop, cB)
            posA, posB = bim_pos(bA), bim_pos(bB)
            A, _, _ = dosages_in(pA, posA, loA - FLANK, hiA + FLANK)
            B, _, _ = dosages_in(pB, posB, loB - FLANK, hiB + FLANK)
            if A.size == 0 or B.size == 0:
                print(f"  {pop} {gA}x{gB}: no variants", flush=True)
                continue
            A, B = thin(A, rng), thin(B, rng)
            if A.shape[0] < 2 or B.shape[0] < 2:
                print(f"  {pop} {gA}x{gB}: too few SNPs after MAF/thin", flush=True)
                continue
            cand = ga_of(A, B)

            spanA, spanB = (hiA - loA) + 2 * FLANK, (hiB - loB) + 2 * FLANK
            bg, tries = [], 0
            while len(bg) < N_BACKGROUND and tries < N_BACKGROUND * 40:
                tries += 1
                sA = int(rng.integers(posA[0], max(posA[-1] - spanA, posA[0] + 1)))
                sB = int(rng.integers(posB[0], max(posB[-1] - spanB, posB[0] + 1)))
                if overlaps(art, cA, sA, sA + spanA) or \
                   overlaps(art, cB, sB, sB + spanB):
                    continue
                wA, _, _ = dosages_in(pA, posA, sA, sA + spanA)
                wB, _, _ = dosages_in(pB, posB, sB, sB + spanB)
                if wA.size == 0 or wB.size == 0:
                    continue
                wA, wB = thin(wA, rng), thin(wB, rng)
                if wA.shape[0] < 2 or wB.shape[0] < 2:
                    continue
                v = ga_of(wA, wB)
                if v.size:
                    bg.append(v)
            bgv = np.concatenate(bg) if bg else np.zeros(0)
            # KS on the STATISTIC, which is Rohlfs Figure 1; both sides are
            # computed identically and restricted to 4 df, so they are comparable.
            # Larger GA = more association, so the candidate should be
            # stochastically LARGER -- hence background-vs-candidate here.
            ks = coevo.ks_more_significant(bgv, cand)

            # The KS p-value above is ANTICONSERVATIVE and must not be read as
            # significance: SNP pairs within a gene share LD, so the 64 values
            # are nowhere near 64 independent observations. Rohlfs et al. solve
            # this empirically (their Table 5) by asking where the candidate's
            # KS result falls among KS results from RANDOM gene pairs, which
            # carry exactly the same within-window dependence. That rank is the
            # number to trust.
            null_ks = []
            tries = 0
            while len(null_ks) < N_NULL and tries < N_NULL * 40:
                tries += 1
                sA = int(rng.integers(posA[0], max(posA[-1] - spanA, posA[0] + 1)))
                sB = int(rng.integers(posB[0], max(posB[-1] - spanB, posB[0] + 1)))
                if overlaps(art, cA, sA, sA + spanA) or \
                   overlaps(art, cB, sB, sB + spanB):
                    continue
                wA, _, _ = dosages_in(pA, posA, sA, sA + spanA)
                wB, _, _ = dosages_in(pB, posB, sB, sB + spanB)
                if wA.size == 0 or wB.size == 0:
                    continue
                wA, wB = thin(wA, rng), thin(wB, rng)
                if wA.shape[0] < 2 or wB.shape[0] < 2:
                    continue
                v = ga_of(wA, wB)
                if v.size:
                    null_ks.append(coevo.ks_more_significant(bgv, v)["statistic"])
            null_ks = np.asarray(null_ks, dtype=np.float64)
            emp_p = (1.0 + float(np.sum(null_ks >= ks["statistic"]))) / \
                    (1.0 + null_ks.size) if null_ks.size else float("nan")
            pv = coevo.permutation_pvalues(A, B, n_perm=N_PERM, seed=SEED,
                                           stat="ga")
            rec = dict(pop=pop, pair=f"{gA}x{gB}", n_samples=int(A.shape[1]),
                       n_snps_A=int(A.shape[0]), n_snps_B=int(B.shape[0]),
                       n_cand_pairs=int(cand.size), n_bg_windows=len(bg),
                       n_bg_pairs=int(bgv.size),
                       cand_median_ga=float(np.median(cand)) if cand.size else None,
                       bg_median_ga=float(np.median(bgv)) if bgv.size else None,
                       ks_stat=ks["statistic"], ks_p_naive=ks["pvalue"],
                       ks_p_empirical=emp_p, n_null_pairs=int(null_ks.size),
                       perm_frac_p05=float(np.mean(pv <= 0.05)),
                       perm_min_p=float(np.min(pv)))
            out.append(rec)
            print(f"  {pop:4s} {gA}x{gB:9s} cand_med {rec['cand_median_ga']:5.2f} "
                  f"bg_med {rec['bg_median_ga']:5.2f} "
                  f"KS_naive {ks['pvalue']:9.3g}  KS_empirical {emp_p:6.3f}  "
                  f"perm<=.05 {100*rec['perm_frac_p05']:4.1f}%", flush=True)
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rohlfs_replication.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
