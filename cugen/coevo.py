"""Coevolution between physically unlinked loci, after Rohlfs, Swanson & Weir.

Rohlfs RV, Swanson WJ, Weir BS (2010) "Detecting Coevolution through Allelic
Association between Physically Unlinked Loci", Am J Hum Genet 86:674-685.

The idea is that selection for allele MATCHING between two interacting proteins
can maintain allelic association even with no linkage, because a mismatched
gamete pair is less fit. The evidence is that a candidate gene pair carries more
association than the genome-wide background of unlinked pairs.

Two design points carry the whole method, and both are easy to get wrong:

**The permutation shuffles individuals, not genotypes.** For each replicate the
sample identities at ONE gene are permuted as a unit, so every SNP in that gene
moves together. Within-gene LD is preserved exactly and only the between-gene
association is randomised. Permuting each SNP independently would destroy
within-gene LD and produce a null no real gene pair could match, which makes
everything look significant.

**The inference is relative, not absolute.** Population structure creates
association between unlinked loci (the two-locus Wahlund effect), and this module
does nothing to remove it. It does not have to: structure inflates the background
gene pairs and the candidate alike, so it cancels in the comparison. That is why
the test survives a confound that would wreck an absolute threshold -- and also
why the background must be drawn from the SAME samples as the candidate. Running
a candidate from one cohort against a background from another silently
reintroduces everything this design cancels.

A corollary worth stating: reference-mapping artifact does NOT cancel, because it
is specific to particular region pairs rather than shared by all of them. Exclude
artifact-prone regions from the background before comparing, or the background is
inflated by a term the candidate does not have and the test becomes conservative
by an unknown amount.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .ld import contingency_tables, ld_from_counts

__all__ = ["permute_block", "pair_statistic", "permutation_pvalues",
           "ks_more_significant"]

#: Statistics this module can test. "ga" is the 3x3 genotype test; "cld" is
#: Weir's composite LD as N * r^2, the 1 df statistic. Rohlfs et al. report
#: both, because CLD is blind to association whose dosage covariance cancels.
_STATS = ("ga", "cld")


def permute_block(G, order):
    """Reindex the SAMPLE axis of a (n_variants, n_samples) block.

    Every variant is permuted by the same `order`, which is what preserves
    within-block LD -- the individuals are relabelled, not rebuilt.
    """
    G = np.asarray(G)
    order = np.asarray(order, dtype=np.int64)
    if G.ndim != 2:
        raise ValueError(f"expected (n_variants, n_samples), got shape {G.shape}")
    if order.shape != (G.shape[1],):
        raise ValueError(
            f"order has {order.shape[0]} entries but the block has "
            f"{G.shape[1]} samples; a partial permutation would drop samples "
            f"from one locus only and bias the null.")
    return G[:, order]


def _statistic(tabs, stat):
    res = ld_from_counts(tabs)
    if stat == "ga":
        return np.asarray(res["ga"], dtype=np.float64)
    if stat == "cld":
        r = np.asarray(res["r"], dtype=np.float64)
        return np.asarray(res["n"], dtype=np.float64) * r * r
    raise ValueError(f"stat must be one of {_STATS}, got {stat!r}")


def pair_statistic(A, B, stat: str = "ga"):
    """(nA, nB) statistic for every cross pair between two genotype blocks.

    A and B are (n_variants, n_samples) ALT-dosage arrays with 3 = missing, in
    the SAME sample order. Pairs within A or within B are not computed: this is
    a between-gene test.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"A has {A.shape[1]} samples and B has {B.shape[1]}; the two blocks "
            f"must be the same individuals in the same order.")
    nA, nB = A.shape[0], B.shape[0]
    G = np.vstack([A, B])
    pairs = np.array([(i, nA + j) for i in range(nA) for j in range(nB)],
                     dtype=np.int64)
    return _statistic(contingency_tables(G, pairs), stat).reshape(nA, nB)


def permutation_pvalues(A, B, n_perm: int = 1000, seed: int = 0,
                        stat: str = "ga"):
    """One-sided permutation p-values for every cross pair, as (nA, nB).

    The null holds A fixed and permutes the sample identities of B as a unit.

    p = (1 + #{permuted >= observed}) / (1 + n_perm), so the floor is
    1/(1 + n_perm) and never zero. A zero p-value would misstate the resolution
    the permutation actually bought, and it propagates badly through the KS
    comparison downstream, where ties at zero flatten the candidate CDF.
    """
    if n_perm < 1:
        raise ValueError(f"n_perm must be >= 1, got {n_perm}")
    if stat not in _STATS:
        raise ValueError(f"stat must be one of {_STATS}, got {stat!r}")
    A = np.asarray(A)
    B = np.asarray(B)
    obs = pair_statistic(A, B, stat=stat)
    rng = np.random.default_rng(seed)
    n_samples = B.shape[1]
    ge = np.zeros_like(obs, dtype=np.int64)
    for _ in range(int(n_perm)):
        p = pair_statistic(A, permute_block(B, rng.permutation(n_samples)),
                           stat=stat)
        # NaN in a replicate means a degenerate table there; count it as not
        # exceeding rather than propagating NaN through the whole cell.
        ge += np.where(np.isfinite(p) & np.isfinite(obs), p >= obs, False)
    return (1.0 + ge) / (1.0 + float(n_perm))


def ks_more_significant(candidate, background) -> Dict[str, float]:
    """One-sided KS: is `candidate` stochastically SMALLER than `background`?

    Both are arrays of p-values. Smaller p-values mean more association, so the
    alternative is that the candidate's CDF lies ABOVE the background's. The
    one-sided form matters: a candidate pair that is markedly LESS associated
    than background is not evidence of coevolution, and a two-sided test would
    score it.

    Returns the KS statistic and its p-value. Note that the KS p-value assumes
    independent observations, which SNP pairs within a gene are not -- Rohlfs et
    al. handle this by comparing the candidate's KS result against the
    DISTRIBUTION of KS results from many random gene pairs, rather than reading
    this p-value as a final significance. Use it as a ranking statistic.
    """
    from scipy.stats import ks_2samp

    cand = np.asarray(candidate, dtype=np.float64).ravel()
    bg = np.asarray(background, dtype=np.float64).ravel()
    cand = cand[np.isfinite(cand)]
    bg = bg[np.isfinite(bg)]
    if cand.size == 0 or bg.size == 0:
        return {"statistic": float("nan"), "pvalue": float("nan"),
                "n_candidate": int(cand.size), "n_background": int(bg.size)}
    res = ks_2samp(cand, bg, alternative="greater")
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue),
            "n_candidate": int(cand.size), "n_background": int(bg.size)}
