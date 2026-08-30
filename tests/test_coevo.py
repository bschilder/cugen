"""The Rohlfs et al. (2010) coevolution test: permutation null and gene-pair KS.

Two things the design turns on, both pinned here:

  1. The permutation shuffles SAMPLE IDENTITIES at one gene as a unit, not each
     SNP independently. Shuffling per SNP would destroy within-gene LD and give
     a null that no real gene pair could ever match, making everything
     significant. `test_permutation_preserves_within_gene_ld` is the guard.

  2. The reported quantity is how EXTREME a candidate gene pair is against a
     background of random gene pairs, not its absolute p-values. Population
     structure inflates every unlinked pair, so it lands in the background and
     cancels. That is the whole reason the test survives the Wahlund effect.
"""
import numpy as np
import pytest

from cugen import coevo
from cugen import ld as L


def _block(n_snps, n_samples, rng, latent=None, mix=0.0):
    """Dosages with optional shared latent structure (within-gene LD)."""
    out = np.zeros((n_snps, n_samples), dtype=np.uint8)
    for v in range(n_snps):
        f = rng.uniform(0.2, 0.8)
        s = rng.random(n_samples) if latent is None else \
            mix * latent + (1 - mix) * rng.random(n_samples)
        out[v] = ((s < f).astype(np.uint8) + (rng.random(n_samples) < f))
    return out


def test_permutation_preserves_within_gene_ld():
    """Shuffling identities at a gene must not change that gene's own LD."""
    rng = np.random.default_rng(0)
    lat = rng.random(300)
    G = _block(6, 300, rng, latent=lat, mix=0.8)
    perm = coevo.permute_block(G, rng.permutation(300))
    pairs = np.array([[i, j] for i in range(6) for j in range(i + 1, 6)])
    before = L.ld_from_counts(L.contingency_tables(G, pairs))["r2"]
    after = L.ld_from_counts(L.contingency_tables(perm, pairs))["r2"]
    np.testing.assert_allclose(before, after, rtol=1e-12)


def test_permutation_destroys_between_gene_association():
    rng = np.random.default_rng(1)
    lat = rng.random(400)
    A = _block(3, 400, rng, latent=lat, mix=0.95)
    B = _block(3, 400, rng, latent=lat, mix=0.95)   # A and B share the latent
    obs = coevo.pair_statistic(A, B, stat="ga")
    perm = coevo.pair_statistic(A, coevo.permute_block(B, rng.permutation(400)),
                               stat="ga")
    assert obs.mean() > perm.mean()


def test_pvalues_are_uniform_under_independence():
    rng = np.random.default_rng(2)
    A = _block(4, 250, rng)
    B = _block(4, 250, rng)
    p = coevo.permutation_pvalues(A, B, n_perm=200, seed=3, stat="ga")
    assert p.shape == (4, 4)
    assert ((p > 0) & (p <= 1)).all()
    # a uniform null should not pile up at either end
    assert 0.2 < float(np.mean(p)) < 0.8


def test_planted_association_is_significant():
    rng = np.random.default_rng(4)
    n = 300
    a = rng.integers(0, 3, size=n).astype(np.uint8)
    A = np.vstack([a])
    B = np.vstack([a.copy()])          # perfectly matched genotypes
    p = coevo.permutation_pvalues(A, B, n_perm=200, seed=5, stat="ga")
    assert p[0, 0] <= 1.0 / 201.0 + 1e-12


def test_pvalues_are_never_zero():
    """A zero p-value is a lie about resolution; the floor is 1/(n_perm+1)."""
    rng = np.random.default_rng(6)
    a = rng.integers(0, 3, size=200).astype(np.uint8)
    p = coevo.permutation_pvalues(np.vstack([a]), np.vstack([a]),
                                  n_perm=50, seed=7, stat="ga")
    assert float(p[0, 0]) == pytest.approx(1.0 / 51.0)


def test_ks_detects_a_candidate_shifted_below_background():
    rng = np.random.default_rng(8)
    background = rng.uniform(size=500)
    candidate = rng.uniform(size=60) ** 3      # stochastically smaller
    res = coevo.ks_more_significant(candidate, background)
    assert res["statistic"] > 0.2
    assert res["pvalue"] < 0.01


def test_ks_is_not_significant_for_an_ordinary_gene_pair():
    rng = np.random.default_rng(9)
    res = coevo.ks_more_significant(rng.uniform(size=60), rng.uniform(size=500))
    assert res["pvalue"] > 0.05


def test_ks_is_one_sided():
    """A candidate LESS significant than background must not score."""
    rng = np.random.default_rng(10)
    background = rng.uniform(size=500)
    worse = rng.uniform(size=60) ** 0.3        # stochastically larger
    assert coevo.ks_more_significant(worse, background)["pvalue"] > 0.5
