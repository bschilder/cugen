"""Genotype association (GA): the 3x3 genotype test of Rohlfs et al. (2010).

GA is NOT a variant of r^2, and the test that matters here is
`test_ga_catches_what_cld_cannot`. Composite LD asks whether ALT dosages
covary; it is a one-degree-of-freedom additive summary. Two loci can be
strongly non-independent as GENOTYPES while their dosages have exactly zero
covariance, and CLD is blind to that by construction. Rohlfs, Swanson & Weir
(2010) AJHG 86:674 use both statistics for precisely this reason.

The oracle is `scipy.stats.chi2_contingency`, a different implementation of the
same Pearson statistic, so agreement is evidence rather than tautology.
"""
import numpy as np
import pytest
from scipy.stats import chi2 as _chi2
from scipy.stats import chi2_contingency

from cugen import ld as L


def _tab(rows):
    return np.asarray(rows, dtype=np.int64).reshape(1, 3, 3)


# The construction that separates GA from CLD.
#
#         B=0   B=1   B=2          conditional mean of B is 1.0 for EVERY
#  A=0      0   100     0          value of A, so Cov(A, B) = 0 exactly and
#  A=1     50     0    50          r = 0, chi2 = 0. But the genotype table is
#  A=2      0   100     0          wildly non-independent: GA = 300.
CLD_BLIND = _tab([[0, 100, 0], [50, 0, 50], [0, 100, 0]])


def test_ga_catches_what_cld_cannot():
    res = L.ld_from_counts(CLD_BLIND)
    assert abs(float(res["r"][0])) < 1e-12, "construction must give r == 0"
    assert float(res["ga"][0]) == pytest.approx(300.0, rel=1e-9)


def test_ga_matches_scipy_on_random_tables():
    rng = np.random.default_rng(0)
    tabs = rng.integers(1, 60, size=(40, 3, 3))
    got = L.ld_from_counts(tabs)["ga"]
    want = [chi2_contingency(t, correction=False)[0] for t in tabs]
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_ga_is_near_zero_under_exact_independence():
    outer = np.outer([30, 50, 20], [40, 40, 20])  # rank-1 => independent
    assert float(L.ld_from_counts(_tab(outer))["ga"][0]) == pytest.approx(0.0, abs=1e-9)


def test_ga_invariant_to_allele_recoding():
    """Recoding ALT<->REF at a locus reverses that axis; GA must not change."""
    base = _tab([[10, 20, 5], [7, 30, 12], [21, 3, 9]])
    flipped = base[:, ::-1, :]
    both = base[:, ::-1, ::-1]
    g = L.ld_from_counts(base)["ga"][0]
    assert float(L.ld_from_counts(flipped)["ga"][0]) == pytest.approx(g, rel=1e-12)
    assert float(L.ld_from_counts(both)["ga"][0]) == pytest.approx(g, rel=1e-12)


def test_ga_df_is_four_for_a_full_table():
    assert int(L.ld_from_counts(CLD_BLIND)["ga_df"][0]) == 4


def test_ga_df_drops_when_a_genotype_is_absent():
    """Rohlfs et al. exclude these pairs. Emit the realised df, don't pretend."""
    no_hom_alt = _tab([[10, 20, 5], [7, 30, 12], [0, 0, 0]])
    res = L.ld_from_counts(no_hom_alt)
    assert int(res["ga_df"][0]) == 2, "2x3 table has (2-1)*(3-1) = 2 df"
    want = chi2_contingency(np.asarray(no_hom_alt[0])[:2], correction=False)[0]
    assert float(res["ga"][0]) == pytest.approx(want, rel=1e-10)


def test_ga_df_zero_table_is_nan():
    res = L.ld_from_counts(_tab(np.zeros((3, 3))))
    assert np.isnan(res["ga"][0])
    assert int(res["ga_df"][0]) == 0


def test_p_ga_is_neglog10_of_the_upper_tail_at_the_realised_df():
    res = L.ld_from_counts(CLD_BLIND)
    want = -_chi2.logsf(300.0, 4) / np.log(10.0)
    assert float(res["p_ga"][0]) == pytest.approx(want, rel=1e-9)


def test_p_ga_stays_finite_far_past_scipy_sf_underflow():
    """sf underflows to 0.0 around chi2 ~ 1450; -log10(p) must still be real."""
    big = _tab([[3000, 0, 0], [0, 3000, 0], [0, 0, 3000]])
    v = float(L.ld_from_counts(big)["p_ga"][0])
    assert np.isfinite(v) and v > 300.0


def test_ga_is_requestable_through_ld_matrix(tmp_path):
    """End-to-end through the public API on the reference path."""
    from cugen.write import write_cugen

    # Built exactly, not sampled, so r is 0 to the bit rather than merely small:
    # 100 x (0,1), 100 x (1,0), 100 x (1,2), 100 x (2,1).
    a = np.repeat([0, 1, 1, 2], 100).astype(np.uint8)
    b = np.repeat([1, 0, 2, 1], 100).astype(np.uint8)
    dos = np.column_stack([a, b])          # (n_samples, n_variants)
    path = tmp_path / "ga.cugen"
    write_cugen(str(path), dos)

    df = L.ld_matrix(str(path), stats=["r", "chi2", "ga", "ga_df", "p_ga"],
                     backend="numpy", verbose=False)
    assert len(df) == 1
    row = df.iloc[0]
    assert abs(float(row["R"])) < 1e-6
    assert float(row["CHI2"]) < 1e-6
    assert float(row["GA"]) == pytest.approx(400.0, rel=1e-5)
    assert int(row["GA_DF"]) == 4
    assert float(row["NEG_LOG10_P_GA"]) > 80.0
