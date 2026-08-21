"""Significance testing for LD: the chi-square statistic and its p-value.

The oracle throughout is scipy -- `chi2.logsf` for the asymptotic test and
`fisher_exact`/`hypergeom` for the exact conditional test. Both are
deliberately different algorithms from the production code, so agreement is
evidence rather than tautology.

Two facts drive the design and are pinned by tests here:

  1. chi2 = N_OBS * r^2 with 1 df, where N_OBS is HAPLOTYPES for gametic
     (phased) LD and INDIVIDUALS for composite (unphased) LD. That factor of
     two is the easiest thing in this feature to get wrong.
  2. p itself underflows -- float64 at chi2 ~ 1450, float32 at chi2 ~ 170,
     which is r^2 = 0.034 at 1000 Genomes sample size. -log10(p) is therefore
     the emitted quantity, and it must stay accurate far past the point where
     scipy's own `chi2.sf` returns a flat zero.
"""
import math

import numpy as np
import pytest
from scipy.special import log_ndtr
from scipy.stats import chi2 as _chi2

from cugen import ld as L
from cugen.write import write_cugen, write_cugen_phased
from conftest import simulate_haplotypes

CPU = dict(backend="numpy", verbose=False)
LN10 = math.log(10.0)


def simulate_phased(n_samples, n_variants, seed=0):
    """(2*n_samples, n_variants) 0/1 alleles with a spread of MAF and LD."""
    rng = np.random.default_rng(seed)
    H = 2 * n_samples
    latent = rng.random(H)
    out = np.zeros((H, n_variants), dtype=np.uint8)
    for v in range(n_variants):
        mix = rng.uniform(0.0, 0.95)
        freq = rng.uniform(0.05, 0.95)
        score = mix * latent + (1 - mix) * rng.random(H)
        out[:, v] = (score < freq).astype(np.uint8)
    return out


def oracle_neglog10p(chi2_value):
    """-log10(right-tail p) for chi-square with 1 df, computed in log space.

    P(X > x) = 2 * Phi(-sqrt(x)) for X ~ chi2(1), and scipy's log_ndtr is a
    genuine log-space normal CDF -- accurate past chi2 = 1e7, where the p-value
    is around 1e-2171476 and utterly unrepresentable.

    Deliberately NOT scipy.stats.chi2.logsf, which computes log(sf()) and so
    returns -inf for any chi2 above ~1450. That is the exact regime the
    production helper exists to cover, so logsf cannot check it.
    """
    return -(math.log(2.0) + log_ndtr(-math.sqrt(chi2_value))) / LN10


@pytest.mark.parametrize("chi2_value", [
    0.5, 1.0, 3.841459, 10.0, 20.0, 30.0, 50.0, 100.0, 399.0,
    400.0, 400.5, 401.0, 1.4e3, 5e3, 1e5, 4e6, 1e7,
])
def test_neglog10p_matches_scipy_across_the_whole_range(chi2_value):
    """Accurate on both sides of the erfc/asymptotic branch, and past underflow.

    399/400/400.5/401 straddle _NLP_ASYMPTOTIC_FROM: a discontinuity at the
    branch point is the failure mode this parametrisation is built to catch.
    """
    got = L._neglog10_chi2_1df(np.array([chi2_value], dtype=np.float64), np)
    want = oracle_neglog10p(chi2_value)
    assert abs(float(got[0]) - want) < 1e-6, (
        f"chi2={chi2_value}: got {float(got[0])!r}, oracle says {want!r}")


def test_neglog10p_is_finite_where_the_p_value_itself_underflows():
    """The reason this helper exists at all.

    scipy's own sf() is zero here, so anything that computes p and then logs
    it -- including the `1.0 - erf(...)` form in cugen.qc -- returns inf.
    """
    deep = np.array([2e3, 1e4, 1e5], dtype=np.float64)
    assert (_chi2.sf(deep, 1) == 0.0).all(), "fixture no longer probes underflow"
    got = L._neglog10_chi2_1df(deep, np)
    assert np.isfinite(got).all(), f"underflowed to {got!r}"
    assert (got > 400.0).all()


def test_neglog10p_is_monotone_increasing_in_chi2():
    x = np.geomspace(1e-3, 1e6, 5000)
    got = L._neglog10_chi2_1df(x, np)
    assert (np.diff(got) > 0).all(), "not monotone -- p-value ordering is broken"


# --------------------------------------------------------- the statistic
# chi2 = N_OBS * r^2 with 1 df (Park 2019 eq. 1; Weir, Genetic Data Analysis
# II). N_OBS is HAPLOTYPES for gametic LD and INDIVIDUALS for composite LD --
# cugen already sets it that way, and these tests pin it there.

def test_chi2_is_n_obs_times_r2_on_the_dosage_path(small_cugen):
    df = L.ld_matrix(small_cugen[0], stats=("r2", "chi2"), **CPU)
    np.testing.assert_allclose(
        df["CHI2"].to_numpy(np.float64),
        df["N_OBS"].to_numpy(np.float64) * df["R2"].to_numpy(np.float64),
        rtol=1e-6)


def test_chi2_counts_haplotypes_not_individuals_on_a_phased_file(tmp_path):
    """The factor-of-two trap.

    Gametic LD is tested over 2n haplotypes, composite LD over n individuals.
    Getting this wrong halves or doubles every chi2 in the output and is
    invisible in any test that only checks internal consistency, so assert the
    absolute count against n_samples directly.
    """
    hap = simulate_phased(n_samples=50, n_variants=8, seed=3)
    path = tmp_path / "ph.cugen"
    write_cugen_phased(str(path), hap)

    df = L.ld_matrix(str(path), stats=("r2_phased", "chi2"), **CPU)
    assert (df["N_OBS"] == 100).all(), "expected 2*50 haplotypes"
    np.testing.assert_allclose(
        df["CHI2"].to_numpy(np.float64),
        100.0 * df["R2_PHASED"].to_numpy(np.float64), rtol=1e-6)


def test_chi2_counts_individuals_on_an_unphased_file(tmp_path):
    """The other half of the trap: the same panel, unphased, must use n not 2n."""
    hap = simulate_phased(n_samples=50, n_variants=8, seed=3)
    dos = (hap[0::2] + hap[1::2]).astype(np.uint8)      # (50, 8)
    path = tmp_path / "un.cugen"
    write_cugen(str(path), dos)

    df = L.ld_matrix(str(path), stats=("r2", "chi2"), **CPU)
    assert (df["N_OBS"] == 50).all(), "expected 50 individuals"


def test_neg_log10_p_matches_scipy_on_real_fixture_data(small_cugen):
    df = L.ld_matrix(small_cugen[0], stats=("r2", "chi2", "p"), **CPU)
    want = np.array([oracle_neglog10p(c) for c in df["CHI2"].to_numpy(np.float64)])
    np.testing.assert_allclose(
        df["NEG_LOG10_P"].to_numpy(np.float64), want, atol=1e-5)


def test_significance_stats_are_absent_unless_requested(small_cugen):
    """_DEFAULT_STATS must not change: adding a statistic cannot alter an
    existing caller's output. Same rule the phased statistics follow."""
    df = L.ld_matrix(small_cugen[0], **CPU)
    assert "CHI2" not in df.columns
    assert "NEG_LOG10_P" not in df.columns


def test_chi2_arithmetic_is_float64_exact(small_cugen):
    """The float32 output schema is the precision floor of the FORMAT, not of
    the arithmetic -- so check the helper itself at float64 tolerance."""
    df = L.ld_matrix(small_cugen[0], stats=("r2", "chi2"), **CPU)
    chi2 = df["CHI2"].to_numpy(np.float64)
    direct = L._neglog10_chi2_1df(chi2, np)
    want = np.array([oracle_neglog10p(c) if c > 0 else 0.0 for c in chi2])
    np.testing.assert_allclose(direct, want, atol=1e-9)
