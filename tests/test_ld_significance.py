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


# ------------------------------------------------- filtering and correction

def _bh_reject_count(neglog10p, m, alpha):
    """Textbook Benjamini-Hochberg, done the slow obvious way in log space.

    Independent of the production implementation: sorts descending, walks every
    rank, takes the largest k that satisfies the BH inequality. O(K log K) and
    allocates freely -- this is the oracle, not the shipping path.
    """
    order = np.sort(np.asarray(neglog10p, dtype=np.float64))[::-1]
    best = 0
    for k in range(1, order.size + 1):
        if order[k - 1] >= math.log10(m / (k * alpha)):
            best = k
    return best


def test_max_p_is_exactly_equivalent_to_the_matching_r2_threshold(small_cugen):
    """The whole reason p-filtering is free.

    With N constant across pairs, p is a strictly monotone function of r^2, so
    a p-value cut IS an r^2 cut and the existing in-kernel min_r2 filter does
    the work. If these two calls ever diverge, the filter is not free any more.
    """
    path, dos = small_cugen
    n = dos.shape[1]
    max_p = 1e-3
    by_p = L.ld_matrix(path, stats=("r2", "p"), max_p=max_p, **CPU)
    by_r2 = L.ld_matrix(path, stats=("r2", "p"),
                        min_r2=_chi2.isf(max_p, 1) / n, **CPU)
    assert len(by_p) == len(by_r2) > 0
    np.testing.assert_array_equal(by_p["gidx_a"].to_numpy(),
                                 by_r2["gidx_a"].to_numpy())
    np.testing.assert_array_equal(by_p["gidx_b"].to_numpy(),
                                 by_r2["gidx_b"].to_numpy())


def test_max_p_actually_bounds_the_emitted_p_values(small_cugen):
    df = L.ld_matrix(small_cugen[0], stats=("r2", "p"), max_p=1e-3, **CPU)
    assert (df["NEG_LOG10_P"] >= 3.0 - 1e-6).all()


def test_max_p_and_min_r2_together_are_refused(small_cugen):
    with pytest.raises(ValueError, match="max_p.*min_r2|min_r2.*max_p"):
        L.ld_matrix(small_cugen[0], stats=("r2",), max_p=1e-3, min_r2=0.2, **CPU)


@pytest.mark.parametrize("bad", [0.0, -1.0, 1.5])
def test_max_p_out_of_range_is_refused(small_cugen, bad):
    with pytest.raises(ValueError, match="max_p"):
        L.ld_matrix(small_cugen[0], stats=("r2",), max_p=bad, **CPU)


def test_bonferroni_threshold_is_alpha_over_the_number_of_tests(small_cugen):
    path, dos = small_cugen
    p_var = dos.shape[0]
    m = p_var * (p_var - 1) // 2
    got = L.ld_matrix(path, stats=("r2", "p"), correction="bonferroni",
                      alpha=0.05, **CPU)
    want = L.ld_matrix(path, stats=("r2", "p"), max_p=0.05 / m, **CPU)
    assert len(got) == len(want)
    np.testing.assert_array_equal(got["gidx_a"].to_numpy(),
                                 want["gidx_a"].to_numpy())


def test_fdr_matches_a_textbook_bh_over_the_same_tests(small_cugen):
    path, dos = small_cugen
    p_var = dos.shape[0]
    m = p_var * (p_var - 1) // 2
    every = L.ld_matrix(path, stats=("r2", "p"), **CPU)
    assert len(every) == m, "fixture must emit every pair for the oracle to hold"
    want_k = _bh_reject_count(every["NEG_LOG10_P"].to_numpy(np.float64), m, 0.05)
    got = L.ld_matrix(path, stats=("r2", "p"), correction="fdr", alpha=0.05,
                      **CPU)
    assert len(got) == want_k


def test_fdr_is_never_stricter_than_bonferroni(small_cugen):
    path = small_cugen[0]
    fdr = L.ld_matrix(path, stats=("r2", "p"), correction="fdr", alpha=0.05,
                      **CPU)
    bon = L.ld_matrix(path, stats=("r2", "p"), correction="bonferroni",
                      alpha=0.05, **CPU)
    assert len(fdr) >= len(bon)


def test_unknown_correction_is_refused(small_cugen):
    with pytest.raises(ValueError, match="correction"):
        L.ld_matrix(small_cugen[0], stats=("r2",), correction="holm", **CPU)


def test_correction_requires_the_p_statistic(small_cugen):
    """Asking for FDR without asking for p is a mistake worth naming, not
    something to silently paper over by adding the column."""
    with pytest.raises(ValueError, match="correction.*'p'|'p'.*correction"):
        L.ld_matrix(small_cugen[0], stats=("r2",), correction="fdr", **CPU)


@pytest.fixture
def linked_cugen(tmp_path):
    """A panel with genuinely strong LD, so a tight r^2 screen keeps some pairs.

    The `dosages` fixture is independent draws, where no pair reaches r^2 = 0.9
    and a tight screen leaves nothing at all -- which exercises the empty-result
    path rather than the guard under test.
    """
    rng = np.random.default_rng(11)
    n, p_var = 100, 12
    base = rng.integers(0, 3, size=n).astype(np.uint8)
    dos = np.empty((n, p_var), dtype=np.uint8)
    for v in range(p_var):
        q = 0.02 + 0.10 * (v / (p_var - 1))     # 2%..12% of calls corrupted
        noisy = rng.integers(0, 3, size=n).astype(np.uint8)
        dos[:, v] = np.where(rng.random(n) < q, noisy, base)
    path = tmp_path / "linked.cugen"
    write_cugen(str(path), dos)
    return str(path)


def test_fdr_refuses_when_the_screen_may_have_truncated_discoveries(linked_cugen):
    """A screen tight enough that BH rejects everything left is a trap.

    BH's threshold depends on where the k-th smallest p-value sits. If every
    survivor is rejected, the true cut may lie below the screen, and pairs that
    BH would also have rejected were discarded before it ran. Returning the
    truncated set would quietly under-report discoveries, so refuse instead.
    """
    unscreened = L.ld_matrix(linked_cugen, stats=("r2", "p"), **CPU)
    assert (unscreened["R2"] >= 0.5).all(), "fixture has no strong LD to screen on"
    with pytest.raises(ValueError, match="survived the screen"):
        L.ld_matrix(linked_cugen, stats=("r2", "p"), min_r2=0.5,
                    correction="fdr", alpha=0.5, **CPU)


def test_fdr_with_a_harmless_screen_still_agrees_with_textbook_bh(small_cugen):
    """A screen that keeps more pairs than BH rejects cannot change the answer,
    because everything it dropped has a larger p than the cut."""
    path, dos = small_cugen
    p_var = dos.shape[0]
    m = p_var * (p_var - 1) // 2
    every = L.ld_matrix(path, stats=("r2", "p"), **CPU)
    want_k = _bh_reject_count(every["NEG_LOG10_P"].to_numpy(np.float64), m, 0.05)
    assert want_k > 0, "fixture must have some discoveries for this to mean anything"
    screened = L.ld_matrix(path, stats=("r2", "p"), min_r2=1e-6,
                           correction="fdr", alpha=0.05, **CPU)
    assert len(screened) == want_k


def test_max_p_converts_using_the_haplotype_count_on_a_phased_file(tmp_path):
    """The factor-of-two trap again, this time in the THRESHOLD conversion.

    max_p becomes min_r2 = chi2.isf(max_p, 1) / N. Using n_samples instead of
    2*n_samples on a phased file doubles the required r^2 and silently drops
    real pairs, which no internal-consistency check would notice.
    """
    hap = simulate_phased(n_samples=60, n_variants=10, seed=7)
    path = tmp_path / "ph.cugen"
    write_cugen_phased(str(path), hap)
    max_p = 1e-4

    by_p = L.ld_matrix(str(path), stats=("r2_phased", "p"), max_p=max_p, **CPU)
    by_hap = L.ld_matrix(str(path), stats=("r2_phased", "p"),
                         min_r2=_chi2.isf(max_p, 1) / 120, **CPU)
    by_ind = L.ld_matrix(str(path), stats=("r2_phased", "p"),
                         min_r2=_chi2.isf(max_p, 1) / 60, **CPU)
    assert len(by_p) == len(by_hap)
    assert len(by_hap) > len(by_ind), (
        "fixture cannot tell the two conversions apart; pick a max_p where "
        "some pair falls between the haplotype and individual thresholds")


def test_fdr_uses_the_test_count_not_the_survivor_count(small_cugen):
    """BH's threshold is m*t/alpha, and m is every test PLANNED, not every test
    that survived the screen. Screening 66 pairs down to 43 must not make the
    remaining 43 easier to call significant."""
    path, dos = small_cugen
    p_var = dos.shape[0]
    m = p_var * (p_var - 1) // 2
    every = L.ld_matrix(path, stats=("r2", "p"), **CPU)
    survivors = int((every["R2"] >= 0.01).sum())
    want_k = _bh_reject_count(every["NEG_LOG10_P"].to_numpy(np.float64), m, 0.05)
    assert want_k < survivors < m, (
        f"fixture must screen strictly and still keep more than BH rejects; "
        f"got k={want_k}, survivors={survivors}, m={m}")

    got = L.ld_matrix(path, stats=("r2", "p"), min_r2=0.01, correction="fdr",
                      alpha=0.05, **CPU)
    assert len(got) == want_k
