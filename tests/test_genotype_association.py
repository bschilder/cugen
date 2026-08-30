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

from conftest import requires_gpu

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


def _two_locus_file(tmp_path, name="ga.cugen"):
    from cugen.write import write_cugen
    a = np.repeat([0, 1, 1, 2], 100).astype(np.uint8)
    b = np.repeat([1, 0, 2, 1], 100).astype(np.uint8)
    p = tmp_path / name
    write_cugen(str(p), np.column_stack([a, b]))
    return str(p)


def test_ga_is_gpu_eligible_like_d_not_cpu_only_like_r2_s():
    """GA runs on the GPU counts path; it is not a reference-path statistic.

    Two decisions in `ld_matrix` fix which backend a statistic gets, and this
    pins both without needing a device:

      * `_TABLE_STATS` selects the tiled GPU scan with `need_table=True`, which
        builds the 3x3 counts on device via `_counts_block`. GA belongs here,
        with d/dp, because it is a function of that same table.
      * The GLS-corrected statistics (r2_s / r2_v / r2_vs) are the ones that
        genuinely force CPU -- each needs an n x n eigendecomposition and a
        dense sample-axis transform. GA must NOT be among them.

    An end-to-end `backend="gpu"` check cannot discriminate here, because the
    CuPy-missing guard (ld.py:4102) fires before the GLS downgrade (ld.py:4266),
    so on a CPU-only box every statistic raises alike.
    """
    assert {"ga", "ga_df", "p_ga"} <= L._TABLE_STATS
    assert {"d", "dp"} <= L._TABLE_STATS, "GA must ride the same path as d/dp"

    forces_cpu = set(L._CORRECTED_STATS) - set(L._ANCESTRY_STATS)
    assert {"r2_s", "r2_v", "r2_vs"} <= forces_cpu
    assert not ({"ga", "ga_df", "p_ga"} & forces_cpu)


@requires_gpu
def test_ga_gpu_matches_reference_bit_for_bit(tmp_path):
    """The GPU counts path and the NumPy counts path must agree exactly.

    Both feed the same `ld_from_counts`, and the 3x3 cells are exact integers on
    either side, so this is an equality test rather than a tolerance test.
    """
    rng = np.random.default_rng(11)
    from cugen.write import write_cugen
    dos = rng.integers(0, 3, size=(500, 40)).astype(np.uint8)
    path = str(tmp_path / "gpu_parity.cugen")
    write_cugen(path, dos)

    cols = ["GA", "GA_DF", "NEG_LOG10_P_GA", "D", "DP"]
    kw = dict(stats=["ga", "ga_df", "p_ga", "d", "dp"], verbose=False)
    cpu = L.ld_matrix(path, backend="numpy", **kw).sort_values(
        ["gidx_a", "gidx_b"]).reset_index(drop=True)
    gpu = L.ld_matrix(path, backend="gpu", **kw).sort_values(
        ["gidx_a", "gidx_b"]).reset_index(drop=True)

    assert len(cpu) == len(gpu) and len(cpu) > 0
    np.testing.assert_array_equal(cpu["gidx_a"], gpu["gidx_a"])
    np.testing.assert_array_equal(cpu["GA_DF"], gpu["GA_DF"])
    for c in cols:
        np.testing.assert_allclose(cpu[c], gpu[c], rtol=1e-6, equal_nan=True,
                                   err_msg=f"{c} differs between backends")
