"""Genomic relationship matrix.

popstruct.grm was a declared-but-stubbed public API (exported as `grm`,
`make_grm` and `realized_relationship_matrix`). It lands here because the
kinship-corrected LD measures need it, but it is useful on its own -- it is what
PCA and mixed-model association both start from.

Oracle is the GCTA/Yang (2011) definition written out directly in numpy:

    A_jk = (1/M) sum_i (x_ij - 2p_i)(x_ik - 2p_i) / (2 p_i (1 - p_i))

so agreement is evidence rather than tautology.
"""
import numpy as np
import pytest

from cugen import popstruct
from cugen.write import write_cugen

CPU = dict(backend="numpy", verbose=False)


def oracle_grm(dos):
    """(n_samples, n_samples) GCTA-standardised GRM from (n_samples, n_var)."""
    x = np.asarray(dos, dtype=np.float64)
    p = x.mean(axis=0) / 2.0
    keep = (p > 0.0) & (p < 1.0)
    x, p = x[:, keep], p[keep]
    z = (x - 2.0 * p) / np.sqrt(2.0 * p * (1.0 - p))
    return z @ z.T / z.shape[1]


@pytest.fixture
def panel(tmp_path):
    rng = np.random.default_rng(17)
    n, m = 40, 600
    freq = rng.uniform(0.1, 0.9, size=m)
    dos = (rng.random((n, m)) < freq).astype(np.uint8) + \
          (rng.random((n, m)) < freq).astype(np.uint8)
    path = tmp_path / "panel.cugen"
    write_cugen(str(path), dos)
    return str(path), dos


def test_grm_matches_the_gcta_definition(panel):
    path, dos = panel
    got = popstruct.grm(path, **CPU)
    np.testing.assert_allclose(got, oracle_grm(dos), rtol=1e-6, atol=1e-9)


def test_grm_is_symmetric_and_correctly_shaped(panel):
    path, dos = panel
    got = popstruct.grm(path, **CPU)
    assert got.shape == (dos.shape[0], dos.shape[0])
    np.testing.assert_allclose(got, got.T, atol=1e-12)


def test_unrelated_samples_have_diagonal_near_one_and_off_diagonal_near_zero(panel):
    got = popstruct.grm(panel[0], **CPU)
    n = got.shape[0]
    off = got[~np.eye(n, dtype=bool)]
    assert abs(np.mean(np.diag(got)) - 1.0) < 0.05
    assert abs(np.mean(off)) < 0.05


def test_a_duplicated_sample_is_related_to_itself_at_its_own_diagonal(tmp_path):
    """The one relatedness value with an unambiguous right answer.

    Identical genotypes give identical standardised vectors, so A[0, clone] is
    z_0 . z_0 / M, which IS A[0, 0] -- exactly, not approximately. Asserting
    equality with the diagonal is a much sharper check than asserting closeness
    to 1, because the GCTA diagonal estimates 1 + F and is biased low at small
    n: p is estimated from the same samples, so at n = 22 the diagonal sits
    around 0.88 rather than 1. That bias is a property of the estimator (it is
    what GCTA's --grm-adj addresses), not of this implementation.
    """
    rng = np.random.default_rng(18)
    n, m = 60, 2000
    freq = rng.uniform(0.2, 0.8, size=m)
    dos = (rng.random((n, m)) < freq).astype(np.uint8) + \
          (rng.random((n, m)) < freq).astype(np.uint8)
    dup = np.vstack([dos, dos[:2]])          # samples n, n+1 clone 0, 1
    path = tmp_path / "dup.cugen"
    write_cugen(str(path), dup)
    got = popstruct.grm(str(path), **CPU)

    np.testing.assert_allclose(got[0, n], got[0, 0], atol=1e-9)
    np.testing.assert_allclose(got[1, n + 1], got[1, 1], atol=1e-9)
    # and a clone must stand far above the unrelated background
    unrelated = got[2:n, 2:n][~np.eye(n - 2, dtype=bool)]
    assert got[0, n] > 0.8
    assert got[0, n] > np.abs(unrelated).max()


def test_grm_tiling_does_not_change_the_answer(panel):
    """The variant axis is streamed so p can exceed memory; every tile size must
    give the same matrix."""
    path, _ = panel
    full = popstruct.grm(path, **CPU)
    for tile in (7, 64, 100000):
        np.testing.assert_allclose(popstruct.grm(path, tile_size=tile, **CPU),
                                   full, rtol=1e-9, atol=1e-12)


def test_grm_rejects_a_phased_file(tmp_path):
    from cugen.write import write_cugen_phased
    rng = np.random.default_rng(19)
    hap = (rng.random((40, 50)) < 0.5).astype(np.uint8)
    path = tmp_path / "ph.cugen"
    write_cugen_phased(str(path), hap)
    with pytest.raises(ValueError, match="hap2bit|phased|dosage"):
        popstruct.grm(str(path), **CPU)


@pytest.fixture
def offset_panel(tmp_path):
    """Same data, but gidx PERMUTED rather than equal to row position.

    A .cugen written by cugen.convert with gidx_start set -- the normal case for
    a per-chromosome file -- already has gidx that differ from row positions,
    but an offset alone is a weak control: the offset gidx are out of range as
    positions, so a positional misreading would crash rather than mislead. A
    permutation keeps every value a VALID row position while sending it to a
    different row, so the two readings both succeed and disagree.
    """
    rng = np.random.default_rng(23)
    n, m = 30, 40
    freq = rng.uniform(0.15, 0.85, size=m)
    dos = (rng.random((n, m)) < freq).astype(np.uint8) + \
          (rng.random((n, m)) < freq).astype(np.uint8)
    g = rng.permutation(m)
    path = tmp_path / "offset.cugen"
    write_cugen(str(path), dos, gidx=g)
    return str(path), dos, g


def test_grm_accepts_a_scattered_variant_subset(panel):
    """The workflow pcs_from_grm's docstring prescribes.

    Its docstring says to feed an LD-pruned subset, and ld_prune returns a
    scattered set of gidx. Without this, that instruction was unfollowable:
    variant_range takes a contiguous slice only.
    """
    path, dos = panel
    want = np.array([3, 11, 12, 40, 41, 99, 500, 599])
    got = popstruct.grm(path, variants=want, **CPU)
    np.testing.assert_allclose(got, oracle_grm(dos[:, want]),
                               rtol=1e-6, atol=1e-9)


def test_grm_variants_selects_by_gidx_not_row_position(offset_panel):
    """The bug this test exists for.

    ld_matrix resolves `variants` against the STORED gidx, so grm must too.
    Treating them as row positions silently builds the GRM from the wrong
    markers -- and it still returns a plausible PSD matrix.
    """
    path, dos, g = offset_panel
    rows = [0, 2, 4, 6]
    want = [int(g[r]) for r in rows]          # the gidx stored at those rows
    got = popstruct.grm(path, variants=want, **CPU)
    np.testing.assert_allclose(got, oracle_grm(dos[:, rows]),
                               rtol=1e-6, atol=1e-9)
    # The positional misreading is a different, still-valid selection. If it
    # were not, this test would pass under the bug it exists to catch.
    assert sorted(want) != sorted(rows), "fixture permutation is too tame"
    assert not np.allclose(got, oracle_grm(dos[:, want]))


def test_grm_refuses_a_variant_selection_that_matches_nothing(offset_panel):
    """Silently returning a GRM over zero markers would be worse than raising;
    the caller cannot tell an empty selection from a valid one."""
    path, _, _ = offset_panel
    with pytest.raises(ValueError, match="none of the"):
        popstruct.grm(path, variants=[900, 901, 902], **CPU)


def test_grm_variants_composes_with_variant_range(panel):
    """Both filters apply, so a subset outside the range yields the
    intersection -- not the union, and not the subset alone."""
    path, dos = panel
    want = np.array([5, 10, 300, 400])
    got = popstruct.grm(path, variants=want, variant_range=(0, 100), **CPU)
    np.testing.assert_allclose(got, oracle_grm(dos[:, [5, 10]]),
                               rtol=1e-6, atol=1e-9)


def oracle_grm_cov(dos):
    """Mean-centred (covariance) GRM: no 1/sqrt(2p(1-p)) variance weighting."""
    x = np.asarray(dos, dtype=np.float64)
    p = x.mean(axis=0) / 2.0
    keep = (p > 0.0) & (p < 1.0)
    x, p = x[:, keep], p[keep]
    z = x - 2.0 * p
    return z @ z.T / z.shape[1]


@pytest.fixture
def maf_skewed(tmp_path):
    """Half the markers common, half rare -- the shape LD pruning produces.

    Pruning at a fixed r2 preferentially retains LOW-MAF markers, because the
    maximum attainable r2 between markers of frequency a < b is a(1-b)/(b(1-a)),
    so a rare marker simply cannot exceed the threshold against a common one.
    """
    rng = np.random.default_rng(101)
    n, m = 60, 400
    freq = np.concatenate([rng.uniform(0.30, 0.50, m // 2),
                           rng.uniform(0.01, 0.05, m // 2)])
    dos = (rng.random((n, m)) < freq).astype(np.uint8) + \
          (rng.random((n, m)) < freq).astype(np.uint8)
    path = tmp_path / "skew.cugen"
    write_cugen(str(path), dos)
    return str(path), dos


def test_center_standardisation_matches_the_covariance_definition(panel):
    path, dos = panel
    got = popstruct.grm(path, standardize="center", **CPU)
    np.testing.assert_allclose(got, oracle_grm_cov(dos), rtol=1e-6, atol=1e-9)


def test_center_is_not_merely_a_rescaling_of_yang(maf_skewed):
    """If it were, the option would be pointless for PCA -- eigenvectors are
    invariant to a scalar. It must change the RELATIVE marker weighting."""
    path, _ = maf_skewed
    yang = popstruct.grm(path, **CPU)
    cent = popstruct.grm(path, standardize="center", **CPU)
    s = np.trace(cent) / np.trace(yang)
    assert not np.allclose(cent, yang * s, rtol=1e-3), (
        "center differs from yang only by a scalar; no reweighting happened")


def test_center_gives_rare_markers_less_weight_than_yang(maf_skewed):
    """The reason the option exists.

    Yang divides by sqrt(2p(1-p)), so a MAF-0.02 marker gets ~13x the variance
    weight of a MAF-0.40 one and a pruned, rare-enriched marker set produces a
    GRM dominated by its rarest markers. Measured as the spread of the diagonal,
    which is what carries into the leading eigenvectors.
    """
    path, dos = maf_skewed
    spread = lambda A: float(np.diag(A).max() / np.diag(A).min())
    assert spread(popstruct.grm(path, standardize="center", **CPU)) < \
           spread(popstruct.grm(path, **CPU))


def test_an_unknown_standardisation_raises_rather_than_silently_defaulting(panel):
    with pytest.raises(ValueError, match="standardize"):
        popstruct.grm(panel[0], standardize="gcta2", **CPU)
