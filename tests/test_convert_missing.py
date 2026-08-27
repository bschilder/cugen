"""Missing-genotype policies for the .cugen converters.

The 2-bit encoding has codes 0/1/2 for dosage and 3 for missing. cugen's
association kernels are complete-case and skip code 3, which is why `keep` is
the historical behaviour. The LD scan cannot do that: its fused path -- the only
one supporting stream= or count_only= -- requires a file with no missing calls
at all, and the FLAG_HAS_MISSING bit is file-wide, so one no-call anywhere
disqualifies the whole file. These policies exist to resolve that at conversion
time, which is the only place it can be resolved.
"""
import numpy as np
import pytest

from cugen.convert import _apply_missing_policy

MISSING = 3.0


def test_keep_leaves_the_missing_code_in_place():
    d = np.array([0.0, 1.0, MISSING, 2.0])
    out, n_miss = _apply_missing_policy(d, "keep")
    assert n_miss == 1
    np.testing.assert_array_equal(out, [0.0, 1.0, MISSING, 2.0])


def test_ref_fills_missing_with_hom_ref():
    d = np.array([0.0, 1.0, MISSING, 2.0])
    out, n_miss = _apply_missing_policy(d, "ref")
    assert n_miss == 1
    np.testing.assert_array_equal(out, [0.0, 1.0, 0.0, 2.0])


def test_mean_fills_missing_with_the_rounded_observed_mean():
    # observed 0, 2, 2 -> mean 4/3 = 1.333 -> 1
    d = np.array([0.0, 2.0, MISSING, 2.0])
    out, n_miss = _apply_missing_policy(d, "mean")
    assert n_miss == 1
    np.testing.assert_array_equal(out, [0.0, 2.0, 1.0, 2.0])


def test_mean_rounds_rather_than_storing_a_fraction():
    """2-bit has no fractional code. The stored value must be an integer
    dosage, and a reader must never see a rounded value it cannot represent."""
    d = np.array([0.0, 1.0, 1.0, MISSING])
    out, _ = _apply_missing_policy(d, "mean")
    assert set(np.unique(out)) <= {0.0, 1.0, 2.0}


def test_mean_never_invents_a_call_for_a_fully_missing_variant():
    """No observed calls means no mean. Filling anything is a fabrication, so
    the variant is reported as undefined rather than silently set to 0."""
    d = np.full(4, MISSING)
    out, n_miss = _apply_missing_policy(d, "mean")
    assert n_miss == 4
    assert out is None, "a variant with no observed call must be dropped"


def test_drop_returns_none_when_any_call_is_missing():
    d = np.array([0.0, 1.0, MISSING, 2.0])
    out, n_miss = _apply_missing_policy(d, "drop")
    assert out is None
    assert n_miss == 1


def test_drop_keeps_a_variant_with_no_missing_calls():
    d = np.array([0.0, 1.0, 1.0, 2.0])
    out, n_miss = _apply_missing_policy(d, "drop")
    assert n_miss == 0
    np.testing.assert_array_equal(out, d)


@pytest.mark.parametrize("policy", ["keep", "ref", "mean", "drop"])
def test_observed_calls_are_never_altered(policy):
    """Only the missing entries may change. A policy that clamps or rescales an
    observed dosage would corrupt data that was fine."""
    d = np.array([0.0, 1.0, 2.0, MISSING])
    out, _ = _apply_missing_policy(d, policy)
    if out is not None:
        np.testing.assert_array_equal(out[:3], [0.0, 1.0, 2.0])


@pytest.mark.parametrize("policy", ["keep", "ref", "mean", "drop"])
def test_a_variant_with_no_missing_calls_is_untouched_by_every_policy(policy):
    d = np.array([0.0, 1.0, 2.0, 1.0])
    out, n_miss = _apply_missing_policy(d, policy)
    assert n_miss == 0
    np.testing.assert_array_equal(out, d)


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="missing"):
        _apply_missing_policy(np.zeros(3), "interpolate")


def test_the_input_array_is_not_modified_in_place():
    """Callers reuse one buffer across variants (pgen2cugen does), so mutating
    the argument would leak one variant's fill into the next."""
    d = np.array([0.0, 1.0, MISSING, 2.0])
    before = d.copy()
    _apply_missing_policy(d, "ref")
    np.testing.assert_array_equal(d, before)


# ------------------------------------------------ end-to-end through pgen2cugen

pgenlib = pytest.importorskip("pgenlib", reason="pgen2cugen needs pgenlib")

from cugen.convert import pgen2cugen          # noqa: E402
from cugen.io import read_cugen_header        # noqa: E402


def _write_pgen(tmp_path, G):
    """A real .pgen/.psam pair. G is (n_variants, n_samples) with -9 = missing."""
    G = np.ascontiguousarray(np.asarray(G, dtype=np.int8))
    n_var, n_samp = G.shape
    prefix = tmp_path / "t"
    w = pgenlib.PgenWriter(str(f"{prefix}.pgen").encode(), n_samp, n_var, False)
    for v in range(n_var):
        w.append_biallelic(np.ascontiguousarray(G[v]))
    w.close()
    with open(f"{prefix}.psam", "w") as f:
        f.write("#IID\tSEX\n")
        for i in range(n_samp):
            f.write(f"s{i}\t1\n")
    return prefix


def _dosages(cugen_path, n_samples=None, n_variants=None):
    """Variant-major dosages, via cugen's OWN reader.

    An earlier version of this helper unpacked the 2-bit payload by hand and got
    the byte layout wrong, which reported a correct converter as broken. Read
    through the library instead: the packing is its business, not the test's.
    read_to_numpy() is sample-major, so transpose.
    """
    from cugen.io import read_cugen
    return np.asarray(read_cugen(str(cugen_path)).read_to_numpy()).T


# One clean variant, one with a single no-call, one with two.
_G = np.array([
    [0, 1, 2, 1, 0, 1, 2, 0],          # clean
    [2, 2, 2, 2, 2, 2, 2, -9],         # 1 missing, observed mean 2.0
    [0, 0, 0, 0, -9, -9, 1, 1],        # 2 missing, observed mean 2/6 = 0.33
], dtype=np.int8)
_N, _P = _G.shape[1], _G.shape[0]


def test_keep_sets_the_file_wide_missing_flag(tmp_path):
    """One no-call anywhere sets FLAG_HAS_MISSING for the WHOLE file, which is
    exactly why the fused LD scan cannot use such a file at any missingness."""
    prefix = _write_pgen(tmp_path, _G)
    out = tmp_path / "keep.cugen"
    pgen2cugen(f"{prefix}.pgen", str(out), missing="keep", verbose=False)
    assert read_cugen_header(str(out))["has_missing"] is True


@pytest.mark.parametrize("policy", ["ref", "mean"])
def test_filling_clears_the_missing_flag(tmp_path, policy):
    """The whole point: a filled file has no missing code, so the fused scan --
    the only path with stream= or count_only= -- will accept it."""
    prefix = _write_pgen(tmp_path, _G)
    out = tmp_path / f"{policy}.cugen"
    pgen2cugen(f"{prefix}.pgen", str(out), missing=policy, verbose=False)
    assert read_cugen_header(str(out))["has_missing"] is False


def test_ref_fills_no_calls_with_hom_ref(tmp_path):
    prefix = _write_pgen(tmp_path, _G)
    out = tmp_path / "ref.cugen"
    pgen2cugen(f"{prefix}.pgen", str(out), missing="ref", verbose=False)
    d = _dosages(out)
    assert d[1][7] == 0
    assert d[2][4] == 0 and d[2][5] == 0
    np.testing.assert_array_equal(d[0], _G[0])          # clean row untouched


def test_mean_fills_no_calls_with_the_rounded_observed_mean(tmp_path):
    prefix = _write_pgen(tmp_path, _G)
    out = tmp_path / "mean.cugen"
    pgen2cugen(f"{prefix}.pgen", str(out), missing="mean", verbose=False)
    d = _dosages(out)
    assert d[1][7] == 2, "observed mean 2.0 -> 2"
    assert d[2][4] == 0 and d[2][5] == 0, "observed mean 0.33 -> 0"
    np.testing.assert_array_equal(d[0], _G[0])


def test_mean_and_ref_differ_where_the_variant_is_common(tmp_path):
    """If they never differed the option would be pointless. The difference is
    the AF bias ref introduces, and it lands on common variants."""
    prefix = _write_pgen(tmp_path, _G)
    a, b = tmp_path / "a.cugen", tmp_path / "b.cugen"
    pgen2cugen(f"{prefix}.pgen", str(a), missing="ref", verbose=False)
    pgen2cugen(f"{prefix}.pgen", str(b), missing="mean", verbose=False)
    assert not np.array_equal(_dosages(a), _dosages(b))


def test_drop_writes_only_the_variants_with_no_missing_calls(tmp_path):
    prefix = _write_pgen(tmp_path, _G)
    out = tmp_path / "drop.cugen"
    pgen2cugen(f"{prefix}.pgen", str(out), missing="drop", verbose=False)
    info = read_cugen_header(str(out))
    assert info["n_variants"] == 1, "only the clean variant survives"
    assert info["has_missing"] is False
    np.testing.assert_array_equal(_dosages(out)[0], _G[0])


def test_drop_refuses_when_it_would_leave_nothing(tmp_path):
    """At a real sample count nearly every variant carries a no-call, so this
    is the normal outcome of --geno 0 thinking, not an edge case."""
    prefix = _write_pgen(tmp_path, _G[1:])          # both rows have missing
    out = tmp_path / "empty.cugen"
    with pytest.raises(ValueError, match="leaves 0"):
        pgen2cugen(f"{prefix}.pgen", str(out), missing="drop", verbose=False)


def test_mean_refuses_a_variant_with_no_observed_call(tmp_path):
    """No observed call means no mean. Filling anything would be a fabrication."""
    G = np.array([[0, 1, 2, 1, 0, 1, 2, 0],
                  [-9] * 8], dtype=np.int8)
    prefix = _write_pgen(tmp_path, G)
    out = tmp_path / "undef.cugen"
    with pytest.raises(ValueError, match="no observed call"):
        pgen2cugen(f"{prefix}.pgen", str(out), missing="mean", verbose=False)


def test_the_default_leaves_a_file_the_fused_scan_can_use(tmp_path):
    """The default is what the AoU pipeline relies on; if it ever reverts to
    'keep', every scan there fails at preflight."""
    prefix = _write_pgen(tmp_path, _G)
    out = tmp_path / "default.cugen"
    pgen2cugen(f"{prefix}.pgen", str(out), verbose=False)
    assert read_cugen_header(str(out))["has_missing"] is False


def test_an_unknown_policy_is_refused_before_any_file_is_written(tmp_path):
    prefix = _write_pgen(tmp_path, _G)
    out = tmp_path / "bad.cugen"
    with pytest.raises(ValueError, match="missing"):
        pgen2cugen(f"{prefix}.pgen", str(out), missing="nearest", verbose=False)
    assert not out.exists(), "a refused policy must not leave a partial file"


def test_stats_are_computed_over_the_imputed_dosages(tmp_path):
    """This is the distinction from the failure this module's docstring warns
    about. That bug wrote 0 for missing while leaving sxx over only the
    non-missing calls, so the data and the stats disagreed and high-missingness
    probes manufactured false positives.

    Imputing before add_variant means variant_stats runs over the imputed
    vector, so mu_x/sxx/maf describe exactly the dosages on disk.
    """
    from cugen.io import read_cugen
    prefix = _write_pgen(tmp_path, _G)
    out = tmp_path / "mean.cugen"
    pgen2cugen(f"{prefix}.pgen", str(out), missing="mean", verbose=False)
    rd = read_cugen(str(out))
    d = _dosages(out)
    np.testing.assert_allclose(np.asarray(rd.mu_x), d.mean(axis=1), rtol=1e-6)


def test_rounded_mean_imputation_can_inflate_sxx(tmp_path):
    """The cost of rounding, measured rather than assumed.

    EXACT mean-imputation would leave sxx untouched: the filled samples sit on
    the mean, contribute nothing to the centred sum of squares, and do not move
    the mean. Only the variance estimate sxx/n shrinks, by (1 - missingness).

    2-bit has no fractional code, so the fill is round(mean) and can sit up to
    0.49 away from it. That shifts the mean AND adds spread, so sxx goes UP --
    the same direction as the failure this module's docstring warns about,
    though self-consistent here because the stats are recomputed over the
    imputed vector rather than left over the observed calls.
    """
    from cugen.io import read_cugen
    prefix = _write_pgen(tmp_path, _G)
    kept, filled = tmp_path / "k.cugen", tmp_path / "m.cugen"
    pgen2cugen(f"{prefix}.pgen", str(kept), missing="keep", verbose=False)
    pgen2cugen(f"{prefix}.pgen", str(filled), missing="mean", verbose=False)
    # Variant 2 is [0,0,0,0,.,.,1,1]: observed mean 0.333, so the fill rounds
    # to 0 and lands 0.333 below the mean.
    assert np.asarray(read_cugen(str(filled)).sxx)[2] > \
        np.asarray(read_cugen(str(kept)).sxx)[2]


def test_rounded_mean_is_identical_to_ref_below_af_one_quarter(tmp_path):
    """The degeneracy that matters for a MAF-filtered panel.

    round(mean dosage) is 0 whenever the mean is <= 0.5, i.e. AF <= 0.25. Most
    variants in a MAF>1% panel sit below that, so 'mean' and 'ref' produce
    byte-identical files for them and the choice only bites above AF 0.25.
    """
    low = np.array([[0, 0, 0, 0, 0, 0, 1, -9]], dtype=np.int8)   # observed mean 1/7
    prefix = _write_pgen(tmp_path, low)
    a, b_ = tmp_path / "ref.cugen", tmp_path / "mean.cugen"
    pgen2cugen(f"{prefix}.pgen", str(a), missing="ref", verbose=False)
    pgen2cugen(f"{prefix}.pgen", str(b_), missing="mean", verbose=False)
    np.testing.assert_array_equal(_dosages(a), _dosages(b_))


def test_rounded_mean_differs_from_ref_above_af_one_quarter(tmp_path):
    """...and above it they genuinely differ, which is the whole reason the
    option is not just an alias."""
    high = np.array([[2, 2, 1, 1, 2, 1, 2, -9]], dtype=np.int8)  # observed mean 11/7
    prefix = _write_pgen(tmp_path, high)
    a, b_ = tmp_path / "ref2.cugen", tmp_path / "mean2.cugen"
    pgen2cugen(f"{prefix}.pgen", str(a), missing="ref", verbose=False)
    pgen2cugen(f"{prefix}.pgen", str(b_), missing="mean", verbose=False)
    assert _dosages(a)[0][7] == 0
    assert _dosages(b_)[0][7] == 2
