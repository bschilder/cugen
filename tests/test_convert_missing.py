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
