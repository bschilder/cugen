"""p-values must be recoverable from every format that can be written.

`.cugenld` deliberately does not store -log10 p: it is a closed form in
N * r^2, so a stored copy would be 4-8 redundant bytes against a 2.0 B/pair
budget. Deriving it needs N, and that is where the subtlety is.

Without missingness every pair shares one N and the scalar in the header is
exact. With `missing="pairwise"` each pair rests on its own sample set, so N
varies per pair and a scalar N gives the WRONG p -- silently, and in the
direction that overstates significance for pairs with more missingness. The
format therefore stores a per-pair N, but only when N actually varies, as a
narrow DEFICIT (n_samples - n_obs). Measured: a constant-N file pays zero, a
varying-N file pays 0.5-0.9 B/pair (2.34 -> 2.87-3.24, 23-38%) -- cheaper than
the raw 2 bytes because deficits compress, but not free.

The interop formats (parquet/feather/tsv/csv/npz/zarr) store whatever columns
the frame carries, so for them the requirement is just that the column
survives the round trip.
"""
import numpy as np
import pandas as pd
import pytest

from cugen import ld as L
from cugen import ldio

CPU = dict(backend="numpy", verbose=False)
STATS = ("r", "r2", "p")


def _frame_p(df):
    return dict(zip(zip(df["gidx_a"], df["gidx_b"]), df["NEG_LOG10_P"]))


# ------------------------------------------------- the no-missingness case
def test_cugenld_recovers_p_without_missingness(small_cugen, tmp_path):
    out = tmp_path / "a.cugenld"
    df = L.ld_matrix(small_cugen[0], stats=STATS, output=str(out), **CPU)
    back = ldio.read_pairs(str(out))

    assert "NEG_LOG10_P" in back.columns, "no p column recovered from .cugenld"
    want = _frame_p(df)
    got = _frame_p(back)
    assert set(got) == set(want)
    for key, w in want.items():
        assert got[key] == pytest.approx(w, abs=2e-3), key


def test_a_constant_n_file_stores_no_per_pair_n(small_cugen, tmp_path):
    """Zero bytes for a column that carries no information."""
    out = tmp_path / "b.cugenld"
    L.ld_matrix(small_cugen[0], stats=STATS, output=str(out), **CPU)
    rd = ldio.read_ld(str(out))
    assert rd.has_per_pair_n is False
    assert rd.params["n_obs"], "scalar n_obs must still be recorded"


# ---------------------------------------------------- the missingness case
def test_cugenld_recovers_exact_p_under_missingness(missing_cugen, tmp_path):
    """The case a scalar N gets wrong.

    Asserts two things: that the recovered p matches the computed p, AND that
    a scalar-N reconstruction would NOT have. Without the second assertion
    this test would pass on a build that never stored per-pair N at all.
    """
    out = tmp_path / "c.cugenld"
    df = L.ld_matrix(missing_cugen[0], stats=STATS, missing="pairwise",
                     output=str(out), **CPU)
    assert df["N_OBS"].nunique() > 1, "fixture has constant N; test is vacuous"

    rd = ldio.read_ld(str(out))
    assert rd.has_per_pair_n is True

    back = ldio.read_pairs(str(out))
    want, got = _frame_p(df), _frame_p(back)
    assert set(got) == set(want)
    for key, w in want.items():
        assert got[key] == pytest.approx(w, abs=2e-3), key

    # and the scalar-N answer really is different, so the column earns its bytes
    i, j, r = rd.rows()
    scalar_n = float(rd.params["n_obs"])
    from cugen.ld import _neglog10_chi2_1df
    naive = _neglog10_chi2_1df(scalar_n * r ** 2)
    exact = np.array([want[(a, b)] for a, b in zip(i, j)])
    assert np.abs(naive - exact).max() > 1e-2, (
        "scalar N happens to agree here, so this fixture cannot show the "
        "difference the per-pair column exists for")


def test_per_pair_n_round_trips_exactly(missing_cugen, tmp_path):
    out = tmp_path / "d.cugenld"
    df = L.ld_matrix(missing_cugen[0], stats=STATS, missing="pairwise",
                     output=str(out), **CPU)
    back = ldio.read_pairs(str(out))
    want = dict(zip(zip(df["gidx_a"], df["gidx_b"]), df["N_OBS"]))
    got = dict(zip(zip(back["gidx_a"], back["gidx_b"]), back["N_OBS"]))
    assert got == pytest.approx(want)


# ------------------------------------------------------- refusal, not guessing
def test_p_is_refused_when_it_cannot_be_derived(small_cugen, tmp_path):
    """A file with no N recorded must refuse p rather than invent one."""
    out = tmp_path / "e.cugenld"
    L.ld_matrix(small_cugen[0], stats=("r", "r2"), output=str(out), **CPU)
    rd = ldio.read_ld(str(out))
    rd.params["n_obs"] = None                      # simulate an older writer
    with pytest.raises(ValueError, match="n_obs"):
        rd.rows(with_p=True)


# ------------------------------------------------------- the interop formats
@pytest.mark.parametrize("ext", [".parquet", ".feather", ".tsv", ".csv",
                                 ".tsv.gz", ".npz"])
def test_every_interop_format_round_trips_p(small_cugen, tmp_path, ext):
    out = tmp_path / f"x{ext}"
    df = L.ld_matrix(small_cugen[0], stats=STATS, output=str(out), **CPU)
    back = ldio.read_pairs(str(out))
    assert "NEG_LOG10_P" in back.columns, f"{ext} lost the p column"
    want, got = _frame_p(df), _frame_p(back)
    assert set(got) == set(want)
    for key, w in want.items():
        assert got[key] == pytest.approx(w, rel=1e-5), key
