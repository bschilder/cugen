"""Tests for cugen.ld.ld_clump.

Almost all of this runs without a GPU. That is by construction, not luck: the
greedy assignment -- the part that decides which variant indexes which clump --
is a pure function of (p-values, edge set), so it is exercised directly. Only
the r^2 that produces the edge set needs a device, and that path is already
covered by test_ld.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cugen.ld import (_CLUMP_BINS, _bin_columns, _clump_bin_counts,
                      _clumps_to_frame, _empty_clumps, _greedy_clump,
                      _load_sumstats, ld_clump)


def order_by_p(pv):
    """Row order the implementation uses: p ascending, row index breaking ties."""
    return np.lexsort((np.arange(len(pv)), np.asarray(pv, dtype=float)))


# ---------------------------------------------------------------------------
# the greedy core
# ---------------------------------------------------------------------------
def test_most_significant_variant_indexes_its_clump():
    pv = np.array([1e-8, 1e-6, 1e-3])
    nb = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    got = _greedy_clump(order_by_p(pv), pv, nb, p1=1e-4)
    assert got == [(0, [1, 2])], "the smallest p must win the index slot"


def test_a_clumped_variant_cannot_become_an_index():
    """The regression that makes clumping clumping.

    Both variants clear p1, so both are index CANDIDATES; the second is in LD
    with the first and must be absorbed rather than opening its own clump.
    """
    pv = np.array([1e-9, 1e-8])
    got = _greedy_clump(order_by_p(pv), pv, {0: [1], 1: [0]}, p1=1e-4)
    assert got == [(0, [1])]
    assert len(got) == 1


def test_membership_is_not_gated_by_p2():
    """p2 is a DISPLAY threshold, which is the opposite of what it looks like.

    Measured: on a 400-variant fixture plink2 v2.0.0-a.7.1 reports the same 182
    clumps and the same TOTAL (149) at p2=0.01 and p2=1.0 -- only SP2 changes.
    So a variant far above p2 still joins the clump and still counts in TOTAL
    and in the NONSIG bin. Gating membership on p2 (the obvious reading) makes
    TOTAL too small and NONSIG always zero.
    """
    pv = np.array([1e-9, 0.5])           # 0.5 is far above any sane p2
    got = _greedy_clump(order_by_p(pv), pv, {0: [1], 1: [0]}, p1=1e-4)
    assert got == [(0, [1])], "p > p2 must not remove a variant from the clump"


def test_variant_above_p1_never_indexes():
    """p1 gates indexing; a variant above it can only ever be a member."""
    pv = np.array([1e-3, 2e-3])          # both > p1
    got = _greedy_clump(order_by_p(pv), pv, {0: [1], 1: [0]}, p1=1e-4)
    assert got == []


def test_variants_out_of_ld_form_separate_clumps():
    pv = np.array([1e-9, 1e-8])
    got = _greedy_clump(order_by_p(pv), pv, {}, p1=1e-4)
    assert got == [(0, []), (1, [])]


def test_allow_overlap_lets_a_member_join_twice():
    """Without overlap the shared variant goes to the stronger index only."""
    pv = np.array([1e-9, 1e-8, 1e-5])    # 2 is in LD with both 0 and 1
    nb = {0: [2], 1: [2], 2: [0, 1]}
    strict = _greedy_clump(order_by_p(pv), pv, nb, 1e-4, allow_overlap=False)
    loose = _greedy_clump(order_by_p(pv), pv, nb, 1e-4, allow_overlap=True)
    assert strict == [(0, [2]), (1, [])]
    assert loose == [(0, [2]), (1, [2])]


def test_allow_overlap_does_not_let_an_absorbed_variant_index():
    """Pinned to measured plink2 behaviour, not to a reading of the docs.

    Fixture mirrors the one used to check it: rsA p=1e-9 and rsC p=1e-5 at
    r^2=0.94, both clearing p1=1e-4. plink2 v2.0.0-a.7.1 produced BYTE-IDENTICAL
    output with and without --clump-allow-overlap, and rsC never indexed a
    clump of its own. So allow_overlap governs membership only; absorption
    always removes a variant from index candidacy.
    """
    pv = np.array([1e-9, 1e-5])          # both < p1
    nb = {0: [1], 1: [0]}
    for overlap in (False, True):
        got = _greedy_clump(order_by_p(pv), pv, nb, 1e-4, allow_overlap=overlap)
        assert got == [(0, [1])], (
            f"allow_overlap={overlap} let an absorbed variant index a clump")


def test_members_are_reported_in_position_order_not_p_order():
    """plink2's SP2 lists members by coordinate: its output reads
    `rs247,rs249,rs252,rs256`, which is position order, not ascending p."""
    pv = np.array([1e-9, 1e-3, 1e-7, 1e-5])
    nb = {0: [3, 1, 2], 1: [0], 2: [0], 3: [0]}   # deliberately unsorted
    (_, members), = _greedy_clump(order_by_p(pv), pv, nb, 1e-4)
    assert members == [1, 2, 3]
    assert list(pv[members]) != sorted(pv[members]), \
        "fixture must distinguish position order from p order"


def test_p_ties_resolve_deterministically():
    """Identical p-values must not make the result depend on dict ordering."""
    pv = np.array([1e-9, 1e-9, 1e-9])
    nb = {0: [1], 1: [0], 2: []}
    first = _greedy_clump(order_by_p(pv), pv, nb, 1e-4)
    for _ in range(5):
        assert _greedy_clump(order_by_p(pv), pv, nb, 1e-4) == first
    assert first[0][0] == 0, "lowest row index breaks the tie"


def test_every_variant_is_used_at_most_once_without_overlap():
    rng = np.random.default_rng(0)
    n = 200
    pv = 10 ** (-rng.uniform(1, 12, size=n))
    nb = {i: [j for j in range(max(0, i - 4), min(n, i + 5)) if j != i]
          for i in range(n)}
    clumps = _greedy_clump(order_by_p(pv), pv, nb, 1e-4)
    seen = [i for i, m in clumps] + [j for _, m in clumps for j in m]
    assert len(seen) == len(set(seen)), "a variant appeared in two clumps"


def test_greedy_is_gpu_free():
    """The decision logic must not import cupy -- it is the part CI can run."""
    import inspect
    src = inspect.getsource(_greedy_clump)
    assert "cp." not in src and "cupy" not in src


# ---------------------------------------------------------------------------
# sumstats parsing
# ---------------------------------------------------------------------------
def test_sumstats_field_search_order_prefers_earlier_names():
    df = pd.DataFrame({"SNP": ["a"], "ID": ["b"], "P": [0.1]})
    out = _load_sumstats(df, ("ID", "SNP"), ("P",), False)
    assert out["ID"].tolist() == ["b"]
    out = _load_sumstats(df, ("SNP", "ID"), ("P",), False)
    assert out["ID"].tolist() == ["a"]


def test_sumstats_missing_column_names_what_it_looked_for():
    df = pd.DataFrame({"variant": ["a"], "pval": [0.1]})
    with pytest.raises(ValueError, match="no id column"):
        _load_sumstats(df, ("ID", "SNP"), ("P",), False)


def test_log10_pvalues_are_converted_not_compared_inverted():
    df = pd.DataFrame({"ID": ["a", "b"], "P": [8.0, 2.0]})   # -log10(p)
    out = _load_sumstats(df, ("ID",), ("P",), log10=True)
    assert np.isclose(out["P"].to_numpy(), [1e-8, 1e-2]).all()


def test_non_numeric_and_missing_pvalues_are_dropped():
    df = pd.DataFrame({"ID": ["a", "b", "c"], "P": ["1e-9", "NA", ""]})
    assert _load_sumstats(df, ("ID",), ("P",), False)["ID"].tolist() == ["a"]


def test_duplicate_ids_keep_the_first():
    df = pd.DataFrame({"ID": ["a", "a"], "P": [1e-9, 0.5]})
    out = _load_sumstats(df, ("ID",), ("P",), False)
    assert len(out) == 1 and out["P"].iloc[0] == 1e-9


# ---------------------------------------------------------------------------
# schema and reporting
# ---------------------------------------------------------------------------
def test_empty_result_has_the_full_schema_and_dtypes():
    e = _empty_clumps()
    assert list(e.columns) == ["CHR", "POS", "ID", "P", "TOTAL", "NONSIG",
                               "S0.05", "S0.01", "S0.001", "S0.0001",
                               "SP2", "gidx"]
    assert len(e) == 0
    assert e["TOTAL"].sum() == 0          # must not raise
    assert e["P"].dtype == np.float64


def test_p2_filters_sp2_but_not_total_or_bins():
    """The asymmetry that a whole-output diff against plink2 exposed.

    TOTAL and the bins count every member; SP2 lists only members at p <= p2.
    Before this was measured, membership was gated on p2 and the result had
    the right index variants with systematically low TOTAL -- the kind of
    near-miss that a spot check passes and a full diff catches.
    """
    rel = pd.DataFrame({"gidx": [0, 1, 2], "ID": ["a", "b", "c"],
                        "POS": [1, 2, 3], "P": [1e-9, 1e-3, 0.5],
                        "CHR": ["1", "1", "1"]})
    clumps = [(0, [1, 2])]
    frame = _clumps_to_frame(clumps, rel, _CLUMP_BINS, p2=0.01)
    row = frame.iloc[0]
    assert row["TOTAL"] == 2, "TOTAL must count the member above p2"
    assert row["NONSIG"] == 1, "the p=0.5 member belongs in NONSIG"
    assert row["SP2"] == "b", "SP2 must list only members at p <= p2"


def test_total_always_equals_the_bin_counts():
    """TOTAL and the bins are two views of one member list and must agree."""
    rng = np.random.default_rng(11)
    rel = pd.DataFrame({
        "gidx": np.arange(40), "ID": [f"v{i}" for i in range(40)],
        "POS": np.arange(40) * 100,
        "P": 10.0 ** (-rng.uniform(0, 9, size=40)), "CHR": "1"})
    clumps = [(0, list(range(1, 25))), (25, list(range(26, 40)))]
    frame = _clumps_to_frame(clumps, rel, _CLUMP_BINS, p2=0.01)
    bins = frame[_bin_columns(_CLUMP_BINS)].to_numpy().sum(axis=1)
    assert (frame["TOTAL"].to_numpy() == bins).all()


def test_empty_member_list_renders_as_plink_dot():
    rel = pd.DataFrame({"gidx": [0], "ID": ["a"], "POS": [1], "P": [1e-9],
                        "CHR": ["1"]})
    frame = _clumps_to_frame([(0, [])], rel, _CLUMP_BINS, p2=0.01)
    assert frame["SP2"].iloc[0] == "."
    assert frame["TOTAL"].iloc[0] == 0


def test_bin_columns_match_plink2_names_and_order():
    """plink2 v2.0.0-a.7.1 writes NONSIG then descending bounds; verified
    against its .clumps output rather than assumed."""
    assert _bin_columns((0.0001, 0.001, 0.01, 0.05)) == [
        "NONSIG", "S0.05", "S0.01", "S0.001", "S0.0001"]


def test_bins_count_members_per_band():
    bins = (0.0001, 0.001, 0.01, 0.05)
    # one member in each band, least significant first
    got = _clump_bin_counts(np.array([0.9, 2e-2, 5e-3, 5e-4, 1e-5]), bins)
    assert got == [1, 1, 1, 1, 1]
    assert _clump_bin_counts(np.empty(0), bins) == [0, 0, 0, 0, 0]
    # a member exactly on a boundary belongs to the more significant band
    assert _clump_bin_counts(np.array([0.01]), bins) == [0, 0, 1, 0, 0]


def test_bin_counts_sum_to_total_members():
    rng = np.random.default_rng(3)
    mp = 10 ** (-rng.uniform(0, 10, size=57))
    assert sum(_clump_bin_counts(mp, _CLUMP_BINS)) == len(mp)


# ---------------------------------------------------------------------------
# plink2 parity, asserted from committed golden files so it holds in CI with
# no plink installed. Runs on the numpy backend, so it needs no GPU either.
# ---------------------------------------------------------------------------
DATA = Path(__file__).parent / "data"
PARITY_COLS = ["POS", "ID", "P", "TOTAL", "NONSIG", "S0.05", "S0.01",
               "S0.001", "S0.0001", "SP2"]


def _clump_fixture(tmp_path):
    from cugen.write import write_cugen
    z = np.load(DATA / "clump_fixture.npz")
    G, pos = z["G"], z["POS"]
    path = tmp_path / "clump.cugen"
    write_cugen(str(path), G.T)          # write_cugen takes (samples, variants)
    ann = pd.DataFrame({"gidx": np.arange(G.shape[0]),
                        "ID": [f"rs{i}" for i in range(G.shape[0])],
                        "POS": pos, "CHR": "1"})
    return str(path), ann


@pytest.mark.parametrize("overlap,gold", [(False, "clump_gold.clumps"),
                                          (True, "clump_gold_overlap.clumps")])
def test_matches_plink2_clump_golden(tmp_path, overlap, gold):
    """Every column of every clump, against plink2 v2.0.0-a.7.1.

    --clump-unphased is the reference: plink2's default --clump uses PHASED
    r^2 and .cugen discards phase, so the unphased form is the only one we can
    match. Same caveat as D/D' in the module docstring.
    """
    path, ann = _clump_fixture(tmp_path)
    got = ld_clump(path, DATA / "clump_sumstats.tsv", annotation=ann,
                   backend="numpy", allow_overlap=overlap, verbose=False)
    want = pd.read_csv(DATA / gold, sep="\t").rename(
        columns=lambda c: c.lstrip("#"))
    assert len(got) == len(want), "different number of clumps"
    a = want.sort_values("ID").reset_index(drop=True)[PARITY_COLS]
    b = got.sort_values("ID").reset_index(drop=True)[PARITY_COLS]
    np.testing.assert_allclose(a["P"].to_numpy(float), b["P"].to_numpy(float),
                               rtol=1e-6)
    for c in [x for x in PARITY_COLS if x != "P"]:
        assert (a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all(), \
            f"column {c} differs from plink2"


def test_golden_exercises_the_asymmetry_it_is_meant_to():
    """A parity test that cannot fail is worthless.

    The golden must contain at least one clump whose TOTAL exceeds its SP2
    count -- i.e. a member above p2 -- otherwise the p2-is-presentational rule
    is never tested and the earlier bug would have passed.
    """
    want = pd.read_csv(DATA / "clump_gold.clumps", sep="\t").rename(
        columns=lambda c: c.lstrip("#"))
    listed = want["SP2"].astype(str).apply(
        lambda s: 0 if s == "." else len(s.split(",")))
    assert (want["TOTAL"] > listed).any(), \
        "golden has no member above p2; it would not catch the TOTAL bug"
    assert want["NONSIG"].sum() > 0, "golden has no NONSIG member"


# ---------------------------------------------------------------------------
# argument validation, before any genotype is read
# ---------------------------------------------------------------------------
def test_annotation_is_required_with_an_actionable_message():
    with pytest.raises(ValueError, match="needs annotation"):
        ld_clump("nofile.cugen", pd.DataFrame({"ID": ["a"], "P": [1e-9]}))


def test_swapped_thresholds_are_rejected_not_silently_degenerate():
    ann = pd.DataFrame({"gidx": [0], "ID": ["a"], "POS": [1]})
    with pytest.raises(ValueError, match="Did you swap them"):
        ld_clump("nofile.cugen", pd.DataFrame({"ID": ["a"], "P": [1e-9]}),
                 annotation=ann, p1=0.01, p2=1e-4)


def test_r2_outside_zero_one_is_rejected():
    ann = pd.DataFrame({"gidx": [0], "ID": ["a"], "POS": [1]})
    with pytest.raises(ValueError, match=r"r2 must be in"):
        ld_clump("nofile.cugen", pd.DataFrame({"ID": ["a"], "P": [1e-9]}),
                 annotation=ann, r2=1.5)


def test_annotation_missing_a_column_says_which_one():
    ann = pd.DataFrame({"gidx": [0], "ID": ["a"]})          # no POS
    with pytest.raises(ValueError, match="'POS'"):
        ld_clump("nofile.cugen", pd.DataFrame({"ID": ["a"], "P": [1e-9]}),
                 annotation=ann)


def test_unmatched_ids_name_the_plink_make_bed_trap():
    ann = pd.DataFrame({"gidx": [0], "ID": ["."], "POS": [1]})
    with pytest.raises(ValueError, match="make-bed"):
        ld_clump("nofile.cugen", pd.DataFrame({"ID": ["rs1"], "P": [1e-9]}),
                 annotation=ann)


def test_nothing_reaching_p1_returns_empty_schema_without_touching_the_file():
    """The file does not exist; reaching it would raise. Returning the empty
    frame proves the p-value screen happened first."""
    ann = pd.DataFrame({"gidx": [0, 1], "ID": ["a", "b"], "POS": [1, 2]})
    ss = pd.DataFrame({"ID": ["a", "b"], "P": [0.5, 0.4]})
    out = ld_clump("nofile.cugen", ss, annotation=ann, verbose=False)
    assert len(out) == 0
    assert list(out.columns) == list(_empty_clumps().columns)


def test_p2_screen_runs_before_any_io():
    """Same idea one threshold up: everything is below p1 but above p2."""
    ann = pd.DataFrame({"gidx": [0], "ID": ["a"], "POS": [1]})
    ss = pd.DataFrame({"ID": ["a"], "P": [0.9]})
    assert len(ld_clump("nofile.cugen", ss, annotation=ann, p2=0.01,
                        verbose=False)) == 0
