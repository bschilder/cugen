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
from conftest import requires_gpu

from cugen.ld import (_CLUMP_BINS, _bin_columns, _clump_bin_counts,
                      _clumps_from_pairs, _clumps_to_frame, _empty_clumps,
                      _greedy_clump, _load_sumstats, clump_core, ld_clump,
                      membership_pairs)


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
# the array-parallel core, with the sequential greedy as its oracle
# ---------------------------------------------------------------------------
def _random_clump_case(rng):
    """A windowed graph with p-value ties -- like a kb window on sorted rows."""
    n = int(rng.integers(4, 80))
    pv = 10.0 ** (-rng.uniform(0, 11, size=n))
    if rng.random() < 0.35:                       # force ties deliberately
        pv[rng.integers(0, n, size=max(2, n // 4))] = 1e-6
    w = int(rng.integers(1, 9))
    eu, ev = [], []
    for i in range(n):
        for j in range(i + 1, min(n, i + w + 1)):
            if rng.random() < 0.5:
                eu.append(i)
                ev.append(j)
    eu = np.array(eu, dtype=np.int32)
    ev = np.array(ev, dtype=np.int32)
    order = np.lexsort((np.arange(n), pv))
    rank = np.empty(n, dtype=np.int32)
    rank[order] = np.arange(n, dtype=np.int32)
    nb = {}
    for i, j in zip(eu.tolist(), ev.tolist()):
        nb.setdefault(i, []).append(j)
        nb.setdefault(j, []).append(i)
    return n, pv, eu, ev, order, rank, nb


def test_parallel_core_equals_sequential_greedy():
    """The load-bearing claim: the parallel formulation is not an
    approximation of greedy clumping, it computes the identical answer.

    Index selection is the lexicographically-first maximal independent set
    under p-priority; membership is an argmin rather than a sequence. 400
    random windowed graphs x both overlap modes, with ties forced in a third
    of them -- ties being exactly where a parallel tie-break could diverge.
    """
    rng = np.random.default_rng(1)
    for _ in range(400):
        n, pv, eu, ev, order, rank, nb = _random_clump_case(rng)
        p1 = float(rng.choice([1e-4, 1e-2, 1.0]))
        for ov in (False, True):
            want = _greedy_clump(order, pv, nb, p1, ov)
            is_idx, owner, _ = clump_core(eu, ev, rank, pv <= p1, ov, xp=np)
            a, b = membership_pairs(is_idx, owner, rank, eu, ev, ov, xp=np)
            assert _clumps_from_pairs(is_idx, rank, a, b) == want


def test_mis_converges_in_a_handful_of_rounds():
    """Round count is what makes the parallel form worth having: at O(p)
    rounds it would be the sequential loop with extra steps."""
    rng = np.random.default_rng(7)
    rounds = []
    for _ in range(120):
        n, pv, eu, ev, order, rank, nb = _random_clump_case(rng)
        _, _, r = clump_core(eu, ev, rank, pv <= 1.0, False, xp=np)
        rounds.append(r)
    assert max(rounds) <= 8, f"MIS needed {max(rounds)} rounds"
    assert np.mean(rounds) < 4


def test_no_overlap_membership_never_reads_the_edge_list():
    """Under no-overlap the owner array alone determines membership, which is
    why that path costs O(p) on the host instead of O(edges)."""
    rng = np.random.default_rng(3)
    n, pv, eu, ev, order, rank, nb = _random_clump_case(rng)
    is_idx, owner, _ = clump_core(eu, ev, rank, pv <= 1.0, False, xp=np)
    empty = np.empty(0, dtype=np.int32)
    a1, b1 = membership_pairs(is_idx, owner, rank, eu, ev, False, xp=np)
    a2, b2 = membership_pairs(is_idx, owner, rank, empty, empty, False, xp=np)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


def test_clump_kernel_source_is_pure_ascii():
    """Regression guard for 34b4a59 -- an NVRTC crash from one non-ASCII
    character, which cannot be reproduced without a GPU."""
    from cugen.ld import _CLUMP_SRC
    assert _CLUMP_SRC.isascii()


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


def test_p1_above_p2_is_allowed_it_is_the_prs_configuration():
    """p1=1, p2=0.01 is clumping-and-thresholding for polygenic scores: index
    on every variant, list only the significant members. An earlier version
    rejected it on the false premise that p2 gates membership, which blocked
    the most valuable configuration. plink2 accepts it, so we must too.

    Reaching the missing file proves validation let it through.
    """
    ann = pd.DataFrame({"gidx": [0], "ID": ["a"], "POS": [1]})
    ss = pd.DataFrame({"ID": ["a"], "P": [1e-9]})
    with pytest.raises((FileNotFoundError, OSError)) as e:
        ld_clump("nofile.cugen", ss, annotation=ann, p1=1.0, p2=0.01,
                 backend="numpy", verbose=False)
    assert "swap" not in str(e.value).lower()


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


# ---------------------------------------------------------------------------
# GPU. These are the only tests here that need a device; everything above
# validates the same algorithm through its NumPy path.
# ---------------------------------------------------------------------------
@requires_gpu
def test_clump_core_gpu_matches_cpu():
    """The kernels and np.minimum.at must agree exactly -- same algorithm,
    two atomicMin implementations."""
    import cupy as cp
    rng = np.random.default_rng(11)
    for _ in range(60):
        n, pv, eu, ev, order, rank, nb = _random_clump_case(rng)
        p1 = float(rng.choice([1e-4, 1e-2, 1.0]))
        cand = pv <= p1
        for ov in (False, True):
            ic, oc, rc = clump_core(eu, ev, rank, cand, ov, xp=np)
            ig, og, rg = clump_core(cp.asarray(eu), cp.asarray(ev),
                                    cp.asarray(rank), cp.asarray(cand),
                                    ov, xp=cp)
            assert rc == rg, "different round counts"
            assert np.array_equal(ic, cp.asnumpy(ig))
            assert np.array_equal(oc, cp.asnumpy(og))


@requires_gpu
@pytest.mark.parametrize("overlap", [False, True])
def test_gpu_clump_matches_plink2_golden(tmp_path, overlap):
    """End to end on the device, against the same plink2 goldens the CPU path
    is held to. This is the test that would have caught the fused path being
    silently bypassed."""
    gold = "clump_gold_overlap.clumps" if overlap else "clump_gold.clumps"
    path, ann = _clump_fixture(tmp_path)
    got = ld_clump(path, DATA / "clump_sumstats.tsv", annotation=ann,
                   backend="gpu", allow_overlap=overlap, verbose=False)
    want = pd.read_csv(DATA / gold, sep="\t").rename(
        columns=lambda c: c.lstrip("#"))
    a = want.sort_values("ID").reset_index(drop=True)[PARITY_COLS]
    b = got.sort_values("ID").reset_index(drop=True)[PARITY_COLS]
    assert len(a) == len(b)
    np.testing.assert_allclose(a["P"].to_numpy(float), b["P"].to_numpy(float),
                               rtol=1e-6)
    for c in [x for x in PARITY_COLS if x != "P"]:
        assert (a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all(), \
            f"column {c} differs from plink2 on the GPU path"


@requires_gpu
@pytest.mark.parametrize("overlap", [False, True])
def test_gpu_and_cpu_backends_agree(tmp_path, overlap):
    path, ann = _clump_fixture(tmp_path)
    kw = dict(annotation=ann, allow_overlap=overlap, verbose=False)
    g = ld_clump(path, DATA / "clump_sumstats.tsv", backend="gpu", **kw)
    c = ld_clump(path, DATA / "clump_sumstats.tsv", backend="numpy", **kw)
    pd.testing.assert_frame_equal(g.reset_index(drop=True),
                                  c.reset_index(drop=True))


@requires_gpu
def test_gpu_path_handles_p1_of_one(tmp_path):
    """The C+T configuration: every variant an index candidate, low r2. This
    is the regime the device path exists for."""
    path, ann = _clump_fixture(tmp_path)
    kw = dict(annotation=ann, p1=1.0, p2=0.01, r2=0.1, verbose=False)
    g = ld_clump(path, DATA / "clump_sumstats.tsv", backend="gpu", **kw)
    c = ld_clump(path, DATA / "clump_sumstats.tsv", backend="numpy", **kw)
    pd.testing.assert_frame_equal(g.reset_index(drop=True),
                                  c.reset_index(drop=True))
    assert len(g) > 0


def test_vectorised_frame_matches_the_list_builder():
    """The production path builds the table straight from arrays; the list
    builder survives only as the tests' oracle. They must not drift.

    The list form boxes every membership into a Python int, which is fine here
    and cost an OOM kill on real chr22 -- so the fast path is the one that
    ships, and this is what keeps it honest.
    """
    from cugen.ld import _clumps_frame_vec
    rng = np.random.default_rng(5)
    for _ in range(40):
        n, pv, eu, ev, order, rank, nb = _random_clump_case(rng)
        rel = pd.DataFrame({
            "gidx": np.arange(n), "ID": [f"v{i}" for i in range(n)],
            "POS": np.arange(n) * 1000, "P": pv, "CHR": "1"})
        for ov in (False, True):
            is_idx, owner, _ = clump_core(eu, ev, rank, pv <= 1.0, ov, xp=np)
            a, b = membership_pairs(is_idx, owner, rank, eu, ev, ov, xp=np)
            slow = _clumps_to_frame(
                _clumps_from_pairs(is_idx, rank, a, b), rel, _CLUMP_BINS, 0.01)
            fast = _clumps_frame_vec(is_idx, rank, a, b, rel, _CLUMP_BINS, 0.01)
            pd.testing.assert_frame_equal(slow.reset_index(drop=True),
                                          fast.reset_index(drop=True))


def test_sp2_false_drops_only_the_member_id_column():
    """sp2=False is the O(clumps) escape hatch: counts must be unaffected."""
    from cugen.ld import _clumps_frame_vec
    rng = np.random.default_rng(9)
    n, pv, eu, ev, order, rank, nb = _random_clump_case(rng)
    rel = pd.DataFrame({"gidx": np.arange(n), "ID": [f"v{i}" for i in range(n)],
                        "POS": np.arange(n) * 1000, "P": pv, "CHR": "1"})
    is_idx, owner, _ = clump_core(eu, ev, rank, pv <= 1.0, False, xp=np)
    a, b = membership_pairs(is_idx, owner, rank, eu, ev, False, xp=np)
    on = _clumps_frame_vec(is_idx, rank, a, b, rel, _CLUMP_BINS, 0.01, sp2=True)
    off = _clumps_frame_vec(is_idx, rank, a, b, rel, _CLUMP_BINS, 0.01, sp2=False)
    cols = [c for c in on.columns if c != "SP2"]
    pd.testing.assert_frame_equal(on[cols], off[cols])
    assert (off["SP2"] == ".").all()


def test_gpu_backend_never_silently_runs_the_cpu_reference(tmp_path,
                                                           monkeypatch):
    """Regression: a .cugen flagged HAS_MISSING used to send backend='gpu'
    into a hardcoded backend='numpy' call.

    That is not a slow path, it is a different algorithm's cost model --
    O(p * window * n) -- and on chr22 it was an OOM kill that looked like a
    memory bug in the device code. The device code had never run.
    """
    import cugen.ld as L
    seen = {}
    real = L.ld_matrix

    def spy(*a, **kw):
        seen["backend"] = kw.get("backend")
        return real(*a, **kw)

    monkeypatch.setattr(L, "ld_matrix", spy)
    path, ann = _clump_fixture(tmp_path)
    # Force the non-fused branch regardless of the fixture's flags.
    monkeypatch.setattr(L, "HAS_CUPY", False)
    L.ld_clump(path, DATA / "clump_sumstats.tsv", annotation=ann,
               backend="numpy", verbose=False)
    assert seen["backend"] == "numpy", (
        "the requested backend must be passed through, not overridden")


# ---------------------------------------------------------------------------
# the rectangular scan. The committed clump fixture is 74-100% candidates, so
# left alone it exercises the BANDED path only -- these force the other one.
# ---------------------------------------------------------------------------
SPARSE_SS = DATA / "clump_sumstats_sparse.tsv"
SPARSE_GOLD = DATA / "clump_gold_sparse.clumps"


def test_sparse_fixture_actually_selects_few_candidates():
    """Guard the guard: if this fixture ever drifts dense, the rectangular
    tests below would silently start testing the banded path instead."""
    ss = pd.read_csv(SPARSE_SS, sep="\t")
    frac = float((ss["P"] <= 1e-4).mean())
    assert frac < 0.2, f"{frac:.1%} candidates -- too dense to force rect scan"


def test_sparse_golden_exercises_the_p2_asymmetry():
    want = pd.read_csv(SPARSE_GOLD, sep="\t").rename(
        columns=lambda c: c.lstrip("#"))
    listed = want["SP2"].astype(str).apply(
        lambda s: 0 if s == "." else len(s.split(",")))
    assert (want["TOTAL"] > listed).any()
    assert want["NONSIG"].sum() > 0


def test_matches_plink2_on_sparse_candidates(tmp_path):
    """End-to-end parity where only 4% of variants are index candidates --
    the standard-GWAS regime, and the one the banded scan was 23x too slow
    for on real chr22."""
    path, ann = _clump_fixture(tmp_path)
    got = ld_clump(path, SPARSE_SS, annotation=ann, backend="numpy",
                   verbose=False)
    want = pd.read_csv(SPARSE_GOLD, sep="\t").rename(
        columns=lambda c: c.lstrip("#"))
    assert len(got) == len(want)
    a = want.sort_values("ID").reset_index(drop=True)[PARITY_COLS]
    b = got.sort_values("ID").reset_index(drop=True)[PARITY_COLS]
    np.testing.assert_allclose(a["P"].to_numpy(float), b["P"].to_numpy(float),
                               rtol=1e-6)
    for c in [x for x in PARITY_COLS if x != "P"]:
        assert (a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all(), \
            f"column {c} differs from plink2"


@requires_gpu
def test_rectangular_and_banded_scans_agree(tmp_path, monkeypatch):
    """The two scan shapes must be interchangeable. They evaluate different
    SETS of pairs -- rectangular skips pairs where neither side is a candidate
    -- so agreement is evidence that the skipped pairs genuinely cannot matter.
    """
    import cugen.ld as L
    path, ann = _clump_fixture(tmp_path)
    kw = dict(annotation=ann, backend="gpu", verbose=False)
    monkeypatch.setattr(L, "_CLUMP_DENSE_FRAC", 1.01)      # force rectangular
    rect = L.ld_clump(path, SPARSE_SS, **kw)
    monkeypatch.setattr(L, "_CLUMP_DENSE_FRAC", -0.01)     # force banded
    band = L.ld_clump(path, SPARSE_SS, **kw)
    pd.testing.assert_frame_equal(rect.reset_index(drop=True),
                                  band.reset_index(drop=True))


@requires_gpu
def test_rectangular_scan_matches_plink2_golden(tmp_path, monkeypatch):
    import cugen.ld as L
    monkeypatch.setattr(L, "_CLUMP_DENSE_FRAC", 1.01)      # force rectangular
    path, ann = _clump_fixture(tmp_path)
    got = L.ld_clump(path, SPARSE_SS, annotation=ann, backend="gpu",
                     verbose=False)
    want = pd.read_csv(SPARSE_GOLD, sep="\t").rename(
        columns=lambda c: c.lstrip("#"))
    a = want.sort_values("ID").reset_index(drop=True)[PARITY_COLS]
    b = got.sort_values("ID").reset_index(drop=True)[PARITY_COLS]
    assert len(a) == len(b)
    for c in [x for x in PARITY_COLS if x != "P"]:
        assert (a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all(), \
            f"column {c} differs from plink2 on the rectangular path"


@requires_gpu
def test_variant_moments_kernel_is_exact():
    """The moments kernel replaced a pre-pass that built three fp32 planes to
    produce two per-variant sums. It must be EXACT, not merely close: the
    whole module rests on the contingency data being exact integers, and these
    sums feed the r denominator.
    """
    import cupy as cp
    from cugen.ld import _variant_moments
    from cugen.io import read_cugen
    from cugen.write import write_cugen
    import tempfile, os
    rng = np.random.default_rng(21)
    for n, p in ((200, 30), (1000, 64), (4001, 40)):     # 4001: ragged tail
        G = rng.integers(0, 3, size=(p, n)).astype(np.uint8)
        d = tempfile.mkdtemp()
        path = os.path.join(d, "m.cugen")
        write_cugen(path, G.T)
        rd = read_cugen(path)
        packed = cp.asarray(np.frombuffer(rd.read_packed_bytes(),
                                          dtype=np.uint8))
        packed = packed.reshape(int(rd.n_variants), int(rd.bytes_per_variant))
        s, q = _variant_moments(packed, p, int(rd.n_samples),
                                int(rd.bytes_per_variant))
        want_s = G.sum(axis=1, dtype=np.int64)
        want_q = (G.astype(np.int64) ** 2).sum(axis=1)
        assert np.array_equal(cp.asnumpy(s).astype(np.int64), want_s), \
            f"sum(g) wrong at n={n}"
        assert np.array_equal(cp.asnumpy(q).astype(np.int64), want_q), \
            f"sum(g*g) wrong at n={n}"


# ---------------------------------------------------------------------------
# candidate tiling. Pure position arithmetic, so it is testable without a GPU,
# and it is what stops scattered candidates dragging one tile across a
# chromosome.
# ---------------------------------------------------------------------------
def test_scattered_candidates_do_not_share_one_tile():
    """The regression that made the rectangular scan slow.

    Candidates 20 Mb apart have disjoint windows; grouping them into one tile
    makes its union span everything between them, so the scan evaluates the
    whole range for every candidate in the tile.
    """
    from cugen.ld import _plan_cand_tiles
    pos = np.arange(20000, dtype=np.int64) * 1000        # 1 variant per kb
    cand = np.array([100, 5000, 10000, 15000, 19000])    # far apart
    tiles = _plan_cand_tiles(cand, pos, span=250_000, max_cands=256,
                             row_budget=4096)
    assert len(tiles) == len(cand), (
        f"{len(cand)} scattered candidates collapsed into {len(tiles)} tile(s)")


def test_clustered_candidates_do_share_a_tile():
    """The other half: candidates inside one window SHOULD batch, or the scan
    rebuilds the same neighbour plane once per candidate."""
    from cugen.ld import _plan_cand_tiles
    pos = np.arange(20000, dtype=np.int64) * 1000
    cand = np.array([5000, 5010, 5020, 5030])            # within 250 kb
    tiles = _plan_cand_tiles(cand, pos, span=250_000, max_cands=256,
                             row_budget=4096)
    assert len(tiles) == 1, f"clustered candidates split into {len(tiles)} tiles"


def test_tiles_partition_the_candidates_in_order():
    from cugen.ld import _plan_cand_tiles
    rng = np.random.default_rng(2)
    pos = np.sort(rng.choice(np.arange(1, 30_000_000), size=8000,
                             replace=False))
    cand = np.sort(rng.choice(8000, size=200, replace=False))
    tiles = _plan_cand_tiles(cand, pos, 250_000, 256, 4096)
    assert np.array_equal(np.concatenate(tiles), cand)
    assert all(len(t) for t in tiles)


def test_tile_count_is_bounded_by_max_cands():
    from cugen.ld import _plan_cand_tiles
    pos = np.arange(50000, dtype=np.int64) * 10          # very dense
    cand = np.arange(0, 1000)                            # all in one window
    tiles = _plan_cand_tiles(cand, pos, 250_000, max_cands=64,
                             row_budget=10**9)
    assert max(len(t) for t in tiles) <= 64


# ---------------------------------------------------------------------------
# ranged reads. Profiling put 96% of standard-GWAS clumping in the read, so
# this is the hot path -- and getting run boundaries wrong would silently
# permute genotype rows, which no correctness test downstream would attribute
# to the reader.
# ---------------------------------------------------------------------------
def test_contiguous_runs_splits_correctly():
    from cugen.ld import _contiguous_runs
    assert _contiguous_runs([]) == []
    assert _contiguous_runs([5]) == [(5, 6)]
    assert _contiguous_runs([0, 1, 2, 3]) == [(0, 4)]
    assert _contiguous_runs([0, 1, 5, 6, 7, 20]) == [(0, 2), (5, 8), (20, 21)]


def test_contiguous_runs_covers_every_row_exactly_once():
    """The invariant that matters: runs must reproduce the input, in order."""
    rng = np.random.default_rng(8)
    for _ in range(200):
        rows = np.sort(rng.choice(5000, size=int(rng.integers(1, 400)),
                                  replace=False))
        from cugen.ld import _contiguous_runs
        runs = _contiguous_runs(rows)
        rebuilt = np.concatenate([np.arange(lo, hi) for lo, hi in runs])
        assert np.array_equal(rebuilt, rows)
        # and runs must be disjoint and ascending
        assert all(runs[k][1] <= runs[k + 1][0] for k in range(len(runs) - 1))


def test_contiguous_runs_of_a_window_union_is_few_runs():
    """A clumping relevant-set is a union of kb windows, so it should be a
    handful of long runs -- which is what makes ranged reads worth doing. If
    this ever becomes thousands of singletons, the read strategy is wrong."""
    from cugen.ld import _contiguous_runs
    pos = np.arange(20000, dtype=np.int64) * 1000
    cand = np.array([100, 5000, 10000, 15000, 19000])
    span = 250_000
    keep = np.zeros(len(pos), bool)
    for c in cand:
        lo = np.searchsorted(pos, pos[c] - span, "left")
        hi = np.searchsorted(pos, pos[c] + span, "right")
        keep[lo:hi] = True
    runs = _contiguous_runs(np.flatnonzero(keep))
    assert len(runs) == len(cand), f"{len(runs)} runs for {len(cand)} windows"
    assert sum(hi - lo for lo, hi in runs) < len(pos) / 3, \
        "windows should cover a fraction of the file, not most of it"


@requires_gpu
def test_ranged_read_matches_a_whole_file_read(tmp_path):
    """The ranged read must produce byte-identical rows to reading everything
    and gathering. Row order is the failure mode: a wrong run boundary
    permutes genotypes, and every downstream test would blame the kernels."""
    import cupy as cp
    from cugen.ld import _load_packed_rows
    from cugen.io import read_cugen
    from cugen.write import write_cugen
    rng = np.random.default_rng(13)
    G = rng.integers(0, 3, size=(400, 1000)).astype(np.uint8)
    path = str(tmp_path / "r.cugen")
    write_cugen(path, G.T)
    rd = read_cugen(path)
    bpv = int(rd.bytes_per_variant)
    whole = cp.asarray(np.frombuffer(rd.read_packed_bytes(), dtype=np.uint8))
    whole = whole.reshape(int(rd.n_variants), bpv)
    for rows in (np.arange(400),                        # identity
                 np.arange(50, 150),                    # one run
                 np.r_[0:20, 100:140, 380:400],         # three runs
                 np.array([7])):                        # single row
        got = _load_packed_rows(rd, rows, bpv)
        want = whole[cp.asarray(rows)]
        assert cp.array_equal(got, want), f"mismatch for {len(rows)} rows"
