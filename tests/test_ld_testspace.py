"""Test-space parameters: which pairs enter the computation, and so set m.

cugen already had the RETENTION family (min_r2, max_p, correction, alpha) --
which computed pairs get written. This is the other half: which pairs are
computed at all. The two are coupled and the coupling is the point. m comes
from _count_pairs over the test space, so narrowing the test space lowers m,
which LOOSENS the Bonferroni/BH threshold. A run is only reproducible if both
families travel with the file.

  maf_max                  upper MAF bound, the partner to maf_min
  min_dist_kb              exclude near-diagonal pairs (long-range scans)
  max_dist_kb              band limit; the same predicate window_kb applies
  scope = all|cis|trans    same-chromosome only, or different-chromosome only
  top_k                    RETENTION, not test space: keep each variant's K
                           strongest partners

Two semantics worth stating because they are choices, not consequences:

  * A bp predicate implies cis. Base-pair distance between chromosomes is not
    a number, so window_kb / min_dist_kb / max_dist_kb never emit a
    cross-chromosome pair -- which is what plink2's --ld-window-kb does. That
    makes scope="trans" plus any bp predicate a contradiction, and it raises
    rather than returning the empty frame it would otherwise produce.
  * top_k is SYMMETRIC. A pair is kept if it is in the top K of EITHER
    endpoint, not of whichever endpoint happened to be sorted first. An
    asymmetric top-K would depend on scan order, and a pair's survival would
    depend on which side of the upper triangle it landed on.
"""
import numpy as np
import pandas as pd
import pytest

from cugen import ld as L
from conftest import requires_gpu, simulate_haplotypes
from cugen.write import write_cugen

CPU = dict(backend="numpy", verbose=False)


def _idx(df):
    return set(zip(df["gidx_a"].tolist(), df["gidx_b"].tolist()))


@pytest.fixture
def panel(tmp_path):
    """One chromosome, 24 variants at known 10 kb spacing."""
    dos = simulate_haplotypes(120, 24, seed=7).T.astype(np.float32)
    p = tmp_path / "chr1.cugen"
    write_cugen(str(p), dos)
    ann = pd.DataFrame({"gidx": np.arange(24, dtype=np.int64),
                        "CHR": ["1"] * 24,
                        "POS": (np.arange(24, dtype=np.int64) + 1) * 10_000,
                        "ID": [f"v{i}" for i in range(24)]})
    return str(p), ann


@pytest.fixture
def two_chrom(tmp_path):
    """Two chromosomes as contiguous row blocks, positions restarting."""
    dos = simulate_haplotypes(120, 20, seed=11).T.astype(np.float32)
    p = tmp_path / "merged.cugen"
    write_cugen(str(p), dos)
    ann = pd.DataFrame({"gidx": np.arange(20, dtype=np.int64),
                        "CHR": ["1"] * 12 + ["2"] * 8,
                        "POS": np.concatenate([
                            (np.arange(12, dtype=np.int64) + 1) * 10_000,
                            (np.arange(8, dtype=np.int64) + 1) * 10_000]),
                        "ID": [f"v{i}" for i in range(20)]})
    return str(p), ann


# ------------------------------------------------------------------- maf_max
def test_maf_max_bounds_the_test_space_from_above(panel):
    path, ann = panel
    allp = L.ld_matrix(path, annotation=ann, **CPU)
    hi = L.ld_matrix(path, annotation=ann, maf_max=0.25, **CPU)
    assert _idx(hi) < _idx(allp), "maf_max did not remove any pair"
    assert (hi["MAF_A"] <= 0.25).all() and (hi["MAF_B"] <= 0.25).all()


def test_maf_min_and_maf_max_form_a_band(panel):
    path, ann = panel
    df = L.ld_matrix(path, annotation=ann, maf_min=0.1, maf_max=0.35, **CPU)
    for col in ("MAF_A", "MAF_B"):
        assert (df[col] >= 0.1).all() and (df[col] <= 0.35).all()


def test_maf_max_below_maf_min_raises(panel):
    path, ann = panel
    with pytest.raises(ValueError, match="maf_max"):
        L.ld_matrix(path, annotation=ann, maf_min=0.4, maf_max=0.2, **CPU)


# ------------------------------------------------------- distance predicates
def test_max_dist_kb_matches_window_kb(panel):
    """Same predicate, so the same pairs. If these diverge one is wrong."""
    path, ann = panel
    a = L.ld_matrix(path, annotation=ann, window_kb=50.0, **CPU)
    b = L.ld_matrix(path, annotation=ann, max_dist_kb=50.0, **CPU)
    assert _idx(a) == _idx(b)


def test_min_dist_kb_excludes_the_near_diagonal(panel):
    path, ann = panel
    pos = dict(zip(ann["gidx"], ann["POS"]))
    df = L.ld_matrix(path, annotation=ann, min_dist_kb=50.0, **CPU)
    assert len(df) > 0, "fixture emits nothing; the test would be vacuous"
    d = np.array([pos[b] - pos[a] for a, b in _idx(df)])
    assert (d >= 50_000).all()


def test_min_and_max_dist_partition_the_full_scan(panel):
    """A distance cut splits the pair set exactly: no loss, no overlap."""
    path, ann = panel
    allp = _idx(L.ld_matrix(path, annotation=ann, **CPU))
    near = _idx(L.ld_matrix(path, annotation=ann, max_dist_kb=80.0, **CPU))
    far = _idx(L.ld_matrix(path, annotation=ann, min_dist_kb=80.001, **CPU))
    assert near & far == set()
    assert near | far == allp


def test_min_dist_above_max_dist_raises(panel):
    path, ann = panel
    with pytest.raises(ValueError, match="min_dist_kb"):
        L.ld_matrix(path, annotation=ann, min_dist_kb=100.0,
                    max_dist_kb=50.0, **CPU)


def test_distance_predicates_require_annotation(panel):
    path, _ = panel
    for kw in ({"min_dist_kb": 10.0}, {"max_dist_kb": 10.0}):
        with pytest.raises(ValueError, match="annotation"):
            L.ld_matrix(path, **kw, **CPU)


# -------------------------------------------------------------- cis vs trans
def test_cis_and_trans_partition_the_pair_set(two_chrom):
    path, ann = two_chrom
    chrom = dict(zip(ann["gidx"], ann["CHR"]))
    allp = _idx(L.ld_matrix(path, annotation=ann, **CPU))
    cis = _idx(L.ld_matrix(path, annotation=ann, scope="cis", **CPU))
    trans = _idx(L.ld_matrix(path, annotation=ann, scope="trans", **CPU))

    assert cis and trans, "fixture must exercise both halves"
    assert cis & trans == set()
    assert cis | trans == allp
    assert all(chrom[a] == chrom[b] for a, b in cis)
    assert all(chrom[a] != chrom[b] for a, b in trans)


def test_scope_requires_chromosomes(two_chrom):
    path, _ = two_chrom
    with pytest.raises(ValueError, match="annotation"):
        L.ld_matrix(path, scope="cis", **CPU)


def test_unknown_scope_raises(two_chrom):
    path, ann = two_chrom
    with pytest.raises(ValueError, match="scope"):
        L.ld_matrix(path, annotation=ann, scope="sideways", **CPU)


def test_trans_with_a_bp_predicate_raises(two_chrom):
    """bp distance between chromosomes is not a number. Refuse, don't return 0."""
    path, ann = two_chrom
    for kw in ({"max_dist_kb": 50.0}, {"min_dist_kb": 50.0},
               {"window_kb": 50.0}):
        with pytest.raises(ValueError, match="trans"):
            L.ld_matrix(path, annotation=ann, scope="trans", **kw, **CPU)


def test_a_bp_window_never_emits_a_cross_chromosome_pair(two_chrom):
    """The plink2 semantic: a bp window is implicitly within-chromosome."""
    path, ann = two_chrom
    chrom = dict(zip(ann["gidx"], ann["CHR"]))
    df = L.ld_matrix(path, annotation=ann, max_dist_kb=1e6, **CPU)
    assert len(df) > 0
    assert all(chrom[a] == chrom[b] for a, b in _idx(df))


def test_interleaved_chromosomes_are_refused(tmp_path):
    """cis/trans banding assumes each chromosome is one contiguous row block."""
    dos = simulate_haplotypes(80, 6, seed=3).T.astype(np.float32)
    p = tmp_path / "interleaved.cugen"
    write_cugen(str(p), dos)
    ann = pd.DataFrame({"gidx": np.arange(6, dtype=np.int64),
                        "CHR": ["1", "2", "1", "2", "1", "2"],
                        "POS": (np.arange(6, dtype=np.int64) + 1) * 1000,
                        "ID": [f"v{i}" for i in range(6)]})
    with pytest.raises(ValueError, match="contiguous"):
        L.ld_matrix(str(p), annotation=ann, scope="cis", **CPU)


# --------------------------------------------------------- coupling with m
def test_narrowing_the_test_space_loosens_the_bonferroni_threshold(panel, tmp_path):
    """The coupling, asserted directly.

    m is _count_pairs over the test space, so a narrower space means a smaller
    m and therefore a LARGER alpha/m -- a looser r^2 cut. A build that stopped
    feeding the test space into m would give both runs the same threshold.
    """
    from cugen import ldio
    path, ann = panel

    def run(name, **kw):
        out = tmp_path / name
        L.ld_matrix(path, annotation=ann, correction="bonferroni", alpha=0.05,
                    stats=("r", "r2", "p"), output=str(out), **kw, **CPU)
        return ldio.read_ld(str(out)).params

    wide = run("wide.cugenld")
    narrow = run("narrow.cugenld", max_dist_kb=60.0)

    assert wide["m_tests"] > narrow["m_tests"], "m did not track the test space"
    # smaller m -> larger alpha/m -> LOWER r^2 cut. The threshold loosens.
    assert narrow["min_r2"] < wide["min_r2"]


def test_the_test_space_is_recorded_in_the_written_params(panel, tmp_path):
    from cugen import ldio
    path, ann = panel
    out = tmp_path / "d.cugenld"
    L.ld_matrix(path, annotation=ann, maf_min=0.05, maf_max=0.4,
                min_dist_kb=20.0, max_dist_kb=90.0, output=str(out), **CPU)
    got = ldio.read_ld(str(out)).params
    assert got["maf_max"] == pytest.approx(0.4)
    assert got["min_dist_kb"] == pytest.approx(20.0)
    assert got["max_dist_kb"] == pytest.approx(90.0)
    assert got["scope"] == "all"


# --------------------------------------------------------------------- top_k
def test_top_k_bounds_every_variants_degree(panel):
    path, ann = panel
    k = 3
    df = L.ld_matrix(path, annotation=ann, top_k=k, **CPU)
    deg = {}
    for a, b in _idx(df):
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    # symmetric union: a variant can appear in its own top-K and in others'
    assert min(deg.values()) >= 1, "top_k dropped a variant entirely"
    assert len(df) <= k * 24, "row count is not bounded by k * n_variants"


def test_top_k_keeps_the_strongest_and_is_symmetric(panel):
    """The kept set is exactly the union of each endpoint's top K by |r|."""
    path, ann = panel
    k = 2
    allp = L.ld_matrix(path, annotation=ann, **CPU)
    got = _idx(L.ld_matrix(path, annotation=ann, top_k=k, **CPU))

    both = pd.concat([
        allp[["gidx_a", "gidx_b", "R"]],
        allp[["gidx_b", "gidx_a", "R"]].rename(
            columns={"gidx_b": "gidx_a", "gidx_a": "gidx_b"})])
    both["absr"] = both["R"].abs()
    want = set()
    for v, g in both.groupby("gidx_a"):
        for _, row in g.nlargest(k, "absr").iterrows():
            a, b = int(row["gidx_a"]), int(row["gidx_b"])
            want.add((a, b) if a < b else (b, a))
    assert got == want


def test_top_k_must_be_positive(panel):
    path, ann = panel
    with pytest.raises(ValueError, match="top_k"):
        L.ld_matrix(path, annotation=ann, top_k=0, **CPU)


# ----------------------------------------------------------------- GPU parity
@requires_gpu
def test_gpu_agrees_with_the_reference_on_every_new_predicate(two_chrom):
    path, ann = two_chrom
    for kw in ({"maf_max": 0.3}, {"scope": "cis"}, {"scope": "trans"},
               {"max_dist_kb": 40.0}, {"min_dist_kb": 40.0},
               {"min_dist_kb": 20.0, "max_dist_kb": 80.0}, {"top_k": 2}):
        ref = _idx(L.ld_matrix(path, annotation=ann, backend="numpy",
                               verbose=False, **kw))
        gpu = _idx(L.ld_matrix(path, annotation=ann, backend="gpu",
                               verbose=False, **kw))
        assert ref == gpu, f"gpu and reference disagree on {kw}"
