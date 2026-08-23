"""The fused kernel takes PER-ROW column bounds, so cis/trans and bp-distance
scans reach the fast path.

Before this the kernel took `window` as a scalar, so every predicate needing
per-row bounds fell back to the tiled path -- and `stream=True` and
`count_only` both REQUIRE the fused kernel. That made those scans unmeasurable
at genome scale: count_only is the only way to run every GEMM and write
nothing, so without it a cis/trans scan cannot be timed without also paying
for terabytes of output.

_pair_bounds already returned exactly the (starts, hi) arrays the kernel
needs, and the tiled path already consumed them, so this is plumbing plus a
two-line guard swap:

    if (gj <= gi) return;                                  -- both subsumed by
    if (window > 0 && (gj - gi) > window) return;              one range check

    if (gj < starts[gi] || gj >= his[gi]) return;

starts[i] is i+1 in the unwindowed case, so the upper-triangle guard is not
lost -- it is expressed in the same array as everything else.
"""
import numpy as np
import pandas as pd
import pytest

from cugen import ld as L
from cugen import ldio
from conftest import requires_gpu, simulate_haplotypes
from cugen.write import write_cugen

STATS = ("r", "r2")   # D/D' need the 3x3 table the fused epilogue omits

PRED = [
    pytest.param({"scope": "cis"}, id="cis"),
    pytest.param({"scope": "trans"}, id="trans"),
    pytest.param({"max_dist_kb": 40.0}, id="max_dist"),
    pytest.param({"min_dist_kb": 40.0}, id="min_dist"),
    pytest.param({"min_dist_kb": 20.0, "max_dist_kb": 90.0}, id="band"),
    pytest.param({"window_kb": 60.0}, id="window_kb"),
]


def _idx(df):
    return set(zip(df["gidx_a"].tolist(), df["gidx_b"].tolist()))


@pytest.fixture
def two_chrom(tmp_path):
    dos = simulate_haplotypes(240, 40, seed=23).T.astype(np.float32)
    p = tmp_path / "m.cugen"
    write_cugen(str(p), dos)
    ann = pd.DataFrame({
        "gidx": np.arange(40, dtype=np.int64),
        "CHR": ["1"] * 24 + ["2"] * 16,
        "POS": np.concatenate([(np.arange(24, dtype=np.int64) + 1) * 10_000,
                               (np.arange(16, dtype=np.int64) + 1) * 10_000]),
        "ID": [f"v{i}" for i in range(40)]})
    return str(p), ann


@requires_gpu
@pytest.mark.parametrize("kw", PRED)
def test_count_only_works_and_agrees(two_chrom, kw):
    """count_only is what makes a genome-scale scan measurable at all."""
    path, ann = two_chrom
    ref = L.ld_matrix(path, annotation=ann, backend="numpy", verbose=False,
                      stats=STATS, min_r2=0.05, **kw)
    n = L.ld_matrix(path, annotation=ann, backend="gpu", verbose=False,
                    stats=STATS, min_r2=0.05, count_only=True, **kw)
    assert isinstance(n, (int, np.integer)), f"count_only returned {type(n)}"
    assert n == len(ref), f"count_only says {n}, reference emits {len(ref)}"


@requires_gpu
@pytest.mark.parametrize("kw", PRED)
def test_stream_works_and_agrees(two_chrom, tmp_path, kw):
    path, ann = two_chrom
    ref = L.ld_matrix(path, annotation=ann, backend="numpy", verbose=False,
                      stats=STATS, min_r2=0.05, **kw)
    out = tmp_path / f"s_{abs(hash(tuple(sorted(kw.items()))))}.cugenld"
    n = L.ld_matrix(path, annotation=ann, backend="gpu", verbose=False,
                    stats=STATS, min_r2=0.05, stream=True, output=str(out), **kw)
    assert n == len(ref)
    i, j, _ = ldio.open_ld(str(out)).rows()
    assert set(zip(i.tolist(), j.tolist())) == _idx(ref)


@requires_gpu
@pytest.mark.parametrize("kw", PRED)
def test_count_only_succeeding_IS_the_proof_of_the_fused_path(two_chrom, kw):
    """No string matching needed: count_only raises unless the scan is fused.

    ld_matrix refuses count_only when the fused path is unavailable, because
    returning a DataFrame where the caller expects a number is worse than an
    error. So a count_only call that returns an int cannot have fallen back --
    which makes test_count_only_works_and_agrees a path assertion as well as a
    correctness one, and makes a brittle check of the verbose text unnecessary.
    """
    path, ann = two_chrom
    n = L.ld_matrix(path, annotation=ann, backend="gpu", verbose=False,
                    stats=STATS, min_r2=0.05, count_only=True, **kw)
    assert isinstance(n, (int, np.integer))


@requires_gpu
def test_the_gate_still_refuses_what_genuinely_cannot_be_fused(two_chrom):
    """The control for the above.

    Widening the gate to admit cis/trans must not widen it to admit everything.
    D and D-prime need the 3x3 haplotype table, which the fused epilogue does
    not emit at all, so count_only must still refuse them. Without this test,
    replacing the whole gate with `True` would leave every other test green.
    """
    path, ann = two_chrom
    with pytest.raises(ValueError, match="count_only"):
        L.ld_matrix(path, annotation=ann, backend="gpu", verbose=False,
                    stats=("r", "d", "dp"), min_r2=0.05, count_only=True,
                    scope="cis")


@requires_gpu
def test_unwindowed_still_matches_after_the_guard_swap(two_chrom):
    """starts[i] == i+1, so the upper-triangle guard must survive."""
    path, ann = two_chrom
    a = L.ld_matrix(path, annotation=ann, backend="numpy", verbose=False,
                    stats=STATS, min_r2=0.05)
    b = L.ld_matrix(path, annotation=ann, backend="gpu", verbose=False,
                    stats=STATS, min_r2=0.05)
    assert _idx(a) == _idx(b)
    assert all(x < y for x, y in _idx(b)), "emitted a non-upper-triangle pair"
