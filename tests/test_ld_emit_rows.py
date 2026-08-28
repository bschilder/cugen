"""Row-bounded emission, which is what makes block-chunked conversion possible.

The AoU chr1 job needs a 0.64 TB .cugen on disk before the scan can read it, and
that is what refused the job. A windowed scan does not need the whole file: with
window = W, every pair has |i - j| <= W, so converting variants [s, e+W) and
emitting only pairs whose FIRST variant lies in [s, e) covers each pair exactly
once. Peak disk becomes one block instead of the chromosome.

`variant_range` cannot express that -- it restricts both axes, so pairs
straddling a block boundary are lost rather than assigned to one block. The
property these tests pin is the one chunking depends on: partition the rows,
union the results, get the unrestricted scan back exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from cugen import ld as L


def _bounds(n_rows, window, emit_rows=None):
    return L._pair_bounds(n_rows, None, window, None, emit_rows=emit_rows)


def test_pair_bounds_empties_rows_outside_the_emit_range():
    starts, hi = _bounds(20, 4, emit_rows=(5, 9))
    counts = np.maximum(hi - starts, 0)
    assert counts[:5].sum() == 0
    assert counts[9:].sum() == 0
    assert counts[5:9].sum() > 0


def test_pair_bounds_row_counts_are_unchanged_inside_the_range():
    _, hi_all = _bounds(20, 4)
    s_all, _ = _bounds(20, 4)
    s_sub, hi_sub = _bounds(20, 4, emit_rows=(5, 9))
    full = np.maximum(hi_all - s_all, 0)
    sub = np.maximum(hi_sub - s_sub, 0)
    assert np.array_equal(sub[5:9], full[5:9]), "in-range rows must not change"


def test_count_pairs_partitions_exactly():
    n, w = 50, 7
    total = L._count_pairs(n, None, w, None)
    parts = [(0, 11), (11, 26), (26, 40), (40, 50)]
    got = sum(L._count_pairs(n, None, w, None, emit_rows=p) for p in parts)
    assert got == total


def test_emit_rows_none_is_the_unrestricted_default():
    a = _bounds(30, 5)
    b = _bounds(30, 5, emit_rows=None)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@pytest.fixture
def _cugen(tmp_path):
    from cugen.write import CugenWriter, ENCODING_2BIT
    n_samples, n_variants = 48, 120
    rng = np.random.default_rng(11)
    out = str(tmp_path / "e.cugen")
    with CugenWriter(out, n_samples, n_variants, ENCODING_2BIT) as w:
        for v in range(n_variants):
            af = rng.uniform(0.15, 0.5)
            w.add_variant(v, rng.binomial(2, af, n_samples).astype(np.float64))
    return out


def test_ld_matrix_partition_by_emit_rows_reproduces_the_whole_scan(_cugen):
    """The end-to-end property: no pair lost, none emitted twice."""
    kw = dict(window=9, min_r2=0.0, backend="numpy", verbose=False)
    whole = L.ld_matrix(_cugen, **kw)
    parts = [(0, 30), (30, 61), (61, 95), (95, 120)]
    pieces = [L.ld_matrix(_cugen, emit_rows=p, **kw) for p in parts]
    import pandas as pd
    got = pd.concat(pieces, ignore_index=True)
    assert len(got) == len(whole), (len(got), len(whole))
    key = lambda d: set(zip(d["gidx_a"].tolist(), d["gidx_b"].tolist()))
    assert key(got) == key(whole)
