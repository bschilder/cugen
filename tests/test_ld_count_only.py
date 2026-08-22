"""count_only: the exact survivor count, with no output buffer allocated.

Genome-scale all-pairs cannot materialise its result in device memory. A
single chromosome already proves it: chr22's full variant set (p=1,055,454)
emitted 187,252,868 rows against the 200e6-row cap in _scan_gpu_fused --
6.4% of headroom. Genome-wide MAF>=1% extrapolates to ~3.2e10 rows, which
is 630 GB of index+r arrays and 7.9x an 80 GB A100.

The epilogue kernel already increments its counter unconditionally and only
writes when ``slot < capacity``, so a zero-capacity run performs every GEMM,
writes nothing, and still counts exactly. count_only exposes that, which
makes genome-scale compute measurable before the streaming writer exists.
"""
import numpy as np
import pytest

import cugen.ld as L
from conftest import requires_cudf, requires_gpu, simulate_haplotypes


@requires_gpu
@requires_cudf
def test_count_only_equals_the_row_count_of_the_written_result(
        tmp_path, write_cugen_file):
    """The count must be the SAME number the full path emits, not an estimate."""
    path = write_cugen_file(simulate_haplotypes(200, 400, seed=11))
    df = L.ld_matrix(path, min_r2=0.2, stats=("r", "r2"),
                     sign_reference="major", output=str(tmp_path / "o.tsv"),
                     max_pairs=10 ** 12, verbose=False)
    n = L.ld_matrix(path, min_r2=0.2, stats=("r", "r2"),
                    sign_reference="major", count_only=True,
                    max_pairs=10 ** 12, verbose=False)
    assert isinstance(n, int), f"count_only must return an int, got {type(n)}"
    assert n == len(df), f"count_only said {n}, full path emitted {len(df)}"
    assert n > 100, f"fixture emitted only {n} pairs -- not exercising the kernel"


@requires_gpu
def test_count_only_needs_no_output_path(tmp_path, write_cugen_file):
    """Counting writes nothing, so demanding output= would be nonsense.

    The fused path is gated on ``output is not None`` because it exists to
    avoid a host round trip when serialising. Counting has nothing to
    serialise, so that gate must not apply.
    """
    path = write_cugen_file(simulate_haplotypes(120, 250, seed=12))
    n = L.ld_matrix(path, min_r2=0.2, stats=("r", "r2"),
                    sign_reference="major", output=None, count_only=True,
                    max_pairs=10 ** 12, verbose=False)
    assert isinstance(n, int) and n > 0


@requires_gpu
def test_count_only_is_not_blocked_by_max_pairs(tmp_path, write_cugen_file):
    """max_pairs guards runaway OUTPUT; counting produces none.

    This is the guard that forces benchmarks/genomewide.sh to pass 10**15 --
    a magic number meaning 'disable this'. It must not stop a count.
    """
    path = write_cugen_file(simulate_haplotypes(120, 300, seed=13))
    n = L.ld_matrix(path, min_r2=0.2, stats=("r", "r2"),
                    sign_reference="major", count_only=True,
                    max_pairs=1, verbose=False)
    assert isinstance(n, int) and n > 0


@requires_gpu
@requires_cudf
def test_count_only_allocates_nothing_sized_to_the_result(
        tmp_path, write_cugen_file, monkeypatch):
    """The whole point: no allocation may scale with the number of survivors.

    Measured by recording every cp.empty request rather than by reading the
    memory pool -- _scan_gpu_fused calls free_all_blocks() before returning,
    so pool.total_bytes() reads 0 afterwards and cannot see the peak.
    """
    import cupy as cp
    path = write_cugen_file(simulate_haplotypes(300, 900, seed=14))

    sizes = []
    real_empty = cp.empty

    def spy(shape, *a, **k):
        sizes.append(shape if isinstance(shape, int)
                     else int(np.prod(shape)) if shape else 0)
        return real_empty(shape, *a, **k)

    monkeypatch.setattr(cp, "empty", spy)

    sizes.clear()
    L.ld_matrix(path, min_r2=0.05, stats=("r", "r2"), sign_reference="major",
                output=str(tmp_path / "full.tsv"), max_pairs=10 ** 12,
                verbose=False)
    full_max = max(sizes)

    sizes.clear()
    n = L.ld_matrix(path, min_r2=0.05, stats=("r", "r2"),
                    sign_reference="major", count_only=True,
                    max_pairs=10 ** 12, verbose=False)
    count_max = max(sizes)

    assert n > 1000, f"only {n} survivors -- raise the fixture's LD"
    assert full_max >= n, (
        f"full path's largest allocation was {full_max:,} elements for {n:,} "
        f"survivors -- test is not observing the output buffer")
    assert count_max < full_max, (
        f"count_only's largest allocation was {count_max:,} elements vs the "
        f"full path's {full_max:,} -- an output buffer was still sized")
    # Independence, not smallness, is the invariant. The largest remaining
    # allocation is the B x n plane pair, which is O(B*n) and legitimately
    # unrelated to the result -- so assert the peak does not MOVE when the
    # survivor count changes by an order of magnitude.
    sizes.clear()
    n_strict = L.ld_matrix(path, min_r2=0.9, stats=("r", "r2"),
                           sign_reference="major", count_only=True,
                           max_pairs=10 ** 12, verbose=False)
    count_max_strict = max(sizes)
    assert n_strict * 5 < n, (
        f"min_r2 0.9 gave {n_strict:,} vs 0.05's {n:,} -- not a big enough "
        f"spread to detect result-dependent allocation")
    assert count_max_strict == count_max, (
        f"peak allocation moved from {count_max:,} to {count_max_strict:,} "
        f"elements when survivors fell from {n:,} to {n_strict:,} -- "
        f"something is still sized by the result")


@requires_gpu
def test_count_only_refuses_when_the_fused_path_is_unavailable(
        tmp_path, write_cugen_file):
    """Returning a DataFrame when a count was asked for is worse than failing.

    D/D' force the 3x3-table path, which has no survivor counter. Falling
    back silently would hand the caller a frame where they expect a number,
    and ``if n > 0`` would still pass on it.
    """
    path = write_cugen_file(simulate_haplotypes(80, 120, seed=15))
    with pytest.raises(ValueError, match="count_only"):
        L.ld_matrix(path, min_r2=0.2, stats=("r", "r2", "d", "dp"),
                    sign_reference="major", count_only=True,
                    max_pairs=10 ** 12, verbose=False)
