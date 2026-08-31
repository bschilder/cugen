"""Row-blocked packed-row residency for the fused LD scan.

cugen's fused scan held every selected variant's packed bytes in ONE device
allocation -- _load_packed_rows does cp.empty((p, ceil(n/4))). At n=414,830 that
is 103,708 bytes per variant, so chr1's millions of variants want ~830 GB
against a T4's 14.5 GB, and the scan OOMs before its first GEMM.

The scan does not need them all. It touches packed rows only as [i0:i1] and
[j0:j1] with j0 >= i0 and j1 <= i1 + window, so at most B + window rows are live
at any moment. These tests pin that: the cache serves the same bytes, and the
scan's results are unchanged.
"""
import numpy as np
import pytest

from cugen import ld as L

# The cache's correctness is index arithmetic and refill bookkeeping, which is
# array-module agnostic, so it takes xp= and these run on CPU. Without that the
# whole file would skip on any machine lacking cupy -- which is where it is
# being written.
XP = np


class _FakeReader:
    """Serves packed bytes from an in-memory array, counting host reads."""

    def __init__(self, packed):
        self._packed = np.asarray(packed, dtype=np.uint8)
        self.n_variants = self._packed.shape[0]
        self.bytes_per_variant = self._packed.shape[1]
        self.reads = 0
        self.rows_read = 0

    def read_packed_bytes(self, lo, hi):
        self.reads += 1
        self.rows_read += hi - lo
        return self._packed[lo:hi].tobytes()


def _packed(p, bpv, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (p, bpv), dtype=np.uint8)


def test_cache_serves_the_same_bytes_as_a_whole_file_load():
    ref = _packed(200, 7)
    rd = _FakeReader(ref)
    cache = L._PackedRowCache(rd, np.arange(200), 7, cap_rows=40, xp=XP)
    for lo in range(0, 200, 13):
        hi = min(200, lo + 11)
        got = np.asarray(cache[lo:hi])
        np.testing.assert_array_equal(got, ref[lo:hi])


def test_cache_never_holds_more_than_its_capacity():
    ref = _packed(500, 5)
    rd = _FakeReader(ref)
    cache = L._PackedRowCache(rd, np.arange(500), 5, cap_rows=64, xp=XP)
    for lo in range(0, 500, 20):
        cache[lo:min(500, lo + 20)]
        assert cache.resident <= 64


def test_cache_reads_far_fewer_rows_than_the_whole_file_when_windowed():
    """The point of the exercise. A sliding read over a 5,000-row file with a
    64-row cache must not degenerate into re-reading everything each step."""
    p = 5_000
    rd = _FakeReader(_packed(p, 4))
    cache = L._PackedRowCache(rd, np.arange(p), 4, cap_rows=256, xp=XP)
    step = 64
    for lo in range(0, p, step):
        cache[lo:min(p, lo + step)]
    # Amplification is bounded by cap/(cap - step): re-reading the whole file
    # many times over would show up here immediately.
    assert rd.rows_read < 4 * p, f"read {rd.rows_read:,} rows for a {p:,}-row file"


def test_cache_serves_a_range_wider_than_one_refill_step():
    """The scan asks for [i0, i1 + window), which is wider than B. A cache that
    only ever held B rows would thrash or, worse, return a short view."""
    ref = _packed(300, 3)
    rd = _FakeReader(ref)
    cache = L._PackedRowCache(rd, np.arange(300), 3, cap_rows=128, xp=XP)
    got = np.asarray(cache[10:120])
    np.testing.assert_array_equal(got, ref[10:120])


def test_cache_refuses_a_range_it_cannot_ever_hold():
    """Silently returning a truncated view would corrupt a GEMM. Fail loudly."""
    rd = _FakeReader(_packed(300, 3))
    cache = L._PackedRowCache(rd, np.arange(300), 3, cap_rows=32, xp=XP)
    with pytest.raises(ValueError, match="cap"):
        cache[0:100]


def test_cache_honours_a_row_subset_rather_than_file_positions():
    """rows= may select a subset, so cache[k] must mean the k-th SELECTED row,
    matching _load_packed_rows' contract, not the k-th row of the file."""
    ref = _packed(100, 3)
    sel = np.arange(0, 100, 2)                       # every other variant
    rd = _FakeReader(ref)
    cache = L._PackedRowCache(rd, sel, 3, cap_rows=20, xp=XP)
    got = np.asarray(cache[5:15])
    np.testing.assert_array_equal(got, ref[sel[5:15]])


# ------------------------------------------------------- cap selection

def test_cap_covers_the_widest_range_the_scan_will_ask_for():
    """The scan asks for [i0, i1 + window), so the cap must be at least
    B + window. Anything less makes _PackedRowCache raise mid-scan."""
    B, window, p = 1_280, 5_000, 8_000_000
    cap = L._row_cache_cap(p, B, window, bpv=103_708, free_bytes=14.5e9)
    assert cap >= B + window


def test_cap_never_exceeds_the_number_of_rows():
    cap = L._row_cache_cap(500, 1_280, 5_000, bpv=4, free_bytes=14.5e9)
    assert cap == 500


def test_cap_takes_forward_context_when_memory_allows():
    """Refilling exactly B + window rows per step re-reads the window band every
    time. With room to spare the cap should be larger, so the overlap amortises."""
    B, window = 1_280, 5_000
    tight = L._row_cache_cap(8_000_000, B, window, bpv=103_708, free_bytes=8e9)
    roomy = L._row_cache_cap(8_000_000, B, window, bpv=103_708, free_bytes=60e9)
    assert roomy > tight >= B + window


def test_cap_allows_a_wide_window_above_the_read_ahead_target():
    """The read-ahead target is not a hard ceiling on the live window.

    AoU chr22 on a T4 needs 51,280 packed rows for a 50,000-variant
    window.  That is about 44% of free VRAM: above the normal 25% read-ahead
    target, but still below the 50% hard cache ceiling that leaves 15% after
    the fused path's 35% compute-buffer budget.
    """
    n_samples = 535_662
    bpv = (n_samples + 3) // 4
    tile, window, free = 1_280, 50_000, 15.7e9

    cap = L._row_cache_cap(426_463, tile, window, bpv=bpv, free_bytes=free)

    assert cap >= tile + window
    assert cap * bpv <= 0.50 * free


def test_cap_fits_the_memory_budget_it_is_given():
    bpv, free = 103_708, 14.5e9
    cap = L._row_cache_cap(8_000_000, 1_280, 5_000, bpv=bpv, free_bytes=free)
    assert cap * bpv < free, "the cache alone must not exhaust the device"


def test_cap_refuses_when_even_the_minimum_range_cannot_fit():
    """B + window rows is the floor. If that does not fit, no cap works, and
    saying so beats an OutOfMemoryError with no explanation."""
    with pytest.raises(ValueError, match="window"):
        L._row_cache_cap(8_000_000, 1_280, 5_000, bpv=103_708, free_bytes=1e8)


def test_all_pairs_needs_every_row_from_i0_onward():
    """With window=None the inner loop runs to p, so residency cannot be
    bounded and the caller must fall back to a whole-file load."""
    assert L._row_cache_cap(1000, 128, None, bpv=4, free_bytes=1e12) is None


# ----------------------------------------- end-to-end equality (needs a GPU)

@pytest.fixture
def _small_cugen(tmp_path):
    """A tiny .cugen with no missing calls, so the fused scan accepts it."""
    from cugen.write import CugenWriter, ENCODING_2BIT
    n_samples, n_variants = 64, 400
    rng = np.random.default_rng(7)
    out = str(tmp_path / "t.cugen")
    with CugenWriter(out, n_samples, n_variants, ENCODING_2BIT) as w:
        for v in range(n_variants):
            # A real AF so variants are not monomorphic and survive MAF filters.
            af = rng.uniform(0.1, 0.5)
            d = rng.binomial(2, af, n_samples).astype(np.float64)
            w.add_variant(v, d)
    return out


def _scan(path, out, cap, monkeypatch):
    """Run the scan through the STREAMING path and return (i, j, r).

    It has to be the streaming path. `on_device` requires count_only, a native
    .cugenld stream, or cuDF plus an output -- a plain DataFrame-returning call
    has none of them, so it never reaches _scan_gpu_fused and never touches the
    row cache. An earlier version of these tests did exactly that and compared
    two identical whole-file runs, which is why the companion spy test below
    exists. This is also the path AoU.genome.ld.run_scan uses.
    """
    from cugen import ld as LD
    from cugen.ldio import open_ld
    monkeypatch.setattr(LD, "_ROW_CACHE_MAX_ROWS", cap)
    n = LD.ld_matrix(path, stats=("r2",), window=32, min_r2=0.0,
                     backend="gpu", maf_min=0.01, tile_size=32,
                     stream=True, output=str(out))
    i, j, r = open_ld(str(out)).rows()
    order = np.lexsort((j, i))
    return int(n), i[order], j[order], r[order]


@pytest.mark.parametrize("cap_rows", [64, 96, 130])
def test_a_bounded_cache_gives_bit_identical_results(_small_cugen, cap_rows,
                                                     tmp_path, monkeypatch):
    """The whole point: bounding residency must not change a single value.

    Same tiles, same kernels, same integer accumulation -- only where the bytes
    live changes. Any difference means the cache served the wrong rows.
    """
    pytest.importorskip("cupy", reason="fused scan is GPU-only")
    n_w, i_w, j_w, r_w = _scan(_small_cugen, tmp_path / "whole.cugenld",
                               None, monkeypatch)
    n_b, i_b, j_b, r_b = _scan(_small_cugen, tmp_path / f"blk{cap_rows}.cugenld",
                               cap_rows, monkeypatch)
    assert n_w == n_b, f"row counts differ: {n_w:,} vs {n_b:,}"
    np.testing.assert_array_equal(i_w, i_b)
    np.testing.assert_array_equal(j_w, j_b)
    np.testing.assert_array_equal(r_w, r_b)          # bit-identical, not close


def test_the_cache_path_is_actually_taken_when_forced(_small_cugen, tmp_path,
                                                      monkeypatch):
    """Guards against the equality test passing vacuously because the cap was
    ignored and the whole file loaded on both sides. It caught exactly that."""
    pytest.importorskip("cupy", reason="fused scan is GPU-only")
    from cugen import ld as LD

    made = []
    real = LD._PackedRowCache

    class _Spy(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(LD, "_PackedRowCache", _Spy)
    _scan(_small_cugen, tmp_path / "spy.cugenld", 64, monkeypatch)
    assert made, "the bounded cache was never constructed"
    assert made[0].refills >= 2, "a 64-row cap over 400 rows must refill"


def test_the_whole_file_path_is_taken_when_not_forced(_small_cugen, tmp_path,
                                                      monkeypatch):
    """The other half of the guard: the baseline arm must NOT be using the
    cache, or 'blocked vs whole' compares two blocked runs."""
    pytest.importorskip("cupy", reason="fused scan is GPU-only")
    from cugen import ld as LD

    made = []
    real = LD._PackedRowCache

    class _Spy(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(LD, "_PackedRowCache", _Spy)
    _scan(_small_cugen, tmp_path / "base.cugenld", None, monkeypatch)
    assert not made, "the baseline arm must load the whole file"
