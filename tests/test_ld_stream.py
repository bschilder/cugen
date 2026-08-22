"""Streaming the epilogue: peak VRAM must not scale with the result.

_scan_gpu_fused materialised the entire result in device memory, so peak
memory grew as O(p^2 * f) while every other stage is O(p) or O(B*n). The
min(200e6, ...) cap was a symptom: chr22's full variant set alone emitted
187,252,868 rows against it, 6.4% of headroom, and genome-wide MAF>=1%
extrapolates to ~3.2e10 rows = 596 GB, past both the cap and the card.

A tile emits at most bi*bj <= B*B rows, so a buffer of >= 2*B*B that is
flushed whenever the count comes within B*B of capacity can never overflow.
That makes the old "re-run the entire O(p^2) GEMM at exact size" retry
impossible rather than merely unlikely.
"""
import numpy as np
import pandas as pd
import pytest

import cugen.ld as L
from conftest import requires_cudf, requires_gpu, simulate_haplotypes

# p > tile_size so the scan actually tiles; otherwise one tile holds
# everything and no flush boundary is ever crossed.
TILE = 256


def _fixture(write_cugen_file, n=300, p=900, seed=21):
    return write_cugen_file(simulate_haplotypes(n, p, seed=seed))


def _rows(path):
    df = pd.read_csv(path, sep="\t")
    key = [c for c in ("gidx_a", "gidx_b") if c in df.columns]
    return df.sort_values(key).reset_index(drop=True), key


@requires_gpu
@requires_cudf
def test_streamed_file_holds_exactly_the_unstreamed_rows(
        tmp_path, write_cugen_file):
    """Streaming may reorder rows; it may not add, drop or alter any."""
    path = _fixture(write_cugen_file)
    ref = tmp_path / "ref.tsv"
    df = L.ld_matrix(path, min_r2=0.05, stats=("r", "r2"),
                     sign_reference="major", output=str(ref),
                     tile_size=TILE, max_pairs=10 ** 12, verbose=False)
    got = tmp_path / "stream.tsv"
    n = L.ld_matrix(path, min_r2=0.05, stats=("r", "r2"),
                    sign_reference="major", output=str(got), stream=True,
                    flush_rows=1 << 15, tile_size=TILE,
                    max_pairs=10 ** 12, verbose=False)
    assert n == len(df), f"streamed {n} rows, unstreamed {len(df)}"
    a, key = _rows(str(ref))
    b, _ = _rows(str(got))
    assert len(a) == len(b) == n
    pd.testing.assert_frame_equal(a[key], b[key])
    np.testing.assert_allclose(a["R"].to_numpy(), b["R"].to_numpy(), atol=0)


@requires_gpu
@requires_cudf
def test_stream_flushes_more_than_once(tmp_path, write_cugen_file, monkeypatch):
    """If it only ever flushed at the end this would all be theatre."""
    path = _fixture(write_cugen_file)
    calls = []
    real = L._ChunkWriter.write

    def spy(self, df):
        calls.append(len(df))
        return real(self, df)

    monkeypatch.setattr(L._ChunkWriter, "write", spy)
    n = L.ld_matrix(path, min_r2=0.05, stats=("r", "r2"),
                    sign_reference="major", output=str(tmp_path / "s.tsv"),
                    stream=True, flush_rows=1 << 15, tile_size=TILE,
                    max_pairs=10 ** 12, verbose=False)
    assert len(calls) > 1, f"only {len(calls)} flush(es) for {n:,} rows"
    assert sum(calls) == n, f"flushes summed to {sum(calls)}, count said {n}"


@requires_gpu
@requires_cudf
def test_stream_loses_nothing_with_the_smallest_legal_buffer(
        tmp_path, write_cugen_file):
    """The old path DROPPED rows past capacity and then re-ran everything.

    Asking for an absurdly small buffer must clamp to the safe size and
    still return every row, not silently truncate.
    """
    path = _fixture(write_cugen_file)
    ref = L.ld_matrix(path, min_r2=0.05, stats=("r", "r2"),
                      sign_reference="major", output=str(tmp_path / "r.tsv"),
                      tile_size=TILE, max_pairs=10 ** 12, verbose=False)
    n = L.ld_matrix(path, min_r2=0.05, stats=("r", "r2"),
                    sign_reference="major", output=str(tmp_path / "t.tsv"),
                    stream=True, flush_rows=1, tile_size=TILE,
                    max_pairs=10 ** 12, verbose=False)
    assert n == len(ref)


@requires_gpu
@requires_cudf
def test_stream_peak_allocation_does_not_track_the_result_size(
        tmp_path, write_cugen_file, monkeypatch):
    """The invariant the whole change exists to establish."""
    import cupy as cp
    path = _fixture(write_cugen_file)
    sizes, real_empty = [], cp.empty

    def spy(shape, *a, **k):
        sizes.append(shape if isinstance(shape, int)
                     else int(np.prod(shape)) if shape else 0)
        return real_empty(shape, *a, **k)

    monkeypatch.setattr(cp, "empty", spy)

    def peak_for(min_r2, name):
        sizes.clear()
        n = L.ld_matrix(path, min_r2=min_r2, stats=("r", "r2"),
                        sign_reference="major",
                        output=str(tmp_path / name), stream=True,
                        tile_size=TILE, max_pairs=10 ** 12, verbose=False)
        return n, max(sizes)

    n_loose, peak_loose = peak_for(0.05, "a.tsv")
    n_tight, peak_tight = peak_for(0.9, "b.tsv")
    assert n_tight * 5 < n_loose, (
        f"{n_tight:,} vs {n_loose:,} survivors -- spread too small to test")
    assert peak_loose == peak_tight, (
        f"peak allocation moved {peak_tight:,} -> {peak_loose:,} elements when "
        f"survivors went {n_tight:,} -> {n_loose:,}; something tracks the result")


@requires_gpu
def test_stream_requires_an_output_path(tmp_path, write_cugen_file):
    """Streaming with nowhere to stream to is a programming error."""
    path = _fixture(write_cugen_file, p=300)
    with pytest.raises(ValueError, match="stream"):
        L.ld_matrix(path, min_r2=0.2, stats=("r", "r2"),
                    sign_reference="major", output=None, stream=True,
                    max_pairs=10 ** 12, verbose=False)
