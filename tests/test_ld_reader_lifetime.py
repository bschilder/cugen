"""Reader-ownership regressions for the public LD API."""

from __future__ import annotations

import numpy as np
import pytest

from cugen import ld as L
from cugen.write import ENCODING_FLOAT32, write_cugen

CPU = {
    "backend": "numpy",
    "stats": ("r", "r2"),
    "precision": "fp32",
    "verbose": False,
}


@pytest.fixture
def synthetic_cugen(tmp_path):
    path = tmp_path / "synthetic.cugen"
    calls = np.array([[0, 0], [1, 2], [2, 1]], dtype=np.uint8)
    write_cugen(str(path), calls, gidx=[30, 10], encoding=0)
    return path


@pytest.fixture
def opened_readers(monkeypatch):
    """Capture real readers constructed by the public function."""
    real_read_cugen = L.read_cugen
    readers = []

    def tracked_read_cugen(*args, **kwargs):
        reader = real_read_cugen(*args, **kwargs)
        readers.append(reader)
        return reader

    monkeypatch.setattr(L, "read_cugen", tracked_read_cugen)
    yield readers
    for reader in readers:
        if not reader._mmap.closed:
            reader.close()


def assert_reader_closed(readers):
    assert len(readers) == 1
    assert (readers[0]._file.closed, readers[0]._mmap.closed) == (True, True)


def test_ld_matrix_closes_reader_and_result_remains_usable(
    synthetic_cugen, opened_readers
):
    result = L.ld_matrix(synthetic_cugen, **CPU)

    assert_reader_closed(opened_readers)
    assert result[["gidx_a", "gidx_b"]].to_dict("records") == [
        {"gidx_a": 30, "gidx_b": 10}
    ]
    np.testing.assert_allclose(result["R"], [0.5], rtol=0, atol=1e-7)
    np.testing.assert_allclose(result["R2"], [0.25], rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variant_range": (0, 0)},
        {"variant_range": (0, 1)},
        {"min_r2": 1.01},
    ],
    ids=["no-selected-rows", "no-planned-pairs", "no-surviving-pairs"],
)
def test_ld_matrix_closes_reader_for_empty_results(
    synthetic_cugen, opened_readers, kwargs
):
    result = L.ld_matrix(synthetic_cugen, **CPU, **kwargs)

    assert_reader_closed(opened_readers)
    assert result.empty
    assert list(result.columns) == list(L._empty_pairs(CPU["stats"]).columns)


def test_ld_matrix_closes_reader_when_encoding_is_rejected(tmp_path, opened_readers):
    path = tmp_path / "float32.cugen"
    calls = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    write_cugen(str(path), calls, gidx=[30, 10], encoding=ENCODING_FLOAT32)

    with pytest.raises(NotImplementedError) as raised:
        L.ld_matrix(path, **CPU)

    assert str(raised.value) == (
        "cugen.ld requires 2-bit encoding (encoding=0); this file has encoding=3. "
        "Re-convert with cg.convert."
    )
    assert_reader_closed(opened_readers)


def test_ld_matrix_closes_reader_when_pair_limit_is_rejected(
    synthetic_cugen, opened_readers
):
    with pytest.raises(ValueError) as raised:
        L.ld_matrix(synthetic_cugen, max_pairs=0, **CPU)

    assert str(raised.value) == (
        "plan would emit 1 pairs, above max_pairs=0. Narrow it with window= or "
        "window_kb=, raise min_r2, or raise max_pairs if you really want this."
    )
    assert_reader_closed(opened_readers)


def test_ld_matrix_closes_reader_and_preserves_scan_exception(
    synthetic_cugen, opened_readers, monkeypatch
):
    failure = RuntimeError("synthetic scan failure")

    def fail_scan(*args, **kwargs):
        raise failure

    monkeypatch.setattr(L, "contingency_tables", fail_scan)

    with pytest.raises(RuntimeError) as raised:
        L.ld_matrix(synthetic_cugen, **CPU)

    assert raised.value is failure
    assert_reader_closed(opened_readers)


def test_ld_matrix_closes_reader_and_preserves_output_exception(
    synthetic_cugen, opened_readers, monkeypatch, tmp_path
):
    failure = OSError("synthetic output failure")

    def fail_write(*args, **kwargs):
        raise failure

    monkeypatch.setattr(L, "_write_df", fail_write)

    with pytest.raises(OSError) as raised:
        L.ld_matrix(synthetic_cugen, output=tmp_path / "ld.tsv", **CPU)

    assert raised.value is failure
    assert_reader_closed(opened_readers)
