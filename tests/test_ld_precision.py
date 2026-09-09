"""Regression cases for cancellation in the nonfused GPU LD epilogues."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from conftest import requires_gpu

from cugen import ld as L
from cugen.write import write_cugen

# Independent integer-moment results, not values from either CuGen backend.
CASES = [
    (3072, False, -0.0004605808764446619),
    (3072, True, 0.706991645342556),
    (4096, False, -0.0003453934724429183),
    (4096, True, 0.7070204380906536),
]


def _near_fixed(n, overlap, missing, flips):
    calls = np.full((n, 2), 2, dtype=np.uint8)
    calls[0, 0] = 1
    calls[[0, 1] if overlap else [1, 2], 1] = 1
    if missing:
        # These people are not jointly observed; n and the correlation stay fixed.
        calls = np.vstack((calls, np.array([[3, 2], [2, 3], [3, 3]], dtype=np.uint8)))
    for col in flips:
        called = calls[:, col] != 3
        calls[called, col] = 2 - calls[called, col]
    return calls


def _run(tmp_path, calls, gidx, *, tile, stats=("r", "r2"), min_r2=0):
    path = tmp_path / "synthetic.cugen"
    write_cugen(path, calls, gidx=gidx, encoding=0)
    annotation = pd.DataFrame({
        "gidx": gidx, "CHR": ["1"] * len(gidx),
        "POS": [101 + 100 * i for i in range(len(gidx))],
        "ID": [f"synthetic-{i}" for i in range(len(gidx))],
    })
    return L.ld_matrix(
        path, annotation=annotation, backend="gpu", precision="fp32",
        stats=stats, dprime_method="composite", sign_reference="alt",
        missing="pairwise", min_obs=2, maf_min=0, min_r2=min_r2,
        window=None, window_kb=None, output_format="pairs", output=None,
        tile_size=tile, max_pairs=21, verbose=False,
    )


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("n,overlap,want", CASES)
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("flips", [(), (0,), (1,), (0, 1)])
@pytest.mark.parametrize("tile", [1, 3])
def test_near_fixed_r_only_preserves_correlation(tmp_path, n, overlap, want,
                                                missing, flips, tile):
    """FP32 raw-moment subtraction loses covariance; flips must not cure a bug."""
    calls = _near_fixed(n, overlap, missing, flips)
    result = _run(tmp_path, calls, [30, 10], tile=tile)
    assert len(result) == 1
    row = result.iloc[0]
    assert (row["gidx_a"], row["gidx_b"], row["N_OBS"]) == (30, 10, n)
    assert (row["CHR_A"], row["CHR_B"]) == (1, 1)
    assert (row["POS_A"], row["POS_B"]) == (101, 201)
    assert (row["ID_A"], row["ID_B"]) == ("synthetic-0", "synthetic-1")
    signed = -want if len(flips) == 1 else want
    assert row["R"] == pytest.approx(signed, rel=0, abs=1e-5)
    assert row["R2"] == pytest.approx(want * want, rel=0, abs=2e-5)
    assert result["R"].dtype == np.dtype("float32")
    assert result["R2"].dtype == np.dtype("float32")


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("n", [3072, 4096])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("flips", [(), (0, 1)])
@pytest.mark.parametrize("tile", [1, 3])
def test_table_prefilter_retains_small_nonzero_correlation(tmp_path, n, missing,
                                                          flips, tile):
    """An FP32 zero must not discard a pair before the FP64 host calculation."""
    calls = _near_fixed(n, False, missing, flips)
    # Requesting D exercises the contingency-table prefilter, not the r-only path.
    result = _run(tmp_path, calls, [30, 10], tile=tile, stats=("r", "r2", "d"),
                  min_r2=1e-7)
    assert len(result) == 1
    row = result.iloc[0]
    assert (row["gidx_a"], row["gidx_b"], row["N_OBS"]) == (30, 10, n)
    want = -2 / math.sqrt((n - 1) * (2 * n - 4))
    assert row["R"] == pytest.approx(want, rel=0, abs=1e-5)
    assert row["R2"] == pytest.approx(want * want, rel=0, abs=2e-5)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("tile", [1, 3])
@pytest.mark.parametrize("stats", [("r", "r2"), ("r", "r2", "d")])
def test_moment_precision_preserves_missing_and_undefined_pairs(tmp_path, tile, stats):
    """Higher precision must preserve pairwise counts and exclude undefined LD."""
    calls = np.array([
        [2, 0, 2, 3, 1, 1, 1], [0, 0, 2, 3, 1, 0, 3],
        [2, 2, 0, 3, 1, 2, 3], [1, 1, 1, 3, 1, 3, 3],
    ], dtype=np.uint8)
    expected = {
        (30, 10): (4, 5 / 11), (30, 70): (4, -5 / 11),
        (30, 40): (3, math.sqrt(3) / 2), (10, 70): (4, -1),
        (10, 40): (3, math.sqrt(3) / 2), (70, 40): (3, -math.sqrt(3) / 2),
    }
    result = _run(tmp_path, calls, [30, 10, 70, 20, 60, 40, 50],
                  tile=tile, stats=stats)
    assert len(result) == len(expected)
    assert set(zip(result["gidx_a"], result["gidx_b"])) == set(expected)
    for row in result.to_dict(orient="records"):
        n, r = expected[(row["gidx_a"], row["gidx_b"])]
        assert row["N_OBS"] == n
        assert row["R"] == pytest.approx(r, rel=0, abs=1e-5)
        assert row["R2"] == pytest.approx(r * r, rel=0, abs=2e-5)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("stats", [("r", "r2"), ("r", "r2", "d")])
def test_moment_precision_at_accumulation_limit(tmp_path, missing, overlap, stats):
    """Exact raw sums do not suffice: their products need more than 24 bits."""
    # At this source size 4*n is still an exactly representable FP32 integer.
    n = 4_194_304 - (3 if missing else 0)
    calls = _near_fixed(n, overlap, missing, ())
    result = _run(tmp_path, calls, [30, 10], tile=2, stats=stats, min_r2=1e-14)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["N_OBS"] == n
    numerator = n - 2 if overlap else -2
    want = numerator / math.sqrt((n - 1) * (2 * n - 4))
    assert row["R"] == pytest.approx(want, rel=0, abs=1e-5 if overlap else 1e-12)
    assert row["R2"] == pytest.approx(want * want, rel=0, abs=2e-5)
