"""Fixtures for the cugen test suite.

Everything here is CPU-only and network-free: .cugen files are synthesised in
tmp_path via cugen.write, which needs no GPU.
"""
import numpy as np
import pytest

from cugen.write import write_cugen

try:
    import cupy  # noqa: F401
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

requires_gpu = pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not available")

try:
    import cudf  # noqa: F401
    _HAS_CUDF = True
except ImportError:
    _HAS_CUDF = False

# The fused epilogue is only selected when cuDF is present (ld_matrix gates
# `on_device` on it), so a test that ASSERTS the fused path ran cannot run
# without cuDF. Skipping is the honest outcome there -- failing would report a
# missing optional dependency as a code defect.
requires_cudf = pytest.mark.skipif(
    not (_HAS_CUPY and _HAS_CUDF), reason="CuPy+cuDF required for fused path")


def simulate_haplotypes(n_samples, n_variants, seed=0, missing_rate=0.0,
                        straddle_half=True):
    """Genotypes with a spread of LD and MAF.

    straddle_half puts ALT frequency above 0.5 for half the variants. Without
    that the 'alt' and 'major' sign conventions coincide and any orientation
    test passes vacuously -- a real defect we hit while building this.
    """
    rng = np.random.default_rng(seed)
    latent = rng.random(n_samples)
    dos = np.zeros((n_variants, n_samples), dtype=np.uint8)
    for v in range(n_variants):
        mix = rng.uniform(0.0, 0.95)
        lat = mix * latent + (1 - mix) * rng.random(n_samples)
        if straddle_half and v % 2 == 0:
            af = rng.uniform(0.55, 0.92)
        else:
            af = rng.uniform(0.08, 0.45)
        t1, t2 = np.quantile(lat, [(1 - af) ** 2, 1 - af ** 2])
        dos[v] = (lat > t1).astype(np.uint8) + (lat > t2).astype(np.uint8)
    if missing_rate > 0:
        mask = rng.random((n_variants, n_samples)) < missing_rate
        dos[mask] = 3
    return dos


@pytest.fixture
def dosages():
    """(n_variants, n_samples) uint8, 3 = missing."""
    return simulate_haplotypes(200, 12, seed=11, missing_rate=0.0)


@pytest.fixture
def dosages_missing():
    d = simulate_haplotypes(200, 12, seed=12, missing_rate=0.0)
    rng = np.random.default_rng(5)
    d[3, rng.choice(200, 30, replace=False)] = 3
    d[7, rng.choice(200, 45, replace=False)] = 3
    return d


def _write(tmp_path, dos, name="chr22.cugen"):
    # write_cugen takes (n_samples, n_variants) -- transpose in.
    p = tmp_path / name
    write_cugen(str(p), dos.T.astype(np.uint8))
    return str(p)


@pytest.fixture
def small_cugen(tmp_path, dosages):
    return _write(tmp_path, dosages), dosages


@pytest.fixture
def missing_cugen(tmp_path, dosages_missing):
    return _write(tmp_path, dosages_missing), dosages_missing


@pytest.fixture
def write_cugen_file(tmp_path):
    def _f(dos, name="chr22.cugen"):
        return _write(tmp_path, dos, name)
    return _f
