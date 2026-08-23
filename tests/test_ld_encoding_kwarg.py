"""`ld_encoding=` on ld_matrix: reach float32 .cugenld from the scan.

int16 is the right default -- measured 1,310x tighter than the sampling SE of
r at n = 2,504, so for analysis, clumping, fine-mapping and visualisation it is
not a compromise. But its quantum is 3.05e-5, which is 29x COARSER than the
5.3e-7 cugen/plink2 agreement floor, so the one thing int16 cannot do is
measure agreement between implementations. Quantising a parity benchmark
silently is the failure mode to avoid.

LDDatasetWriter and write_ld both took `encoding=` already; ld_matrix did not,
so float32 was unreachable from the scan -- which left TSV as the only lossless
option and cost 78 B/pair for three columns of information.
"""
import numpy as np
import pytest

from cugen import ld as L
from cugen import ldio
from conftest import requires_gpu, simulate_haplotypes
from cugen.write import write_cugen

CPU = dict(backend="numpy", verbose=False)


@pytest.fixture
def panel(tmp_path):
    dos = simulate_haplotypes(200, 60, seed=19).T.astype(np.float32)
    p = tmp_path / "e.cugen"
    write_cugen(str(p), dos)
    return str(p)


def _by_pair(t):
    i, j, r = t
    return dict(zip(zip(i.tolist(), j.tolist()), r.tolist()))


def test_float32_round_trips_exactly(panel, tmp_path):
    """The whole point: lossless, so a parity benchmark is not quantised."""
    ref = L.ld_matrix(panel, stats=("r", "r2"), **CPU)
    out = tmp_path / "f32.cugenld"
    L.ld_matrix(panel, stats=("r", "r2"), output=str(out),
                ld_encoding="float32", **CPU)

    rd = ldio.read_ld(str(out))
    assert rd.encoding == "float32"
    got = _by_pair(rd.rows())
    want = dict(zip(zip(ref["gidx_a"], ref["gidx_b"]), ref["R"]))
    assert set(got) == set(want)
    for k, w in want.items():
        # float32 storage of a float32 r: bit-exact, not merely close
        assert np.float32(got[k]) == np.float32(w), k


def test_int16_is_still_the_default_and_is_lossy_at_its_quantum(panel, tmp_path):
    ref = L.ld_matrix(panel, stats=("r", "r2"), **CPU)
    out = tmp_path / "i16.cugenld"
    L.ld_matrix(panel, stats=("r", "r2"), output=str(out), **CPU)

    rd = ldio.read_ld(str(out))
    assert rd.encoding == "int16"
    got = _by_pair(rd.rows())
    want = dict(zip(zip(ref["gidx_a"], ref["gidx_b"]), ref["R"]))
    err = max(abs(got[k] - w) for k, w in want.items())
    # within a half-quantum, and NOT exact -- if this became exact the encoding
    # silently changed under us
    assert err <= 1.0 / 32767 + 1e-12
    assert err > 0.0, "int16 returned exact values; is it really int16?"


def test_int8_is_reachable_and_coarser_still(panel, tmp_path):
    out = tmp_path / "i8.cugenld"
    L.ld_matrix(panel, stats=("r", "r2"), output=str(out),
                ld_encoding="int8", **CPU)
    assert ldio.read_ld(str(out)).encoding == "int8"


def test_the_encoding_is_recorded_so_a_reader_need_not_guess(panel, tmp_path):
    for enc in ("int16", "float32", "int8"):
        out = tmp_path / f"rec_{enc}.cugenld"
        L.ld_matrix(panel, stats=("r", "r2"), output=str(out),
                    ld_encoding=enc, **CPU)
        assert ldio.read_ld(str(out)).encoding == enc


def test_an_unknown_encoding_is_refused(panel, tmp_path):
    with pytest.raises(ValueError, match="encoding"):
        L.ld_matrix(panel, stats=("r", "r2"),
                    output=str(tmp_path / "bad.cugenld"),
                    ld_encoding="float16", **CPU)


def test_float32_costs_more_bytes_than_int16(panel, tmp_path):
    """Sanity: the lossless option is bigger, or it is not really float32."""
    import os
    a = tmp_path / "cmp16.cugenld"
    b = tmp_path / "cmp32.cugenld"
    L.ld_matrix(panel, stats=("r", "r2"), output=str(a), **CPU)
    L.ld_matrix(panel, stats=("r", "r2"), output=str(b),
                ld_encoding="float32", **CPU)
    assert os.path.getsize(b) > os.path.getsize(a)


@requires_gpu
def test_streaming_honours_the_encoding(panel, tmp_path):
    """The streaming branch is the one a genome-scale run takes."""
    out = tmp_path / "stream.cugenld"
    n = L.ld_matrix(panel, stats=("r", "r2"), output=str(out), stream=True,
                    ld_encoding="float32", backend="gpu", verbose=False)
    assert n > 0
    import os
    ds = ldio.open_ld(str(out))
    assert ds.encoding == "float32", "the manifest did not record float32"
    # and the shards themselves, not just the manifest that describes them
    encs = {ldio.read_ld(os.path.join(str(out), sh["file"])).encoding
            for sh in ds.shards}
    assert encs == {"float32"}, f"streaming wrote {encs}, not float32"
