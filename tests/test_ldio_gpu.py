"""Device-side .cugenld encoding, byte-identical to the host encoder.

Why byte-identical is the acceptance criterion rather than "round-trips":
the format is already written and read in production, so a device encoder that
produced merely *equivalent* bytes would fork the format. Identical bytes mean
the reader needs no change, existing files stay valid, and the two encoders can
be swapped per call without a compatibility matrix.

Why this is worth doing at all: measured on one real flush of 16,677,861
survivors, .cugenld holds 2.64 B/pair against the 62.18 B/pair cugen writes as
CSV -- 23.5x smaller -- but takes 2.447 s against 0.399 s, because the encoder
runs on the host. Writing its 44 MB at the 2.8 GB/s the raw path achieves would
take 0.016 s, so 99.4% of that 2.447 s is host encoding, not I/O.
"""
import numpy as np
import pytest

from cugen import ldio
from conftest import requires_gpu


def _sorted_pairs(n_rows=400, per_row=40, seed=5):
    """(i, j, r) sorted by (i, j), j ascending within each row variant."""
    rng = np.random.default_rng(seed)
    i, j = [], []
    for v in range(n_rows):
        k = rng.integers(1, per_row)
        js = np.sort(rng.choice(np.arange(v + 1, v + 1 + 4 * per_row),
                                size=k, replace=False))
        i.append(np.full(k, v, dtype=np.int64))
        j.append(js.astype(np.int64))
    i = np.concatenate(i)
    j = np.concatenate(j)
    r = rng.uniform(-1, 1, size=i.size).astype(np.float32)
    return i, j, r


@requires_gpu
def test_run_starts_gpu_matches_host():
    import cupy as cp
    i, _, _ = _sorted_pairs()
    want = ldio._run_starts(i)
    got = cp.asnumpy(ldio._run_starts_gpu(cp.asarray(i)))
    np.testing.assert_array_equal(want, got)


@requires_gpu
def test_tier_of_gpu_matches_host():
    """Tier assignment decides which block a pair lands in, so a single
    off-by-one would silently reorder the file rather than corrupt it."""
    import cupy as cp
    rng = np.random.default_rng(11)
    r2 = rng.uniform(0, 1, size=100_000)
    # include exact tier edges: comparisons at the boundary are where a
    # `<` vs `<=` disagreement between two implementations would show up
    r2[:len(ldio.DEFAULT_TIERS)] = np.asarray(ldio.DEFAULT_TIERS)
    want = ldio._tier_of(r2, ldio.DEFAULT_TIERS)
    got = cp.asnumpy(ldio._tier_of_gpu(cp.asarray(r2), ldio.DEFAULT_TIERS))
    np.testing.assert_array_equal(want, got)


@requires_gpu
@pytest.mark.parametrize("encoding", ["int16", "float32", "int8"])
def test_encode_block_gpu_is_byte_identical_to_host(encoding):
    import cupy as cp
    i, j, r = _sorted_pairs()
    rs = ldio._run_starts(i)
    blob_h, meta_h = ldio.encode_block(j, r, encoding=encoding, row_starts=rs)
    blob_d, meta_d = ldio.encode_block_gpu(
        cp.asarray(j), cp.asarray(r), encoding=encoding,
        row_starts=cp.asarray(rs))
    assert blob_d == blob_h, "compressed bytes differ"
    assert meta_d == meta_h, f"metadata differs: {meta_d} vs {meta_h}"


@requires_gpu
def test_encode_block_gpu_round_trips_through_the_host_decoder():
    """The bytes must be readable by the existing decoder, unchanged."""
    import cupy as cp
    i, j, r = _sorted_pairs()
    rs = ldio._run_starts(i)
    blob, meta = ldio.encode_block_gpu(cp.asarray(j), cp.asarray(r),
                                       row_starts=cp.asarray(rs))
    # decode_block needs row_starts in meta to undo the per-row delta reset;
    # the writer supplies it, encode_block does not put it there.
    meta = dict(meta, row_starts=rs.tolist())
    gj, gr = ldio.decode_block(blob, meta)[:2]
    np.testing.assert_array_equal(np.asarray(gj, dtype=np.int64), j)
    # int16 quantisation: half a quantum, and no more
    assert np.abs(np.asarray(gr) - r).max() <= 0.5 / 32767 + 1e-9


@requires_gpu
def test_encode_block_gpu_carries_the_n_deficit():
    import cupy as cp
    i, j, r = _sorted_pairs(n_rows=60)
    rs = ldio._run_starts(i)
    nd = np.random.default_rng(3).integers(0, 500, size=i.size).astype(np.int64)
    bh, mh = ldio.encode_block(j, r, row_starts=rs, n_deficit=nd)
    bd, md = ldio.encode_block_gpu(cp.asarray(j), cp.asarray(r),
                                   row_starts=cp.asarray(rs),
                                   n_deficit=cp.asarray(nd))
    assert bd == bh and md == mh


@requires_gpu
def test_encode_block_gpu_refuses_a_negative_gap():
    """Same guard as the host: a descending j would wrap in an unsigned width
    and decode to a different variant entirely, silently."""
    import cupy as cp
    j = np.array([10, 5, 20], dtype=np.int64)      # 5 < 10 inside one row
    r = np.array([0.5, 0.4, 0.3], dtype=np.float32)
    with pytest.raises(ValueError, match="ascend"):
        ldio.encode_block_gpu(cp.asarray(j), cp.asarray(r),
                              row_starts=cp.zeros(1, dtype=cp.int64))


@requires_gpu
def test_gpu_written_shard_is_byte_identical_to_host_written(tmp_path):
    """The whole pipeline, not just encode_block.

    This is the claim the device path has to earn: a shard written from device
    arrays must be the same file as one written from host arrays. Anything less
    forks the format -- the reader, the manifest and every file already on disk
    would then depend on which encoder produced them.
    """
    import cupy as cp
    i, j, r = _sorted_pairs(n_rows=9000, per_row=25, seed=17)
    dh, dd = tmp_path / "h.cugenld", tmp_path / "d.cugenld"

    wh = ldio.LDDatasetWriter(str(dh), params={})
    wh.write_shard((0, 0), i, j, r, presorted=True)
    wh.mark_complete()
    wh.close()

    wd = ldio.LDDatasetWriter(str(dd), params={})
    wd.write_shard_gpu((0, 0), cp.asarray(i), cp.asarray(j), cp.asarray(r))
    wd.mark_complete()
    wd.close()

    bh = (dh / "0-0.ldz").read_bytes()
    bd = (dd / "0-0.ldz").read_bytes()
    assert len(bd) == len(bh), f"shard sizes differ: {len(bd)} vs {len(bh)}"
    assert bd == bh, "shard bytes differ between the host and device encoders"


@requires_gpu
def test_gpu_written_shard_reads_back_correctly(tmp_path):
    """And the existing reader, untouched, must return the input."""
    import cupy as cp
    i, j, r = _sorted_pairs(n_rows=3000, per_row=20, seed=23)
    d = tmp_path / "d.cugenld"
    w = ldio.LDDatasetWriter(str(d), params={})
    w.write_shard_gpu((0, 0), cp.asarray(i), cp.asarray(j), cp.asarray(r))
    w.mark_complete()
    w.close()
    gi, gj, gr = ldio.read_ld(str(d / "0-0.ldz")).rows()
    order = np.lexsort((np.asarray(gj), np.asarray(gi)))
    np.testing.assert_array_equal(np.asarray(gi)[order], i)
    np.testing.assert_array_equal(np.asarray(gj)[order], j)
    assert np.abs(np.asarray(gr)[order] - r).max() <= 0.5 / 32767 + 1e-9


@requires_gpu
@pytest.mark.parametrize("mbp", [1 << 16, 1 << 22])
def test_gpu_shard_is_byte_identical_at_any_block_size(tmp_path, mbp):
    """Byte identity must not depend on max_block_pairs.

    Per-block overhead was the whole wall: at the 65,536 default a 368 M-row
    scan became 5,701 blocks and 26.05 s, while 4,194,304 gave 12.37 s AND
    1.95 B/pair against 2.05 -- fewer footer entries. Since that parameter is
    now worth tuning, the device encoder has to agree with the host at every
    setting, not just the default.
    """
    import cupy as cp
    i, j, r = _sorted_pairs(n_rows=6000, per_row=30, seed=41)
    dh, dd = tmp_path / f"h{mbp}.cugenld", tmp_path / f"d{mbp}.cugenld"
    wh = ldio.LDDatasetWriter(str(dh), params={}, max_block_pairs=mbp)
    wh.write_shard((0, 0), i, j, r, presorted=True)
    wh.mark_complete(); wh.close()
    wd = ldio.LDDatasetWriter(str(dd), params={}, max_block_pairs=mbp)
    wd.write_shard_gpu((0, 0), cp.asarray(i), cp.asarray(j), cp.asarray(r))
    wd.mark_complete(); wd.close()
    assert (dd / "0-0.ldz").read_bytes() == (dh / "0-0.ldz").read_bytes()
