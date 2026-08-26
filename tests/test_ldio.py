"""On-disk LD results: the .cugenld format, and the writer/reader registry.

The design target is genome-wide all-by-all, which is 10^9 to 10^13 emitted
pairs depending only on the retention threshold -- and the threshold is set by
the number of tests and the sample size, neither of which the format controls.
Bytes per pair is what the format controls, and at 10^12 rows it is the
difference between 7.7 TB and 178 TB. See the design notes at the head of
cugen/ldio.py.

Everything here is checked against an independent implementation or against
brute force, never against the production path restated.
"""
import numpy as np
import pytest

from cugen import ldio
from conftest import requires_cudf


# ------------------------------------------------------------------ encodings
# Half a quantisation step, not a whole one: rounding cannot exceed 0.5 steps
# while truncation reaches a full step, so a loose bound here would accept a
# truncating implementation. The mutation sweep caught exactly that.
@pytest.mark.parametrize("encoding,tol", [
    ("int16", 0.51 / 32767),
    ("int8", 0.51 / 127),
    # float32 is the "lossless" option in the sense that it stores r as given,
    # but it is not bit-exact against a float64 source: 1e-6 and 0.9999 are not
    # representable, so the bound is float32's own relative precision.
    ("float32", float(np.finfo(np.float32).eps)),
])
def test_r_round_trips_within_the_encoding_resolution(encoding, tol):
    """r lives in [-1, 1], so a fixed-point scale is the right quantisation.

    int16 gives uniform ~3e-5 resolution, which is finer than the six
    significant figures of plink2's own text output that cugen is validated
    against, and two to three orders below LD's own estimation error.
    """
    r = np.array([-1.0, -0.9999, -0.5, -1e-6, 0.0, 1e-6, 0.5, 0.9999, 1.0],
                 dtype=np.float64)
    back = ldio.dequantize_r(ldio.quantize_r(r, encoding), encoding)
    assert np.abs(back - r).max() <= tol + 1e-12
    # the endpoints must be exact -- r = +/-1 is a real value, not a rounding
    assert back[0] == pytest.approx(-1.0, abs=tol + 1e-12)
    assert back[-1] == pytest.approx(1.0, abs=tol + 1e-12)


def test_quantisation_is_monotone():
    """Ordering by stored r must equal ordering by true r, or every
    threshold query and every zone map silently returns the wrong set."""
    r = np.linspace(-1.0, 1.0, 20001)
    q = ldio.quantize_r(r, "int16")
    assert (np.diff(q.astype(np.int64)) >= 0).all()


def test_int16_is_the_declared_default():
    assert ldio.DEFAULT_ENCODING == "int16"


def test_an_unknown_encoding_is_refused():
    with pytest.raises(ValueError, match="encoding"):
        ldio.quantize_r(np.zeros(3), "float16")


# --------------------------------------------------------------------- header
def test_header_round_trips_every_field():
    fields = dict(version=1, encoding="int16", payload="banded",
                  n_row_variants=12345, n_pairs=9_876_543_210_123,
                  index_offset=256, blocks_offset=4096,
                  footer_offset=1 << 40, r_scale=32767.0,
                  block_a=7, block_b=9, flags=0b101)
    buf = ldio.pack_header(**fields)
    assert len(buf) == ldio.HEADER_SIZE
    assert buf[:8] == ldio.MAGIC
    got = ldio.parse_header(buf)
    for k, v in fields.items():
        assert got[k] == v, f"{k}: {got[k]!r} != {v!r}"


def test_header_counts_are_64_bit():
    """A variant in an all-by-all scan can have more than 2**31 partners, and a
    dataset can hold more than 2**31 pairs. int32 counts do not survive the
    design target."""
    big = (1 << 42) + 7
    got = ldio.parse_header(ldio.pack_header(
        version=1, encoding="int16", payload="banded", n_row_variants=1,
        n_pairs=big, index_offset=256, blocks_offset=512,
        footer_offset=big, r_scale=32767.0, block_a=0, block_b=0, flags=0))
    assert got["n_pairs"] == big
    assert got["footer_offset"] == big


def test_a_foreign_magic_is_refused():
    buf = bytearray(ldio.pack_header(
        version=1, encoding="int16", payload="banded", n_row_variants=1,
        n_pairs=1, index_offset=256, blocks_offset=512, footer_offset=1024,
        r_scale=32767.0, block_a=0, block_b=0, flags=0))
    buf[:8] = b"NOTCUGEN"
    with pytest.raises(ValueError, match="magic|not a .cugenld"):
        ldio.parse_header(bytes(buf))


def test_a_future_version_is_refused_rather_than_guessed():
    buf = bytearray(ldio.pack_header(
        version=1, encoding="int16", payload="banded", n_row_variants=1,
        n_pairs=1, index_offset=256, blocks_offset=512, footer_offset=1024,
        r_scale=32767.0, block_a=0, block_b=0, flags=0))
    buf[8:12] = (ldio.FORMAT_VERSION + 1).to_bytes(4, "little")
    with pytest.raises(ValueError, match="version"):
        ldio.parse_header(bytes(buf))


# ------------------------------------------------------------- delta coding
def test_delta_coding_round_trips_sorted_partner_indices():
    rng = np.random.default_rng(0)
    j = np.unique(rng.integers(0, 5_000_000, size=100_000)).astype(np.int64)
    for width in (1, 2, 4):
        enc, used = ldio.delta_encode(j, width)
        if used is None:
            continue                       # this width cannot hold the deltas
        np.testing.assert_array_equal(ldio.delta_decode(enc, j[0], used), j)


def test_delta_width_is_chosen_from_the_actual_gaps():
    near = np.arange(0, 1000, 3, dtype=np.int64)          # gaps of 3 -> u8
    far = np.array([0, 1_000_000, 5_000_000], dtype=np.int64)   # -> u32
    assert ldio.delta_width_for(near) == 1
    assert ldio.delta_width_for(far) == 4


def test_delta_coding_refuses_unsorted_input():
    """Deltas are only meaningful on an ascending index. Silently encoding a
    negative gap would produce a file that decodes to garbage."""
    with pytest.raises(ValueError, match="ascending|sorted"):
        ldio.delta_encode(np.array([5, 3, 9], dtype=np.int64), 4)


# ------------------------------------------------------------------- blocks
def test_block_payload_round_trips_through_zstd():
    rng = np.random.default_rng(1)
    j = np.sort(rng.choice(200_000, size=50_000, replace=False)).astype(np.int64)
    r = rng.uniform(-1, 1, j.size)
    blob, meta = ldio.encode_block(j, r, encoding="int16")
    gj, gr = ldio.decode_block(blob, meta, encoding="int16")
    np.testing.assert_array_equal(gj, j)
    assert np.abs(gr - r).max() <= 2.0 / 32767 + 1e-12


def test_block_meta_carries_a_zone_map():
    """min_r/max_r per block is what lets a threshold query skip whole blocks
    without decompressing them."""
    j = np.arange(10, dtype=np.int64)
    r = np.array([0.1, -0.9, 0.3, 0.2, -0.05, 0.8, 0.0, -0.2, 0.4, 0.15])
    _, meta = ldio.encode_block(j, r, encoding="int16")
    assert meta["min_r"] == pytest.approx(-0.9, abs=1e-4)
    assert meta["max_r"] == pytest.approx(0.8, abs=1e-4)
    assert meta["max_abs_r"] == pytest.approx(0.9, abs=1e-4)
    assert meta["n"] == 10


def test_a_corrupt_block_is_detected_not_silently_decoded():
    j = np.arange(100, dtype=np.int64)
    blob, meta = ldio.encode_block(j, np.zeros(100), encoding="int16")
    torn = blob[:-4]
    with pytest.raises(Exception):
        ldio.decode_block(torn, meta, encoding="int16")


# ------------------------------------------------------ shard writer / reader
def brute_pairs(seed=0, p=400, n=300, min_r2=0.0):
    """An honest reference set of (i, j, r), computed with numpy corrcoef."""
    rng = np.random.default_rng(seed)
    lat = rng.random(n)
    X = np.empty((n, p))
    for v in range(p):
        mix = rng.uniform(0.0, 0.9)
        X[:, v] = mix * lat + (1 - mix) * rng.random(n)
    R = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices(p, k=1)
    i, j, r = iu[0], iu[1], R[iu]
    keep = r ** 2 >= min_r2
    return i[keep].astype(np.int64), j[keep].astype(np.int64), r[keep]


PARAMS = dict(maf_min=0.01, maf_max=0.5, window=None, window_kb=None,
              min_dist_kb=None, max_dist_kb=None, scope="all",
              min_r2=0.05, max_p=None, correction=None, alpha=0.05,
              top_k=None, n_obs=600, m_tests=79800)


def write_shard(tmp_path, i, j, r, *, chunks=1, encoding="int16",
                block_variants=64, params=None):
    tmp_path = __import__("pathlib").Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "shard.cugenld"
    w = ldio.LDShardWriter(str(path), encoding=encoding,
                           block_variants=block_variants,
                           params=params if params is not None else PARAMS)
    for a, b, c in zip(np.array_split(i, chunks), np.array_split(j, chunks),
                       np.array_split(r, chunks)):
        w.append(a, b, c)
    w.close()
    return str(path)


def test_shard_round_trips_every_pair(tmp_path):
    i, j, r = brute_pairs(min_r2=0.05)
    assert i.size > 5000, "fixture too small to be meaningful"
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r))
    gi, gj, gr = rd.rows()
    order = np.lexsort((gj, gi))
    np.testing.assert_array_equal(gi[order], i)
    np.testing.assert_array_equal(gj[order], j)
    assert np.abs(gr[order] - r).max() <= 2.0 / 32767 + 1e-12
    assert rd.n_pairs == i.size


def test_streaming_in_many_chunks_gives_the_same_file(tmp_path):
    """append() is called once per scan tile, so the result must not depend on
    how the stream was cut up."""
    i, j, r = brute_pairs(min_r2=0.05)
    one = write_shard(tmp_path / "a", i, j, r, chunks=1)
    many = write_shard(tmp_path / "b", i, j, r, chunks=17)

    def sorted_rows(path):
        gi, gj, gr = ldio.read_ld(path).rows()
        o = np.lexsort((gj, gi))
        return gi[o], gj[o], gr[o]

    # Sorted, not raw: blocks are cut by row-variant group AND by r^2 tier, so
    # block order depends on how the stream was flushed. Ordering is guaranteed
    # within a block, never across the file -- that is what makes shards
    # independently writable, and globally sorting 10^12 rows would cost more
    # than the LD scan.
    for x, y in zip(sorted_rows(one), sorted_rows(many)):
        np.testing.assert_allclose(x, y)
    assert ldio.read_ld(one).n_pairs == ldio.read_ld(many).n_pairs == i.size


def test_variant_query_matches_brute_force(tmp_path):
    i, j, r = brute_pairs(min_r2=0.05)
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r))
    for v in (0, 1, 7, 100, 399):
        gj, gr = rd.variant(v)
        want = j[i == v]
        np.testing.assert_array_equal(gj, want)
        if want.size:
            assert np.abs(gr - r[i == v]).max() <= 2.0 / 32767 + 1e-12


def test_above_matches_brute_force_and_actually_skips_blocks(tmp_path):
    """A zone map that is correct but never skips is decorative."""
    i, j, r = brute_pairs(min_r2=0.05)
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r))
    rd.rows()
    all_blocks = rd.blocks_read

    rd.reset_counters()
    t = 0.6
    gi, gj, gr = rd.above(min_r2=t)
    m = r ** 2 >= t
    assert m.sum() > 0, "fixture has nothing above the cut"

    # above() filters on the STORED r, which is quantised, so a pair whose true
    # r^2 sits within one quantisation step of the cut may fall either side.
    # That band is the honest comparison; anything outside it is a real defect.
    got = set(zip(gi.tolist(), gj.tolist()))
    want = set(zip(i[m].tolist(), j[m].tolist()))
    step = 2.0 / 32767
    band = {(int(a), int(b)) for a, b, rr in zip(i, j, r)
            if abs(rr ** 2 - t) <= 2 * step}
    assert (got ^ want) <= band, (
        f"{len((got ^ want) - band)} pairs differ outside the quantisation "
        f"band at the threshold")
    assert rd.blocks_read < all_blocks, (
        f"zone map skipped nothing: read {rd.blocks_read} of {all_blocks}")


def test_reader_refuses_a_query_looser_than_what_was_stored(tmp_path):
    """The file was written at min_r2=0.05. Everything below that is gone, so
    above(0.01) cannot be answered -- and a short list would be a wrong answer,
    not a partial one. LDmat's stated limitation is exactly that its lossy
    parameters are fixed at creation with no record of them."""
    i, j, r = brute_pairs(min_r2=0.05)
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r))
    with pytest.raises(ValueError, match="looser|discarded|min_r2"):
        rd.above(min_r2=0.01)
    rd.above(min_r2=0.05)          # exactly the stored cut is fine
    rd.above(min_r2=0.5)           # stricter is fine


def test_reader_refuses_a_dense_request_against_a_thresholded_file(tmp_path):
    """Reconstructing a dense R from a thresholded store would replace every
    sub-threshold entry with 0 rather than its true small value."""
    i, j, r = brute_pairs(min_r2=0.05)
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r))
    with pytest.raises(ValueError, match="dense|threshold"):
        rd.dense()


def test_dense_is_served_from_an_unthresholded_file(tmp_path):
    i, j, r = brute_pairs(min_r2=0.0)
    params = dict(PARAMS, min_r2=0.0)
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r, params=params))
    R = rd.dense()
    assert R.shape == (400, 400)
    np.testing.assert_allclose(np.diag(R), 1.0)
    np.testing.assert_allclose(R, R.T, atol=1e-12)
    assert np.abs(R[i, j] - r).max() <= 2.0 / 32767 + 1e-12


def test_the_header_round_trips_the_test_space_and_retention_params(tmp_path):
    """A run is only reproducible if both parameter families are recorded, and
    the retention family is what the reader enforces."""
    i, j, r = brute_pairs(min_r2=0.05)
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r))
    for k, v in PARAMS.items():
        assert rd.params[k] == v, f"{k}: {rd.params[k]!r} != {v!r}"


def test_appends_out_of_variant_order_are_refused(tmp_path):
    """Blocks are ordered by row variant; a backwards append would either
    corrupt the index or force a global sort of 10^12 rows."""
    w = ldio.LDShardWriter(str(tmp_path / "s.cugenld"), params=PARAMS)
    w.append(np.array([5, 6]), np.array([9, 9]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="order|decreasing"):
        w.append(np.array([1]), np.array([9]), np.array([0.5]))


@pytest.mark.parametrize("encoding", ["int16", "int8", "float32"])
def test_every_encoding_round_trips_a_shard(tmp_path, encoding):
    i, j, r = brute_pairs(min_r2=0.05)
    tol = {"int16": 2 / 32767, "int8": 2 / 127, "float32": 1e-6}[encoding]
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r, encoding=encoding))
    gi, gj, gr = rd.rows()
    order = np.lexsort((gj, gi))
    np.testing.assert_array_equal(gi[order], i)
    assert np.abs(gr[order] - r).max() <= tol + 1e-12


def test_bytes_per_pair_beats_the_dataframe_schema(tmp_path):
    """The whole point. _empty_pairs is 76 B/row for stats=("r2","p")."""
    import os
    i, j, r = brute_pairs(min_r2=0.05)
    path = write_shard(tmp_path, i, j, r)
    per = os.path.getsize(path) / i.size
    assert per < 12.0, f"{per:.2f} B/pair is no better than the 76 B schema"


def sparse_trans_pairs(p=6000, n_hap=5008, seed=7):
    """The all-by-all regime: almost every pair is null, a few are real.

    r on a null pair is ~N(0, 1/sqrt(N)), so the distribution is bimodal and
    strong pairs are scattered across the whole variant axis rather than
    clustered -- which is exactly why a zone map keyed on position alone cannot
    skip anything.
    """
    rng = np.random.default_rng(seed)
    I, J, R = [], [], []
    for v in range(p - 1):
        k = min(int(rng.integers(20, 60)), p - v - 1)
        j = np.sort(rng.choice(np.arange(v + 1, p), size=k, replace=False))
        r = rng.normal(0, 1 / np.sqrt(n_hap), k)
        hot = rng.random(k) < 0.02
        r[hot] = rng.uniform(0.5, 0.99, int(hot.sum()))
        I.append(np.full(k, v, dtype=np.int64))
        J.append(j.astype(np.int64))
        R.append(r)
    i, j, r = np.concatenate(I), np.concatenate(J), np.concatenate(R)
    keep = r ** 2 >= 1e-4
    return i[keep], j[keep], r[keep]


def test_tiering_makes_the_zone_map_skip_most_blocks(tmp_path):
    """Blocks are cut by r^2 tier as well as by row-variant range.

    Keyed on position alone the map was decorative -- measured at 94 of 95
    blocks read on this fixture, because 4% strong pairs spread over the variant
    axis put something strong in every block. Cutting by value makes each block
    homogeneous, and the skip becomes proportional to the query.
    """
    i, j, r = sparse_trans_pairs()
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r, block_variants=256,
                                  params=dict(PARAMS, min_r2=1e-4)))
    rd.rows()
    total = rd.blocks_read
    assert total > 20, "fixture must produce enough blocks to skip any"

    seen = {}
    for t in (0.5, 0.8):
        rd.reset_counters()
        gi, gj, _ = rd.above(min_r2=t)
        m = r ** 2 >= t
        assert set(zip(gi.tolist(), gj.tolist())) == set(
            zip(i[m].tolist(), j[m].tolist()))
        seen[t] = rd.blocks_read

    assert seen[0.8] <= 0.35 * total, (
        f"r2>=0.8 read {seen[0.8]}/{total} blocks; tiering is not skipping")
    assert seen[0.8] < seen[0.5], "a stricter cut must read fewer blocks"


def test_a_whole_tier_above_the_cut_needs_no_per_pair_filter(tmp_path):
    """Correctness check on the shortcut: when a block's tier floor is already
    at or above the threshold, every pair in it qualifies and the filter is
    skipped. Getting that wrong would emit sub-threshold pairs."""
    i, j, r = sparse_trans_pairs()
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r, block_variants=256,
                                  params=dict(PARAMS, min_r2=1e-4)))
    for t in (0.2, 0.5, 0.8):
        _, _, gr = rd.above(min_r2=t)
        assert (gr ** 2 >= t).all(), f"above({t}) emitted a sub-threshold pair"


def test_rows_returns_pairs_sitting_exactly_on_the_stored_threshold(tmp_path):
    """rows() must not re-apply the stored cut.

    r is quantised, so a pair written at true r^2 = min_r2 can come back with
    stored r^2 a hair under it. Re-filtering on the way out would silently drop
    pairs the file demonstrably contains -- and the drop would be invisible,
    because the count would still look plausible.
    """
    t = 0.25
    r_at = np.sqrt(t)
    # a spread of pairs straddling the cut by less than one quantisation step
    step = 1.0 / 32767
    r = np.array([r_at, r_at + step / 4, r_at - step / 4, r_at + step,
                  0.9, 0.5, -r_at, -(r_at + step / 4)])
    i = np.zeros(r.size, dtype=np.int64)
    j = np.arange(1, r.size + 1, dtype=np.int64)

    rd = ldio.read_ld(write_shard(tmp_path, i, j, r,
                                  params=dict(PARAMS, min_r2=t)))
    gi, gj, gr = rd.rows()
    assert gi.size == r.size, (
        f"rows() returned {gi.size} of {r.size} stored pairs -- the stored cut "
        f"is being re-applied to quantised values")
    np.testing.assert_array_equal(np.sort(gj), j)

    # and the boundary pairs really are within a step of the cut, so this
    # fixture would not catch the bug by accident
    assert np.abs(np.abs(r) ** 2 - t).min() < 2 * step * r_at


# ------------------------------------------------------- the writer registry
# The user picks the format; the point is options, not one imposed container.

def test_every_backend_round_trips_the_same_pairs(tmp_path, small_cugen):
    """Write one LD result to every supported container and read each back.

    A format that cannot reproduce the pairs it was given is not an option, it
    is a data-loss bug with a file extension.
    """
    from cugen import ld as L
    path, _ = small_cugen
    df = L.ld_matrix(path, stats=("r", "r2"), backend="numpy", verbose=False)
    want = set(zip(df["gidx_a"].tolist(), df["gidx_b"].tolist()))

    for ext in (".tsv", ".csv", ".parquet", ".feather", ".npz", ".cugenld"):
        out = tmp_path / f"ld{ext}"
        ldio.write_ld(df, str(out), params=dict(PARAMS, min_r2=0.0))
        assert out.exists(), f"{ext} wrote nothing"
        got = ldio.read_pairs(str(out))
        assert set(zip(got["gidx_a"].tolist(), got["gidx_b"].tolist())) == want, (
            f"{ext} did not round-trip the pair set")
        assert np.abs(np.sort(got["R"].to_numpy(np.float64))
                      - np.sort(df["R"].to_numpy(np.float64))).max() < 1e-4, (
            f"{ext} did not round-trip r")


def test_npz_is_readable_with_plain_numpy(tmp_path, small_cugen):
    """The reason .npz is on the menu: one line, no cugen, no new dependency."""
    from cugen import ld as L
    path, _ = small_cugen
    df = L.ld_matrix(path, stats=("r", "r2"), backend="numpy", verbose=False)
    out = tmp_path / "ld.npz"
    ldio.write_ld(df, str(out), params=PARAMS)

    z = np.load(out)
    assert {"gidx_a", "gidx_b", "R"} <= set(z.files)
    np.testing.assert_array_equal(z["gidx_a"], df["gidx_a"].to_numpy())


def test_parquet_is_written_with_zstd_and_statistics(tmp_path, small_cugen):
    """Untuned parquet was 24.3 B/row against 19.0 with zstd, and without row
    statistics a bp-range predicate cannot be pushed down at all -- which is how
    the live downstream consumer queries these files."""
    import pyarrow.parquet as pq
    from cugen import ld as L
    path, _ = small_cugen
    df = L.ld_matrix(path, stats=("r", "r2"), backend="numpy", verbose=False)
    out = tmp_path / "ld.parquet"
    ldio.write_ld(df, str(out), params=PARAMS)

    md = pq.ParquetFile(out).metadata
    codecs = {md.row_group(g).column(c).compression
              for g in range(md.num_row_groups)
              for c in range(md.num_columns)}
    assert codecs <= {"ZSTD"}, f"expected zstd, got {codecs}"
    assert md.row_group(0).column(0).statistics is not None, (
        "no column statistics -- predicate pushdown will read every row group")


def test_dead_columns_are_not_written(tmp_path, small_cugen):
    """POS is literal zeros and ID is "." on the device path, and CHR is
    constant per file. Writing them is pure overhead."""
    from cugen import ld as L
    path, _ = small_cugen
    df = L.ld_matrix(path, stats=("r", "r2"), backend="numpy", verbose=False)
    out = tmp_path / "ld.tsv"
    ldio.write_ld(df, str(out), params=PARAMS, drop_dead=True)
    # pyarrow's CSV writer quotes header names; that is pre-existing behaviour
    # and not what this test is about
    header = [c.strip('"') for c in out.read_text().splitlines()[0].split("\t")]
    for dead in ("POS_A", "POS_B", "ID_A", "ID_B"):
        assert dead not in header, f"{dead} is all-placeholder but was written"
    assert "gidx_a" in header and "R" in header


def test_an_unknown_extension_is_refused(tmp_path, small_cugen):
    from cugen import ld as L
    df = L.ld_matrix(small_cugen[0], stats=("r2",), backend="numpy",
                     verbose=False)
    with pytest.raises(ValueError, match="extension|format"):
        ldio.write_ld(df, str(tmp_path / "ld.rubbish"), params=PARAMS)


def test_ld_matrix_can_write_the_native_format_directly(tmp_path, small_cugen):
    from cugen import ld as L
    out = tmp_path / "chr22.cugenld"
    df = L.ld_matrix(small_cugen[0], stats=("r", "r2"), min_r2=0.01,
                     output=str(out), backend="numpy", verbose=False)
    rd = ldio.read_ld(str(out))
    assert rd.n_pairs == len(df)
    assert rd.params["min_r2"] == 0.01
    gi, gj, _ = rd.rows()
    assert set(zip(gi.tolist(), gj.tolist())) == set(
        zip(df["gidx_a"].tolist(), df["gidx_b"].tolist()))


def test_variant_lookup_reads_only_the_blocks_that_hold_it(tmp_path):
    """variant() must not scan the whole shard.

    Scanning every block's row_variants list per lookup measured 367 ms on a
    66-block file from real chr22 -- linear in blocks and in Python. The reader
    builds a row-variant index at open instead.
    """
    i, j, r = sparse_trans_pairs()
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r, block_variants=256,
                                  params=dict(PARAMS, min_r2=1e-4)))
    rd.rows()
    total = rd.blocks_read
    assert total > 20

    rd.reset_counters()
    gj, gr = rd.variant(100)
    want = j[i == 100]
    np.testing.assert_array_equal(gj, want)
    assert rd.blocks_read <= 6, (
        f"one variant read {rd.blocks_read} of {total} blocks; the index is "
        f"not being used")


def test_blocks_are_capped_by_pair_count_not_only_by_variant_count(tmp_path):
    """zstd frames are not seekable, so a one-variant lookup pays for the whole
    block it lands in. On real chr22 a 4096-variant block held 2.5 M pairs and
    variant() cost 360 ms -- the cap bounds that directly."""
    i, j, r = brute_pairs(seed=5, p=1200, n=400, min_r2=0.0)
    assert i.size > 400_000, "fixture must exceed the cap to test it"
    cap = 1 << 14
    path = tmp_path / "s.cugenld"
    w = ldio.LDShardWriter(str(path), block_variants=100_000,
                           max_block_pairs=cap,
                           params=dict(PARAMS, min_r2=0.0))
    w.append(i, j, r)
    w.close()

    rd = ldio.read_ld(str(path))
    assert rd.n_pairs == i.size
    biggest = max(b["n"] for b in rd.blocks)
    # a single row variant can exceed the cap on its own; the guarantee is that
    # blocks are cut at variant boundaries as soon as the cap is passed
    assert biggest <= cap + int(np.bincount(i).max()), (
        f"largest block holds {biggest} pairs against a cap of {cap}")
    gi, gj, _ = rd.rows()
    assert gi.size == i.size


# ------------------------------------------- sharded datasets & resumability
# One 100 TB object cannot be written by concurrent GPUs, cannot be resumed, and
# cannot be partially recomputed. A shard is one scan tile's output.

def tile_stream(i, j, r, n_tiles=6):
    """Split pairs into (A, B) tile keys the way the scan walks them."""
    lo, hi = int(i.min()), int(j.max()) + 1
    edges = np.linspace(lo, hi, n_tiles + 1).astype(np.int64)
    out = []
    for a in range(n_tiles):
        for b in range(a, n_tiles):
            m = ((i >= edges[a]) & (i < edges[a + 1])
                 & (j >= edges[b]) & (j < edges[b + 1]))
            if m.any():
                out.append(((a, b), i[m], j[m], r[m]))
    return out


def test_dataset_round_trips_across_many_shards(tmp_path):
    i, j, r = brute_pairs(min_r2=0.05)
    d = str(tmp_path / "ds")
    with ldio.LDDatasetWriter(d, params=PARAMS) as w:
        for key, ti, tj, tr in tile_stream(i, j, r):
            w.write_shard(key, ti, tj, tr)

    ds = ldio.open_ld(d)
    assert ds.n_pairs == i.size
    assert ds.n_shards > 1, "fixture must produce more than one shard"
    gi, gj, gr = ds.rows()
    order = np.lexsort((gj, gi))
    np.testing.assert_array_equal(gi[order], i)
    np.testing.assert_array_equal(gj[order], j)
    assert np.abs(gr[order] - r).max() <= 0.51 / 32767 + 1e-12


def test_a_dataset_resumes_and_skips_what_is_already_written(tmp_path):
    """A 3.3 h genome-wide job that dies at hour 2 must not start over."""
    i, j, r = brute_pairs(min_r2=0.05)
    tiles = tile_stream(i, j, r)
    d = str(tmp_path / "ds")

    # first attempt dies partway
    w = ldio.LDDatasetWriter(d, params=PARAMS)
    for key, ti, tj, tr in tiles[:4]:
        w.write_shard(key, ti, tj, tr)
    w.close()
    partial = ldio.open_ld(d)
    assert partial.n_shards == 4
    assert not partial.complete

    # resume: the writer reports what is already done and does not redo it
    w2 = ldio.LDDatasetWriter(d, params=PARAMS, resume=True)
    done = set(w2.completed_shards())
    assert done == {t[0] for t in tiles[:4]}
    redone = 0
    for key, ti, tj, tr in tiles:
        if key in done:
            continue
        w2.write_shard(key, ti, tj, tr)
        redone += 1
    w2.mark_complete()
    w2.close()
    assert redone == len(tiles) - 4

    ds = ldio.open_ld(d)
    assert ds.complete
    assert ds.n_pairs == i.size
    gi, gj, _ = ds.rows()
    assert set(zip(gi.tolist(), gj.tolist())) == set(
        zip(i.tolist(), j.tolist()))


def test_a_resumed_dataset_equals_one_written_in_a_single_pass(tmp_path):
    i, j, r = brute_pairs(min_r2=0.05)
    tiles = tile_stream(i, j, r)

    one = str(tmp_path / "one")
    with ldio.LDDatasetWriter(one, params=PARAMS) as w:
        for key, ti, tj, tr in tiles:
            w.write_shard(key, ti, tj, tr)

    two = str(tmp_path / "two")
    w = ldio.LDDatasetWriter(two, params=PARAMS)
    for key, ti, tj, tr in tiles[:3]:
        w.write_shard(key, ti, tj, tr)
    w.close()
    w = ldio.LDDatasetWriter(two, params=PARAMS, resume=True)
    for key, ti, tj, tr in tiles[3:]:
        w.write_shard(key, ti, tj, tr)
    w.mark_complete()
    w.close()

    a, b = ldio.open_ld(one), ldio.open_ld(two)
    assert a.n_pairs == b.n_pairs
    for x, y in zip(a.rows(), b.rows()):
        np.testing.assert_allclose(np.sort(x), np.sort(y))


def test_an_interrupted_shard_is_not_trusted(tmp_path):
    """A shard half-written when the process died must not be read as complete.

    Shards land by atomic rename, so a torn file never appears in the manifest.
    """
    i, j, r = brute_pairs(min_r2=0.05)
    d = str(tmp_path / "ds")
    with ldio.LDDatasetWriter(d, params=PARAMS) as w:
        for key, ti, tj, tr in tile_stream(i, j, r)[:3]:
            w.write_shard(key, ti, tj, tr)

    import pathlib
    (pathlib.Path(d) / "0-99.ldz").write_bytes(b"torn garbage")
    ds = ldio.open_ld(d)
    assert ds.n_shards == 3, "a file not in the manifest must be ignored"
    ds.rows()                                   # must not raise


def test_queries_route_across_shards(tmp_path):
    i, j, r = brute_pairs(min_r2=0.05)
    d = str(tmp_path / "ds")
    with ldio.LDDatasetWriter(d, params=PARAMS) as w:
        for key, ti, tj, tr in tile_stream(i, j, r):
            w.write_shard(key, ti, tj, tr)
    ds = ldio.open_ld(d)

    for v in (0, 5, 100, 399):
        gj, _ = ds.variant(v)
        np.testing.assert_array_equal(gj, np.sort(j[i == v]))

    gi, gj, _ = ds.above(min_r2=0.5)
    m = r ** 2 >= 0.5
    step = 1.0 / 32767
    band = {(int(a), int(b)) for a, b, rr in zip(i, j, r)
            if abs(rr ** 2 - 0.5) <= 2 * step}
    assert (set(zip(gi.tolist(), gj.tolist()))
            ^ set(zip(i[m].tolist(), j[m].tolist()))) <= band


def test_the_manifest_records_the_params_and_the_shard_list(tmp_path):
    i, j, r = brute_pairs(min_r2=0.05)
    d = str(tmp_path / "ds")
    tiles = tile_stream(i, j, r)
    with ldio.LDDatasetWriter(d, params=PARAMS) as w:
        for key, ti, tj, tr in tiles:
            w.write_shard(key, ti, tj, tr)

    import json
    import pathlib
    man = json.loads((pathlib.Path(d) / "manifest.json").read_text())
    for k, v in PARAMS.items():
        assert man["params"][k] == v
    assert len(man["shards"]) == len(tiles)
    assert sum(sh["n_pairs"] for sh in man["shards"]) == i.size
    assert man["format"] == "cugenld" and man["version"] == ldio.FORMAT_VERSION


def test_resuming_with_different_params_is_refused(tmp_path):
    """Half a dataset at maf_min=0.01 and half at 0.05 is not a dataset. The
    test space sets m, so mixing them corrupts every corrected threshold."""
    i, j, r = brute_pairs(min_r2=0.05)
    d = str(tmp_path / "ds")
    w = ldio.LDDatasetWriter(d, params=PARAMS)
    w.write_shard((0, 0), i[:100], j[:100], r[:100])
    w.close()
    with pytest.raises(ValueError, match="params|differ"):
        ldio.LDDatasetWriter(d, params=dict(PARAMS, maf_min=0.25), resume=True)


def test_the_manifest_skips_shards_before_opening_them(tmp_path):
    """Two levels of skipping: whole shards on the manifest's per-shard index
    and max_abs_r, then whole blocks on each shard's own zone map. A shard that
    cannot contain the answer must not be opened at all."""
    i, j, r = sparse_trans_pairs(p=6000)
    d = str(tmp_path / "ds")
    with ldio.LDDatasetWriter(d, params=dict(PARAMS, min_r2=1e-4)) as w:
        for key, ti, tj, tr in tile_stream(i, j, r, n_tiles=10):
            w.write_shard(key, ti, tj, tr)

    ds = ldio.open_ld(d)
    assert ds.n_shards > 20
    ds.rows()
    total = ds.blocks_read

    n_shards = ds.n_shards

    ds.reset_counters()
    ds.above(min_r2=0.8)
    assert ds.blocks_read <= 0.35 * total, "block zone map is not skipping"
    # NOT asserting a shard-level skip here. This fixture spreads strong pairs
    # uniformly, so every shard genuinely contains one and no shard CAN be
    # skipped -- the manifest max_abs_r is correct but unexercised. See
    # test_shard_r_skip_works_when_strong_ld_is_local for the realistic case.

    ds.reset_counters()
    gj, _ = ds.variant(100)
    np.testing.assert_array_equal(gj, np.sort(j[i == 100]))
    assert ds.shards_read <= 0.3 * n_shards, (
        f"one variant opened {ds.shards_read} of {n_shards} shards; the "
        f"manifest min_i/max_i skip is dead")
    assert ds.blocks_read <= 0.15 * total

    ds.reset_counters()
    gi, gj, _ = ds.region(1000, 2000)
    m = (i >= 1000) & (i < 2000) & (j >= 1000) & (j < 2000)
    assert set(zip(gi.tolist(), gj.tolist())) == set(
        zip(i[m].tolist(), j[m].tolist()))
    assert ds.shards_read <= 0.4 * n_shards, "region shard skip is dead"
    assert ds.blocks_read <= 0.25 * total


def test_sharding_costs_a_fixed_overhead_per_shard(tmp_path):
    """Honest accounting: every shard pays a 256-byte header and its own footer,
    so shards must be sized to hold a meaningful number of pairs. This pins the
    direction so nobody shards a small dataset into uselessness."""
    i, j, r = brute_pairs(min_r2=0.05)
    few, many = str(tmp_path / "few"), str(tmp_path / "many")
    for d, n_tiles in ((few, 2), (many, 12)):
        with ldio.LDDatasetWriter(d, params=PARAMS) as w:
            for key, ti, tj, tr in tile_stream(i, j, r, n_tiles=n_tiles):
                w.write_shard(key, ti, tj, tr)
    a, b = ldio.open_ld(few), ldio.open_ld(many)
    assert a.n_pairs == b.n_pairs == i.size
    assert b.n_shards > a.n_shards
    assert b.bytes_per_pair() > a.bytes_per_pair(), (
        "more shards should cost more per pair; if not, the accounting is wrong")


def test_a_manifest_entry_whose_file_vanished_is_dropped_on_resume(tmp_path):
    """The manifest is the record, but a file it names may not survive -- a
    partial upload, a cleaned scratch dir. Resuming must re-do that shard rather
    than assume it is present."""
    import os
    i, j, r = brute_pairs(min_r2=0.05)
    tiles = tile_stream(i, j, r)
    d = str(tmp_path / "ds")
    w = ldio.LDDatasetWriter(d, params=PARAMS)
    for key, ti, tj, tr in tiles[:4]:
        w.write_shard(key, ti, tj, tr)
    w.close()

    gone = ldio.open_ld(d).shards[0]["file"]
    os.remove(os.path.join(d, gone))

    w2 = ldio.LDDatasetWriter(d, params=PARAMS, resume=True)
    assert len(w2.completed_shards()) == 3, (
        "a manifest entry with no file on disk was trusted")
    assert gone not in {_shard_file(k) for k in w2.completed_shards()}


def _shard_file(key):
    return f"{int(key[0])}-{int(key[1])}.ldz"


def local_ld_pairs(p=6000, n_hap=5008, seed=11):
    """Realistic LD: strength decays with separation, as it does on a chromosome.

    The uniform-signal fixture cannot exercise a shard-level r skip, because
    every shard contains something strong. Real strong LD is near-diagonal, so
    off-diagonal shards genuinely have a low max |r| and are skippable -- which
    is the whole reason the manifest carries one.
    """
    rng = np.random.default_rng(seed)
    I, J, R = [], [], []
    for v in range(p - 1):
        k = min(int(rng.integers(30, 80)), p - v - 1)
        j = np.sort(rng.choice(np.arange(v + 1, p), size=k, replace=False))
        sep = (j - v).astype(np.float64)
        # r^2 ~ 1/(1 + sep/scale): strong nearby, noise far away
        base = np.sqrt(1.0 / (1.0 + sep / 30.0))
        r = base * rng.uniform(0.6, 1.0, k) + rng.normal(
            0, 1 / np.sqrt(n_hap), k)
        R.append(np.clip(r, -1, 1))
        I.append(np.full(k, v, dtype=np.int64))
        J.append(j.astype(np.int64))
    i, j, r = np.concatenate(I), np.concatenate(J), np.concatenate(R)
    keep = r ** 2 >= 1e-4
    return i[keep], j[keep], r[keep]


def test_shard_r_skip_works_when_strong_ld_is_local(tmp_path):
    """The manifest's per-shard max_abs_r earns its keep on realistic data."""
    i, j, r = local_ld_pairs()
    d = str(tmp_path / "ds")
    with ldio.LDDatasetWriter(d, params=dict(PARAMS, min_r2=1e-4)) as w:
        for key, ti, tj, tr in tile_stream(i, j, r, n_tiles=10):
            w.write_shard(key, ti, tj, tr)

    ds = ldio.open_ld(d)
    n_shards = ds.n_shards
    assert n_shards > 20

    ds.reset_counters()
    gi, gj, _ = ds.above(min_r2=0.8)
    m = r ** 2 >= 0.8
    step = 1.0 / 32767
    band = {(int(a), int(b)) for a, b, rr in zip(i, j, r)
            if abs(rr ** 2 - 0.8) <= 2 * step}
    assert (set(zip(gi.tolist(), gj.tolist()))
            ^ set(zip(i[m].tolist(), j[m].tolist()))) <= band
    assert ds.shards_read < 0.5 * n_shards, (
        f"opened {ds.shards_read} of {n_shards} shards for r2>=0.8 on "
        f"near-diagonal data; the manifest max_abs_r skip is dead")


def test_repeated_queries_do_not_reparse_every_shard_footer(tmp_path):
    """Opening a shard parses its footer and rebuilds its row-variant index.

    A cross-shard variant() touches several shards, and re-parsing each time
    took a GPU-scale lookup from 2.94 ms on one shard to 21.74 ms across 28.
    Readers are cached; the files are immutable so it is always safe.
    """
    i, j, r = local_ld_pairs()
    d = str(tmp_path / "ds")
    with ldio.LDDatasetWriter(d, params=dict(PARAMS, min_r2=1e-4)) as w:
        for key, ti, tj, tr in tile_stream(i, j, r, n_tiles=10):
            w.write_shard(key, ti, tj, tr)

    ds = ldio.open_ld(d)
    first = ds.variant(100)
    n_cached = len(ds._cache)
    assert n_cached > 1, "expected several shards to hold one variant"

    for _ in range(5):
        again = ds.variant(100)
    np.testing.assert_array_equal(first[0], again[0])
    assert len(ds._cache) == n_cached, "the cache is not being reused"
    assert len(ds._cache) <= ds._cache_max


# -------------------------------------------- loops opened earlier, now closed
def test_gz_output_is_actually_compressed(tmp_path, small_cugen):
    """The cuDF branch of the legacy writer never requested gzip, so a .tsv.gz
    got uncompressed bytes under a .gz name. Detectable without a GPU by simply
    asking whether the file is a gzip stream."""
    import gzip
    from cugen import ld as L
    out = tmp_path / "ld.tsv.gz"
    L.ld_matrix(small_cugen[0], stats=("r", "r2"), output=str(out),
                backend="numpy", verbose=False)
    assert out.read_bytes()[:2] == b"\x1f\x8b", "not a gzip stream"
    with gzip.open(out, "rt") as f:
        head = f.readline()
    assert "gidx_a" in head.replace('"', "")


def test_output_format_matrix_returns_an_LDMatrix(small_cugen):
    """output_format="matrix" was validated and then ignored: LDMatrix was
    declared, exported, documented, and never constructed anywhere in the repo,
    so passing "matrix" silently returned a pairs DataFrame."""
    from cugen import ld as L
    path, dos = small_cugen
    m = L.ld_matrix(path, stats=("r", "r2"), output_format="matrix",
                    backend="numpy", verbose=False)
    assert isinstance(m, L.LDMatrix), f"got {type(m).__name__}"
    p = dos.shape[0]
    assert m.r.shape == (p, p) and m.r2.shape == (p, p)
    assert m.gidx.size == p and m.n_samples == dos.shape[1]

    # unit diagonal, symmetric, and agreeing with the pairs path
    np.testing.assert_allclose(np.diag(m.r), 1.0)
    np.testing.assert_allclose(m.r, m.r.T, atol=1e-6)
    df = L.ld_matrix(path, stats=("r",), backend="numpy", verbose=False)
    pos = {int(g): k for k, g in enumerate(m.gidx)}
    for a, b, r in zip(df["gidx_a"], df["gidx_b"], df["R"]):
        assert abs(m.r[pos[int(a)], pos[int(b)]] - r) < 1e-6


def test_matrix_output_refuses_a_threshold(small_cugen):
    """A dense matrix built from a thresholded scan would silently replace every
    sub-threshold correlation with zero rather than its true value -- the same
    refusal LDReader.dense() makes, for the same reason."""
    from cugen import ld as L
    with pytest.raises(ValueError, match="matrix|threshold|min_r2"):
        L.ld_matrix(small_cugen[0], stats=("r",), output_format="matrix",
                    min_r2=0.2, backend="numpy", verbose=False)


# ------------------------------------------------ streaming into .cugenld
# #4 added stream=True with an on_flush callback whose flushes land on tile
# boundaries. A shard IS one tile's output, so the two compose: each flush
# becomes a shard, and nothing is ever accumulated.

@requires_cudf
@pytest.mark.gpu
def test_streaming_to_cugenld_matches_the_buffered_result(tmp_path,
                                                          write_cugen_file):
    from cugen import ld as L
    from conftest import simulate_haplotypes
    # simulate_haplotypes returns (n_variants, n_samples)
    dos = simulate_haplotypes(400, 900, seed=4)
    assert dos.shape == (900, 400)
    path = write_cugen_file(dos)

    # min_r2=0 on purpose: the flush buffer clamps at 65,536 rows and drains at
    # half full, so a thresholded scan on this fixture emits one flush and the
    # sharding assertion below would be vacuous.
    buffered = L.ld_matrix(path, min_r2=0.0, stats=("r", "r2"),
                           output=str(tmp_path / "b.parquet"),
                           max_pairs=10**15, verbose=False)

    d = str(tmp_path / "streamed.cugenld")
    # tile_size forces many tiles: the buffer drains BETWEEN tiles, so a scan
    # that fits in one tile flushes exactly once however large the result is,
    # and the sharding assertion below would be vacuous.
    n = L.ld_matrix(path, min_r2=0.0, stats=("r", "r2"), output=d,
                    stream=True, flush_rows=1 << 16, tile_size=128,
                    max_pairs=10**15, verbose=False)
    assert isinstance(n, int), "stream=True must return a row count"
    assert n == len(buffered)

    ds = ldio.open_ld(d)
    assert ds.complete and ds.n_pairs == n
    assert ds.n_shards > 1, "flush_rows was too large to exercise sharding"
    gi, gj, gr = ds.rows()
    want = set(zip(buffered["gidx_a"].to_numpy().tolist(),
                   buffered["gidx_b"].to_numpy().tolist()))
    assert set(zip(gi.tolist(), gj.tolist())) == want
    # r survives quantisation
    ref = dict(zip(zip(buffered["gidx_a"].to_numpy().tolist(),
                       buffered["gidx_b"].to_numpy().tolist()),
                   buffered["R"].to_numpy(np.float64).tolist()))
    err = max(abs(r - ref[(int(a), int(b))])
              for a, b, r in zip(gi, gj, gr))
    assert err <= 0.51 / 32767 + 1e-9, f"max |dr| {err:.2e}"


@requires_cudf
@pytest.mark.gpu
def test_streaming_to_cugenld_records_its_params_and_resumes(tmp_path,
                                                            write_cugen_file):
    """A streamed dataset is an ordinary sharded dataset: it carries the run's
    parameters and a resumed writer sees what already landed."""
    from cugen import ld as L
    from conftest import simulate_haplotypes
    dos = simulate_haplotypes(300, 600, seed=9)
    path = write_cugen_file(dos)
    d = str(tmp_path / "s.cugenld")
    L.ld_matrix(path, min_r2=0.05, stats=("r", "r2"), output=d, stream=True,
                flush_rows=1 << 15, tile_size=128, max_pairs=10**15,
                verbose=False)

    ds = ldio.open_ld(d)
    assert ds.params["min_r2"] == 0.05
    assert ds.params["n_obs"] == dos.shape[1] or ds.params["n_obs"] > 0
    w = ldio.LDDatasetWriter(d, params=ds.params, resume=True)
    assert len(w.completed_shards()) == ds.n_shards


# ------------------------------------------------------- ancestry adjustment
# An adjusted r is a correlation between PC-RESIDUALISED genotypes. chi2 = N*r^2
# does not transfer to it: after rank-K residualisation the effective sample
# size is not N, and cugen.ld already refuses p-values for the corrected stats
# for exactly this reason (see the module docstring at cugen/ld.py:245). The
# format has to carry that refusal too, or a stored adjusted r silently yields
# p-values computed under the wrong null the moment anyone reads the file back.

ADJ = dict(PARAMS, adjust=10)


def test_adjust_round_trips_through_the_header(tmp_path):
    """It has to be recorded, or a reader cannot know to refuse."""
    i = np.array([0, 0, 1]); j = np.array([1, 2, 2])
    r = np.array([0.9, 0.5, 0.3])
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r, params=ADJ))
    assert rd.params["adjust"] == 10


def test_neglog10p_refuses_on_an_adjusted_file(tmp_path):
    i = np.array([0, 0, 1]); j = np.array([1, 2, 2])
    r = np.array([0.9, 0.5, 0.3])
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r, params=ADJ))
    with pytest.raises(ValueError, match="adjust"):
        rd.neglog10p(r)


def test_above_with_max_p_refuses_on_an_adjusted_file(tmp_path):
    """The p-value path into `above` is the one a caller reaches by accident:
    max_p is converted to an equivalent r^2 using n_obs, which is exactly the
    conversion that does not hold after residualisation."""
    i = np.array([0, 0, 1]); j = np.array([1, 2, 2])
    r = np.array([0.9, 0.5, 0.3])
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r, params=ADJ))
    with pytest.raises(ValueError, match="adjust"):
        rd.above(max_p=1e-3)


def test_above_with_p_columns_refuses_on_an_adjusted_file(tmp_path):
    i = np.array([0, 0, 1]); j = np.array([1, 2, 2])
    r = np.array([0.9, 0.5, 0.3])
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r, params=ADJ))
    with pytest.raises(ValueError, match="adjust"):
        rd.above(0.1, with_p=True)


def test_an_unadjusted_file_still_yields_p_values(tmp_path):
    """The guard must key on `adjust` being set, not on its presence."""
    i = np.array([0, 0, 1]); j = np.array([1, 2, 2])
    r = np.array([0.9, 0.5, 0.3])
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r, params=PARAMS))
    p = rd.neglog10p(r)
    assert np.all(np.isfinite(p)) and p[0] > p[2]
    rd.above(max_p=1e-10)        # tighter than the stored min_r2; must not raise


def test_adjust_zero_is_not_an_adjusted_file(tmp_path):
    """k = 0 is the intercept-only case, which reproduces plain r exactly, so
    p-values remain valid. Treating 0 as "adjusted" would refuse a file that is
    not adjusted at all."""
    i = np.array([0, 0, 1]); j = np.array([1, 2, 2])
    r = np.array([0.9, 0.5, 0.3])
    rd = ldio.read_ld(write_shard(tmp_path, i, j, r, params=dict(PARAMS, adjust=0)))
    assert np.all(np.isfinite(rd.neglog10p(r)))
