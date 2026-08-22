"""cugen.ldio - on-disk LD results: formats, streaming writers, and queries.

    write_ld(...)     pluggable writer, chosen by file extension
    read_ld(path)     reader for the native .cugenld format
    LDReader          .region() / .variant() / .above() / .dense()

Why this module exists
----------------------
Writing already costs as much as computing. The headline LD benchmark
decomposes as

    t(s) = pairs/2.07e10 + rows/1.63e7 + 0.194
    scan 0.706 s (47%) | write 0.647 s (43%) | fixed 0.194 s (13%)

-- 61.5 ns/row on the cuDF path. And the schema that cost is paying for is
mostly dead: of the thirteen columns in cugen.ld._empty_pairs, only gidx_a,
gidx_b and one float carry information. CHR is constant per file (recovered by
regex on the filename), POS is written as literal zeros and ID as "." on the
device path, MAF is in the .cugen header, N_OBS is constant without
missingness, R2 = R*R, and NEG_LOG10_P is a closed-form function of N_OBS * r2.
The device fast path -- the only one a large run takes -- is gated on
`annotation is None`, which is exactly the condition that makes those columns
degenerate, so the largest-output path writes the most dead bytes.

The design target is genome-wide all-by-all
-------------------------------------------
At a fixed threshold, emission is LINEAR in p (61.5 rows/variant at MAF >= 1%),
because each variant has a bounded number of partners above any r^2, set by its
LD block rather than by how many variants were scanned. The quadratic term does
not blow up either: at a family-wise threshold the expected number of retained
NULL pairs is alpha by construction, so raising the test count raises the
threshold exactly enough to compensate. Trans pairs are self-limiting.

What is not self-limiting is the sample size. Since chi2 = N * r^2, a larger
cohort pushes the significance threshold down and the cis partner count up --
and the threshold is by far the dominant lever on output volume. Measured on
real chr22 (51,100 variants at MAF >= 0.01, all pairs, see
benchmarks/results/STORAGE.md):

    min_r2   rows          rows/variant   % of all pairs
    0.2      1,415,371     27.7           0.108%
    0.1      3,448,914     67.5           0.264%
    0.05     12,730,642    249.1          0.975%
    0.02     66,765,884    1,306.6        5.114%
    0.01     164,079,591   3,211.0        12.568%

That is 116x between 0.2 and 0.01, and the log-log slope is **t^-1.64**, not the
1/t an earlier version of this docstring assumed. Two things follow, in opposite
directions. Tightening the threshold buys more than 1/t suggests; but the power
law cannot continue, because at r2 >= 0.01 the retained set is already 12.6% of
ALL pairs, so it saturates toward the full within-LD-span pair space rather than
growing without bound. Extrapolating t^-1.64 into the biobank regime therefore
overshoots -- the real ceiling is
p * (variants within the LD span), roughly 7e10 rows genome-wide for a ~1 Mb
span, plus a trans contribution that stays alpha-limited.

So: 1e6 to 1e11 rows across the useful threshold range, and bytes-per-pair is
the only lever this module owns.

Shape of the format
-------------------
A sharded dataset with a manifest, not one file. A single 100 TB object cannot
be written by concurrent GPUs, cannot be resumed, and cannot be partially
recomputed. Shards are keyed by the variant-block pair (A, B) that the scan
already tiles over, so one shard is one tile's output and writers need no
cross-shard coordination.

Two payload kinds, because banding is an assumption all-by-all breaks. Delta
coding a partner index only pays when partners are nearby -- true on the
diagonal, false everywhere else -- so BANDED keeps cis near 3 B/pair and
SCATTER stores explicit (j, r) for trans rather than degrading to nonsense.

The block directory is a FOOTER, never interleaved with the payload. Pan-UKBB's
BlockMatrix interleaves its LZ4 chunk lengths with the data, so there is no
index that would allow reading only the needed rows, and the measured
consequence is a ~400x read amplification. Offsets go in a footer or they are
useless.

Compression is zstd through pyarrow, which is already a hard dependency, so
this costs no new package.

References
----------
Only what was checked before it went into the code.

LDmat, Kim et al. (2023) Bioinformatics 39(2):btad092
    https://doi.org/10.1093/bioinformatics/btad092
    1 Mb HDF5 groups; lossless at 59% of source, lossy at <1%; 1 Mb region
    query in <2 s. Heritability estimates unaffected down to 3 decimal places
    of r, which is the empirical precision budget int16 clears by two orders.
    Its stated limitation -- lossy parameters fixed at creation, with no record
    of them -- is why this format stores its retention predicate and refuses
    queries that fall outside it.
Pan-UKBB LD release (2020), via seq2gwas-upscaler wp3-ld-substrate.md
    Banded upper triangle only, blockSize 4096, 9.7 TB. fp16 storage error
    measured at 4.1-6.8e-4, "two to three orders of magnitude below LD's own
    estimation error". int16 fixed-point beats it at the same two bytes, since
    r is bounded on [-1, 1].
"""
from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

MAGIC = b"CUGENLD1"
HEADER_SIZE = 256
FORMAT_VERSION = 1

# ---------------------------------------------------------------------------
# Value encodings. r is bounded on [-1, 1], so a fixed-point scale beats a
# float of the same width: int16 gives a uniform 3.05e-5 everywhere, where
# fp16 spends its precision near zero and loses it near +/-1.
# ---------------------------------------------------------------------------
ENCODINGS = ("int16", "float32", "int8")
DEFAULT_ENCODING = "int16"
_ENC_CODE = {"int16": 0, "float32": 1, "int8": 2}
_ENC_NAME = {v: k for k, v in _ENC_CODE.items()}
_ENC_DTYPE = {"int16": np.int16, "float32": np.float32, "int8": np.int8}
_ENC_SCALE = {"int16": 32767.0, "float32": 1.0, "int8": 127.0}

PAYLOADS = ("banded", "scatter")
_PAY_CODE = {"banded": 0, "scatter": 1}
_PAY_NAME = {v: k for k, v in _PAY_CODE.items()}

# zstd through pyarrow rather than a new dependency
_CODEC = "zstd"

# r^2 tier edges, descending. A block is cut by VALUE as well as by position,
# because a zone map over position alone does not skip: strong LD is spread
# across the variant axis rather than clustered into contiguous row ranges, so
# on a sparse trans-like fixture a max-|r| map read 94 of 95 blocks. Cutting by
# tier makes each block value-homogeneous, and a threshold query then touches
# only the tiers it needs.
DEFAULT_TIERS = (0.8, 0.5, 0.2, 0.05, 0.0)

# A block is also capped by PAIRS, not only by row variants. zstd frames are not
# seekable, so a single-variant lookup pays for the whole block it lands in: at
# block_variants=4096 on real chr22 a block held 2.5 M pairs and variant() cost
# 360 ms. Capping pairs bounds that cost directly.
MAX_BLOCK_PAIRS = 1 << 16


def _check_encoding(encoding: str) -> str:
    if encoding not in ENCODINGS:
        raise ValueError(
            f"unknown encoding {encoding!r}; valid are {list(ENCODINGS)}. "
            f"r is bounded on [-1, 1], so these are fixed-point scales, not "
            f"float formats -- fp16 is deliberately absent because int16 beats "
            f"it at the same width for a bounded value.")
    return encoding


def quantize_r(r, encoding: str = DEFAULT_ENCODING) -> np.ndarray:
    """Signed r -> the stored representation.

    Rounds rather than truncates, and clips to the representable range so that
    r = +/-1 lands exactly on the endpoint instead of wrapping.
    """
    _check_encoding(encoding)
    a = np.asarray(r, dtype=np.float64)
    if encoding == "float32":
        return a.astype(np.float32)
    scale = _ENC_SCALE[encoding]
    lim = float(np.iinfo(_ENC_DTYPE[encoding]).max)
    return np.clip(np.rint(a * scale), -lim, lim).astype(_ENC_DTYPE[encoding])


def dequantize_r(q, encoding: str = DEFAULT_ENCODING) -> np.ndarray:
    """The stored representation -> signed r as float64."""
    _check_encoding(encoding)
    a = np.asarray(q)
    if encoding == "float32":
        return a.astype(np.float64)
    return a.astype(np.float64) / _ENC_SCALE[encoding]


def r_scale_for(encoding: str = DEFAULT_ENCODING) -> float:
    return _ENC_SCALE[_check_encoding(encoding)]


# ---------------------------------------------------------------------------
# Header. Fixed 256 bytes, little-endian, magic + version + explicit section
# offsets that a reader honours rather than recomputes -- the same contract the
# .cugen header states, for the same reason: layout may change, offsets may not.
# ---------------------------------------------------------------------------
_H = [
    #  offset, struct code, field
    (8, "<I", "version"),
    (12, "<I", "_encoding"),
    (16, "<I", "_payload"),
    (20, "<I", "flags"),
    (24, "<Q", "n_row_variants"),
    (32, "<Q", "n_pairs"),
    (40, "<Q", "index_offset"),
    (48, "<Q", "blocks_offset"),
    (56, "<Q", "footer_offset"),
    (64, "<d", "r_scale"),
    (72, "<q", "block_a"),
    (80, "<q", "block_b"),
]


def pack_header(*, version: int = FORMAT_VERSION, encoding: str,
                payload: str, n_row_variants: int, n_pairs: int,
                index_offset: int, blocks_offset: int, footer_offset: int,
                r_scale: float, block_a: int, block_b: int,
                flags: int = 0) -> bytes:
    """Serialise a shard header into exactly HEADER_SIZE bytes."""
    _check_encoding(encoding)
    if payload not in PAYLOADS:
        raise ValueError(f"unknown payload {payload!r}; valid are {list(PAYLOADS)}")
    buf = bytearray(HEADER_SIZE)
    buf[0:8] = MAGIC
    vals = dict(version=version, _encoding=_ENC_CODE[encoding],
                _payload=_PAY_CODE[payload], flags=flags,
                n_row_variants=n_row_variants, n_pairs=n_pairs,
                index_offset=index_offset, blocks_offset=blocks_offset,
                footer_offset=footer_offset, r_scale=r_scale,
                block_a=block_a, block_b=block_b)
    for off, code, name in _H:
        struct.pack_into(code, buf, off, vals[name])
    return bytes(buf)


def parse_header(buf: bytes) -> dict:
    """Inverse of pack_header. Refuses a foreign magic or a future version."""
    if len(buf) < HEADER_SIZE:
        raise ValueError(
            f"header is {len(buf)} bytes, need {HEADER_SIZE}; this is not a "
            f".cugenld shard")
    if bytes(buf[0:8]) != MAGIC:
        raise ValueError(
            f"bad magic {bytes(buf[0:8])!r}, expected {MAGIC!r} -- this is not "
            f"a .cugenld shard")
    out = {}
    for off, code, name in _H:
        (out[name],) = struct.unpack_from(code, buf, off)
    if out["version"] > FORMAT_VERSION:
        raise ValueError(
            f"file declares .cugenld version {out['version']} but this build "
            f"understands up to {FORMAT_VERSION}. Refusing rather than "
            f"guessing at a layout that may have moved.")
    out["encoding"] = _ENC_NAME[out.pop("_encoding")]
    out["payload"] = _PAY_NAME[out.pop("_payload")]
    return out


# ---------------------------------------------------------------------------
# Delta coding of partner indices. Pays on the diagonal, where a variant's
# partners are its neighbours; does not pay off it, which is what the SCATTER
# payload kind is for.
# ---------------------------------------------------------------------------
_DELTA_DTYPE = {1: np.uint8, 2: np.uint16, 4: np.uint32}


def delta_width_for(j) -> int:
    """Narrowest delta width (1, 2 or 4 bytes) that holds every gap."""
    a = np.asarray(j, dtype=np.int64)
    if a.size < 2:
        return 1
    gap = int(np.diff(a).max())
    for w in (1, 2, 4):
        if gap <= np.iinfo(_DELTA_DTYPE[w]).max:
            return w
    return 8


def delta_encode(j, width: int) -> Tuple[np.ndarray, Optional[int]]:
    """Gaps between consecutive partner indices, at the given byte width.

    Returns ``(deltas, width)``, or ``(empty, None)`` when the requested width
    cannot hold the gaps -- the caller then widens rather than silently
    wrapping, which would decode to a different variant entirely.
    """
    a = np.asarray(j, dtype=np.int64)
    if a.size and (np.diff(a) < 0).any():
        raise ValueError(
            "delta_encode needs ascending partner indices; a negative gap "
            "would wrap in an unsigned width and decode to the wrong variant")
    if a.size < 2:
        return np.zeros(0, dtype=_DELTA_DTYPE.get(width, np.uint32)), width
    d = np.diff(a)
    if width not in _DELTA_DTYPE or int(d.max()) > np.iinfo(
            _DELTA_DTYPE[width]).max:
        return np.zeros(0, dtype=np.uint32), None
    return d.astype(_DELTA_DTYPE[width]), width


def delta_decode(deltas, first: int, width: int) -> np.ndarray:
    """Inverse of delta_encode, given the first index."""
    d = np.asarray(deltas)
    out = np.empty(d.size + 1, dtype=np.int64)
    out[0] = int(first)
    if d.size:
        np.cumsum(d.astype(np.int64), out=out[1:])
        out[1:] += int(first)
    return out


# ---------------------------------------------------------------------------
# Blocks. Independently compressed so a region or threshold query decompresses
# only what it needs, with the zone map (min_r / max_r) in the metadata so a
# threshold query can skip a block without touching its bytes at all.
# ---------------------------------------------------------------------------
def encode_block(j, r, *, encoding: str = DEFAULT_ENCODING,
                 payload: str = "banded", row_starts=None) -> Tuple[bytes, dict]:
    """One block of (partner index, r) into a compressed blob plus metadata.

    A block spans several ROW variants, and j restarts at each one, so the
    sequence is only piecewise ascending. Deltas therefore reset at every row
    boundary and each row's first partner is stored outright -- delta coding
    straight across the block would produce a negative gap at every boundary
    and decode to the wrong variant.
    """
    import pyarrow as pa                                    # noqa: PLC0415

    _check_encoding(encoding)
    ja = np.asarray(j, dtype=np.int64)
    q = quantize_r(r, encoding)
    rs = (np.zeros(1, dtype=np.int64) if row_starts is None
          else np.asarray(row_starts, dtype=np.int64))
    if ja.size and rs.size:
        d = np.empty(ja.size, dtype=np.int64)
        d[0] = 0
        if ja.size > 1:
            d[1:] = np.diff(ja)
        d[rs] = 0                                # reset at each row start
        if (d < 0).any():
            raise ValueError(
                "partner indices must ascend within each row variant; a "
                "negative gap means the block was not sorted by (i, j)")
        firsts = ja[rs]
        gap = int(d.max()) if d.size else 0
    else:
        d = np.zeros(0, dtype=np.int64)
        firsts = np.zeros(0, dtype=np.int64)
        gap = 0

    width = next((w for w in (1, 2, 4)
                  if gap <= np.iinfo(_DELTA_DTYPE[w]).max), 8)
    if payload == "banded" and width == 8:
        payload = "scatter"                      # gaps outgrew every width
    if payload == "banded":
        idx_bytes = d.astype(_DELTA_DTYPE[width]).tobytes() + firsts.tobytes()
        first_len = firsts.nbytes
    else:
        width, first_len = 8, 0
        idx_bytes = ja.tobytes()

    raw = idx_bytes + q.tobytes()
    comp = pa.compress(raw, codec=_CODEC)
    rr = dequantize_r(q, encoding)
    meta = {
        "n": int(ja.size),
        "payload": payload,
        "delta_width": int(width),
        "first": int(ja[0]) if ja.size else 0,
        "raw_len": len(raw),
        "idx_len": len(idx_bytes),
        "first_len": int(first_len),
        "comp_len": len(comp),
        "min_r": float(rr.min()) if rr.size else 0.0,
        "max_r": float(rr.max()) if rr.size else 0.0,
        "max_abs_r": float(np.abs(rr).max()) if rr.size else 0.0,
    }
    return bytes(comp), meta


def decode_block(blob: bytes, meta: dict, *,
                 encoding: str = DEFAULT_ENCODING) -> Tuple[np.ndarray,
                                                            np.ndarray]:
    """Inverse of encode_block. Raises on a truncated or corrupt blob."""
    import pyarrow as pa                                    # noqa: PLC0415

    _check_encoding(encoding)
    raw = bytes(pa.decompress(blob, decompressed_size=meta["raw_len"],
                              codec=_CODEC))
    if len(raw) != meta["raw_len"]:
        raise ValueError(
            f"block decompressed to {len(raw)} bytes, footer says "
            f"{meta['raw_len']} -- refusing to decode a truncated block")
    n = meta["n"]
    idx_len, first_len = meta["idx_len"], meta.get("first_len", 0)
    if meta["payload"] == "banded":
        w = meta["delta_width"]
        d = np.frombuffer(raw[:idx_len - first_len], dtype=_DELTA_DTYPE[w])
        firsts = np.frombuffer(raw[idx_len - first_len:idx_len], dtype=np.int64)
        starts = np.asarray(meta.get("row_starts", [0]), dtype=np.int64)
        if n:
            c = np.cumsum(d.astype(np.int64))
            j = c + np.repeat(firsts - c[starts],
                              np.diff(np.append(starts, n)))
        else:
            j = np.zeros(0, dtype=np.int64)
    else:
        j = np.frombuffer(raw[:idx_len], dtype=np.int64)
    q = np.frombuffer(raw[idx_len:], dtype=_ENC_DTYPE[encoding])
    if j.size != n or q.size != n:
        raise ValueError(
            f"block holds {j.size} indices and {q.size} values but declares "
            f"n={n}")
    return np.asarray(j, dtype=np.int64), dequantize_r(q, encoding)


# ---------------------------------------------------------------------------
# Shard writer. Streaming: append() is called once per scan tile and nothing
# accumulates beyond one block, so peak memory is a block rather than a result.
# A shard covers one contiguous run of ROW variants, which is what the (A, B)
# tile key gives you -- so the monotone-i requirement below is satisfied by
# construction by the scan, and violating it means the caller is not tiling.
# ---------------------------------------------------------------------------
_PARAM_KEYS = (
    # test space -- what entered the computation, and so the denominator m
    "maf_min", "maf_max", "window", "window_kb", "min_dist_kb", "max_dist_kb",
    "scope",
    # retention -- what was written, and so what the reader may be asked
    "min_r2", "max_p", "correction", "alpha", "top_k",
    # provenance needed to derive every other statistic from r
    "n_obs", "m_tests",
)


def _run_starts(a) -> np.ndarray:
    """Indices where a SORTED array starts a new value. O(n), no sort.

    np.unique(a, return_index=True) sorts internally, which is wasted on data
    that is already sorted -- and the writer called it once per flush and again
    per block. Measured at 8 M rows: 0.020 s for the outer call and 0.066 s for
    the per-block ones, 14% of all host write work between them.
    """
    a = np.asarray(a)
    if a.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(
        (np.zeros(1, dtype=np.int64),
         (np.flatnonzero(a[1:] != a[:-1]) + 1).astype(np.int64)))


def _tier_of(r2, tiers) -> np.ndarray:
    """Tier index per row, 0 = strongest. Vectorised over the whole flush.

    The alternative -- five boolean masks and fifteen fancy-indexed copies per
    variant group -- measured 0.242 s at 8 M rows, 51% of all host write work
    and by far the largest single term. One searchsorted plus one stable sort
    replaces it, and the same two calls exist on a GPU.
    """
    edges = np.asarray(sorted(float(t) for t in tiers))      # ascending
    # r2 in [edges[k], edges[k+1]) -> descending tier index
    k = np.searchsorted(edges, np.asarray(r2), side="right") - 1
    return (len(edges) - 1 - np.clip(k, 0, len(edges) - 1)).astype(np.int64)


class LDShardWriter:
    """Append (i, j, r) in row-variant order; get a queryable .cugenld shard.

    Blocks are cut at `block_variants` row variants, compressed independently,
    and their directory is written as a FOOTER -- so a reader can seek to the
    blocks it needs instead of walking the file. Interleaving that directory
    with the payload is what costs Pan-UKBB's BlockMatrix a ~400x read
    amplification.
    """

    def __init__(self, path: str, *, encoding: str = DEFAULT_ENCODING,
                 block_variants: int = 4096, params: Optional[dict] = None,
                 block_a: int = 0, block_b: int = 0, tiers=DEFAULT_TIERS,
                 max_block_pairs: int = MAX_BLOCK_PAIRS):
        _check_encoding(encoding)
        self.tiers = tuple(sorted((float(t) for t in tiers), reverse=True))
        self.path = str(path)
        self.encoding = encoding
        self.block_variants = int(block_variants)
        self.max_block_pairs = int(max_block_pairs)
        self.params = {k: (params or {}).get(k) for k in _PARAM_KEYS}
        self.block_a, self.block_b = int(block_a), int(block_b)
        self._f = open(self.path, "wb")
        self._f.write(b"\x00" * HEADER_SIZE)     # placeholder, rewritten last
        self._blocks: list = []                  # footer entries
        self._buf_i: list = []
        self._buf_j: list = []
        self._buf_r: list = []
        self._buf_rows = 0
        self._block_lo = None                    # first row variant in buffer
        self._last_i = -1
        self.n_pairs = 0
        self._payload_scatter = 0
        self._payload_banded = 0

    # -- writing ------------------------------------------------------------
    def append(self, i, j, r, presorted: bool = False) -> None:
        """One tile's survivors. Any order within the chunk; row variants must
        not go backwards across chunks.

        ``presorted`` skips the host lexsort, which profiling showed to be 88%
        of all host write work (2.03 s of 2.31 s at 8 M rows) -- far more than
        the delta+zstd encode at 9%. A caller holding the data on a GPU should
        sort there instead; cp.lexsort does the same 8 M rows in 0.031 s warm.

        After that change the remaining host term is the per-block pack loop
        (0.130 s at 8 M rows), not compression -- zstd itself is 0.005 s. See
        benchmarks/results/STORAGE.md.
        """
        ia = np.asarray(i, dtype=np.int64).ravel()
        if ia.size == 0:
            return
        ja = np.asarray(j, dtype=np.int64).ravel()
        ra = np.asarray(r, dtype=np.float64).ravel()
        if not (ia.size == ja.size == ra.size):
            raise ValueError(
                f"append got {ia.size} i, {ja.size} j and {ra.size} r values")
        lo = int(ia.min())
        if lo < self._last_i:
            raise ValueError(
                f"append is out of order: row variant {lo} follows {self._last_i}. "
                f"Blocks are indexed by row variant, so a decreasing append "
                f"would need a global sort -- at 1e12 rows that is a bigger job "
                f"than the LD scan. Shard by variant block instead.")
        if self._block_lo is None:
            self._block_lo = lo
        self._buf_i.append(ia)
        self._buf_j.append(ja)
        self._buf_r.append(ra)
        self._buf_rows += ia.size
        self._presorted = getattr(self, "_presorted", True) and presorted
        self._last_i = max(self._last_i, int(ia.max()))
        if self._last_i - self._block_lo + 1 >= self.block_variants:
            self._flush()

    def _flush(self, final: bool = False) -> None:
        """Cut the buffer into blocks of `block_variants` row variants.

        One block per flush would defeat the point: a block is the unit of both
        compression and zone-map skipping, so a single giant block means a
        threshold query decompresses everything and a region query reads the
        whole shard.

        Unless `final`, the highest row variant in the buffer is held back --
        appends are monotone but not exclusive, so a later tile can still add
        partners to it, and a block must contain a row variant's partners
        entirely or `variant()` would return a fragment.
        """
        if not self._buf_rows:
            return
        i = np.concatenate(self._buf_i)
        j = np.concatenate(self._buf_j)
        r = np.concatenate(self._buf_r)
        # A single pre-sorted append needs no re-sort. More than one does: the
        # concatenation of two sorted runs is not sorted.
        if not (getattr(self, "_presorted", False) and len(self._buf_i) == 1):
            order = np.lexsort((j, i))
            i, j, r = i[order], j[order], r[order]

        starts = _run_starts(i)
        uniq = i[starts]
        counts = np.diff(np.append(starts, i.size))
        n_emit = uniq.size if final else uniq.size - 1
        if n_emit <= 0:
            self._buf_i, self._buf_j, self._buf_r = [i], [j], [r]
            self._buf_rows = i.size
            return

        # Split off the held-back tail FIRST, before any reordering: the
        # highest row variant may still receive partners from a later tile, and
        # a block must hold a variant's partners entirely.
        cut = int(starts[n_emit]) if n_emit < uniq.size else int(i.size)
        tail_i, tail_j, tail_r = i[cut:], j[cut:], r[cut:]
        i, j, r = i[:cut], j[:cut], r[:cut]

        # Partition each variant group into r^2 tiers with ONE permutation and
        # three gathers, instead of five boolean masks and fifteen fancy-indexed
        # copies. flatnonzero returns ascending indices, so the (i, j) order
        # already present inside each tier survives without a sort.
        #
        # Measured at 8 M rows, four ways:
        #   masks + 15 gathers (the obvious form)          0.246 s
        #   one global stable argsort over the flush       0.396 s
        #   per-group stable argsort, 3 gathers            0.350 s
        #   this: per-group flatnonzero perm, 3 gathers    0.103 s
        # A full argsort of 8 M int64 costs more than the twenty linear passes
        # it replaces, so the sort-based forms are slower than doing nothing.
        ntier = len(self.tiers)
        tier_lo = sorted((float(t) for t in self.tiers), reverse=True)
        for lo in range(0, n_emit, self.block_variants):
            hi = min(lo + self.block_variants, n_emit)
            s0 = int(starts[lo])
            s1 = int(starts[hi]) if hi < uniq.size else int(i.size)
            if s1 <= s0:
                continue
            gr = r[s0:s1]
            tl = _tier_of(gr * gr, self.tiers)
            parts = [np.flatnonzero(tl == k) for k in range(ntier)]
            perm = np.concatenate(parts) if ntier > 1 else parts[0]
            pi, pj, pr = i[s0:s1][perm], j[s0:s1][perm], gr[perm]

            off = 0
            for k, part in enumerate(parts):
                m = part.size
                if m == 0:
                    continue
                ti = pi[off:off + m]
                tj = pj[off:off + m]
                tr = pr[off:off + m]
                off += m
                # split on row-variant boundaries so a variant's partners never
                # straddle two blocks within a tier
                if m > self.max_block_pairs:
                    rs = _run_starts(ti)
                    cutp = [0]
                    for st in rs[1:]:
                        if st - cutp[-1] >= self.max_block_pairs:
                            cutp.append(int(st))
                    cutp.append(m)
                    for a, b in zip(cutp[:-1], cutp[1:]):
                        if b > a:
                            self._write_block(ti[a:b], tj[a:b], tr[a:b],
                                              tier_lo=tier_lo[k])
                else:
                    self._write_block(ti, tj, tr, tier_lo=tier_lo[k])

        if tail_i.size:
            self._buf_i, self._buf_j, self._buf_r = (
                [tail_i], [tail_j], [tail_r])
            self._buf_rows = int(tail_i.size)
            self._block_lo = int(tail_i[0])
            self._presorted = True          # the tail is still (i, j) ordered
        else:
            self._buf_i, self._buf_j, self._buf_r = [], [], []
            self._buf_rows = 0
            self._block_lo = None

    def _write_block(self, i, j, r, *, tier_lo: float) -> None:
        """One compressed block for one (row-variant group, r^2 tier)."""
        starts = _run_starts(i)
        uniq = i[starts]
        counts = np.diff(np.append(starts, i.size))
        blob, meta = encode_block(j, r, encoding=self.encoding,
                                  row_starts=starts.astype(np.int64))
        if meta["payload"] == "scatter":
            self._payload_scatter += 1
        else:
            self._payload_banded += 1
        off = self._f.tell()
        self._f.write(blob)
        meta.update(offset=off, tier_lo=float(tier_lo),
                    row_variants=uniq.astype(np.int64).tolist(),
                    row_starts=starts.astype(np.int64).tolist(),
                    row_counts=counts.astype(np.int64).tolist())
        self._blocks.append(meta)
        self.n_pairs += int(i.size)

    def close(self) -> None:
        import json                                          # noqa: PLC0415

        self._flush(final=True)
        footer_off = self._f.tell()
        footer = json.dumps({
            "blocks": self._blocks,
            "params": self.params,
            "encoding": self.encoding,
            "block_variants": self.block_variants,
            "payload_mix": {"banded": self._payload_banded,
                            "scatter": self._payload_scatter},
        }, separators=(",", ":")).encode()
        self._f.write(footer)
        n_rv = sum(len(b["row_variants"]) for b in self._blocks)
        self._f.seek(0)
        self._f.write(pack_header(
            encoding=self.encoding,
            payload="scatter" if self._payload_scatter else "banded",
            n_row_variants=n_rv, n_pairs=self.n_pairs,
            index_offset=footer_off, blocks_offset=HEADER_SIZE,
            footer_offset=footer_off, r_scale=r_scale_for(self.encoding),
            block_a=self.block_a, block_b=self.block_b))
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Reader. Header and footer only up front; block payloads on demand.
# ---------------------------------------------------------------------------
class LDReader:
    """Query a .cugenld shard.

    Refuses questions the file cannot answer rather than under-reporting. A
    file written at min_r2 = 0.2 has discarded everything below it, so
    above(0.05) has no short answer -- it has no answer.
    """

    def __init__(self, path: str):
        import json                                          # noqa: PLC0415

        self.path = str(path)
        with open(self.path, "rb") as f:
            self.header = parse_header(f.read(HEADER_SIZE))
            f.seek(self.header["footer_offset"])
            foot = json.loads(f.read().decode())
        self.blocks = foot["blocks"]
        self.params = foot["params"]
        self.encoding = foot["encoding"]
        self.payload_mix = foot["payload_mix"]
        self.n_pairs = self.header["n_pairs"]
        self.blocks_read = 0
        # row variant -> [(block index, slot)]. Built once, because scanning
        # every block's row_variants list per lookup made variant() cost 367 ms
        # on a 66-block file -- linear in blocks and in Python. A variant's
        # partners are spread across r^2 tiers, so a variant maps to several
        # blocks and this has to be a multimap.
        self._where: dict = {}
        for bi, b in enumerate(self.blocks):
            for slot, v in enumerate(b["row_variants"]):
                self._where.setdefault(int(v), []).append((bi, slot))

    def reset_counters(self) -> None:
        self.blocks_read = 0

    # -- internals ----------------------------------------------------------
    def _block(self, b: dict):
        with open(self.path, "rb") as f:
            f.seek(b["offset"])
            blob = f.read(b["comp_len"])
        self.blocks_read += 1
        return decode_block(blob, b, encoding=self.encoding)

    def _stored_min_r2(self) -> float:
        v = self.params.get("min_r2")
        return 0.0 if v is None else float(v)

    def _check_threshold(self, min_r2: Optional[float]) -> Optional[float]:
        """None means "no filter beyond what the file already applied".

        It must NOT mean "re-apply the stored cut": r is quantised, so a pair
        stored at true r^2 = 0.05000001 can come back with stored r^2 just under
        0.05, and re-filtering would silently drop pairs the file contains.
        """
        stored = self._stored_min_r2()
        if min_r2 is None:
            return None
        t = float(min_r2)
        if t < stored - 1e-12:
            raise ValueError(
                f"this file was written at min_r2={stored:g}; every pair below "
                f"that was discarded, so min_r2={t:g} cannot be answered from "
                f"it. Returning the pairs that happen to be present would be a "
                f"wrong answer, not a partial one. Re-run the scan at the "
                f"looser threshold.")
        return t

    # -- queries ------------------------------------------------------------
    def rows(self):
        """Every stored pair, as (i, j, r)."""
        return self.above(min_r2=None)

    def above(self, min_r2: Optional[float] = None,
              max_p: Optional[float] = None):
        """Pairs at or above a threshold, skipping blocks by their zone map.

        p is monotone in r^2 at fixed N, so a p-value cut is the same skip: it
        is converted to the equivalent r^2 using n_obs from the header.
        """
        if max_p is not None:
            n = self.params.get("n_obs")
            if not n:
                raise ValueError(
                    "max_p needs n_obs, which this file does not record")
            from scipy.stats import chi2 as _c                # noqa: PLC0415
            min_r2 = max(min_r2 or 0.0, float(_c.isf(max_p, 1)) / float(n))
        t = self._check_threshold(min_r2)
        oi, oj, orr = [], [], []
        for b in self.blocks:
            # zone map: nothing in this block can clear the cut, so its bytes
            # are never touched. Blocks are cut by tier as well as position, so
            # this actually skips instead of being decorative.
            if t is not None and t > 0.0 and b["max_abs_r"] ** 2 < t:
                continue
            j, r = self._block(b)
            i = np.repeat(np.asarray(b["row_variants"], dtype=np.int64),
                          np.asarray(b["row_counts"], dtype=np.int64))
            # a whole tier at or above the cut needs no per-pair filter
            if t is not None and t > 0.0 and b.get("tier_lo", -1.0) < t:
                keep = r ** 2 >= t
                i, j, r = i[keep], j[keep], r[keep]
            oi.append(i)
            oj.append(j)
            orr.append(r)
        if not oi:
            e = np.zeros(0, dtype=np.int64)
            return e, e.copy(), np.zeros(0)
        return np.concatenate(oi), np.concatenate(oj), np.concatenate(orr)

    def variant(self, gidx: int):
        """One variant's partners, as (j, r), ascending in j.

        A variant's partners are spread across tiers, so this gathers every
        block that lists it -- typically a handful, not the whole shard.
        """
        v = int(gidx)
        oj, orr = [], []
        for bi, slot in self._where.get(v, ()):
            b = self.blocks[bi]
            j, r = self._block(b)
            s = b["row_starts"][slot]
            e = s + b["row_counts"][slot]
            oj.append(j[s:e])
            orr.append(r[s:e])
        if not oj:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        j = np.concatenate(oj)
        r = np.concatenate(orr)
        o = np.argsort(j, kind="stable")
        return j[o], r[o]

    def dense(self, n_variants: Optional[int] = None) -> np.ndarray:
        """Symmetric dense r matrix, unit diagonal.

        Only available on an unthresholded file. A thresholded store cannot
        produce this: every absent entry would become 0 rather than its true
        small value, and SuSiE would then be handed a matrix that is wrong in a
        way nothing downstream can detect.
        """
        stored = self._stored_min_r2()
        if stored > 0.0:
            raise ValueError(
                f"dense() needs an unthresholded file, but this one was written "
                f"at min_r2={stored:g}. Filling the missing entries with 0 "
                f"would silently replace every sub-threshold correlation with "
                f"zero instead of its true value. Re-run with min_r2=0.")
        i, j, r = self.rows()
        p = int(n_variants) if n_variants else int(max(i.max(), j.max()) + 1)
        R = np.eye(p, dtype=np.float64)
        R[i, j] = r
        R[j, i] = r
        return R

    def bytes_per_pair(self) -> float:
        import os                                            # noqa: PLC0415
        return os.path.getsize(self.path) / max(self.n_pairs, 1)


def read_ld(path: str) -> LDReader:
    """Open a .cugenld shard for querying."""
    return LDReader(path)


# ---------------------------------------------------------------------------
# Writer registry. The point is options: some people want a compact native
# container, some want .npz they can open in one line with no new dependency,
# some want parquet a downstream service already queries. All of them are
# served, and the extension picks.
# ---------------------------------------------------------------------------
# Placeholder columns the device path fills with constants: POS is literal
# zeros and ID is "." whenever annotation is None, which is exactly the
# condition the fused path requires. Writing them is pure overhead.
_DEAD_WHEN_PLACEHOLDER = ("POS_A", "POS_B", "ID_A", "ID_B")

# One row group per this many rows. Small enough that a bp-range predicate
# skips most of the file, large enough that per-group metadata stays cheap.
PARQUET_ROW_GROUP = 1 << 20


def _to_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _drop_placeholder_columns(df):
    """Drop columns that carry no information in this frame."""
    out = df
    for c in _DEAD_WHEN_PLACEHOLDER:
        if c not in out.columns:
            continue
        col = out[c]
        if c.startswith("POS"):
            dead = bool((col == 0).all())
        else:
            dead = bool((col.astype(str) == ".").all())
        if dead:
            out = out.drop(columns=[c])
    for c in ("CHR_A", "CHR_B"):
        if c in out.columns and out[c].nunique(dropna=False) <= 1:
            out = out.drop(columns=[c])
    return out


def write_ld(df, path: str, *, params: Optional[dict] = None,
             encoding: str = DEFAULT_ENCODING, block_variants: int = 4096,
             drop_dead: bool = False, tiers=DEFAULT_TIERS) -> None:
    """Write an LD pairs frame, in the format the extension asks for.

    ``.cugenld`` is the compact native container; the rest are interop formats.
    ``params`` records the test-space and retention parameters of the run --
    the native format needs them to answer queries honestly, and the others
    keep them alongside where the container allows it.
    """
    p = str(path)
    if p.endswith(".cugenld"):
        return _write_cugenld(df, p, params=params, encoding=encoding,
                              block_variants=block_variants, tiers=tiers)
    if p.endswith(".npz"):
        return _write_npz(df, p, params=params)
    if p.endswith(".zarr"):
        return _write_zarr(df, p, params=params)
    if p.endswith(".parquet"):
        return _write_parquet(df, p, drop_dead=drop_dead)
    if p.endswith(".feather"):
        h = _to_pandas(df)
        return (_drop_placeholder_columns(h) if drop_dead else h).reset_index(
            drop=True).to_feather(p)
    if p.endswith((".tsv", ".csv", ".tsv.gz", ".csv.gz", ".gz")):
        return _write_text(df, p, drop_dead=drop_dead)
    raise ValueError(
        f"no writer for {p!r}: unrecognised extension. Supported formats are "
        f".cugenld (compact native), .parquet, .feather, .npz, .zarr, .tsv, "
        f".csv, and the .gz variants of the last two.")


def _r_column(h):
    for c in ("R", "R_PHASED"):
        if c in h.columns:
            return c
    raise ValueError(
        "the native and array formats store signed r, from which r2, D, chi2 "
        "and p are all derived -- so 'r' or 'r_phased' must be in stats=. Got "
        f"columns {list(h.columns)}.")


def _write_cugenld(df, path, *, params, encoding, block_variants, tiers):
    h = _to_pandas(df)
    rc = _r_column(h)
    w = LDShardWriter(path, encoding=encoding, block_variants=block_variants,
                      params=params, tiers=tiers)
    i = h["gidx_a"].to_numpy(np.int64)
    j = h["gidx_b"].to_numpy(np.int64)
    r = h[rc].to_numpy(np.float64)
    order = np.lexsort((j, i))
    w.append(i[order], j[order], r[order])
    w.close()


def _write_npz(df, path, *, params):
    import json                                              # noqa: PLC0415

    h = _to_pandas(df)
    arrays = {c: h[c].to_numpy() for c in h.columns
              if h[c].dtype != object}
    arrays["params_json"] = np.array(json.dumps(params or {}))
    np.savez_compressed(path, **arrays)


def _write_zarr(df, path, *, params):
    import json                                              # noqa: PLC0415
    try:
        import zarr                                          # noqa: PLC0415
    except ImportError as e:                                 # noqa: BLE001
        raise ImportError(
            "the .zarr backend needs the optional 'zarr' package: "
            "pip install zarr") from e

    h = _to_pandas(df)
    g = zarr.open_group(path, mode="w")
    for c in h.columns:
        if h[c].dtype == object:
            continue
        g.create_array(c, shape=(len(h),), dtype=h[c].dtype,
                       chunks=(min(len(h), 1 << 20),))[:] = h[c].to_numpy()
    g.attrs["params"] = json.dumps(params or {})


def _write_parquet(df, path, *, drop_dead):
    import pyarrow as pa                                     # noqa: PLC0415
    import pyarrow.parquet as pq                             # noqa: PLC0415

    h = _to_pandas(df)
    if drop_dead:
        h = _drop_placeholder_columns(h)
    tbl = pa.Table.from_pandas(h, preserve_index=False)
    # zstd over snappy (19.0 vs 24.3 B/row measured), statistics on so a
    # bp-range predicate can skip row groups instead of reading all of them,
    # and dictionary encoding off for the wide integer keys where it only adds
    # a level of indirection.
    pq.write_table(tbl, path, compression="zstd",
                   row_group_size=PARQUET_ROW_GROUP, write_statistics=True,
                   use_dictionary=False)


def _write_text(df, path, *, drop_dead):
    h = _to_pandas(df)
    if drop_dead:
        h = _drop_placeholder_columns(h)
    sep = "\t" if ".tsv" in path else ","
    if not path.endswith(".gz"):
        try:
            import pyarrow as pa                             # noqa: PLC0415
            from pyarrow import csv as pacsv                  # noqa: PLC0415
            pacsv.write_csv(
                pa.Table.from_pandas(h, preserve_index=False), path,
                pacsv.WriteOptions(include_header=True, delimiter=sep,
                                   quoting_style="none"))
            return
        except Exception:                                    # noqa: BLE001
            pass
    h.to_csv(path, sep=sep, index=False,
             compression="gzip" if path.endswith(".gz") else None)


def read_pairs(path: str):
    """Read any container this module writes back into a pairs DataFrame.

    Exists so a round-trip is testable across every backend with one call.
    """
    import pandas as pd                                      # noqa: PLC0415

    p = str(path)
    if p.endswith(".cugenld"):
        rd = read_ld(p)
        i, j, r = rd.rows()
        return pd.DataFrame({"gidx_a": i, "gidx_b": j, "R": r})
    if p.endswith(".npz"):
        z = np.load(p, allow_pickle=False)
        return pd.DataFrame({k: z[k] for k in z.files
                             if k != "params_json" and z[k].ndim == 1})
    if p.endswith(".zarr"):
        import zarr                                          # noqa: PLC0415
        g = zarr.open_group(p, mode="r")
        return pd.DataFrame({k: g[k][:] for k in g.array_keys()})
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    if p.endswith(".feather"):
        return pd.read_feather(p)
    if p.endswith((".tsv", ".tsv.gz")):
        return pd.read_csv(p, sep="\t")
    if p.endswith((".csv", ".csv.gz")):
        return pd.read_csv(p)
    raise ValueError(f"no reader for {p!r}")


# ---------------------------------------------------------------------------
# Sharded datasets. A shard is one scan tile's output, keyed by the (A, B)
# variant-block pair the scan already walks, so shards are written
# independently -- concurrently, across GPUs, and across a resumed run -- with
# no cross-shard coordination and no global sort.
#
# Resumability is not a convenience at this scale. A genome-wide all-by-all run
# is hours of GPU time, and on spot or preemptible capacity a format that forces
# all-or-nothing writes makes the job impractical. The manifest is the record of
# what exists; a shard lands by atomic rename, so a process killed mid-write
# leaves a temp file that no reader ever sees.
# ---------------------------------------------------------------------------
MANIFEST = "manifest.json"


def _shard_name(key) -> str:
    a, b = key
    return f"{int(a)}-{int(b)}.ldz"


class LDDatasetWriter:
    """Write a sharded LD dataset; resume one that was interrupted.

    ``write_shard(key, i, j, r)`` is the whole interface. Each call produces one
    self-contained, independently queryable shard and appends it to the
    manifest, so progress is durable at shard granularity rather than at the end
    of the run.
    """

    def __init__(self, path: str, *, params: Optional[dict] = None,
                 encoding: str = DEFAULT_ENCODING, block_variants: int = 4096,
                 max_block_pairs: int = MAX_BLOCK_PAIRS, tiers=DEFAULT_TIERS,
                 resume: bool = False):
        import json                                          # noqa: PLC0415
        import os                                            # noqa: PLC0415

        _check_encoding(encoding)
        self.path = str(path)
        self.encoding = encoding
        self.block_variants = int(block_variants)
        self.max_block_pairs = int(max_block_pairs)
        self.tiers = tiers
        self.params = {k: (params or {}).get(k) for k in _PARAM_KEYS}
        os.makedirs(self.path, exist_ok=True)
        man_path = os.path.join(self.path, MANIFEST)

        self._shards: list = []
        self._complete = False
        if resume and os.path.exists(man_path):
            man = json.loads(open(man_path).read())
            if man.get("params") != self.params:
                diff = {k for k in _PARAM_KEYS
                        if man.get("params", {}).get(k) != self.params[k]}
                raise ValueError(
                    f"refusing to resume: the existing dataset was written with "
                    f"different parameters ({sorted(diff)}). The test space sets "
                    f"the number of tests and therefore every corrected "
                    f"threshold, so half a dataset at one setting and half at "
                    f"another is not a dataset. Write to a new path.")
            if man.get("encoding") != self.encoding:
                raise ValueError(
                    f"refusing to resume: existing encoding is "
                    f"{man.get('encoding')!r}, requested {self.encoding!r}")
            # Trust the manifest, not the directory listing: a shard killed
            # mid-write leaves a temp file, and a torn file must never be read.
            self._shards = [sh for sh in man.get("shards", [])
                            if os.path.exists(os.path.join(self.path,
                                                           sh["file"]))]
            self._complete = bool(man.get("complete", False))
        self._flush_manifest()

    # -- writing ------------------------------------------------------------
    def completed_shards(self):
        """Keys already durably written, so a resumed run can skip them."""
        return [tuple(sh["key"]) for sh in self._shards]

    def write_shard(self, key, i, j, r, presorted: bool = False) -> str:
        """One tile's survivors as one shard. Lands by atomic rename."""
        import os                                            # noqa: PLC0415

        name = _shard_name(key)
        final = os.path.join(self.path, name)
        tmp = final + f".tmp{os.getpid()}"
        ia = np.asarray(i, dtype=np.int64).ravel()
        ja = np.asarray(j, dtype=np.int64).ravel()
        ra = np.asarray(r, dtype=np.float64).ravel()
        order = (np.arange(ia.size) if presorted else np.lexsort((ja, ia)))
        w = LDShardWriter(tmp, encoding=self.encoding,
                          block_variants=self.block_variants,
                          max_block_pairs=self.max_block_pairs,
                          params=self.params, tiers=self.tiers,
                          block_a=int(key[0]), block_b=int(key[1]))
        w.append(ia[order], ja[order], ra[order], presorted=True)
        w.close()
        os.replace(tmp, final)                   # atomic: no torn shard is ever
                                                 # visible under its real name
        self._shards = [sh for sh in self._shards
                        if tuple(sh["key"]) != tuple(key)]
        self._shards.append({
            "key": [int(key[0]), int(key[1])], "file": name,
            "n_pairs": int(w.n_pairs),
            "min_i": int(ia.min()) if ia.size else 0,
            "max_i": int(ia.max()) if ia.size else -1,
            "min_j": int(ja.min()) if ja.size else 0,
            "max_j": int(ja.max()) if ja.size else -1,
            "max_abs_r": float(np.abs(ra).max()) if ra.size else 0.0,
        })
        self._flush_manifest()
        return final

    def mark_complete(self) -> None:
        """Record that every planned shard is present."""
        self._complete = True
        self._flush_manifest()

    def _flush_manifest(self) -> None:
        import json                                          # noqa: PLC0415
        import os                                            # noqa: PLC0415

        man = {
            "format": "cugenld", "version": FORMAT_VERSION,
            "encoding": self.encoding, "params": self.params,
            "block_variants": self.block_variants,
            "max_block_pairs": self.max_block_pairs,
            "tiers": list(self.tiers),
            "complete": self._complete,
            "n_pairs": int(sum(sh["n_pairs"] for sh in self._shards)),
            "shards": sorted(self._shards, key=lambda sh: sh["key"]),
        }
        p = os.path.join(self.path, MANIFEST)
        tmp = p + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(json.dumps(man, indent=1))
        os.replace(tmp, p)

    def close(self) -> None:
        self._flush_manifest()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is None:
            self.mark_complete()
        self.close()


class LDDatasetReader:
    """Query a sharded LD dataset. Routes each query to the shards that can
    answer it, using the manifest's per-shard index and r ranges."""

    def __init__(self, path: str):
        import json                                          # noqa: PLC0415
        import os                                            # noqa: PLC0415

        self.path = str(path)
        man_path = os.path.join(self.path, MANIFEST)
        if not os.path.exists(man_path):
            raise ValueError(
                f"{self.path!r} has no {MANIFEST}; it is not a cugenld dataset. "
                f"For a single shard use read_ld() on the .ldz/.cugenld file.")
        self.manifest = json.loads(open(man_path).read())
        self.params = self.manifest["params"]
        self.encoding = self.manifest["encoding"]
        self.complete = bool(self.manifest.get("complete", False))
        self.shards = self.manifest["shards"]
        self.n_pairs = int(self.manifest.get("n_pairs", 0))
        self.blocks_read = 0
        # Shards OPENED, counted separately from blocks decompressed. The
        # manifest-level skip saves file opens, not block decodes, so a
        # blocks_read counter cannot see whether it works -- opening a shard and
        # finding nothing reads zero blocks. Without this the skip is unfalsifiable.
        self.shards_read = 0
        # Opening a shard parses its footer and rebuilds its row-variant index.
        # A cross-shard variant() touches ~10 shards, and re-parsing each time
        # took a GPU-scale lookup from 2.94 ms (single shard) to 21.74 ms. Cache
        # the readers; the footers are small and the files are immutable.
        self._cache: dict = {}
        self._cache_max = 64

    @property
    def n_shards(self) -> int:
        return len(self.shards)

    def reset_counters(self) -> None:
        self.blocks_read = 0
        self.shards_read = 0

    def _open(self, sh) -> LDReader:
        import os                                            # noqa: PLC0415
        self.shards_read += 1
        f = sh["file"]
        rd = self._cache.get(f)
        if rd is None:
            if len(self._cache) >= self._cache_max:
                self._cache.pop(next(iter(self._cache)))
            rd = LDReader(os.path.join(self.path, f))
            self._cache[f] = rd
        rd.reset_counters()
        return rd

    def _gather(self, pick, call):
        oi, oj, orr = [], [], []
        for sh in self.shards:
            if not pick(sh):
                continue
            rd = self._open(sh)
            out = call(rd)
            self.blocks_read += rd.blocks_read
            if out is None:
                continue
            oi.append(out[0])
            oj.append(out[1])
            orr.append(out[2])
        if not oi:
            e = np.zeros(0, dtype=np.int64)
            return e, e.copy(), np.zeros(0)
        return np.concatenate(oi), np.concatenate(oj), np.concatenate(orr)

    # -- queries ------------------------------------------------------------
    def rows(self):
        return self._gather(lambda sh: True, lambda rd: rd.rows())

    def above(self, min_r2: Optional[float] = None,
              max_p: Optional[float] = None):
        """Skips whole SHARDS on the manifest's max_abs_r before opening them,
        then whole blocks on each shard's own zone map."""
        def pick(sh):
            if min_r2 is None or min_r2 <= 0.0:
                return True
            return sh["max_abs_r"] ** 2 >= min_r2
        return self._gather(pick, lambda rd: rd.above(min_r2=min_r2,
                                                      max_p=max_p))

    def variant(self, gidx: int):
        v = int(gidx)
        oj, orr = [], []
        for sh in self.shards:
            if not (sh["min_i"] <= v <= sh["max_i"]):
                continue
            rd = self._open(sh)
            j, r = rd.variant(v)
            self.blocks_read += rd.blocks_read
            if j.size:
                oj.append(j)
                orr.append(r)
        if not oj:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        j = np.concatenate(oj)
        r = np.concatenate(orr)
        o = np.argsort(j, kind="stable")
        return j[o], r[o]

    def region(self, lo: int, hi: int):
        """Every stored pair with both endpoints in [lo, hi)."""
        def pick(sh):
            return not (sh["max_i"] < lo or sh["min_i"] >= hi
                        or sh["max_j"] < lo or sh["min_j"] >= hi)

        def call(rd):
            i, j, r = rd.rows()
            m = (i >= lo) & (i < hi) & (j >= lo) & (j < hi)
            return i[m], j[m], r[m]
        return self._gather(pick, call)

    def bytes_per_pair(self) -> float:
        import os                                            # noqa: PLC0415
        total = sum(os.path.getsize(os.path.join(self.path, sh["file"]))
                    for sh in self.shards)
        return total / max(self.n_pairs, 1)


def open_ld(path: str):
    """Open a sharded dataset (directory) or a single shard (file)."""
    import os                                                # noqa: PLC0415
    if os.path.isdir(path):
        return LDDatasetReader(path)
    return LDReader(path)
