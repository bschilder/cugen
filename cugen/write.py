"""cugen.write - create .cugen files.

The public package could read .cugen but not write one: ``io.write_cugen``
raised NotImplementedError and pointed at legacy ``build_*_cugen.sh`` scripts
that are not in the repository. So anyone outside this cluster could consume
the format but never produce it. This module closes that.

    CugenWriter        streaming writer, one variant at a time
    write_cugen        array -> file convenience wrapper
    validate_cugen     structural check of an existing file

ON-DISK FORMAT (v1) - little-endian throughout
----------------------------------------------
    offset  size  type      field
         0     8  char[8]   magic "CUPGEN01"   (legacy magic, kept post-rename)
         8     4  uint32    version = 1
        12     4  uint32    encoding            0=2bit 1=uint8 2=f16 3=f32
        16     8  uint64    n_samples
        24     8  uint64    n_variants
        32     8  uint64    bytes_per_variant
        40     8  uint64    stats_offset
        48     8  uint64    data_offset
        56     8  uint64    gidx_offset
        64     4  uint32    flags               bit0 HAS_MISSING, bit1 HAS_GIDX
        68   188  --        zero padding to 256

    stats  [stats_offset]  float32 x n_variants x 3, as three CONTIGUOUS arrays
                           in order: mu_x, then sxx, then maf. NOT interleaved.
    gidx   [gidx_offset]   int64 x n_variants (present iff HAS_GIDX)
    data   [data_offset]   n_variants x bytes_per_variant

Section order as written by this module (and by the production converter) is
header, stats, gidx, data, so data_offset = 256 + 12*n_variants + 8*n_variants.
A reader must honour the offsets rather than assume that layout.

2-BIT ENCODING - big-endian WITHIN each byte
--------------------------------------------
Sample i lives in byte i>>2 at bit shift 6-2*(i&3), so sample 0 is the HIGH
pair. Values 0/1/2 are dosages; 3 is missing. bytes_per_variant = ceil(n/4),
and trailing bits in the last byte are zero (which decodes as dosage 0, not
missing - readers must bound loops by n_samples).

PER-VARIANT STATS ARE SAMPLE-SET-SPECIFIC
------------------------------------------
mu_x, sxx and maf are computed over THIS file's samples, over VALID genotypes
only -- finite and within [0, 2], which is the production converter's exact
definition (pgen_to_cugen.py compute_variant_stats). Subsetting samples
invalidates them - recompute (cg.subset /
cg.repair) rather than copying them across. This is the single most common way
to corrupt a .cugen silently: stale stats make beta = num/sxx meaningless while
every structural check still passes.
"""
import os
import struct

import numpy as np

CUGEN_MAGIC = b"CUPGEN01"
CUGEN_VERSION = 1
HEADER_SIZE = 256

ENCODING_2BIT = 0
ENCODING_UINT8 = 1
ENCODING_FLOAT16 = 2
ENCODING_FLOAT32 = 3

FLAG_HAS_MISSING = 1
FLAG_HAS_GIDX_MAP = 2

_ENC_BYTES = {ENCODING_2BIT: lambda n: (n + 3) // 4,
              ENCODING_UINT8: lambda n: n,
              ENCODING_FLOAT16: lambda n: n * 2,
              ENCODING_FLOAT32: lambda n: n * 4}


def pack_2bit(geno_u8):
    """Pack uint8 genotypes (0,1,2,3) into 2-bit, big-endian within byte."""
    g = np.asarray(geno_u8, dtype=np.uint8)
    n = g.size
    pad = (-n) % 4
    if pad:
        g = np.concatenate([g, np.zeros(pad, dtype=np.uint8)])
    r = g.reshape(-1, 4)
    return ((r[:, 0] << 6) | (r[:, 1] << 4) | (r[:, 2] << 2) | r[:, 3]).astype(np.uint8)


def unpack_2bit(packed, n_samples):
    """Inverse of pack_2bit; returns uint8 with 3 = missing."""
    b = np.frombuffer(packed, dtype=np.uint8)
    out = np.empty(b.size * 4, dtype=np.uint8)
    out[0::4] = (b >> 6) & 3
    out[1::4] = (b >> 4) & 3
    out[2::4] = (b >> 2) & 3
    out[3::4] = b & 3
    return out[:n_samples]


def variant_stats(dosages):
    """mu_x, sxx, maf over VALID samples (finite, in [0, 2]).

    sxx is the centred sum of squares of the non-missing dosages, which is what
    the association code divides by; maf is folded to <= 0.5.
    """
    d = np.asarray(dosages, dtype=np.float64)
    # Valid mask matches the production converter (pgen_to_cugen.py
    # compute_variant_stats) EXACTLY: finite and within [0, 2]. That excludes
    # the 2-bit missing code 3, but also excludes out-of-range dosages, which
    # `d != 3` would silently have let through for the float encodings.
    ok = np.isfinite(d) & (d >= 0) & (d <= 2)
    n = int(ok.sum())
    if n == 0:
        return 0.0, 0.0, 0.0, True
    x = d[ok]
    mu = float(x.mean())
    sxx = float(((x - mu) ** 2).sum())
    af = mu / 2.0
    maf = float(min(af, 1.0 - af))
    return mu, sxx, maf, bool(n < d.size)


class CugenWriter:
    """Streaming .cugen writer.

    Usage:
        with CugenWriter(path, n_samples, n_variants) as w:
            for gidx, dosages in source:
                w.add_variant(gidx, dosages)

    Variants must be added in the order they should appear. Stats, gidx and the
    header are written on close, so the file is only valid after the context
    manager exits.
    """

    def __init__(self, path, n_samples, n_variants, encoding=ENCODING_2BIT):
        if encoding not in _ENC_BYTES:
            raise ValueError(f"unknown encoding {encoding}")
        self.path = str(path)
        self.n_samples = int(n_samples)
        self.n_variants = int(n_variants)
        self.encoding = int(encoding)
        self.bytes_per_variant = _ENC_BYTES[encoding](self.n_samples)

        self.stats_offset = HEADER_SIZE
        self.stats_size = self.n_variants * 4 * 3
        self.gidx_offset = self.stats_offset + self.stats_size
        self.gidx_size = self.n_variants * 8
        self.data_offset = self.gidx_offset + self.gidx_size

        self.mu_x = np.zeros(self.n_variants, dtype=np.float32)
        self.sxx = np.zeros(self.n_variants, dtype=np.float32)
        self.maf = np.zeros(self.n_variants, dtype=np.float32)
        self.gidx = np.zeros(self.n_variants, dtype=np.int64)
        self.flags = FLAG_HAS_GIDX_MAP
        self.i = 0
        self.f = None

    def __enter__(self):
        self.f = open(self.path, "wb")
        self.f.write(b"\x00" * (HEADER_SIZE + self.stats_size + self.gidx_size))
        return self

    def add_variant(self, gidx, dosages):
        if self.i >= self.n_variants:
            raise IndexError(f"more than the declared {self.n_variants} variants")
        d = np.asarray(dosages)
        if d.size != self.n_samples:
            raise ValueError(f"variant {self.i}: {d.size} dosages != "
                             f"n_samples {self.n_samples}")
        mu, sxx, maf, has_missing = variant_stats(d)
        self.mu_x[self.i] = mu
        self.sxx[self.i] = sxx
        self.maf[self.i] = maf
        self.gidx[self.i] = int(gidx)
        if has_missing:
            self.flags |= FLAG_HAS_MISSING

        if self.encoding == ENCODING_2BIT:
            g = np.asarray(d, dtype=np.float64)
            u = np.where(np.isfinite(g), np.rint(np.nan_to_num(g, nan=3.0)), 3.0)
            u = np.clip(u, 0, 3).astype(np.uint8)
            self.f.write(pack_2bit(u).tobytes())
        elif self.encoding == ENCODING_UINT8:
            u = np.clip(np.nan_to_num(np.asarray(d), nan=3.0), 0, 3).astype(np.uint8)
            self.f.write(u.tobytes())
        elif self.encoding == ENCODING_FLOAT16:
            self.f.write(np.asarray(d, dtype=np.float16).tobytes())
        else:
            self.f.write(np.asarray(d, dtype=np.float32).tobytes())
        self.i += 1

    def _finalize(self):
        if self.i != self.n_variants:
            raise ValueError(f"declared {self.n_variants} variants, wrote {self.i}")
        self.f.seek(self.stats_offset)
        self.f.write(self.mu_x.tobytes())
        self.f.write(self.sxx.tobytes())
        self.f.write(self.maf.tobytes())
        self.f.seek(self.gidx_offset)
        self.f.write(self.gidx.tobytes())

        h = bytearray(HEADER_SIZE)
        h[0:8] = CUGEN_MAGIC
        struct.pack_into("<I", h, 8, CUGEN_VERSION)
        struct.pack_into("<I", h, 12, self.encoding)
        struct.pack_into("<Q", h, 16, self.n_samples)
        struct.pack_into("<Q", h, 24, self.n_variants)
        struct.pack_into("<Q", h, 32, self.bytes_per_variant)
        struct.pack_into("<Q", h, 40, self.stats_offset)
        struct.pack_into("<Q", h, 48, self.data_offset)
        struct.pack_into("<Q", h, 56, self.gidx_offset)
        struct.pack_into("<I", h, 64, self.flags)
        self.f.seek(0)
        self.f.write(bytes(h))

    def __exit__(self, exc_type, exc, tb):
        if self.f:
            if exc_type is None:
                self._finalize()
            self.f.close()
            self.f = None
        return False


def write_cugen(path, dosages, gidx=None, encoding=ENCODING_2BIT):
    """Write a whole (n_samples, n_variants) dosage array to `path`."""
    a = np.asarray(dosages)
    if a.ndim != 2:
        raise ValueError("dosages must be 2-D (n_samples, n_variants)")
    n_samples, n_variants = a.shape
    g = np.arange(n_variants, dtype=np.int64) if gidx is None else \
        np.asarray(gidx, dtype=np.int64)
    if g.size != n_variants:
        raise ValueError(f"gidx has {g.size} entries, need {n_variants}")
    with CugenWriter(path, n_samples, n_variants, encoding=encoding) as w:
        for j in range(n_variants):
            w.add_variant(int(g[j]), a[:, j])
    return path


def validate_cugen(path, verbose=True):
    """Structural check. Returns a dict; raises nothing, reports `ok`.

    NOTE this validates STRUCTURE only. It cannot tell you whether the stored
    stats match the stored genotypes -- for that use cg.repair(dry_run=True),
    which recomputes them. A file with stale stats passes every check here.
    """
    r = {"path": str(path), "ok": False, "problems": []}
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h = f.read(HEADER_SIZE)
    if len(h) < HEADER_SIZE:
        r["problems"].append("file shorter than a header")
        return r
    if h[0:8] != CUGEN_MAGIC:
        r["problems"].append(f"bad magic {h[0:8]!r}")
    g = lambda fmt, off: struct.unpack_from(fmt, h, off)[0]      # noqa: E731
    r.update(version=g("<I", 8), encoding=g("<I", 12),
             n_samples=g("<Q", 16), n_variants=g("<Q", 24),
             bytes_per_variant=g("<Q", 32), stats_offset=g("<Q", 40),
             data_offset=g("<Q", 48), gidx_offset=g("<Q", 56), flags=g("<I", 64))
    if r["version"] != CUGEN_VERSION:
        r["problems"].append(f"version {r['version']} != {CUGEN_VERSION}")
    exp_bpv = _ENC_BYTES.get(r["encoding"], lambda n: -1)(r["n_samples"])
    if r["bytes_per_variant"] != exp_bpv:
        r["problems"].append(
            f"bytes_per_variant {r['bytes_per_variant']} != {exp_bpv} "
            f"for encoding {r['encoding']} at n_samples {r['n_samples']}")
    exp_size = r["data_offset"] + r["n_variants"] * r["bytes_per_variant"]
    if size != exp_size:
        r["problems"].append(f"file size {size} != expected {exp_size}")
    if r["stats_offset"] < HEADER_SIZE:
        r["problems"].append("stats_offset inside the header")
    if r["data_offset"] < r["stats_offset"] + 12 * r["n_variants"]:
        r["problems"].append("data_offset overlaps the stats block")
    r["has_gidx"] = bool(r["flags"] & FLAG_HAS_GIDX_MAP)
    r["has_missing"] = bool(r["flags"] & FLAG_HAS_MISSING)
    r["ok"] = not r["problems"]
    if verbose:
        print(f"{'OK  ' if r['ok'] else 'BAD '} {path}")
        print(f"     n_samples={r['n_samples']:,} n_variants={r['n_variants']:,} "
              f"enc={r['encoding']} bpv={r['bytes_per_variant']:,} "
              f"gidx={r['has_gidx']} missing={r['has_missing']}")
        for p in r["problems"]:
            print(f"     - {p}")
    return r
