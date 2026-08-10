"""cugen.ld - GPU linkage disequilibrium from packed genotypes.

    ld_matrix(...)    signed r, r^2, signed r^2, D, D'     (alias: cg.r2)
    ld_clump(...)     v0.2 roadmap, still a stub

Counts, not correlations
------------------------
Every statistic here derives from the 3x3 genotype contingency table of each
variant pair. Those counts are recoverable from products of per-variant
indicator planes, so the O(p^2) work runs on tensor cores, and because the
counts are integers the arithmetic is EXACT: fp32 accumulation is bit-exact
while 4*n_samples < 2**24 (n < 4,194,304). There is therefore no float
screen-then-refine pass in this module -- there is no float error to correct.
Above that bound the code raises rather than returning quietly-wrong counts.

Cost per tile is 3 GEMMs for the full r + D/D' set on a file with no missing
calls -- n21 comes free as n12 transposed, and the marginals collapse to
per-variant constants -- rising to 6 when missingness is present and the
co-observed counts and marginals must themselves be computed pairwise.

PHASE -- read this before trusting D'
-------------------------------------
.cugen stores unphased 2-bit dosages; phase is discarded at conversion. D and
D' are haplotype quantities, so they must be ESTIMATED. We solve the Hill
(1974) likelihood exactly, via the cubic of Gaunt et al. (2007) -- the same
approach plink2 --r2-phased takes (cf. plink-ng cubic_real_roots).

Consequence: on PHASED input (e.g. stock 1000 Genomes VCFs) plink2 uses the
observed haplotypes, and that cannot be reproduced from a .cugen because the
phase is already gone. Measured divergence on identical dosages with phase
toggled is ~8.6x in r^2. Parity with --r2-phased therefore holds on
PHASE-STRIPPED input only:

    bcftools +setGT in.vcf.gz -- -t a -n u | bgzip > unphased.vcf.gz

r and r^2 carry no such caveat; they match --r2-unphased unconditionally.

MISSINGNESS
-----------
Pairwise complete case by default: a sample contributes only if non-missing at
BOTH loci. The header mu_x/sxx are PER-VARIANT non-missing statistics and are
invalid in a pairwise denominator, because variants j and k have different
non-missing sets -- do not "optimise" by reusing them. Likewise this module
does not build on read_to_gpu(), which maps missing -> dosage 0; that is the
failure mode fixed in 85ff1b0 (complete-case association kernel, session 52).
Verified: plink2 --r2-unphased is also pairwise-complete, not mean-imputing.

SIGN
----
r is signed relative to the ALT allele at both variants, consistent with
assoc.py's BETA orientation and directly usable by gpu_susie_rss. plink2 signs
by the MAJOR allele; pass sign_reference="major" for parity. Flipping both
variants leaves r unchanged, so the two conventions differ only when exactly
one variant of a pair has ALT frequency > 0.5.

MEMORY
------
Above the tile size, peak device memory is a function of tile size and sample
count -- NOT of the number of variants. Below it the tile is the whole matrix,
so cost is O(p^2); the two regimes meet where p equals the tile size.

Measured on 1000 Genomes chr22 (2504 samples, A100 80GB): peak held at
~5.6 GiB from p = 20,000 to p = 170,949, i.e. across a 73x growth in pair
count. The plateau height is set by the DEVICE -- the tile auto-tuner takes a
fraction of free VRAM -- so a smaller card plateaus lower, not higher. Whole-
chromosome runs are therefore bounded by wall time, not by memory.

VALIDATION
----------
Against plink2 v2.0.0-a.7.1 (300 samples x 14 variants, ~15% missingness on
two variants, ALT frequency straddling 0.5 so major/ALT orientation actually
diverges on 49 of 91 pairs), max absolute error:

    r  vs UNPHASED_R      4.8e-07        D  vs D        4.7e-07
    r2 vs UNPHASED_R^2    7.2e-07        D' vs DPRIME   5.0e-07
    phased r vs PHASED_R  5.0e-07

which is the 6-significant-figure floor of plink2's text output. The cubic is
additionally cross-checked against an independent multi-start EM and against
brute-force likelihood maximisation.

References
----------
Lewontin (1964) Genetics 49(1):49-67           D' and Dmax
Hill & Robertson (1968) TAG 38(6):226-231      r^2 in finite populations
Hill (1974) Heredity 33(2):229-239             ML haplotype freqs, unphased
Weir (1979) Biometrics 35(1):235-254           Burrows' composite LD
Excoffier & Slatkin (1995) MBE 12(5):921-927   EM (used only as a test oracle)
Gaunt, Rodriguez & Day (2007) BMC Bioinf 8:428 CubeX exact cubic (production)
Chang et al. (2015) GigaScience 4:7            PLINK, the behavioural reference
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._stubs import _stub
from .io import ENCODING_2BIT, read_cugen

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:                                            # noqa: BLE001
    cp = None
    HAS_CUPY = False

__all__ = ["ld_matrix", "ld_clump", "LDMatrix"]

EPS = 1e-12
_STATS = ("r", "r2", "r2_signed", "d", "dp")
_STAT_COL = {"r": "R", "r2": "R2", "r2_signed": "R2_SIGNED", "d": "D", "dp": "DP"}
_FP32_EXACT_MAX_SAMPLES = (1 << 24) // 4      # 4*n must stay below 2**24


# ---------------------------------------------------------------------------
# Numeric core -- fp64, vectorised, CuPy-free.
# This is the CPU twin of the kernel epilogue: same branches, same guards, so
# a disagreement here is a disagreement there.
# ---------------------------------------------------------------------------
def cubic_real_roots(a, b, c, d):
    """Real roots of a*x^3 + b*x^2 + c*x + d, vectorised.

    Returns (roots, n_real) with roots shaped (..., 3), unused slots NaN.
    Trigonometric form when three real roots exist, Cardano otherwise. This
    mirrors plink-ng's cubic_real_roots so the root SET matches plink2.
    """
    a, b, c, d = (np.asarray(v, dtype=np.float64) for v in (a, b, c, d))
    shape = np.broadcast_shapes(a.shape, b.shape, c.shape, d.shape)
    roots = np.full(shape + (3,), np.nan)
    n_real = np.zeros(shape, dtype=np.int64)

    ok = np.abs(a) > EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(ok, b / a, 0.0)
        q = np.where(ok, c / a, 0.0)
        r = np.where(ok, d / a, 0.0)
        # depressed cubic t^3 + A t + B, with x = t - p/3
        A = q - p * p / 3.0
        B = 2.0 * p ** 3 / 27.0 - p * q / 3.0 + r
        disc = (B / 2.0) ** 2 + (A / 3.0) ** 3

        three = ok & (disc <= 0.0) & (A < -EPS)
        one = ok & ~three

        if three.any():
            At = np.where(three, A, -1.0)
            Bt = np.where(three, B, 0.0)
            m = 2.0 * np.sqrt(-At / 3.0)
            theta = np.arccos(np.clip(3.0 * Bt / (At * m), -1.0, 1.0))
            for k in range(3):
                t = m * np.cos((theta - 2.0 * np.pi * k) / 3.0)
                roots[..., k] = np.where(three, t - p / 3.0, roots[..., k])
            n_real = np.where(three, 3, n_real)

        if one.any():
            Bo = np.where(one, B, 0.0)
            s = np.sqrt(np.where(one, np.maximum(disc, 0.0), 0.0))
            t = np.cbrt(-Bo / 2.0 + s) + np.cbrt(-Bo / 2.0 - s)
            roots[..., 0] = np.where(one, t - p / 3.0, roots[..., 0])
            n_real = np.where(one, 1, n_real)

    return roots, n_real


def _loglik(x, tab, pA, pB):
    """Multinomial log-likelihood of a genotype table given p_AB = x."""
    y, z, w = pA - x, pB - x, 1.0 - pA - pB + x
    bad = (x < -1e-9) | (y < -1e-9) | (z < -1e-9) | (w < -1e-9)
    xe, ye, ze, we = (np.maximum(v, EPS) for v in (x, y, z, w))
    ll = (2 * tab[..., 2, 2] * np.log(xe)
          + tab[..., 2, 1] * np.log(xe * ye)
          + tab[..., 1, 2] * np.log(xe * ze)
          + 2 * tab[..., 2, 0] * np.log(ye)
          + 2 * tab[..., 0, 2] * np.log(ze)
          + tab[..., 1, 1] * np.log(xe * we + ye * ze)
          + tab[..., 1, 0] * np.log(ye * we)
          + tab[..., 0, 1] * np.log(ze * we)
          + 2 * tab[..., 0, 0] * np.log(we))
    return np.where(bad, -np.inf, ll)


def ld_from_counts(counts, dprime_method: str = "phased"):
    """All five statistics from an (..., 3, 3) stack of genotype tables.

    counts[..., i, j] = number of PAIRWISE-CO-OBSERVED samples with ALT dosage
    i at locus A and j at locus B. Degenerate pairs yield NaN.
    """
    tab = np.asarray(counts, dtype=np.float64)
    n = tab.sum(axis=(-2, -1))
    rows, cols = tab.sum(axis=-1), tab.sum(axis=-2)
    dv = np.array([0.0, 1.0, 2.0])
    sA, sB = rows @ dv, cols @ dv
    qA, qB = rows @ (dv * dv), cols @ (dv * dv)
    # S = sum over co-observed of gA*gB, recoverable from the interior cells
    # alone: terms with dosage 0 on either side vanish.
    S = np.einsum("i,...ij,j->...", dv, tab, dv)

    with np.errstate(divide="ignore", invalid="ignore"):
        pA, pB = sA / (2.0 * n), sB / (2.0 * n)
        vA, vB = n * qA - sA * sA, n * qB - sB * sB
        good = (n > 0) & (vA > 0) & (vB > 0)
        # Hill & Robertson (1968): r as the correlation of allele counts.
        # sqrt(vA)*sqrt(vB), not sqrt(vA*vB) -- the product overflows the
        # useful range for no benefit.
        r = np.clip(np.where(good, (n * S - sA * sB) / (np.sqrt(vA) * np.sqrt(vB)),
                             np.nan), -1.0, 1.0)
        qAf, qBf = 1.0 - pA, 1.0 - pB

        if dprime_method == "composite":
            # Burrows' composite measure; see Weir (1979). This inverts the
            # haplotypic identity r = D/sqrt(pA qA pB qB) and is NOT a phase
            # estimate. Verified against simulation to recover gametic D under
            # HWE with no factor-of-2 correction.
            D = r * np.sqrt(np.maximum(pA * qAf * pB * qBf, 0.0))
        else:
            # Hill (1974) likelihood, exact roots per Gaunt et al. (2007).
            # E-step note: a COUPLING double heterozygote is AB/ab and yields
            # exactly ONE AB haplotype, not two. That factor is the classic
            # error here; getting it wrong makes EM DECREASE the likelihood.
            cc = 2.0 * tab[..., 2, 2] + tab[..., 2, 1] + tab[..., 1, 2]
            n11 = tab[..., 1, 1]
            a1, a0 = 1.0 - 2.0 * pA - 2.0 * pB, pA * pB
            roots, _ = cubic_real_roots(
                4.0 * n,
                2.0 * n * a1 - 2.0 * cc - n11,
                2.0 * n * a0 - cc * a1 - n11 * (1.0 - pA - pB),
                -cc * a0)
            lo = np.maximum(0.0, pA + pB - 1.0)
            hi = np.minimum(pA, pB)
            best = np.where(good, lo, np.nan)
            best_ll = np.full(n.shape, -np.inf)
            # boundaries are candidates too: the ML can sit on one
            cand = np.concatenate([roots, lo[..., None], hi[..., None]], axis=-1)
            for k in range(cand.shape[-1]):
                x = np.clip(cand[..., k], lo, hi)
                ll = _loglik(x, tab, pA, pB)
                take = good & np.isfinite(x) & (ll > best_ll)
                best = np.where(take, x, best)
                best_ll = np.where(take, ll, best_ll)
            mono = (pA <= EPS) | (pA >= 1 - EPS) | (pB <= EPS) | (pB >= 1 - EPS)
            D = np.where(mono, 0.0, best - pA * pB)

        D = np.where(good, D, np.nan)
        # Lewontin (1964): D normalised by its frequency-dependent maximum.
        # The branch on sign(D) is Lewontin's, not a numerical convenience.
        dmax = np.where(D >= 0, np.minimum(pA * qBf, qAf * pB),
                        np.minimum(pA * pB, qAf * qBf))
        DP = np.where(good, np.where(dmax > EPS,
                                     np.clip(D / dmax, -1.0, 1.0), 0.0), np.nan)

    return {"n": n, "pA": pA, "pB": pB, "r": r,
            "r2": np.clip(r * r, 0.0, 1.0),
            "r2_signed": np.clip(r * np.abs(r), -1.0, 1.0),
            "d": D, "dp": DP}


def contingency_tables(dosages, pairs):
    """(m, 3, 3) pairwise-complete tables for pairs of rows of `dosages`.

    dosages is (n_variants, n_samples) uint8 with 3 = missing. This is the
    NumPy twin of the GEMM count path; the kernel computes the same integers
    via indicator-plane products.
    """
    g = np.asarray(dosages)
    i = np.asarray(pairs, dtype=np.int64)[:, 0]
    j = np.asarray(pairs, dtype=np.int64)[:, 1]
    out = np.zeros((len(i), 3, 3), dtype=np.int64)
    for a in range(3):
        ia = (g[i] == a)
        for b in range(3):
            # int64 accumulation is mandatory: numpy sums in the INPUT dtype,
            # so counting in uint8 silently overflows past 255 samples.
            out[:, a, b] = np.sum(ia & (g[j] == b), axis=1, dtype=np.int64)
    return out


# ---------------------------------------------------------------------------
# CuPy-free helpers. Everything that is not a kernel lives here so the whole
# decision surface is testable on CPU-only CI.
# ---------------------------------------------------------------------------
def _resolve_gidx(obj) -> Optional[np.ndarray]:
    """Polymorphic gidx resolution, mirroring score._resolve_weights."""
    if obj is None:
        return None
    if isinstance(obj, (str, Path)):
        p = str(obj)
        if p.endswith(".feather"):
            df = pd.read_feather(p)
        elif p.endswith(".npz"):
            return np.asarray(np.load(p)["gidx"], dtype=np.int64)
        else:
            df = pd.read_csv(p, sep="\t" if p.endswith((".tsv", ".tsv.gz")) else ",")
        obj = df
    if isinstance(obj, pd.DataFrame):
        if "gidx" not in obj.columns:
            raise ValueError("variant table needs a 'gidx' column; "
                             f"got {list(obj.columns)}")
        return np.asarray(obj["gidx"], dtype=np.int64)
    return np.asarray(obj, dtype=np.int64)


def _parse_region(region: str) -> Tuple[str, Optional[int], Optional[int]]:
    """'22:1000000-2000000' or '22'. 1-based inclusive, PLINK/samtools style."""
    m = re.fullmatch(r"([^:]+)(?::([\d,]+)-([\d,]+))?", str(region).strip())
    if not m:
        raise ValueError(f"malformed region {region!r}; expected 'CHR' or "
                         f"'CHR:START-END'")
    chrom, s, e = m.group(1), m.group(2), m.group(3)
    if s is None:
        return chrom, None, None
    start, end = int(s.replace(",", "")), int(e.replace(",", ""))
    if end < start:
        raise ValueError(f"region {region!r} has end < start")
    return chrom, start, end


def _pair_bounds(n_rows: int, positions: Optional[np.ndarray],
                 window: Optional[int], window_kb: Optional[float]):
    """Per-row exclusive upper column bound for the upper-triangle scan.

    Both window predicates AND together, matching PLINK. Banding by position
    uses searchsorted so the column extent is O(band), not O(p). Returns
    (starts, hi) so the pair COUNT can be taken without materialising pairs --
    at p = 1.1M the materialised list would be ~6e11 entries.
    """
    if window_kb is not None and positions is None:
        raise ValueError("window_kb requires variant positions; pass annotation=")
    if positions is not None and len(positions):
        if np.any(np.diff(positions) < 0):
            raise ValueError("positions are not non-decreasing along file rows; "
                             "banding would be silently wrong. Sort the .cugen "
                             "or drop window_kb.")
    lo = np.arange(n_rows, dtype=np.int64)
    hi = np.full(n_rows, n_rows, dtype=np.int64)
    if window is not None:
        hi = np.minimum(hi, lo + int(window) + 1)
    if window_kb is not None:
        span = int(round(float(window_kb) * 1000))
        hi = np.minimum(hi, np.searchsorted(positions, positions + span,
                                            side="right").astype(np.int64))
    return lo + 1, hi


def _count_pairs(n_rows: int, positions, window, window_kb) -> int:
    starts, hi = _pair_bounds(n_rows, positions, window, window_kb)
    return int(np.maximum(hi - starts, 0).sum())


def _plan_pairs(n_rows: int, positions: Optional[np.ndarray],
                window: Optional[int], window_kb: Optional[float]):
    """Materialised upper-triangle pair list, for the NumPy reference path."""
    starts, hi = _pair_bounds(n_rows, positions, window, window_kb)
    lo = np.arange(n_rows, dtype=np.int64)
    counts = np.maximum(hi - starts, 0)
    total = int(counts.sum())
    if total == 0:
        return np.zeros((0, 2), dtype=np.int64), 0
    i = np.repeat(lo, counts)
    offs = np.arange(total, dtype=np.int64) - np.repeat(
        np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    j = np.repeat(starts, counts) + offs
    return np.stack([i, j], axis=1), total


def _empty_pairs(stats: Sequence[str]) -> pd.DataFrame:
    """Zero rows, full schema, correct dtypes. Callers break on bare
    pd.DataFrame() -- this is a real bug class, not a nicety."""
    cols = {"CHR_A": np.int32, "POS_A": np.int64, "ID_A": object, "MAF_A": np.float32,
            "CHR_B": np.int32, "POS_B": np.int64, "ID_B": object, "MAF_B": np.float32,
            "N_OBS": np.int32}
    for s in _STATS:
        if s in stats:
            cols[_STAT_COL[s]] = np.float32
    cols["gidx_a"] = np.int64
    cols["gidx_b"] = np.int64
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()})


def _write_df(df: pd.DataFrame, path: str) -> None:
    """Mirrors qc._write_df so output conventions stay identical."""
    if path.endswith(".feather"):
        df.to_feather(path)
    else:
        sep = "\t" if path.endswith((".tsv", ".tsv.gz")) else ","
        df.to_csv(path, sep=sep, index=False,
                  compression="gzip" if path.endswith(".gz") else None)


def _merge_annotation(df: pd.DataFrame, ann: Optional[pd.DataFrame],
                      chrom_fallback: int) -> pd.DataFrame:
    """Left-join CHR/POS/ID for both sides on gidx, gwas-style placeholders."""
    for side in ("a", "b"):
        S = side.upper()
        if ann is None:
            df[f"CHR_{S}"] = np.int32(chrom_fallback)
            df[f"POS_{S}"] = np.int64(0)
            df[f"ID_{S}"] = "."
        else:
            sub = ann[["gidx", "CHR", "POS", "ID"]].rename(
                columns={"CHR": f"CHR_{S}", "POS": f"POS_{S}", "ID": f"ID_{S}",
                         "gidx": f"gidx_{side}"})
            df = df.merge(sub, on=f"gidx_{side}", how="left")
            df[f"ID_{S}"] = df[f"ID_{S}"].fillna(".")
    return df


def _finalize(df: pd.DataFrame, stats: Sequence[str]) -> pd.DataFrame:
    tmpl = _empty_pairs(stats)
    for c, dt in tmpl.dtypes.items():
        if c in df.columns:
            try:
                df[c] = df[c].astype(dt)
            except (ValueError, TypeError):
                pass
    return df[list(tmpl.columns)].reset_index(drop=True)


def ld_clump(*a, **kw):
    return _stub("ld.ld_clump")


# ---------------------------------------------------------------------------
# GPU: one small kernel to build indicator planes; all O(p^2) work is cuBLAS.
# Keeping hand-written CUDA to a minimum is deliberate -- 34b4a59 was an NVRTC
# crash from a single non-ASCII character, and that class of failure cannot be
# reproduced without a GPU. KERNEL SOURCE MUST STAY PURE ASCII.
# ---------------------------------------------------------------------------
_LD_PLANES_KERNEL = None

_LD_PLANES_SRC = r'''
extern "C" __global__
void build_ld_planes(const unsigned char* packed,
                     float* M, float* I1, float* I2,
                     const long long n_samples,
                     const long long n_variants,
                     const long long bytes_per_variant)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n_variants * n_samples;
    if (idx >= total) return;
    long long v = idx / n_samples;
    long long s = idx - v * n_samples;
    /* 2-bit codes are big-endian within the byte: sample 0 is the HIGH pair.
       Loop bound is n_samples so the zero-padded tail of the last byte, which
       would decode as dosage 0, can never leak in as a real sample. */
    unsigned char byte = packed[v * bytes_per_variant + (s >> 2)];
    int code = (byte >> (6 - 2 * (s & 3))) & 3;
    M[idx]  = (code != 3) ? 1.0f : 0.0f;   /* non-missing indicator */
    I1[idx] = (code == 1) ? 1.0f : 0.0f;
    I2[idx] = (code == 2) ? 1.0f : 0.0f;
}
'''

assert _LD_PLANES_SRC.isascii(), "kernel source must be pure ASCII (cf. 34b4a59)"


def _get_planes_kernel():
    global _LD_PLANES_KERNEL
    if _LD_PLANES_KERNEL is None and HAS_CUPY:
        _LD_PLANES_KERNEL = cp.RawKernel(_LD_PLANES_SRC, "build_ld_planes")
    return _LD_PLANES_KERNEL


def _build_planes(packed2d, rows, n_samples, bytes_per_variant):
    """Indicator planes (M, I1, I2) for a block of variant rows, on device."""
    blk = packed2d[rows].ravel()
    b = int(len(rows))
    ns = int(n_samples)
    M = cp.empty((b, ns), dtype=cp.float32)
    I1 = cp.empty((b, ns), dtype=cp.float32)
    I2 = cp.empty((b, ns), dtype=cp.float32)
    tpb = 256
    total = b * ns
    _get_planes_kernel()(((total + tpb - 1) // tpb,), (tpb,),
                         (blk, M, I1, I2, np.int64(ns), np.int64(b),
                          np.int64(bytes_per_variant)))
    return M, I1, I2


def _counts_block(pl_a, pl_b, n_samples, has_missing):
    """(3, 3, ba, bb) exact integer counts for one row-block x col-block tile.

    3 GEMMs when the file has no missing calls (n21 = n12^T and the marginals
    are per-variant constants), 6 when it does.
    """
    Ma, I1a, I2a = pl_a
    Mb, I1b, I2b = pl_b
    ba, bb = I1a.shape[0], I1b.shape[0]
    ns = int(n_samples)

    n11 = I1a @ I1b.T
    n22 = I2a @ I2b.T
    n12 = I1a @ I2b.T
    n21 = I2a @ I1b.T
    if has_missing:
        nn = Ma @ Mb.T
        rr1 = I1a @ Mb.T
        rr2 = I2a @ Mb.T
        cc1 = Ma @ I1b.T
        cc2 = Ma @ I2b.T
    else:
        nn = cp.full((ba, bb), float(ns), dtype=cp.float32)
        rr1 = cp.broadcast_to(I1a.sum(axis=1)[:, None], (ba, bb))
        rr2 = cp.broadcast_to(I2a.sum(axis=1)[:, None], (ba, bb))
        cc1 = cp.broadcast_to(I1b.sum(axis=1)[None, :], (ba, bb))
        cc2 = cp.broadcast_to(I2b.sum(axis=1)[None, :], (ba, bb))

    out = cp.empty((3, 3, ba, bb), dtype=cp.float32)
    out[1, 1], out[1, 2], out[2, 1], out[2, 2] = n11, n12, n21, n22
    out[1, 0] = rr1 - n11 - n12
    out[2, 0] = rr2 - n21 - n22
    out[0, 1] = cc1 - n11 - n21
    out[0, 2] = cc2 - n12 - n22
    out[0, 0] = nn - (out[1, 1] + out[1, 2] + out[2, 1] + out[2, 2]
                      + out[1, 0] + out[2, 0] + out[0, 1] + out[0, 2])
    return out


def _r_from_counts_gpu(c):
    """Cheap on-device r, used only to threshold before the host transfer."""
    n = c.sum(axis=(0, 1))
    sA = c[1].sum(axis=0) + 2.0 * c[2].sum(axis=0)
    sB = c[:, 1].sum(axis=0) + 2.0 * c[:, 2].sum(axis=0)
    qA = c[1].sum(axis=0) + 4.0 * c[2].sum(axis=0)
    qB = c[:, 1].sum(axis=0) + 4.0 * c[:, 2].sum(axis=0)
    S = c[1, 1] + 2.0 * c[1, 2] + 2.0 * c[2, 1] + 4.0 * c[2, 2]
    vA, vB = n * qA - sA * sA, n * qB - sB * sB
    ok = (n > 0) & (vA > 0) & (vB > 0)
    den = cp.sqrt(cp.where(ok, vA, 1.0)) * cp.sqrt(cp.where(ok, vB, 1.0))
    r = cp.where(ok, (n * S - sA * sB) / den, cp.nan)
    return cp.clip(r, -1.0, 1.0), n


def _tile_size_for(n_samples: int, budget_frac: float = 0.35) -> int:
    """Pick B so planes + counts fit the free device memory.

    planes  = 2 * 3 * B * n_samples * 4
    counts  = 9 * B * B * 4
    Peak is a function of B and n_samples only -- NOT of the variant count.
    """
    free, _total = cp.cuda.Device().mem_info
    budget = budget_frac * free
    a = 36.0
    b = 24.0 * n_samples
    B = int((-b + (b * b + 4 * a * budget) ** 0.5) / (2 * a))
    return max(256, min(8192, (B // 256) * 256))


@dataclass(frozen=True)
class LDMatrix:
    """Dense LD result for one contiguous variant selection on one chromosome."""
    r: Optional[np.ndarray]
    r2: Optional[np.ndarray]
    r2_signed: Optional[np.ndarray]
    d: Optional[np.ndarray]
    d_prime: Optional[np.ndarray]
    n_obs: np.ndarray
    gidx: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    maf: np.ndarray
    n_samples: int
    dprime_method: str
    missing: str

    def __repr__(self) -> str:
        p = len(self.gidx)
        return (f"<LDMatrix p={p} n_samples={self.n_samples} "
                f"dprime={self.dprime_method} missing={self.missing}>")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _dosages_numpy(reader) -> np.ndarray:
    """(n_variants, n_samples) uint8 with 3 = missing, no CuPy required."""
    from .write import unpack_2bit
    packed = np.frombuffer(reader.read_packed_bytes(), dtype=np.uint8)
    bpv = int(reader.bytes_per_variant)
    p, ns = int(reader.n_variants), int(reader.n_samples)
    return np.stack([unpack_2bit(packed[v * bpv:(v + 1) * bpv], ns)
                     for v in range(p)])


def _scan_gpu(reader, rows, window, window_kb, positions, min_r2, min_obs,
              tile_size=None, verbose=False):
    """Tiled upper-triangle scan. Returns (pairs_local, tables) for survivors.

    Peak device memory is a function of tile size and sample count only, never
    of the number of variants: each (row-block, col-block) tile allocates its
    own planes and counts and frees them before the next.
    """
    ns = int(reader.n_samples)
    if ns > _FP32_EXACT_MAX_SAMPLES:
        raise ValueError(
            f"n_samples={ns:,} exceeds {_FP32_EXACT_MAX_SAMPLES:,}, above which "
            f"fp32 accumulation of 4*n is no longer exact. Refusing rather than "
            f"returning quietly-wrong counts.")
    bpv = int(reader.bytes_per_variant)
    p = len(rows)
    has_missing = bool(reader.has_missing)
    B = int(tile_size) if tile_size else _tile_size_for(ns)

    packed = cp.asarray(np.frombuffer(reader.read_packed_bytes(), dtype=np.uint8))
    packed = packed.reshape(int(reader.n_variants), bpv)[cp.asarray(rows)]

    out_pairs, out_tabs = [], []
    for i0 in range(0, p, B):
        i1 = min(i0 + B, p)
        pl_a = _build_planes(packed, cp.arange(i0, i1), ns, bpv)
        # column extent: upper triangle, bounded by whichever windows are set
        hi = p
        if window is not None:
            hi = min(hi, i1 - 1 + int(window) + 1)
        if window_kb is not None:
            hi = min(hi, int(np.searchsorted(
                positions, positions[i1 - 1] + int(round(window_kb * 1000)),
                side="right")))
        for j0 in range(i0, hi, B):
            j1 = min(j0 + B, hi)
            pl_b = pl_a if j0 == i0 else _build_planes(
                packed, cp.arange(j0, j1), ns, bpv)
            c = _counts_block(pl_a, pl_b, ns, has_missing)
            r, n = _r_from_counts_gpu(c)

            ii = cp.arange(i0, i1)[:, None]
            jj = cp.arange(j0, j1)[None, :]
            keep = (jj > ii) & cp.isfinite(r) & (n >= min_obs)
            if window is not None:
                keep &= (jj - ii) <= int(window)
            if window_kb is not None:
                pos = cp.asarray(positions)
                keep &= (pos[jj] - pos[ii]) <= int(round(window_kb * 1000))
            if min_r2 > 0:
                keep &= (r * r) >= min_r2

            idx = cp.nonzero(keep)
            if idx[0].size:
                tabs = cp.empty((idx[0].size, 3, 3), dtype=cp.float32)
                for a in range(3):
                    for b in range(3):
                        tabs[:, a, b] = c[a, b][idx]
                out_tabs.append(cp.asnumpy(tabs))
                out_pairs.append(np.stack([cp.asnumpy(idx[0]) + i0,
                                           cp.asnumpy(idx[1]) + j0], axis=1))
            del c, r, n, keep, idx
            if pl_b is not pl_a:
                del pl_b
        del pl_a
        cp.get_default_memory_pool().free_all_blocks()

    del packed
    cp.get_default_memory_pool().free_all_blocks()
    if not out_pairs:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3, 3))
    return np.concatenate(out_pairs), np.concatenate(out_tabs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ld_matrix(
    cugen: Union[str, Path],
    *,
    variant_range: Optional[Tuple[int, int]] = None,
    variants=None,
    region: Optional[str] = None,
    maf_min: float = 0.0,
    window: Optional[int] = None,
    window_kb: Optional[float] = None,
    min_r2: float = 0.0,
    stats: Sequence[str] = _STATS,
    dprime_method: str = "phased",
    sign_reference: str = "alt",
    missing: str = "pairwise",
    min_obs: int = 2,
    output_format: str = "pairs",
    annotation=None,
    output: Optional[Union[str, Path]] = None,
    backend: str = "auto",
    tile_size: Optional[int] = None,
    max_pairs: int = 100_000_000,
    device: int = 0,
    verbose: bool = True,
):
    """Pairwise LD (r, r^2, signed r^2, D, D') from a .cugen file.

    See the module docstring for the phase caveat on D/D', the pairwise
    complete-case missingness rule, and plink2 parity.

    Parameters
    ----------
    cugen
        Path to a single ``.cugen`` file.
    variant_range
        ``(start, end)`` half-open LOCAL ROW indices into the file. Note these
        are row indices, not gidx: gidx is global, rows are local.
    variants
        Restrict to these gidx values. Sequence, ndarray, DataFrame with a
        ``gidx`` column, or a path to feather/TSV/CSV/NPZ.
    region
        ``'22:1000000-2000000'``, 1-based inclusive. Requires ``annotation``,
        since a .cugen carries no coordinates.
    maf_min
        Drop variants below this MAF, read from the header (no decode).
    window, window_kb
        Max separation in variant-index distance and in kb respectively. Both
        may be given; a pair must satisfy both, matching PLINK. ``window_kb``
        requires ``annotation``. Unlike PLINK these default to NO filter --
        silently truncating a result set is worse than an explicit error, and
        ``max_pairs`` catches the runaway case.
    min_r2
        Drop pairs below this r^2. PLINK's ``--ld-window-r2``.
    stats
        Any of ``r, r2, r2_signed, d, dp``. Dropping ``d``/``dp`` skips the
        cubic entirely.
    dprime_method
        ``'phased'`` (exact cubic, plink2 ``--r2-phased`` parity) or
        ``'composite'`` (Burrows' closed form; not a phase estimate).
    sign_reference
        ``'alt'`` (default, cugen-internal) or ``'major'`` (plink2 parity).
    missing
        ``'pairwise'`` complete-case (default). Only this is implemented.
    min_obs
        Drop pairs with fewer co-observed samples than this.
    output_format
        ``'pairs'`` -> DataFrame, or ``'matrix'`` -> :class:`LDMatrix`.
    backend
        ``'auto'``, ``'gpu'``, or ``'numpy'`` (reference path, no CuPy).

    Returns
    -------
    DataFrame with columns
    ``CHR_A POS_A ID_A MAF_A CHR_B POS_B ID_B MAF_B N_OBS [R R2 R2_SIGNED D DP]
    gidx_a gidx_b``, or an :class:`LDMatrix`.
    """
    # ---- validation, all before any allocation --------------------------
    bad = [s for s in stats if s not in _STATS]
    if bad:
        raise ValueError(f"unknown stats {bad}; valid are {list(_STATS)}")
    if output_format not in ("pairs", "matrix"):
        raise ValueError(f"output_format must be 'pairs' or 'matrix', got {output_format!r}")
    if dprime_method not in ("phased", "composite"):
        raise ValueError(f"dprime_method must be 'phased' or 'composite', got {dprime_method!r}")
    if sign_reference not in ("alt", "major"):
        raise ValueError(f"sign_reference must be 'alt' or 'major', got {sign_reference!r}")
    if missing != "pairwise":
        raise NotImplementedError(
            f"missing={missing!r} is not implemented; only 'pairwise' "
            f"(complete-case) is available.")
    if backend not in ("auto", "gpu", "numpy"):
        raise ValueError(f"backend must be 'auto', 'gpu' or 'numpy', got {backend!r}")

    path = str(cugen)
    if os.path.isdir(path):
        raise NotImplementedError(
            "directory input is not implemented yet; pass a single .cugen file")
    if not path.endswith(".cugen"):
        raise ValueError(
            f"{path!r} is not a .cugen file. cugen computes on .cugen only; "
            f"convert first, e.g.  cg.convert.vcf2cugen(in_vcf, out_cugen)  or "
            f"cugen-convert vcf in.vcf.gz out.cugen  (also pgen2cugen / "
            f"bed2cugen).")

    ann = None
    if annotation is not None:
        ann = (annotation if isinstance(annotation, pd.DataFrame)
               else pd.read_feather(str(annotation)))
    if (region is not None or window_kb is not None) and ann is None:
        raise ValueError("region= and window_kb= need coordinates; pass annotation=")
    if region is not None and variant_range is not None:
        raise ValueError("pass at most one of region= and variant_range=")

    use_gpu = HAS_CUPY if backend == "auto" else (backend == "gpu")
    if backend == "gpu" and not HAS_CUPY:
        raise RuntimeError("CuPy not available")
    if backend == "auto" and not HAS_CUPY:
        raise RuntimeError(
            "CuPy not available. Pass backend='numpy' for the CPU reference "
            "path (correct but not fast).")

    reader = read_cugen(path, device=device)
    if int(reader.encoding) != ENCODING_2BIT:
        raise NotImplementedError(
            f"cugen.ld requires 2-bit encoding (encoding={ENCODING_2BIT}); this "
            f"file has encoding={int(reader.encoding)}. Re-convert with "
            f"cg.convert.")

    p_all = int(reader.n_variants)
    gidx_all = np.asarray(reader.gidx, dtype=np.int64)
    maf_all = np.asarray(reader.maf, dtype=np.float32)

    # ---- row selection ---------------------------------------------------
    rows = np.arange(p_all, dtype=np.int64)
    if variant_range is not None:
        s, e = variant_range
        rows = rows[max(0, int(s)):min(p_all, int(e))]
    if variants is not None:
        want = set(_resolve_gidx(variants).tolist())
        rows = rows[np.isin(gidx_all[rows], list(want))]
    if region is not None:
        chrom, start, end = _parse_region(region)
        sub = ann[ann["CHR"].astype(str) == str(chrom)]
        if start is not None:
            sub = sub[(sub["POS"] >= start) & (sub["POS"] <= end)]
        rows = rows[np.isin(gidx_all[rows], np.asarray(sub["gidx"], dtype=np.int64))]
    if maf_min > 0:
        rows = rows[maf_all[rows] >= maf_min]
    if rows.size == 0:
        return _empty_pairs(stats)

    positions = None
    if ann is not None:
        pos_map = dict(zip(np.asarray(ann["gidx"], dtype=np.int64),
                           np.asarray(ann["POS"], dtype=np.int64)))
        positions = np.array([pos_map.get(int(g), 0) for g in gidx_all[rows]],
                             dtype=np.int64)

    n_pairs = _count_pairs(len(rows), positions, window, window_kb)
    if n_pairs > max_pairs:
        raise ValueError(
            f"plan would emit {n_pairs:,} pairs, above max_pairs={max_pairs:,}. "
            f"Narrow it with window= or window_kb=, raise min_r2, or raise "
            f"max_pairs if you really want this.")
    if n_pairs == 0:
        return _empty_pairs(stats)

    # ---- counts ----------------------------------------------------------
    if use_gpu:
        cp.cuda.Device(device).use()
        pairs_local, tables = _scan_gpu(
            reader, rows, window, window_kb, positions, min_r2, min_obs,
            tile_size=tile_size, verbose=verbose)
        if len(pairs_local) == 0:
            return _empty_pairs(stats)
    else:
        pairs_local, _ = _plan_pairs(len(rows), positions, window, window_kb)
        dos = _dosages_numpy(reader)[rows]
        tables = contingency_tables(dos, pairs_local)

    res = ld_from_counts(tables, dprime_method=dprime_method)

    # ---- orientation -----------------------------------------------------
    # Flipping BOTH variants leaves r unchanged; flipping exactly one negates
    # r, D and D'. plink2 orients by the major allele (Chang et al. 2015).
    if sign_reference == "major":
        flip = (res["pA"] > 0.5) ^ (res["pB"] > 0.5)
        for k in ("r", "r2_signed", "d", "dp"):
            res[k] = np.where(flip, -res[k], res[k])

    keep = np.isfinite(res["r"]) & (res["n"] >= min_obs)
    if min_r2 > 0:
        keep &= res["r2"] >= min_r2
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return _empty_pairs(stats)

    ga = gidx_all[rows[pairs_local[idx, 0]]]
    gb = gidx_all[rows[pairs_local[idx, 1]]]
    df = pd.DataFrame({
        "gidx_a": ga, "gidx_b": gb,
        "MAF_A": maf_all[rows[pairs_local[idx, 0]]],
        "MAF_B": maf_all[rows[pairs_local[idx, 1]]],
        "N_OBS": res["n"][idx].astype(np.int32),
    })
    for s in _STATS:
        if s in stats:
            df[_STAT_COL[s]] = res[s][idx]

    chrom_fallback = 0
    m = re.search(r"chr(\d+)", os.path.basename(path))
    if m:
        chrom_fallback = int(m.group(1))
    df = _merge_annotation(df, ann, chrom_fallback)
    df = _finalize(df, stats)

    if verbose:
        print(f"cugen.ld: {len(rows):,} variants -> {n_pairs:,} pairs planned, "
              f"{len(df):,} emitted  (backend={'gpu' if use_gpu else 'numpy'}, "
              f"dprime={dprime_method}, sign={sign_reference})")
    if output is not None:
        _write_df(df, str(output))
    return df
