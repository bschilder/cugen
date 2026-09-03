"""Genomic region masks for LD scans.

Exclude variants in regions known to produce spurious long-range LD -- segmental
duplications, low-mappability sequence, satellite arrays, pericentromeres,
assembly gaps -- before scanning rather than after. Filtering up front is not
merely tidier: pairs grow quadratically in the variant count, so removing 11% of
variants removes ~21% of pairs and the scan gets cheaper as well as cleaner.

Two facts drive every design decision here, because both fail SILENTLY:

* **BED is 0-based half-open; variant positions are 1-based.** A variant at
  1-based ``p`` lies in ``[s, e)`` iff ``s < p <= e``. An off-by-one shifts
  every boundary by a base and no summary statistic reveals it.
* **Chromosome naming is not standardised.** UCSC ships ``chr1``; many .pvar
  files carry ``1``. A mismatch masks NOTHING and the scan looks clean, which is
  the worst possible failure: an unmasked artifact reported as a filtered result.

Both are handled by normalising to a bare contig name and comparing on 1-based
inclusive coordinates.
"""

from __future__ import annotations

import gzip
import os
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

__all__ = ["read_bed", "merge_intervals", "mask_variants", "normalise_contig",
           "fetch_bundle", "resolve_regions", "BUNDLES"]

Intervals = Dict[str, List[Tuple[int, int]]]


def normalise_contig(name) -> str:
    """'chr1' -> '1', 'CHR1' -> '1', 1 -> '1'. Leaves 'X'/'MT' alone."""
    s = str(name).strip()
    low = s.lower()
    if low.startswith("chr"):
        s = s[3:]
    return s.upper() if s.upper() in ("X", "Y", "M", "MT") else s


def merge_intervals(iv: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort and coalesce overlapping OR adjacent half-open intervals."""
    s = sorted(iv)
    if not s:
        return []
    out = [s[0]]
    for a, b in s[1:]:
        la, lb = out[-1]
        if a <= lb:                     # <=, so [10,20) and [20,30) coalesce
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def read_bed(path: Union[str, os.PathLike]) -> Intervals:
    """Read a BED (optionally gzipped) into {contig: merged half-open intervals}.

    Track/browser/comment lines are skipped. A line that does not carry at least
    three fields RAISES rather than being skipped -- a silently dropped interval
    means an unmasked artifact region, which is exactly the failure this module
    exists to prevent.
    """
    p = str(path)
    opener = gzip.open if p.endswith(".gz") else open
    raw: Dict[str, List[Tuple[int, int]]] = {}
    with opener(p, "rt") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            f = line.split()
            if len(f) < 3:
                raise ValueError(
                    f"{p} line {n}: expected at least 3 tab-separated fields "
                    f"(chrom start end), got {len(f)}: {line[:60]!r}")
            try:
                start, end = int(f[1]), int(f[2])
            except ValueError:
                raise ValueError(
                    f"{p} line {n}: start/end are not integers: {line[:60]!r}")
            raw.setdefault(normalise_contig(f[0]), []).append((start, end))
    return {c: merge_intervals(v) for c, v in raw.items()}


def mask_variants(chrom, pos, beds: Union[Intervals, Sequence[Intervals]],
                  pad: int = 0) -> np.ndarray:
    """True where a variant falls inside any interval. `pos` is 1-BASED.

    `beds` is one interval dict or several, in which case their union is used.
    `pad` widens every interval symmetrically, clamped at 0.
    """
    if isinstance(beds, dict):
        beds = [beds]
    chrom = np.asarray([normalise_contig(c) for c in np.asarray(chrom).ravel()])
    pos = np.asarray(pos, dtype=np.int64).ravel()
    if chrom.size != pos.size:
        raise ValueError(f"{chrom.size} contigs but {pos.size} positions")
    out = np.zeros(pos.size, dtype=bool)

    merged: Dict[str, List[Tuple[int, int]]] = {}
    for b in beds:
        for c, iv in b.items():
            merged.setdefault(c, []).extend(iv)
    for c in merged:
        iv = merge_intervals(
            (max(0, a - pad), e + pad) for a, e in merged[c]) if pad else \
            merge_intervals(merged[c])
        sel = np.nonzero(chrom == c)[0]
        if sel.size == 0:
            continue
        starts = np.fromiter((a for a, _ in iv), dtype=np.int64, count=len(iv))
        ends = np.fromiter((e for _, e in iv), dtype=np.int64, count=len(iv))
        # 1-based p is inside half-open [s, e) iff s < p <= e. searchsorted on
        # `starts` with side="left" over p-1 gives the count of starts <= p-1,
        # so the candidate interval is the one before that index.
        k = np.searchsorted(starts, pos[sel] - 1, side="right") - 1
        ok = k >= 0
        hit = np.zeros(sel.size, dtype=bool)
        hit[ok] = pos[sel][ok] <= ends[k[ok]]
        out[sel] = hit
    return out


# --------------------------------------------------------------------------
# Curated bundles, fetched on demand
# --------------------------------------------------------------------------
#: Where the curated BED sets live. A dataset repo rather than a package data
#: dir: these files are tens of megabytes and change independently of releases.
BUNDLE_REPO = "standardmodelbio/cugen"
BUNDLE_PREFIX = "regions"

#: name -> (assembly, [files within the repo]).
#:
#: `artifact` collects the annotations that make a variant's mapping
#: unreliable, and therefore make an LD pair between two such regions suspect:
#: low-mappability, segmental duplications, satellite and alpha-satellite
#: arrays, pericentromeres, acrocentric proximal arms, assembly gaps and the
#: ENCODE blacklist.
#:
#: Acrocentric arms are INTERVALS here, deliberately. An earlier annotation set
#: encoded "acrocentric" as whole-chromosome membership of chr13/14/15/21/22,
#: which is a chromosome label rather than a region. Masking on the label would
#: discard those five chromosomes' entire long arms -- a large fraction of any
#: panel -- while removing no artifact, because the mapping problem is confined
#: to the proximal arms. Enrichment for artifact is a property of the PAIR (how
#: many of its two legs land in such a region), not of a single leg, so a
#: per-leg chromosome flag cannot express it.
BUNDLES = {
    "grch38-artifact": ("GRCh38", [
        "low_mappability.bed.gz",
        "segdup.bed.gz",
        "satellite.bed.gz",
        "pericentromere.bed.gz",
        "acrocentric_arms.bed.gz",
        "assembly_gaps.bed.gz",
        "encode_blacklist_v2.bed.gz",
    ]),
    "grch38-segdup": ("GRCh38", ["segdup.bed.gz"]),
    "grch38-mappability": ("GRCh38", ["low_mappability.bed.gz"]),
    "grch38-satellite": ("GRCh38", ["satellite.bed.gz", "pericentromere.bed.gz",
                                    "acrocentric_arms.bed.gz"]),
}


def fetch_bundle(name: str, cache_dir=None, token=None) -> List[Intervals]:
    """Download a curated bundle and return its interval dicts.

    Needs `huggingface_hub`. The repo is private, so a token with read access is
    required -- from the argument, or HF_TOKEN in the environment.
    """
    if name not in BUNDLES:
        raise KeyError(
            f"unknown bundle {name!r}; known: {sorted(BUNDLES)}. Pass a path "
            f"to your own BED instead to use a custom set.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "fetch_bundle needs huggingface_hub (pip install huggingface_hub); "
            "or download the BED yourself and pass its path.")
    assembly, files = BUNDLES[name]
    out = []
    for f in files:
        p = hf_hub_download(
            BUNDLE_REPO, f"{BUNDLE_PREFIX}/{assembly.lower()}/{f}",
            repo_type="dataset", cache_dir=cache_dir,
            token=token or os.environ.get("HF_TOKEN"))
        out.append(read_bed(p))
    return out


def resolve_regions(spec, cache_dir=None, token=None) -> List[Intervals]:
    """Turn a user's `exclude_regions=` into interval dicts.

    Accepts a bundle name, a path to a BED, an already-parsed interval dict, or
    any sequence mixing those -- so a caller can combine a curated bundle with
    their own file in one call.
    """
    if spec is None:
        return []
    if isinstance(spec, dict):
        return [spec]
    if isinstance(spec, (str, os.PathLike)):
        spec = [spec]
    out: List[Intervals] = []
    for item in spec:
        if isinstance(item, dict):
            out.append(item)
        elif str(item) in BUNDLES:
            out.extend(fetch_bundle(str(item), cache_dir=cache_dir, token=token))
        else:
            out.append(read_bed(item))
    return out
