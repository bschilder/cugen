"""Population structure: PCA / KING / GRM.

`grm` and `pcs_from_grm` are implemented. `pca` (variant-axis), `king` and
`pc_project` remain v0.2 stubs -- `pcs_from_grm` covers the case cugen.ld
needs, which is a `structure=` basis for ancestry-adjusted LD.
"""

from typing import Optional, Union
from pathlib import Path

import numpy as np

from ._stubs import _stub
from .io import ENCODING_2BIT, read_cugen

__all__ = ["grm", "pcs_from_grm", "king", "pca", "pc_project"]

# 2-bit codes are big-endian within the byte: sample 0 is the HIGH pair. Same
# convention as the LD kernels (see cugen/ld.py) and cugen/write.pack_2bit.
_SHIFTS = np.array([6, 4, 2, 0], dtype=np.uint8)


def _unpack_tile(packed, n_samples):
    """(tile_variants, n_samples) uint8 dosage codes, 3 = missing.

    Vectorised on purpose. CugenReader.read_to_numpy does this with a nested
    Python loop over every (variant, sample) cell -- its own docstring says
    "Slow" -- which is fine for a 12-variant fixture and unusable for a GRM,
    where the whole point is to stream a few hundred thousand variants.
    """
    codes = (packed[:, :, None] >> _SHIFTS) & np.uint8(3)
    return codes.reshape(packed.shape[0], -1)[:, :n_samples]


def grm(cugen: Union[str, Path], *, variant_range=None, variants=None,
        maf_min: float = 0.0, standardize: str = "yang",
        tile_size: Optional[int] = None, backend: str = "auto",
        device: int = 0, verbose: bool = True):
    """Genomic relationship matrix, (n_samples, n_samples).

    The GCTA/Yang (2011) standardised form, which is what `--make-grm` produces
    and what HWE-normalised PCA and mixed-model association both start from:

        A_jk = (1/M) sum_i (x_ij - 2p_i)(x_ik - 2p_i) / (2 p_i (1 - p_i))

    over the M variants that pass the frequency filter. The diagonal estimates
    1 + F rather than exactly 1, and it is biased LOW at small n because p_i is
    estimated from the same samples: around 0.88 at n = 22, closing on 1 as n
    grows. That is a property of the estimator, which is what GCTA's --grm-adj
    exists to address, not of this implementation. A sample duplicated against
    itself sits at exactly its own diagonal value.

    Streamed over the variant axis in tiles, accumulating Z.T @ Z, so peak
    memory is the (n_samples, n_samples) accumulator plus one tile rather than
    the whole genotype matrix. The GEMM is over the SAMPLE axis, which is the
    transpose of what cugen.ld does -- same shape of work, different contraction.

    Missing calls are mean-imputed, i.e. standardised to zero. That is the
    conventional choice and it keeps the accumulation a single GEMM; it does
    bias A_jk toward zero for sample pairs with heavy missingness.

    Parameters
    ----------
    variants
        Restrict to these variants, resolved the same way ``cugen.ld`` resolves
        its own ``variants=``: an array of gidx, a DataFrame with a ``gidx``
        column, or a path to one. Matching is on the STORED gidx, not on row
        position -- a per-chromosome file written with ``gidx_start`` has gidx
        that do not equal row positions, and reading them as positions builds
        the GRM from the wrong markers while still returning a plausible PSD
        matrix. This is how an LD-pruned subset gets in; see
        :func:`pcs_from_grm`. Composes with ``variant_range``: both apply, so
        the result is the intersection.

        The scan stays sequential and masks within each tile rather than
        seeking per variant. Reading a pruned tenth of a file costs the whole
        file's sequential I/O, which still beats a million scattered preads.
    maf_min
        Drop variants below this minor allele frequency. Monomorphic variants
        are always dropped -- their standardisation divides by zero.
    standardize
        ``'yang'`` (default) is the GCTA form above, dividing by
        ``sqrt(2p(1-p))`` so every marker contributes equal variance.
        ``'center'`` mean-centres only, which is plink2 ``--make-rel cov``.

        This matters most on an LD-pruned marker set, which is the main input
        here. Pruning at a fixed r^2 preferentially retains LOW-frequency
        markers, because the maximum attainable r^2 between markers of
        frequency a < b is a(1-b)/(b(1-a)) -- a rare marker cannot exceed the
        threshold against a common one, so it survives. Under ``'yang'`` every
        one of those survivors is then upweighted by 1/sqrt(2p(1-p)), and the
        GRM is dominated by its rarest markers. ``'center'`` weights by
        frequency naturally and defuses that without discarding markers.
        Eigenvectors are invariant to a scalar, so this is a genuine change of
        relative marker weighting, not a rescaling.
    backend
        ``'auto'``, ``'gpu'`` or ``'numpy'``.

    Returns
    -------
    (n_samples, n_samples) float64 ndarray, symmetric.
    """
    path = str(cugen)
    if standardize not in ("yang", "center"):
        raise ValueError(
            f"standardize must be 'yang' or 'center', got {standardize!r}")
    if backend not in ("auto", "gpu", "numpy"):
        raise ValueError(
            f"backend must be 'auto', 'gpu' or 'numpy', got {backend!r}")

    reader = read_cugen(path, device=device)
    if int(reader.encoding) != ENCODING_2BIT:
        raise ValueError(
            f"{path} is not 2bit dosage data (encoding={int(reader.encoding)}). "
            f"A GRM is defined on dosages; a hap2bit file carries phased "
            f"alleles, where the 2-bit codes mean something different. Convert "
            f"with cg.convert, or read haplotypes directly.")

    xp = np
    if backend != "numpy":
        try:
            import cupy as _cp  # noqa: PLC0415
            _cp.cuda.Device(device).use()
            xp = _cp
        except Exception:  # noqa: BLE001
            if backend == "gpu":
                raise
            xp = np

    n = int(reader.n_samples)
    p_all = int(reader.n_variants)
    lo, hi = (0, p_all) if variant_range is None else (
        max(0, int(variant_range[0])), min(p_all, int(variant_range[1])))
    bpv = int(reader.bytes_per_variant)
    tile = int(tile_size) if tile_size else 8192

    keep_rows = None
    if variants is not None:
        from .ld import _resolve_gidx  # noqa: PLC0415  (lazy: ld is heavy)
        want = _resolve_gidx(variants)
        gidx_all = np.asarray(reader.gidx, dtype=np.int64)
        keep_rows = np.isin(gidx_all, want)
        if not keep_rows[lo:hi].any():
            raise ValueError(
                f"none of the {want.size:,} requested gidx are in {path}"
                f"{f' within variant_range {(lo, hi)}' if variant_range else ''}"
                f"; its gidx run {gidx_all.min():,}..{gidx_all.max():,}. A GRM "
                f"over zero markers is undefined, and returning one would hide "
                f"the mismatch.")

    A = xp.zeros((n, n), dtype=xp.float64)
    m_used = 0
    for s0 in range(lo, hi, tile):
        s1 = min(s0 + tile, hi)
        if keep_rows is not None and not keep_rows[s0:s1].any():
            continue
        raw = np.frombuffer(reader.read_packed_bytes(s0, s1), dtype=np.uint8)
        codes = _unpack_tile(raw.reshape(s1 - s0, bpv), n)
        if keep_rows is not None:
            codes = codes[keep_rows[s0:s1]]
        codes = xp.asarray(codes)

        obs = codes != 3
        x = xp.where(obs, codes, 0).astype(xp.float64)
        n_obs = obs.sum(axis=1)
        # allele frequency over the NON-MISSING calls only, per variant
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = x.sum(axis=1) / (2.0 * xp.maximum(n_obs, 1))
        keep = (n_obs > 0) & (freq > 0.0) & (freq < 1.0)
        if maf_min > 0:
            keep &= xp.minimum(freq, 1.0 - freq) >= maf_min
        if not bool(keep.any()):
            continue
        x, obs, freq = x[keep], obs[keep], freq[keep]

        two_p = 2.0 * freq[:, None]
        centred = x - two_p
        z = centred if standardize == "center" else \
            centred / xp.sqrt(two_p * (1.0 - freq[:, None]))
        z = xp.where(obs, z, 0.0)
        A += z.T @ z
        m_used += int(keep.sum())

    if m_used == 0:
        raise ValueError(
            f"no variant in {path} passed the filter (maf_min={maf_min}); "
            f"the GRM is undefined with zero markers.")
    A /= float(m_used)
    if verbose:
        print(f"cugen.popstruct: GRM {n} x {n} from {m_used:,} of "
              f"{hi - lo:,} variants  (standardize={standardize}, "
              f"backend={'gpu' if xp is not np else 'numpy'})")
    return xp.asnumpy(A) if xp is not np else A


def pcs_from_grm(grm_matrix, k: int, return_eigenvalues: bool = False):
    """Top-k principal components from a GRM, for `structure=` in cugen.ld.

    A GRM is already the sample-by-sample cross-product PCA decomposes, so this
    is one `eigh` -- trivial at cohort n (2,504 x 2,504 in well under a second)
    and the reason `pca` on the variant axis is not needed for this purpose.
    Eigenvectors come back in DESCENDING eigenvalue order, which `eigh` does not
    give.

    The columns are mean-centred to machine precision because `grm` is built
    from centred genotypes, so `1` already lies in their span's orthogonal
    complement. `cugen.ld._ancestry_basis` re-orthonormalises `[1 | PCs]`
    regardless rather than relying on that.

    FEED IT AN LD-PRUNED SUBSET. This is the failure that actually bites: on an
    unpruned panel a tight LD block acts like a large set of near-duplicate
    markers and dominates the leading eigenvectors, so the "PCs" describe local
    haplotype structure instead of ancestry. Residualising LD on such a basis
    then removes real LD and leaves the population term behind -- the exact
    opposite of the intent. Prune with `cugen.ld.ld_prune` first and pass the
    result straight through -- `grm(path, variants=keep)` takes the gidx frame
    ld_prune returns. A few hundred thousand markers is ample.

    Parameters
    ----------
    grm_matrix
        (n_samples, n_samples) from :func:`grm`, or any symmetric PSD matrix in
        the same sample order as the .cugen file.
    k
        Number of components. ``k = 0`` returns an (n, 0) array, which is the
        no-correction case and is accepted deliberately.
    return_eigenvalues
        Also return the k leading eigenvalues, for a scree plot or for choosing
        k. There is no rule for k; report the sensitivity rather than picking
        one silently.
    """
    A = np.asarray(grm_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"grm must be square (n_samples, n_samples), got {A.shape}")
    n = A.shape[0]
    k = int(k)
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, n_samples={n}], got {k}")
    if k == 0:
        empty = np.zeros((n, 0), dtype=np.float64)
        return (empty, np.zeros(0)) if return_eigenvalues else empty
    w, V = np.linalg.eigh(A)
    order = np.argsort(w)[::-1][:k]
    pcs = np.ascontiguousarray(V[:, order])
    return (pcs, w[order]) if return_eigenvalues else pcs


def pca(*a, **kw):
    return _stub("popstruct.pca")


def king(*a, **kw):
    return _stub("popstruct.king")


def pc_project(*a, **kw):
    return _stub("popstruct.pc_project")
