"""Population structure: PCA / KING / GRM.

`grm`, `pcs_from_grm` and `king` are implemented. `pca` (variant-axis) and
`pc_project` remain v0.2 stubs -- `pcs_from_grm` covers the case cugen.ld
needs, which is a `structure=` basis for ancestry-adjusted LD.

`grm` and `king` answer different questions and are not substitutes. A GRM
centres on the sample's own allele frequencies, so it measures similarity
relative to the cohort mean and cannot separate "related" from "same
ancestry" -- which is exactly why its eigenvectors are ancestry PCs. `king`
conditions on each pair's heterozygosity instead, so shared ancestry does not
become apparent kinship. Use the GRM to describe structure, KING to detect
relatives.
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


def king(cugen: Union[str, Path], *, variant_range=None, variants=None,
         maf_min: float = 0.0, estimator: str = "between-family",
         tile_size: Optional[int] = None, backend: str = "auto",
         device: int = 0, verbose: bool = True):
    """KING-robust kinship coefficients, (n_samples, n_samples).

    Manichaikul et al. (2010), Bioinformatics 26:2867. The between-family form
    is what plink2 ``--make-king`` computes, and it is the default here for the
    same reason plink2 makes it one: it ignores pedigree, so it applies to every
    pair without being told which pairs are supposed to be related.

        phi_ij = (2*N_HetHet - 4*N_IBS0 - N_Het_i - N_Het_j + 2*min(N_Het_i, N_Het_j))
                 / (4 * min(N_Het_i, N_Het_j))

    over the markers that pass the filter, where N_HetHet counts variants at
    which both are heterozygous, N_IBS0 counts variants at which one is
    homozygous reference and the other homozygous alternate, and N_Het_i counts
    variants at which i is heterozygous. Duplicates give exactly 0.5,
    parent-offspring and full sibs ~0.25, unrelated ~0, and the diagonal is 0.5
    because phi_ii = (1+F)/2.

    WHY THIS AND NOT A GRM. A GRM centres each marker on the SAMPLE's allele
    frequency, so it measures relatedness relative to the sample mean: two
    unrelated individuals from the same subpopulation share drift history and
    score positively, and nothing in the matrix separates "second cousins" from
    "same continental group". KING conditions on each PAIR's own heterozygosity
    instead, so unrelated pairs WITHIN a population sit at ~0 regardless of how
    differentiated that population is from the rest of the cohort. Use a GRM to
    describe ancestry and KING to ask whether two samples are actually related;
    they are not substitutes.

    WHAT ROBUSTNESS DOES NOT MEAN. Across a population split KING goes
    NEGATIVE, it does not stay at zero -- opposite homozygotes are commoner
    between differentiated groups, which inflates N_IBS0 and drives phi down.
    Measured on two synthetic subpopulations at Fst ~0.25: within-group
    unrelated pairs average -0.006, between-group pairs average -0.202, and
    plink2 returns the same values to 5e-07. So a markedly negative entry is a
    statement about ancestry difference, not about relatedness, and it is the
    conventional signal that a pair should not be compared on this scale at
    all. What the estimator guarantees is the direction that matters for QC: it
    does not turn shared ancestry into apparent kinship.

    Computed as two matrix products over the sample axis -- the same shape of
    work as :func:`grm`, transposed relative to ``cugen.ld``. Writing H, R and A
    for the indicator planes of heterozygous, homozygous-reference and
    homozygous-alternate calls, ``N_HetHet = H'H`` and ``N_IBS0 = R'A + (R'A)'``,
    so the second product serves both orderings and only two GEMMs are needed.
    Missingness costs a third, ``H'M``, because the het counts in the
    denominator must then be PAIRWISE complete: a variant where i is
    heterozygous but j was not called contributes to neither the numerator nor
    the pair's marker count, and using i's total het count there would deflate
    every kinship involving a poorly-called sample.

    Parameters
    ----------
    estimator
        ``'between-family'`` (default, plink2's) or ``'within-family'``, the
        latter being ``(N_HetHet - 2*N_IBS0) / (N_Het_i + N_Het_j)``. The
        within-family form assumes the pair is drawn from one population and is
        biased upward by any structure between them; it is offered because it is
        the more powerful of the two once relatedness is already established.
    maf_min
        Drop variants below this minor allele frequency. Unlike a GRM, KING
        needs no frequency weighting, so filtering here is about excluding
        unreliable markers rather than about the estimator's scale.

    Returns
    -------
    (n_samples, n_samples) float64 ndarray, symmetric, diagonal 0.5.
    """
    path = str(cugen)
    if estimator not in ("between-family", "within-family"):
        raise ValueError(
            f"estimator must be 'between-family' or 'within-family', "
            f"got {estimator!r}")
    if backend not in ("auto", "gpu", "numpy"):
        raise ValueError(
            f"backend must be 'auto', 'gpu' or 'numpy', got {backend!r}")

    reader = read_cugen(path, device=device)
    if int(reader.encoding) != ENCODING_2BIT:
        raise ValueError(
            f"{path} is not 2bit dosage data (encoding={int(reader.encoding)}). "
            f"KING counts GENOTYPE classes (het / hom-ref / hom-alt), which a "
            f"hap2bit file does not carry -- its 2-bit codes are phased "
            f"alleles. Convert with cg.convert.")

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
                f"; its gidx run {gidx_all.min():,}..{gidx_all.max():,}. KING "
                f"over zero markers is undefined.")

    # Decided ONCE from the header, not per tile. The pairwise het count has to
    # accumulate over every marker, so "does this file have missing calls" must
    # not be answered tile by tile: with missingness confined to later tiles, a
    # per-tile test skips H'M on the fully-called tiles and then uses the
    # resulting partial matrix as the denominator for all of them. That deflated
    # every pair by up to 0.64 against plink2 before this was pinned by
    # test_missingness_confined_to_later_tiles.
    has_missing = bool(reader.has_missing)
    need_obs = has_missing or maf_min > 0

    HH = xp.zeros((n, n), dtype=xp.float64)
    RA = xp.zeros((n, n), dtype=xp.float64)
    HM = xp.zeros((n, n), dtype=xp.float64) if has_missing else None
    het_tot = xp.zeros(n, dtype=xp.float64)
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

        obs = None
        if need_obs:
            obs = codes != 3
            n_obs = obs.sum(axis=1)
            keep = n_obs > 0
            if maf_min > 0:
                # only computed when it is actually needed; on a fully-called
                # file with maf_min=0 this pass over the tile is skipped
                with np.errstate(invalid="ignore", divide="ignore"):
                    freq = (xp.where(obs, codes, 0).sum(axis=1)
                            / (2.0 * xp.maximum(n_obs, 1)))
                keep &= xp.minimum(freq, 1.0 - freq) >= maf_min
            if not bool(keep.any()):
                continue
            codes, obs = codes[keep], obs[keep]

        # float32 is exact here: the planes are 0/1, so each product term needs
        # one mantissa bit, and a tile accumulates at most `tile` of them --
        # far inside fp32's exact-integer range of 2**24. The running totals are
        # float64, so the sum over tiles stays exact for any number of markers.
        #
        # H'H is written as `Hf.T @ Hf` deliberately: BLAS recognises the
        # symmetric form and dispatches SYRK, which is measurably half the cost
        # of the general product (43.7 ms vs 82.3 ms per 8192-marker tile at
        # n=2504). R'A cannot take that path -- the operands differ -- and the
        # symmetric rewrites of IBS0 all cost more, e.g. O'O - R'R - A'A is
        # three SYRKs where this is one GEMM.
        Hf = (codes == 1).astype(xp.float32)
        HH += Hf.T @ Hf
        RA += (codes == 0).astype(xp.float32).T @ (codes == 2).astype(xp.float32)
        if has_missing:
            HM += Hf.T @ obs.astype(xp.float32)
        else:
            het_tot += Hf.sum(axis=0, dtype=xp.float64)
        m_used += int(codes.shape[0])

    if m_used == 0:
        raise ValueError(
            f"no variant in {path} passed the filter (maf_min={maf_min}); "
            f"KING is undefined with zero markers.")

    ibs0 = RA + RA.T
    if has_missing:
        # h_i restricted to the pair's co-observed markers, and its transpose
        h_i, h_j = HM, HM.T
    else:
        h_i = xp.broadcast_to(het_tot[:, None], (n, n))
        h_j = xp.broadcast_to(het_tot[None, :], (n, n))

    with np.errstate(invalid="ignore", divide="ignore"):
        if estimator == "between-family":
            mn = xp.minimum(h_i, h_j)
            num = 2.0 * HH - 4.0 * ibs0 - h_i - h_j + 2.0 * mn
            den = 4.0 * mn
        else:
            num = HH - 2.0 * ibs0
            den = h_i + h_j
        K = xp.where(den > 0, num / xp.where(den > 0, den, 1.0), xp.nan)
    K = 0.5 * (K + K.T)          # kill fp asymmetry from the two products

    if verbose:
        print(f"cugen.popstruct: KING {n} x {n} from {m_used:,} of "
              f"{hi - lo:,} variants  (estimator={estimator}, "
              f"missing={'yes' if has_missing else 'no'}, "
              f"backend={'gpu' if xp is not np else 'numpy'})")
    return xp.asnumpy(K) if xp is not np else K


def pc_project(*a, **kw):
    return _stub("popstruct.pc_project")
