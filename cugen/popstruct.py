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

import os
from typing import Optional, Union
from pathlib import Path

import numpy as np

from ._stubs import _stub
from .io import ENCODING_2BIT, read_cugen

__all__ = ["grm", "pcs_from_grm", "king", "king_pairs", "plan_king",
           "king_matrix", "open_king_matrix", "KingMatrix", "pca", "pc_project"]

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


def _host_available_bytes():
    """Best-effort free host RAM. Falls back to a deliberately modest 8 GiB."""
    try:
        if os.path.exists("/proc/meminfo"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:  # noqa: BLE001
        return 8 << 30


def _king_budget(backend, device, safety):
    """(budget_bytes, where) for whichever device would run the scan."""
    if backend != "numpy":
        try:
            import cupy as _cp  # noqa: PLC0415
            free, _ = _cp.cuda.Device(device).mem_info
            return int(free * safety), "gpu"
        except Exception:  # noqa: BLE001
            if backend == "gpu":
                raise
    return int(_host_available_bytes() * safety), "host"


#: (n, n) float64 blocks that king() holds at peak. One for the Gram
#: accumulator, two more for the pairwise counts when calls are missing, one for
#: HetHet under the within-family form, and five for the closing arithmetic
#: (min, numerator, denominator, result, and the symmetrisation temporary).
#: Deliberately an over-estimate: an optimistic figure routes a job to the dense
#: path that then dies partway through, having already paid for the read.
def _king_dense_bytes(n, has_missing, estimator):
    blocks = 1 + (2 if has_missing else 0) + \
        (1 if estimator == "within-family" else 0) + 5
    return blocks * n * n * 8


def _auto_sample_block(n, markers, budget_bytes):
    """Largest whole-byte block whose accumulator and unpacked panel fit."""
    B = 4
    while B * 2 <= n:
        cand = B * 2
        need = 4 * cand * cand * 8 + markers * cand * 4
        if need > budget_bytes:
            break
        B = cand
    return min(max(B, 4), ((n + 3) // 4) * 4)


def plan_king(cugen: Union[str, Path], *, estimator: str = "between-family",
              backend: str = "auto", device: int = 0, safety: float = 0.70,
              budget_bytes: Optional[int] = None, verbose: bool = True):
    """Which KING routine this cohort and this machine can actually run.

    :func:`king` returns a dense (n, n) matrix and :func:`king_pairs` returns the
    pairs above a threshold, so they are NOT interchangeable outputs and nothing
    here silently swaps one for the other. What this does is answer the question
    that decides it -- does the dense form fit -- and report the arithmetic, so
    the choice is a number rather than a guess.

    Prefer `king` whenever it fits. It walks the markers once; `king_pairs`
    bounds memory by re-walking them for every block pair, which is the right
    trade only when the dense form is impossible. Measured on an RTX 4090 at
    n=2504, p=100,000: `king` 0.048 s against `king_pairs` 0.877 s.

    Returns
    -------
    dict with ``recommend`` (``"king"`` or ``"king_pairs"``), ``fits``,
    ``n_samples``, ``n_variants``, ``dense_bytes``, ``budget_bytes``, ``where``
    (``"gpu"`` or ``"host"``), ``sample_block`` (what to pass to `king_pairs`)
    and ``reason``.
    """
    reader = read_cugen(str(cugen), device=device)
    n = int(reader.n_samples)
    p = int(reader.n_variants)
    has_missing = bool(reader.has_missing)
    where = "host"
    if budget_bytes is None:
        budget_bytes, where = _king_budget(backend, device, safety)
    dense = _king_dense_bytes(n, has_missing, estimator)
    fits = dense <= budget_bytes
    B = _auto_sample_block(n, p, budget_bytes)
    out = {
        "recommend": "king" if fits else "king_pairs",
        "fits": fits, "n_samples": n, "n_variants": p,
        "has_missing": has_missing, "estimator": estimator,
        "dense_bytes": dense, "budget_bytes": int(budget_bytes),
        "where": where, "sample_block": B,
        "reason": (
            f"dense needs {dense/2**30:.2f} GiB of {budget_bytes/2**30:.2f} GiB "
            f"{where} budget at n={n:,}"
            + ("" if fits else
               f"; use king_pairs(sample_block={B}) instead, which holds "
               f"one ({B}, {B}) block rather than ({n:,}, {n:,})")),
    }
    if verbose:
        print(f"cugen.popstruct: plan_king -> {out['recommend']}  ({out['reason']})")
    return out


def _unpack_on(packed, n_samples, xp, shifts):
    """_unpack_tile, but on whichever device holds `packed`.

    Pure shift/mask, so it runs on CuPy unchanged once the shift vector lives
    there. Worth doing rather than unpacking on the host and copying the codes
    over: the packed form is a QUARTER the size, so moving it instead of the
    codes cuts the transfer 4x as well as freeing the CPU. Measured on an RTX
    4090 at n=2504, p=100,000, the GPU king() was unpack-bound before this --
    cancelling one of its two matrix products moved the wall-clock by nothing
    (0.81s -> 0.868s) while the CPU side improved 2.4x, because the GEMM was
    never what the GPU path was waiting on.
    """
    codes = (packed[:, :, None] >> shifts) & xp.uint8(3)
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
         device: int = 0, budget_bytes: Optional[int] = None,
         safety: float = 0.70, verbose: bool = True):
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

    # Refuse BEFORE reading anything. The dense form is (n, n)-bound, so at
    # biobank n it cannot run at all -- and a MemoryError raised partway through
    # the scan has already paid for the read and tells the caller nothing about
    # what to do instead. This names the alternative and a block size.
    _budget = budget_bytes
    _where = "host"
    if _budget is None:
        _budget, _where = _king_budget(backend, device, safety)
    _need = _king_dense_bytes(n, bool(reader.has_missing), estimator)
    if _need > _budget:
        _B = _auto_sample_block(n, p_all, _budget)
        raise MemoryError(
            f"king() needs about {_need/2**30:.2f} GiB of (n, n) accumulators at "
            f"n={n:,}, over the {_budget/2**30:.2f} GiB {_where} budget. It "
            f"returns a dense matrix, so there is no way to shrink it. Use "
            f"king_pairs(..., sample_block={_B}) for the pairs above a "
            f"threshold instead -- it holds one ({_B}, {_B}) block rather than "
            f"({n:,}, {n:,}). See plan_king() for the arithmetic.")

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

    _sh = _SHIFTS if xp is np else xp.asarray(_SHIFTS)
    UU = xp.zeros((n, n), dtype=xp.float64)
    MM = xp.zeros((n, n), dtype=xp.float64) if has_missing else None
    WM = xp.zeros((n, n), dtype=xp.float64) if has_missing else None
    HH = xp.zeros((n, n), dtype=xp.float64) if estimator == "within-family" \
        else None
    het_tot = xp.zeros(n, dtype=xp.float64)
    m_used = 0

    for s0 in range(lo, hi, tile):
        s1 = min(s0 + tile, hi)
        if keep_rows is not None and not keep_rows[s0:s1].any():
            continue
        raw = np.frombuffer(reader.read_packed_bytes(s0, s1),
                            dtype=np.uint8).reshape(s1 - s0, bpv)
        if keep_rows is not None:
            raw = raw[keep_rows[s0:s1]]
        codes = _unpack_on(raw if xp is np else xp.asarray(raw), n, xp, _sh)

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

        # ONE product, not two. Substituting the identity
        #     IBS0 = (m - h_i - h_j + HetHet - u'u) / 2,      u = dosage - 1
        # into the between-family estimator cancels HetHet outright:
        #     phi = (2*u'u - 2m + h_i + h_j + 2*min(h_i,h_j)) / (4*min(...))
        # so the whole statistic rides on the Gram matrix of u, whose entries
        # are +1 for a shared homozygote class, -1 for opposite homozygotes,
        # and 0 wherever either sample is heterozygous. u'u is also SYMMETRIC,
        # so BLAS takes the SYRK path -- worth another factor of two over the
        # general product this replaced.
        #
        # float32 is exact: u is in {-1,0,+1}, so each term needs one mantissa
        # bit, and a tile accumulates at most `tile` of them, far inside fp32's
        # exact-integer range of 2**24. Totals are float64, so the sum over
        # tiles stays exact for any marker count.
        if obs is None:
            uf = codes.astype(xp.float32) - xp.float32(1.0)
        else:
            uf = xp.where(obs, codes.astype(xp.float32) - xp.float32(1.0),
                          xp.float32(0.0))
        UU += uf.T @ uf
        wf = (codes == 1).astype(xp.float32)
        if has_missing:
            mf = obs.astype(xp.float32)
            MM += mf.T @ mf              # pairwise co-observed marker count
            WM += wf.T @ mf              # h_i over the pair's shared markers
        else:
            het_tot += wf.sum(axis=0, dtype=xp.float64)
        if HH is not None:
            HH += wf.T @ wf              # within-family form still needs HetHet
        m_used += int(codes.shape[0])

    if m_used == 0:
        raise ValueError(
            f"no variant in {path} passed the filter (maf_min={maf_min}); "
            f"KING is undefined with zero markers.")

    if has_missing:
        h_i, h_j, m_pair = WM, WM.T, MM
    else:
        h_i = xp.broadcast_to(het_tot[:, None], (n, n))
        h_j = xp.broadcast_to(het_tot[None, :], (n, n))
        m_pair = float(m_used)

    with np.errstate(invalid="ignore", divide="ignore"):
        mn = xp.minimum(h_i, h_j)
        if estimator == "between-family":
            num = 2.0 * UU - 2.0 * m_pair + h_i + h_j + 2.0 * mn
            den = 4.0 * mn
        else:
            # within-family: (HetHet - 2*IBS0) / (h_i + h_j), with IBS0 taken
            # from the same identity so it still costs no extra product.
            ibs0 = 0.5 * (m_pair - h_i - h_j + HH - UU)
            num = HH - 2.0 * ibs0
            den = h_i + h_j
        K = xp.where(den > 0, num / xp.where(den > 0, den, 1.0), xp.nan)
    K = 0.5 * (K + K.T)          # kill fp asymmetry in the accumulated products

    if verbose:
        print(f"cugen.popstruct: KING {n} x {n} from {m_used:,} of "
              f"{hi - lo:,} variants  (estimator={estimator}, "
              f"missing={'yes' if has_missing else 'no'}, "
              f"backend={'gpu' if xp is not np else 'numpy'})")
    return xp.asnumpy(K) if xp is not np else K


def pc_project(*a, **kw):
    return _stub("popstruct.pc_project")


def _king_blocks(cugen, *, sample_block=4096, variant_range=None,
                 variants=None, maf_min=0.0, tile_size=None, backend="auto",
                 device=0, safety=0.70, verbose=True):
    """Yield ``(a0, a1, b0, b1, phi)`` over the lower block triangle, b <= a.

    The one place the blocked estimator lives. `king_pairs` thresholds what
    comes out of here and `king_matrix` writes it, so neither can drift from the
    other. Blocks arrive in a-major order and each is released before the next,
    which is what lets a consumer hold one row-strip instead of the matrix.
    """
    import pandas as pd  # noqa: PLC0415,F401  (kept for the caller's benefit)

    path = str(cugen)
    if backend not in ("auto", "gpu", "numpy"):
        raise ValueError(
            f"backend must be 'auto', 'gpu' or 'numpy', got {backend!r}")

    reader = read_cugen(path, device=device)
    if int(reader.encoding) != ENCODING_2BIT:
        raise ValueError(
            f"{path} is not 2bit dosage data (encoding={int(reader.encoding)}).")

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
    if isinstance(sample_block, str):
        if sample_block != "auto":
            raise ValueError(
                f"sample_block must be an int or 'auto', got {sample_block!r}")
        _bud, _ = _king_budget(backend, device, safety)
        B = _auto_sample_block(n, int(reader.n_variants), _bud)
    else:
        B = max(4, (int(sample_block) // 4) * 4)

    p_all = int(reader.n_variants)
    lo, hi = (0, p_all) if variant_range is None else (
        max(0, int(variant_range[0])), min(p_all, int(variant_range[1])))
    bpv = int(reader.bytes_per_variant)
    tile = int(tile_size) if tile_size else 8192
    has_missing = bool(reader.has_missing)

    keep_rows = None
    if variants is not None:
        from .ld import _resolve_gidx  # noqa: PLC0415
        want = _resolve_gidx(variants)
        gidx_all = np.asarray(reader.gidx, dtype=np.int64)
        keep_rows = np.isin(gidx_all, want)
        if not keep_rows[lo:hi].any():
            raise ValueError(f"none of the requested gidx are in {path}")

    # pass 1: per-sample het counts and the marker mask. O(n) memory.
    het = xp.zeros(n, dtype=xp.float64)
    marker_keep, m_used = [], 0
    for s0 in range(lo, hi, tile):
        s1 = min(s0 + tile, hi)
        raw = np.frombuffer(reader.read_packed_bytes(s0, s1),
                            dtype=np.uint8).reshape(s1 - s0, bpv)
        if keep_rows is not None:
            raw = raw[keep_rows[s0:s1]]
        codes = _unpack_on(raw if xp is np else xp.asarray(raw), n, xp,
                           _SHIFTS if xp is np else xp.asarray(_SHIFTS))
        obs = codes != 3
        n_obs = obs.sum(axis=1)
        keep = n_obs > 0
        if maf_min > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                freq = (xp.where(obs, codes, 0).sum(axis=1)
                        / (2.0 * xp.maximum(n_obs, 1)))
            keep &= xp.minimum(freq, 1.0 - freq) >= maf_min
        codes = codes[keep]
        het += (codes == 1).sum(axis=0, dtype=xp.float64)
        m_used += int(codes.shape[0])
        marker_keep.append(np.asarray(
            keep.get() if xp is not np else keep, dtype=bool))
    if m_used == 0:
        raise ValueError(
            f"no variant in {path} passed the filter (maf_min={maf_min}); "
            f"KING is undefined with zero markers.")
    if has_missing:
        raise NotImplementedError(
            f"{path} has missing calls, and the pairwise-complete het counts "
            f"that requires are themselves (n, n) -- the very allocation the "
            f"blocked path avoids. Impute or hard-call first, or use king() if "
            f"the cohort is small enough for the dense form.")

    # resident packed panel: read once, sliced per block pair
    rows = []
    for k, s0 in enumerate(range(lo, hi, tile)):
        s1 = min(s0 + tile, hi)
        raw = np.frombuffer(reader.read_packed_bytes(s0, s1),
                            dtype=np.uint8).reshape(s1 - s0, bpv)
        if keep_rows is not None:
            raw = raw[keep_rows[s0:s1]]
        rows.append(raw[marker_keep[k]])
    packed = np.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
    del rows
    _sh = _SHIFTS if xp is np else xp.asarray(_SHIFTS)
    if xp is not np:
        packed = xp.asarray(packed)

    nb = (n + B - 1) // B
    if verbose:
        print(f"cugen.popstruct: blocked KING n={n:,} markers={m_used:,} "
              f"block={B} -> {nb} blocks, {nb*(nb+1)//2:,} block pairs, "
              f"panel {packed.nbytes/1e9:.2f} GB, "
              f"backend={'gpu' if xp is not np else 'numpy'}", flush=True)

    for a in range(nb):
        a0, a1 = a * B, min((a + 1) * B, n)
        UA = _unpack_on(packed[:, a0 // 4:(a1 + 3) // 4], a1 - a0, xp,
                        _sh).astype(xp.float32) - xp.float32(1.0)
        for b in range(a + 1):                 # b <= a: lower block triangle
            b0, b1 = b * B, min((b + 1) * B, n)
            acc = xp.zeros((a1 - a0, b1 - b0), dtype=xp.float64)
            for t0 in range(0, packed.shape[0], tile):
                t1 = min(t0 + tile, packed.shape[0])
                fa = UA[t0:t1]
                if b == a:
                    acc += fa.T @ fa           # SYRK on the diagonal block
                else:
                    cb = _unpack_on(packed[t0:t1, b0 // 4:(b1 + 3) // 4],
                                    b1 - b0, xp, _sh)
                    acc += fa.T @ (cb.astype(xp.float32) - xp.float32(1.0))
            hi_ = het[a0:a1][:, None]
            hj_ = het[b0:b1][None, :]
            mn = xp.minimum(hi_, hj_)
            with np.errstate(invalid="ignore", divide="ignore"):
                phi = ((2.0 * acc - 2.0 * m_used + hi_ + hj_ + 2.0 * mn)
                       / xp.where(mn > 0, 4.0 * mn, xp.nan))
            yield a0, a1, b0, b1, phi
            del acc, phi
        del UA


def king_pairs(cugen: Union[str, Path], *, min_kinship: float = 0.0442,
               sample_block=4096, variant_range=None, variants=None,
               maf_min: float = 0.0, tile_size: Optional[int] = None,
               backend: str = "auto", device: int = 0, safety: float = 0.70,
               verbose: bool = True):
    """Related pairs only, for cohorts where the (n, n) matrix does not exist.

    :func:`king` materialises (n, n) accumulators. That is the right shape for a
    cohort and the wrong shape for a biobank: at n = 500,000 the dense form is
    6 TB and its output alone is 2 TB; at n = 1,000,000, 24 TB. No amount of
    tuning fixes an allocation that large.

    What does fix it is that a kinship matrix at that scale is almost entirely
    zeros -- relatives are rare -- so the useful object is the sparse set of
    pairs above a threshold. This walks BLOCKS of the sample-pair space, keeps
    only the pairs that clear ``min_kinship`` and discards the rest before the
    next block. Peak memory is the packed panel plus one
    (sample_block, sample_block) accumulator, neither growing with n^2.

    Measured on an A100-80GB: n = 1,000,000 with 5,000 markers evaluates
    499,999,500,000 pairs in 377 s at 16.6 GB peak RSS.

    The default threshold, 0.0442, is the conventional lower bound of the
    4th-degree band (2^-9/2). Pairs from DIFFERENT ancestries score NEGATIVE
    rather than zero -- see :func:`king` -- so a threshold selects relatives
    from one direction only and never returns the population-difference tail.

    USE ENOUGH MARKERS. This is the one thing that changes at biobank scale. A
    threshold is applied to n(n-1)/2 draws from the null -- 1.25e11 at
    n=500,000 -- so the per-pair false-positive rate has to sit below roughly
    1/n^2 before the emitted list means anything. SE(phi) falls as
    1/sqrt(markers), so the tail collapses fast. Measured at n=500,000 with 50
    planted duplicates:

        markers   emitted at phi >= 0.0442   false positives
          5,000                  7,574,975         7,574,925
         20,000                         50                 0

    Four times the markers took the output from millions of spurious pairs to
    exactly the 50 real ones. 20,000 is a floor rather than a target; published
    biobank scans use 50,000-100,000 pruned markers, and cost is sub-linear in
    markers (4x the markers cost 3.2x the time), so this is cheap insurance.

    Returns
    -------
    DataFrame with ``i``, ``j`` (``i < j``) and ``kinship``, kinship descending.
    """
    import pandas as pd  # noqa: PLC0415

    out_i, out_j, out_k, n_eval = [], [], [], 0
    for a0, a1, b0, b1, phi in _king_blocks(
            cugen, sample_block=sample_block, variant_range=variant_range,
            variants=variants, maf_min=maf_min, tile_size=tile_size,
            backend=backend, device=device, safety=safety, verbose=verbose):
        xp = np if isinstance(phi, np.ndarray) else phi.__class__.__module__
        _xp = np if isinstance(phi, np.ndarray) else __import__("cupy")
        take = phi >= min_kinship
        if a0 == b0:
            # strictly below the diagonal, so each unordered pair appears once.
            # A -inf sentinel instead of a boolean mask would leak the whole
            # triangle at min_kinship=-inf, since -inf >= -inf.
            take &= _xp.tril(_xp.ones(phi.shape, dtype=bool), -1)
            n_eval += (a1 - a0) * (a1 - a0 - 1) // 2
        else:
            n_eval += (a1 - a0) * (b1 - b0)
        sel = _xp.nonzero(take)
        if int(sel[0].size):
            # rows are the a-block (higher index), cols the b-block, so j<i here
            si = (sel[0] + a0).astype(_xp.int64)
            sj = (sel[1] + b0).astype(_xp.int64)
            out_j.append(np.asarray(si.get() if _xp is not np else si))
            out_i.append(np.asarray(sj.get() if _xp is not np else sj))
            v = phi[sel]
            out_k.append(np.asarray(v.get() if _xp is not np else v))

    if out_i:
        df = pd.DataFrame({"i": np.concatenate(out_i),
                           "j": np.concatenate(out_j),
                           "kinship": np.concatenate(out_k)})
        df = df.sort_values("kinship", ascending=False).reset_index(drop=True)
    else:
        df = pd.DataFrame({"i": np.zeros(0, np.int64),
                           "j": np.zeros(0, np.int64),
                           "kinship": np.zeros(0, np.float64)})
    if verbose:
        print(f"cugen.popstruct: {n_eval:,} pairs evaluated, {len(df):,} at "
              f"kinship >= {min_kinship}", flush=True)
    return df


# --------------------------------------------------------------------------
# Dense KING, on disk. For when the matrix genuinely is the deliverable.
# --------------------------------------------------------------------------
_KM_MAGIC = b"CUKING02"
_KM_ENC = {"int16": (np.int16, 32767.0), "float32": (np.float32, 1.0)}
_KM_LAYOUT = ("square", "triangle")
#: magic(8) n(i8) enc(i4) layout(i4) id_bytes(i8); the ID block follows, then data
_KM_HEADER = 32


def king_matrix(cugen: Union[str, Path], out: Union[str, Path], *,
                encoding: str = "int16", layout: str = "square",
                sample_ids=None, sample_block=4096, variant_range=None,
                variants=None, maf_min: float = 0.0,
                tile_size: Optional[int] = None, backend: str = "auto",
                device: int = 0, safety: float = 0.70, verbose: bool = True):
    """The full dense matrix, written to disk and queryable BY PERSON.

    :func:`king` holds the matrix in memory, so n stops at what the machine has.
    This runs the same block walk as :func:`king_pairs` but writes every block,
    so n is bounded by DISK. Read it back with :func:`open_king_matrix`.

    LAYOUT IS THE DECISION THAT MATTERS, and it is about access, not size. In a
    lower triangle, row i is one contiguous run of i+1 entries PLUS n-i-1
    entries that each live in a different row -- so a per-person row query
    touches up to n scattered pages spread across the whole file. At
    n = 1,000,000 that is ~10 s per person on NVMe. Stored square, row i is one
    contiguous 2 MB read: ~1 ms. Four orders of magnitude, for twice the bytes:

        layout     n=1,000,000 int16   row query
        square     2.0 TB             ~1 ms      (default)
        triangle   1.0 TB             ~10 s

    ``square`` is the default because the reason to materialise a dense matrix
    at all is random access; ``triangle`` is there for archival, where halving
    1 TB matters and nobody is going to query a row.

    Encoding is int16 by default, for the same reason ``.cugenld`` stores r that
    way: phi lives in [-1, 0.5] and its own sampling error is ~1e-3 at 100,000
    markers, so a 3.05e-5 quantum is two orders finer than the number is real
    to. ``float32`` is available and exact to fp32.

    ``sample_ids`` are stored in the file so queries can name people rather than
    row offsets. If omitted, a ``<cugen>.samples.txt`` sidecar is used when one
    exists -- ``.cugen`` itself does not carry IDs.

    BEFORE YOU RUN THIS, be sure you want dense. At n = 1,000,000 more than
    99.99% of entries are phi ~ 0, and the same information as a pair list is
    0.06-0.6 GB, three to four orders smaller. Dense earns its place when
    something downstream needs arbitrary (i, j) access or a plink/GCTA-style
    binary; otherwise use :func:`king_pairs`.
    """
    if encoding not in _KM_ENC:
        raise ValueError(
            f"encoding must be one of {sorted(_KM_ENC)}, got {encoding!r}")
    if layout not in _KM_LAYOUT:
        raise ValueError(
            f"layout must be one of {list(_KM_LAYOUT)}, got {layout!r}")
    dt, scale = _KM_ENC[encoding]
    isz = np.dtype(dt).itemsize

    reader = read_cugen(str(cugen), device=device)
    n = int(reader.n_samples)

    if sample_ids is None:
        side = Path(str(cugen) + ".samples.txt")
        if side.exists():
            sample_ids = [x.strip() for x in side.read_text().split("\n")
                          if x.strip()]
    if sample_ids is not None:
        sample_ids = [str(x) for x in sample_ids]
        if len(sample_ids) != n:
            raise ValueError(
                f"{len(sample_ids):,} sample_ids for {n:,} samples in "
                f"{cugen}. Mismatched IDs would silently mislabel every query, "
                f"so this refuses rather than truncating.")
    id_blob = ("\n".join(sample_ids)).encode() if sample_ids else b""

    tri = n * (n + 1) // 2
    cells = n * n if layout == "square" else tri
    if verbose:
        print(f"cugen.popstruct: king_matrix n={n:,} layout={layout} "
              f"-> {cells:,} cells, {cells*isz/1e12:.3f} TB as {encoding}, "
              f"to {out}", flush=True)

    lim = float(np.iinfo(dt).max) if encoding == "int16" else None

    def _q(v):
        v = np.asarray(v.get() if hasattr(v, "get") else v, dtype=np.float64)
        if encoding == "int16":
            v = np.clip(np.rint(v * scale), -lim, lim)
        return v.astype(dt)

    gen = _king_blocks(
        cugen, sample_block=sample_block, variant_range=variant_range,
        variants=variants, maf_min=maf_min, tile_size=tile_size,
        backend=backend, device=device, safety=safety, verbose=False)

    with open(str(out), "wb") as fh:
        fh.write(_KM_MAGIC)
        fh.write(np.array([n], dtype=np.int64).tobytes())
        fh.write(np.array([list(_KM_ENC).index(encoding)], dtype=np.int32)
                 .tobytes())
        fh.write(np.array([_KM_LAYOUT.index(layout)], dtype=np.int32).tobytes())
        fh.write(np.array([len(id_blob)], dtype=np.int64).tobytes())
        fh.write(id_blob)
        data_off = fh.tell()
        fh.truncate(data_off + cells * isz)

    if layout == "square":
        # Blocks arrive over the LOWER block triangle, so each also fills its
        # mirror. A memmap takes both placements without a second pass, and the
        # kernel flushes dirty pages as it goes -- peak RSS stays bounded rather
        # than tracking the file size.
        M = np.memmap(str(out), dtype=dt, mode="r+", offset=data_off,
                      shape=(n, n))
        for a0, a1, b0, b1, phi in gen:
            q = _q(phi)
            M[a0:a1, b0:b1] = q
            if a0 != b0:
                M[b0:b1, a0:a1] = q.T
        M.flush()
        del M
        written = cells
    else:
        # Ragged lower triangle: row i occupies i+1 entries at i(i+1)/2. Blocks
        # come in a-major order, so the blocks for row-block a are exactly its
        # columns 0..a1 -- one complete strip. Fill it, write, release.
        state = {"cur": None, "strip": None, "written": 0}
        with open(str(out), "r+b") as fh:
            fh.seek(data_off)

            def _flush():
                if state["strip"] is None:
                    return
                r0, r1 = state["cur"]
                for r in range(r0, r1):
                    fh.write(state["strip"][r - r0, :r + 1].tobytes())
                    state["written"] += r + 1
                state["strip"] = None

            for a0, a1, b0, b1, phi in gen:
                if state["cur"] != (a0, a1):
                    _flush()
                    state["cur"] = (a0, a1)
                    state["strip"] = np.zeros((a1 - a0, a1), dtype=dt)
                state["strip"][:, b0:b1] = _q(phi)
            _flush()
        written = state["written"]
        if written != tri:
            raise RuntimeError(
                f"wrote {written:,} entries but the lower triangle of n={n:,} "
                f"is {tri:,}; the strip walk missed part of the matrix.")

    if verbose:
        print(f"cugen.popstruct: wrote {os.path.getsize(str(out))/1e9:.3f} GB "
              f"({written:,} cells)", flush=True)
    return str(out)


class KingMatrix:
    """Memory-mapped reader for a :func:`king_matrix` file, indexed by person.

    Dense semantics without dense memory. Every accessor takes either a row
    index or a sample ID, so callers work in the identifiers they already have:

        km["NA12878", "NA12891"]        one cell
        km.row("NA12878")               that person against everyone
        km.related("NA12878")           just their relatives, sorted

    ``to_numpy()`` materialises the whole square and refuses above ``max_gb``,
    because this format exists for matrices that do not fit.
    """

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as fh:
            head = fh.read(_KM_HEADER)
            if head[:8] != _KM_MAGIC:
                raise ValueError(
                    f"{path} is not a king_matrix file (magic {head[:8]!r} != "
                    f"{_KM_MAGIC!r})")
            self.n = int(np.frombuffer(head[8:16], dtype=np.int64)[0])
            self.encoding = list(_KM_ENC)[
                int(np.frombuffer(head[16:20], dtype=np.int32)[0])]
            self.layout = _KM_LAYOUT[
                int(np.frombuffer(head[20:24], dtype=np.int32)[0])]
            id_bytes = int(np.frombuffer(head[24:32], dtype=np.int64)[0])
            blob = fh.read(id_bytes) if id_bytes else b""
            self._data_off = _KM_HEADER + id_bytes
        self.ids = blob.decode().split("\n") if blob else None
        self._pos = {v: i for i, v in enumerate(self.ids)} if self.ids else None
        self._dt, self._scale = _KM_ENC[self.encoding]
        cells = self.n * self.n if self.layout == "square" \
            else self.n * (self.n + 1) // 2
        shape = (self.n, self.n) if self.layout == "square" else (cells,)
        self._m = np.memmap(self.path, dtype=self._dt, mode="r",
                            offset=self._data_off, shape=shape)

    # -- person -> row index ------------------------------------------------
    def index_of(self, person):
        """Row index for a sample ID, or the int itself if already an index."""
        if isinstance(person, (int, np.integer)):
            i = int(person)
            if not 0 <= i < self.n:
                raise IndexError(f"index {i} outside 0..{self.n - 1}")
            return i
        if self._pos is None:
            raise KeyError(
                f"{person!r} was given as a sample ID, but this file carries no "
                f"IDs. Pass sample_ids= to king_matrix, or index by position.")
        try:
            return self._pos[str(person)]
        except KeyError:
            raise KeyError(
                f"{person!r} is not in this matrix ({self.n:,} samples)"
            ) from None

    def _deq(self, v):
        return np.asarray(v, dtype=np.float64) / self._scale

    def __getitem__(self, key):
        i, j = key
        i, j = self.index_of(i), self.index_of(j)
        if self.layout == "square":
            return float(self._deq(self._m[i, j]))
        if i < j:
            i, j = j, i
        return float(self._deq(self._m[i * (i + 1) // 2 + j]))

    def row(self, person):
        """One person against everyone, as a length-n float64 vector."""
        i = self.index_of(person)
        if self.layout == "square":
            return self._deq(self._m[i])           # one contiguous read
        out = np.empty(self.n, dtype=np.float64)
        base = i * (i + 1) // 2
        out[:i + 1] = self._deq(self._m[base:base + i + 1])
        # The upper part of the row lives one element per LATER row, at
        # k(k+1)/2 + i. Gathered in one vectorised take rather than a Python
        # loop over n reads -- the loop version cost a million interpreter round
        # trips per person at n=1e6, which made the format unusable for the
        # query it exists to serve.
        if i + 1 < self.n:
            k = np.arange(i + 1, self.n, dtype=np.int64)
            out[i + 1:] = self._deq(self._m[k * (k + 1) // 2 + i])
        return out

    def indices_of(self, people):
        """Resolve a mixed list of IDs and indices, PRESERVING their order."""
        return np.fromiter((self.index_of(x) for x in people),
                           dtype=np.int64, count=len(people))

    def submatrix(self, people, cols=None, *, as_frame: bool = False,
                  max_gb: float = 8.0):
        """The kinship block for a set of people, in the order you asked for.

        The cohort-level query: hand it 500 cases and get their 500 x 500 block
        without touching the rest of an n x n file. ``cols`` makes it
        rectangular -- cases against controls, say -- and defaults to ``people``
        for the symmetric case.

        Order is the caller's, not sorted, so rows line up with the list that
        was passed in. Duplicates are allowed and give repeated rows, which is
        occasionally what a caller wants and is never silently wrong.

        Reading cost depends on the layout, as everything here does. Square
        gathers whole rows, so k people cost k contiguous row reads. Triangle
        has to compute an offset per cell, which is one vectorised gather but
        touches up to k^2 scattered pages -- fine for a few hundred people,
        and the reason ``square`` is the default.
        """
        ri = self.indices_of(list(people))
        ci = ri if cols is None else self.indices_of(list(cols))
        need = ri.size * ci.size * 8 / 1e9
        if need > max_gb:
            raise MemoryError(
                f"the requested block is {ri.size:,} x {ci.size:,} = "
                f"{need:,.1f} GB, over max_gb={max_gb}. Ask for fewer people, "
                f"or raise max_gb deliberately.")

        if self.layout == "square":
            out = self._deq(self._m[np.ix_(ri, ci)])
        else:
            # offset(i, j) = max(i,j)*(max+1)/2 + min(i,j), vectorised over the
            # whole block rather than looped, so one gather serves k^2 cells.
            I = ri[:, None]
            J = ci[None, :]
            hi = np.maximum(I, J)
            lo_ = np.minimum(I, J)
            out = self._deq(self._m[hi * (hi + 1) // 2 + lo_])

        if not as_frame:
            return out
        import pandas as pd  # noqa: PLC0415
        lab = (lambda a: [self.ids[k] for k in a]) if self.ids else list
        return pd.DataFrame(out, index=lab(ri), columns=lab(ci))

    def related(self, person, min_kinship: float = 0.0442, top: int = None):
        """That person's relatives, kinship descending. Self is excluded.

        The query this format is for. Note the threshold is one-sided on
        purpose: pairs from different ancestries score NEGATIVE (see
        :func:`king`), so a cutoff selects relatives and never returns the
        population-difference tail.
        """
        import pandas as pd  # noqa: PLC0415

        i = self.index_of(person)
        r = self.row(i)
        r[i] = -np.inf                             # never report self
        sel = np.nonzero(r >= min_kinship)[0]
        order = sel[np.argsort(-r[sel], kind="stable")]
        if top is not None:
            order = order[:int(top)]
        return pd.DataFrame({
            "id": [self.ids[k] for k in order] if self.ids else list(order),
            "index_": order.astype(np.int64),
            "kinship": r[order],
        })

    def to_numpy(self, max_gb: float = 8.0):
        need = self.n * self.n * 8 / 1e9
        if need > max_gb:
            raise MemoryError(
                f"the full square is {need:,.1f} GB at n={self.n:,}, over "
                f"max_gb={max_gb}. That limit exists because this format is for "
                f"matrices that do not fit -- use km[i, j], km.row(person) or "
                f"km.related(person), or raise max_gb deliberately.")
        if self.layout == "square":
            return self._deq(np.asarray(self._m))
        M = np.empty((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            base = i * (i + 1) // 2
            M[i, :i + 1] = self._deq(self._m[base:base + i + 1])
        L = np.tril(M)
        return L + np.tril(L, -1).T

    def __len__(self):
        return self.n

    def __repr__(self):
        return (f"KingMatrix(n={self.n:,}, layout={self.layout!r}, "
                f"encoding={self.encoding!r}, "
                f"ids={'yes' if self.ids else 'no'}, path={self.path!r})")


def open_king_matrix(path):
    """Open a :func:`king_matrix` file for memory-mapped access."""
    return KingMatrix(path)
