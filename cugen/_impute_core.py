"""cugen._impute_core - the Li and Stephens imputation numerics, CuPy-free.

Everything here runs on plain numpy so the statistics are testable without a
GPU. `impute.py` wraps it with file I/O, windowing and device kernels.

    aggregate_markers        cluster target markers within `cluster` cM
    allele_sequence_codes    identical allele sequences -> equal integers
    transition_tau           Li-Stephens switch probability
    forward_backward_ref     the obvious implementation; the ORACLE
    forward_backward_blocked the memory-bounded one used in production
    dose_dense               posterior -> allele probability, dense
    dose_sparse              the same, over minor-allele carriers only

METHOD AND ITS SOURCE
---------------------
Browning, Zhou & Browning (2018), "A one-penny imputed genome from next
generation reference panels", Am J Hum Genet 103(3):338-348. Quoting the
transition, because a paraphrase is where this kind of thing goes wrong:

    "If the HMM state at marker m-1 is reference haplotype h then with
     probability tau_m = 1 - e^{-4 N_e d_m / |H|} a historical recombination
     event will cause the HMM state at marker m to be randomly chosen from the
     |H| reference haplotypes."

so P((h,m-1) -> (h,m)) = (1 - tau_m) + tau_m/|H| and P(-> (h',m)) = tau_m/|H|.
d_m is in MORGANS; see _genmap.morgans().

Emission uses an error rate eps: a state emits its own allele with probability
(1 - eps) and a different allele with probability eps.

WHY THE TRANSITION COSTS O(K) AND NOT O(K^2)
---------------------------------------------
The off-diagonal tau_m/|H| does not depend on the source state, so the
transition matrix is a diagonal plus a rank-one term. The sum over source
states collapses to a single scalar per marker:

    alpha_m(k) = e_m(k) * [ (1 - tau_m) alpha_{m-1}(k) + tau_m/|H| * S_{m-1} ]

with S_{m-1} = sum_k alpha_{m-1}(k). Normalising alpha at each marker makes
S == 1 and doubles as underflow protection, which a 50,000-marker chromosome
needs: unnormalised alpha underflows float64 within a few hundred markers.

AGGREGATE MARKERS
-----------------
Target markers within `cluster` cM (0.005 by default) collapse into one
aggregate marker "whose alleles are the sequences of alleles at the constituent
markers". Its genetic position is the MEAN OF THE FIRST AND LAST constituent --
not the mean of all of them, and not the first -- and for l constituents the
mismatch probability is l*eps, "since the emission of a different allele at any
of the l constituent markers will cause a different haplotype to be emitted".

Two haplotypes therefore match at an aggregate marker only if they match at
EVERY constituent, which allele_sequence_codes() turns into an integer compare.

THE DOSE STEP IS WHERE THE TIME GOES
-------------------------------------
Sized on the paper's own chr20 fixture (K=4904, T=104, 54,885 target markers,
1,718,742 reference markers) before any of this was written:

    forward-backward      1.1e11
    dose, dense           1.7e12      15x the forward-backward
    dose, sparse          7e9 - 3e10  50-245x cheaper than dense

The dense form is a real matmul, but it multiplies the posterior by an allele
matrix that is nearly all one value: most reference haplotypes carry the major
allele at any given marker. Summing over minor-allele carriers only gives the
identical answer for a fraction of the work, which is what bref3's allele
coding exists to enable. dose_dense is kept as the oracle for dose_sparse.
"""
import numpy as np

__all__ = [
    "aggregate_markers", "allele_sequence_codes", "transition_tau",
    "forward_backward_ref", "forward_backward_blocked",
    "dose_dense", "dose_sparse", "build_carriers", "interpolation_weights",
    "default_err", "impute_haplotypes",
]


def default_err(n_ref_hap):
    """Beagle 5.5's default allele mismatch probability.

    From the 5.5 manual: "If no err parameter is specified, the err parameter
    will be set equal to theta/(2(theta+H)) where H is the number of haplotypes
    and theta = 1/(0.5 + ln H)."

    NOTE this disagrees with Browning et al. (2018), which states a flat
    "error rate eps (0.0001 by default)". At H = 4,904 this formula gives
    ~9e-4, an order of magnitude larger. The manual describes the version we
    validate against, so it wins here; pass err= explicitly for the paper's
    constant. Which one the 5.5 binary actually applies is on the list to
    settle by running it rather than by reading about it.
    """
    H = float(n_ref_hap)
    if H <= 1:
        raise ValueError(f"need at least 2 reference haplotypes, got {n_ref_hap}")
    theta = 1.0 / (0.5 + np.log(H))
    return theta / (2.0 * (theta + H))


def aggregate_markers(cm, cluster=0.005):
    """Group markers into aggregate markers of at most `cluster` cM span.

    Returns (starts, stops, agg_cm) with half-open [start, stop) index ranges
    into the marker axis and the aggregate's genetic position, which is the
    mean of the FIRST and LAST constituent positions.

    cluster <= 0 disables aggregation (one marker per aggregate).
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 1:
        raise ValueError("cm must be 1-D")
    n = cm.size
    if n == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, np.empty(0, dtype=np.float64)
    if np.any(np.diff(cm) < 0):
        raise ValueError("marker cM positions must be non-decreasing")

    if cluster is None or cluster <= 0:
        starts = np.arange(n, dtype=np.int64)
        return starts, starts + 1, cm.copy()

    starts, stops = [], []
    i = 0
    while i < n:
        j = i + 1
        while j < n and (cm[j] - cm[i]) <= cluster:
            j += 1
        starts.append(i)
        stops.append(j)
        i = j
    starts = np.asarray(starts, dtype=np.int64)
    stops = np.asarray(stops, dtype=np.int64)
    agg_cm = 0.5 * (cm[starts] + cm[stops - 1])
    return starts, stops, agg_cm


def allele_sequence_codes(ref_bits, tgt_bits, starts, stops):
    """Integer codes per aggregate marker: equal code iff identical sequence.

    ref_bits : (K, M) uint8 0/1        reference haplotypes
    tgt_bits : (T, M) uint8 0/1        target haplotypes
    returns  : (C, K) int32, (C, T) int32

    Codes are assigned jointly over reference and target so they are directly
    comparable. Reference haplotypes sharing a code are the "reference
    haplotypes that have identical allele sequences in the interval" the paper
    merges before interpolating.
    """
    ref_bits = np.asarray(ref_bits)
    tgt_bits = np.asarray(tgt_bits)
    K, M = ref_bits.shape
    T = tgt_bits.shape[0]
    if tgt_bits.shape[1] != M:
        raise ValueError(f"ref has {M} markers, target has {tgt_bits.shape[1]}")
    C = starts.size

    ref_codes = np.empty((C, K), dtype=np.int32)
    tgt_codes = np.empty((C, T), dtype=np.int32)
    for c in range(C):
        a, b = int(starts[c]), int(stops[c])
        if b - a == 1:
            # the overwhelmingly common case; skip the unique() machinery
            ref_codes[c] = ref_bits[:, a]
            tgt_codes[c] = tgt_bits[:, a]
            continue
        block = np.concatenate([ref_bits[:, a:b], tgt_bits[:, a:b]], axis=0)
        _, inv = np.unique(block, axis=0, return_inverse=True)
        inv = inv.reshape(-1).astype(np.int32)
        ref_codes[c] = inv[:K]
        tgt_codes[c] = inv[K:]
    return ref_codes, tgt_codes


def transition_tau(agg_morgans, ne, n_ref_hap):
    """tau_m = 1 - exp(-4 * Ne * d_m / |H|), with d_m in MORGANS.

    Element 0 is 0: there is no transition into the first marker, whose state
    distribution is uniform 1/|H|.
    """
    g = np.asarray(agg_morgans, dtype=np.float64)
    d = np.empty_like(g)
    d[0] = 0.0
    d[1:] = np.diff(g)
    if np.any(d < 0):
        raise ValueError("aggregate marker positions must be non-decreasing")
    tau = 1.0 - np.exp(-4.0 * float(ne) * d / float(n_ref_hap))
    tau[0] = 0.0
    return tau


def _emission(ref_codes_c, tgt_code_t, mism_c):
    """(K,) emission probabilities for one target haplotype at one aggregate."""
    match = (ref_codes_c == tgt_code_t)
    return np.where(match, 1.0 - mism_c, mism_c)


def forward_backward_ref(ref_codes, tgt_codes, tau, mism):
    """Posterior state probabilities. Stores everything; the ORACLE.

    ref_codes : (C, K) int
    tgt_codes : (C, T) int
    tau, mism : (C,) float
    returns   : (C, K, T) float64 posteriors, each column summing to 1

    Memory is C*K*T floats, which is fine for tests and hopeless for a
    chromosome -- forward_backward_blocked() is the one to use in anger. Both
    must agree to ~1e-12; test_blocked_matches_reference pins that.
    """
    C, K = ref_codes.shape
    T = tgt_codes.shape[1]
    alpha = np.empty((C, K, T), dtype=np.float64)

    a = np.full((K, T), 1.0 / K, dtype=np.float64)
    for c in range(C):
        e = (ref_codes[c][:, None] == tgt_codes[c][None, :])
        e = np.where(e, 1.0 - mism[c], mism[c])
        if c == 0:
            a = e * (1.0 / K)
        else:
            a = e * ((1.0 - tau[c]) * a + tau[c] / K)   # S == 1 after scaling
        a /= a.sum(axis=0, keepdims=True)
        alpha[c] = a

    post = np.empty_like(alpha)
    b = np.ones((K, T), dtype=np.float64)
    for c in range(C - 1, -1, -1):
        p = alpha[c] * b
        post[c] = p / p.sum(axis=0, keepdims=True)
        if c == 0:
            break
        e = (ref_codes[c][:, None] == tgt_codes[c][None, :])
        e = np.where(e, 1.0 - mism[c], mism[c])
        eb = e * b
        b = (1.0 - tau[c]) * eb + tau[c] / K * eb.sum(axis=0, keepdims=True)
        b /= b.sum(axis=0, keepdims=True)
    return post


def forward_backward_blocked(ref_codes, tgt_codes, tau, mism, block=None,
                             emit=None):
    """Memory-bounded forward-backward; posteriors delivered in marker order.

    Same result as forward_backward_ref but stores O(block*K*T + (C/block)*K*T)
    instead of O(C*K*T). With block = sqrt(C) that is O(sqrt(C)*K*T), which is
    what makes a whole chromosome fit: at C=18,295 and K=4,904 the full array is
    359 MB per target haplotype, and the blocked one is ~5 MB.

    `emit(c, posterior_KT)` is called for each aggregate marker in increasing c.
    If `emit` is None the full (C, K, T) array is returned, which defeats the
    point and exists only so tests can compare against the reference.

    The scheme: one backward sweep storing beta at block boundaries only, then
    per block a short backward replay to rebuild the block's betas followed by a
    forward sweep that pairs them with alpha. Two backward passes and one
    forward pass in exchange for sqrt-scale memory.
    """
    C, K = ref_codes.shape
    T = tgt_codes.shape[1]
    if block is None:
        block = max(1, int(np.ceil(np.sqrt(C))))
    block = int(min(max(block, 1), C)) if C else 1
    if C == 0:
        return np.empty((0, K, T), dtype=np.float64)

    def emis(c):
        e = (ref_codes[c][:, None] == tgt_codes[c][None, :])
        return np.where(e, 1.0 - mism[c], mism[c])

    def bstep(b, c):
        """beta_{c-1} from beta_c: fold in emission at c, then transition."""
        eb = emis(c) * b
        out = (1.0 - tau[c]) * eb + tau[c] / K * eb.sum(axis=0, keepdims=True)
        return out / out.sum(axis=0, keepdims=True)

    bounds = list(range(0, C, block))
    # beta at the START of each block, swept once from the end
    saved = [None] * len(bounds)
    b = np.ones((K, T), dtype=np.float64)
    for bi in range(len(bounds) - 1, -1, -1):
        lo = bounds[bi]
        hi = min(lo + block, C)
        for c in range(hi - 1, lo - 1, -1):
            if c == C - 1:
                b = np.ones((K, T), dtype=np.float64)
            else:
                b = bstep(b, c + 1)
        saved[bi] = b.copy()

    out = None if emit is not None else np.empty((C, K, T), dtype=np.float64)
    a = None
    for bi, lo in enumerate(bounds):
        hi = min(lo + block, C)
        # Rebuild this block's betas by sweeping DOWN from beta_hi, which is the
        # next block's saved boundary. beta_c cannot be stepped forward from
        # beta_{c-1} -- the recursion only runs one way -- so the saved
        # boundaries are entry points for short backward replays, not a forward
        # start state.
        betas = [None] * (hi - lo)
        bcur = saved[bi + 1] if bi + 1 < len(bounds) else \
            np.ones((K, T), dtype=np.float64)
        for c in range(hi - 1, lo - 1, -1):
            if c == C - 1:
                bcur = np.ones((K, T), dtype=np.float64)
            else:
                bcur = bstep(bcur, c + 1)
            betas[c - lo] = bcur

        for c in range(lo, hi):
            e = emis(c)
            if c == 0:
                a = e * (1.0 / K)
            else:
                a = e * ((1.0 - tau[c]) * a + tau[c] / K)
            a /= a.sum(axis=0, keepdims=True)
            p = a * betas[c - lo]
            p = p / p.sum(axis=0, keepdims=True)
            if emit is not None:
                emit(c, p)
            else:
                out[c] = p
    return out


def build_carriers(ref_bits):
    """Sparse minor-allele representation of an allele matrix.

    ref_bits : (K, M) uint8 0/1
    returns  : (indptr, indices, major) with
                 indices[indptr[m]:indptr[m+1]]  reference haplotypes carrying
                                                 the NON-major allele at m
                 major[m]                        the major allele, 0 or 1

    The allele-1 probability at marker m is then

        sum over carriers of P(k)            if major[m] == 0
        1 - sum over carriers of P(k)        if major[m] == 1

    using sum_k P(k) == 1. Storing the minority side bounds nnz at K/2 per
    marker and in practice at the minor allele count, which for a sequence
    reference panel is small for most markers.
    """
    ref_bits = np.asarray(ref_bits)
    K, M = ref_bits.shape
    ones = ref_bits.sum(axis=0).astype(np.int64)
    major = (ones * 2 > K).astype(np.uint8)          # 1 iff allele 1 is major
    counts = np.where(major == 1, K - ones, ones).astype(np.int64)
    indptr = np.zeros(M + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for m in range(M):
        want = 0 if major[m] == 1 else 1
        idx = np.flatnonzero(ref_bits[:, m] == want).astype(np.int32)
        indices[indptr[m]:indptr[m + 1]] = idx
    return indptr, indices, major


def interpolation_weights(agg_cm, marker_cm):
    """Which aggregate pair brackets each marker, and the left weight.

    Returns (left, lam) where the state probability at a marker is

        P = lam * P[left] + (1 - lam) * P[left + 1]

    "State probabilities at all imputed markers between two genotyped markers
    are obtained from the state probabilities at the bounding genotyped
    markers", by linear interpolation on GENETIC distance. Markers outside the
    aggregate range take the nearest aggregate's posterior outright (lam 1 or
    0), which is the only defined answer -- there is nothing to interpolate
    towards.
    """
    agg_cm = np.asarray(agg_cm, dtype=np.float64)
    marker_cm = np.asarray(marker_cm, dtype=np.float64)
    C = agg_cm.size
    if C == 0:
        raise ValueError("no aggregate markers")
    if C == 1:
        return (np.zeros(marker_cm.size, dtype=np.int64),
                np.ones(marker_cm.size, dtype=np.float64))

    right = np.searchsorted(agg_cm, marker_cm, side="left")
    left = np.clip(right - 1, 0, C - 2)
    gl = agg_cm[left]
    gr = agg_cm[left + 1]
    span = gr - gl
    lam = np.where(span > 0, (gr - marker_cm) / np.where(span > 0, span, 1.0), 1.0)
    return left, np.clip(lam, 0.0, 1.0)


def dose_dense(post_left, post_right, lam, ref_bits_block):
    """Allele-1 probability by dense contraction. The ORACLE for dose_sparse.

    post_left, post_right : (K, T)
    lam                   : (n,) left weights for n markers
    ref_bits_block        : (K, n) uint8
    returns               : (T, n)
    """
    A = np.asarray(ref_bits_block, dtype=np.float64)
    gl = post_left.T @ A          # (T, n)
    gr = post_right.T @ A
    return lam[None, :] * gl + (1.0 - lam[None, :]) * gr


def dose_sparse(post_left, post_right, lam, indptr, indices, major, cols):
    """Allele-1 probability by summing over minor-allele carriers only.

    Identical result to dose_dense for a fraction of the work: the dense form
    contracts over all K states when at most min(MAC, K-MAC) of them contribute
    anything the major-allele complement does not already account for.
    """
    T = post_left.shape[1]
    n = len(cols)
    out = np.empty((T, n), dtype=np.float64)
    for j, m in enumerate(cols):
        s, e = int(indptr[m]), int(indptr[m + 1])
        idx = indices[s:e]
        if idx.size:
            sl = post_left[idx].sum(axis=0)
            sr = post_right[idx].sum(axis=0)
        else:
            sl = np.zeros(T, dtype=np.float64)
            sr = np.zeros(T, dtype=np.float64)
        if major[m] == 1:
            sl = 1.0 - sl
            sr = 1.0 - sr
        out[:, j] = lam[j] * sl + (1.0 - lam[j]) * sr
    return out


def impute_haplotypes(ref_bits, tgt_bits, tgt_idx, marker_cm, *, ne=100_000,
                      err=None, cluster=0.005, block=None, sparse=True,
                      carriers=None, timers=None):
    """End-to-end allele-1 probabilities for every reference marker.

    ref_bits  : (K, M_ref) uint8 0/1   phased reference panel
    tgt_bits  : (T, M_tgt) uint8 0/1   phased target haplotypes
    tgt_idx   : (M_tgt,) int           each target marker's index in ref
    marker_cm : (M_ref,) float         genetic position of every ref marker
    returns   : (T, M_ref) float64     P(allele 1) on each target haplotype

    Posteriors are consumed as they are produced, so the (C, K, T) array is
    never materialised: only two aggregate markers' worth are live at a time,
    which is what lets a chromosome run in bounded memory.

    `timers` is an optional dict that accumulates per-phase seconds. Phase
    costs here differ by more than an order of magnitude, and the previous
    round of this project spent three optimisation passes on a phase that
    turned out to be 0.9% of runtime -- so the instrumentation is not optional
    scaffolding to be added later.
    """
    import time
    ref_bits = np.asarray(ref_bits)
    tgt_bits = np.asarray(tgt_bits)
    tgt_idx = np.asarray(tgt_idx, dtype=np.int64)
    marker_cm = np.asarray(marker_cm, dtype=np.float64)
    K, M_ref = ref_bits.shape
    T, M_tgt = tgt_bits.shape
    if tgt_idx.size != M_tgt:
        raise ValueError(f"{tgt_idx.size} target indices but {M_tgt} target markers")
    if marker_cm.size != M_ref:
        raise ValueError(f"{marker_cm.size} cM values but {M_ref} reference markers")
    if np.any(np.diff(tgt_idx) <= 0):
        raise ValueError("tgt_idx must be strictly increasing")
    if err is None:
        err = default_err(K)

    t = {} if timers is None else timers
    def tick(name, t0):
        t[name] = t.get(name, 0.0) + (time.perf_counter() - t0)

    t0 = time.perf_counter()
    tgt_cm = marker_cm[tgt_idx]
    starts, stops, agg_cm = aggregate_markers(tgt_cm, cluster)
    ref_codes, tgt_codes = allele_sequence_codes(
        ref_bits[:, tgt_idx], tgt_bits, starts, stops)
    C = starts.size
    tau = transition_tau(agg_cm / 100.0, ne, K)          # cM -> Morgans
    mism = np.minimum((stops - starts).astype(np.float64) * err, 0.5)
    tick("aggregate", t0)

    t0 = time.perf_counter()
    left, lam = interpolation_weights(agg_cm, marker_cm)
    order = np.argsort(left, kind="stable")
    group_start = np.searchsorted(left[order], np.arange(C), side="left")
    group_stop = np.searchsorted(left[order], np.arange(C), side="right")
    tick("interp_plan", t0)

    if sparse and carriers is None:
        t0 = time.perf_counter()
        carriers = build_carriers(ref_bits)
        tick("carriers", t0)

    out = np.zeros((T, M_ref), dtype=np.float64)
    state = {"prev": None, "prev_c": -1}

    def flush(c_left, post_l, post_r):
        cols = order[group_start[c_left]:group_stop[c_left]]
        if cols.size == 0:
            return
        t0 = time.perf_counter()
        if sparse:
            indptr, indices, major = carriers
            out[:, cols] = dose_sparse(post_l, post_r, lam[cols],
                                       indptr, indices, major, cols)
        else:
            out[:, cols] = dose_dense(post_l, post_r, lam[cols],
                                      ref_bits[:, cols])
        tick("dose", t0)

    fb0 = time.perf_counter()

    def emit(c, post):
        if state["prev"] is not None:
            flush(state["prev_c"], state["prev"], post)
        state["prev"] = post.copy()
        state["prev_c"] = c

    forward_backward_blocked(ref_codes, tgt_codes, tau, mism,
                             block=block, emit=emit)
    # Markers past the last aggregate interpolate against nothing; their left
    # index is C-2 (clipped) and they were flushed above, except when C == 1,
    # where every marker takes the single posterior outright.
    if C == 1 and state["prev"] is not None:
        flush(0, state["prev"], state["prev"])
    t["forward_backward"] = time.perf_counter() - fb0 - t.get("dose", 0.0)
    return out
