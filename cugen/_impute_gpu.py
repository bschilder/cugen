"""cugen._impute_gpu - CUDA kernels for Li-Stephens imputation.

    fb_posteriors_gpu   forward-backward, posteriors in (C, K, T) layout
    dose_sparse_gpu     posteriors -> allele probabilities, minor-allele carriers
    impute_haplotypes_gpu   the same contract as the numpy impute_haplotypes

Every kernel here has a numpy counterpart in _impute_core.py that it must agree
with to ~1e-5; tests/test_impute_gpu.py pins that. The numpy versions are the
specification, not a fallback.

LAYOUT: POSTERIORS ARE (C, K, T)
---------------------------------
Chosen from the DOSE step's access pattern, because the dose step is 15x the
forward-backward and therefore decides. It reads post[c, k, t] for a fixed
(c, k) across many t, summing over the reference haplotypes that carry a
marker's minor allele. Under (T, C, K) adjacent t are C*K floats apart -- about
30 million at chr20 scale -- so every warp lane touches a different cache line.
Under (C, K, T) they are adjacent and the read coalesces.

The forward-backward pays for this: it wants k contiguous for its reduction and
gets stride T instead. That is the right trade at a 15:1 cost ratio, and it is
worth stating explicitly, because the layout looks backwards if you only read
the forward-backward kernel.

MEMORY
------
The posterior array is C * K * T floats and is the binding constraint: 11.6 GiB
for one 40 cM window of the paper's chr20 fixture. The T tile is therefore
SIZED FROM THE ACTUAL DIMENSIONS AND FREE DEVICE MEMORY, never from a constant.
Three separate bugs in the LD work in this repo were a tile constant that
ignored a dimension it multiplies -- 8,192 rows regardless of sample count
allocating a 16.4 GB plane, three full-size planes built to produce two
per-variant scalars, candidate tiles grouped by count while their windows
spanned positions. The pattern is the same every time and it does not announce
itself; it just runs out of memory on data one size larger than the test.

ASCII ONLY
----------
Kernel source must contain no non-ASCII character. Commit 34b4a59 in this repo
was an NVRTC crash caused by one in a comment, and it cannot be reproduced on a
machine without CUDA -- which is where this file was written. The assertion
below runs at import.
"""
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:                                            # noqa: BLE001
    cp = None
    HAS_CUPY = False

__all__ = ["fb_posteriors_gpu", "dose_sparse_gpu", "impute_haplotypes_gpu",
           "build_carriers_gpu", "plan_t_tile", "IMPUTE_SRC"]

_MODULE = None

# Threads per block: TX haplotypes wide (contiguous t, so loads coalesce) by TY
# state-workers deep (the axis the per-marker reduction runs over).
_TX = 32
_TY = 8

IMPUTE_SRC = r'''
extern "C" {

// Emission probability of state k against target t at aggregate marker c.
__device__ __forceinline__ float emit(int rc, int tc, float mism) {
    return (rc == tc) ? (1.0f - mism) : mism;
}

// ---------------------------------------------------------------------------
// Backward sweep. Writes beta for every aggregate marker into `post`, which the
// forward kernel then overwrites in place with the posterior.
//
// beta_{c-1}(k) = (1 - tau_c) * eb_c(k) + tau_c/K * sum_k eb_c(k),
//     eb_c(k) = e_c(k) * beta_c(k)
//
// Two passes over k per marker: one computing eb (stored back into post) and
// accumulating the partial sums, one applying the transition. The second pass
// re-reads eb rather than holding K/TY values per thread in registers -- 154 of
// them at chr20 scale, which would spill.
// ---------------------------------------------------------------------------
__global__ void fb_backward(
    const int* __restrict__ ref_codes,     // (C, K)
    const int* __restrict__ tgt_codes,     // (C, T)
    const float* __restrict__ tau,         // (C,)
    const float* __restrict__ mism,        // (C,)
    float* __restrict__ post,              // (C, K, T)
    const int C, const int K, const int T)
{
    extern __shared__ float sh[];          // blockDim.x * blockDim.y partials
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int t = blockIdx.x * blockDim.x + tx;
    const int lane = ty * blockDim.x + tx;
    const bool live = (t < T);

    // beta at the last marker is uniformly 1
    for (int k = ty; k < K; k += blockDim.y) {
        if (live) post[((long long)(C - 1) * K + k) * T + t] = 1.0f;
    }
    __syncthreads();

    for (int c = C - 1; c > 0; --c) {
        const float mm = mism[c];
        const int tc = live ? tgt_codes[(long long)c * T + t] : 0;
        float partial = 0.0f;

        // pass 1: accumulate sum_k eb WITHOUT storing eb. An earlier version
        // wrote eb back into post[c], which destroys beta_c -- and the forward
        // kernel reads post[c] expecting beta_c, so the posterior would have
        // come out as alpha * e * beta, an extra emission factor at every
        // marker. Recomputing e in pass 2 is one integer compare and costs less
        // than the write plus re-read it replaces.
        for (int k = ty; k < K; k += blockDim.y) {
            const long long o = ((long long)c * K + k) * T + t;
            if (live) {
                partial += emit(ref_codes[(long long)c * K + k], tc, mm)
                           * post[o];
            }
        }
        sh[lane] = partial;
        __syncthreads();
        // reduce down the ty axis; tx (the haplotype axis) stays independent
        for (int s = blockDim.y / 2; s > 0; s >>= 1) {
            if (ty < s) sh[lane] += sh[(ty + s) * blockDim.x + tx];
            __syncthreads();
        }
        const float S = sh[tx];
        __syncthreads();

        // pass 2: apply the transition into marker c-1, and renormalise.
        // sum_k[(1-tau)eb(k) + tau*S/K] = (1-tau)S + tau*S = S exactly, so the
        // normaliser is the same S already reduced -- no second reduction.
        const float keep = 1.0f - tau[c];
        const float jump = (S > 0.0f) ? (tau[c] * S / (float)K) : 0.0f;
        const float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
        for (int k = ty; k < K; k += blockDim.y) {
            const long long o = ((long long)c * K + k) * T + t;
            if (live) {
                const float eb = emit(ref_codes[(long long)c * K + k], tc, mm)
                                 * post[o];
                post[((long long)(c - 1) * K + k) * T + t] =
                    (keep * eb + jump) * inv;
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Forward sweep. Reads beta from `post` and overwrites it with the normalised
// posterior alpha*beta. `alpha` is a (K, T) scratch buffer carried marker to
// marker.
// ---------------------------------------------------------------------------
__global__ void fb_forward(
    const int* __restrict__ ref_codes,
    const int* __restrict__ tgt_codes,
    const float* __restrict__ tau,
    const float* __restrict__ mism,
    float* __restrict__ post,              // (C, K, T) beta in, posterior out
    float* __restrict__ alpha,             // (K, T) scratch
    const int C, const int K, const int T)
{
    extern __shared__ float sh[];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int t = blockIdx.x * blockDim.x + tx;
    const int lane = ty * blockDim.x + tx;
    const bool live = (t < T);

    for (int c = 0; c < C; ++c) {
        const float mm = mism[c];
        const int tc = live ? tgt_codes[(long long)c * T + t] : 0;
        const float keep = 1.0f - tau[c];
        const float jump = tau[c] / (float)K;

        float partial = 0.0f;
        for (int k = ty; k < K; k += blockDim.y) {
            const float e = emit(ref_codes[(long long)c * K + k], tc, mm);
            float a;
            if (c == 0) {
                a = e / (float)K;
            } else {
                a = e * (keep * alpha[(long long)k * T + t] + jump);
            }
            if (live) alpha[(long long)k * T + t] = a;
            partial += a;
        }
        sh[lane] = partial;
        __syncthreads();
        for (int s = blockDim.y / 2; s > 0; s >>= 1) {
            if (ty < s) sh[lane] += sh[(ty + s) * blockDim.x + tx];
            __syncthreads();
        }
        const float SA = sh[tx];
        __syncthreads();

        // normalise alpha, fold in beta, and accumulate the posterior norm
        float pp = 0.0f;
        const float inv = (SA > 0.0f) ? (1.0f / SA) : 0.0f;
        for (int k = ty; k < K; k += blockDim.y) {
            const long long ka = (long long)k * T + t;
            const long long o = ((long long)c * K + k) * T + t;
            if (live) {
                const float a = alpha[ka] * inv;
                alpha[ka] = a;
                const float p = a * post[o];
                post[o] = p;
                pp += p;
            }
        }
        sh[lane] = pp;
        __syncthreads();
        for (int s = blockDim.y / 2; s > 0; s >>= 1) {
            if (ty < s) sh[lane] += sh[(ty + s) * blockDim.x + tx];
            __syncthreads();
        }
        const float SP = sh[tx];
        __syncthreads();

        const float invp = (SP > 0.0f) ? (1.0f / SP) : 0.0f;
        for (int k = ty; k < K; k += blockDim.y) {
            const long long o = ((long long)c * K + k) * T + t;
            if (live) post[o] *= invp;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Allele probabilities, summing over minor-allele carriers only.
//
//   P(allele 1) = lam * S(post[left]) + (1 - lam) * S(post[left+1])
//   S(p)        = sum over carriers, complemented when allele 1 is the major
//
// One block per marker, threads over target haplotypes. Every lane walks the
// SAME carrier list, so indices[] is broadcast, and post[...] + t is contiguous
// across the warp. That is the whole reason for the (C, K, T) layout.
// ---------------------------------------------------------------------------
__global__ void dose_sparse(
    const float* __restrict__ post,        // (C, K, T)
    const long long* __restrict__ indptr,  // (M + 1,)
    const int* __restrict__ indices,       // (nnz,)
    const unsigned char* __restrict__ major,
    const int* __restrict__ left,          // (M,) bracketing aggregate index
    const float* __restrict__ lam,         // (M,)
    float* __restrict__ out,               // (T, M)
    const int C, const int K, const int T, const int M)
{
    const int m = blockIdx.x;
    if (m >= M) return;
    const long long s = indptr[m];
    const long long e = indptr[m + 1];
    const int cl = left[m];
    const int cr = (cl + 1 < C) ? (cl + 1) : cl;
    const float w = lam[m];

    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float sl = 0.0f;
        float sr = 0.0f;
        for (long long j = s; j < e; ++j) {
            const int k = indices[j];
            sl += post[((long long)cl * K + k) * T + t];
            sr += post[((long long)cr * K + k) * T + t];
        }
        if (major[m] == (unsigned char)1) {
            sl = 1.0f - sl;
            sr = 1.0f - sr;
        }
        out[(long long)t * M + m] = w * sl + (1.0f - w) * sr;
    }
}

// ---------------------------------------------------------------------------
// Minority-allele carrier lists, straight from the PACKED reference bytes.
//
// This replaced two host phases that between them were 49.2s of a 76.7s run and
// depend only on the reference panel -- building the carrier lists (30.7s) and
// unpacking the whole window to (K, M) bytes just to scan it (18.5s). Neither
// touches target data, so neither belonged on the critical path at all.
//
// One thread per marker, walking that marker's bytes serially. Adjacent threads
// are bytes_per_variant apart so the reads do not coalesce, which is worth it:
// the alternative orderings either need a block-wide scan to keep indices
// sorted, or drop the ordering entirely. Sorted output keeps this
// bit-comparable against the numpy build_carriers.
// ---------------------------------------------------------------------------
__global__ void carrier_counts(
    const unsigned char* __restrict__ packed,   // (M, bpv)
    long long* __restrict__ ones,               // (M,) out: count of allele 1
    const int M, const int n_hap, const int bpv)
{
    const long long m = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    const unsigned char* row = packed + m * (long long)bpv;
    int c = 0;
    const int full = n_hap >> 3;
    for (int b = 0; b < full; ++b) c += __popc((unsigned int)row[b]);
    const int rem = n_hap & 7;               // trailing bits are padding zeros,
    if (rem) {                               // but mask anyway rather than trust
        const unsigned char mask = (unsigned char)(0xFF << (8 - rem));
        c += __popc((unsigned int)(row[full] & mask));
    }
    ones[m] = c;
}

__global__ void carrier_scatter(
    const unsigned char* __restrict__ packed,   // (M, bpv)
    const long long* __restrict__ indptr,       // (M+1,)
    const unsigned char* __restrict__ major,    // (M,)
    int* __restrict__ indices,                  // (nnz,) out
    const int M, const int n_hap, const int bpv)
{
    const long long m = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    const unsigned char* row = packed + m * (long long)bpv;
    const int want = (major[m] == (unsigned char)1) ? 0 : 1;
    long long w = indptr[m];
    const long long end = indptr[m + 1];
    for (int j = 0; j < n_hap && w < end; ++j) {
        const int bit = (row[j >> 3] >> (7 - (j & 7))) & 1;
        if (bit == want) indices[w++] = j;
    }
}

}   // extern "C"
'''

# NVRTC dies on a non-ASCII byte anywhere in the source, including comments, and
# the failure is invisible on a machine without CUDA. Commit 34b4a59 was exactly
# this. Assert at import so it cannot reach a pod.
assert IMPUTE_SRC.isascii(), "kernel source must be pure ASCII (see 34b4a59)"


def _module():
    global _MODULE
    if not HAS_CUPY:
        raise RuntimeError("CuPy is not available; cugen.impute GPU path needs it")
    if _MODULE is None:
        _MODULE = cp.RawModule(code=IMPUTE_SRC, options=("-std=c++11",))
    return _MODULE


def plan_t_tile(C, K, T, budget_bytes=None, safety=0.70):
    """How many target haplotypes fit in one posterior array.

    Sized from C, K and free device memory. A constant here would be the same
    bug that cost the LD work three separate fixes: a tile dimensioned without
    reference to a dimension it multiplies.
    """
    if budget_bytes is None:
        if HAS_CUPY:
            free, _total = cp.cuda.Device().mem_info
            budget_bytes = int(free * safety)
        else:
            budget_bytes = 8 << 30
    per_hap = C * K * 4 + K * 4          # posterior slab plus alpha scratch
    tile = int(budget_bytes // max(per_hap, 1))
    if tile < 1:
        raise MemoryError(
            f"a single target haplotype needs {per_hap / 2**30:.2f} GiB at "
            f"C={C:,} aggregate markers and K={K:,} states, over the "
            f"{budget_bytes / 2**30:.2f} GiB budget. Reduce `window` (fewer "
            f"aggregate markers per window) or the reference panel size.")
    return min(tile, int(T))


def fb_posteriors_gpu(ref_codes, tgt_codes, tau, mism):
    """Posteriors as a (C, K, T) CuPy array. Mirrors forward_backward_ref."""
    mod = _module()
    C, K = ref_codes.shape
    T = tgt_codes.shape[1]
    rc = cp.ascontiguousarray(cp.asarray(ref_codes, dtype=cp.int32))
    tc = cp.ascontiguousarray(cp.asarray(tgt_codes, dtype=cp.int32))
    ta = cp.ascontiguousarray(cp.asarray(tau, dtype=cp.float32))
    mm = cp.ascontiguousarray(cp.asarray(mism, dtype=cp.float32))
    post = cp.empty((C, K, T), dtype=cp.float32)
    alpha = cp.zeros((K, T), dtype=cp.float32)

    grid = ((T + _TX - 1) // _TX,)
    blk = (_TX, _TY)
    shmem = _TX * _TY * 4
    mod.get_function("fb_backward")(
        grid, blk, (rc, tc, ta, mm, post, np.int32(C), np.int32(K),
                    np.int32(T)), shared_mem=shmem)
    mod.get_function("fb_forward")(
        grid, blk, (rc, tc, ta, mm, post, alpha, np.int32(C), np.int32(K),
                    np.int32(T)), shared_mem=shmem)
    return post


def dose_sparse_gpu(post, indptr, indices, major, left, lam):
    """Allele-1 probabilities as a (T, M) CuPy array. Mirrors dose_sparse."""
    mod = _module()
    C, K, T = post.shape
    M = int(len(lam))
    ip = cp.ascontiguousarray(cp.asarray(indptr, dtype=cp.int64))
    ix = cp.ascontiguousarray(cp.asarray(indices, dtype=cp.int32))
    mj = cp.ascontiguousarray(cp.asarray(major, dtype=cp.uint8))
    lf = cp.ascontiguousarray(cp.asarray(left, dtype=cp.int32))
    lm = cp.ascontiguousarray(cp.asarray(lam, dtype=cp.float32))
    out = cp.empty((T, M), dtype=cp.float32)
    threads = min(256, max(32, ((T + 31) // 32) * 32))
    mod.get_function("dose_sparse")(
        (M,), (threads,), (post, ip, ix, mj, lf, lm, out,
                           np.int32(C), np.int32(K), np.int32(T), np.int32(M)))
    return out


def build_carriers_gpu(packed, n_hap, n_markers, bytes_per_variant):
    """Carrier lists from PACKED reference bytes, on the device.

    packed : (n_markers, bytes_per_variant) uint8, exactly as stored on disk
    returns: (indptr, indices, major) matching _impute_core.build_carriers

    Takes the packed bytes rather than an unpacked (K, M) matrix, which is the
    point: the host never has to expand the window at all. Together with
    computing the counts here instead of in a Python loop, this removes the two
    phases that were 49.2s of a 76.7s run and that depend only on the reference
    panel.
    """
    mod = _module()
    pk = cp.ascontiguousarray(cp.asarray(packed, dtype=cp.uint8))
    M = int(n_markers)
    ones = cp.empty(M, dtype=cp.int64)
    threads = 256
    blocks = (M + threads - 1) // threads
    mod.get_function("carrier_counts")(
        (blocks,), (threads,), (pk, ones, np.int32(M), np.int32(n_hap),
                                np.int32(bytes_per_variant)))
    major = ((ones * 2) > n_hap).astype(cp.uint8)
    counts = cp.where(major == 1, n_hap - ones, ones)
    indptr = cp.zeros(M + 1, dtype=cp.int64)
    cp.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1].get())
    indices = cp.empty(max(nnz, 1), dtype=cp.int32)
    mod.get_function("carrier_scatter")(
        (blocks,), (threads,), (pk, indptr, major, indices, np.int32(M),
                                np.int32(n_hap), np.int32(bytes_per_variant)))
    return indptr, indices[:nnz], major


def impute_haplotypes_gpu(ref_bits, tgt_bits, tgt_idx, marker_cm, *,
                          ne=100_000, err=None, cluster=0.005, carriers=None,
                          timers=None, budget_bytes=None, ref_packed=None,
                          n_hap=None, bytes_per_variant=None):
    """GPU counterpart of _impute_core.impute_haplotypes. Returns (T, M) float32.

    Pass `ref_packed` (the on-disk bytes, with n_hap and bytes_per_variant) to
    take the fast path: carriers are built on the device straight from the
    packed form, and only the genotyped columns are ever unpacked on the host.
    `ref_bits` may then be None. That skips expanding the window to a (K, M)
    byte matrix, which for one 40 cM window of chr20 is 2.4 GiB the host reads,
    writes and then scans -- and which nothing except the carrier build needed.
    """
    import time

    from ._impute_core import (aggregate_markers, aggregate_mismatch,
                               allele_sequence_codes, build_carriers,
                               default_err, interpolation_weights,
                               transition_tau)

    t = {} if timers is None else timers

    def tick(name, t0):
        t[name] = t.get(name, 0.0) + (time.perf_counter() - t0)

    tgt_bits = np.asarray(tgt_bits)
    tgt_idx = np.asarray(tgt_idx, dtype=np.int64)
    marker_cm = np.asarray(marker_cm, dtype=np.float64)
    T = tgt_bits.shape[0]

    packed_path = ref_packed is not None
    if packed_path:
        if n_hap is None or bytes_per_variant is None:
            raise ValueError("ref_packed needs n_hap and bytes_per_variant")
        ref_packed = np.ascontiguousarray(
            np.asarray(ref_packed, dtype=np.uint8).reshape(-1, bytes_per_variant))
        K, M = int(n_hap), int(ref_packed.shape[0])
    else:
        ref_bits = np.asarray(ref_bits)
        K, M = ref_bits.shape
    if err is None:
        err = default_err(K)

    t0 = time.perf_counter()
    starts, stops, agg_cm = aggregate_markers(marker_cm[tgt_idx], cluster)
    if packed_path:
        # Unpack ONLY the genotyped columns -- tens of thousands, against the
        # window's hundreds of thousands.
        sub = np.unpackbits(ref_packed[tgt_idx], axis=1).T[:K, :]
    else:
        sub = ref_bits[:, tgt_idx]
    ref_codes, tgt_codes = allele_sequence_codes(
        sub, tgt_bits, starts, stops)
    tau = transition_tau(agg_cm / 100.0, ne, K)
    mism = aggregate_mismatch(starts, stops, err)
    C = starts.size
    tick("aggregate", t0)

    t0 = time.perf_counter()
    left, lam = interpolation_weights(agg_cm, marker_cm)
    tick("interp_plan", t0)

    if carriers is None:
        t0 = time.perf_counter()
        if packed_path:
            carriers = build_carriers_gpu(ref_packed, K, M, bytes_per_variant)
            cp.cuda.Device().synchronize()
        else:
            carriers = build_carriers(ref_bits)
        tick("carriers", t0)
    indptr, indices, major = carriers

    tile = plan_t_tile(C, K, T, budget_bytes=budget_bytes)
    out = np.empty((T, M), dtype=np.float32)
    for lo in range(0, T, tile):
        hi = min(lo + tile, T)
        t0 = time.perf_counter()
        post = fb_posteriors_gpu(ref_codes, tgt_codes[:, lo:hi], tau, mism)
        cp.cuda.Device().synchronize()
        tick("forward_backward", t0)

        t0 = time.perf_counter()
        d = dose_sparse_gpu(post, indptr, indices, major, left, lam)
        cp.cuda.Device().synchronize()
        tick("dose", t0)
        out[lo:hi] = cp.asnumpy(d)
        del post, d
        cp.get_default_memory_pool().free_all_blocks()
    # NOT in the seconds dict: it is a count, and it printed as
    # "t_tile 104.00s" at the top of the phase table, which reads
    # exactly like the dominant cost.
    t["_t_tile_count"] = tile
    return out
