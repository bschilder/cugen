"""Kernel tests for cugen.impute.

Two tiers:

  * The `emulate_*` tests transcribe each kernel's exact control flow into numpy
    and compare against _impute_core. They run on CPU CI, with no GPU and no
    compiler, and they catch the algorithm being wrong -- which is most of what
    a kernel written on a machine without CUDA gets wrong. One real bug was
    found this way before any pod existed: the backward kernel wrote eb back
    into post[c], destroying beta_c, so the posterior would have carried an
    extra emission factor at every marker.

  * The `@requires_gpu` tests run the real kernels and require agreement with
    the numpy reference. Only these can catch an NVRTC failure or a race.
"""
import numpy as np
import pytest

from cugen._impute_core import (aggregate_mismatch, build_carriers, dose_sparse,
                                forward_backward_ref, impute_haplotypes,
                                interpolation_weights)
from cugen._impute_gpu import IMPUTE_SRC, plan_t_tile

from conftest import requires_gpu, scaled_ne, simulate_mosaic


# --------------------------------------------------------------------------
# Source-level invariants -- these run everywhere, including macOS
# --------------------------------------------------------------------------

def test_kernel_source_is_ascii():
    """NVRTC dies on a non-ASCII byte anywhere in the source, comments
    included, and it cannot be reproduced without CUDA. Commit 34b4a59 in this
    repo was exactly that."""
    assert IMPUTE_SRC.isascii()
    bad = [(i, c) for i, c in enumerate(IMPUTE_SRC) if ord(c) > 127]
    assert not bad, f"non-ASCII at {bad[:3]}"


def test_kernels_are_declared_extern_c():
    for name in ("fb_backward", "fb_forward", "dose_sparse"):
        assert f"void {name}(" in IMPUTE_SRC
    assert 'extern "C"' in IMPUTE_SRC


def test_t_tile_is_sized_from_the_data_not_a_constant():
    """A tile constant that ignores a dimension it multiplies was three
    separate bugs in the LD work here. Doubling either C or K must halve the
    tile; a constant would not move."""
    base = plan_t_tile(1000, 1000, 10_000, budget_bytes=8 << 30)
    assert plan_t_tile(2000, 1000, 10_000, budget_bytes=8 << 30) == pytest.approx(base / 2, rel=0.02)
    assert plan_t_tile(1000, 2000, 10_000, budget_bytes=8 << 30) == pytest.approx(base / 2, rel=0.02)
    assert plan_t_tile(1000, 1000, 10_000, budget_bytes=16 << 30) == pytest.approx(base * 2, rel=0.02)


def test_t_tile_never_exceeds_the_haplotype_count():
    assert plan_t_tile(10, 10, 7, budget_bytes=8 << 30) == 7


def test_t_tile_raises_rather_than_returning_zero():
    with pytest.raises(MemoryError, match="single target haplotype"):
        plan_t_tile(10_000_000, 10_000, 4, budget_bytes=1 << 20)


# --------------------------------------------------------------------------
# Kernel logic, emulated on CPU
# --------------------------------------------------------------------------

def emulate_fb(ref_codes, tgt_codes, tau, mism):
    """Follows fb_backward and fb_forward statement for statement."""
    C, K = ref_codes.shape
    T = tgt_codes.shape[1]
    post = np.empty((C, K, T))

    def emis(c):
        return np.where(ref_codes[c][:, None] == tgt_codes[c][None, :],
                        1 - mism[c], mism[c])

    post[C - 1] = 1.0                                    # fb_backward
    for c in range(C - 1, 0, -1):
        e = emis(c)
        S = (e * post[c]).sum(axis=0)                    # pass 1: reduce only
        keep = 1.0 - tau[c]
        jump = tau[c] * S / K
        inv = np.where(S > 0, 1.0 / np.where(S > 0, S, 1.0), 0.0)
        eb = e * post[c]                                 # pass 2: recompute
        post[c - 1] = (keep * eb + jump[None, :]) * inv[None, :]

    alpha = np.zeros((K, T))                             # fb_forward
    for c in range(C):
        e = emis(c)
        a = e / K if c == 0 else e * ((1 - tau[c]) * alpha + tau[c] / K)
        SA = a.sum(axis=0)
        a = a / np.where(SA > 0, SA, 1.0)
        alpha = a
        p = a * post[c]
        SP = p.sum(axis=0)
        post[c] = p / np.where(SP > 0, SP, 1.0)
    return post


@pytest.mark.parametrize("seed", range(6))
def test_fb_kernel_logic_matches_reference(seed):
    rng = np.random.default_rng(31 + seed)
    K = int(rng.integers(5, 40)); T = int(rng.integers(1, 9))
    C = int(rng.integers(3, 30))
    rc = rng.integers(0, 3, size=(C, K)).astype(np.int32)
    tc = rng.integers(0, 3, size=(C, T)).astype(np.int32)
    tau = np.concatenate([[0.0], rng.uniform(1e-4, 0.3, size=C - 1)])
    mism = np.full(C, float(rng.uniform(1e-4, 0.05)))
    assert np.abs(forward_backward_ref(rc, tc, tau, mism)
                  - emulate_fb(rc, tc, tau, mism)).max() < 1e-12


def test_backward_kernel_preserves_beta_for_the_forward_pass():
    """Regression guard for a real bug.

    The first version stored eb back into post[c] to avoid recomputing the
    emission. But the forward kernel reads post[c] expecting beta_c, so the
    posterior came out as alpha * e * beta. Nothing crashes; the numbers are
    merely wrong, and wrong in a direction that still looks like a probability.
    """
    rng = np.random.default_rng(3)
    C, K, T = 8, 6, 2
    rc = rng.integers(0, 2, size=(C, K)).astype(np.int32)
    tc = rng.integers(0, 2, size=(C, T)).astype(np.int32)
    tau = np.concatenate([[0.0], np.full(C - 1, 0.05)])
    mism = np.full(C, 0.01)
    good = emulate_fb(rc, tc, tau, mism)

    def emulate_broken():
        post = np.empty((C, K, T)); post[C - 1] = 1.0
        for c in range(C - 1, 0, -1):
            e = np.where(rc[c][:, None] == tc[c][None, :], 1 - mism[c], mism[c])
            eb = e * post[c]
            post[c] = eb                                  # the bug
            S = eb.sum(axis=0)
            post[c - 1] = ((1 - tau[c]) * eb + tau[c] * S / K) / S
        alpha = np.zeros((K, T))
        for c in range(C):
            e = np.where(rc[c][:, None] == tc[c][None, :], 1 - mism[c], mism[c])
            a = e / K if c == 0 else e * ((1 - tau[c]) * alpha + tau[c] / K)
            a /= a.sum(axis=0); alpha = a
            p = a * post[c]; post[c] = p / p.sum(axis=0)
        return post

    assert not np.allclose(good, emulate_broken()), \
        "the fixture cannot distinguish the bug from the fix"
    assert np.abs(forward_backward_ref(rc, tc, tau, mism) - good).max() < 1e-12


def test_dose_kernel_logic_matches_reference():
    rng = np.random.default_rng(12)
    K, T, M, C = 30, 5, 50, 6
    post = rng.random((C, K, T)); post /= post.sum(axis=1, keepdims=True)
    bits = (rng.random((K, M)) < rng.uniform(0.02, 0.98, size=M)).astype(np.uint8)
    indptr, indices, major = build_carriers(bits)
    left, lam = interpolation_weights(np.sort(rng.random(C)), np.sort(rng.random(M)))
    got = np.empty((T, M))
    for m in range(M):                                    # one block per marker
        s, e = indptr[m], indptr[m + 1]
        cl = int(left[m]); cr = min(cl + 1, C - 1)
        for t in range(T):                                # threads over haps
            sl = post[cl, indices[s:e], t].sum()
            sr = post[cr, indices[s:e], t].sum()
            if major[m] == 1:
                sl, sr = 1 - sl, 1 - sr
            got[t, m] = lam[m] * sl + (1 - lam[m]) * sr
    exp = np.stack([dose_sparse(post[int(left[m])], post[min(int(left[m]) + 1, C - 1)],
                                np.array([lam[m]]), indptr, indices, major, [m])[:, 0]
                    for m in range(M)], axis=1)
    assert np.abs(got - exp).max() < 1e-12


# --------------------------------------------------------------------------
# The real kernels
# --------------------------------------------------------------------------

@requires_gpu
def test_kernels_compile():
    """The one failure class invisible on macOS."""
    from cugen._impute_gpu import _module
    mod = _module()
    for name in ("fb_backward", "fb_forward", "dose_sparse"):
        assert mod.get_function(name) is not None


@requires_gpu
@pytest.mark.parametrize("C,K,T", [(3, 5, 1), (12, 33, 7), (40, 64, 32),
                                   (25, 100, 33), (17, 7, 64)])
def test_gpu_forward_backward_matches_numpy(C, K, T):
    """Odd K and T on purpose: the kernel strides k by blockDim.y and t by
    blockDim.x, so sizes that are not multiples of 8 and 32 are where an
    out-of-bounds lane or a missed tail iteration shows up."""
    import cupy as cp
    from cugen._impute_gpu import fb_posteriors_gpu
    rng = np.random.default_rng(C * 1000 + K * 10 + T)
    rc = rng.integers(0, 3, size=(C, K)).astype(np.int32)
    tc = rng.integers(0, 3, size=(C, T)).astype(np.int32)
    tau = np.concatenate([[0.0], rng.uniform(1e-4, 0.3, size=C - 1)]).astype(np.float64)
    mism = np.full(C, 1e-3)
    exp = forward_backward_ref(rc, tc, tau, mism)
    got = cp.asnumpy(fb_posteriors_gpu(rc, tc, tau, mism))
    assert got.shape == exp.shape
    assert np.abs(got - exp).max() < 2e-5, np.abs(got - exp).max()
    assert np.abs(got.sum(axis=1) - 1.0).max() < 2e-5


@requires_gpu
def test_gpu_dose_matches_numpy():
    import cupy as cp
    from cugen._impute_gpu import dose_sparse_gpu
    rng = np.random.default_rng(77)
    C, K, T, M = 9, 60, 21, 300
    post = rng.random((C, K, T)).astype(np.float32)
    post /= post.sum(axis=1, keepdims=True)
    bits = (rng.random((K, M)) < rng.uniform(0.01, 0.99, size=M)).astype(np.uint8)
    indptr, indices, major = build_carriers(bits)
    left, lam = interpolation_weights(np.sort(rng.random(C)), np.sort(rng.random(M)))
    got = cp.asnumpy(dose_sparse_gpu(cp.asarray(post), indptr, indices, major,
                                     left, lam))
    exp = np.empty((T, M))
    for m in range(M):
        cl = int(left[m]); cr = min(cl + 1, C - 1)
        exp[:, m] = dose_sparse(post[cl], post[cr], np.array([lam[m]]),
                                indptr, indices, major, [m])[:, 0]
    assert np.abs(got - exp).max() < 2e-5


@requires_gpu
def test_gpu_end_to_end_matches_numpy():
    from cugen._impute_gpu import impute_haplotypes_gpu
    rng = np.random.default_rng(101)
    K, M = 200, 1500
    ref = simulate_mosaic(K, M, seed=101)
    tgt_idx = np.sort(rng.choice(M, size=M // 8, replace=False))
    tgt = simulate_mosaic(16, M, seed=102)[:, tgt_idx]
    cm = np.linspace(0, 6, M)
    kw = dict(ne=scaled_ne(K), cluster=0.005)
    cpu = impute_haplotypes(ref, tgt, tgt_idx, cm, **kw)
    gpu = impute_haplotypes_gpu(ref, tgt, tgt_idx, cm, **kw)
    assert np.abs(cpu - gpu).max() < 5e-4, np.abs(cpu - gpu).max()


@requires_gpu
def test_gpu_tiling_is_transparent():
    """Forcing several T tiles must not change the answer -- the tile loop
    reallocates the posterior slab each pass and must not carry state."""
    from cugen._impute_gpu import impute_haplotypes_gpu
    rng = np.random.default_rng(5)
    K, M = 120, 600
    ref = simulate_mosaic(K, M, seed=5)
    tgt_idx = np.sort(rng.choice(M, size=80, replace=False))
    tgt = simulate_mosaic(24, M, seed=6)[:, tgt_idx]
    cm = np.linspace(0, 3, M)
    kw = dict(ne=scaled_ne(K), cluster=0.005)
    whole = impute_haplotypes_gpu(ref, tgt, tgt_idx, cm, **kw)
    # The budget must be expressed per AGGREGATE marker (C), not per reference
    # marker (M). Sizing it with M made the budget ~7x too generous and no
    # tiling happened -- caught only because the precondition below is asserted
    # rather than assumed.
    from cugen._impute_core import aggregate_markers
    C = aggregate_markers(cm[tgt_idx], 0.005)[0].size
    t = {}
    tiled = impute_haplotypes_gpu(ref, tgt, tgt_idx, cm, timers=t,
                                  budget_bytes=int(4 * (C * K * 4 + K * 4)),
                                  **kw)
    assert t["_t_tile_count"] < 24, f"budget did not force tiling (tile={t['t_tile']})"
    assert np.abs(whole - tiled).max() < 1e-6


# --------------------------------------------------------------------------
# Carrier lists from packed bytes
# --------------------------------------------------------------------------

def emulate_carriers(packed, n_hap, M, bpv):
    """Follows carrier_counts and carrier_scatter statement for statement."""
    ones = np.empty(M, dtype=np.int64)
    for m in range(M):                                   # carrier_counts
        row = packed[m]
        c = 0
        full = n_hap >> 3
        for b in range(full):
            c += int(row[b]).bit_count()
        rem = n_hap & 7
        if rem:
            c += int(row[full] & ((0xFF << (8 - rem)) & 0xFF)).bit_count()
        ones[m] = c
    major = ((ones * 2) > n_hap).astype(np.uint8)
    counts = np.where(major == 1, n_hap - ones, ones)
    indptr = np.zeros(M + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for m in range(M):                                   # carrier_scatter
        row = packed[m]
        want = 0 if major[m] == 1 else 1
        w = int(indptr[m]); end = int(indptr[m + 1])
        j = 0
        while j < n_hap and w < end:
            if ((row[j >> 3] >> (7 - (j & 7))) & 1) == want:
                indices[w] = j; w += 1
            j += 1
    return indptr, indices, major


@pytest.mark.parametrize("n_hap,M", [(8, 5), (13, 7), (4904, 40), (37, 21)])
def test_carrier_kernel_logic_matches_numpy(n_hap, M):
    """Non-multiples of 8 on purpose: the trailing bits of the last byte are
    padding, and counting them would inflate the allele-1 count and flip
    `major` on markers near 50% frequency."""
    from cugen.write import pack_hap2bit
    rng = np.random.default_rng(n_hap * 100 + M)
    bits = (rng.random((n_hap, M)) < rng.uniform(0.02, 0.98, size=M)).astype(np.uint8)
    bpv = (n_hap + 7) // 8
    packed = np.stack([pack_hap2bit(bits[:, m]) for m in range(M)])
    assert packed.shape == (M, bpv)
    got = emulate_carriers(packed, n_hap, M, bpv)
    exp = build_carriers(bits)
    for a, b, name in zip(got, exp, ("indptr", "indices", "major")):
        assert np.array_equal(a, b), name


def test_carrier_padding_bits_are_masked():
    """A marker where every haplotype carries allele 1 must count exactly
    n_hap, not the padded byte width."""
    from cugen.write import pack_hap2bit
    n_hap = 13
    bits = np.ones((n_hap, 1), dtype=np.uint8)
    packed = np.stack([pack_hap2bit(bits[:, 0])])
    indptr, indices, major = emulate_carriers(packed, n_hap, 1, (n_hap + 7) // 8)
    assert major[0] == 1 and indptr[-1] == 0     # no minority carriers at all


@requires_gpu
def test_gpu_carriers_match_numpy():
    import cupy as cp
    from cugen._impute_gpu import build_carriers_gpu
    from cugen.write import pack_hap2bit
    rng = np.random.default_rng(4)
    for n_hap, M in ((4904, 5000), (37, 300), (256, 1024)):
        bits = (rng.random((n_hap, M)) < rng.uniform(0.001, 0.999, size=M)
                ).astype(np.uint8)
        bpv = (n_hap + 7) // 8
        packed = np.stack([pack_hap2bit(bits[:, m]) for m in range(M)])
        ip, ix, mj = build_carriers_gpu(packed, n_hap, M, bpv)
        eip, eix, emj = build_carriers(bits)
        assert np.array_equal(cp.asnumpy(ip), eip), f"indptr n_hap={n_hap}"
        assert np.array_equal(cp.asnumpy(ix), eix), f"indices n_hap={n_hap}"
        assert np.array_equal(cp.asnumpy(mj), emj), f"major n_hap={n_hap}"
