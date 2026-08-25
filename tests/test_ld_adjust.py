"""Ancestry-adjusted r^2 on the fused GPU path (`r_adj` / `r2_adj`).

Same estimator as `r2_s` -- residualise the genotypes on [1 | structure], then
correlate -- but reached by a route that keeps the fused scan, and with it
`count_only` and `stream`. `r2_s` cannot: it forces the reference path and
builds a dense (p, p) host Gram, which is 1.25 PB at p = 12.5M.

The identity that makes it cheap. With U an orthonormal basis of [1 | S] and
P = U U^T:

    g_i^T (I - P) g_j  =  g_i^T g_j - (U^T g_i)^T (U^T g_j)

so with Y = G^T U (variants x K), one GEMM over AUGMENTED planes

    [G | Y] @ [G | -Y]^T  =  G^T G - Y Y^T  =  G^T (I - P) G

gives the residual cross-product, and `q_adj = q - ||Y||^2` gives the residual
variances. Because 1 is in col(P) the epilogue's centring term vanishes, so
handing the existing kernel `s = 0` and `q = q_adj` reproduces r2_s with the
kernel unmodified -- see the note at the head of cugen/ld.py about zero sum
vectors collapsing `(n*S - sA*sB)/sqrt(...)` to `S/sqrt(qA*qB)`.

Cost is the K extra columns: +K/n flops, 0.84% at n = 2504, K = 21.

What these tests are FOR, individually, is written on each one. Two of them
(test_annihilated_..., test_count_only_matches_...) exist because the failure
they catch is invisible at unit scale and corrupts genome-scale output.
"""
import numpy as np
import pytest

from cugen import ld as L
from cugen.write import write_cugen
from test_ld_corrected import oracle_r2s

CPU = dict(backend="numpy", verbose=False)


# --------------------------------------------------------------------- fixture
@pytest.fixture
def structured(tmp_path):
    """Two populations with divergent allele frequencies, plus PCs.

    This is the Wahlund setup: no LD is simulated WITHIN either population, so
    every pair's true r^2 is 0 and anything the naive estimator reports is the
    two-locus Wahlund effect.
    """
    rng = np.random.default_rng(11)
    n, m, k = 240, 30, 3
    pop = np.repeat([0, 1], n // 2)
    dos = np.empty((n, m), dtype=np.uint8)
    for v in range(m):
        f = 0.2 if v % 2 == 0 else 0.75
        fr = np.where(pop == 0, f, 1.0 - f)
        dos[:, v] = rng.binomial(2, fr)
    path = tmp_path / "structured.cugen"
    write_cugen(str(path), dos)

    G = dos.astype(np.float64)
    Gc = G - G.mean(0)
    grm = Gc @ Gc.T / m
    w, V = np.linalg.eigh(grm)
    pcs = V[:, ::-1][:, :k]
    return str(path), dos, pcs, pop


def _adj_reference(dos, pcs):
    """Residualised r^2 for every pair, in float64 on the host.

    Deliberately written as an explicit projection rather than reusing the
    production algebra, so a shared mistake cannot pass both.
    """
    G = np.asarray(dos, dtype=np.float64)
    n = G.shape[0]
    Z = np.column_stack([np.ones(n), np.asarray(pcs, dtype=np.float64)])
    U, _ = np.linalg.qr(Z)
    R = G - U @ (U.T @ G)
    q = (R * R).sum(0)
    C = R.T @ R
    m = G.shape[1]
    out = np.full((m, m), np.nan)
    for i in range(m):
        for j in range(m):
            if q[i] > 0 and q[j] > 0:
                out[i, j] = C[i, j] / np.sqrt(q[i] * q[j])
    return out


# ----------------------------------------------------------------------- tests
def test_r2_adj_matches_the_ldcorsv_oracle(structured):
    """The estimator is r2_s. Anchor it to the authors' own R code.

    oracle_r2s is a Schur complement of covariance matrices; the production
    path is a residual projection. Structurally different routes to the same
    number, so agreement here is evidence about the algebra, not just the code.
    """
    path, dos, pcs, _ = structured
    df = L.ld_matrix(path, stats=("r2_adj",), structure=pcs, **CPU)
    worst, checked = 0.0, 0
    for a, b, got in zip(df["gidx_a"], df["gidx_b"], df["R2_ADJ"]):
        want = oracle_r2s(dos[:, a].astype(float), dos[:, b].astype(float), pcs)
        worst = max(worst, abs(got - want))
        checked += 1
    assert checked > 100, f"only {checked} pairs compared"
    assert worst < 1e-6, f"max |r2_adj - oracle_r2s| = {worst:.3e}"


def test_r2_adj_matches_an_independent_host_projection(structured):
    """Second oracle, written as an explicit (I - UU^T) projection."""
    path, dos, pcs, _ = structured
    ref = _adj_reference(dos, pcs)
    df = L.ld_matrix(path, stats=("r2_adj",), structure=pcs, **CPU)
    worst = 0.0
    for a, b, got in zip(df["gidx_a"], df["gidx_b"], df["R2_ADJ"]):
        worst = max(worst, abs(got - ref[a, b] ** 2))
    assert worst < 1e-6, f"max |r2_adj - projection| = {worst:.3e}"


def test_structure_of_pure_noise_leaves_r2_alone(structured):
    """Catches a missing intercept in the projector.

    Residualising on noise should remove nothing. If the intercept is left out
    of the basis, r becomes an UNCENTRED correlation and this diverges hard --
    the fixture's allele frequencies (0.2 / 0.75) are far enough from 0.5 to
    make that unmissable.
    """
    path, dos, _, _ = structured
    rng = np.random.default_rng(3)
    noise = rng.normal(size=(dos.shape[0], 3))
    plain = L.ld_matrix(path, stats=("r2",), **CPU)
    adj = L.ld_matrix(path, stats=("r2_adj",), structure=noise, **CPU)
    key = {(a, b): v for a, b, v in
           zip(plain["gidx_a"], plain["gidx_b"], plain["R2"])}
    worst = max(abs(key[(a, b)] - v) for a, b, v in
                zip(adj["gidx_a"], adj["gidx_b"], adj["R2_ADJ"]))
    assert worst < 0.05, f"noise structure moved r2 by {worst:.4f}"


def test_rank_deficient_structure_is_a_no_op(structured):
    """A constant column spans the intercept already, so nothing is removed."""
    path, dos, _, _ = structured
    const = np.ones((dos.shape[0], 1))
    plain = L.ld_matrix(path, stats=("r2",), **CPU)
    adj = L.ld_matrix(path, stats=("r2_adj",), structure=const, **CPU)
    key = {(a, b): v for a, b, v in
           zip(plain["gidx_a"], plain["gidx_b"], plain["R2"])}
    worst = max(abs(key[(a, b)] - v) for a, b, v in
                zip(adj["gidx_a"], adj["gidx_b"], adj["R2_ADJ"]))
    assert worst < 1e-8, f"constant structure moved r2 by {worst:.3e}"


def test_annihilated_variant_is_dropped_not_emitted_as_one(tmp_path):
    """A variant that IS the structure has zero residual.

    Its r_adj is 0/0. Emitting it as +-1 would, at genome scale, produce
    billions of spurious r2 = 1 rows -- the failure mode is invisible on a
    small fixture unless it is asked for directly.
    """
    rng = np.random.default_rng(7)
    n, m = 200, 8
    pop = np.repeat([0, 1], n // 2)
    dos = np.empty((n, m), dtype=np.uint8)
    for v in range(m - 1):
        f = rng.uniform(0.25, 0.7)
        dos[:, v] = rng.binomial(2, f)
    dos[:, m - 1] = (pop * 2).astype(np.uint8)      # exactly 2 * indicator
    path = tmp_path / "annih.cugen"
    write_cugen(str(path), dos)
    S = pop.reshape(-1, 1).astype(float)

    df = L.ld_matrix(path, stats=("r2_adj",), structure=S, **CPU)
    involved = [(a, b, v) for a, b, v in
                zip(df["gidx_a"], df["gidx_b"], df["R2_ADJ"])
                if m - 1 in (a, b)]
    assert all(not np.isfinite(v) or v < 0.999 for _, _, v in involved), (
        f"annihilated variant emitted near-perfect r2: {involved[:5]}")


def test_adjustment_collapses_wahlund_but_spares_real_cis_ld(tmp_path):
    """The headline control, and the one that catches a SIGN error.

    Flipping the sign of the augmented block ADDS the Wahlund term instead of
    removing it. Cis would still look fine, so only the trans arm reveals it --
    which is why both arms are asserted here.
    """
    rng = np.random.default_rng(19)
    n, per_chrom = 300, 20
    pop = np.repeat([0, 1], n // 2)
    cols, chrom = [], []
    for c in (0, 1):                       # two independent "chromosomes"
        # one tight cis block with real within-population LD
        base = rng.binomial(2, np.where(pop == 0, 0.35, 0.6))
        for v in range(per_chrom):
            if v < 5:                      # cis block: correlated with base
                flip = rng.random(n) < 0.08
                col = np.where(flip, rng.binomial(2, 0.4), base)
            else:
                f = 0.25 if v % 2 else 0.7
                col = rng.binomial(2, np.where(pop == 0, f, 1 - f))
            cols.append(col.astype(np.uint8)); chrom.append(c)
    dos = np.stack(cols, axis=1)
    chrom = np.asarray(chrom)
    path = tmp_path / "twochrom.cugen"
    write_cugen(str(path), dos)

    # Ancestry is given directly rather than estimated. Deriving PCs from THIS
    # panel does not work and the reason matters: 10 of its 40 variants are
    # near-copies of two block founders, so those blocks dominate the top
    # eigenvectors of the GRM and the resulting "PCs" describe LD structure
    # instead of the population split -- residualising on them removes real cis
    # LD and leaves the Wahlund term behind. That is why PCs on real data must
    # come from an LD-PRUNED subset (see cugen.ld.ld_prune). Here the confounder
    # is known exactly, so use it and keep this test about the estimator.
    pcs = pop.reshape(-1, 1).astype(float)

    raw = L.ld_matrix(path, stats=("r2",), **CPU)
    adj = L.ld_matrix(path, stats=("r2_adj",), structure=pcs, **CPU)

    def split(df, col):
        tr, ci = [], []
        for a, b, v in zip(df["gidx_a"], df["gidx_b"], df[col]):
            if not np.isfinite(v):
                continue
            (tr if chrom[a] != chrom[b] else ci).append(v)
        return np.asarray(tr), np.asarray(ci)

    tr_raw, ci_raw = split(raw, "R2")
    tr_adj, ci_adj = split(adj, "R2_ADJ")

    # trans: pure Wahlund, must collapse toward the LE null
    assert tr_raw.mean() > 5 * tr_adj.mean(), (
        f"trans r2 did not collapse: raw {tr_raw.mean():.4f} "
        f"adj {tr_adj.mean():.4f}")
    assert (tr_adj >= 0.1).sum() < 0.2 * max((tr_raw >= 0.1).sum(), 1)

    # cis: real LD inside the tight block must largely survive
    strong_raw = np.sort(ci_raw)[-10:].mean()
    strong_adj = np.sort(ci_adj)[-10:].mean()
    assert strong_adj > 0.5 * strong_raw, (
        f"adjustment destroyed real cis LD: {strong_raw:.3f} -> {strong_adj:.3f}")


def test_k_zero_reproduces_the_unadjusted_scan(structured):
    """No structure columns means no correction. A free correctness check."""
    path, dos, _, _ = structured
    plain = L.ld_matrix(path, stats=("r2",), **CPU)
    adj = L.ld_matrix(path, stats=("r2_adj",),
                      structure=np.zeros((dos.shape[0], 0)), **CPU)
    key = {(a, b): v for a, b, v in
           zip(plain["gidx_a"], plain["gidx_b"], plain["R2"])}
    worst = max(abs(key[(a, b)] - v) for a, b, v in
                zip(adj["gidx_a"], adj["gidx_b"], adj["R2_ADJ"]))
    assert worst < 1e-9, f"k=0 changed r2 by {worst:.3e}"


def test_r2_adj_refuses_to_carry_a_p_value(structured):
    """chi2 = N * r^2 does not transfer after rank-K residualisation.

    Same refusal the module already makes for r2_s: the effective sample size
    is not N and nothing says what it is, so a p-value here would look like
    rigour and be the opposite.
    """
    path, _, pcs, _ = structured
    with pytest.raises(ValueError, match="null distribution|p-value|chi2"):
        L.ld_matrix(path, stats=("r2_adj", "p"), structure=pcs, **CPU)


def test_r2_adj_requires_a_structure_matrix(structured):
    path, _, _, _ = structured
    with pytest.raises(ValueError, match="structure"):
        L.ld_matrix(path, stats=("r2_adj",), **CPU)


@pytest.mark.skipif(not L.HAS_CUPY, reason="needs a GPU")
def test_gpu_agrees_with_the_reference_path(structured):
    """The fused path is the point of all this. It must match the CPU oracle."""
    path, dos, pcs, _ = structured
    cpu = L.ld_matrix(path, stats=("r2_adj",), structure=pcs, **CPU)
    gpu = L.ld_matrix(path, stats=("r2_adj",), structure=pcs,
                      backend="gpu", verbose=False)
    key = {(a, b): v for a, b, v in
           zip(cpu["gidx_a"], cpu["gidx_b"], cpu["R2_ADJ"])}
    worst = max(abs(key[(a, b)] - v) for a, b, v in
                zip(gpu["gidx_a"], gpu["gidx_b"], gpu["R2_ADJ"]))
    assert worst < 1e-4, f"GPU vs CPU max |dr2_adj| = {worst:.3e}"


@pytest.mark.skipif(not L.HAS_CUPY, reason="needs a GPU")
def test_count_only_matches_the_streamed_row_count(structured, tmp_path):
    """Catches a q_adj omitted from the OVERFLOW-RETRY kernel launch.

    That launch fires only when a single tile overflows the buffer, so a
    version missing it passes every other test here and silently corrupts
    genome-scale shards. flush_rows is set small to force it.
    """
    path, _, pcs, _ = structured
    n = L.ld_matrix(path, stats=("r2_adj",), structure=pcs, min_r2=0.0,
                    count_only=True, verbose=False)
    out = tmp_path / "s.cugenld"
    m = L.ld_matrix(path, stats=("r2_adj",), structure=pcs, min_r2=0.0,
                    stream=True, output=str(out), flush_rows=1 << 16,
                    verbose=False)
    assert n == m, f"count_only {n} != streamed {m}"
