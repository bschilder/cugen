"""Fine-mapping from a stored LD matrix.

`gpu_susie_rss` takes XtX/Xty and derives (R, z) internally, so there was no way
to fine-map from a stored LD panel -- the case where you have summary statistics
and an LD reference but no individual genotypes. That is exactly what
LDReader.dense() and LDMatrix provide, and until now nothing could consume them.

What this does NOT do is replace the in-sample XtX path in
_step5b_finemapping. Three reasons, all structural:

  * XtX and Xty come from one pass over the same centred X, and Xty needs the
    genotypes. Sourcing XtX from a file instead means reading X anyway, so a
    cold run does strictly more I/O.
  * _step5b needs XtX on the COVARIANCE scale -- `beta_hat = Xty / diag(XtX)`
    and the FISTA path both use it. A .cugenld stores correlation, so the
    per-variant variances are gone (recoverable from the .cugen header's sxx,
    but that is a coupling, not a simplification).
  * ld_matrix is pairwise complete-case, so each pair may rest on a different
    sample set and the resulting R is not guaranteed positive semi-definite.
    SuSiE solves against R. One X gives a Gram matrix and PSD for free.

    Measured, 800 samples x 150 variants, against the mean-imputed single-X R
    (benchmarks/pairwise_vs_single_x.py), min eigenvalue of the pairwise R:

        missing   max |dR|   mean |dR|   min eig   PSD
             0%     0.0000     0.00001   +2.99e-1  yes
             2%     0.0111     0.00154   +2.88e-1  yes
             5%     0.0222     0.00368   +2.62e-1  yes
            10%     0.0414     0.00749   +2.17e-1  yes
            20%     0.0923     0.01538   +1.25e-1  yes

    So PSD is a real risk in principle but did not bite here: the smallest
    eigenvalue shrinks monotonically with missingness yet stays far from 0,
    and dR grows roughly linearly at ~0.077 * rate. The reason to keep the
    genotype path is the first two bullets, which hold unconditionally; this
    one is a caveat with a margin, and the 1e-4 ridge this module already adds
    covers more than the drift above. Worth re-checking on a panel with
    non-random missingness, where complete-case sample sets diverge much
    harder than the uniform draw used here.

So the in-sample path keeps its genotype GEMM, and this adds the summary-
statistics entry point it never had.
"""
import numpy as np
import pytest

from conftest import requires_gpu

cp = pytest.importorskip("cupy")

from cugen import ldio                                        # noqa: E402
from cugen._step5b_finemapping import (gpu_susie_rss,          # noqa: E402
                                       gpu_susie_rss_from_ld)


def locus(n=600, p=120, causal=(17, 63), beta=0.55, seed=3):
    """Genotypes with real LD, no missingness, and a known causal set."""
    rng = np.random.default_rng(seed)
    lat = rng.random(n)
    X = np.zeros((n, p), dtype=np.uint8)
    for v in range(p):
        mix = 0.55 if (v // 8) % 2 == 0 else 0.15
        f = rng.uniform(0.15, 0.85)
        score = mix * lat + (1 - mix) * rng.random(n)
        X[:, v] = (score < f).astype(np.uint8) + (rng.random(n) < f)
    X = np.clip(X, 0, 2)
    y = X[:, list(causal)].astype(np.float64) @ np.full(len(causal), beta)
    y = y + rng.normal(0, 1.0, n)
    return X, y - y.mean()


def suff_stats(X, y):
    """The genotype path: one pass over centred X, exactly as _step5b does."""
    Xg = cp.asarray(X.astype(np.float32))
    Xc = Xg - Xg.mean(axis=0)[None, :]
    n = float(X.shape[0])
    return (Xc.T @ Xc / n, Xc.T @ cp.asarray(y.astype(np.float32)) / n,
            float(np.var(y)), X.shape[0])


@requires_gpu
def test_from_ld_reproduces_the_xtx_entry_point_exactly(tmp_path):
    """Pure-refactor equivalence: fed the R and z that gpu_susie_rss derives
    internally, the new entry point must return the same PIPs."""
    X, y = locus()
    XtX, Xty, var_y, n = suff_stats(X, y)

    pips_a, cs_a = gpu_susie_rss(XtX, Xty, var_y, n, L=5, verbose=False)

    d = cp.maximum(cp.diag(XtX).copy(), cp.float32(1e-10))
    sd = cp.sqrt(d)
    R = XtX / (sd[:, None] * sd[None, :])
    R[cp.arange(R.shape[0]), cp.arange(R.shape[0])] = 1.0
    z = cp.sqrt(cp.float32(n)) * (Xty / d) * sd / cp.float32(np.sqrt(var_y))

    pips_b, cs_b = gpu_susie_rss_from_ld(R, z, n, L=5, verbose=False)
    np.testing.assert_allclose(pips_a, pips_b, atol=1e-6)
    assert len(cs_a) == len(cs_b)


@requires_gpu
def test_fine_mapping_from_a_stored_cugenld_matches_the_genotype_path(tmp_path):
    """The point of the feature: R off disk, z from summary statistics.

    With no missing calls the pairwise-complete r in the file and the
    single-X correlation agree, so any PIP difference is the format's
    quantisation, not a change of estimator.
    """
    from cugen import ld as L
    from cugen.write import write_cugen

    X, y = locus()
    path = tmp_path / "locus.cugen"
    write_cugen(str(path), X)
    XtX, Xty, var_y, n = suff_stats(X, y)
    pips_geno, _ = gpu_susie_rss(XtX, Xty, var_y, n, L=5, verbose=False)

    # z-scores are summary statistics -- they do not come from the LD store
    d = cp.maximum(cp.diag(XtX).copy(), cp.float32(1e-10))
    z = (cp.sqrt(cp.float32(n)) * (Xty / d) * cp.sqrt(d)
         / cp.float32(np.sqrt(var_y)))

    out = str(tmp_path / "locus.cugenld")
    L.ld_matrix(str(path), stats=("r", "r2"), min_r2=0.0, output=out,
                backend="numpy", verbose=False)
    R_disk = cp.asarray(ldio.read_ld(out).dense(n_variants=X.shape[1])
                        .astype(np.float32))

    pips_disk, _ = gpu_susie_rss_from_ld(R_disk, z, n, L=5, verbose=False)
    assert np.abs(pips_disk - pips_geno).max() < 0.02, (
        f"max |dPIP| {np.abs(pips_disk - pips_geno).max():.4f}")
    # and the causal variants still lead
    assert set(np.argsort(pips_disk)[-2:]) == set(np.argsort(pips_geno)[-2:])


@requires_gpu
def test_from_ld_refuses_a_shape_mismatch():
    """Both guards, each matched on its OWN message.

    The loose `match="shape|length|z"` this started as passed even with the
    z-length check deleted, because cupy's broadcast failure is also a
    ValueError that says "shape" -- the test was green for the wrong reason
    (mutation_sweep.py, 'from_ld accepts z and R of different lengths').
    Anchoring on text only cugen emits is what makes it bite.
    """
    with pytest.raises(ValueError, match="same variant order"):
        gpu_susie_rss_from_ld(cp.eye(10, dtype=cp.float32),
                              cp.zeros(7, dtype=cp.float32), 500,
                              verbose=False)

    with pytest.raises(ValueError, match="must be square"):
        gpu_susie_rss_from_ld(cp.zeros((4, 9), dtype=cp.float32),
                              cp.zeros(9, dtype=cp.float32), 500,
                              verbose=False)


@requires_gpu
def test_from_ld_leaves_the_callers_R_untouched():
    """The ridge goes on a copy.

    A caller's R is very often not theirs to lose: dense() can hand back a view
    over a mapped .cugenld, and the same panel is reused across every locus in
    a run. Adding 1e-4 in place would compound the ridge once per call, so the
    tenth locus would be solving against a different matrix than the first.
    """
    X, _ = locus()
    Xc = X - X.mean(axis=0)
    C = Xc.T @ Xc / X.shape[0]
    sd = np.sqrt(np.maximum(np.diag(C), 1e-12))
    R_host = (C / np.outer(sd, sd)).astype(np.float32)
    np.fill_diagonal(R_host, 1.0)

    R = cp.asarray(R_host)                  # float32 already: astype can alias
    z = cp.asarray(np.linspace(-3, 3, R_host.shape[0]).astype(np.float32))
    before = R.copy()

    gpu_susie_rss_from_ld(R, z, n_samples=X.shape[0], L=3, max_iter=5,
                          verbose=False)

    assert cp.array_equal(R, before), (
        "gpu_susie_rss_from_ld mutated the caller's R; the diagonal moved by "
        f"{float(cp.abs(cp.diag(R) - cp.diag(before)).max()):.2e}")
