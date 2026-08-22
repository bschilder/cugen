"""Structure- and kinship-corrected r^2 (Mangin et al. 2012, Heredity 108:285-291).

Three bias-corrected extensions of r^2, for samples that are structured, related,
or both:

    r2_s   corrected for population structure    (their eq. 1)
    r2_v   corrected for relatedness / kinship   (their eq. 2)
    r2_vs  corrected for both                    (their eq. 3)

These are ESTIMATORS ONLY. The paper proves them unbiased for unlinked loci and
links them to association-test power, but derives no null sampling distribution,
so chi2 = N * r^2 does not transfer to them -- after GLS centring and rank-K
residualisation the effective sample size is not N and nothing says what it is.
Asking for them alongside chi2/p raises rather than inventing a
degrees-of-freedom. See the References block in cugen/ld.py.

The oracles below are literal transcriptions of the authors' own R code
(LDcorSV 1.3.3, Measure.R2V / Measure.R2S / Measure.R2VS), which is a genuinely
different implementation in a different language rather than a restatement of
the production path.

Note on the paper's Table 1: its published numbers are not reproduced here.
Pooling two populations induces Hardy-Weinberg disequilibrium (Wahlund), so the
composite genotypic r^2 exceeds the gametic one -- a two-population sample with
true r^2 = 0.01 and allele frequencies 0.9/0.1 at both loci gives 0.65 on
dosages against the paper's 0.460, which is the gametic value. The paper does
not say which its simulation produced, so matching that table would be testing
a reconstruction of their simulation rather than this implementation of their
formula. The structural claim is tested instead, in
test_r2_s_removes_the_inflation_that_structure_causes.
"""
import numpy as np
import pytest

from cugen import ld as L
from cugen.write import write_cugen

CPU = dict(backend="numpy", verbose=False)


def pinv_psd(V, tol=1e-5):
    """Moore-Penrose inverse via eigendecomposition, eigenvalues < tol zeroed.

    Transcribed from LDcorSV's Inv.proj.matrix.sdp. The paper specifies the
    Moore-Penrose inverse V^- "which is always defined"; the 1e-5 floor is the
    R package's choice, not the paper's.
    """
    w, U = np.linalg.eigh(np.asarray(V, dtype=np.float64))
    wi = np.where(w < tol, 0.0, 1.0 / np.where(w < tol, 1.0, w))
    return U @ np.diag(wi) @ U.T


def _cov(A, B=None):
    A = np.asarray(A, dtype=np.float64)
    A = A - A.mean(axis=0)
    if B is None:
        return A.T @ A / (A.shape[0] - 1)
    B = np.asarray(B, dtype=np.float64)
    B = B - B.mean(axis=0)
    return A.T @ B / (A.shape[0] - 1)


def oracle_r2v(x, y, V):
    """LDcorSV Measure.R2V, transcribed."""
    n = len(x)
    Vi = pinv_psd(V)
    one = np.ones(n)
    DATA = np.column_stack([x, y]).astype(np.float64)
    FACT = np.outer(one / (one @ Vi @ one), one) @ Vi
    MAT = DATA - FACT @ DATA
    SIG = MAT.T @ Vi @ MAT
    return SIG[0, 1] ** 2 / (SIG[0, 0] * SIG[1, 1])


def oracle_r2s(x, y, S):
    """LDcorSV Measure.R2S, transcribed."""
    B = np.column_stack([x, y]).astype(np.float64)
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    if S.shape[0] != B.shape[0]:
        S = S.T
    SIG = _cov(B) - _cov(B, S) @ pinv_psd(_cov(S)) @ _cov(S, B)
    return SIG[0, 1] ** 2 / (SIG[0, 0] * SIG[1, 1])


def oracle_r2vs(x, y, V, S):
    """LDcorSV Measure.R2VS, transcribed."""
    n = len(x)
    Vi = pinv_psd(V)
    one = np.ones(n)
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    if S.shape[0] != n:
        S = S.T
    DATA = np.column_stack([x, y, S]).astype(np.float64)
    FACT = np.outer(one / (one @ Vi @ one), one) @ Vi
    MAT = DATA - FACT @ DATA
    SIG = MAT.T @ Vi @ MAT
    iss = pinv_psd(SIG[2:, 2:])
    s1, s2 = SIG[0, 2:], SIG[1, 2:]
    num = (SIG[0, 1] - s1 @ iss @ s2) ** 2
    d11 = SIG[0, 0] - s1 @ iss @ s1
    d22 = SIG[1, 1] - s2 @ iss @ s2
    return num / (d11 * d22)


# ------------------------------------------------------------------ fixtures
def two_pop_panel(n_per, freqs1, freqs2, r2, seed):
    """Two populations, same within-population r^2, divergent allele freqs."""
    rng = np.random.default_rng(seed)
    blocks = []
    for pa, pb in ((freqs1, freqs2)[0], (freqs1, freqs2)[1]):
        D = np.sqrt(r2 * pa * (1 - pa) * pb * (1 - pb))
        h = np.array([pa * pb + D, pa * (1 - pb) - D,
                      (1 - pa) * pb - D, (1 - pa) * (1 - pb) + D])
        h = np.clip(h, 0.0, None)
        h /= h.sum()
        idx = rng.choice(4, size=(n_per, 2), p=h)
        A = np.isin(idx, [0, 1]).sum(axis=1)
        B = np.isin(idx, [0, 2]).sum(axis=1)
        blocks.append(np.stack([A, B], axis=1).astype(np.uint8))
    return blocks


@pytest.fixture
def panel(tmp_path):
    """20 variants x 120 samples, plus a matching kinship and structure matrix."""
    rng = np.random.default_rng(5)
    n, m = 120, 20
    lat = rng.random(n)
    dos = np.empty((n, m), dtype=np.uint8)
    for v in range(m):
        mix = rng.uniform(0.0, 0.8)
        f = rng.uniform(0.2, 0.8)
        score = mix * lat + (1 - mix) * rng.random(n)
        dos[:, v] = (score < f).astype(np.uint8) + (rng.random(n) < f)
    dos = np.clip(dos, 0, 2)
    path = tmp_path / "p.cugen"
    write_cugen(str(path), dos)
    # a valid PSD kinship: a GRM of independent markers plus a ridge
    Z = rng.normal(size=(n, 400))
    V = Z @ Z.T / 400.0 + 0.1 * np.eye(n)
    S = np.column_stack([lat, rng.random(n)])
    return str(path), dos, V, S


# --------------------------------------------------------------------- tests
def test_r2_v_matches_the_authors_r_implementation(panel):
    path, dos, V, _ = panel
    df = L.ld_matrix(path, stats=("r2_v",), kinship=V, **CPU)
    worst, checked = 0.0, 0
    for a, b, got in zip(df["gidx_a"], df["gidx_b"], df["R2_V"]):
        want = oracle_r2v(dos[:, int(a)], dos[:, int(b)], V)
        worst = max(worst, abs(float(got) - want))
        checked += 1
    assert checked >= 150, f"only {checked} pairs compared"
    assert worst < 1e-6, f"max |diff| vs LDcorSV Measure.R2V: {worst:.3e}"


def test_r2_s_matches_the_authors_r_implementation(panel):
    path, dos, _, S = panel
    df = L.ld_matrix(path, stats=("r2_s",), structure=S, **CPU)
    worst, checked = 0.0, 0
    for a, b, got in zip(df["gidx_a"], df["gidx_b"], df["R2_S"]):
        want = oracle_r2s(dos[:, int(a)], dos[:, int(b)], S)
        worst = max(worst, abs(float(got) - want))
        checked += 1
    assert checked >= 150
    assert worst < 1e-6, f"max |diff| vs LDcorSV Measure.R2S: {worst:.3e}"


def test_r2_vs_matches_the_authors_r_implementation(panel):
    path, dos, V, S = panel
    df = L.ld_matrix(path, stats=("r2_vs",), kinship=V, structure=S, **CPU)
    worst, checked = 0.0, 0
    for a, b, got in zip(df["gidx_a"], df["gidx_b"], df["R2_VS"]):
        want = oracle_r2vs(dos[:, int(a)], dos[:, int(b)], V, S)
        worst = max(worst, abs(float(got) - want))
        checked += 1
    assert checked >= 150
    assert worst < 1e-6, f"max |diff| vs LDcorSV Measure.R2VS: {worst:.3e}"


def test_identity_kinship_reduces_r2_v_to_ordinary_r2(panel):
    """With V = I the GLS mean is the ordinary mean and the weighting vanishes,
    so eq. (2) must collapse onto the usual r^2. A correction that does not
    reduce to the thing it corrects is wrong."""
    path, dos, _, _ = panel
    n = dos.shape[0]
    df = L.ld_matrix(path, stats=("r2", "r2_v"), kinship=np.eye(n), **CPU)
    np.testing.assert_allclose(df["R2_V"].to_numpy(np.float64),
                               df["R2"].to_numpy(np.float64), atol=1e-6)


def test_structure_uncorrelated_with_the_loci_leaves_r2_alone(panel):
    """r2_s residualises against S. If S carries no information about the loci
    there is nothing to remove, so r2_s must stay close to r2."""
    path, dos, _, _ = panel
    rng = np.random.default_rng(99)
    noise = rng.random((dos.shape[0], 2))
    df = L.ld_matrix(path, stats=("r2", "r2_s"), structure=noise, **CPU)
    d = np.abs(df["R2_S"].to_numpy(np.float64) - df["R2"].to_numpy(np.float64))
    assert d.max() < 0.05, f"irrelevant structure shifted r2 by {d.max():.3f}"


def test_r2_s_removes_the_inflation_that_structure_causes(tmp_path):
    """The paper's central claim, as a direction and a magnitude.

    Two populations, allele frequencies 0.9 vs 0.1 at both loci, true within-
    population r^2 = 0.01. Pooling them inflates r^2 enormously; conditioning on
    the population indicator must bring it back to roughly the within-population
    value.
    """
    p1, p2 = two_pop_panel(300, (0.9, 0.9), (0.1, 0.1), 0.01, seed=7)
    within = tmp_path / "w.cugen"
    write_cugen(str(within), p1)
    r2_within = float(L.ld_matrix(str(within), stats=("r2",), **CPU)["R2"].iloc[0])

    pooled = np.vstack([p1, p2])
    path = tmp_path / "p.cugen"
    write_cugen(str(path), pooled)
    S = np.concatenate([np.zeros(len(p1)), np.ones(len(p2))])[:, None]
    df = L.ld_matrix(str(path), stats=("r2", "r2_s"), structure=S, **CPU)
    r2_pooled = float(df["R2"].iloc[0])
    r2_s = float(df["R2_S"].iloc[0])

    assert r2_pooled > 0.4, f"fixture is not confounded (pooled r2 = {r2_pooled:.3f})"
    assert r2_s < 0.1, f"r2_s did not remove the structure inflation: {r2_s:.3f}"
    assert abs(r2_s - r2_within) < 0.05, (
        f"r2_s {r2_s:.4f} should land near the within-population r^2 "
        f"{r2_within:.4f}, not the pooled {r2_pooled:.4f}")


def two_loci(n, r2, seed, pa=0.5, pb=0.5):
    """n genotypes at two loci with a specified within-sample r^2."""
    rng = np.random.default_rng(seed)
    D = np.sqrt(r2 * pa * (1 - pa) * pb * (1 - pb))
    h = np.array([pa * pb + D, pa * (1 - pb) - D,
                  (1 - pa) * pb - D, (1 - pa) * (1 - pb) + D])
    h = np.clip(h, 0.0, None)
    h /= h.sum()
    idx = rng.choice(4, size=(n, 2), p=h)
    return np.stack([np.isin(idx, [0, 1]).sum(axis=1),
                     np.isin(idx, [0, 2]).sum(axis=1)], axis=1).astype(np.uint8)


def test_r2_v_corrects_the_inflation_that_clones_cause(tmp_path):
    """The paper's Table 2 clone scenario, reproduced.

    Duplicating a genotype over-represents one haplotype and inflates r^2.
    Handing r2_v a kinship matrix in which the clones sit at 1 must pull the
    estimate back to what the independent samples alone would have given.

    Averaged over replicates because a single r^2 at n = 80 is noisy; the
    quantity under test is the bias, which is a property of the mean.
    """
    def run(dos, **kw):
        path = tmp_path / "x.cugen"
        write_cugen(str(path), dos)
        return L.ld_matrix(str(path), **kw, **CPU)

    n_ind, n_clone, true_r2 = 80, 20, 0.05
    naive, corrected, independent = [], [], []
    for seed in range(60):
        base = two_loci(n_ind, true_r2, seed)
        sample = np.vstack([base, np.repeat(base[:1], n_clone, axis=0)])
        N = sample.shape[0]
        # kinship 1 within the clone group, 1 on the diagonal, 0 elsewhere --
        # exactly the V the paper describes for this scenario
        V = np.eye(N)
        group = [0] + list(range(n_ind, N))
        for i in group:
            for j in group:
                V[i, j] = 1.0

        independent.append(float(run(base, stats=("r2",))["R2"].iloc[0]))
        got = run(sample, stats=("r2", "r2_v"), kinship=V)
        naive.append(float(got["R2"].iloc[0]))
        corrected.append(float(got["R2_V"].iloc[0]))

    m_ind, m_naive, m_corr = np.mean(independent), np.mean(naive), np.mean(corrected)
    assert m_naive > m_ind + 0.008, (
        f"clones did not inflate r^2 ({m_naive:.4f} vs {m_ind:.4f}); there is "
        f"nothing here to correct")
    assert abs(m_corr - m_ind) < abs(m_naive - m_ind), (
        f"r2_v ({m_corr:.4f}) is no closer to the independent-sample value "
        f"({m_ind:.4f}) than the naive r^2 ({m_naive:.4f})")
    assert m_corr < m_naive


# ---------------------------------------------------------------- refusals
def test_corrected_measures_refuse_to_carry_a_p_value(panel):
    """The paper gives no null distribution for these, so there is no honest
    p-value to emit. Refuse rather than invent a degrees-of-freedom."""
    path, _, V, _ = panel
    with pytest.raises(ValueError, match="no null distribution|null distribution"):
        L.ld_matrix(path, stats=("r2_v", "p"), kinship=V, **CPU)


def test_r2_v_without_a_kinship_matrix_is_refused(panel):
    with pytest.raises(ValueError, match="kinship"):
        L.ld_matrix(panel[0], stats=("r2_v",), **CPU)


def test_r2_s_without_a_structure_matrix_is_refused(panel):
    with pytest.raises(ValueError, match="structure"):
        L.ld_matrix(panel[0], stats=("r2_s",), **CPU)


def test_a_misshaped_kinship_matrix_is_refused(panel):
    path, dos, _, _ = panel
    with pytest.raises(ValueError, match="shape|n_samples"):
        L.ld_matrix(path, stats=("r2_v",), kinship=np.eye(dos.shape[0] - 3),
                    **CPU)
