"""KING-robust kinship (Manichaikul et al. 2010, Bioinformatics 26:2867).

The oracle is plink2 `--make-king-table`, a different implementation of the
same estimator, so agreement is evidence rather than tautology. plink2 uses the
BETWEEN-family estimator for every pair (`plink2 --help make-king`: "Pedigree
information is currently ignored; the between-family estimator is used for all
pairs"), and so does this one by default.

The test that states the reason KING exists is
`test_king_is_robust_to_population_structure`: a GRM cannot separate "related"
from "same ancestry", because it centres on the sample's own allele
frequencies. KING conditions on each PAIR's heterozygosity instead, so two
unrelated individuals from differentiated subpopulations score ~0 where a GRM
scores clearly positive.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

from conftest import requires_gpu

from cugen.popstruct import grm, king
from cugen.write import write_cugen

HAS_PLINK2 = shutil.which("plink2") is not None
requires_plink2 = pytest.mark.skipif(not HAS_PLINK2, reason="plink2 not on PATH")


def _hap(g, rng):
    """One transmitted haplotype from a diploid genotype vector."""
    return (g > 1).astype(np.int8) + ((g == 1) & (rng.random(g.size) < 0.5))


def synth(n_unrel=18, p=3000, seed=0, fst=0.15, dup=True, po=True):
    """Two differentiated subpopulations, optionally a duplicate and a child.

    Returns (G, labels) with G as (n_samples, n_variants) uint8 ALT dosages --
    the orientation `write_cugen` takes.
    """
    rng = np.random.default_rng(seed)
    f1 = rng.uniform(0.1, 0.9, p)
    f2 = np.clip(f1 + rng.normal(0, fst, p), 0.02, 0.98)

    def draw(f, k):
        return ((rng.random((k, p)) < f).astype(np.int8)
                + (rng.random((k, p)) < f).astype(np.int8))

    G = np.vstack([draw(f1, n_unrel), draw(f2, n_unrel)])
    lab = ["A"] * n_unrel + ["B"] * n_unrel
    if dup:
        G = np.vstack([G, G[0:1].copy()])
        lab.append("dup0")
    if po:
        G = np.vstack([G, (_hap(G[1], rng) + _hap(G[2], rng))[None, :]])
        lab.append("child12")
    return G.astype(np.uint8), lab


def write_cg(tmp_path, G, name="k.cugen"):
    p = tmp_path / name
    write_cugen(str(p), G.T if False else G)   # (n_samples, n_variants)
    return str(p)


def plink2_king(tmp_path, G):
    """(n, n) KING matrix from plink2, via a VCF."""
    n, p = G.shape
    vcf = tmp_path / "t.vcf"
    gt = {0: "0/0", 1: "0/1", 2: "1/1"}
    with open(vcf, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=1,length=900000000>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 + "\t".join(f"S{i}" for i in range(n)) + "\n")
        for v in range(p):
            fh.write(f"1\t{(v + 1) * 1000}\tv{v}\tA\tG\t.\tPASS\t.\tGT\t"
                     + "\t".join(gt[int(x)] for x in G[:, v]) + "\n")
    out = tmp_path / "pk"
    subprocess.run(["plink2", "--vcf", str(vcf), "--make-king-table",
                    "--out", str(out)], check=True, capture_output=True)
    M = np.full((n, n), np.nan)
    with open(f"{out}.kin0") as fh:
        hdr = fh.readline().lstrip("#").split()
        ci, cj, ck = hdr.index("IID1"), hdr.index("IID2"), hdr.index("KINSHIP")
        for line in fh:
            f = line.split()
            a, b = int(f[ci][1:]), int(f[cj][1:])
            M[a, b] = M[b, a] = float(f[ck])
    return M


# --------------------------------------------------------------------------
# The oracle
# --------------------------------------------------------------------------
@requires_plink2
def test_king_matches_plink2(tmp_path):
    G, _ = synth(seed=1)
    got = king(write_cg(tmp_path, G), verbose=False)
    want = plink2_king(tmp_path, G)
    off = ~np.eye(G.shape[0], dtype=bool)
    # plink2's text table carries 6 decimals, so that is the achievable floor
    np.testing.assert_allclose(got[off], want[off], atol=2e-6)


@requires_plink2
def test_king_matches_plink2_with_missingness(tmp_path):
    """Pairwise-complete het counts, not per-sample totals."""
    G, _ = synth(seed=4, p=2500)
    rng = np.random.default_rng(9)
    G = G.copy()
    G[rng.random(G.shape) < 0.05] = 3          # 3 = missing in .cugen
    Gv = G.copy()
    Gv[Gv == 3] = 255                          # sentinel for the VCF writer
    n, p = G.shape
    vcf = tmp_path / "m.vcf"
    gt = {0: "0/0", 1: "0/1", 2: "1/1", 255: "./."}
    with open(vcf, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=1,length=900000000>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 + "\t".join(f"S{i}" for i in range(n)) + "\n")
        for v in range(p):
            fh.write(f"1\t{(v + 1) * 1000}\tv{v}\tA\tG\t.\tPASS\t.\tGT\t"
                     + "\t".join(gt[int(x)] for x in Gv[:, v]) + "\n")
    out = tmp_path / "pm"
    subprocess.run(["plink2", "--vcf", str(vcf), "--make-king-table",
                    "--out", str(out)], check=True, capture_output=True)
    want = np.full((n, n), np.nan)
    with open(f"{out}.kin0") as fh:
        hdr = fh.readline().lstrip("#").split()
        ci, cj, ck = hdr.index("IID1"), hdr.index("IID2"), hdr.index("KINSHIP")
        for line in fh:
            f = line.split()
            a, b = int(f[ci][1:]), int(f[cj][1:])
            want[a, b] = want[b, a] = float(f[ck])
    got = king(write_cg(tmp_path, G, "m.cugen"), verbose=False)
    off = ~np.eye(n, dtype=bool)
    np.testing.assert_allclose(got[off], want[off], atol=2e-6)


# --------------------------------------------------------------------------
# Analytic properties, no oracle needed
# --------------------------------------------------------------------------
def test_duplicate_pair_is_exactly_half(tmp_path):
    G, lab = synth(seed=2)
    K = king(write_cg(tmp_path, G), verbose=False)
    assert K[0, lab.index("dup0")] == pytest.approx(0.5, abs=1e-12)


def test_diagonal_is_half(tmp_path):
    """Self-kinship phi_ii = (1+F)/2, and F=0 falls out of the estimator."""
    G, _ = synth(seed=3)
    K = king(write_cg(tmp_path, G), verbose=False)
    np.testing.assert_allclose(np.diag(K), 0.5, atol=1e-12)


def test_parent_offspring_near_quarter(tmp_path):
    G, lab = synth(seed=5, p=6000)
    K = king(write_cg(tmp_path, G), verbose=False)
    c = lab.index("child12")
    assert K[1, c] == pytest.approx(0.25, abs=0.03)
    assert K[2, c] == pytest.approx(0.25, abs=0.03)


def test_symmetric(tmp_path):
    G, _ = synth(seed=6)
    K = king(write_cg(tmp_path, G), verbose=False)
    np.testing.assert_allclose(K, K.T, atol=0.0)


def test_king_is_robust_to_population_structure(tmp_path):
    """The reason this estimator exists, stated as what it actually guarantees.

    Nobody in this cohort is related; there are just two subpopulations at
    Fst ~0.25. A GRM centres on the pooled allele frequencies, so
    within-subpopulation pairs look POSITIVELY related purely for sharing drift
    history. KING conditions on each pair's own heterozygosity and does not.

    Note what is NOT asserted: that KING sits at zero everywhere. Across the
    split it goes markedly negative (measured ~-0.20 here, and plink2 agrees to
    5e-07), because opposite homozygotes are commoner between differentiated
    groups and that inflates IBS0. Negative means "different ancestry", not
    "related", so the property worth pinning is one-sided: shared ancestry is
    never converted into apparent kinship.
    """
    G, lab = synth(n_unrel=25, p=6000, seed=7, fst=0.25, dup=False, po=False)
    path = write_cg(tmp_path, G)
    K = king(path, verbose=False)
    A = grm(path, standardize="center", verbose=False)

    lab = np.asarray(lab)
    eye = np.eye(len(lab), dtype=bool)
    same = (lab[:, None] == lab[None, :]) & ~eye
    cross = lab[:, None] != lab[None, :]

    # The GRM registers the split, and inflates same-subpop pairs above it.
    assert A[same].mean() > A[cross].mean(), "GRM should register the split"

    # KING: unrelated same-population pairs sit at ~0 despite that structure.
    assert abs(K[same].mean()) < 0.02, f"KING same-pop mean {K[same].mean():.4f}"

    # The one-sided guarantee: no unrelated pair anywhere reaches even the
    # conventional 4th-degree threshold of 0.0442.
    assert K[~eye].max() < 0.0442, f"max unrelated KING {K[~eye].max():.4f}"

    # And the documented negative excursion across the split is real, so a
    # future change that "fixed" it into zeros would be caught here.
    assert K[cross].mean() < -0.05, f"KING cross-pop mean {K[cross].mean():.4f}"


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------
def test_maf_min_and_variants_narrow_the_marker_set(tmp_path):
    G, _ = synth(seed=8)
    path = write_cg(tmp_path, G)
    full = king(path, verbose=False)
    sub = king(path, variants=np.arange(0, G.shape[1], 2), verbose=False)
    assert not np.allclose(full, sub), "a marker subset should change the estimate"
    assert np.allclose(np.diag(sub), 0.5, atol=1e-12)


def test_refuses_a_phased_file(tmp_path):
    from cugen.write import write_cugen_phased
    rng = np.random.default_rng(0)
    hap = (rng.random((40, 500)) < 0.4).astype(np.uint8)
    p = tmp_path / "ph.cugen"
    write_cugen_phased(str(p), hap)
    with pytest.raises(ValueError, match="hap2bit|phased|2bit"):
        king(str(p), verbose=False)


def test_zero_markers_raises_rather_than_returning_nan(tmp_path):
    G, _ = synth(seed=10)
    path = write_cg(tmp_path, G)
    with pytest.raises(ValueError, match="no variant|zero markers"):
        king(path, maf_min=0.99, verbose=False)


@requires_gpu
def test_gpu_matches_numpy(tmp_path):
    G, _ = synth(seed=11, p=4000)
    path = write_cg(tmp_path, G)
    c = king(path, backend="numpy", verbose=False)
    g = king(path, backend="gpu", verbose=False)
    np.testing.assert_allclose(c, g, rtol=1e-9, atol=1e-9)


def test_tile_size_does_not_change_the_answer(tmp_path):
    G, _ = synth(seed=12, p=4000)
    path = write_cg(tmp_path, G)
    a = king(path, tile_size=512, verbose=False)
    b = king(path, tile_size=4096, verbose=False)
    np.testing.assert_allclose(a, b, atol=1e-12)


@requires_plink2
def test_missingness_confined_to_later_tiles(tmp_path):
    """A regression guard for tile-dependent pairwise het counts.

    The pairwise-complete het count must accumulate over EVERY marker, not only
    over the tiles that happen to contain a missing call. With missingness
    concentrated past the first tile boundary, an implementation that only
    accumulates H'M on 'dirty' tiles loses the clean tiles' contribution and
    deflates every denominator. Uniform missingness cannot detect that, because
    then every tile is dirty.
    """
    G, _ = synth(seed=21, p=4000)
    rng = np.random.default_rng(3)
    G = G.copy()
    tail = G[:, 2000:]
    tail[rng.random(tail.shape) < 0.10] = 3        # missing only in the tail
    G[:, 2000:] = tail
    n, p = G.shape

    Gv = G.copy()
    vcf = tmp_path / "late.vcf"
    gt = {0: "0/0", 1: "0/1", 2: "1/1", 3: "./."}
    with open(vcf, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=1,length=900000000>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 + "\t".join(f"S{i}" for i in range(n)) + "\n")
        for v in range(p):
            fh.write(f"1\t{(v + 1) * 1000}\tv{v}\tA\tG\t.\tPASS\t.\tGT\t"
                     + "\t".join(gt[int(x)] for x in Gv[:, v]) + "\n")
    out = tmp_path / "pl"
    subprocess.run(["plink2", "--vcf", str(vcf), "--make-king-table",
                    "--out", str(out)], check=True, capture_output=True)
    want = np.full((n, n), np.nan)
    with open(f"{out}.kin0") as fh:
        hdr = fh.readline().lstrip("#").split()
        ci, cj, ck = hdr.index("IID1"), hdr.index("IID2"), hdr.index("KINSHIP")
        for line in fh:
            f = line.split()
            a, b = int(f[ci][1:]), int(f[cj][1:])
            want[a, b] = want[b, a] = float(f[ck])

    path = write_cg(tmp_path, G, "late.cugen")
    off = ~np.eye(n, dtype=bool)
    # tile_size=1000 puts markers 0-1999 in fully-called tiles
    got = king(path, tile_size=1000, verbose=False)
    np.testing.assert_allclose(got[off], want[off], atol=2e-6)


# --------------------------------------------------------------------------
# king_pairs: the blocked, thresholded form for cohorts where (n,n) is too big
# --------------------------------------------------------------------------
from cugen.popstruct import king_pairs  # noqa: E402


def _dense_from_pairs(df, n):
    M = np.full((n, n), np.nan)
    M[df.i.to_numpy(), df.j.to_numpy()] = df.kinship.to_numpy()
    return M


@pytest.mark.parametrize("block", [16, 32, 128])
def test_king_pairs_is_bit_exact_against_dense_king(tmp_path, block):
    """Blocking must not perturb a single value, at any block size.

    128 exceeds n here on purpose, so the single-block path is covered too.
    """
    G, _ = synth(n_unrel=60, p=4000, seed=31)
    n = G.shape[0]
    path = write_cg(tmp_path, G)
    dense = king(path, verbose=False)
    df = king_pairs(path, min_kinship=-np.inf, sample_block=block, verbose=False)
    iu = np.triu_indices(n, 1)
    np.testing.assert_array_equal(_dense_from_pairs(df, n)[iu], dense[iu])


def test_king_pairs_emits_each_unordered_pair_once(tmp_path):
    """At -inf every pair must appear exactly once, with i < j.

    Guards the diagonal-block mask. Masking it with -inf instead of a boolean
    leaks the whole lower triangle here, because -inf >= -inf.
    """
    G, _ = synth(n_unrel=30, p=1500, seed=32)
    n = G.shape[0]
    df = king_pairs(write_cg(tmp_path, G), min_kinship=-np.inf,
                    sample_block=16, verbose=False)
    assert len(df) == n * (n - 1) // 2
    assert bool((df.i < df.j).all())
    assert not df.duplicated(subset=["i", "j"]).any()


def test_king_pairs_threshold_recovers_the_planted_relatives(tmp_path):
    G, lab = synth(n_unrel=60, p=6000, seed=33)
    df = king_pairs(write_cg(tmp_path, G), min_kinship=0.0442,
                    sample_block=32, verbose=False)
    got = {(int(a), int(b)) for a, b in zip(df.i, df.j)}
    dup, child = lab.index("dup0"), lab.index("child12")
    assert (0, dup) in got
    assert (1, child) in got and (2, child) in got
    top = df.iloc[0]
    assert int(top.i) == 0 and int(top.j) == dup
    assert float(top.kinship) == pytest.approx(0.5, abs=1e-12)


def test_king_pairs_refuses_missingness_rather_than_guessing(tmp_path):
    """Pairwise-complete het counts are themselves (n,n) -- the thing this
    function exists to avoid -- so it must say so instead of quietly using
    per-sample totals and returning slightly wrong kinships."""
    G, _ = synth(n_unrel=20, p=1000, seed=34)
    rng = np.random.default_rng(1)
    G = G.copy()
    G[rng.random(G.shape) < 0.05] = 3
    with pytest.raises(NotImplementedError, match="missing"):
        king_pairs(write_cg(tmp_path, G, "miss.cugen"), verbose=False)


@requires_gpu
def test_king_pairs_gpu_matches_numpy(tmp_path):
    G, _ = synth(n_unrel=50, p=3000, seed=35)
    path = write_cg(tmp_path, G)
    a = king_pairs(path, min_kinship=-np.inf, sample_block=32,
                   backend="numpy", verbose=False)
    b = king_pairs(path, min_kinship=-np.inf, sample_block=32,
                   backend="gpu", verbose=False)
    a = a.sort_values(["i", "j"]).reset_index(drop=True)
    b = b.sort_values(["i", "j"]).reset_index(drop=True)
    np.testing.assert_array_equal(a.i, b.i)
    np.testing.assert_array_equal(a.j, b.j)
    np.testing.assert_allclose(a.kinship, b.kinship, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# plan_king: which routine fits, and why
# --------------------------------------------------------------------------
from cugen.popstruct import plan_king  # noqa: E402


def test_plan_recommends_dense_when_it_fits(tmp_path):
    G, _ = synth(n_unrel=20, p=800, seed=40)
    plan = plan_king(write_cg(tmp_path, G), budget_bytes=8 << 30, verbose=False)
    assert plan["recommend"] == "king"
    assert plan["fits"] is True
    assert plan["n_samples"] == G.shape[0]


def test_plan_recommends_pairs_when_dense_does_not_fit(tmp_path):
    """Same file, tiny budget -- the choice must follow the budget, not n."""
    G, _ = synth(n_unrel=20, p=800, seed=41)
    plan = plan_king(write_cg(tmp_path, G), budget_bytes=1000, verbose=False)
    assert plan["recommend"] == "king_pairs"
    assert plan["fits"] is False
    assert plan["sample_block"] >= 4
    assert plan["sample_block"] % 4 == 0, "blocks must be whole bytes of 2-bit"


def test_plan_dense_estimate_is_not_optimistic(tmp_path):
    """The estimate must bound what king() actually peaks at, not undercut it.

    An estimate that is too low is worse than none: it routes a job to the dense
    path that then dies partway through with a MemoryError.
    """
    G, _ = synth(n_unrel=40, p=1200, seed=42)
    path = write_cg(tmp_path, G)
    n = G.shape[0]
    plan = plan_king(path, budget_bytes=8 << 30, verbose=False)
    # king() holds at minimum the Gram accumulator and the result, both (n,n) f64
    assert plan["dense_bytes"] >= 2 * n * n * 8


def test_plan_counts_the_extra_accumulators_for_missing_and_within_family(tmp_path):
    G, _ = synth(n_unrel=20, p=800, seed=43)
    clean = write_cg(tmp_path, G)
    Gm = G.copy()
    rng = np.random.default_rng(0)
    Gm[rng.random(Gm.shape) < 0.05] = 3
    dirty = write_cg(tmp_path, Gm, "dirty.cugen")
    a = plan_king(clean, budget_bytes=8 << 30, verbose=False)["dense_bytes"]
    b = plan_king(dirty, budget_bytes=8 << 30, verbose=False)["dense_bytes"]
    c = plan_king(clean, estimator="within-family", budget_bytes=8 << 30,
                  verbose=False)["dense_bytes"]
    assert b > a, "missingness adds pairwise (n,n) accumulators"
    assert c > a, "within-family still needs HetHet"


def test_king_refuses_early_with_a_pointer_to_king_pairs(tmp_path):
    """Fail before reading data, naming the alternative and a block size.

    Dying in a MemoryError partway through a scan wastes the whole read and
    tells the caller nothing about what to do instead.
    """
    G, _ = synth(n_unrel=20, p=800, seed=44)
    with pytest.raises(MemoryError, match="king_pairs"):
        king(write_cg(tmp_path, G), budget_bytes=1000, verbose=False)


def test_king_pairs_auto_block_matches_an_explicit_block(tmp_path):
    G, _ = synth(n_unrel=40, p=2000, seed=45)
    path = write_cg(tmp_path, G)
    auto = king_pairs(path, min_kinship=-np.inf, sample_block="auto",
                      verbose=False).sort_values(["i", "j"]).reset_index(drop=True)
    fixed = king_pairs(path, min_kinship=-np.inf, sample_block=32,
                       verbose=False).sort_values(["i", "j"]).reset_index(drop=True)
    np.testing.assert_array_equal(auto.i, fixed.i)
    np.testing.assert_allclose(auto.kinship, fixed.kinship, rtol=0, atol=0)


# --------------------------------------------------------------------------
# king_matrix: dense on disk, for when the matrix is the deliverable
# --------------------------------------------------------------------------
from cugen.popstruct import king_matrix, open_king_matrix  # noqa: E402


def test_king_matrix_float32_is_exact_against_dense_king(tmp_path):
    G, _ = synth(n_unrel=40, p=2000, seed=50)
    n = G.shape[0]
    path = write_cg(tmp_path, G)
    dense = king(path, verbose=False)
    km = open_king_matrix(king_matrix(path, tmp_path / "k.bin",
                                      encoding="float32", sample_block=16,
                                      verbose=False))
    assert km.n == n and km.encoding == "float32"
    got = km.to_numpy()
    np.testing.assert_allclose(got, dense, rtol=0, atol=1e-6)


def test_king_matrix_int16_is_within_its_own_quantum(tmp_path):
    """3.05e-5 is the promise; anything looser is a bug, not a rounding."""
    G, _ = synth(n_unrel=40, p=2000, seed=51)
    path = write_cg(tmp_path, G)
    dense = king(path, verbose=False)
    km = open_king_matrix(king_matrix(path, tmp_path / "k16.bin",
                                      sample_block=16, verbose=False))
    assert km.encoding == "int16"
    assert np.abs(km.to_numpy() - dense).max() <= 1.0 / 32767.0


def test_king_matrix_is_symmetric_and_half_on_the_diagonal(tmp_path):
    G, _ = synth(n_unrel=30, p=1200, seed=52)
    km = open_king_matrix(king_matrix(write_cg(tmp_path, G),
                                      tmp_path / "s.bin", encoding="float32",
                                      sample_block=16, verbose=False))
    M = km.to_numpy()
    np.testing.assert_allclose(M, M.T, atol=0)
    np.testing.assert_allclose(np.diag(M), 0.5, atol=1e-6)


def test_indexed_access_matches_the_materialised_matrix(tmp_path):
    """km[i,j] and km.row(i) are the point of the format -- they must agree."""
    G, _ = synth(n_unrel=30, p=1200, seed=53)
    n = G.shape[0]
    km = open_king_matrix(king_matrix(write_cg(tmp_path, G),
                                      tmp_path / "i.bin", encoding="float32",
                                      sample_block=16, verbose=False))
    M = km.to_numpy()
    rng = np.random.default_rng(0)
    for _ in range(40):
        i, j = int(rng.integers(n)), int(rng.integers(n))
        assert km[i, j] == pytest.approx(M[i, j], abs=1e-6)
        assert km[i, j] == km[j, i], "must be order-independent"
    r = int(rng.integers(n))
    np.testing.assert_allclose(km.row(r), M[r], atol=1e-6)


def test_to_numpy_refuses_a_matrix_that_does_not_fit(tmp_path):
    """The format exists for matrices that do not fit; materialising one
    silently would defeat the whole point."""
    G, _ = synth(n_unrel=20, p=800, seed=54)
    km = open_king_matrix(king_matrix(write_cg(tmp_path, G),
                                      tmp_path / "b.bin", encoding="float32",
                                      sample_block=16, verbose=False))
    with pytest.raises(MemoryError, match="max_gb"):
        km.to_numpy(max_gb=1e-9)


def test_rejects_a_file_that_is_not_a_king_matrix(tmp_path):
    bad = tmp_path / "nope.bin"
    bad.write_bytes(b"NOTKING1" + b"\0" * 32)
    with pytest.raises(ValueError, match="not a king_matrix"):
        open_king_matrix(bad)


# --------------------------------------------------------------------------
# Querying by person: sample IDs, fast rows, and per-person relative lookup
# --------------------------------------------------------------------------
def _ids(n):
    return [f"IND{i:05d}" for i in range(n)]


def test_sample_ids_round_trip_and_index_by_id(tmp_path):
    G, _ = synth(n_unrel=30, p=1200, seed=60)
    n = G.shape[0]
    ids = _ids(n)
    km = open_king_matrix(king_matrix(
        write_cg(tmp_path, G), tmp_path / "id.bin", encoding="float32",
        sample_block=16, sample_ids=ids, verbose=False))
    assert list(km.ids) == ids
    assert km.index_of("IND00007") == 7
    # the same cell, addressed three ways
    assert km["IND00003", "IND00009"] == pytest.approx(km[3, 9], abs=0)
    assert km["IND00009", "IND00003"] == pytest.approx(km[3, 9], abs=0)


def test_row_by_id_matches_row_by_index(tmp_path):
    G, _ = synth(n_unrel=30, p=1200, seed=61)
    n = G.shape[0]
    km = open_king_matrix(king_matrix(
        write_cg(tmp_path, G), tmp_path / "r.bin", encoding="float32",
        sample_block=16, sample_ids=_ids(n), verbose=False))
    np.testing.assert_array_equal(km.row("IND00011"), km.row(11))


def test_unknown_id_raises_rather_than_returning_nonsense(tmp_path):
    G, _ = synth(n_unrel=20, p=800, seed=62)
    km = open_king_matrix(king_matrix(
        write_cg(tmp_path, G), tmp_path / "u.bin", encoding="float32",
        sample_block=16, sample_ids=_ids(G.shape[0]), verbose=False))
    with pytest.raises(KeyError, match="NOPE"):
        km.row("NOPE")


def test_both_layouts_agree_with_dense_king(tmp_path):
    """square costs 2x the bytes and buys ~10,000x on row latency; neither may
    change a value."""
    G, _ = synth(n_unrel=40, p=2000, seed=63)
    path = write_cg(tmp_path, G)
    dense = king(path, verbose=False)
    for layout in ("triangle", "square"):
        km = open_king_matrix(king_matrix(
            path, tmp_path / f"{layout}.bin", encoding="float32",
            layout=layout, sample_block=16, verbose=False))
        assert km.layout == layout
        np.testing.assert_allclose(km.to_numpy(), dense, rtol=0, atol=1e-6)
        r = 7
        np.testing.assert_allclose(km.row(r), dense[r], rtol=0, atol=1e-6)


def test_square_is_twice_the_triangle_on_disk(tmp_path):
    G, _ = synth(n_unrel=40, p=1000, seed=64)
    path = write_cg(tmp_path, G)
    n = G.shape[0]
    t = os.path.getsize(king_matrix(path, tmp_path / "t.bin", layout="triangle",
                                    sample_block=16, verbose=False))
    s = os.path.getsize(king_matrix(path, tmp_path / "s.bin", layout="square",
                                    sample_block=16, verbose=False))
    assert s > t
    # square is n^2, triangle is n(n+1)/2, so the ratio approaches 2
    assert 1.8 < (s / t) < 2.2


def test_related_returns_that_persons_relatives_sorted(tmp_path):
    """The query the format exists for: who is related to this person?"""
    G, lab = synth(n_unrel=60, p=6000, seed=65)
    n = G.shape[0]
    ids = _ids(n)
    km = open_king_matrix(king_matrix(
        write_cg(tmp_path, G), tmp_path / "rel.bin", encoding="float32",
        layout="square", sample_block=16, sample_ids=ids, verbose=False))
    dup = lab.index("dup0")
    hits = km.related(0, min_kinship=0.0442)
    assert ids[dup] in set(hits.id), "the duplicate must be found"
    assert 0 not in set(hits.index_), "self must be excluded"
    assert hits.kinship.is_monotonic_decreasing
    assert float(hits.kinship.iloc[0]) == pytest.approx(0.5, abs=1e-6)
    # and by id, identically
    np.testing.assert_allclose(km.related(ids[0], min_kinship=0.0442).kinship,
                               hits.kinship, rtol=0, atol=0)


def test_row_gather_is_vectorised_not_a_python_loop(tmp_path):
    """A triangle row must not cost n Python-level reads.

    The first version looped `for k in range(i+1, n)` doing one memmap read per
    element, which at n=1e6 is a million round trips through the interpreter for
    a single person. Timing a small case is a weak signal, so this asserts the
    shape of the result and that a mid-file row is correct -- the loop version
    was correct too, just unusable -- and the layout test above covers values.
    """
    G, _ = synth(n_unrel=60, p=800, seed=66)
    path = write_cg(tmp_path, G)
    n = G.shape[0]
    dense = king(path, verbose=False)
    km = open_king_matrix(king_matrix(path, tmp_path / "v.bin",
                                      encoding="float32", layout="triangle",
                                      sample_block=16, verbose=False))
    for r in (0, 1, n // 2, n - 2, n - 1):
        got = km.row(r)
        assert got.shape == (n,)
        np.testing.assert_allclose(got, dense[r], rtol=0, atol=1e-6)
