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
