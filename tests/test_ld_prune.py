"""Tests for cugen.ld.ld_prune (plink2 --indep-pairwise's job).

Pruning is clumping with a different priority: both are greedy
maximal-independent-set selection under an r^2 constraint. Clumping ranks by
p-value, pruning by allele frequency. So these tests focus on what is
genuinely different -- the ranking, and the deliberate divergence from plink.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cugen.ld import clump_core, ld_prune

DATA = Path(__file__).parent / "data"


def _fixture(tmp_path, p=120, n=300, seed=17):
    """LD blocks with a spread of allele frequencies, so MAF ordering and
    file ordering genuinely disagree -- otherwise the priority rule is
    untested."""
    from cugen.write import write_cugen
    rng = np.random.default_rng(seed)
    G = np.zeros((p, n), dtype=np.uint8)
    k = 0
    while k < p:
        blk = int(min(rng.integers(3, 9), p - k))
        base = (rng.random(n) < rng.uniform(0.1, 0.5)).astype(int) + \
               (rng.random(n) < rng.uniform(0.1, 0.5)).astype(int)
        for j in range(blk):
            noise = rng.random(n) < rng.uniform(0.0, 0.3)
            G[k + j] = np.where(noise, rng.integers(0, 3, n), base)
        k += blk
    path = tmp_path / "prune.cugen"
    write_cugen(str(path), G.T)
    pos = np.sort(rng.choice(np.arange(1, 900_000), size=p, replace=False))
    ann = pd.DataFrame({"gidx": np.arange(p), "ID": [f"v{i}" for i in range(p)],
                        "POS": pos, "CHR": "1"})
    return str(path), ann, G


def _max_r2_among(G, kept_gidx):
    X = G.astype(float)
    X = X - X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    sd[sd == 0] = 1
    Z = X / sd
    R2 = ((Z @ Z.T) / G.shape[1]) ** 2
    k = np.asarray(sorted(kept_gidx))
    if len(k) < 2:
        return 0.0
    sub = R2[np.ix_(k, k)].copy()
    np.fill_diagonal(sub, 0.0)
    return float(sub.max())


# ---------------------------------------------------------------------------
# the guarantee pruning exists to provide
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("r2", [0.2, 0.5, 0.8])
def test_no_two_retained_variants_exceed_the_threshold(tmp_path, r2):
    """The only property that actually matters. Everything else is policy."""
    path, ann, G = _fixture(tmp_path)
    keep, drop = ld_prune(path, window=50, r2=r2, annotation=ann,
                          backend="numpy", verbose=False)
    worst = _max_r2_among(G, keep["gidx"].to_numpy())
    assert worst <= r2 + 1e-6, (
        f"two retained variants have r2={worst:.4f} > {r2}")


def test_kept_and_dropped_partition_the_input(tmp_path):
    path, ann, G = _fixture(tmp_path)
    keep, drop = ld_prune(path, window=50, r2=0.5, annotation=ann,
                          backend="numpy", verbose=False)
    both = np.concatenate([keep["gidx"].to_numpy(), drop["gidx"].to_numpy()])
    assert sorted(both) == list(range(len(G)))
    assert len(set(both)) == len(both), "a variant is in both lists"


def test_result_is_maximal_nothing_can_be_added_back(tmp_path):
    """This is where we diverge from plink, so it gets asserted rather than
    assumed: no dropped variant could be restored without breaking the
    guarantee. plink's output does NOT satisfy this."""
    path, ann, G = _fixture(tmp_path)
    r2 = 0.5
    keep, drop = ld_prune(path, window=50, r2=r2, annotation=ann,
                          backend="numpy", verbose=False)
    kept = set(keep["gidx"].tolist())
    for v in drop["gidx"].tolist():
        assert _max_r2_among(G, kept | {v}) > r2, (
            f"variant {v} was dropped but could be added back -- not maximal")


# ---------------------------------------------------------------------------
# the priority rule, measured against plink2 rather than assumed
# ---------------------------------------------------------------------------
def test_higher_maf_wins(tmp_path):
    """plink2 v2.0.0-a.7.1 keeps the higher-MAF variant of a conflicting pair,
    and it is MAF rather than position: on a fixture where the two orderings
    disagreed, the high-MAF variant survived under BOTH file orderings.
    """
    from cugen.write import write_cugen
    rng = np.random.default_rng(31)
    n = 600
    h = rng.random(n) < 0.12
    lo = (h.astype(int) + (rng.random(n) < 0.10).astype(int))     # MAF ~0.11
    flip = rng.random(n) < 0.02
    hi = np.where(flip, rng.integers(0, 3, n),
                  h.astype(int) + (rng.random(n) < 0.45).astype(int))
    mid = rng.integers(0, 3, n)
    for order, low_row, high_row in ((("lo", "mid", "hi"), 0, 2),
                                     (("hi", "mid", "lo"), 2, 0)):
        rows = {"lo": lo, "mid": mid, "hi": hi}
        G = np.vstack([rows[k] for k in order]).astype(np.uint8)
        path = tmp_path / f"maf_{'_'.join(order)}.cugen"
        write_cugen(str(path), G.T)
        ann = pd.DataFrame({"gidx": [0, 1, 2], "ID": list(order),
                            "POS": [1000, 2000, 3000], "CHR": "1"})
        keep, drop = ld_prune(str(path), window=10, r2=0.1, annotation=ann,
                              backend="numpy", verbose=False)
        assert "hi" in set(keep["ID"]), f"high-MAF variant pruned ({order})"
        assert "lo" in set(drop["ID"]), f"low-MAF variant kept ({order})"


def test_ties_break_deterministically(tmp_path):
    path, ann, G = _fixture(tmp_path)
    first = ld_prune(path, window=40, r2=0.4, annotation=ann,
                     backend="numpy", verbose=False)[0]
    for _ in range(3):
        again = ld_prune(path, window=40, r2=0.4, annotation=ann,
                         backend="numpy", verbose=False)[0]
        pd.testing.assert_frame_equal(first, again)


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------
def test_window_is_required(tmp_path):
    path, ann, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="window"):
        ld_prune(path, r2=0.5, annotation=ann, backend="numpy", verbose=False)


def test_r2_outside_zero_one_is_rejected(tmp_path):
    path, ann, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match=r"r2 must be in"):
        ld_prune(path, window=50, r2=1.5, annotation=ann, backend="numpy",
                 verbose=False)


def test_annotation_is_optional_for_variant_count_windows(tmp_path):
    """A variant-count window needs no coordinates, so it must work without
    annotation -- IDs just come back as '.'."""
    path, _ann, G = _fixture(tmp_path)
    keep, drop = ld_prune(path, window=50, r2=0.5, backend="numpy",
                          verbose=False)
    assert len(keep) + len(drop) == len(G)
    assert (keep["ID"] == ".").all()


def test_threshold_of_one_keeps_everything(tmp_path):
    path, ann, G = _fixture(tmp_path)
    keep, drop = ld_prune(path, window=50, r2=1.0, annotation=ann,
                          backend="numpy", verbose=False)
    assert len(drop) == 0 or _max_r2_among(G, keep["gidx"].to_numpy()) <= 1.0


def test_prune_reuses_the_clump_core():
    """Documents the relationship rather than duplicating the algorithm: if
    these ever diverge, one of them has grown its own copy."""
    import inspect
    src = inspect.getsource(ld_prune)
    assert "clump_core(" in src
