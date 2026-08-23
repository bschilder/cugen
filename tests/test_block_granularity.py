"""The block-pair cap is a write-vs-query trade; pin what it is FOR.

Asserting `MAX_BLOCK_PAIRS == 262144` would be a tautology. The property that
matters is that a point lookup stays bounded: the constant exists because an
uncapped block held 2.5 M pairs and variant() cost 360 ms, and raising the cap
to 4 M would undo that (measured 23.83 ms against 1.68 ms at 65,536 on real
chr22 -- see the table in ldio.py).
"""
import numpy as np
import pytest

from cugen import ldio


def _write(tmp_path, n_row_variants, partners, cap):
    """A shard with a known, uniform degree per row variant."""
    i = np.repeat(np.arange(n_row_variants, dtype=np.int64), partners)
    j = np.tile(np.arange(partners, dtype=np.int64), n_row_variants) \
        + i + 1
    # r concentrated in the TOP tier on purpose. Spread uniformly over [-1, 1]
    # the 200,000 pairs split ~40,000 per tier, under the 65,536 cap, so the
    # cap never engaged and the sweep tests compared two identical files. The
    # fixture has to make the cap bind for a test of the cap to mean anything.
    rng = np.random.default_rng(5)
    r = rng.uniform(0.9, 1.0, i.size)
    p = tmp_path / f"cap{cap}.cugenld"
    w = ldio.LDShardWriter(str(p), max_block_pairs=cap,
                           params={"n_obs": 1000})
    w.append(i, j, r, presorted=True)
    w.close()
    return p, i.size


def test_the_cap_actually_bounds_block_size(tmp_path):
    """Every block must respect the cap, or the bound is decorative."""
    p, n = _write(tmp_path, 400, 500, ldio.MAX_BLOCK_PAIRS)
    rd = ldio.read_ld(str(p))
    assert n == 200_000
    biggest = max(b["n"] for b in rd.blocks)
    assert biggest <= ldio.MAX_BLOCK_PAIRS, (
        f"a block holds {biggest:,} pairs against a cap of "
        f"{ldio.MAX_BLOCK_PAIRS:,}")


def test_a_point_lookup_decompresses_a_bounded_number_of_pairs(tmp_path):
    """The real invariant: variant() cost is bounded by the cap, not by file size.

    A lookup pays for whole blocks because zstd frames are not seekable, so the
    quantity to bound is pairs-decompressed-per-lookup.
    """
    p, _ = _write(tmp_path, 400, 500, ldio.MAX_BLOCK_PAIRS)
    rd = ldio.read_ld(str(p))
    rd.reset_counters()
    rd.variant(120)
    pairs_touched = sum(b["n"] for b in rd.blocks
                        if 120 in b["row_variants"])
    # one variant's partners live in at most one block per r^2 tier
    assert pairs_touched <= ldio.MAX_BLOCK_PAIRS * len(ldio.DEFAULT_TIERS)
    assert rd.blocks_read <= len(ldio.DEFAULT_TIERS), (
        f"a single lookup read {rd.blocks_read} blocks; a variant should live "
        f"in at most one block per tier")


def test_a_larger_cap_really_does_make_blocks_coarser(tmp_path):
    """Guard the direction: the knob must still do something.

    If this stopped holding, the cap would have become inert and the measured
    trade-off in ldio.py would no longer describe the code.
    """
    small, n = _write(tmp_path, 400, 500, 1 << 16)
    large, _ = _write(tmp_path, 400, 500, 1 << 20)
    assert n > (1 << 16), "fixture does not exceed the small cap; test is vacuous"
    assert n < (1 << 20), "fixture exceeds the large cap too; nothing to compare"
    n_small = len(ldio.read_ld(str(small)).blocks)
    n_large = len(ldio.read_ld(str(large)).blocks)
    assert n_large < n_small, (
        f"cap 1<<20 produced {n_large} blocks, cap 1<<16 produced {n_small}; "
        f"a larger cap must produce fewer, coarser blocks")


def test_fewer_blocks_means_fewer_bytes(tmp_path):
    """Every block carries a footer entry, so over-blocking inflates size.

    This is the mirror of the over-sharding finding: bigger blocks were
    measured both faster AND smaller (2.011 -> 1.957 B/pair).
    """
    import os
    small, n = _write(tmp_path, 400, 500, 1 << 16)
    large, _ = _write(tmp_path, 400, 500, 1 << 20)
    assert os.path.getsize(large) < os.path.getsize(small)
