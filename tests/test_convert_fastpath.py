"""The int8 fast path must match the float64 path it replaces, exactly.

pgen2cugen promoted every variant to float64 -- 4.09 MiB at n=535,662 -- and
then built roughly a dozen more arrays that size between the missing policy,
variant_stats and the 2-bit packing. Measured at 2.89 ms per variant, which is
2.04 h for chr1's 2,537,153 SNVs before any I/O.

Dosages are 0/1/2 plus a no-call code. None of that needs float64, and the
statistics follow from counts of 0/1/2 rather than from the vector. This file
pins the replacement to the behaviour of the code it replaces, because a
converter that is fast and subtly different is worse than a slow one.
"""

import numpy as np
import pytest

from cugen.convert import _MISSING_CODE, _apply_missing_policy, _convert_codes
from cugen.write import pack_2bit, variant_stats


def _reference(codes, policy):
    """The pre-optimisation path, kept here as the oracle."""
    d = codes.astype(np.float64)
    d[d < 0] = _MISSING_CODE
    dd, n_miss = _apply_missing_policy(d, policy)
    if dd is None:
        return None, n_miss
    mu, sxx, maf, has_missing = variant_stats(dd)
    g = np.asarray(dd, dtype=np.float64)
    u = np.where(np.isfinite(g), np.rint(np.nan_to_num(g, nan=3.0)), 3.0)
    u = np.clip(u, 0, 3).astype(np.uint8)
    return (pack_2bit(u).tobytes(), mu, sxx, maf, has_missing), n_miss


EDGE_CASES = [
    ("no missing", np.array([0, 1, 2, 0, 1, 2, 2, 1], np.int8)),
    ("all missing", np.array([-9, -9, -9, -9], np.int8)),
    ("single observed call", np.array([-9, -9, 2, -9], np.int8)),
    # np.rint is banker's rounding: 0.5 -> 0 and 1.5 -> 2, not 1 and 2.
    ("mean exactly 0.5", np.array([0, 1, -9, -9], np.int8)),
    ("mean exactly 1.5", np.array([1, 2, -9, -9], np.int8)),
    ("mean 2.0", np.array([2, 2, 2, -9], np.int8)),
    ("monomorphic ref", np.zeros(1000, np.int8)),
    ("monomorphic alt", np.full(1000, 2, np.int8)),
    ("length not a multiple of 4", np.array([0, 1, 2, 1, 0], np.int8)),
]


@pytest.mark.parametrize("policy", ["mean", "ref", "keep"])
@pytest.mark.parametrize("name,codes", EDGE_CASES, ids=[c[0] for c in EDGE_CASES])
def test_fastpath_packs_identical_bytes(name, codes, policy):
    ref, ref_miss = _reference(codes.copy(), policy)
    got = _convert_codes(codes.copy(), policy)
    if ref is None:
        assert got[0] is None
        return
    assert got[0] == ref[0], "packed bytes differ"
    assert got[1] == ref[1], "mu differs"
    assert got[3] == ref[3], "maf differs"
    assert got[4] == ref[4], "has_missing differs"
    assert got[5] == ref_miss, "n_missing differs"


@pytest.mark.parametrize("policy", ["mean", "ref", "keep"])
def test_fastpath_matches_on_random_variants(policy):
    rng = np.random.default_rng(7)
    for _ in range(30):
        n = int(rng.integers(1, 3000))
        pm = float(rng.choice([0.0, 0.001, 0.02, 0.3, 0.9]))
        codes = rng.choice(
            np.array([0, 1, 2, -9], np.int8), size=n,
            p=[(1 - pm) * 0.6, (1 - pm) * 0.3, (1 - pm) * 0.1, pm],
        ).astype(np.int8)
        ref, ref_miss = _reference(codes.copy(), policy)
        got = _convert_codes(codes.copy(), policy)
        if ref is None:
            assert got[0] is None
            continue
        assert got[0] == ref[0]
        assert got[1] == ref[1]
        assert got[3] == ref[3]
        assert got[4] == ref[4]
        assert got[5] == ref_miss


def test_sxx_is_identical_once_stored_as_float32():
    """sxx uses the closed form s2 - s1^2/cnt instead of sum((x-mu)^2).

    In float64 the two differ in the last few ulps -- the closed form carries
    two roundings where the reference carries about n, so the closed form is the
    MORE accurate one. But CugenWriter stores sxx as float32, and that cast
    erases the difference entirely: the written file is bit-identical, not
    merely close. This test pins both halves, because "close enough" and
    "identical" are different promises and only one of them is true here."""
    rng = np.random.default_rng(11)
    worst64 = 0.0
    n_float32_differ = 0
    for _ in range(200):
        size = int(rng.integers(50, 20000))
        codes = rng.choice(np.array([0, 1, 2, -9], np.int8), size=size,
                           p=[0.6, 0.25, 0.14, 0.01]).astype(np.int8)
        ref, _ = _reference(codes.copy(), "mean")
        got = _convert_codes(codes.copy(), "mean")
        worst64 = max(worst64, abs(ref[2] - got[2]) / max(abs(ref[2]), 1e-12))
        if np.float32(ref[2]) != np.float32(got[2]):
            n_float32_differ += 1
    assert worst64 < 1e-12, f"float64 sxx drifted by {worst64:.2e} relative"
    assert n_float32_differ == 0, (
        f"{n_float32_differ} variants differ in the float32 actually stored")


def test_negative_codes_other_than_minus_nine_are_still_no_calls():
    """The fast path collapses no-calls with a single minimum() on the uint8
    view: every negative int8 lands >= 128 unsigned, so it clamps to 3. That
    holds for any negative sentinel, not just pgenlib's -9."""
    for sentinel in (-1, -9, -128):
        codes = np.array([0, 1, 2, sentinel], np.int8)
        got = _convert_codes(codes.copy(), "keep")
        assert got[5] == 1, f"sentinel {sentinel} not counted as missing"
        assert got[4] is True


def test_add_variant_packed_writes_the_same_file_as_add_variant(tmp_path):
    """The fast path packs and computes stats itself, so the writer needs a door
    that takes them directly. That door must produce a byte-identical file --
    otherwise the optimisation is a format change wearing a speed costume."""
    from cugen.io import read_cugen_header
    from cugen.write import CugenWriter, ENCODING_2BIT

    rng = np.random.default_rng(3)
    n = 1013
    variants = [
        rng.choice(np.array([0, 1, 2, -9], np.int8), size=n,
                   p=[0.6, 0.25, 0.14, 0.01]).astype(np.int8)
        for _ in range(25)
    ]

    slow = tmp_path / "slow.cugen"
    with CugenWriter(slow, n, len(variants), ENCODING_2BIT) as w:
        for k, codes in enumerate(variants):
            d = codes.astype(np.float64)
            d[d < 0] = _MISSING_CODE
            dd, _ = _apply_missing_policy(d, "mean")
            w.add_variant(k, dd)

    fast = tmp_path / "fast.cugen"
    with CugenWriter(fast, n, len(variants), ENCODING_2BIT) as w:
        for k, codes in enumerate(variants):
            packed, mu, sxx, maf, hm, _ = _convert_codes(codes, "mean")
            w.add_variant_packed(k, packed, mu, sxx, maf, hm)

    assert slow.stat().st_size == fast.stat().st_size

    ha, hb = read_cugen_header(str(slow)), read_cugen_header(str(fast))
    for key in ("n_samples", "n_variants", "bytes_per_variant", "encoding",
                "has_missing", "flags", "phased"):
        assert ha[key] == hb[key], f"header {key} differs: {ha[key]} vs {hb[key]}"

    # Compare through the reader, not by byte offset: the genotypes are what
    # must be identical, and this does not depend on the header layout.
    from cugen.io import CugenReader

    ra, rb = CugenReader(str(slow)), CugenReader(str(fast))
    try:
        assert np.array_equal(ra.read_to_numpy(0, len(variants)),
                              rb.read_to_numpy(0, len(variants))), "genotypes differ"
        mua, sxxa, mafa = ra.get_stats(0, len(variants))
        mub, sxxb, mafb = rb.get_stats(0, len(variants))
        assert np.array_equal(mua, mub), "mu differs"
        assert np.array_equal(mafa, mafb), "maf differs"
        # sxx is stored as float32, which erases the float64 difference.
        assert np.array_equal(sxxa, sxxb), "sxx differs"
    finally:
        ra.close(); rb.close()
