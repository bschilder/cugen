"""pgen2cugen reads one variant per pgenlib call; batching must not change output.

pgen stores runs of variants as diffs against a reference variant, so a
per-variant `read()` defeats that compression while `read_range()` exploits it.
The optimisation is only admissible if it is bit-identical, and the case that
breaks a naive implementation is the one AoU actually uses: a SPARSE
variant_idx, because the ACAF callset is under snpindel/ and only its biallelic
SNVs are converted. A `read_range(first, last)` over a sparse index would
silently convert the indels sitting between the wanted rows.
"""

from __future__ import annotations

import numpy as np
import pytest

pgenlib = pytest.importorskip("pgenlib", reason="pgen2cugen needs pgenlib")

from cugen.convert import pgen2cugen          # noqa: E402


def _write_pgen(tmp_path, G):
    G = np.ascontiguousarray(np.asarray(G, dtype=np.int8))
    n_var, n_samp = G.shape
    prefix = tmp_path / "t"
    w = pgenlib.PgenWriter(str(f"{prefix}.pgen").encode(), n_samp, n_var, False)
    for v in range(n_var):
        w.append_biallelic(np.ascontiguousarray(G[v]))
    w.close()
    with open(f"{prefix}.psam", "w") as f:
        f.write("#IID\tSEX\n")
        for i in range(n_samp):
            f.write(f"s{i}\t1\n")
    return prefix


def _panel(n_var=61, n_samp=37, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=(n_var, n_samp)).astype(np.int8)


def _convert(prefix, tmp_path, tag, **kw):
    out = tmp_path / f"{tag}.cugen"
    pgen2cugen(f"{prefix}.pgen", str(out), verbose=False, **kw)
    return out.read_bytes()


@pytest.mark.parametrize("batch", [2, 7, 64, 4096])
def test_batched_read_is_byte_identical_over_all_variants(tmp_path, batch):
    prefix = _write_pgen(tmp_path, _panel())
    one = _convert(prefix, tmp_path, "one", read_batch=1)
    many = _convert(prefix, tmp_path, f"many{batch}", read_batch=batch)
    assert one == many


@pytest.mark.parametrize("batch", [2, 7, 64])
def test_batched_read_is_byte_identical_on_a_sparse_variant_idx(tmp_path, batch):
    """The AoU shape: convert only the biallelic SNVs, which are not contiguous."""
    prefix = _write_pgen(tmp_path, _panel())
    keep = [0, 1, 5, 6, 7, 20, 33, 34, 35, 36, 60]
    one = _convert(prefix, tmp_path, "s_one", variant_idx=keep, read_batch=1)
    many = _convert(prefix, tmp_path, f"s_many{batch}", variant_idx=keep,
                    read_batch=batch)
    assert one == many


def test_batched_read_respects_a_sample_subset(tmp_path):
    prefix = _write_pgen(tmp_path, _panel())
    cols = [0, 3, 4, 5, 9, 30, 36]
    one = _convert(prefix, tmp_path, "c_one", sample_idx=cols, read_batch=1)
    many = _convert(prefix, tmp_path, "c_many", sample_idx=cols, read_batch=16)
    assert one == many


def test_batched_read_is_byte_identical_under_every_missing_policy(tmp_path):
    G = _panel(seed=3)
    G[4, 2] = -9
    G[17, :5] = -9      # partly missing: every policy can handle it. A FULLY
                        # missing variant is refused by 'mean' on purpose, which
                        # is correct behaviour and not a batching question.
    G[40, 11] = -9
    G[41, 0] = -9
    prefix = _write_pgen(tmp_path, G)
    for policy in ("keep", "ref", "mean", "drop"):
        one = _convert(prefix, tmp_path, f"m1_{policy}", missing=policy,
                       read_batch=1)
        many = _convert(prefix, tmp_path, f"mN_{policy}", missing=policy,
                        read_batch=8)
        assert one == many, policy


def test_read_batch_defaults_to_batching_not_one(tmp_path):
    """The default must actually take the fast path, or nothing is gained."""
    import inspect
    sig = inspect.signature(pgen2cugen)
    assert "read_batch" in sig.parameters
    assert sig.parameters["read_batch"].default > 1


# ------------------------------------------------------------------ profiling
#
# Whether a GPU port of _convert_codes is worth building depends on how convert
# splits between pgenlib decode, the numpy transform, and the write. Measured on
# a laptop the transform is 0.317 ms/variant at n=535,662 -- 0.42 h for chr1 --
# but the other two terms were never measured on the VM that runs the job.


def test_profile_dict_reports_the_three_phases(tmp_path):
    import time
    prefix = _write_pgen(tmp_path, _panel(n_var=200, n_samp=400))
    prof: dict = {}
    t0 = time.perf_counter()
    pgen2cugen(f"{prefix}.pgen", str(tmp_path / "p.cugen"), verbose=False,
               profile=prof)
    wall = time.perf_counter() - t0
    assert {"read_s", "transform_s", "write_s", "n_variants"} <= set(prof)
    assert prof["n_variants"] == 200
    total = prof["read_s"] + prof["transform_s"] + prof["write_s"]
    assert 0 < total <= wall, "phases cannot exceed the wall time"
    assert total > 0.25 * wall, "the three phases should dominate the loop"


def test_profile_is_optional_and_default_changes_nothing(tmp_path):
    prefix = _write_pgen(tmp_path, _panel(n_var=40, n_samp=25))
    a = _convert(prefix, tmp_path, "np_a")
    prof: dict = {}
    out = tmp_path / "np_b.cugen"
    pgen2cugen(f"{prefix}.pgen", str(out), verbose=False, profile=prof)
    assert out.read_bytes() == a
    assert prof["read_s"] >= 0.0


# ------------------------------------------------------- parallel conversion
#
# Conversion is CPU-bound per variant and independent across variants, so it
# fans out across processes. Workers write their own bytes into disjoint slots
# of ONE preallocated file, which avoids a merge -- concatenating 0.34 TB of
# shards would cost 0.68 TB of sequential I/O and eat the entire gain.
#
# Workers must read the STAGED LOCAL pgen. Fanning out over the AoU mount would
# put N concurrent readers on the CDR bucket, which raises egress alerts; that
# guard is AoU.genome.ld.resolve_convert_source, enforced before calling here.


@pytest.mark.parametrize("workers", [2, 3, 5])
def test_parallel_conversion_is_byte_identical(tmp_path, workers):
    prefix = _write_pgen(tmp_path, _panel(n_var=61, n_samp=37))
    one = _convert(prefix, tmp_path, "w1")
    many = _convert(prefix, tmp_path, f"w{workers}", workers=workers)
    assert one == many


def test_parallel_conversion_byte_identical_on_a_sparse_variant_idx(tmp_path):
    prefix = _write_pgen(tmp_path, _panel(n_var=61, n_samp=37))
    keep = [0, 1, 5, 6, 7, 20, 33, 34, 35, 36, 60]
    one = _convert(prefix, tmp_path, "sp1", variant_idx=keep)
    many = _convert(prefix, tmp_path, "sp4", variant_idx=keep, workers=4)
    assert one == many


def test_parallel_conversion_handles_more_workers_than_variants(tmp_path):
    prefix = _write_pgen(tmp_path, _panel(n_var=3, n_samp=12))
    one = _convert(prefix, tmp_path, "tiny1")
    many = _convert(prefix, tmp_path, "tiny9", workers=9)
    assert one == many


def test_parallel_conversion_refuses_the_drop_policy(tmp_path):
    """'drop' changes the variant count, so slots cannot be assigned up front."""
    G = _panel(seed=5)
    G[3, 1] = -9
    prefix = _write_pgen(tmp_path, G)
    with pytest.raises(ValueError, match="drop"):
        pgen2cugen(f"{prefix}.pgen", str(tmp_path / "d.cugen"),
                   missing="drop", workers=4, verbose=False)


def test_parallel_conversion_preserves_missingness_stats(tmp_path):
    G = _panel(seed=9)
    G[4, 2] = -9
    G[40, 11] = -9
    prefix = _write_pgen(tmp_path, G)
    for policy in ("keep", "ref", "mean"):
        one = _convert(prefix, tmp_path, f"p1_{policy}", missing=policy)
        many = _convert(prefix, tmp_path, f"p4_{policy}", missing=policy,
                        workers=4)
        assert one == many, policy
