"""Positional writes into one .cugen, so conversion can be parallelised.

The AoU conversion is CPU-bound per variant and embarrassingly parallel across
variants. The format already permits a fan-out: data sits at
``data_offset + v * bytes_per_variant`` and stats/gidx are preallocated arrays
indexed by variant number. Only the writer was sequential.

Workers must read the STAGED LOCAL pgen, never the AoU mount -- N concurrent
readers on the CDR bucket is the pattern AoU raises egress alerts on. That guard
lives in AoU.genome.ld.resolve_convert_source; this file is only about the write.

The failure mode being guarded here is a worker silently skipping a slot, which
would leave a zeroed variant in an otherwise valid file. So the parallel writer
tracks which slots were filled and refuses to finalise an incomplete file --
strictly stronger than the sequential ``i != n_variants`` count, which cannot
tell a gap from a short write.
"""

from __future__ import annotations

import numpy as np
import pytest

from cugen.write import CugenWriter, ENCODING_2BIT
from cugen.convert import pack_2bit


def _panel(n_var=37, n_samp=53, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=(n_var, n_samp)).astype(np.int8)


def _sequential(path, G):
    n_var, n_samp = G.shape
    with CugenWriter(str(path), n_samp, n_var, ENCODING_2BIT) as w:
        for v in range(n_var):
            w.add_variant(v, G[v].astype(np.float64))
    return path.read_bytes()


def _packed_and_stats(row):
    from cugen.convert import _convert_codes
    packed, mu, sxx, maf, has_missing, _ = _convert_codes(
        np.ascontiguousarray(row, dtype=np.int8), "mean")
    return packed, mu, sxx, maf, has_missing


def test_create_mode_preallocates_the_whole_file(tmp_path):
    G = _panel()
    n_var, n_samp = G.shape
    p = tmp_path / "c.cugen"
    w = CugenWriter(str(p), n_samp, n_var, ENCODING_2BIT, parallel="create")
    with w:
        for v in range(n_var):
            packed, mu, sxx, maf, hm = _packed_and_stats(G[v])
            w.add_variant_packed_at(v, v, packed, mu, sxx, maf, hm)
    assert p.stat().st_size == w.data_offset + n_var * w.bytes_per_variant


@pytest.mark.parametrize("nchunk", [1, 2, 3, 8, 37])
def test_positional_writes_are_byte_identical_to_sequential(tmp_path, nchunk):
    G = _panel()
    n_var, n_samp = G.shape
    ref = _sequential(tmp_path / "seq.cugen", G)
    p = tmp_path / f"par{nchunk}.cugen"
    edges = np.linspace(0, n_var, nchunk + 1).astype(int)
    w = CugenWriter(str(p), n_samp, n_var, ENCODING_2BIT, parallel="create")
    with w:
        # out of order on purpose: position, not arrival order, must decide layout
        for a, b in reversed(list(zip(edges[:-1], edges[1:]))):
            for v in range(a, b):
                packed, mu, sxx, maf, hm = _packed_and_stats(G[v])
                w.add_variant_packed_at(v, v, packed, mu, sxx, maf, hm)
    assert p.read_bytes() == ref


def test_a_skipped_slot_is_refused(tmp_path):
    """The anti-corruption guard: a gap must not produce a valid-looking file."""
    G = _panel()
    n_var, n_samp = G.shape
    p = tmp_path / "gap.cugen"
    with pytest.raises(ValueError, match="slot|unfilled|gap"):
        w = CugenWriter(str(p), n_samp, n_var, ENCODING_2BIT, parallel="create")
        with w:
            for v in range(n_var):
                if v == 11:
                    continue
                packed, mu, sxx, maf, hm = _packed_and_stats(G[v])
                w.add_variant_packed_at(v, v, packed, mu, sxx, maf, hm)


def test_attach_mode_does_not_truncate_what_create_preallocated(tmp_path):
    """Workers attach to a file the coordinator made; they must not zero it."""
    G = _panel()
    n_var, n_samp = G.shape
    p = tmp_path / "a.cugen"
    coord = CugenWriter(str(p), n_samp, n_var, ENCODING_2BIT, parallel="create")
    coord.__enter__()
    size_after_prealloc = p.stat().st_size
    worker = CugenWriter(str(p), n_samp, n_var, ENCODING_2BIT, parallel="attach")
    with worker:
        for v in range(0, 5):
            packed, mu, sxx, maf, hm = _packed_and_stats(G[v])
            worker.add_variant_packed_at(v, v, packed, mu, sxx, maf, hm)
    assert p.stat().st_size == size_after_prealloc, "attach truncated the file"
    coord.f.close()


def test_attach_plus_create_finalize_equals_sequential(tmp_path):
    """The real shape: workers attach and write, the coordinator finalises."""
    G = _panel(n_var=40, n_samp=31)
    n_var, n_samp = G.shape
    ref = _sequential(tmp_path / "s2.cugen", G)
    p = tmp_path / "split.cugen"
    coord = CugenWriter(str(p), n_samp, n_var, ENCODING_2BIT, parallel="create")
    coord.__enter__()
    for a, b in ((0, 13), (13, 27), (27, 40)):
        wk = CugenWriter(str(p), n_samp, n_var, ENCODING_2BIT, parallel="attach")
        with wk:
            for v in range(a, b):
                packed, mu, sxx, maf, hm = _packed_and_stats(G[v])
                wk.add_variant_packed_at(v, v, packed, mu, sxx, maf, hm)
        coord.absorb(wk)
    coord.__exit__(None, None, None)
    assert p.read_bytes() == ref
