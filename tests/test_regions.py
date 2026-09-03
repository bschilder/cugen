"""BED-based region exclusion.

The failure modes this pins down are the ones that silently produce wrong
answers rather than errors:

  * BED is 0-based half-open [start, end). Variant positions in .bim/.pvar are
    1-based. A variant at 1-based p is inside [s, e) iff s < p <= e. Getting
    this wrong shifts every boundary by one base, which no test of totals
    would catch.
  * Chromosome naming differs between sources ("chr1" vs "1"). UCSC ships
    "chr1", many pvar files use "1". A mismatch silently excludes NOTHING and
    the scan looks like it ran clean.
  * BED files are not required to be sorted or non-overlapping.
"""

import numpy as np
import pytest

from cugen.regions import mask_variants, merge_intervals, read_bed


def _write(tmp_path, text, name="t.bed"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_bed_is_zero_based_half_open_against_one_based_positions(tmp_path):
    """chr1 100-200 in BED covers 1-based positions 101..200 inclusive."""
    bed = _write(tmp_path, "chr1\t100\t200\n")
    chrom = np.array(["chr1"] * 5)
    pos = np.array([100, 101, 150, 200, 201], dtype=np.int64)
    m = mask_variants(chrom, pos, read_bed(bed))
    assert m.tolist() == [False, True, True, True, False]


def test_chromosome_naming_is_normalised_both_ways(tmp_path):
    """A 'chr1' BED must mask a '1' pvar and vice versa. Silently matching
    nothing is the dangerous outcome: the scan looks clean."""
    for bed_name, var_name in (("chr1", "1"), ("1", "chr1")):
        bed = _write(tmp_path, f"{bed_name}\t100\t200\n", name=f"{bed_name}{var_name}.bed")
        m = mask_variants(np.array([var_name]), np.array([150]), read_bed(bed))
        assert m.tolist() == [True], f"{bed_name} BED failed to mask {var_name}"


def test_unsorted_and_overlapping_intervals_are_handled(tmp_path):
    bed = _write(tmp_path, "chr1\t300\t400\nchr1\t100\t200\nchr1\t150\t350\n")
    m = mask_variants(np.array(["chr1"] * 4), np.array([50, 150, 250, 450]),
                      read_bed(bed))
    assert m.tolist() == [False, True, True, False]


def test_merge_intervals_collapses_overlaps_and_adjacency():
    got = merge_intervals([(10, 20), (15, 25), (25, 30), (40, 50)])
    assert got == [(10, 30), (40, 50)]


def test_padding_widens_symmetrically_and_clamps_at_zero(tmp_path):
    bed = _write(tmp_path, "chr1\t100\t200\n")
    b = read_bed(bed)
    m = mask_variants(np.array(["chr1"] * 3), np.array([90, 210, 260]), b, pad=50)
    assert m.tolist() == [True, True, False]
    near0 = read_bed(_write(tmp_path, "chr1\t10\t20\n", name="z.bed"))
    m2 = mask_variants(np.array(["chr1"]), np.array([1]), near0, pad=1000)
    assert m2.tolist() == [True]


def test_comment_and_track_lines_are_skipped(tmp_path):
    bed = _write(tmp_path, "# comment\ntrack name=x\nbrowser hide all\nchr1\t100\t200\n")
    assert read_bed(bed)["1"] == [(100, 200)]


def test_variants_on_a_chromosome_absent_from_the_bed_are_kept(tmp_path):
    bed = _write(tmp_path, "chr1\t100\t200\n")
    m = mask_variants(np.array(["chr2", "chr1"]), np.array([150, 150]), read_bed(bed))
    assert m.tolist() == [False, True]


def test_multiple_beds_union(tmp_path):
    a = read_bed(_write(tmp_path, "chr1\t100\t200\n", name="a.bed"))
    b = read_bed(_write(tmp_path, "chr1\t500\t600\n", name="b.bed"))
    pos = np.array([150, 550, 800])
    m = mask_variants(np.array(["chr1"] * 3), pos, [a, b])
    assert m.tolist() == [True, True, False]


def test_empty_bed_masks_nothing(tmp_path):
    m = mask_variants(np.array(["chr1"]), np.array([150]),
                      read_bed(_write(tmp_path, "", name="e.bed")))
    assert m.tolist() == [False]


def test_gzipped_bed_is_read(tmp_path):
    import gzip
    p = tmp_path / "g.bed.gz"
    with gzip.open(p, "wt") as f:
        f.write("chr1\t100\t200\n")
    assert read_bed(str(p))["1"] == [(100, 200)]


def test_malformed_line_raises_rather_than_being_skipped(tmp_path):
    """A truncated or mis-delimited line must not silently reduce the mask."""
    bed = _write(tmp_path, "chr1\t100\n")
    with pytest.raises(ValueError, match="line 1"):
        read_bed(bed)


# ------------------- integration with ld_matrix

def _tiny_panel(tmp_path, n_samples=40, n_variants=12, seed=0):
    """A .cugen plus the CHR/POS annotation ld_matrix needs for coordinates."""
    import pandas as pd
    from cugen.write import CugenWriter, ENCODING_2BIT

    rng = np.random.default_rng(seed)
    path = tmp_path / "p.cugen"
    with CugenWriter(path, n_samples, n_variants, ENCODING_2BIT) as w:
        for k in range(n_variants):
            # keep every variant polymorphic so none is dropped for being
            # monomorphic rather than for being excluded
            d = rng.integers(0, 3, size=n_samples).astype(np.float64)
            d[0], d[1] = 0.0, 2.0
            w.add_variant(k, d)
    ann = pd.DataFrame({
        "gidx": np.arange(n_variants, dtype=np.int64),
        "CHR": ["1"] * n_variants,
        "POS": np.arange(1, n_variants + 1) * 1000,
        "ID": [f"v{k}" for k in range(n_variants)],
    })
    return str(path), ann


def test_ld_matrix_excludes_variants_inside_the_bed(tmp_path):
    """Excluding must remove the variants themselves, so no PAIR involving them
    is emitted -- filtering after the fact would still have paid for the scan."""
    from cugen.ld import ld_matrix

    path, ann = _tiny_panel(tmp_path)
    bed = tmp_path / "x.bed"
    # POS 3000..6000 (1-based) -> BED half-open [2999, 6000)
    bed.write_text("chr1\t2999\t6000\n")

    full = ld_matrix(path, annotation=ann, backend="numpy", min_r2=0.0,
                     max_pairs=10**9, verbose=False)
    cut = ld_matrix(path, annotation=ann, backend="numpy", min_r2=0.0,
                    max_pairs=10**9, verbose=False, exclude_regions=str(bed))
    kept_pos = set(cut["POS_A"]).union(cut["POS_B"])
    assert not ({3000, 4000, 5000, 6000} & kept_pos), \
        "excluded positions still appear in the output"
    assert {1000, 2000, 7000} <= kept_pos, "non-excluded positions were dropped"
    assert len(cut) < len(full)


def test_ld_matrix_exclude_requires_coordinates(tmp_path):
    """Without annotation= there are no coordinates to test against, and
    silently skipping the exclusion would be the dangerous outcome."""
    from cugen.ld import ld_matrix

    path, _ = _tiny_panel(tmp_path)
    bed = tmp_path / "x.bed"
    bed.write_text("chr1\t2999\t6000\n")
    with pytest.raises(ValueError, match="annotation"):
        ld_matrix(path, backend="numpy", verbose=False, exclude_regions=str(bed))


def test_ld_matrix_exclude_pad_widens(tmp_path):
    from cugen.ld import ld_matrix

    path, ann = _tiny_panel(tmp_path)
    bed = tmp_path / "x.bed"
    bed.write_text("chr1\t3999\t4000\n")           # POS 4000 only
    narrow = ld_matrix(path, annotation=ann, backend="numpy", min_r2=0.0,
                       max_pairs=10**9, verbose=False, exclude_regions=str(bed))
    wide = ld_matrix(path, annotation=ann, backend="numpy", min_r2=0.0,
                     max_pairs=10**9, verbose=False, exclude_regions=str(bed),
                     exclude_pad=1000)
    nw = set(narrow["POS_A"]).union(narrow["POS_B"])
    ww = set(wide["POS_A"]).union(wide["POS_B"])
    assert 3000 in nw and 5000 in nw
    assert 3000 not in ww and 5000 not in ww, "pad did not widen the exclusion"


def test_excluded_variants_are_never_scanned_not_merely_filtered(tmp_path):
    """The requirement is that excluded regions are never COMPUTED.

    Output length cannot distinguish pre- from post-filtering: dropping the
    variants and dropping every pair touching them both yield C(8,2). The
    discriminating property is that exclusion acts at VARIANT SELECTION, so it
    must be byte-for-byte identical to naming the survivors with variants= --
    the same code path, reached two ways. A post-hoc pair filter would compute
    all 66 pairs and could not be identical to an 8-variant scan.
    """
    from cugen.ld import ld_matrix

    path, ann = _tiny_panel(tmp_path)
    bed = tmp_path / "x.bed"
    bed.write_text("chr1\t2999\t6000\n")        # POS 3000,4000,5000,6000

    kept = [0, 1, 6, 7, 8, 9, 10, 11]             # gidx of POS not in the BED
    by_hand = ld_matrix(path, annotation=ann, backend="numpy", min_r2=0.0,
                        max_pairs=10**9, verbose=False, variants=kept)
    by_bed = ld_matrix(path, annotation=ann, backend="numpy", min_r2=0.0,
                       max_pairs=10**9, verbose=False,
                       exclude_regions=str(bed))
    assert len(by_bed) == len(by_hand) == 28, (
        f"expected C(8,2)=28 pairs, got {len(by_bed)} / {len(by_hand)}")
    import pandas as pd
    pd.testing.assert_frame_equal(
        by_bed.reset_index(drop=True), by_hand.reset_index(drop=True))


def test_exclusion_reports_what_it_dropped(tmp_path, capsys):
    """A silent mask is indistinguishable from a mask that matched nothing --
    the failure mode where a naming mismatch makes an unmasked scan look
    filtered. The count must be printed."""
    from cugen.ld import ld_matrix

    path, ann = _tiny_panel(tmp_path)
    bed = tmp_path / "x.bed"
    bed.write_text("chr1\t2999\t6000\n")
    ld_matrix(path, annotation=ann, backend="numpy", min_r2=0.0,
              max_pairs=10**9, verbose=True, exclude_regions=str(bed))
    out = capsys.readouterr().out
    assert "exclude_regions" in out and "dropped 4 of 12" in out, out
