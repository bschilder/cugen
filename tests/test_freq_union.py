"""Per-ancestry MAF ascertainment (the ACAF rule).

A panel filtered on POOLED allele frequency silently discards variants that are
common inside one ancestry group but rare overall. For a group holding fraction
f of the cohort, a variant private to it has pooled AF ~= AF_group * f, so a
pooled --maf t drops everything with AF_group < t/f -- on 1kGP superpopulation
fractions that reaches AF < 7.2% for AMR. Those are the highest-Fst variants in
the panel, and they are exactly what a multi-ancestry LD reference exists to
describe.

All of Us ships its ACAF callset on the opposite rule -- "AF > 1% or AC > 100 in
ANY computed ancestry subpopulation" -- which is what these helpers implement.
"""
import pytest

from cugen.freq import pooled_af, read_afreq, union_maf_pass


def test_pooled_af_of_a_group_private_variant():
    # AMR is 347/2504 of 1kGP; a 5% AMR-private variant is 0.69% pooled.
    assert pooled_af(0.05, 347 / 2504) == pytest.approx(0.00693, rel=1e-3)


def test_pooled_filter_drops_a_variant_common_in_one_group():
    """The defect, as arithmetic rather than argument."""
    assert pooled_af(0.05, 347 / 2504) < 0.01 <= 0.05


def test_union_keeps_a_variant_passing_in_any_one_group():
    freqs = {"AFR": {"v1": 0.30, "v2": 0.001},
             "EUR": {"v1": 0.002, "v2": 0.002},
             "EAS": {"v1": 0.000, "v2": 0.004}}
    assert union_maf_pass(freqs, 0.01) == {"v1"}


def test_union_drops_a_variant_rare_in_every_group():
    assert union_maf_pass({"AFR": {"v": 0.004}, "EUR": {"v": 0.003}}, 0.01) == set()


def test_union_folds_alt_frequencies_above_one_half():
    """plink2 --freq reports ALT frequency, which may exceed 0.5. MAF is
    min(af, 1-af), so 0.98 is a MAF of 0.02 and must pass."""
    assert union_maf_pass({"EUR": {"v": 0.98}}, 0.01) == {"v"}


def test_a_group_missing_a_variant_cannot_veto_it():
    """Groups are filtered independently, so a variant monomorphic in one group
    is simply absent from its table. Absent must not be read as zero."""
    assert union_maf_pass({"AFR": {"v": 0.20}, "EUR": {}}, 0.01) == {"v"}


def test_union_is_a_union_not_an_intersection():
    """An intersection would keep only globally-common variants -- the same
    ascertainment the pooled filter imposes, by another route."""
    freqs = {"AFR": {"a": 0.2, "b": 0.001}, "EUR": {"a": 0.001, "b": 0.2}}
    assert union_maf_pass(freqs, 0.01) == {"a", "b"}


def test_read_afreq_parses_plink2_output(tmp_path):
    p = tmp_path / "g.afreq"
    p.write_text(
        "#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT\n"
        "chr1\tchr1:100:A:G\tA\tG\t0.0432\t1322\n"
        "chr1\tchr1:200:C:T\tC\tT\t0.9910\t1322\n")
    got = read_afreq(p)
    assert got == {"chr1:100:A:G": pytest.approx(0.0432),
                   "chr1:200:C:T": pytest.approx(0.9910)}


def test_read_afreq_skips_rows_with_an_unusable_frequency(tmp_path):
    """plink2 writes NA for a variant with no observations in the subset. A
    crash there would take down a 22-chromosome build at chromosome 9."""
    p = tmp_path / "g.afreq"
    p.write_text("#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT\n"
                 "chr1\tgood\tA\tG\t0.05\t100\n"
                 "chr1\tbad\tA\tG\tNA\t0\n")
    assert read_afreq(p) == {"good": pytest.approx(0.05)}


def test_read_afreq_locates_columns_by_name_not_position(tmp_path):
    """plink2's column set varies with --freq cols=; positional parsing would
    read the wrong field silently."""
    p = tmp_path / "g.afreq"
    p.write_text("#CHROM\tPOS\tID\tREF\tALT\tOBS_CT\tALT_FREQS\n"
                 "chr1\t100\tv\tA\tG\t1322\t0.07\n")
    assert read_afreq(p) == {"v": pytest.approx(0.07)}


# ----------------------------------------- what the pooled filter would keep

def test_pooled_from_groups_weights_by_sample_count():
    """Pooled AF is the sample-weighted mean of the group frequencies, not the
    unweighted one. AFR is 661/2504 and AMR 347/2504; treating them equally
    would misstate the pooled frequency and therefore the size of the defect."""
    from cugen.freq import pooled_from_groups
    freqs = {"AFR": {"v": 0.10}, "AMR": {"v": 0.00}}
    sizes = {"AFR": 661, "AMR": 347}
    assert pooled_from_groups(freqs, sizes)["v"] == pytest.approx(
        0.10 * 661 / 1008, rel=1e-9)


def test_pooled_from_groups_treats_an_absent_variant_as_absent_not_missing():
    """A variant monomorphic in a group is AF 0 there for the POOLED average --
    the opposite of union_maf_pass, where absent must not mean zero. The two
    rules differ on purpose and the difference is easy to get backwards."""
    from cugen.freq import pooled_from_groups
    freqs = {"AFR": {"v": 0.20}, "EUR": {}}
    sizes = {"AFR": 661, "EUR": 503}
    assert pooled_from_groups(freqs, sizes)["v"] == pytest.approx(
        0.20 * 661 / 1164, rel=1e-9)


def test_the_union_is_a_strict_superset_of_a_pooled_filter():
    """The whole claim, as a test. Anything a pooled filter keeps is common
    enough somewhere to survive the union too; the reverse does not hold."""
    from cugen.freq import pooled_from_groups
    sizes = {"AFR": 661, "AMR": 347, "EAS": 504, "EUR": 503, "SAS": 489}
    freqs = {
        "AFR": {"shared": 0.30, "afr_only": 0.030, "rare": 0.001},
        "AMR": {"shared": 0.28, "afr_only": 0.000, "rare": 0.001},
        "EAS": {"shared": 0.31, "afr_only": 0.000, "rare": 0.002},
        "EUR": {"shared": 0.29, "afr_only": 0.000, "rare": 0.001},
        "SAS": {"shared": 0.30, "afr_only": 0.000, "rare": 0.000},
    }
    union = union_maf_pass(freqs, 0.01)
    pooled_af_by_id = pooled_from_groups(freqs, sizes)
    pooled = {v for v, a in pooled_af_by_id.items() if min(a, 1 - a) >= 0.01}
    assert pooled < union, "union must strictly contain the pooled set"
    assert union - pooled == {"afr_only"}, (
        "a 3% AFR-private variant is 0.79% pooled and is exactly what the old "
        "filter discarded")
