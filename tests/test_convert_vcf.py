"""VCF -> .cugen dosage conversion.

The genotype codes are the whole risk here. cyvcf2's `gt_types` is
0=HOM_REF, 1=HET, 2=UNKNOWN, 3=HOM_ALT by default, and only becomes the
intuitive 0/1/2-plus-3-for-missing when the VCF is opened with gts012=True.
Getting that backwards maps every homozygous-ALT call to missing and every
missing call to homozygous-ALT -- silently, with plausible-looking output.
"""
import sys

import numpy as np
import pytest

cyvcf2 = pytest.importorskip("cyvcf2")

from cugen.convert import vcf2cugen          # noqa: E402
from cugen.io import read_cugen              # noqa: E402
from cugen.write import unpack_2bit          # noqa: E402

# One variant per genotype class, plus a mixed one, and a missing call.
VCF = """##fileformat=VCFv4.2
##contig=<ID=chr22>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4
chr22\t100\tv0\tA\tG\t.\t.\t.\tGT\t0|0\t0|1\t1|0\t1|1
chr22\t200\tv1\tC\tT\t.\t.\t.\tGT\t1|1\t1|1\t1|1\t1|1
chr22\t300\tv2\tG\tA\t.\t.\t.\tGT\t0/0\t0/1\t1/1\t./.
"""

# what each row must decode to: dosage 0/1/2, 3 = missing
WANT = np.array([[0, 1, 1, 2],
                 [2, 2, 2, 2],
                 [0, 1, 2, 3]], dtype=np.uint8)


def _decode(path, n, p):
    r = read_cugen(path)
    packed = np.frombuffer(r.read_packed_bytes(), dtype=np.uint8)
    bpv = int(r.bytes_per_variant)
    return np.stack([unpack_2bit(packed[v * bpv:(v + 1) * bpv], n)
                     for v in range(p)])


def test_homozygous_alt_is_dosage_two_not_missing(tmp_path):
    """The bug this file exists for.

    Measured on real 1000 Genomes chr22 before the fix: a variant with 2,321
    `1|1` genotypes produced 2,321 missing calls and ZERO dosage-2 calls.
    """
    src = tmp_path / "t.vcf"
    src.write_text(VCF)
    out = str(tmp_path / "t.cugen")
    vcf2cugen(str(src), out, verbose=False)

    got = _decode(out, 4, 3)
    np.testing.assert_array_equal(got, WANT)


def test_the_all_homozygous_alt_variant_is_not_reported_as_missing(tmp_path):
    """Sharpest form: a variant where every call is 1|1 must have allele
    frequency 1.0 and no missingness -- not 0.0 and entirely missing."""
    src = tmp_path / "t.vcf"
    src.write_text(VCF)
    out = str(tmp_path / "t.cugen")
    vcf2cugen(str(src), out, verbose=False)

    got = _decode(out, 4, 3)
    assert (got[1] == 2).all(), f"all-1|1 variant decoded as {got[1]}"
    assert (got[1] == 3).sum() == 0


def test_a_vcf_with_no_missing_calls_does_not_set_has_missing(tmp_path):
    """Consequence worth pinning separately: a spurious HAS_MISSING flag
    silently disables the fused GPU path, and with it stream= and count_only.
    """
    src = tmp_path / "clean.vcf"
    src.write_text("\n".join(VCF.splitlines()[:-1]) + "\n")   # drop the ./. row
    out = str(tmp_path / "clean.cugen")
    vcf2cugen(str(src), out, verbose=False)
    assert not read_cugen(out).has_missing, (
        "HAS_MISSING set on a VCF with no missing calls; this makes the fused "
        "scan, stream=True and count_only unreachable")


# A triallelic site (REF=G, ALT=A,C) alongside a biallelic control.
MULTI = """##fileformat=VCFv4.2
##contig=<ID=chr22>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4
chr22\t100\tv0\tA\tG\t.\t.\t.\tGT\t0|0\t0|1\t1|0\t1|1
chr22\t200\ttri\tG\tA,C\t.\t.\t.\tGT\t0|2\t1|2\t2|2\t0|1
"""


def test_a_multiallelic_record_is_refused_not_silently_miscoded(tmp_path):
    """A triallelic site has no honest 0/1/2 dosage column, and before this
    guard each backend invented a different wrong one.

    Measured on REF=G ALT=A,C over every diploid genotype:

        GT       0|0  0|1  1|0  0|2  2|0  1|1  1|2  2|1  2|2   mu_x
        cyvcf2     0    1    1    1    1    2    1    1    2  1.1111
        pysam      0    1    1    2    2    2    3    3    3  1.3333

    cyvcf2 pools every ALT into one non-reference allele; pysam writes the
    0|2 heterozygote as a homozygote and then loses 1|2 and 2|2 to the missing
    code. A biallelic control row is identical under both, so the divergence
    is specific to multi-allelic records -- which means the same VCF produced
    different .cugen depending on which library happened to be installed.
    """
    src = tmp_path / "multi.vcf"
    src.write_text(MULTI)
    with pytest.raises(ValueError, match="multi-allelic"):
        vcf2cugen(str(src), str(tmp_path / "multi.cugen"), verbose=False)


def test_the_refusal_names_the_bcftools_split_command(tmp_path):
    """Refusing is only useful if it says what to run instead. Splitting is
    upstream's job -- bcftools renormalises REF/ALT, which cugen cannot do
    from genotypes alone."""
    src = tmp_path / "multi.vcf"
    src.write_text(MULTI)
    with pytest.raises(ValueError) as exc:
        vcf2cugen(str(src), str(tmp_path / "multi.cugen"), verbose=False)
    assert "bcftools norm" in str(exc.value)
    assert "chr22:200" in str(exc.value), "should name the offending record"


def test_the_refusal_does_not_depend_on_the_vcf_library(tmp_path, monkeypatch):
    """The bug was backend-dependent, so the guard must not be."""
    pytest.importorskip("pysam")
    monkeypatch.setitem(sys.modules, "cyvcf2", None)   # force the pysam path
    src = tmp_path / "multi.vcf"
    src.write_text(MULTI)
    with pytest.raises(ValueError, match="multi-allelic"):
        vcf2cugen(str(src), str(tmp_path / "multi.cugen"), verbose=False)


def test_a_purely_biallelic_file_is_untouched_by_the_guard(tmp_path):
    """The guard must not cost anything on split input, which is the normal
    case -- the 1kGP 30x panel carries zero records with more than one ALT."""
    src = tmp_path / "bi.vcf"
    src.write_text(VCF)
    out = str(tmp_path / "bi.cugen")
    vcf2cugen(str(src), out, verbose=False)
    np.testing.assert_array_equal(_decode(out, 4, 3), WANT)
