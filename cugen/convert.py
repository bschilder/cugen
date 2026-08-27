"""cugen.convert - build .cugen files from PGEN, VCF/BCF or PLINK1 BED.

    pgen2cugen   PLINK2 .pgen/.pvar/.psam   (needs pgenlib)
    vcf2cugen    VCF/VCF.gz/BCF             (needs cyvcf2, else pysam)
    bed2cugen    PLINK1 .bed/.bim/.fam      (no dependencies -- pure numpy)

All three stream variant-by-variant through CugenWriter, so peak memory is one
variant, not the genotype matrix.

SAMPLE ORDER IS THE CONTRACT
----------------------------
Every downstream cugen object -- cohort .npz `sample_idx_sorted`, the LOCO
matrices, the covariate sidecar -- is positional against the sample order of
the .cugen it was built from. Converting the same source with a different
--keep list produces a file that is structurally valid and silently
incompatible with those objects. Each converter writes a companion
`<out>.samples.txt` recording the exact order used; check it before pairing a
new .cugen with existing cohorts.

MISSING GENOTYPES
-----------------
Encoded as 2-bit value 3 and EXCLUDED from mu_x/sxx/maf. The association
kernels are complete-case (io.py:108): they skip 3 rather than treating it as
dosage 0. Writing missing as 0 instead would inflate the effective variance
relative to the stored non-missing sxx and manufacture false positives on
high-missingness probes -- that failure mode cost a lot of debugging, so do not
"simplify" it here.

BED note: PLINK1 is already 2-bit but uses a DIFFERENT code (01 = missing,
little-endian within byte). bed2cugen remaps rather than copying bytes.
"""
import os
import sys

import numpy as np

from .write import CugenWriter, ENCODING_2BIT, pack_2bit


#: 2-bit code for a no-call. Dosages are 0/1/2; 3 means missing.
_MISSING_CODE = 3.0
#: The same code as a 2-bit integer, for the int8 fast path.
_MISSING_U8 = 3

MISSING_POLICIES = ("keep", "ref", "mean", "drop")


def _apply_missing_policy(dosages, policy):
    """Resolve no-calls in one variant. Returns ``(dosages_or_None, n_missing)``.

    ``None`` means drop the variant.

    Why this exists
    ---------------
    cugen's association kernels are complete-case and skip code 3, which is why
    ``keep`` is the historical behaviour and remains the default. The LD scan
    cannot do that: its fused path -- the only one supporting ``stream=`` or
    ``count_only=`` -- requires ``not reader.has_missing``, and
    ``FLAG_HAS_MISSING`` is set for the WHOLE file if a single genotype anywhere
    is a no-call. So for LD the missingness has to be resolved at conversion
    time; there is no scan-time option.

    Policies
    --------
    keep    Leave code 3 in place. Complete-case downstream. Sets the file flag.
    ref     Fill with 0 (hom REF).
    mean    Fill with the observed mean, ROUNDED to the nearest integer dosage.
    drop    Return None for any variant carrying a no-call.

    Statistics, stated plainly
    --------------------------
    Imputation here happens BEFORE ``CugenWriter.add_variant``, so
    ``variant_stats`` runs over the imputed vector and mu_x/sxx/maf are
    self-consistent with the stored dosages. That is the crucial difference from
    the failure this module's docstring warns about, which was writing 0 for
    missing while keeping sxx over only the non-missing calls -- data and stats
    disagreeing.

    Both fills bias, and rounding makes ``mean`` bias differently than you would
    expect. Measured, not assumed -- see tests/test_convert_missing.py:

    * EXACT mean-imputation would leave sxx UNCHANGED: filled samples sit on the
      mean, contribute nothing to the centred sum of squares, and do not move
      the mean. Only the variance estimate ``sxx/n`` shrinks, by
      ``1 - missingness``.
    * ``mean`` here ROUNDS, because 2-bit has no fractional code, so the fill
      can sit up to 0.49 away from the mean. That shifts the mean AND adds
      spread, so sxx goes UP -- the same direction as the failure this module's
      docstring warns about. It is self-consistent, because the stats are
      recomputed over the imputed vector rather than left over the observed
      calls, but it is not the benign case exact imputation would be.
    * ``round(mean)`` is 0 for any variant whose mean dosage is <= 0.5, i.e.
      AF <= 0.25. Most of a MAF>1% panel sits below that, so for those variants
      ``mean`` and ``ref`` produce BYTE-IDENTICAL files. The choice only bites
      above AF 0.25.
    * ``ref`` shifts mu_x and maf downward in proportion to the missingness
      rate, hardest on common variants.

    Exact mean-imputation needs a float encoding, at 4-16x the bytes per
    variant -- which re-breaks the device-residency ceiling that motivates
    resolving missingness at all. Rounding is numpy's half-to-even, so an
    observed mean of exactly 0.5 fills 0 and exactly 1.5 fills 2.
    """
    if policy not in MISSING_POLICIES:
        raise ValueError(
            f"unknown missing= policy {policy!r}; expected one of "
            f"{list(MISSING_POLICIES)}")
    d = np.asarray(dosages, dtype=np.float64)
    miss = ~(np.isfinite(d) & (d >= 0) & (d <= 2))
    n_missing = int(miss.sum())
    if n_missing == 0:
        return d, 0
    if policy == "keep":
        # Never mutate the caller's buffer: pgen2cugen reuses one array across
        # every variant, so an in-place fill would leak into the next record.
        out = d.copy()
        out[miss] = _MISSING_CODE
        return out, n_missing
    if policy == "drop":
        return None, n_missing
    if policy == "ref":
        out = d.copy()
        out[miss] = 0.0
        return out, n_missing
    # mean
    obs = d[~miss]
    if obs.size == 0:
        # No observed call means no mean. Filling anything would be a
        # fabrication, so the variant is dropped rather than invented.
        return None, n_missing
    out = d.copy()
    out[miss] = float(np.rint(obs.mean()))
    return out, n_missing


def _convert_codes(codes, policy):
    """One variant of pgenlib hardcalls -> what CugenWriter would have written.

    Returns ``(packed_bytes, mu, sxx, maf, has_missing, n_missing)``, or a
    leading ``None`` when the policy drops the variant.

    This replaces a float64 round-trip. The old path promoted each variant to
    float64 -- 4.09 MiB at n=535,662, well past L2 -- and then built roughly a
    dozen more arrays that size across the missing policy, ``variant_stats`` and
    the 2-bit packing. That measured 2.89 ms per variant, i.e. 2.04 h for chr1's
    2,537,153 SNVs before any I/O. Dosages are 0/1/2 plus a no-call, so none of
    it needed 64-bit floats.

    Three things make it fast, all measured rather than assumed:

    * ``int8 -> uint8`` is a free view, and every negative sentinel lands at
      >= 128 unsigned, so ONE ``minimum`` collapses all no-calls to code 3 with
      no comparison pass and no ``where``.
    * ``np.count_nonzero`` is 4x faster than ``(u == k).sum()`` here, and
      ``np.bincount`` on uint8 is 17x SLOWER -- it upcasts to intp.
    * mu and sxx follow from the counts. For dosages in {0,1,2},
      ``sum(x) = n1 + 2*n2`` and ``sum(x**2) = n1 + 4*n2``, both exact integers
      far inside 2**53, so no pass over the vector is needed at all.

    The statistics are computed over the IMPUTED counts, preserving the ordering
    the old path had (policy applied before ``variant_stats``), which is what
    keeps mu/sxx/maf self-consistent with the stored dosages.

    The written file is BIT-IDENTICAL to what the old path produced. ``sxx`` is
    the only value that differs in float64 -- the closed form ``s2 - s1**2/cnt``
    carries two roundings where ``sum((x - mu)**2)`` carries about n, making the
    closed form the more accurate of the two -- and the difference is ~4e-16
    relative, which the float32 the writer stores erases completely.
    tests/test_convert_fastpath.py asserts both halves.
    """
    c = np.ascontiguousarray(codes, dtype=np.int8)
    n = int(c.size)
    u = np.minimum(c.view(np.uint8), np.uint8(_MISSING_U8))
    # Plain ints, not numpy scalars: variant_stats returned bool() and int(),
    # and CugenWriter stores them; a np.bool_ leaking out here would be a
    # silent type change in the header fields.
    n0 = int(np.count_nonzero(u == 0))
    n1 = int(np.count_nonzero(u == 1))
    n2 = int(np.count_nonzero(u == 2))
    cnt = n0 + n1 + n2
    n_missing = n - cnt

    if n_missing:
        if policy == "drop":
            return None, 0.0, 0.0, 0.0, True, n_missing
        if policy == "mean":
            if cnt == 0:
                # No observed call means no mean to fill with; pgen2cugen turns
                # this into an error naming the variant.
                return None, 0.0, 0.0, 0.0, True, n_missing
            # np.rint, not round(): banker's rounding, matching the old path.
            fill = int(np.rint(float(n1 + 2 * n2) / cnt))
        elif policy == "ref":
            fill = 0
        elif policy == "keep":
            fill = _MISSING_U8
        else:
            raise ValueError(
                f"unknown missing= policy {policy!r}; expected one of "
                f"{list(MISSING_POLICIES)}")
        if fill != _MISSING_U8:
            u[u == _MISSING_U8] = fill
            if fill == 0:
                n0 += n_missing
            elif fill == 1:
                n1 += n_missing
            else:
                n2 += n_missing
            cnt += n_missing

    if cnt:
        s1 = float(n1 + 2 * n2)
        s2 = float(n1 + 4 * n2)
        mu = s1 / cnt
        sxx = s2 - s1 * s1 / cnt
        af = mu / 2.0
        maf = min(af, 1.0 - af)
    else:
        mu = sxx = maf = 0.0
    return pack_2bit(u).tobytes(), mu, sxx, maf, bool(cnt < n), n_missing


def _write_samples(out_path, sample_ids):
    p = f"{out_path}.samples.txt"
    with open(p, "w") as f:
        f.write("\n".join(str(s) for s in sample_ids) + "\n")
    return p


def _progress(i, n, every=50000):
    if n and i % every == 0:
        print(f"  {i:,}/{n:,} variants ({100.0*i/n:.0f}%)", flush=True)


def pgen2cugen(pgen, out, psam=None, pvar=None, sample_idx=None,
               variant_idx=None, gidx=None, missing="mean", verbose=True):
    """PLINK2 .pgen -> .cugen. Requires pgenlib.

    ``missing`` resolves no-calls; see :func:`_apply_missing_policy` for the
    policies and their statistical consequences. The short version:

    * ``"mean"`` (default) fills with the observed mean rounded to an integer
      dosage, so the file carries no missing code and cugen's fused LD scan can
      use it. That scan is the only path supporting ``stream=`` or
      ``count_only=``, and it refuses any file with ``has_missing`` set -- a
      file-wide flag that one no-call anywhere trips.
    * ``"keep"`` is the historical behaviour: leave code 3 in place and let
      cugen's complete-case association kernels skip it. Use this for
      association scans, where complete-case is the correct treatment and
      imputing would change the analysis.
    * ``"ref"`` fills with 0. ``"drop"`` discards any variant carrying a
      no-call, and costs a second pass over the .pgen to learn the surviving
      count, since the header declares it up front.

    A missingness summary is printed when ``verbose``, so the choice can be
    checked against the data rather than assumed.
    """
    try:
        import pgenlib
    except ImportError:
        raise ImportError("pgen2cugen needs pgenlib (pip install Pgenlib)")

    psam = psam or os.path.splitext(pgen)[0] + ".psam"
    ids = []
    with open(psam) as f:
        for line in f:
            if line.startswith("##"):
                continue
            p = line.split()
            if p[0].lstrip("#").upper() in ("FID", "IID"):
                continue
            ids.append(p[0])
    raw_n = len(ids)

    si = None if sample_idx is None else np.asarray(sample_idx, dtype=np.uint32)
    reader = pgenlib.PgenReader(str(pgen).encode(), raw_sample_ct=raw_n,
                                sample_subset=si)
    n_samples = reader.get_raw_sample_ct() if si is None else len(si)
    total = reader.get_variant_ct()
    vidx = np.arange(total) if variant_idx is None else np.asarray(variant_idx)
    g = vidx if gidx is None else np.asarray(gidx)
    kept_ids = ids if si is None else [ids[i] for i in si]

    if missing not in MISSING_POLICIES:
        raise ValueError(
            f"unknown missing= policy {missing!r}; expected one of "
            f"{list(MISSING_POLICIES)}")
    if verbose:
        print(f"pgen2cugen: {n_samples:,} samples x {len(vidx):,} variants "
              f"-> {out}  (missing={missing})")
    buf = np.empty(n_samples, dtype=np.int8)

    def _read(v):
        """One variant of raw int8 hardcalls, straight from pgenlib.

        Deliberately NOT promoted to float64 here. That promotion, plus the
        temporaries it forced through the missing policy, variant_stats and the
        packing, was 2.89 ms per variant at n=535,662 -- 2.04 h for chr1 before
        any I/O. :func:`_convert_codes` consumes the int8 directly.
        """
        reader.read(int(v), buf)
        return buf

    # `drop` changes the variant count, and CugenWriter._finalize refuses a
    # short write ("declared N variants, wrote M"). So learn the surviving count
    # first -- the same two-pass shape vcf2cugen already uses, and only paid
    # when the policy actually needs it.
    keep_mask = None
    if missing == "drop":
        keep_mask = np.ones(len(vidx), dtype=bool)
        for k, v in enumerate(vidx):
            keep_mask[k] = _convert_codes(_read(v), "drop")[0] is not None
            if verbose:
                _progress(k, len(vidx))
        if verbose:
            print(f"  drop: {int((~keep_mask).sum()):,} of {len(vidx):,} "
                  f"variants carry a no-call and will be discarded", flush=True)

    n_out = len(vidx) if keep_mask is None else int(keep_mask.sum())
    if n_out == 0:
        raise ValueError(
            f"missing={missing!r} leaves 0 of {len(vidx):,} variants. Every "
            f"variant carries a no-call, which at this sample count is normal "
            f"-- use missing='mean' or 'ref' instead of dropping.")

    n_missing_total = 0
    n_var_with_missing = 0
    n_var_undefined = 0
    with CugenWriter(out, n_samples, n_out, ENCODING_2BIT) as w:
        for k, v in enumerate(vidx):
            if keep_mask is not None and not keep_mask[k]:
                continue
            packed, mu, sxx, maf, has_missing, n_miss = _convert_codes(
                _read(v), missing)
            if n_miss:
                n_missing_total += n_miss
                n_var_with_missing += 1
            if packed is None:
                # Only reachable for 'mean' on a fully-missing variant, which
                # the counting pass above does not run for. Refuse rather than
                # write a short file, which _finalize would reject anyway with
                # a message that does not name the cause.
                n_var_undefined += 1
                raise ValueError(
                    f"variant {int(g[k])} has no observed call, so "
                    f"missing={missing!r} has no mean to fill with. Use "
                    f"missing='drop' to discard such variants, or filter them "
                    f"upstream (plink2 --geno 0.99).")
            w.add_variant_packed(int(g[k]), packed, mu, sxx, maf, has_missing)
            if verbose:
                _progress(k, len(vidx))
    reader.close()
    _write_samples(out, kept_ids)
    if verbose:
        cells = float(n_samples) * max(len(vidx), 1)
        print(f"  missingness: {n_missing_total:,} no-calls "
              f"({100.0 * n_missing_total / cells:.4f}% of genotypes) across "
              f"{n_var_with_missing:,} of {len(vidx):,} variants; "
              f"policy={missing}, {n_out:,} variants written", flush=True)
    return out


def vcf2cugen(vcf, out, region=None, keep=None, gidx_start=0,
              min_ac=0, verbose=True):
    """VCF/BCF -> .cugen. Uses cyvcf2 if available, else pysam.

    Two passes: one to count variants (the header does not carry the count),
    one to write. Pass --region to convert a chromosome at a time, which is how
    the rest of cugen expects the data laid out (chr<N>.cugen).

    INPUT MUST BE BIALLELIC. A record with more than one ALT is refused, with
    the bcftools command to fix it. cugen does not split multi-allelic sites and
    should not: splitting renormalises REF/ALT against the reference sequence,
    which is information cugen never sees. Do it upstream, once:

        bcftools norm -m -any in.vcf.gz -Oz -o split.vcf.gz
        bcftools index -t split.vcf.gz

    That is lossless for per-allele dosage -- a 1|2 genotype at a triallelic
    site becomes dosage 1 at each of the two resulting records -- and it is what
    most reference panels already ship. The 1kGP 30x GRCh38 panel, for one,
    carries zero records with more than one ALT (chr21: 1,002,753 records, 0
    multi-allelic); its multi-allelic sites appear as several biallelic records
    sharing a POS, which is exactly the post-split form.

    Note what survives splitting: the per-ALT records still share a coordinate,
    so a pair of them sits at distance 0 with LD forced by the representation
    rather than by haplotype structure. See yuj1r0/cugen#18.
    """
    try:
        from cyvcf2 import VCF
        backend = "cyvcf2"
    except ImportError:
        try:
            import pysam
            backend = "pysam"
        except ImportError:
            raise ImportError("vcf2cugen needs cyvcf2 or pysam")

    def _open():
        # gts012=True is load-bearing, not a preference. cyvcf2's DEFAULT
        # gt_types encoding is 0=HOM_REF, 1=HET, 2=UNKNOWN, 3=HOM_ALT; only
        # with gts012=True does it become 0/1/2 dosage plus 3 for missing,
        # which is what the mapping below assumes and what ENCODING_2BIT
        # means. Without it every homozygous-ALT call is written as MISSING and
        # every missing call as dosage 2 -- silently, with plausible output.
        # Measured on real 1000 Genomes chr22: a variant with 2,321 `1|1`
        # genotypes produced 2,321 missing and zero dosage-2, and the spurious
        # HAS_MISSING flag then disabled the fused GPU scan, stream= and
        # count_only for the whole file.
        #
        # vcf2cugenh does not need this: it reads allele indices through
        # rec.genotypes, which gts012 does not affect.
        return (VCF(vcf, gts012=True) if backend == "cyvcf2"
                else pysam.VariantFile(vcf))

    v = _open()
    samples = list(v.samples if backend == "cyvcf2" else v.header.samples)
    keep_idx = None
    if keep:
        want = set(keep if isinstance(keep, (list, set, tuple))
                   else open(keep).read().split())
        keep_idx = [i for i, s in enumerate(samples) if s in want]
        samples = [samples[i] for i in keep_idx]
    n_samples = len(samples)

    it = (v(region) if region else v) if backend == "cyvcf2" else \
        (v.fetch(region=region) if region else v)

    def _alts(rec):
        return list(rec.ALT if backend == "cyvcf2" else (rec.alts or ()))

    def _where(rec):
        return (f"{rec.CHROM}:{rec.POS}" if backend == "cyvcf2"
                else f"{rec.chrom}:{rec.pos}")

    # The counting pass already visits every record, so the multi-allelic check
    # is free here and fails before a single byte is written.
    n_var = n_multi = 0
    first_multi = None
    for rec in it:
        n_var += 1
        if len(_alts(rec)) > 1:
            n_multi += 1
            if first_multi is None:
                first_multi = _where(rec)
    v.close() if backend == "cyvcf2" else v.close()
    if n_multi:
        raise ValueError(
            f"{vcf}: {n_multi:,} multi-allelic record(s), first at "
            f"{first_multi}. A site with more than one ALT has no honest "
            f"0/1/2 dosage column, and the two backends disagree about how to "
            f"force one. Measured on REF=G ALT=A,C: cyvcf2 pools every ALT "
            f"into a single non-reference allele (0|2 -> 1, 1|2 -> 1, mu_x "
            f"1.1111), while pysam writes the 0|2 heterozygote as a homozygote "
            f"and loses 1|2 and 2|2 to the missing code (mu_x 1.3333) -- so "
            f"the same file used to convert differently depending on which "
            f"library happened to be installed. Split upstream instead; that "
            f"renormalises REF/ALT, which cannot be done from genotypes "
            f"alone:\n"
            f"    bcftools norm -m -any {vcf} -Oz -o split.vcf.gz\n"
            f"    bcftools index -t split.vcf.gz\n"
            f"Splitting is lossless for per-allele dosage: a 1|2 genotype "
            f"becomes dosage 1 at each of the two records.")
    if verbose:
        print(f"vcf2cugen [{backend}]: {n_samples:,} samples x {n_var:,} "
              f"variants{f' in {region}' if region else ''} -> {out}")

    v = _open()
    it = (v(region) if region else v) if backend == "cyvcf2" else \
        (v.fetch(region=region) if region else v)
    with CugenWriter(out, n_samples, n_var, ENCODING_2BIT) as w:
        for k, rec in enumerate(it):
            if backend == "cyvcf2":
                # gts012=True, so 0,1,2 are dosages and 3 is UNKNOWN.
                gt = rec.gt_types.astype(np.float64)
                d = np.where(gt == 3, 3.0, gt)
            else:
                d = np.empty(len(rec.samples), dtype=np.float64)
                for i, s in enumerate(rec.samples.values()):
                    a = s.get("GT")
                    d[i] = 3.0 if a is None or None in a else float(sum(a))
            if keep_idx is not None:
                d = d[keep_idx]
            w.add_variant(gidx_start + k, d)
            if verbose:
                _progress(k, n_var)
    v.close()
    _write_samples(out, samples)
    return out


def bed2cugen(bed, out, bim=None, fam=None, gidx_start=0, verbose=True):
    """PLINK1 .bed -> .cugen. Pure numpy, no external dependency.

    PLINK1 packs 2 bits per genotype LITTLE-endian within the byte, with codes
    00=hom-A1, 01=MISSING, 10=het, 11=hom-A2. cugen uses BIG-endian within the
    byte with 0/1/2 dosage and 3=missing, so every genotype is remapped.
    """
    bim = bim or os.path.splitext(bed)[0] + ".bim"
    fam = fam or os.path.splitext(bed)[0] + ".fam"
    ids = [l.split()[1] for l in open(fam) if l.strip()]
    n_samples = len(ids)
    n_var = sum(1 for l in open(bim) if l.strip())

    with open(bed, "rb") as f:
        magic = f.read(3)
        if magic[:2] != b"\x6c\x1b":
            raise ValueError(f"{bed}: not a PLINK1 .bed (magic {magic[:2]!r})")
        if magic[2] != 1:
            raise ValueError(f"{bed}: sample-major .bed unsupported; "
                             f"re-encode with --make-bed")
        bpv = (n_samples + 3) // 4
        if verbose:
            print(f"bed2cugen: {n_samples:,} samples x {n_var:,} variants -> {out}")
        # PLINK1 code -> dosage; 1 (01) is missing
        lut = np.array([0.0, 3.0, 1.0, 2.0], dtype=np.float64)
        with CugenWriter(out, n_samples, n_var, ENCODING_2BIT) as w:
            for k in range(n_var):
                raw = np.frombuffer(f.read(bpv), dtype=np.uint8)
                codes = np.empty(raw.size * 4, dtype=np.uint8)
                codes[0::4] = raw & 3           # little-endian within byte
                codes[1::4] = (raw >> 2) & 3
                codes[2::4] = (raw >> 4) & 3
                codes[3::4] = (raw >> 6) & 3
                w.add_variant(gidx_start + k, lut[codes[:n_samples]])
                if verbose:
                    _progress(k, n_var)
    _write_samples(out, ids)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="cugen-convert",
                                 description="build .cugen from pgen/vcf/bed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pgen"); p.add_argument("input"); p.add_argument("output")
    p.add_argument("--psam"); p.add_argument("--pvar")
    v = sub.add_parser("vcf"); v.add_argument("input"); v.add_argument("output")
    v.add_argument("--region"); v.add_argument("--keep")
    v.add_argument("--gidx-start", type=int, default=0)
    b = sub.add_parser("bed"); b.add_argument("input"); b.add_argument("output")
    b.add_argument("--bim"); b.add_argument("--fam")
    b.add_argument("--gidx-start", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "pgen":
        pgen2cugen(a.input, a.output, psam=a.psam, pvar=a.pvar)
    elif a.cmd == "vcf":
        vcf2cugen(a.input, a.output, region=a.region, keep=a.keep,
                  gidx_start=a.gidx_start)
    else:
        bed2cugen(a.input, a.output, bim=a.bim, fam=a.fam,
                  gidx_start=a.gidx_start)
    from .write import validate_cugen
    validate_cugen(a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())


def vcf2cugenh(vcf, out, region=None, keep=None, gidx_start=0,
               gidx=None, require_phased=True, verbose=True):
    """PHASED VCF/BCF -> phased .cugen (ENCODING_HAP2BIT).

    The counterpart of vcf2cugen for haplotype data. Every genotype must be
    diploid, phased and non-missing -- the requirement Beagle places on a
    reference panel, and the one cugen.impute inherits.

    require_phased=True rejects a record carrying an unphased genotype. That
    check matters more than it looks: an unphased het written as if it were
    phased silently invents a haplotype assignment, and the resulting panel
    still imputes, still returns probabilities in [0, 1], and is simply wrong in
    a way no downstream check can detect. Stock 1000 Genomes VCFs are phased;
    many other sources are not, and some are phased for homozygotes only.

    Multi-allelic records are SKIPPED (counted in the verbose line), not
    refused: hap2bit stores one bit per haplotype and so is biallelic by
    construction, and a reference panel missing a site still imputes correctly
    at the sites it has. Skipping does drop them silently from the panel, so
    split first if you need them -- see vcf2cugen for why cugen does not do
    this itself:

        bcftools norm -m -any in.vcf.gz -Oz -o split.vcf.gz

    `gidx` gives an explicit global index per written record, which matters
    whenever this file will be matched against another: cugen.impute pairs a
    target with its reference THROUGH gidx, so numbering each file 0..n-1
    independently silently aligns target marker 0 with reference marker 0 and
    shifts every marker after it. Nothing fails; the imputation simply uses the
    wrong reference markers.
    """
    try:
        from cyvcf2 import VCF
        backend = "cyvcf2"
    except ImportError:
        try:
            import pysam
            backend = "pysam"
        except ImportError:
            raise ImportError("vcf2cugenh needs cyvcf2 or pysam")

    from .write import ENCODING_HAP2BIT

    def _open():
        return VCF(vcf) if backend == "cyvcf2" else pysam.VariantFile(vcf)

    def _iter(v):
        if backend == "cyvcf2":
            return v(region) if region else v
        return v.fetch(region=region) if region else v

    def _biallelic(rec):
        alts = rec.ALT if backend == "cyvcf2" else list(rec.alts or ())
        return len(alts) == 1

    v = _open()
    samples = list(v.samples if backend == "cyvcf2" else v.header.samples)
    keep_idx = None
    if keep:
        want = set(keep if isinstance(keep, (list, set, tuple))
                   else open(keep).read().split())
        keep_idx = [i for i, s in enumerate(samples) if s in want]
        samples = [samples[i] for i in keep_idx]
    n_samples = len(samples)

    n_var = n_skip = 0
    for rec in _iter(v):
        if _biallelic(rec):
            n_var += 1
        else:
            n_skip += 1
    v.close()
    if verbose:
        print(f"vcf2cugenh [{backend}]: {n_samples:,} samples x {n_var:,} "
              f"biallelic variants{f' in {region}' if region else ''} -> {out}"
              + (f"  ({n_skip:,} multi-allelic skipped)" if n_skip else ""))
    if n_var == 0:
        raise ValueError(f"{vcf}: no biallelic records"
                         f"{f' in {region}' if region else ''}")

    if gidx is not None:
        gidx = np.asarray(gidx, dtype=np.int64)
        if gidx.size != n_var:
            raise ValueError(
                f"gidx has {gidx.size} entries but {n_var} biallelic records "
                f"will be written")
    v = _open()
    k = 0
    with CugenWriter(out, n_samples, n_var, ENCODING_HAP2BIT) as w:
        for rec in _iter(v):
            if not _biallelic(rec):
                continue
            if backend == "cyvcf2":
                # rec.genotype.array() hands back a numpy array directly;
                # rec.genotypes builds a Python list-of-lists per record, and
                # np.asarray on it dominates the whole conversion at
                # chromosome scale. Same layout either way: allele columns
                # then a phase flag, with -1 for missing.
                ga = getattr(rec, "genotype", None)
                if ga is not None and hasattr(ga, "array"):
                    arr = ga.array()
                else:
                    arr = np.asarray(rec.genotypes, dtype=np.int16)
                a = arr[:, :2].astype(np.int64, copy=False)
                phased = arr[:, -1].astype(bool)
            else:
                a = np.empty((len(rec.samples), 2), dtype=np.int64)
                phased = np.empty(len(rec.samples), dtype=bool)
                for i, s in enumerate(rec.samples.values()):
                    al = s.get("GT")
                    a[i] = (-1, -1) if al is None or len(al) != 2 else al
                    phased[i] = bool(s.phased)
            if keep_idx is not None:
                a = a[keep_idx]
                phased = phased[keep_idx]
            if (a < 0).any():
                bad = int(np.flatnonzero((a < 0).any(axis=1))[0])
                raise ValueError(
                    f"{vcf}: missing genotype for sample {samples[bad]!r} at "
                    f"record {k} ({rec.CHROM if backend == 'cyvcf2' else rec.chrom}"
                    f":{rec.POS if backend == 'cyvcf2' else rec.pos}). A phased "
                    f".cugen has no code for missing; impute or drop first.")
            # Homozygotes are unambiguous whether or not the record marks them
            # phased, so only heterozygotes need the flag.
            if require_phased:
                het = a[:, 0] != a[:, 1]
                if np.any(het & ~phased):
                    bad = int(np.flatnonzero(het & ~phased)[0])
                    raise ValueError(
                        f"{vcf}: UNPHASED heterozygote for sample "
                        f"{samples[bad]!r} at record {k}. Writing it as phased "
                        f"would invent a haplotype assignment that was never "
                        f"observed, and the panel would still impute and still "
                        f"look correct. Phase first, or pass "
                        f"require_phased=False if you know what you are doing.")
            g = gidx_start + k if gidx is None else int(gidx[k])
            w.add_variant_phased(g, a.astype(np.uint8))
            k += 1
            if verbose:
                _progress(k, n_var)
    v.close()
    _write_samples(out, samples)
    return out


def merge_cugen(paths, out, gidx_start=0, verbose=True):
    """Concatenate per-chromosome .cugen files into one genome-wide file.

    Cross-chromosome LD is not expressible against a directory: ld_matrix takes
    a SINGLE .cugen. A genome-wide all-pairs scan therefore needs every variant
    in one file, with gidx numbered continuously so downstream joins still
    identify variants uniquely.

    Refuses rather than guesses on the two mismatches that would otherwise
    produce a well-formed and wrong file:

    * differing n_samples -- concatenating two cohorts on the variant axis
      silently pairs sample i of one with sample i of the other.
    * differing encoding -- 2bit and hap2bit share bytes but not meaning (see
      cugen.write), so a mixed file decodes correctly for part of its range and
      wrongly for the rest, with no structural signal that anything is amiss.

    Variants keep source order; the caller controls chromosome order by the
    order of `paths`.
    """
    from .io import read_cugen, read_cugen_header
    from .write import ENCODING_HAP2BIT, unpack_2bit, unpack_hap2bit
    paths = [str(x) for x in paths]
    if not paths:
        raise ValueError("no input paths given")

    heads = [read_cugen_header(x) for x in paths]
    ns = {int(h["n_samples"]) for h in heads}
    if len(ns) != 1:
        raise ValueError(
            f"cannot merge: n_samples differs across inputs ({sorted(ns)}). "
            f"These are different cohorts, not different chromosomes.")
    encs = {h["encoding"] for h in heads}
    if len(encs) != 1:
        raise ValueError(
            f"cannot merge: encoding differs across inputs ({sorted(encs)}). "
            f"2bit and hap2bit share bytes but not meaning, so a mixed file "
            f"would decode wrongly for part of its range.")

    n_samples = ns.pop()
    enc_name = encs.pop()
    total = sum(int(h["n_variants"]) for h in heads)
    enc = ENCODING_HAP2BIT if enc_name == "hap2bit" else ENCODING_2BIT
    if verbose:
        print(f"merge_cugen: {len(paths)} files, {total:,} variants x "
              f"{n_samples:,} samples, encoding={enc_name} -> {out}")

    k = 0
    with CugenWriter(out, n_samples, total, encoding=enc) as w:
        for path, h in zip(paths, heads):
            r = read_cugen(path)
            bpv = int(r.bytes_per_variant)
            packed = np.frombuffer(r.read_packed_bytes(), dtype=np.uint8)
            nv = int(h["n_variants"])
            for v in range(nv):
                rec = packed[v * bpv:(v + 1) * bpv]
                # Decode through the encoding's OWN unpacker and re-add. Both
                # round-trip exactly, and this keeps the writer's per-variant
                # stats (mu_x, sxx, maf) correct -- a raw byte copy would
                # produce a file whose stats block describes nothing.
                if enc == ENCODING_HAP2BIT:
                    w.add_variant_phased(gidx_start + k,
                                         unpack_hap2bit(rec, 2 * n_samples))
                else:
                    w.add_variant(gidx_start + k, unpack_2bit(rec, n_samples))
                k += 1
            if verbose:
                print(f"  {path}: +{nv:,} variants ({k:,}/{total:,})")
    return out
