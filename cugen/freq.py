"""Frequency utilities — cugen header is the source of truth for MAF / sxx.

Also hosts the per-ancestry ascertainment rule. A panel filtered on POOLED
allele frequency silently discards variants common inside one ancestry group
but rare overall: a variant private to a group holding fraction f of the cohort
has pooled AF ~= AF_group * f, so a pooled ``--maf t`` drops everything with
AF_group < t/f. On 1kGP superpopulation fractions that band reaches AF < 3.8%
for AFR and AF < 7.2% for AMR -- common variants inside their own group, and
the most population-differentiated slice of the frequency spectrum, which is
exactly what a multi-ancestry LD reference exists to describe.

All of Us ships its ACAF callset on the opposite rule -- "AF > 1% or AC > 100 in
ANY computed ancestry subpopulation" -- which is what :func:`union_maf_pass`
implements.
"""

from pathlib import Path

from .io import CugenReader


def pooled_af(af_in_group: float, group_fraction: float,
              af_elsewhere: float = 0.0) -> float:
    """Pooled allele frequency given an in-group frequency and group size.

    Makes the pooled-filter defect arithmetic rather than argument.
    """
    f = float(group_fraction)
    return float(af_in_group) * f + float(af_elsewhere) * (1.0 - f)


def read_afreq(path) -> dict:
    """``{variant_id: alt_frequency}`` from a plink2 ``--freq`` table.

    Columns are located BY NAME: plink2's column set varies with ``--freq
    cols=``, and positional parsing reads the wrong field silently. Rows whose
    frequency will not parse are skipped -- plink2 writes ``NA`` for a variant
    with no observations in the subset, and a crash there would take down a
    22-chromosome build partway through.
    """
    out = {}
    with open(path) as f:
        header = f.readline().lstrip("#").split()
        try:
            i_id, i_af = header.index("ID"), header.index("ALT_FREQS")
        except ValueError as exc:
            raise ValueError(
                f"{path}: expected plink2 --freq columns ID and ALT_FREQS, "
                f"got {header}") from exc
        for line in f:
            parts = line.split()
            if len(parts) <= max(i_id, i_af):
                continue
            try:
                out[parts[i_id]] = float(parts[i_af])
            except ValueError:
                continue                      # NA / missing
    return out


def pooled_from_groups(freqs_by_group: dict, sizes_by_group: dict) -> dict:
    """Pooled allele frequency per variant: the SAMPLE-WEIGHTED group mean.

    Weighting matters -- AFR is 661/2504 of 1kGP and AMR 347/2504, so an
    unweighted mean would misstate the pooled frequency and therefore the size
    of the ascertainment gap this module exists to quantify.

    A variant absent from a group's table is treated as AF 0 in that group,
    which is the OPPOSITE of :func:`union_maf_pass`, where absent must not mean
    zero. The two rules differ deliberately: for a pooled average a monomorphic
    group genuinely contributes zero copies, while for "common in any group" an
    unscored group must not be able to veto.
    """
    ids = set()
    for t in freqs_by_group.values():
        ids |= set(t)
    total = float(sum(sizes_by_group.get(g, 0) for g in freqs_by_group))
    if total <= 0:
        raise ValueError("sizes_by_group must give a positive total sample count")
    out = {}
    for vid in ids:
        num = sum(float(freqs_by_group[g].get(vid, 0.0)) *
                  float(sizes_by_group.get(g, 0)) for g in freqs_by_group)
        out[vid] = num / total
    return out


def union_maf_pass(freqs_by_group: dict, maf_min: float) -> set:
    """Variant IDs reaching ``maf_min`` in AT LEAST ONE group -- the ACAF rule.

    A UNION, deliberately. An intersection would keep only globally-common
    variants, which is the same ascertainment a pooled filter imposes by
    another route.

    Frequencies are folded (``min(af, 1-af)``) because plink2 reports the ALT
    frequency, which may exceed 0.5. A group missing a variant contributes
    nothing rather than a zero, so one monomorphic group cannot veto a variant
    that is common in another.
    """
    t = float(maf_min)
    keep = set()
    for table in freqs_by_group.values():
        for vid, af in table.items():
            a = float(af)
            if min(a, 1.0 - a) >= t:
                keep.add(vid)
    return keep


def frequency(cugen_path):
    """Read precomputed (mu_x, sxx, maf) arrays from a cugen header.

    Returns a dict with arrays of length n_variants — no genotype I/O.
    """
    p = Path(cugen_path)
    with CugenReader(str(p)) as r:
        mu_x, sxx, maf = r.get_stats(0, r.n_variants)
        return {
            "n_variants": r.n_variants,
            "n_samples": r.n_samples,
            "mu_x": mu_x,
            "sxx": sxx,
            "maf": maf,
        }
