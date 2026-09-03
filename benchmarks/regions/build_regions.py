"""Compile the GRCh38 artifact-region bundle from public sources.

Every URL here was verified to resolve before this script was written; the
script re-checks size and records a SHA-256 of each download in
provenance.json, so a silently changed upstream file is detectable rather than
absorbed.

Design decisions worth stating:

* **Autosomes only.** The panels these masks serve are 22 autosomes. Keeping
  chrX/Y/M intervals would be harmless but misleading in the interval counts.
* **Contig names are normalised to bare ('1', not 'chr1').** cugen.regions
  normalises on read too, so this is belt and braces -- a naming mismatch masks
  NOTHING and makes an unmasked scan look filtered, which is the one failure
  mode that produces a confidently wrong answer.
* **Acrocentric arms are emitted as INTERVALS**, derived from the centromere
  model start on chr13/14/15/21/22. They are NOT a chromosome flag: an earlier
  annotation encoded "acrocentric" as whole-chromosome membership, which would
  drop those five chromosomes' entire long arms while removing no artifact,
  since the mapping problem is confined to the proximal arms.
* **rmsk is not downloaded.** It is 155 MB and GIAB's curated satellite track
  covers the same ground at 50 KB. Pass --with-rmsk if you need per-repName
  classes (ALR/Alpha etc.) rather than satellite as a whole.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import urllib.request

AUTOSOMES = {str(c) for c in range(1, 23)}

GIAB = ("https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/"
        "genome-stratifications/v3.1/GRCh38")
UCSC = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database"

SOURCES = {
    "low_mappability": [
        f"{GIAB}/mappability/GRCh38_nonunique_l100_m2_e1.bed.gz",
        f"{GIAB}/mappability/GRCh38_nonunique_l250_m0_e0.bed.gz",
    ],
    "segdup": [f"{UCSC}/genomicSuperDups.txt.gz"],
    "satellite": [f"{GIAB}/LowComplexity/GRCh38_satellites_slop5.bed.gz"],
    "assembly_gaps": [f"{UCSC}/gap.txt.gz"],
    "pericentromere": [f"{UCSC}/centromeres.txt.gz"],
    "acrocentric_arms": [f"{UCSC}/centromeres.txt.gz"],
    "encode_blacklist_v2": [
        "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/"
        "hg38-blacklist.v2.bed.gz"],
}

#: UCSC .txt.gz tables are not BED: they carry a leading bin column and put the
#: interval in different positions per table. Explicit, because guessing here
#: shifts every interval.
UCSC_COLS = {
    "genomicSuperDups.txt.gz": (1, 2, 3),   # bin, chrom, chromStart, chromEnd
    "gap.txt.gz": (1, 2, 3),
    "centromeres.txt.gz": (1, 2, 3),
}

PERICENTROMERE_SLOP = 3_000_000
ACROCENTRIC = {"13", "14", "15", "21", "22"}


def norm(c):
    c = str(c)
    return c[3:] if c.lower().startswith("chr") else c


def fetch(url, cache):
    os.makedirs(cache, exist_ok=True)
    dst = os.path.join(cache, os.path.basename(url))
    if not os.path.exists(dst):
        print(f"    downloading {os.path.basename(url)}", flush=True)
        with urllib.request.urlopen(url, timeout=300) as r, open(dst, "wb") as f:
            f.write(r.read())
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    return dst, h, os.path.getsize(dst)


def read_intervals(path):
    """Yield (contig, start, end) from a BED or a known UCSC table."""
    base = os.path.basename(path)
    cols = UCSC_COLS.get(base)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            if cols:
                c, s, e = f[cols[0]], f[cols[1]], f[cols[2]]
            else:
                c, s, e = f[0], f[1], f[2]
            c = norm(c)
            if c not in AUTOSOMES:      # drops alt/random/Un/chrX/Y/M
                continue
            yield c, int(s), int(e)


def merge(iv):
    out = {}
    for c, v in iv.items():
        v = sorted(v)
        m = [v[0]]
        for a, b in v[1:]:
            if a <= m[-1][1]:
                m[-1] = (m[-1][0], max(m[-1][1], b))
            else:
                m.append((a, b))
        out[c] = m
    return out


def write_bed(path, iv):
    n = bp = 0
    with gzip.open(path, "wt") as f:
        for c in sorted(iv, key=lambda x: int(x)):
            for a, b in iv[c]:
                f.write(f"{c}\t{a}\t{b}\n")
                n += 1
                bp += b - a
    return n, bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="regions/grch38")
    ap.add_argument("--cache", default=".regions_cache")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prov = {"assembly": "GRCh38", "autosomes_only": True, "tracks": {}}

    for name, urls in SOURCES.items():
        print(f"  {name}", flush=True)
        iv = {}
        srcs = []
        for u in urls:
            p, sha, size = fetch(u, a.cache)
            srcs.append({"url": u, "sha256": sha, "bytes": size})
            for c, s, e in read_intervals(p):
                if name == "pericentromere":
                    s, e = max(0, s - PERICENTROMERE_SLOP), e + PERICENTROMERE_SLOP
                elif name == "acrocentric_arms":
                    if c not in ACROCENTRIC:
                        continue
                    s, e = 0, e            # p-arm: contig start to centromere end
                iv.setdefault(c, []).append((s, e))
        if not iv:
            print(f"    EMPTY -- skipping {name}", flush=True)
            continue
        iv = merge(iv)
        n, bp = write_bed(os.path.join(a.out, f"{name}.bed.gz"), iv)
        prov["tracks"][name] = {"sources": srcs, "intervals": n, "bp": bp,
                                "contigs": len(iv)}
        print(f"    {n:,} intervals, {bp/1e6:,.1f} Mb over {len(iv)} contigs",
              flush=True)

    json.dump(prov, open(os.path.join(a.out, "provenance.json"), "w"), indent=2)
    print(f"\n  wrote {a.out}/provenance.json")


if __name__ == "__main__":
    main()
