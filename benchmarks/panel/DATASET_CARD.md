---
license: other
license_name: 1000-genomes-terms
license_link: https://www.internationalgenome.org/data
pretty_name: cugen 1kGP 30x GRCh38 LD panels
language:
  - en
tags:
  - genomics
  - linkage-disequilibrium
  - population-genetics
  - 1000-genomes
configs: []
---

# cugen — 1000 Genomes 30x GRCh38 LD reference panels

Genotype panels and precomputed linkage-disequilibrium for the 1000 Genomes
Project 30x high-coverage GRCh38 release, built with
[cugen](https://github.com/bschilder/cugen).

**2,504 unrelated samples · 20,276,768 biallelic SNVs · 22 autosomes · phase preserved**

## What makes this panel different

Variants are ascertained **per superpopulation**, not on the pooled cohort. A
variant is kept if its minor-allele frequency reaches 1% in **any** of AFR, AMR,
EAS, EUR or SAS — the same rule All of Us uses for its ACAF callset.

This matters more than it sounds. A variant private to a group holding fraction
`f` of the cohort has pooled `AF ≈ AF_group × f`, so a pooled `--maf 0.01`
filter discards it whenever `AF_group < 0.01/f`:

| superpopulation | n | share | within-group AF a pooled filter would drop |
|---|---|---|---|
| AFR | 661 | 0.264 | 1.00% ≤ AF < 3.79% |
| EAS | 504 | 0.201 | 1.00% ≤ AF < 4.97% |
| EUR | 503 | 0.201 | 1.00% ≤ AF < 4.98% |
| SAS | 489 | 0.195 | 1.00% ≤ AF < 5.12% |
| AMR | 347 | 0.139 | 1.00% ≤ AF < 7.22% |

Those are common variants inside their own group, and they are the most
population-differentiated slice of the frequency spectrum — exactly what a
multi-ancestry LD reference exists to describe.

Measured genome-wide:

| ascertainment | variants |
|---|---|
| unfiltered biallelic SNV | 61,599,150 |
| **per-superpopulation MAF ≥ 1% (this panel)** | **20,276,768** |
| pooled MAF ≥ 1% | 12,528,011 |
| **kept here, absent from a pooled panel** | **7,748,757 — 38.2%** |

`k = union / pooled = 1.6185`, and it is stable across chromosomes (1.580–1.654),
so this is a property of pooled ascertainment rather than an artifact of any one
region. For scale: **the AFR panel alone (1,212,567 chr1 variants) contains more
common variants than a pooled all-sample panel does (990,996).**

One honest caveat: taking the maximum over five groups gives a rare variant five
chances to clear the threshold. At these group sizes a variant needs ≥ 7 minor
alleles, and P(spurious inclusion) is 0.06% at true AF 0.002, 12% at 0.005 and
73% at 0.008. The leak sits just below threshold, where r² is bounded anyway.

## Layout

```
1kg-30x-grch38/
├── perchrom/
│   ├── chrN.cugen              unphased, 2-bit packed (ceil(n/4) B/variant)
│   ├── chrN_ph.cugen           phased haplotypes
│   ├── chrN.bed/.bim/.fam      plink2, same variant set and order
│   ├── superpop/{POP}/chrN.cugen + .bim + .fam
│   └── union/
│       ├── chrN.union.txt              variant IDs kept
│       └── chrN.ascertainment.tsv      unfiltered / union / pooled / union-only
└── ld/
    ├── {POP}/ld_un_cis/chrN.cugenld/   within-chromosome LD, unphased
    ├── ALL/ld_ph_cis/chrN.cugenld/       within-chromosome LD, phased
    └── provenance.json                  scan parameters and conventions
```

`POP` ∈ `ALL, AFR, AMR, EAS, EUR, SAS`. Genome-wide (cross-chromosome) LD is
~1.6 TB and lives on object storage rather than here; the within-chromosome
datasets are the ones most analyses want.

### Per-chromosome variant counts

| chromosome | this panel | pooled would give | never in a pooled panel |
|---|---|---|---|
| chr1 | 1,621,202 | 990,996 | 630,206 |
| chr2 | 1,699,441 | 1,037,587 | 661,854 |
| chr3 | 1,431,477 | 884,354 | 547,123 |
| chr4 | 1,437,180 | 893,807 | 543,373 |
| chr5 | 1,304,495 | 796,063 | 508,432 |
| chr6 | 1,280,219 | 810,423 | 469,796 |
| chr7 | 1,196,278 | 738,529 | 457,749 |
| chr8 | 1,116,290 | 688,941 | 427,349 |
| chr9 | 910,757 | 559,993 | 350,764 |
| chr10 | 1,017,634 | 635,303 | 382,331 |
| chr11 | 989,309 | 614,855 | 374,454 |
| chr12 | 959,205 | 597,948 | 361,257 |
| chr13 | 732,278 | 454,189 | 278,089 |
| chr14 | 656,925 | 407,069 | 249,856 |
| chr15 | 607,811 | 374,021 | 233,790 |
| chr16 | 649,926 | 392,947 | 256,979 |
| chr17 | 561,866 | 344,198 | 217,668 |
| chr18 | 569,385 | 355,173 | 214,212 |
| chr19 | 462,614 | 289,891 | 172,723 |
| chr20 | 470,386 | 289,598 | 180,788 |
| chr21 | 296,653 | 183,645 | 113,008 |
| chr22 | 305,437 | 188,481 | 116,956 |

## Quickstart

```bash
pip install "git+https://github.com/bschilder/cugen.git@ld-rowblock"
```

```python
from huggingface_hub import hf_hub_download

REPO = "standardmodelbio/cugen"
def grab(path):
    return hf_hub_download(REPO, f"1kg-30x-grch38/{path}", repo_type="dataset")

# --- 1. read a genotype panel -------------------------------------------------
from cugen.io import read_cugen

r = read_cugen(grab("perchrom/chr22.cugen"))
print(r.n_samples, r.n_variants, r.phased, r.has_missing)   # 2504 305437 False False
print(r.maf[:5])                                            # precomputed, no I/O

# --- 2. read precomputed within-chromosome LD --------------------------------
from cugen.ldio import open_ld
import numpy as np, os

# A .cugenld is a DIRECTORY of shards plus a manifest, so fetch it as a folder.
from huggingface_hub import snapshot_download
d = snapshot_download(REPO, repo_type="dataset",
                      allow_patterns="1kg-30x-grch38/ld/ALL/ld_un_cis/chr22.cugenld/*")
ld = open_ld(os.path.join(d, "1kg-30x-grch38/ld/ALL/ld_un_cis/chr22.cugenld"))

i, j, r_val = ld.above(min_r2=0.8)      # skips whole shards below the threshold
print(f"{i.size:,} pairs at r2 >= 0.8")

# --- 3. map indices back to variants ------------------------------------------
import pandas as pd

bim = pd.read_csv(grab("perchrom/chr22.bim"), sep="\t", header=None,
                  names=["chrom", "id", "cm", "pos", "a1", "a2"])
# i and j are row positions in the .cugen, which is the .bim order.
top = np.argsort(-(r_val ** 2))[:5]
print(pd.DataFrame({
    "variant_i": bim.id.to_numpy()[i[top]],
    "variant_j": bim.id.to_numpy()[j[top]],
    "bp_apart": np.abs(bim.pos.to_numpy()[j[top]] - bim.pos.to_numpy()[i[top]]),
    "r2": r_val[top] ** 2,
}))
```

### Compute LD yourself

```python
from cugen.ld import ld_matrix

# Within-chromosome, every pair, streamed to a .cugenld dataset.
n = ld_matrix(grab("perchrom/chr22.cugen"), min_r2=0.1, stats=("r", "r2"),
              sign_reference="major", stream=True, output="chr22.cugenld",
              backend="gpu")
print(f"{n:,} pairs written")
```

## Two conventions that will bite you

**1. `r` is signed on the MINOR allele; `.cugen` dosages count ALT.**
These datasets were written with `sign_reference="major"`. Reconstructing a raw
cross-product from a stored `r` therefore needs
`r × (−1)^(major_i XOR major_j)`. Applying that correction to an `"alt"`-signed
file is as wrong as omitting it from a `"major"` one. `r²` is sign-invariant and
unaffected. Check `open_ld(...).params["sign_reference"]`; `None` means
*unrecorded*, which is not the same as `"alt"`.

Omitting this once produced `|r_adj| = 7.7` — impossible for a correlation —
behind a complete and plausible-looking results table.

**2. Shards are row-budgeted, not chromosome-aligned.**
Each shard spans a narrow slab of *both* variant axes. Sampling shards evenly is
not sampling variant pairs evenly; use `.above()` / `.region()`, which consult
the manifest, rather than reading a subset of shard files.

## How it was made

Fully scripted and reproducible, in
[bschilder/cugen](https://github.com/bschilder/cugen) under `benchmarks/panel/`:

| script | step |
|---|---|
| `chromjob.sh` | one chromosome: fetch VCF, verify published MD5, plink2 filter, per-superpop `--freq`, union, convert to `.cugen`/`.bed`, emit all six panels |
| `ldjob.sh` + `ldwrite_r2.py` | LD scan, `MODE=gw` (genome-wide) or `MODE=cis` (per-chromosome) |
| `prepjob.sh` | per-variant annotation, with gidx alignment asserted |
| `mirrorjob.sh` | publish to this dataset |

The ACAF ascertainment rule is `cugen.freq.union_maf_pass` /
`pooled_from_groups`, unit-tested in `tests/test_freq_union.py`.

**Source:** `1kGP_high_coverage_Illumina.chrN.filtered.SNV_INDEL_SV_phased_panel.vcf.gz`
from the 1000 Genomes 30x high-coverage collection
(`20220422_3202_phased_SNV_INDEL_SV`), restricted to the 2,504 unrelated samples.
Every VCF is verified against the published MD5 manifest.

**Filters:** `--max-alleles 2 --snps-only just-acgt`, then the per-superpopulation
union. Phase is preserved through `vcf2cugenh`, which asserts it rather than
assuming it.

**LD scans:** `min_r2=0.1`, `sign_reference="major"`, unphased `("r","r2")` and
phased `("r_phased","r2_phased")`. Run in fp32 on pre-Ampere GPUs and TF32 on
Ampere and later, where |Δr| ≈ 4.4e-4 — about 14× the int16 storage quantum of
3.05e-5, giving ±0.28% jitter at the r²=0.1 cut. Symmetric, and negligible for
aggregate statistics.

## Citation

Cite the 1000 Genomes Project 30x release (Byrska-Bishop et al., *Cell* 2022) for
the underlying data, and [cugen](https://github.com/bschilder/cugen) for the
panels and LD.
