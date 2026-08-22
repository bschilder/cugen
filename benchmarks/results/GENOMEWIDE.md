# Cross-chromosome (genome-wide) LD

`ld_matrix` takes a SINGLE `.cugen`, so before this there was no way to compute
LD between variants on different chromosomes -- the pair space was necessarily
block-diagonal. `cugen.convert.merge_cugen` concatenates per-chromosome files
into one, with gidx numbered continuously, which makes a true genome-wide
all-pairs scan expressible.

## Validated on real data

chr21 + chr22, 1000 Genomes phase 3, `--max-alleles 2 --snps-only --maf 0.01`:

    chr21                170,073 variants
    chr22                170,949 variants
    merged               341,022 variants x 2,504 samples, 0.22 GB

**The merge does not perturb existing pairs.** The same 2,000 chr21 variants,
scanned from chr21 alone and from the merged file:

    chr21-alone rows   42,179
    merged-prefix rows 42,179
    differing               0        VALIDATED

That check is the one worth running. A concatenation that silently shifted
variants would still produce the right variant COUNT and a plausible result.

## Cross-chromosome all-pairs, measured

    p        341,022
    pairs     58,147,831,731
    wall              7.308 s   (1x A100-SXM4-80GB, median of 2)
    rows         27,232,751
    throughput    7.957e9 pairs/s

## Throughput does not degrade off the block diagonal

chr22 alone measured 8.4e9 pairs/s; this cross-chromosome scan measures
7.957e9 -- within 6%. The tiling is oblivious to chromosome boundaries, so
genome-wide is a SCALE problem, not a new regime, and projections from a single
chromosome carry over.

## Projection to the full genome, MAF >= 1% (about 13.7 M variants)

Using measured throughput: cugen 7.957e9 pairs/s (this file), plink2 3.46e8
pairs/s (128 physical cores, full chr22), qLD-GPU 1.98e8 pairs/s.

| scope | pairs | cugen | plink2 @128 | qLD-GPU |
|---|---|---|---|---|
| per-chromosome all-pairs | 5.08e12 | 10.6 min | 4.1 h | 7.1 h |
| true cross-chromosome | 9.38e13 | **3.3 h** | 3.1 d | 5.5 d |

At $1.59/hr the cugen cross-chromosome scan is about **$5.20**.

**plink2 cannot run either row at any time budget.** Its RSS grows as p^2 and
passes 1 TiB at roughly 524,000 variants per chromosome -- below the 1.19 M
common variants on chr1, let alone 13.7 M genome-wide. The day figures above
are what it would cost if the memory existed.

## Caveats

* Two chromosomes, not 22. The 5.815e10 pairs measured here is 0.06% of the
  9.38e13 a full genome-wide scan implies, so the projection is an
  extrapolation of 1,600x on pair count -- justified only by throughput being
  flat between the block-diagonal and cross-chromosome cases, which is the
  measurement above.
* Output volume is the unexamined constraint at full scale. 27.2 M rows here
  extrapolates to billions genome-wide; see the storage analysis for formats.
* min_r2 = 0.2 throughout. Emission scales roughly linearly in p and the
  threshold is the dominant lever on output size.

## Reproducing

    bash benchmarks/genomewide.sh
