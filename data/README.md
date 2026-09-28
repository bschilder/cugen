# Reference resources

## `lrld_grch38.txt`

The 24 long-range-LD regions conventionally excluded before PCA, on GRCh38.
Tab-separated: `chrom  start  end  name`. 135.7 Mb, 4.72% of the autosome.

Provenance, with the traps:

- The region **list** is Price et al. 2008, *AJHG* **83**:132–135 — not 86; the
  Genome Analysis Wiki miscites the volume. It is a Letter rebutting Tang et
  al., and it never recommends excluding these from PCA. The
  exclude-before-PCA instruction comes from Anderson et al. 2010, *Nat Protoc*
  5:1564 and Weale 2010, *MMB* 628. Cite accordingly.
- Only 3 of the 24 regions appear in Table 1 proper. The other 21 — including
  the chr8 inversion and LCT — are in the table's **footnote**, so a filter
  built from "Table 1" alone silently misses them.
- Do **not** substitute the GRCh38 list circulating via plinkQC and the wiki.
  It drops or truncates 7 of the 24 regions (29% of the intended span) and
  reduces **LCT to a 119 bp fragment** at chr2:135,275,091–135,275,210. This
  file is a proper liftover with the wider bounds retained: MHC 10.0 Mb,
  inv8p23 6.0 Mb, LCT 3.5 Mb.
- The list is **European-ascertained** (327 European Americans on Illumina
  550K). It is necessary but not sufficient for a panel that is 74%
  non-European; population-specific long-range LD outside the list is
  documented and, by construction, invisible to it.

Used to build the ancestry PC basis for `stats=("r_adj", "r2_adj")`. Note that
excluding these regions is *not* what Bercovich et al. 2025, gnomAD's
`get_qc_mt`, atgu's notebooks, or PCAone do — PCAone has no region-exclusion
flag in its LD path at all. Report both arms.
