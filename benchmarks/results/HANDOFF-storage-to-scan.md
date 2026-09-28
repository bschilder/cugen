# Handoff: what the storage layer gives the genome-wide scan

For whoever is optimising `cugen.ld` toward genome-wide all-by-all. This is the
storage/significance side of the same problem, on branch `ld-significance`
(bschilder/cugen PR #5). It is written to be *used*, not admired: the seams you
can call today, the numbers you do not need to re-derive, the traps that cost me
time, and the places our two workstreams collide.

Everything below is measured on real chr22 (51,100 variants at MAF >= 0.01,
1000G EUR) or on an RTX 4090 / L40S, unless it says otherwise.

---

## 1. The one result that should change your priorities

**The significance threshold, not the pair count, is the dominant lever on
output volume.** Measured, all pairs, one cohort:

| min_r2 | rows | rows/variant | % of all pairs | wall | size | B/pair |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 1,415,371 | 27.7 | 0.108% | — | — | — |
| 0.1 | 3,448,914 | 67.5 | 0.264% | — | — | — |
| 0.05 | 12,730,642 | 249.1 | 0.975% | 10.2 s | 25.6 MB | 2.01 |
| 0.02 | 66,765,884 | 1,306.6 | 5.114% | 31.0 s | 135.9 MB | 2.04 |
| 0.01 | 164,079,591 | 3,211.0 | 12.568% | — | — | 2.10 |

116x between 0.2 and 0.01, log-log slope **t^-1.64** — not the 1/t I had
assumed. Two consequences pointing opposite ways:

- Tightening the threshold buys *more* than 1/t suggests.
- The power law **cannot continue**. At r2 >= 0.01 the retained set is already
  12.6% of all pairs, so it saturates toward the full within-LD-span pair space.
  Extrapolating t^-1.64 into the biobank regime **overshoots**. The real ceiling
  is `p * (variants within the LD span)` — roughly 7e10 rows genome-wide for a
  ~1 Mb span — plus a trans contribution that stays alpha-limited.

**Why trans does not blow up.** At a family-wise threshold the expected number
of retained *null* pairs is alpha by construction, so raising the test count
raises the threshold exactly enough to compensate. Trans pairs are
self-limiting. What is *not* self-limiting is N: since chi2 = N * r^2, a larger
cohort pushes the threshold down and the cis partner count up. **Sample size is
the thing that makes this hard, not variant count.**

So: 1e6 to 1e11 rows across the useful threshold range. If your scan work
assumes output grows quadratically, it is sized for a regime that does not
happen.

---

## 2. Seams you can call today

| what | where | note |
|---|---|---|
| `write_ld(df, path, params=)` | `cugen/ldio.py` | extension-dispatched: `.cugenld`, `.parquet`, `.feather`, `.npz`, `.zarr`, `.tsv/.csv(.gz)`. `_write_df` fully delegates, so there is one writer contract now |
| `LDShardWriter(path, ...)` | `ldio.py` | `.append(i, j, r, n=None, presorted=True)`. **Pass `presorted=True`** if you sorted on device |
| `LDDatasetWriter` | `ldio.py` | sharded + `manifest.json` + resumable; `write_shard((A,B), i, j, r)` lands by atomic rename |
| `read_ld(path)` / `LDReader` | `ldio.py` | `.region()`, `.variant()`, `.above(min_r2=, max_p=, with_p=)`, `.dense()`, `.rows(with_p=)` |
| `ld_matrix(..., stream=True)` | `cugen/ld.py` | native `.cugenld` streaming branch; device-side `cp.lexsort` then `presorted=True` |
| `gpu_susie_rss_from_ld(R, z, n)` | `_step5b_finemapping.py` | consumes `LDReader.dense()`, so a stored panel can be fine-mapped |

### The two flags that matter most to you

**`presorted=True` is worth 88% of host write work.** Profiling the shard
writer: the host lexsort was 2.03 s of 2.31 s at 8 M rows. `cp.lexsort` does the
same 8 M rows in **0.031 s warm** on device. If you are already holding the
survivors on the GPU, sort there and tell the writer.

**`n=` carries per-pair N.** New as of this branch: p-values are recoverable
from every format. `.cugenld` does not *store* p (it is a closed form in
`N * r^2`, so a stored copy is 4-8 redundant bytes against a 2.0 B/pair
budget) — it derives it. Without missingness the scalar N in the header is
exact; with `missing="pairwise"` each pair rests on its own sample set, so N is
stored per pair as a narrow deficit. Measured cost: **0 for a constant-N file,
0.5-0.9 B/pair when N varies** (2.34 -> 2.87-3.24, i.e. 23-38%).

---

## 3. Numbers you do not need to re-derive

**Format comparison**, same pair set, all round-tripped and asserted equal:

| format | B/pair | vs TSV |
|---|---:|---:|
| `.tsv` | 77.57 | 1.0x |
| `.feather` | 16.85 | 4.6x |
| `.parquet` (zstd, tuned) | 7.57 | 10.2x |
| `.npz` | 6.17 | 12.6x |
| **`.cugenld`** | **3.21** (2.01-2.10 at scale) | **24.1x** |

**The streamed writer, four successive fixes:**

```
masks + 15 gathers, np.unique, identity gather   2.23 s   (baseline)
+ per-group flatnonzero partition, _run_starts   2.06 s   1.08x
+ no identity permutation gather                 1.86 s   1.20x
+ comparison-sum tier assignment                 1.58 s   1.41x total

vs parquet streaming: 5.16 s -> 1.58 s, 3.26x, and 6.1x smaller
```

1.32 s of the remaining 1.58 s is still serialisation, and it is now genuinely
distributed — no single term dominates. The next honest step is structural
(move tier assignment and packing onto the device, where r already lives), not
another micro-fix.

**Nine of thirteen output columns carry no information** on the device fast
path: `CHR` is constant per file, `POS` is written as literal zeros and `ID` as
`"."`, `MAF` is in the `.cugen` header, `R2 = R*R`, `NEG_LOG10_P` is closed-form
in `N_OBS * r2`. The fast path is gated on `annotation is None` — **exactly the
condition that makes those columns degenerate**, so the largest-output path was
writing the most dead bytes.

---

## 4. Six traps that cost me time

1. **Cold vs warm kernel timing.** I quoted `cp.lexsort` at 0.37 s; that was
   cold and included kernel compilation. Warm is **0.031 s** — a 12x error, and
   it inverted the conclusion about where time went. Always warm up.
2. **Never benchmark output onto a network volume.** Identical parquet writes
   swung **1.4-5.4 s** (3.9x on the same operation) on MooseFS. That was
   measuring the filesystem, not the writer. Source data can live on the volume;
   results cannot. Local container disk gave median/best agreeing within 5%.
3. **Guessing where time goes was wrong three times running** — the 2B^2 buffer,
   block size vs indexing, zstd vs packing. Every fix was cheap *once measured*.
   zstd turned out to be 0.005 s of 0.287 s (1.7%), which is why I dropped the
   nvCOMP idea entirely.
4. **A global stable argsort made the writer 0.80x — slower.** A full argsort of
   8 M int64 costs more than the twenty linear passes it replaced. Per-group
   `flatnonzero` is 0.103 s vs 0.246 s for the obvious mask form.
5. **Shards must hold a meaningful number of pairs.** Each pays a 256-byte
   header and its own footer, so over-sharding inverts the size win: 7.43 vs
   4.40 B/pair on a 115 k-pair fixture.
6. **`mutation_sweep.py` had `ROOT` hardcoded to an absolute developer path**
   and `.venv/bin/python` as the interpreter, so it had never once run on a GPU
   box. Fixed — now derived from `__file__` with `$CUGEN_PY`. If you add
   kernels, add sweep entries; GPU-path mutations only get checked where a GPU
   is.

---

## 5. Where the next wins are, ranked

1. **Move tier assignment + block packing onto the device.** 1.32 s of the
   1.58 s streamed write is host serialisation, and `r` is already on the GPU.
   Leave zstd on the host — it is 1.7%, not the bottleneck.
2. **Give the fused kernel per-row column bounds.** Today it takes `window` as
   a *scalar*, so every predicate needing per-row bounds falls off the fast path
   (see §6). `_pair_bounds` already returns exactly the `(starts, hi)` arrays
   the kernel would need, and the tiled path already consumes them.
3. **Tile-local uint16 indices** (20 -> 8 B/row) and **`max_output_gb`
   replacing `max_pairs`** — both already open in your issue #2. `max_pairs` is
   the wrong unit; bytes are what runs out.
4. **`variant()` builds its row-variant index at open in O(row variants) of
   Python.** At 84 M variants that is the cost to beat.
5. **Nothing in the reader is on the GPU.** All query latencies are
   single-threaded host decode.

---

## 6. Where our work collides — read this bit

**The fused-path gate now excludes more cases.** I added a `_needs_row_bounds`
condition:

```python
_needs_row_bounds = (window_kb is not None or min_dist_kb is not None
                     or max_dist_kb is not None or scope != "all")
fused = (on_device and not reader.has_missing and not _needs_row_bounds
         and min_obs <= reader.n_samples and fused_ok_phased)
```

`window_kb` already fell off the fused path before this; the new test-space
params join it for the same reason. **This means `stream=True` and `count_only`
are unavailable for cis/trans and bp-distance scans**, because they require the
fused kernel. Lifting item 2 above fixes all of them at once.

**`_scan_gpu`'s in-tile mask is now one range check, not five predicates:**

```python
keep = (jj >= st_d[ii]) & (jj < hi_d[ii]) & cp.isfinite(r) & (n >= min_obs)
```

Both the pair *count* (which sets `m`) and the emitted set come from the same
`_pair_bounds` call, so **the GPU path agrees with the reference by
construction** rather than by a parity test. If you touch tiling, keep that
property — it is what makes `m` trustworthy.

**`m` is closed-form and must stay that way.** `_count_pairs` derives the test
count from row count and bounds with no pass over data. That is the only reason
`correction="bonferroni"/"fdr"` is affordable at genome scale. Any change that
forces `m` to be counted empirically breaks multiple-testing correction.

**Two smaller ones:** `convert.py`'s `gts012=True` fix exists on both this
branch and `fix-vcf2cugen-hom-alt` (upstream PR #15) with different comment
text — whichever merges second conflicts on a comment hunk. And I removed my
own `phased=False` patch to `_clump_edges_rect_gpu` in favour of your version,
which deleted the branch outright with better reasoning.

---

## 7. What would help me

- **The `benchmarks/win_sweep.py` measurement that was launched and never
  recorded.** Whether cis partners/variant really scales like t^-1.64 outside
  one 10 Mb window of chr22 is the single largest uncertainty in the volume
  model above.
- **Any genome-wide characterisation at all.** Everything in §1 is one window,
  one cohort, one chromosome.
- **A non-random-missingness panel.** Pairwise complete-case R stayed
  positive-definite at every uniform missingness rate I tried (min eigenvalue
  +2.99e-1 at 0% to +1.25e-1 at 20%, `benchmarks/pairwise_vs_single_x.py`), but
  uniform missingness is the easy case — complete-case sample sets diverge much
  harder when missingness is structured.

## 8. Traps you flagged that I am carrying forward

Recorded here so they do not have to be rediscovered:

- `plink2 --r2-phased` on a `.bed` is **not a valid reference** — bed cannot
  store phase, so it runs EM. Deviation 0.616 vs 5.71e-07 against a phased PGEN.
- 1000G phase3 variant IDs are almost all `.`, which silently makes `--extract`
  match everything and blocks `--pmerge-list`. Use
  `--set-all-var-ids '@:#:$r:$a'`.
- **plink2's RSS is not quadratic in p.** Measured 99.5 GiB at p=943,790 vs
  109 GiB at p=170,949. plink2 is a real competitor at genome scale.
- qLD binaries exit nonzero on success. Gate on output, never on `$?`.
