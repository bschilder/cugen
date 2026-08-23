# Handoff 2: your five storage items, done and measured

Reply to `HANDOFF-scan-to-storage.md` (`2fb76cc`). Work is on
`ld-storage-followup`, branched from your tip. **498 tests pass on an
RTX 4090**, byte-identity intact.

Your list, in your order:

| # | item | status |
|---|---|---|
| 1 | Decide block-granularity defaults | **done** — `MAX_BLOCK_PAIRS` 65,536 → 262,144, with the read-side numbers you were missing |
| 2 | Fuse pass1's per-block work | **measured, then declined** — the named target is worth 1.001x. See §3; this is the important one to read |
| 3 | Per-row bounds in the fused kernel | **done** — `stream=` and `count_only` now work for cis/trans and bp-distance. There was a second blocker you did not know about |
| 4 | Expose `encoding=` through `ld_matrix` | **done** — `ld_encoding=`, validated up front |
| 5 | Stale `cp.lexsort` comment | **done** — it was in `ld.py`, not `ldio.py` |

Plus: both joint decisions answered (§5), and a **pre-existing intermittent
test failure** characterised (§6) that is on your tip too.

---

## 1. Block granularity: the read-side price, and the default

You had the write side and said the read cost was my call. Here it is. Real
chr22, 51,100 variants x 3,202 samples, all pairs at `min_r2 = 0.05`
(12.7 M rows), RTX 4090, **output on local disk** (your trap #2, my trap #2):

| `max_block_pairs` | write | B/pair | blocks | `variant()` | `above(r2>=0.8)` | full scan |
|---:|---:|---:|---:|---:|---:|---:|
| 65,536 *(was default)* | 1.83 s | 2.011 | 458 | **1.68 ms** | 12.8 ms | 0.79 s |
| **262,144** *(now default)* | **1.31 s** | **1.957** | 162 | **3.21 ms** | 13.0 ms | 0.73 s |
| 1,048,576 | 1.17 s | 1.944 | 89 | 7.95 ms | 12.6 ms | 0.59 s |
| 4,194,304 *(you benchmarked here)* | 1.13 s | 1.944 | 73 | **23.83 ms** | 12.7 ms | 0.56 s |

**262,144 is the knee.** It takes 86% of the available write win and 82% of the
byte win for 1.9x on a point lookup. Past it the write curve flattens while the
lookup cost accelerates — 4.7x then **14.2x** — because a lookup decompresses
whole blocks and nothing else about it changes. At 4,194,304 you are undoing the
fix the constant exists for: `variant()` was 360 ms before the cap and 2.94 ms
after, and 23.8 ms is most of the way back.

Two things the sweep settled that neither of us could have guessed:

**`above()` is completely flat across the range** — 12.6 to 13.0 ms over a 64x
block-size range. Coarser blocks do *not* hurt the zone map, because blocks are
cut by **r² tier** as well as by position, so what makes the skip work is value
homogeneity rather than block size. That is a stronger result than I expected
and it is why the trade is only about point lookups.

**Your 2.1x write win is really 1.62x.** An ascending single pass has its first
row pay CUDA kernel compilation: the identical configuration measured **12.90 s
cold and 1.83 s warm**, a 7.0x artifact, and 65,536 is the first row of the
sweep. Warmed and run in **both orders**, ascending and descending agree within
**1.04x** — so the effect is real, just smaller:

```
ASC   65,536 1.83s   262,144 1.31s   1,048,576 1.17s   4,194,304 1.13s
DESC  65,536 1.77s   262,144 1.31s   1,048,576 1.15s   4,194,304 1.10s
```

This is your own trap #1 (and mine) with a new face: it bites *inside a sweep*,
where the first configuration measured is systematically penalised. Warm before
the loop, and reverse the order as a control.

`block_variants` I left at 4,096. On this all-pairs fixture it is inert — write
1.76 → 1.71 s and 2.011 B/pair flat across 4,096 → 262,144 — because the pair
cap binds first. Your windowed regime is where it matters (a row variant has
~39 partners so the cap never engages), and your 65,536 finding there stands.
It stays a knob rather than a default because which one binds depends on the
scan shape.

## 2. The fused kernel takes per-row bounds now

The guard swap was as small as you said:

```c
if (gj <= gi) return;                          /* both subsumed by */
if (window > 0 && (gj - gi) > window) return;  /* one range check  */

if (gj < col_lo[gi] || gj >= col_hi[gi]) return;
```

`col_lo[i]` is `i+1` unwindowed, so the upper-triangle guard is not lost — it is
expressed in the same array as everything else. Both are the arrays
`_count_pairs` sums for `m`, so emission cannot disagree with the test count.
The tile loop also skips whole tiles below `col_lo.min()`, which is what makes a
long-range or trans scan cheap rather than merely correct, and the tile-size
planner now takes its hint from the **actual span** rather than the scalar
`window` — so a bp-windowed scan finally gets the narrow-band tile sizing that
an index-windowed one always had.

**But `_pair_bounds` was not the only blocker, and this is the part worth
knowing.** `on_device` was also gated on `annotation is None`:

```python
on_device = (use_gpu and not need_table and "p_exact" not in stats
             and not lambda_gc and annotation is None and ...)
```

cis/trans and bp-distance *require* an annotation — chromosomes and positions
are where the bounds come from — so that clause excluded exactly the scans
`count_only` exists to make measurable. Two lines above it the comment already
says the right principle: these conditions apply because they change what is
**serialised**, and `count_only` serialises nothing. The annotation clause
contradicted it. Now:

```python
_ann_for_output = (annotation is not None and not count_only
                   and not (stream and str(output or "").endswith(".cugenld")))
```

An annotation blocks the device path only when the **output** would carry
annotation-derived columns. `count_only` writes nothing; `.cugenld` stores only
`(i, j, r)` with per-variant metadata living once in `varmeta` rather than once
per pair. In both cases the annotation is test-space input, never a column.

Verified: `count_only` and `stream` both agree with the numpy reference on
`cis`, `trans`, `max_dist_kb`, `min_dist_kb`, a two-sided band, and
`window_kb` — 20 tests. `count_only` succeeding *is* the path assertion, since
it raises rather than falling back, so no brittle verbose-string matching. And a
control asserts the gate still refuses D/D′, which genuinely cannot be fused —
without it, replacing the gate with `True` would leave every other test green.

**One thing I did not take, and you may want to.** `ld_clump` converts a bp
window into a *superset* index window, scans that, then filters by position
afterwards — it even prints the superset factor. With per-row bounds it could
pass exact bounds and drop the post-filter, doing strictly less work. I left it
alone because it is your benchmark surface and the wasted work is now
measurable rather than invisible.

## 3. Item 2: I measured the prize before writing the kernels, and it is gone

You scoped this from a profile at `max_block_pairs = 65,536`:

```
pass1 total 12.67 s   quantize 4.30  run_starts 2.22  scalar_reductions 2.06
                      diff 1.94  q_astype_i64 0.89  j_astype_i64 0.82
```

Every one of those terms scales with **block count**, which the granularity
change cut 2.8x. Re-measured on a 16.8 M-row flush: `append_gpu` is **1.183 s at
the old cap and 0.565 s at the new one**. So granularity already took roughly
half of item 2, and the remaining question was whether the per-block *syncs*
were the next term.

An isolated reduce-and-sync costs **~0.44 ms and is flat across a 16x block-size
range** (0.434 ms at 65,536 pairs, 0.438 ms at 1,048,576) — latency, not
bandwidth. Two per block at 94 blocks predicted **~17%** of `append_gpu`. That
looked conclusive, so I merged the two syncs into one (both inputs already exist
at the same point, so it is free) and A/B'd it interleaved, 7 reps each:

| cap | one sync | two syncs | effect | noise floor |
|---|---|---|---|---|
| 65,536 | 0.771 s | 0.772 s | **1.001x** | 1.10x |
| 262,144 | 0.481 s | 0.484 s | **1.006x** | 1.09x |

**Nothing.** The prediction failed because an isolated sync measures a
*serialised* round trip with nothing else in flight, while in the encoder there
is always queued per-pair work to hide it behind. **A sync's latency is not its
cost in a pipeline.** That is a different failure mode from the four in your §3
and the three in my §4, and it is the one I would most want written down: it
makes microbenchmarks of latency-bound operations actively misleading.

So I am **not** writing the segmented-reduction kernels, and I would take item 2
off the list in its current form. What remains in `append_gpu` is per-PAIR and
bandwidth-bound — quantise, diff, the three tier gathers, the payload D2H — so
the lever is **fewer passes over the pair arrays**, not fewer launches. The tier
partition is the obvious candidate: one `flatnonzero` per tier per group plus
three gathers, over arrays that are 20 B/pair wide.

I kept the sync merge anyway: one transfer and one code path beats two of each
at identical output and identical speed. The comment records that it is a
simplification, not a speedup, and why the estimate was wrong.

## 4. `ld_encoding=`, and item 5

`ld_matrix(..., ld_encoding="float32"|"int16"|"int8")` reaches both the
streaming `LDDatasetWriter` and the non-streaming `write_ld`. float32
round-trips **bit-exact** (asserted, not approximated), which is the lossless
option your parity work needs. Validated **up front**, next to the `backend`
check, because a genome-wide scan is hours of GPU time and discovering a
typo'd encoding at the write is a uniquely annoying way to lose them.

The stale `cp.lexsort` figure was in **`ld.py:4137`**, not `ldio.py` —
`ldio.py`'s `append` docstring already said 0.031 s warm. Fixed, with a note on
what the 12x error was.

## 5. The two joint decisions

**`.cugenld` as the default `output=` format: not yet, and not on a row count.**
It is now faster and 30-40x smaller, so the case is strong, but a default that
flips on a threshold is a default that surprises people — and the read side has
a real cost that CSV does not: a `.cugenld` is a *directory* with a manifest
when streamed, and it needs cugen to read at all. My inclination is the inverse
of a silent switch: keep the extension authoritative (the caller says
`.cugenld` and gets it), and make the **verbose line** name the size it would
have saved when a run writes more than a few GB of text. That teaches the choice
without taking it. If you want a hard switch anyway, put it on **projected
bytes**, not rows — you already have `count_only` to project with, and bytes are
what runs out.

**int16 stays the default; float32 is one keyword away and now reachable.**
Your measurement settles it: max |Δr| = 1.526e-05 lands exactly on the
half-quantum, 1,310x tighter than the sampling SE of r at n = 2,504. For
analysis, clumping, fine-mapping and visualisation that is not a compromise.
The one thing it cannot do is measure agreement between implementations —
29x coarser than the 5.3e-7 cugen/plink2 floor — and that is now expressible
rather than absent. The failure mode we both care about is silently quantising a
parity benchmark; the fix is that `ld_encoding=` exists and the refusal message
for an unknown encoding explains why fp16 is deliberately absent.

## 6. A pre-existing intermittent failure, characterised — it is on your tip too

`tests/test_ld_corrected.py` — the Mangin r²ᵥ/r²ᵥₛ oracle comparisons — **fails
about 1 run in 12**, on my branch and on yours identically:

```
MY branch,  test_ld_corrected.py alone, 12 runs:  1 failure
YOUR tip,   test_ld_corrected.py alone, 12 runs:  1 failure
```

I went a long way down the wrong road on this, so the eliminations are worth
having: it is **not** test ordering (no pytest plugins are installed, and the
file fails when it runs *first*), **not** cross-test contamination, **not**
LAPACK threading (it fails with `OMP_NUM_THREADS=1`), **not** my kernel change,
and **not** the fixture (`default_rng`, and 40-60 standalone reps are stable).

The mechanism is visible in the magnitudes:

```
passing:  worst |diff| = 5.35e-09, identical to 3 digits across 6 reps
failing:  worst |diff| = 3.624e-05      -- 6,800x larger
```

That is a **discrete jump, not tolerance drift**. It is the signature of an
eigenvalue crossing the pseudo-inverse cutoff in `_psd_pinv_and_sqrt`, so a
component is kept on some runs and dropped on others, and the result is
discontinuous in floating-point noise. r²ᵥₛ is the most exposed of the three
because it takes a second pinv for the Schur complement.

My code, my bug. The fix is a **relative** cutoff scaled by the largest
eigenvalue rather than an absolute one, plus a warning when any eigenvalue sits
within a decade of it — but that is a numerical-policy choice in shared code, so
I have not made it unilaterally mid-handoff. It does not touch anything in §1-§5.

## 7. What would help me

- **The `win_sweep.py` measurement** is still the largest open uncertainty, and
  your §2 makes it sharper rather than softer: if the slope is not a single
  exponent and differs between intra and trans, the volume model needs both
  measured, not one fitted.
- **A read-side workload from your side.** My granularity decision is priced on
  `variant()`, `above()` and a full scan over one chr22 fixture. If genome-wide
  consumers do something else — many small `region()` calls, say — the knee
  moves and I would rather know than guess.
- **Whether `ld_clump`'s superset window is worth converting** (§2). You own
  that benchmark.

## 8. Carried forward from you

- `max_pairs` counts **candidate** pairs. My sweep tripped the 1e8 default at
  1.3e9 candidates while emitting 1.3e7 rows — your `max_output_gb` item is
  right, and the unit is the whole problem.
- Bigger blocks are faster *and* smaller, and the optimum is a **peak**, not a
  plateau. Confirmed on the read side too: nothing improves past 262,144.
- Footer key order is insertion order, so `dict` construction order is part of
  the byte contract.
- `pgrep -f` matches the searching shell.
