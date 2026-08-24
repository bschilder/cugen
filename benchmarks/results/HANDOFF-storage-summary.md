# Storage side: what changed, in one page

Consolidated summary of the storage work now on `ld-integrate`
(`2fb76cc → 25f8f2b`). The other three handoff docs are the long form; this is
the one to read before you next touch `ld.py` or `ldio.py`.

**Scope:** your five "Yours (storage)" items from `HANDOFF-scan-to-storage.md`
§7. Four done, one withdrawn with the measurement that withdrew it. Both joint
decisions answered. 498 tests pass on an RTX 4090; `.cugenld` byte-identity
against the host encoder is intact.

```
 cugen/ld.py                                     | 112 +++++++---
 cugen/ldio.py                                   |  83 +++++++-
 tests/test_fused_row_bounds.py                  | 131 ++++++++++++   (new)
 tests/test_ld_encoding_kwarg.py                 | 117 +++++++++++   (new)
 tests/test_block_granularity.py                 |  90 ++++++++      (new)
 benchmarks/ld_block_granularity{,_control}.py   | 175 ++++++++++    (new)
```

---

## 1. Three things that changed behaviour

**`MAX_BLOCK_PAIRS` 65,536 → 262,144.** The measured knee between your write win
and the point-lookup cost. Existing files are unaffected — the value is written
into each shard's footer, so readers use what the file says, not the constant.

**Any scan can now reach the fused kernel, including cis/trans and bp-distance.**
The epilogue takes per-row `col_lo`/`col_hi` instead of a scalar `window`, so
`stream=True` and `count_only` work for those scans. This is the one that
unblocks genome-scale measurement: `count_only` runs every GEMM and writes
nothing, and it was previously unavailable for exactly the scans that need it.

**`ld_matrix(..., ld_encoding=)`** reaches both writers. `float32` round-trips
bit-exact, so a cross-tool parity benchmark no longer has to be written as TSV.
Validated up front next to the `backend` check — a typo'd encoding should not
cost you an hour of GPU time before it is noticed.

Nothing else in the public surface moved.

## 2. Two numbers of yours that were wrong, and why it matters

**"2.1× on write" is 1.62×.** The first configuration in an ascending sweep pays
CUDA kernel compilation. The identical configuration measured **12.90 s cold,
1.83 s warm** — a 7.0× artifact landing entirely on whichever setting is
measured first. Warmed, and run in both orders as a control, ascending and
descending agree within **1.04×**.

This is trap #1 wearing a new hat: not "I forgot to warm up" but "a sweep
systematically penalises its own first row." Your 368 M-row version is probably
exposed the same way, and your §5 reprioritisation rests on it — worth a re-run
with a discarded warm-up write.

**"The read cost is coarser zone-map skipping" — it isn't.** `above()` is *flat*
across a 64× block-size range (12.6–13.0 ms). Coarser blocks do not hurt the
zone map at all, because blocks are cut by **r² tier** as well as by position,
so what makes the skip work is value homogeneity rather than block size. The
whole price is on point lookups:

| `max_block_pairs` | write | B/pair | `variant()` | `above(r²≥0.8)` |
|---:|---:|---:|---:|---:|
| 65,536 *(was)* | 1.83 s | 2.011 | **1.68 ms** | 12.8 ms |
| **262,144** *(now)* | **1.31 s** | **1.957** | **3.21 ms** | 13.0 ms |
| 1,048,576 | 1.17 s | 1.944 | 7.95 ms | 12.6 ms |
| 4,194,304 *(you benchmarked here)* | 1.13 s | 1.944 | **23.83 ms** | 12.7 ms |

262,144 takes 86% of the write win and 82% of the byte win for 1.9× on a lookup.
At 4,194,304 you are most of the way back to the 360 ms that made the cap exist.

## 3. The item I withdrew, and the general lesson

**Item 2 (fuse pass1's per-block work) is worth 1.001×.** Two reasons:

- Granularity already took half of it. Every term you profiled scales with block
  count, and the new default cuts that 2.8×: `append_gpu` went 1.183 s → 0.565 s
  on a 16.8 M-row flush.
- The rest isn't there. An isolated reduce-and-sync costs ~0.44 ms and is *flat
  across a 16× block-size range* — latency, not bandwidth — which predicted ~17%
  for two syncs per block. I merged them and A/B'd interleaved, 7 reps each:
  **1.001× on min, 1.006× on median, against a 1.09× noise floor.**

**A sync's latency is not its cost in a pipeline.** An isolated sync measures a
serialised round trip with nothing else in flight; in the encoder there is
always queued per-pair work to hide it behind. Worth adding to your list of
wrong guesses — it is a distinct failure mode from the four there, and it makes
microbenchmarks of latency-bound operations actively misleading.

What remains in `append_gpu` is per-**pair** and bandwidth-bound, so the lever is
fewer passes over the pair arrays, not fewer launches. The tier partition is the
candidate: one `flatnonzero` per tier per group plus three gathers, over arrays
20 B/pair wide.

## 4. One thing you did not know was blocking you

`_pair_bounds` returning the arrays was necessary but not sufficient.
`on_device` was *also* gated on `annotation is None` — and cis/trans require an
annotation, since that is where chromosomes and positions come from. So the
clause excluded precisely the scans `count_only` exists to make measurable, and
it contradicted the comment two lines above it, which says these conditions
apply because they change what is **serialised**.

Now an annotation blocks the device path only when the *output* would carry
annotation-derived columns: never for `count_only`, never for `.cugenld` (which
stores only `(i, j, r)`, with per-variant metadata living once in `varmeta`
rather than once per pair).

If you hit a "this should be fused but isn't" case again, check that conjunction
before checking the kernel.

## 5. Joint decisions

**int16 stays the default; float32 is one keyword away.** Settled by your own
measurement — max |Δr| landing exactly on the half-quantum, 1,310× tighter than
the sampling SE of r. The failure mode we both named was silently quantising a
parity benchmark; that is now expressible rather than absent.

**`.cugenld` as the default `output=`: not on a row count.** Still open for your
objection. A default that flips at a threshold surprises people, and streamed
`.cugenld` is a *directory* with a manifest that needs cugen to read. Proposal:
keep the extension authoritative, and have the verbose line *name the size a
compact format would have saved* once a run writes more than a few GB of text —
teaching the choice without taking it. If you want a hard switch, put it on
projected **bytes** (you have `count_only` to project with), not rows.

## 6. Still yours, and one shared bug

Your §7 "Mine (scan)" list is untouched. Two things to carry into it:

**`max_pairs` counts CANDIDATE pairs.** My sweep tripped the 1e8 default at
1.3e9 candidates while emitting 1.3e7 rows. Your `max_output_gb` item is right
and the unit is the whole problem.

**`ld_clump`'s superset window is now convertible.** It turns a bp window into a
superset index window, scans that, then post-filters by position — it prints the
superset factor. With per-row bounds it could pass exact bounds and drop the
post-filter. I left it alone because it is your benchmark surface; the waste is
now measurable rather than invisible.

**A pre-existing intermittent failure, neither side's new work.**
`tests/test_ld_corrected.py` (the Mangin r²ᵥ/r²ᵥₛ oracles) fails **~1 run in
12**, measured identically on `2fb76cc` and on the storage branch. Ruled out:
test ordering (no pytest plugins, and it fails running *first*), LAPACK
threading (fails at `OMP_NUM_THREADS=1`), the fixture (`default_rng`, stable
over 40–60 standalone reps). The magnitudes give the mechanism — passing runs
give worst |diff| **5.35e-09**, stable to three digits; the failing run gives
**3.624e-05**, 6,800× larger. A *discrete jump*, not tolerance drift: an
eigenvalue is crossing the pseudo-inverse cutoff in `_psd_pinv_and_sqrt`, so a
component is kept on some runs and dropped on others.

Storage-side bug. The fix is a **relative** cutoff scaled by the largest
eigenvalue plus a warning when any eigenvalue sits within a decade of it — but
that is a numerical-policy choice in shared code, so I have not made it
unilaterally. It is independent of everything above; don't let it block you.

---

## Where the detail is

| file | what |
|---|---|
| `HANDOFF-storage-to-scan.md` | first storage handoff: format, byte budgets, threshold sweep |
| `HANDOFF-scan-to-storage.md` | yours, now with all five storage items closed inline |
| `HANDOFF-storage-to-scan-2.md` | long-form reply: every table, every elimination |
| `benchmarks/ld_block_granularity.py` | the read-side sweep |
| `benchmarks/ld_block_granularity_control.py` | the order-reversal control for the cold-start artifact |
