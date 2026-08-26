# plink2 at genome scale: a storage wall, not a compute wall

Recovered from the benchmark host (AWS `c7a.32xlarge`, **128 physical cores**,
1,600 GiB gp3) before it was decommissioned. Raw logs and scripts are archived
off-box; this file is the result.

## Arm B — inter-chromosomal, like-for-like with cugen

`plink2 --r2-unphased inter-chr allow-ambiguous-allele`, 1000 Genomes 30x,
2,504 unrelated, MAF >= 0.01. Cumulative chromosome sets, so each row is a
strictly larger pair space than the one above it.

| chrs | variants | pairs | wall | Mpair/s | outcome |
|---|---|---|---|---|---|
| 1 | 943,790 | 445,369,310,155 | 1,279.6 s | 348.0 | ok, 164,993,213 rows |
| 1–2 | 1,951,964 | 1,905,080,752,666 | 4,439.8 s | 429.1 | ok, 588,994,585 rows (236,939,687 cross), 35 GiB |
| 1–4 | 3,684,029 | 6,786,032,994,406 | 15,896.6 s | 426.9 | ok, 1,839,316,580 rows (1,201,439,480 cross), 108 GiB |
| **1–22** | **12,057,350** | **72,689,838,482,575** | **28,659.3 s** | — | **exit code 5** |

## What the failure was, and what we can and cannot say

Be careful with this row. Two things in the run log are **our harness talking,
not plink2**:

- `stopping: infeasible from 1-22 up` comes from `gw_plink_interchr.sh:49`,
  which prints it on *any* non-zero exit. "Infeasible" is our word.
- `FAILED:` with nothing after it comes from line 39, which greps plink2's
  output for `^Error` or `no space`. **It matched nothing**, so we captured no
  plink2 error text at all — and plink2's own `/tmp/gwb.log` lived on ephemeral
  storage and was overwritten by the next iteration's `rm -f /tmp/gwb.*`.

So the only direct evidence is the exit code, **5**. In plink2's `PglErr` enum
(`plink2_base.h`) the values run `kPglRetSuccess`=0, `kPglRetSkipped`=1,
`kPglRetNomem`=2, `kPglRetOpenFail`=3, `kPglRetReadFail`=4,
**`kPglRetWriteFail`=5**. So plink2 failed on a **write**, not on the
computation.

**We do not know why the write failed.** The obvious hypothesis is disk
exhaustion, and it does not survive arithmetic. plink2 ran at **r² ≥ 0.2** (the
harness sets `R2=0.2`), and at that threshold the *entire* genome-wide output
projects to **1.25 TB against the 1,535 GB free** that the log recorded — it
fits, barely. At the point of failure it had written perhaps 210 GB. Disk
exhaustion is ruled out, not supported.

The one measurement that would have pinned the failure point — the partial
`.vcor` row count — was destroyed by the harness: the failure branch computes no
row count, and `rm -f /tmp/gwb.vcor*` then runs unconditionally.

## How long a genome-wide scan would take

The scaling is linear at scale, which is what makes extrapolation safe rather
than hopeful:

| step | log-log slope | marginal rate |
|---|---|---|
| 0.45e12 → 1.91e12 pairs | 0.856 | 461.9 Mpair/s |
| **1.91e12 → 6.79e12 pairs** | **1.004** | **426.0 Mpair/s** |

The sub-linearity on the first step is fixed-cost amortisation. By the largest
step the exponent is 1.004. A linear fit on the two largest rungs gives
`wall = 2.347e-9 × pairs − 32 s`; the intercept is negligible.

| | pairs | projected wall |
|---|---|---|
| plink2's own build, p = 12,057,350 | 7.269e13 | **47.4 h** |
| cugen's build, p = 12,528,011 | 7.848e13 | **51.2 h** |
| cugen, measured on one GPU | 7.848e13 | **24.7 min** |

Sensitivity across exponents 1.00–1.05 is 47.3–53.3 h, so ~47–48 h on its own
variant set. The failed run is *consistent* with this — 28,659 s is 16.8% of
47.4 h — but adds no precision, because we cannot verify how far it got.

## Output volume is the real wall, and the thresholds did not match

plink2 was run at r² ≥ 0.2; the cugen genome-wide scan it is being compared
against used r² ≥ 0.1. Compute is unaffected, since the scan visits every pair
either way. Storage is not:

| threshold | rows | plink2 text `.vcor` @ 63.4 B/row | cugen `.cugenld` @ 2.11 B/row |
|---|---|---|---|
| r² ≥ 0.2 (what ran) | 19.7 B | 1.25 TB | — |
| **r² ≥ 0.1 (like-for-like)** | **236 B** | **15.0 TB** | **0.54 TB** |

A like-for-like run therefore needs **15 TB**, which is the hard stop on a
1.6 TB volume. The required write rate is only ~88 MB/s sustained, well within
gp3, so the constraint is volume rather than bandwidth — and it is **28x**
cugen's footprint for the identical result.

## A flag trap that invalidates the naive comparison

Earlier runs on this host passed `--ld-window 999999999 --ld-window-kb 999999`
to defeat plink2's default windowing. plink2 **accepts those flags and silently
restricts the scan anyway**: the same chr1–2 job reported 2,364.2 s and
352,054,898 rows with them, against 4,439.8 s and 588,994,585 rows without.

That is a 1.88x speed difference and 40% of the rows missing, from flags that
produce no warning. The corrected invocation omits them entirely — plink2
rejects them outright when combined with all-pairs mode ("All-pairs
`--r2-unphased` settings cannot be used with
`--ld-window`/`--ld-window-kb`/`--ld-window-cm`"), which is the behaviour you
want and the reason the corrected numbers are the ones above.

Note also that `--ld-window-r2` must come **after** the modifiers, or
`allow-ambiguous-allele` is consumed as its argument.

## Arm A — per-chromosome windowed, for reference

Same host, default windowing, per chromosome. This is the regime plink2 is
built for and it is fast: chr1 in 5.8 s unphased / 6.4 s phased, chr2 in
6.5 / 7.8 s, down to chr22. Full series in the archived `gw_plink.json`.
Nothing here is a general claim about plink2 — it is specific to the all-pairs
inter-chromosomal regime.

## Reproducing

The harness should be fixed before anyone reruns it: capture plink2's `--out`
log to durable storage instead of `/tmp`, do not delete it between rungs, and
grep for the actual plink2 error vocabulary rather than `^Error|no space`. As
written, a failure yields an exit code and nothing else.
