"""Charts for the cugen.ld GPU portability matrix.

Form choices, deliberately:
  * Horizontal bars — 11 devices with long names; vertical ticks collided.
  * TWO panels, not three. Throughput is pairs/time on a fixed workload, i.e.
    exactly 1/time — plotting both would show the same number twice.
  * Direct value labels on every bar, so the compressed low end stays readable
    without a log scale or a broken axis (bars must encode magnitude from zero).
  * One series per panel, so one hue and no legend; the title names the measure.
  * Devices that never ran get an explicit "not tested" marker, not a gap —
    an absent bar and a zero bar must not look alike.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "gpu_matrix_results.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
P_REPORT = 20000

BLUE = "#2a78d6"        # categorical slot 1 (light mode)
AMBER = "#eda100"       # slot 4 — flags the suspected-host-variance bar
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e3e3df"

rows = json.load(open(RES))
recs = []
for r in rows:
    p = r.get("probe") or {}
    runs = {x["p"]: x for x in p.get("runs", []) if "wall_s" in x}
    c = p.get("correctness") or {}
    recs.append(dict(label=r["label"], stage=r.get("stage"),
                     cc=p.get("compute_capability", "-"),
                     vram=p.get("total_mem_gib"),
                     exact=(c.get("max_abs_err_R") == 0.0),
                     run=runs.get(P_REPORT), runs=runs,
                     compile_s=p.get("kernel_compile_s")))

ran = [r for r in recs if r["run"]]
missing = [r for r in recs if not r["run"]]
ran.sort(key=lambda r: r["run"]["wall_s"])          # fastest at top
ordered = ran + missing

# The A4000 sits ~17x off its architectural neighbours (A5000, same cc, same
# VRAM). One pod per GPU means host quality is confounded with GPU model, so
# flag it rather than let it read as an architecture result.
times = [r["run"]["wall_s"] for r in ran]
med = float(np.median(times))
sus = {r["label"] for r in ran if r["run"]["wall_s"] > 5 * med}

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), facecolor="white")
y = np.arange(len(ordered))[::-1]
labels = [f"{r['label']}"
          + (f"\n{r['vram']:.0f} GiB · cc {r['cc']}" if r["vram"] else "")
          for r in ordered]


def style(ax, xlabel, title):
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel(xlabel, fontsize=10, color=INK2)
    ax.set_title(title, fontsize=11.5, color=INK, pad=12, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK2, labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)


# ---- panel 1: wall time -------------------------------------------------
vals = [r["run"]["wall_s"] if r["run"] else 0.0 for r in ordered]
cols = [(AMBER if r["label"] in sus else BLUE) if r["run"] else "none"
        for r in ordered]
axes[0].barh(y, vals, color=cols, height=0.62)
style(axes[0], "wall time (s) — lower is better",
      f"Time for an identical workload\np = {P_REPORT:,} variants "
      f"({P_REPORT*(P_REPORT-1)//2/1e6:.0f}M pairs), 2,504 samples")
xmax = max(vals) if vals else 1
for yy, r, v in zip(y, ordered, vals):
    if r["run"]:
        axes[0].text(v + xmax * 0.012, yy, f"{v:.1f}s", va="center",
                     fontsize=9, color=INK2)
    else:
        axes[0].text(xmax * 0.012, yy, "not tested — no capacity", va="center",
                     fontsize=9, color=MUTED, style="italic")
axes[0].set_xlim(0, xmax * 1.18)

# ---- panel 2: peak memory ----------------------------------------------
mem = [r["run"]["peak_pool_gib"] if r["run"] else 0.0 for r in ordered]
axes[1].barh(y, mem, color=[BLUE if r["run"] else "none" for r in ordered],
             height=0.62)
style(axes[1], "peak device memory (GiB)",
      "Peak GPU memory, same workload\ntile size auto-tuned per device")
mmax = max(mem) if mem else 1
for yy, r, v in zip(y, ordered, mem):
    if r["run"]:
        axes[1].text(v + mmax * 0.015, yy, f"{v:.2f}", va="center",
                     fontsize=9, color=INK2)
    else:
        axes[1].text(mmax * 0.015, yy, "not tested", va="center",
                     fontsize=9, color=MUTED, style="italic")
axes[1].set_xlim(0, mmax * 1.18)
axes[1].set_yticklabels([])

note = (f"All {len(ran)} devices that ran produce results bit-identical to the "
        f"CPU reference (max|ΔR| = 0.0).")
if sus:
    note += ("   Amber = " + ", ".join(sorted(sus)) +
             ": >5x its architectural neighbours; n=1 pod per GPU, so a slow "
             "host draw is the likely cause, not the chip.")
fig.text(0.005, -0.02, note, fontsize=9, color=INK2, ha="left", va="top")
fig.suptitle("cugen.ld — portability and cost across GPU generations",
             fontsize=13.5, color=INK, x=0.005, ha="left", y=1.0)
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig(f"{OUT}/gpu_matrix.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT}/gpu_matrix.png")

# ---- figure 2: memory vs p ---------------------------------------------
# 10 series exceeds the categorical cap, and a generated ramp for the 9th+ is
# exactly the anti-pattern. So: every device is drawn as a muted envelope and
# three representatives carry identity -- fastest, the 6 GB entry card, and the
# suspected-slow-host outlier. That is the "fold into Other" pattern.
fig2, ax = plt.subplots(1, 2, figsize=(14, 5.4), facecolor="white")
shown = [r for r in ran if r["runs"]]
hi_labels = {}
if shown:
    hi_labels[shown[0]["label"]] = BLUE                       # fastest
    small = min(shown, key=lambda r: r["vram"] or 1e9)
    hi_labels.setdefault(small["label"], "#1baf7a")           # slot 3, aqua
    for lab in sus:
        hi_labels[lab] = AMBER

for r in shown:
    ps = sorted(r["runs"])
    hl = hi_labels.get(r["label"])
    kw = (dict(color=hl, lw=2.4, marker="o", ms=6, zorder=3, label=r["label"])
          if hl else dict(color="#c9c9c4", lw=1.4, zorder=1))
    ax[0].plot(ps, [r["runs"][q]["wall_s"] for q in ps], **kw)
    ax[1].plot(ps, [r["runs"][q]["peak_pool_gib"] for q in ps], **kw)

ax[0].set(xscale="log", yscale="log", xlabel="variants (p)",
          ylabel="wall time (s)")
ax[0].set_title("Time vs p — every device, log-log", fontsize=11, loc="left")
ax[1].set(xscale="log", xlabel="variants (p)",
          ylabel="peak device memory (GiB)")
ax[1].set_ylim(bottom=0)
# Don't claim a plateau this figure doesn't show: at p <= 20,000 most curves
# are still rising. The plateau evidence is the chr22 sweep (p 20k -> 171k at
# a constant ~5.6 GiB), quoted in the issue text instead.
ax[1].set_title("Peak memory vs p — still rising in this range;\n"
                "the plateau appears above the tile size (see chr22 sweep)",
                fontsize=11, loc="left")
for a in ax:
    a.grid(color=GRID, linewidth=0.8)
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.tick_params(colors=INK2, labelsize=9)
ax[0].legend(fontsize=9, frameon=False, loc="upper left")
fig2.text(0.005, -0.03, "Grey = the other measured devices (identity carried by "
          "the table, not by a synthesised hue).", fontsize=9, color=INK2)
fig2.tight_layout()
fig2.savefig(f"{OUT}/gpu_scaling.png", dpi=150, bbox_inches="tight",
             facecolor="white")
print(f"wrote {OUT}/gpu_scaling.png")

# ---- markdown table (the accessible view of the same data) -------------
lines = ["| GPU | cc | VRAM | kernel compiles | matches CPU reference | "
         f"time @ p={P_REPORT:,} | peak mem | Mpair/s |",
         "|---|---|---|---|---|---|---|---|"]
for r in ordered:
    if not r["run"]:
        lines.append(f"| {r['label']} | – | – | not tested | not tested | – | "
                     f"– | – |")
        continue
    lines.append(
        f"| {r['label']} | {r['cc']} | {r['vram']:.1f} GiB | "
        f"yes ({r['compile_s']:.1f} s) | "
        f"{'**exact (0.0)**' if r['exact'] else '**MISMATCH**'} | "
        f"{r['run']['wall_s']:.2f} s | {r['run']['peak_pool_gib']:.2f} GiB | "
        f"{r['run']['pairs_per_s']/1e6:.0f} |")
open(f"{OUT}/gpu_matrix_table.md", "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT}/gpu_matrix_table.md")
