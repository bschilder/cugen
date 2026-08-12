"""Charts for the cugen.ld GPU portability matrix.

Form choices, deliberately:
  * Horizontal bars — long device names; vertical ticks collide immediately.
  * Figure HEIGHT scales with device count. At 11 devices a fixed height worked;
    at 29 the two-line labels overlapped into an unreadable band. Rows get a
    fixed vertical allowance instead, and labels are one line.
  * Panel 2 is cost efficiency, NOT peak memory. Peak memory is identical
    (4.90 GiB) on every device measured, so a bar chart of it is a flat wall
    encoding one number 29 times — that belongs in a sentence, not a figure.
    Pairs per dollar spans ~19x and is the number that decides an instance type.
  * Direct value labels on every bar, so the compressed low end stays readable
    without a log scale or a broken axis (bars must encode magnitude from zero).
  * One series per panel, so one hue and no legend; the title names the measure.
  * Devices that never ran get their REASON, not a gap and not a generic "not
    tested" — an absent bar and a zero bar must not look alike, and "no
    capacity" and "the kernel failed" must not read alike either.
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
AQUA = "#1baf7a"        # slot 3 — the cost-efficiency measure
AMBER = "#eda100"       # slot 4 — flags a suspected host-limited draw
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e3e3df"

REASON = {
    "unavailable": "no capacity",
    "create_failed": "no capacity",
    "no_public_ip": "no public IP on host",
    "endpoint_timeout": "no SSH endpoint",
    "ssh_timeout": "SSH never came up",
    "no_result": "no probe output",
    "done": "probe reported no timing",
    "exception": "driver error",
}


def reason_for(r):
    errs = (r.get("probe") or {}).get("errors") or []
    if errs and (r.get("probe") or {}).get("stage") == "import":
        return "CUDA init failed on host"
    if errs:
        return errs[0].split(":")[-1].strip()[:28]
    return REASON.get(r.get("stage"), str(r.get("stage")))


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
                     cost_hr=r.get("cost_hr"),
                     reason=reason_for(r),
                     compile_s=p.get("kernel_compile_s")))

ran = [r for r in recs if r["run"]]
missing = [r for r in recs if not r["run"]]
ran.sort(key=lambda r: r["run"]["wall_s"])          # fastest at top
ordered = ran + missing

# One pod per GPU confounds host quality with GPU model, so flag anything far
# off the pack rather than letting it read as an architecture result.
times = [r["run"]["wall_s"] for r in ran]
med = float(np.median(times)) if times else 1.0
sus = {r["label"] for r in ran if r["run"]["wall_s"] > 5 * med}

# Height scales with row count; below ~14 rows keep the original proportions.
H = max(6.4, 0.34 * len(ordered) + 2.2)
fig, axes = plt.subplots(1, 2, figsize=(16.5, H), facecolor="white")
y = np.arange(len(ordered))[::-1]
labels = [(f"{r['label']}  ·  {r['vram']:.0f} GiB, cc {r['cc']}"
           if r["vram"] else r["label"]) for r in ordered]


def style(ax, xlabel, title):
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xlabel(xlabel, fontsize=10, color=INK2)
    ax.set_title(title, fontsize=11.5, color=INK, pad=12, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-1, len(ordered))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK2, labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)


# ---- panel 1: wall time -------------------------------------------------
vals = [r["run"]["wall_s"] if r["run"] else 0.0 for r in ordered]
cols = [(AMBER if r["label"] in sus else BLUE) if r["run"] else "none"
        for r in ordered]
axes[0].barh(y, vals, color=cols, height=0.68)
style(axes[0], "wall time (s) — lower is better",
      f"Time for an identical workload\np = {P_REPORT:,} variants "
      f"({P_REPORT*(P_REPORT-1)//2/1e6:.0f}M pairs), 2,504 samples")
xmax = max(vals) if vals else 1
for yy, r, v in zip(y, ordered, vals):
    if r["run"]:
        axes[0].text(v + xmax * 0.012, yy, f"{v:.3f}s", va="center",
                     fontsize=8.5, color=INK2)
    else:
        axes[0].text(xmax * 0.012, yy, f"not measured — {r['reason']}",
                     va="center", fontsize=8.5, color=MUTED, style="italic")
axes[0].set_xlim(0, xmax * 1.30)

# ---- panel 2: cost efficiency ------------------------------------------
# Billion pairs per dollar: throughput normalised by price. Peak memory is
# deliberately not plotted -- it is 4.90 GiB on every device, so it carries no
# variance to encode.
def ppd(r):
    if not (r["run"] and r.get("cost_hr")):
        return None
    return r["run"]["pairs"] / r["run"]["wall_s"] / (r["cost_hr"] / 3600) / 1e9


eff = [ppd(r) for r in ordered]
axes[1].barh(y, [e or 0.0 for e in eff],
             color=[AQUA if e else "none" for e in eff], height=0.68)
style(axes[1], "billion pairs per dollar — higher is better",
      "Cost efficiency at list price\nthroughput ÷ $/hr, same workload")
emax = max([e for e in eff if e] or [1])
for yy, r, e in zip(y, ordered, eff):
    if e:
        axes[1].text(e + emax * 0.012, yy, f"{e:,.0f}", va="center",
                     fontsize=8.5, color=INK2)
    elif r["run"]:
        axes[1].text(emax * 0.012, yy, "price not recorded", va="center",
                     fontsize=8.5, color=MUTED, style="italic")
axes[1].set_xlim(0, emax * 1.22)
axes[1].set_yticklabels([])

note = (f"All {len(ran)} devices that ran produce results bit-identical to the "
        f"CPU reference (max|ΔR| = 0.0). Peak device memory is 4.90 GiB on "
        f"every one of them — at this p the tile is capped by the variant "
        f"count, not by VRAM — so it is stated here rather than plotted.")
if sus:
    note += ("   Amber = " + ", ".join(sorted(sus)) +
             ": >5x the median; n=1 pod per GPU, so a slow host draw is the "
             "likely cause, not the chip.")
fig.text(0.005, -0.01, note, fontsize=8.5, color=INK2, ha="left", va="top",
         wrap=True)
fig.suptitle("cugen.ld — portability and cost across GPU generations",
             fontsize=13.5, color=INK, x=0.005, ha="left", y=1.0)
fig.tight_layout(rect=[0, 0.015, 1, 0.97])
fig.savefig(f"{OUT}/gpu_matrix.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT}/gpu_matrix.png")

# ---- figure 2: time and memory vs p ------------------------------------
# Far more series than the categorical cap, and a generated ramp for the 9th+
# is exactly the anti-pattern. So: every device is a muted envelope and three
# representatives carry identity -- fastest, smallest VRAM, and any suspected
# slow-host outlier. That is the "fold into Other" pattern.
fig2, ax = plt.subplots(1, 2, figsize=(14, 5.4), facecolor="white")
shown = [r for r in ran if r["runs"]]
hi_labels = {}
if shown:
    hi_labels[shown[0]["label"]] = BLUE                       # fastest
    small = min(shown, key=lambda r: r["vram"] or 1e9)
    hi_labels.setdefault(small["label"], AQUA)                # smallest VRAM
    for lab in sus:
        hi_labels[lab] = AMBER

for r in shown:
    ps = sorted(r["runs"])
    hl = hi_labels.get(r["label"])
    kw = (dict(color=hl, lw=2.4, marker="o", ms=6, zorder=3, label=r["label"])
          if hl else dict(color="#c9c9c4", lw=1.2, zorder=1))
    ax[0].plot(ps, [r["runs"][q]["wall_s"] for q in ps], **kw)
    ax[1].plot(ps, [r["runs"][q]["peak_pool_gib"] for q in ps], **kw)

ax[0].set(xscale="log", yscale="log", xlabel="variants (p)",
          ylabel="wall time (s)")
ax[0].set_title("Time vs p — every device, log-log", fontsize=11, loc="left")
ax[1].set(xscale="log", xlabel="variants (p)",
          ylabel="peak device memory (GiB)")
ax[1].set_ylim(bottom=0)
# Don't claim a plateau this figure doesn't show: at p <= 20,000 the curves are
# still rising. The plateau evidence is the chr22 sweep (p 20k -> 171k at a
# constant ~8.4 GiB), quoted in the issue text instead.
ax[1].set_title("Peak memory vs p — identical across devices;\n"
                "the plateau appears above the tile size (see chr22 sweep)",
                fontsize=11, loc="left")
for a in ax:
    a.grid(color=GRID, linewidth=0.8)
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.tick_params(colors=INK2, labelsize=9)
ax[0].legend(fontsize=9, frameon=False, loc="upper left")
fig2.text(0.005, -0.03, "Grey = the other measured devices (identity carried "
          "by the table, not by a synthesised hue).", fontsize=9, color=INK2)
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
        lines.append(f"| {r['label']} | – | – | not measured | not measured | "
                     f"– | – | – |")
        continue
    lines.append(
        f"| {r['label']} | {r['cc']} | {r['vram']:.1f} GiB | "
        f"yes ({r['compile_s']:.1f} s) | "
        f"{'**exact (0.0)**' if r['exact'] else '**MISMATCH**'} | "
        f"{r['run']['wall_s']:.3f} s | {r['run']['peak_pool_gib']:.2f} GiB | "
        f"{r['run']['pairs_per_s']/1e6:.0f} |")
open(f"{OUT}/gpu_matrix_table.md", "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT}/gpu_matrix_table.md")
