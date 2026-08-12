"""Line plots for the two scaling axes: variants (p) and samples (n).

Form notes, deliberate:
  * Lines, not bars -- both axes are continuous quantities and the SHAPE is
    the finding (where cugen overtakes plink2, and where it stops gaining).
  * Log-log for time, because both span 3+ decades; a linear axis would hide
    everything below the largest point.
  * A reference slope of 2 is drawn, since pair count grows as p^2 -- it turns
    "is this quadratic?" into something you read off rather than infer.
  * Speedup gets its own panel with a 1.0 rule: crossing it is the event
    readers care about, and it is invisible on a shared time axis.
  * Two series only per panel, so identity is carried by direct labels and no
    legend box is needed.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e3df"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=10, color=INK2)
    ax.set_ylabel(ylabel, fontsize=10, color=INK2)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)


def load(name):
    p = os.path.join(HERE, "results", name)
    return json.load(open(p)) if os.path.exists(p) else None


def panel_time(ax, x, pl, cg, xlabel, title, slope_ref=None):
    xs = np.asarray(x, float)
    have = np.array([v is not None for v in pl])
    if have.any():
        ax.plot(xs[have], np.array(pl, dtype=object)[have].astype(float),
                marker="o", ms=6, lw=2.2, color=ORANGE, label="plink2")
    ax.plot(xs, cg, marker="o", ms=6, lw=2.2, color=BLUE, label="cugen")
    ax.set_xscale("log"); ax.set_yscale("log")
    style(ax, xlabel, "wall time (s)", title)
    if slope_ref:
        x0, x1 = xs.min(), xs.max()
        y0 = min(cg) * 0.55
        ax.plot([x0, x1], [y0, y0 * (x1 / x0) ** 2], ls=":", lw=1.4,
                color=INK2, alpha=0.7)
        ax.annotate("slope 2 (pairs grow as $p^2$)", (x1, y0 * (x1 / x0) ** 2),
                    fontsize=8.5, color=INK2, ha="right", va="bottom")
    # direct labels beat a legend box for two series
    ax.annotate("plink2", (xs[have][-1] if have.any() else xs[-1],
                           float(np.array(pl, dtype=object)[have][-1])
                           if have.any() else cg[-1]),
                color=ORANGE, fontsize=10, fontweight="bold",
                xytext=(6, 2), textcoords="offset points")
    ax.annotate("cugen", (xs[-1], cg[-1]), color=BLUE, fontsize=10,
                fontweight="bold", xytext=(6, -10), textcoords="offset points")


def panel_speedup(ax, x, spd, xlabel, title):
    xs, ys = np.asarray(x, float), np.asarray(spd, float)
    ok = np.isfinite(ys)
    ax.plot(xs[ok], ys[ok], marker="o", ms=6, lw=2.2, color=BLUE)
    ax.axhline(1.0, color=ORANGE, lw=1.6, ls="--")
    ax.annotate("parity with plink2", (xs[ok][0], 1.0), fontsize=8.5,
                color=ORANGE, va="bottom", xytext=(2, 3),
                textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    style(ax, xlabel, "speedup (x, higher = cugen faster)", title)
    for xi, yi in zip(xs[ok], ys[ok]):
        ax.annotate(f"{yi:.0f}x" if yi >= 10 else f"{yi:.1f}x", (xi, yi),
                    fontsize=8.5, color=INK2, xytext=(0, 7),
                    textcoords="offset points", ha="center")


figs = 0
ps = load("psweep.json")
if ps:
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor="white")
    panel_time(ax[0], [r["p"] for r in ps], [r["plink_s"] for r in ps],
               [r["cugen_s"] for r in ps], "variants (p)",
               "Whole-chromosome scaling — 1000 Genomes chr22\n"
               "n = 2,504 samples, all-pairs LD", slope_ref=True)
    panel_speedup(ax[1], [r["p"] for r in ps], [r["speedup"] for r in ps],
                  "variants (p)", "cugen speedup over plink2 (128 CPU cores)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/scaling_variants.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT}/scaling_variants.png")
    figs += 1

ns = load("nscale.json")
if ns:
    fig2, ax2 = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor="white")
    panel_time(ax2[0], [r["n"] for r in ns], [r["plink_s"] for r in ns],
               [r["cugen_s"] for r in ns], "samples (n)",
               "Sample-count scaling — p = 4,000 variants fixed\n"
               "synthetic genotypes, all-pairs LD")
    panel_speedup(ax2[1], [r["n"] for r in ns], [r["speedup"] for r in ns],
                  "samples (n)", "cugen speedup over plink2 (128 CPU cores)")
    fig2.tight_layout()
    fig2.savefig(f"{OUT}/scaling_samples.png", dpi=150, bbox_inches="tight",
                 facecolor="white")
    print(f"wrote {OUT}/scaling_samples.png")
    figs += 1

if not figs:
    sys.exit("no results found in benchmarks/results/")
