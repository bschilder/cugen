"""Scatter plots: cugen vs plink2 for r, r2, D and D'.

Form notes:
  * Identity line drawn on every panel -- for an agreement plot the reference
    IS the message, and without it a reader cannot tell 0.999 from 1.000.
  * Square aspect and equal limits, so a deviation of a given size looks the
    same on both axes. Unequal scaling silently exaggerates or hides bias.
  * Disagreeing points are drawn LAST, in a status colour, over a neutral
    mass. With ~850k points at ~0.005% disagreement, plotting in input order
    buries exactly the thing worth seeing.
  * Hexbin would hide the tail here; the tail is the finding, so keep points --
    but keep alpha low (0.02) so overlap reads as DENSITY rather than a solid
    line. At ~850k points anything above ~0.05 saturates.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The parquet is ~40 MB and is not committed; regenerate it with
#   python benchmarks/concordance.py --bfile <bed> --cugen <cugen> \
#       --p 3000 --window 300 --out concordance.parquet
SRC = sys.argv[1] if len(sys.argv) > 1 else "concordance.parquet"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
TOL = 1e-4

BLUE, RED = "#2a78d6", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e3df"

df = pd.read_parquet(SRC)
panels = [("r", "r_cugen", "r_plink", "plink2 --r-unphased"),
          ("r²", "r2_cugen", "r2_plink", "plink2 UNPHASED_R²"),
          ("D", "d_cugen", "d_plink", "plink2 --r-phased D"),
          ("D'", "dp_cugen", "dp_plink", "plink2 DPRIME")]

# Two rows. The identity scatter alone is degenerate here: agreement is exact
# enough that every point lands on a one-pixel line, so density PERPENDICULAR
# to it is zero and no alpha reveals structure. The residual row rotates that
# line onto the x-axis and gives the y-axis back to the disagreement, which is
# the quantity actually worth seeing.
fig, allax = plt.subplots(2, 4, figsize=(19, 9.6), facecolor="white")
axes = allax[0]
for ax, (name, c1, c2, xlabel) in zip(axes, panels):
    x, y = df[c2].to_numpy(), df[c1].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    bad = np.abs(x - y) > TOL

    lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
    pad = 0.04 * (hi - lo or 1)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.2, ls="--", zorder=1)
    # alpha must be LOW enough that overlap reads as density: at ~850k points
    # anything above ~0.05 saturates to a solid line and hides where the mass
    # actually concentrates. Small marks + low alpha is the pairing that works.
    ax.scatter(x[~bad], y[~bad], s=2, c=BLUE, alpha=0.02, linewidths=0,
               zorder=2, rasterized=True)
    if bad.any():
        ax.scatter(x[bad], y[bad], s=16, c=RED, alpha=0.9, linewidths=0,
                   zorder=3, label=f"{bad.sum():,} disagree (>{TOL:g})")
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    err = np.abs(x - y)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK2)
    ax.set_ylabel(f"cugen {name}", fontsize=9.5, color=INK2)
    ax.set_title(f"{name}   max|Δ| = {err.max():.2e}", fontsize=11,
                 color=INK, loc="left", pad=8)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8.5)

# --- residual row: delta vs value, where density is visible ---------------
for ax, (name, c1, c2, xlabel) in zip(allax[1], panels):
    x, y = df[c2].to_numpy(), df[c1].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = y - x
    bad = np.abs(d) > TOL
    ax.axhline(0.0, color=INK2, lw=1.2, ls="--", zorder=1)
    ax.scatter(x[~bad], d[~bad], s=2, c=BLUE, alpha=0.02, linewidths=0,
               zorder=2, rasterized=True)
    if bad.any():
        ax.scatter(x[bad], d[bad], s=18, c=RED, alpha=0.9, linewidths=0,
                   zorder=3)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK2)
    ax.set_ylabel(f"cugen {name} − plink2", fontsize=9.5, color=INK2)
    rms = float(np.sqrt((d ** 2).mean()))
    ax.set_title(f"{name} residual   rms = {rms:.1e}", fontsize=11, color=INK,
                 loc="left", pad=8)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8.5)

n = int(np.isfinite(df["r_cugen"]).sum())
fig.suptitle(
    f"cugen vs plink2 on 1000 Genomes chr22 — {n:,} matched variant pairs. "
    f"Top: value vs value, dashed line is identity. Bottom: residual vs "
    f"value — the same data with the identity line rotated flat, so the "
    f"spread is legible. Red marks |Δ| > {TOL:g}.",
    fontsize=12.5, color=INK, x=0.005, ha="left", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/concordance.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print(f"wrote {OUT}/concordance.png")

# a focused second figure: what distinguishes the D' outliers?
bad = (df["dp_cugen"] - df["dp_plink"]).abs() > TOL
if bad.any():
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="white")
    ax2[0].scatter(df.loc[~bad, "dp_plink"], df.loc[~bad, "dp_cugen"], s=2,
                   c=BLUE, alpha=0.02, linewidths=0, rasterized=True)
    ax2[0].scatter(df.loc[bad, "dp_plink"], df.loc[bad, "dp_cugen"], s=26,
                   c=RED, alpha=0.95, linewidths=0)
    ax2[0].plot([-1, 1], [-1, 1], color=INK2, lw=1.2, ls="--")
    ax2[0].set(xlabel="plink2 DPRIME", ylabel="cugen D'", xlim=(-1.05, 1.05),
               ylim=(-1.05, 1.05))
    ax2[0].set_aspect("equal")
    ax2[0].set_title(f"D' — {int(bad.sum()):,} of {len(df):,} disagree "
                     f"({100*bad.mean():.3f}%)", fontsize=11, loc="left")

    mm = np.minimum(df["maf_a"], df["maf_b"])
    ax2[1].scatter(mm[~bad], (df["dp_cugen"] - df["dp_plink"]).abs()[~bad],
                   s=2, c=BLUE, alpha=0.03, linewidths=0, rasterized=True)
    ax2[1].scatter(mm[bad], (df["dp_cugen"] - df["dp_plink"]).abs()[bad],
                   s=26, c=RED, alpha=0.95, linewidths=0)
    ax2[1].set(xlabel="min MAF of the pair", ylabel="|Δ D'|", yscale="log")
    ax2[1].set_title("Disagreement vs allele frequency", fontsize=11,
                     loc="left")
    for a in ax2:
        a.grid(color=GRID, linewidth=0.7); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(colors=INK2, labelsize=8.5)
    fig2.tight_layout()
    fig2.savefig(f"{OUT}/concordance_dprime.png", dpi=150,
                 bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}/concordance_dprime.png")
