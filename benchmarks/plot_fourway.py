"""Figures for the four-way LD benchmark: cugen vs plink2 vs qLD-GPU vs qLD-BLIS.

Every number here was MEASURED. Nothing is modelled, smoothed or interpolated.

NAMING. Labels are not written at the call sites -- they live in SERIES below and
are pulled by draw(), so a series cannot be plotted without its canonical name,
colour and marker. The scheme is

    TOOL -- STATISTIC -- HARDWARE, THREADS

with colour FAMILY = tool (cugen greens / plink2 oranges / qLD purples), colour
SHADE = run configuration, MARKER = physical host, LINESTYLE = statistic
(solid unphased, dashed phased).

Naming the host is not pedantry: there are FOUR distinct plink2 configurations
here, and two of them use 32 threads on different machines. "plink2 @32c" would
silently merge a 32-core Runpod box with 32 threads of a 128-core c7a.32xlarge.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.titlesize": 9.5,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

SERIES = {
 # key            label                                                              tick                    colour     fmt
 "cugen":     ("cugen — unphased r² — 1×A100",                                    "cugen\nunphased",     "#0b6b3a", "o-"),
 "cugen_ph":  ("cugen — phased r² — 1×A100",                                      "cugen\nphased",       "#3fae6d", "o--"),
 "pl128":     ("plink2 — unphased r² — c7a.32xlarge (128c), 128 thr",             "plink2\n128c 128thr", "#a83208", "s-"),
 "pl32t":     ("plink2 — unphased r² — c7a.32xlarge (128c), 32 thr",              "plink2\n128c 32thr",  "#e06a1b", "s--"),
 "pl32box":   ("plink2 — unphased r² — 32-core Runpod host, 32 thr",              "plink2\n32c host",    "#c85212", "p-"),
 "pl13":      ("plink2 — unphased r² — 13.6-core cgroup cap, 26 thr",             "plink2\n13.6c cap",   "#f4a86a", "^-"),
 "pl13_ph":   ("plink2 — phased r² (--r2-phased) — 13.6-core cgroup cap",         "plink2 phased\n13.6c","#f4a86a", "^--"),
 "qldgpu":    ("qLD-GPU — phased r² — 1×A100 (OpenCL)",                           "qLD-GPU\n1×A100",     "#5b2d8e", "d-"),
 "qldblis8":  ("qLD-BLIS — phased r² — c7a.32xlarge, 8 thr",                      "qLD-BLIS\n8thr",      "#a07cc9", "v-"),
 "qldblis128":("qLD-BLIS — phased r² — c7a.32xlarge, 128 thr",                    "qLD-BLIS\n128thr",    "#a07cc9", "v--"),
}
lab  = lambda k: SERIES[k][0]
tick = lambda k: SERIES[k][1]
col  = lambda k: SERIES[k][2]

def draw(ax, key, x, y, **kw):
    """Plot a registry series, dropping None y-values (never falsy 0.0)."""
    xs = [a for a, b in zip(x, y) if b is not None]
    ys = [b for b in y if b is not None]
    l, _, c, fmt = SERIES[key]
    return ax.plot(xs, ys, fmt, color=c, label=l, lw=kw.pop("lw", 1.9),
                   ms=kw.pop("ms", 5), **kw)

def box(ax, text, xy=(0.02, 0.97)):
    ax.annotate(text, xy=xy, xycoords="axes fraction", va="top", ha="left",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                      ec="0.75", alpha=0.95))

# ============================================================ fig1 variant axis
P        = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 170949]
D_cugen  = [0.019, 0.066, 0.101, 0.066, 0.149, 0.355, 0.940, 1.734]
D_pl128  = [0.026, 0.035, 0.082, 0.219, 0.767, 5.025, 23.084, 41.465]
D_pl32t  = [0.023, 0.036, 0.107, 0.342, 1.260, 10.355, 31.177, 75.825]
D_pl13   = [0.070, 0.111, 0.318, 1.048, 5.951, 59.095, 230.022, 550.081]
D_qldgpu = [0.416, 0.471, 0.945, 2.329, 9.849, 34.995, 27.895, 73.644]

fig, ax = plt.subplots(figsize=(8.8, 5.4))
for k, d in (("cugen", D_cugen), ("pl128", D_pl128), ("pl32t", D_pl32t),
             ("pl13", D_pl13), ("qldgpu", D_qldgpu)):
    draw(ax, k, P, d)
draw(ax, "pl32box", [170949], [89.41], ms=8)
draw(ax, "qldblis8", [170949], [85.77], ms=8)
ax.set(xscale="log", yscale="log", xlabel="variants (p), all-pairs",
       ylabel="wall time (s)",
       title="Variant axis — 1000G chr22 MAF≥0.01, n=2,504, r²≥0.2")
box(ax, "full chr22 = 1.46e10 pairs\n"
        "cugen 1.734 s  vs  plink2 128c 41.47 s  = 23.9×\n"
        "all three emit 10,517,635 rows\n"
        "(plink2 128c measured twice: 41.47 / 42.21 s)")
ax.legend(fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, -0.13),
          ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_variant_axis.png",
                                bbox_inches="tight", pad_inches=0.3); plt.close(fig)

# ============================================================ fig2 sample axis
N       = [2504, 10000, 50000, 100000, 250000, 500000, 1000000]
S_cg    = [0.0647, 0.1085, 0.1891, 0.3369, 0.4201, 0.6808, 1.1037]   # 2504 re-measured warm
S_cgph  = [0.0699, 0.0745, 0.1838, 0.2567, 0.4821, 0.8268, 1.3238]
S_pl    = [0.092, 0.124, 0.391, 0.766, 2.194, 4.310, 8.510]
S_qgpu  = [1.463, 1.556, 2.312, 4.267, None, None, None]
S_qblis = [0.715, 1.472, 5.638, 10.978, None, None, None]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
for k, d in (("cugen", S_cg), ("cugen_ph", S_cgph), ("pl128", S_pl),
             ("qldgpu", S_qgpu), ("qldblis8", S_qblis)):
    draw(a1, k, N, d)
a1.axvline(500000, color="0.6", ls=":", lw=1)
a1.text(540000, 0.09, "biobank\nscale", fontsize=7.5, color="0.35")
a1.set(xscale="log", yscale="log", xlabel="samples (n), p=4,000 fixed",
       ylabel="wall time (s)", title="Sample axis — cugen's weakest axis")
a1.legend(fontsize=6.6, loc="upper left")
a1.annotate("qLD measured to n=100,000:\nVCF->MDF preprocessing is 487 s at that n",
            xy=(0.97, 0.03), xycoords="axes fraction", ha="right", fontsize=6.8,
            color="0.35")
sp = [p / c for p, c in zip(S_pl, S_cg)]
assert min(sp) > 1.0, f"speedup below 1 -- compile outlier back? {min(sp)}"
a2.plot(N, sp, "o-", color=col("cugen"), lw=2, ms=6)
for x, y in zip(N, sp):
    a2.annotate(f"{y:.1f}×", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7.5)
a2.set(xscale="log", xlabel="samples (n)", ylim=(0, 9),
       ylabel="speedup  (plink2 128c ÷ cugen)",
       title="Speedup grows only slowly with n")
box(a2, "n=2,504 re-measured after a warm-up:\n0.5633 s cold was CUDA JIT, 0.0647 s warm",
    xy=(0.03, 0.95))
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_sample_axis.png",
                                bbox_inches="tight", pad_inches=0.3); plt.close(fig)

# ============================================================ fig3 phased
PP    = [1000, 5000, 20000, 50000, 170949]
C_ph  = [0.0175, 0.0780, 0.1677, 0.4044, 1.8038]
P_ph  = [0.0867, 0.4525, 12.4072, 121.9037, 1013.8759]
fig, (b1, b2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
draw(b1, "cugen_ph", PP, C_ph, ms=6)
draw(b1, "pl13_ph", PP, P_ph, ms=6)
b1.set(xscale="log", yscale="log", xlabel="variants (p)", ylabel="wall time (s)",
       title="Phased LD — 19,888/19,888 pairs agree")
b1.legend(fontsize=7.2, loc="upper left")
b2.plot(PP, [p / c for p, c in zip(P_ph, C_ph)], "o-", color=col("cugen_ph"),
        lw=2, ms=6, label="cugen ÷ plink2 — phased r² — 13.6-core cap")
b2.plot(P, [p / c for p, c in zip(D_pl13, D_cugen)], "^-", color=col("pl13"),
        lw=1.7, ms=5, label="cugen ÷ plink2 — unphased r² — 13.6-core cap")
b2.plot(P, [p / c for p, c in zip(D_pl128, D_cugen)], "s-", color=col("pl128"),
        lw=1.7, ms=5, label="cugen ÷ plink2 — unphased r² — 128c, 128 thr")
b2.set(xscale="log", yscale="log", xlabel="variants (p)", ylabel="cugen speedup",
       title="plink2 pays per pair for phase")
b2.legend(fontsize=6.9, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_phased.png",
                                bbox_inches="tight", pad_inches=0.3); plt.close(fig)

# ============================================================ fig4 threads
fig, (c1, c2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
T  = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192]
TS = [10.79, 8.602, 5.906, 3.53, 2.084, 1.24, 0.885, 0.787, 0.752, 0.725]
c1.plot(T, [TS[0] / t for t in TS], "s-", color=col("pl128"), lw=2, ms=5,
        label=lab("pl128"))
c1.plot(T, T, ":", color="0.65", lw=1.2, label="linear")
c1.set(xscale="log", yscale="log", xlabel="--threads (128 physical cores)",
       ylabel="speedup vs 1 thread", title="plink2 thread scaling, 128 real cores")
box(c1, "14.35× at 128 cores.\nAn Amdahl fit to the 1–32-core\ncurve predicted 7.4× — wrong 2×.")
c1.legend(fontsize=6.9, loc="lower right")
QT, QS = [1, 8, 32, 64, 128], [371.015, 85.773, 125.813, 369.807, 370.687]
c2.plot(QT, QS, "v-", color=col("qldblis8"), lw=2, ms=7, label=lab("qldblis8"))
c2.axhline(371.015, color="0.6", ls=":", lw=1)
for x, y in zip(QT, QS):
    c2.annotate(f"{QS[0]/y:.2f}×", (x, y), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=7.5)
c2.set(xscale="log", xlabel="--threads (45 tasks available)", ylabel="wall time (s)",
       ylim=(0, 470), title="qLD-BLIS collapses at 64+ threads")
c2.annotate("single-thread baseline = 371.0 s;  chunk = tasks // threads = 0",
            xy=(0.5, 0.94), xycoords="axes fraction", ha="center", fontsize=7.6)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_threads.png",
                                bbox_inches="tight", pad_inches=0.3); plt.close(fig)

# ============================================================ fig5 cost
fig, ax = plt.subplots(figsize=(9.0, 4.6))
bars = [("cugen_ph", 7.97e-4), ("cugen", 7.66e-4), ("qldgpu", 3.25e-2),
        ("pl32box", 4.37e-2), ("pl128", 7.70e-2), ("qldblis8", 1.57e-1),
        ("pl13_ph", 4.48e-1), ("qldblis128", 6.76e-1)]
bb = ax.bar([tick(k) for k, _ in bars], [v for _, v in bars],
            color=[col(k) for k, _ in bars])
for k, b in zip([k for k, _ in bars], bb):
    if k in ("qldblis128",):
        b.set_hatch("//")
ax.set(yscale="log", ylabel="$ per job (log)",
       title="Cost per full-chr22 all-pairs job (1.46e10 pairs)")
for b, (_, v) in zip(bb, bars):
    ax.annotate(f"${v:.1e}", (b.get_x() + b.get_width()/2, v),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.2)
ax.legend(handles=[mpatches.Patch(color=col("cugen"), label="cugen"),
                   mpatches.Patch(color=col("pl128"), label="plink2"),
                   mpatches.Patch(color=col("qldgpu"), label="qLD")],
          fontsize=8, loc="upper left", frameon=False)
plt.setp(ax.get_xticklabels(), fontsize=7.2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_cost.png",
                                bbox_inches="tight", pad_inches=0.3); plt.close(fig)

# ============================================================ fig6 correctness
fig, ax = plt.subplots(figsize=(9.0, 4.4))
rows = [("cugen vs plink2\nunphased",            0.0,  5.70e-7, "cugen"),
        ("cugen vs plink2\n--r2-phased, PGEN",   0.0,  5.71e-7, "cugen"),
        ("cugen vs qLD\n-blis",                  0.0,  1.37e-6, "cugen"),
        ("cugen vs qLD\n-gpu",                  60.5,  5.79e-2, "qldgpu"),
        ("qLD -gpu vs plink2\n(bed EM baseline)",97.3,  6.09e-1, "qldgpu")]
FLOOR = 0.6
b = ax.bar([r[0] for r in rows], [r[1] + FLOOR for r in rows],
           color=[col(r[3]) for r in rows])
b[-1].set_hatch("//")
for bb_, r in zip(b, rows):
    ax.annotate(f"{r[1]:.1f}%\nmax Δ={r[2]:.2e}",
                (bb_.get_x() + bb_.get_width()/2, r[1] + FLOOR),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.2)
ax.set(ylabel="% of shared pairs differing >1e-4", ylim=(0, 118),
       title="Correctness — three implementations agree\nqLD's OpenCL kernel does not")
ax.annotate(f"0.0% bars drawn at {FLOOR} for visibility", xy=(0.99, 0.02),
            xycoords="axes fraction", ha="right", fontsize=7, color="0.4")
ax.legend(handles=[mpatches.Patch(color=col("cugen"), label="cugen is a party to this comparison"),
                   mpatches.Patch(color=col("qldgpu"), label="qLD-GPU is a party to this comparison")],
          fontsize=7.4, loc="upper left", frameon=False)
plt.setp(ax.get_xticklabels(), fontsize=7.2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_correctness.png",
                                bbox_inches="tight", pad_inches=0.3); plt.close(fig)
print("wrote 6 figures to", OUT)

# ============================================================ verification
def _verify():
    """Programmatic checks -- this script's output cannot be eyeballed in CI."""
    from PIL import Image
    import collections, sys
    RETIRED = ["#1b6ca8", "#0b9a6d", "#c1440e", "#e08a5a", "#8c2f00",
               "#7b4fa8", "#b08ac9"]
    def rgb(h): return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))
    fails = []
    for f in sorted(os.listdir(OUT)):
        if not f.startswith("fig") or not f.endswith(".png"):
            continue
        im = Image.open(os.path.join(OUT, f)).convert("RGB")
        px = collections.Counter(im.get_flattened_data()
                                 if hasattr(im, "get_flattened_data") else im.getdata())
        def near(target, tol=10):
            t = rgb(target)
            return sum(n for c, n in px.items()
                       if all(abs(a - b) <= tol for a, b in zip(c, t)))
        # tol=0 for the retired palette: several retired colours sit within
        # ~16 of a live one, so any tolerance lets antialiasing blends along a
        # line edge register as a false positive. A DRAWN line reproduces its
        # colour exactly at the core, so exact match is the discriminating test.
        stale = {h: near(h, 0) for h in RETIRED if near(h, 0) > 300}
        live  = {k: near(v[2], 0) for k, v in SERIES.items() if near(v[2], 0) > 100}
        print(f"  {f:<24s} {im.size[0]}x{im.size[1]}  series present: "
              f"{','.join(sorted(live)) or 'NONE'}")
        if stale:
            fails.append(f"{f}: retired palette present {stale}")
        if not live:
            fails.append(f"{f}: no registry colour found")
    if fails:
        print("\n  FAILURES:"); [print("   ", x) for x in fails]; sys.exit(1)
    print("  all figures use only registry colours")

_verify()
