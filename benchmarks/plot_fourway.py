"""Plots for the four-way benchmark: cugen vs plink2 vs qLD-GPU vs qLD-BLIS.

Every number here was MEASURED in this campaign; nothing is modelled. Sources
are named per series so a figure can be traced back to a run.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
C = {"cugen": "#1b6ca8", "cugen_ph": "#0b9a6d", "plink128": "#c1440e",
     "plink13": "#e08a5a", "plink32": "#8c2f00", "qldgpu": "#7b4fa8",
     "qldcpu": "#b08ac9"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- axis 1
P      = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 170949]
cugen  = [0.019, 0.066, 0.101, 0.066, 0.149, 0.355, 0.940, 1.734]
pl13   = [0.070, 0.111, 0.318, 1.048, 5.951, None, None, None]
qldgpu = [0.416, 0.471, 0.945, 2.329, 9.849, 34.995, 27.895, 73.644]
pl128_p, pl128 = [20000, 170949], [0.752, 42.21]
pl32_p,  pl32  = [170949], [89.41]
qcpu_p,  qcpu  = [170949], [85.77]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(P, cugen, "o-", color=C["cugen"], lw=2, ms=5, label="cugen (1×A100)")
ax.plot(pl128_p, pl128, "s-", color=C["plink128"], lw=2, ms=7,
        label="plink2 (128 real cores)")
ax.plot(pl32_p, pl32, "s", color=C["plink32"], ms=7, label="plink2 (32 cores)")
ax.plot([p for p, v in zip(P, pl13) if v], [v for v in pl13 if v], "^--",
        color=C["plink13"], lw=1.4, ms=5, alpha=.9,
        label="plink2 (13.6-core cgroup cap)")
ax.plot(P, qldgpu, "d-", color=C["qldgpu"], lw=1.6, ms=5, label="qLD-GPU (1×A100)")
ax.plot(qcpu_p, qcpu, "v", color=C["qldcpu"], ms=8, label="qLD-BLIS best (8 thr)")
ax.set(xscale="log", yscale="log", xlabel="variants (p), all-pairs",
       ylabel="wall time (s)",
       title="Variant axis — 1000G chr22 MAF≥0.01, n=2,504, r²≥0.2")
ax.annotate(f"24.3× at full chr22\n1.734 s vs 42.21 s", xy=(170949, 42.21),
            xytext=(24000, 120), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=.9, color="0.35"))
ax.legend(fontsize=7.6, loc="upper left", framealpha=.95)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_variant_axis.png"); plt.close(fig)

# ---------------------------------------------------------------- axis 2
N      = [2504, 10000, 50000, 100000, 250000, 500000, 1000000]
cg_un  = [None, 0.1085, 0.1891, 0.3369, 0.4201, 0.6808, 1.1037]   # 2504 = compile outlier
cg_ph  = [0.0643, 0.0745, 0.1838, 0.2567, 0.4821, 0.8268, 1.3238]
pl_n   = [2504, 50000, 100000, 250000, 500000, 1000000]
pl_s   = [0.092, 0.391, 0.766, 2.194, 4.310, 8.510]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
a1.plot([n for n, v in zip(N, cg_un) if v], [v for v in cg_un if v], "o-",
        color=C["cugen"], lw=2, ms=5, label="cugen unphased")
a1.plot(N, cg_ph, "o--", color=C["cugen_ph"], lw=2, ms=5, label="cugen phased")
a1.plot(pl_n, pl_s, "s-", color=C["plink128"], lw=2, ms=6,
        label="plink2 (128 real cores)")
a1.axvline(500000, color="0.6", ls=":", lw=1)
a1.text(520000, 0.075, "biobank\nscale", fontsize=7.5, color="0.35")
a1.set(xscale="log", yscale="log", xlabel="samples (n), p=4,000 fixed",
       ylabel="wall time (s)", title="Sample axis — cugen's weakest axis")
a1.legend(fontsize=8, loc="upper left")
sp = [p / c for p, c, n in zip(pl_s, [0.5633, 0.1891, 0.3369, 0.4201, 0.6808, 1.1037], pl_n)]
a2.plot(pl_n, sp, "o-", color=C["cugen"], lw=2, ms=6)
a2.axhline(1, color="0.6", ls=":", lw=1)
a2.set(xscale="log", xlabel="samples (n)", ylabel="cugen speedup vs plink2 @128",
       title="Speedup grows only slowly with n")
for x, y in zip(pl_n, sp):
    a2.annotate(f"{y:.1f}×", (x, y), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_sample_axis.png"); plt.close(fig)

# ---------------------------------------------------------------- phased
PP  = [1000, 5000, 20000, 50000, 170949]
cph = [0.0175, 0.0780, 0.1677, 0.4044, 1.8038]
pph = [0.0867, 0.4525, 12.4072, 121.9037, 1013.8759]
fig, (b1, b2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
b1.plot(PP, cph, "o-", color=C["cugen_ph"], lw=2, ms=6, label="cugen r2_phased")
b1.plot(PP, pph, "s-", color=C["plink128"], lw=2, ms=6, label="plink2 --r2-phased")
b1.set(xscale="log", yscale="log", xlabel="variants (p)", ylabel="wall time (s)",
       title="Phased LD — identical output, 19,888/19,888 pairs agree")
b1.legend(fontsize=8, loc="upper left")
b2.plot(PP, [p / c for p, c in zip(pph, cph)], "o-", color=C["cugen_ph"], lw=2, ms=6,
        label="phased")
b2.plot([1000, 2000, 5000, 10000, 20000], [0.070/0.019, 0.111/0.066, 0.318/0.101,
        1.048/0.066, 5.951/0.149], "s--", color=C["plink13"], lw=1.6, ms=5,
        label="unphased (13.6c plink2)")
b2.set(xscale="log", yscale="log", xlabel="variants (p)", ylabel="cugen speedup",
       title="Phased scales better: plink2 pays per-pair for phase")
for x, y in zip(PP, [p / c for p, c in zip(pph, cph)]):
    b2.annotate(f"{y:.0f}×", (x, y), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=7.5)
b2.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_phased.png"); plt.close(fig)

# ---------------------------------------------------------------- threads
fig, (c1, c2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
T  = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192]
TS = [10.79, 8.602, 5.906, 3.53, 2.084, 1.24, 0.885, 0.787, 0.752, 0.725]
c1.plot(T, [TS[0] / t for t in TS], "s-", color=C["plink128"], lw=2, ms=5,
        label="plink2 measured")
c1.plot(T, T, ":", color="0.65", lw=1.2, label="linear")
c1.axvline(128, color="0.6", ls=":", lw=1)
c1.annotate("14.35× at 128 cores\n(my Amdahl fit\npredicted 7.4×)", xy=(128, 14.35),
            xytext=(3.2, 30), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=.9, color="0.35"))
c1.set(xscale="log", yscale="log", xlabel="--threads (128 physical cores)",
       ylabel="speedup vs 1 thread", title="plink2 scales — extrapolation from 32 cores failed")
c1.legend(fontsize=8, loc="upper left")
QT, QS = [1, 8, 32, 64, 128], [371.015, 85.773, 125.813, 369.807, 370.687]
c2.plot(QT, QS, "d-", color=C["qldcpu"], lw=2, ms=7)
c2.axhline(371.015, color="0.6", ls=":", lw=1)
c2.text(9, 392, "single-thread time", fontsize=7.5, color="0.4")
for x, y in zip(QT, QS):
    c2.annotate(f"{QS[0]/y:.2f}×", (x, y), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=7.5)
c2.set(xscale="log", xlabel="--threads (45 tasks available)", ylabel="wall time (s)",
       title="qLD-BLIS collapses: chunk = tasks//threads = 0")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_threads.png"); plt.close(fig)

# ---------------------------------------------------------------- cost
fig, ax = plt.subplots(figsize=(7.6, 4.3))
labels = ["cugen\nphased", "cugen", "qLD-GPU", "plink2\n@32c", "plink2\n@128c",
          "qLD-BLIS\nbest", "plink2\n--r2-phased", "qLD-BLIS\n@128c"]
cost = [7.97e-4, 7.66e-4, 3.25e-2, 4.37e-2, 7.70e-2, 1.57e-1, 4.48e-1, 6.76e-1]
cols = [C["cugen_ph"], C["cugen"], C["qldgpu"], C["plink32"], C["plink128"],
        C["qldcpu"], C["plink13"], C["qldcpu"]]
bars = ax.bar(labels, cost, color=cols)
ax.set(yscale="log", ylabel="$ per job (log)",
       title="Cost per full-chr22 all-pairs job (1.46e10 pairs)")
for b, v in zip(bars, cost):
    ax.annotate(f"${v:.1e}", (b.get_x() + b.get_width()/2, v),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.4)
ax.annotate("100.6× cugen", xy=(4, 7.70e-2), xytext=(4.4, 4e-3), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=.9, color="0.35"))
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_cost.png"); plt.close(fig)

# ---------------------------------------------------------------- correctness
fig, ax = plt.subplots(figsize=(7.6, 4.0))
names = ["cugen vs plink2\n(unphased)", "cugen vs plink2\n(--r2-phased, PGEN)",
         "cugen vs qLD\n-blis", "cugen vs qLD\n-gpu",
         "qLD -gpu vs\nplink2 (bed EM)"]
frac  = [0.0, 0.0, 0.0, 60.5, 97.3]
mx    = [5.7e-7, 5.71e-7, 1.37e-6, 5.79e-2, 6.09e-1]
b = ax.bar(names, [f + 0.06 for f in frac],
           color=["#0b9a6d", "#0b9a6d", "#0b9a6d", "#c1440e", "#8c2f00"])
ax.set(ylabel="% of shared pairs differing >1e-4",
       title="Correctness — three implementations agree; qLD's GPU kernel does not")
for bb, f, m in zip(b, frac, mx):
    ax.annotate(f"{f:.1f}%\nmax Δ={m:.2e}", (bb.get_x()+bb.get_width()/2, f+0.06),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_correctness.png"); plt.close(fig)
print("wrote 6 figures to", OUT)
