"""How much does pairwise-complete vs one-X actually change R, and is R PSD?

This is the measurement behind not switching _step5b's XtX path.
"""
import numpy as np, cupy as cp, tempfile, os
from cugen import ld as L, ldio
from cugen.write import write_cugen

rng = np.random.default_rng(5)
n, p = 800, 150
lat = rng.random(n)
X = np.zeros((n, p), dtype=np.uint8)
for v in range(p):
    f = rng.uniform(0.2, 0.8); mix = rng.uniform(0.1, 0.7)
    X[:, v] = ((mix*lat + (1-mix)*rng.random(n)) < f).astype(np.uint8) + (rng.random(n) < f)
X = np.clip(X, 0, 2)

print(f"{n} samples x {p} variants\n")
print(f"  {'missing rate':>12} {'max |dR|':>10} {'mean |dR|':>10} "
      f"{'min eig (pairwise)':>19} {'PSD?':>6}")
for rate in (0.0, 0.02, 0.05, 0.10, 0.20):
    Xm = X.copy()
    if rate > 0:
        Xm[rng.random((n, p)) < rate] = 3          # 3 = missing in 2bit
    d = tempfile.mkdtemp(); path = os.path.join(d, "x.cugen")
    write_cugen(path, Xm)
    out = os.path.join(d, "x.cugenld")
    L.ld_matrix(path, stats=("r","r2"), min_r2=0.0, output=out,
                backend="numpy", verbose=False)
    R_pair = ldio.read_ld(out).dense(n_variants=p)

    # single-X path: missing -> column mean, one Gram matrix
    Xf = Xm.astype(np.float64); Xf[Xm == 3] = np.nan
    col = np.nanmean(Xf, axis=0); Xf = np.where(np.isnan(Xf), col, Xf)
    Xc = Xf - Xf.mean(axis=0)
    C = Xc.T @ Xc / n
    sd = np.sqrt(np.maximum(np.diag(C), 1e-12))
    R_one = C / np.outer(sd, sd); np.fill_diagonal(R_one, 1.0)

    dR = np.abs(R_pair - R_one)
    ev = np.linalg.eigvalsh(R_pair).min()
    print(f"  {rate:>11.0%} {dR.max():>10.4f} {dR.mean():>10.5f} "
          f"{ev:>19.2e} {'yes' if ev > -1e-8 else 'NO':>6}")
