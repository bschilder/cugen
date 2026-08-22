"""Windowed LD across a WINDOW-SIZE sweep, with the scan/write split measured.

Why this script exists: the windowed regime is the one place cugen was ever
measured LOSING to plink2 (`vs_win_cudf.json`, 4.62 s vs 1.95 s at p=170,949,
w=500), and that measurement predates the fused epilogue kernel -- the change
that took SM utilisation from 1-4% to 100%. All-pairs was re-measured on the
fused path (`psweep.json`); windowed never was. So the standing windowed number
is stale by one major optimisation, and every windowed projection since has
been a model rather than a measurement.

Two things it settles that a single wall-clock cannot:

1. WHERE THE TIME GOES. The model says windowed is write-bound (pairs:rows ~16,
   against ~1,389 all-pairs) -- but that is inferred, never measured. This times
   `_scan_gpu_fused` directly and subtracts, so scan and write are separated
   instead of assumed.
2. WHETHER THE FUSED PATH WAS EVEN TAKEN. `on_device` needs `output is not
   None` and `annotation is None`, and `fused` needs no missingness and no
   window_kb. Miss any and you silently measure the host pandas path. That is
   not hypothetical: `vs_win_cudf.json` reports `cugen_gpu_gib = 0.0` for all
   three rows, because the pre-fix sampler read CuPy's pool while cuDF was
   allocating through RMM. A benchmark that cannot say which path it ran is
   not a measurement. This one asserts, and records the versions.

    python benchmarks/win_sweep.py --cugen chr22_maf01.cugen \
        --bfile chr22_maf01 --windows 100,500,2000,10000,50000,none \
        --grid 170949 --min-r2 0.2,0.1,0.05 --out win_sweep.json
"""
import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from _peak import PeakSampler

import cugen as cg
from cugen import ld as _ld
from cugen.ld import ld_matrix


def env_record():
    """Everything needed to know which code path a row came from."""
    rec = {"has_cupy": bool(_ld.HAS_CUPY), "has_cudf": bool(_ld.HAS_CUDF),
           "python": platform.python_version()}
    if _ld.HAS_CUPY:
        import cupy as cp
        dev = cp.cuda.Device()
        rec.update(cupy=cp.__version__, gpu=cp.cuda.runtime.getDeviceProperties(
                       dev.id)["name"].decode(),
                   compute_capability=".".join(map(str, dev.compute_capability)),
                   cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                   total_mem_gib=round(dev.mem_info[1] / 2**30, 2))
        # Which allocator is live matters for reading peak memory at all.
        alloc = cp.cuda.get_allocator()
        rec["cupy_allocator"] = getattr(alloc, "__qualname__", repr(alloc))
    if _ld.HAS_CUDF:
        import cudf
        rec["cudf"] = cudf.__version__
        try:
            import rmm
            rec["rmm"] = rmm.__version__
        except ImportError:
            rec["rmm"] = None
    return rec


def run_plink(bfile, keep, out, window, min_r2, threads):
    cmd = ["plink2", "--bfile", bfile, "--extract", keep,
           "--r2-unphased", "allow-ambiguous-allele", "cols=chrom,pos,id",
           "--ld-window", str(999999 if window is None else window + 1),
           "--ld-window-kb", "999999", "--ld-window-r2", str(min_r2),
           "--threads", str(threads), "--out", out, "--silent"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode:
        return None, None, None, (r.stdout + r.stderr)[-200:]
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2
    n = None
    for ext in (".vcor", ".ld"):
        if os.path.exists(out + ext):
            with open(out + ext) as fh:
                n = sum(1 for _ in fh) - 1
            break
    return dt, rss, n, None


def scan_only(path, p, window, min_r2, reps):
    """Time _scan_gpu_fused alone -- the GPU scan with no frame, no write.

    Uses the same reader/rows/precision the public entry point would, so the
    subtraction against ld_matrix is apples to apples.
    """
    import cupy as cp
    reader = cg.io.read_cugen(path, device=0)
    rows = np.arange(int(reader.n_variants), dtype=np.int64)[:p]
    tf32 = _ld._resolve_precision("auto", int(reader.n_samples), False)
    B = _ld._tile_size_for(int(reader.n_samples), window=window, fused=True)
    B = max(256, min(B, len(rows)))
    ts, found = [], 0
    for _ in range(reps):
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        ii, jj, rr = _ld._scan_gpu_fused(reader, rows, window, min_r2,
                                         tile_size=None, verbose=False,
                                         tf32=tf32)
        cp.cuda.Device().synchronize()
        ts.append(time.perf_counter() - t0)
        found = int(ii.size)
        del ii, jj, rr
    # The scan allocates an optimistic output buffer and re-runs the whole
    # epilogue if it overflows -- that doubles scan work and is invisible in a
    # wall-clock. Recompute the predicted capacity so the row can flag it.
    cap = min(int(200e6), max(1 << 20, p * (int(window) if window else 4096)))
    return statistics.median(ts), found, B, bool(found > cap), bool(tf32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cugen", required=True)
    ap.add_argument("--bfile", default=None, help="plink2 bfile prefix; omit to skip plink2")
    ap.add_argument("--windows", default="100,500,2000,10000,50000,none",
                    help="variant-count windows; 'none' means all-pairs")
    ap.add_argument("--grid", default=None, help="p values (default: whole file)")
    ap.add_argument("--min-r2", default="0.2,0.1,0.05",
                    help="comma-separated thresholds. 0.2 is the only value ever\n                         benchmarked on real data; 0.1/0.05 are here to replace\n                         the Sved-model extrapolation with a measurement.")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="win_sweep.json")
    a = ap.parse_args()

    hdr = cg.io.read_cugen_header(a.cugen)
    pmax = int(hdr["n_variants"])
    grid = [int(x) for x in a.grid.split(",")] if a.grid else [pmax]
    windows = [None if w.strip().lower() == "none" else int(w)
               for w in a.windows.split(",")]
    thresholds = [float(x) for x in a.min_r2.split(",")]

    env = env_record()
    if not env["has_cupy"]:
        sys.exit("CuPy unavailable -- this benchmark measures the GPU path.")
    if not env["has_cudf"]:
        sys.exit("cuDF unavailable: ld_matrix would fall back to the HOST pandas "
                 "path and this would silently measure the wrong thing. "
                 "Install cudf or do not trust the result.")
    print(json.dumps(env, indent=2))
    print(f"\nfile: {a.cugen}  n_samples={hdr['n_samples']}  n_variants={pmax:,}")

    bim = pd.read_csv(a.bfile + ".bim", sep="\t", header=None) if a.bfile else None

    # Warm the kernel and cuBLAS handles ONCE. psweep.py omits this, which is
    # why its p=1,000 row (0.41 s) is slower than its p=5,000 row.
    ld_matrix(a.cugen, variant_range=(0, min(512, pmax)), min_r2=0.5,
              stats=("r", "r2"), sign_reference="major", output="/tmp/_warm.tsv",
              max_pairs=10**15, verbose=False)

    print(f"\n{'p':>8s} {'window':>8s} {'min_r2':>7s} {'pairs':>15s} {'rows':>12s} "
          f"{'total_s':>8s} {'scan_s':>8s} {'write_s':>8s} {'write%':>7s} "
          f"{'tile':>6s} {'GiB':>7s} | {'plink_s':>8s} {'x':>7s}")
    print("-" * 132)

    out = {"env": env, "file": dict(hdr), "runs": []}
    for p in grid:
        for w in windows:
            for mr2 in thresholds:
                sc_s, sc_found, B, retried, tf32 = scan_only(
                    a.cugen, p, w, mr2, a.reps)

                ts = []
                for _ in range(a.reps):
                    with PeakSampler() as smp:
                        t0 = time.perf_counter()
                        df = ld_matrix(a.cugen, variant_range=(0, p), window=w,
                                       min_r2=mr2, stats=("r", "r2"),
                                       sign_reference="major",
                                       output="/tmp/_ws.tsv", max_pairs=10**15,
                                       verbose=False)
                        ts.append(time.perf_counter() - t0)
                    peak = smp.peak_gib
                tot = statistics.median(ts)

                # Sanity: the fused path must have produced the same survivors
                # as the public call, or the subtraction below is meaningless.
                if sc_found != len(df):
                    print(f"  ^ WARNING scan_only found {sc_found:,} but "
                          f"ld_matrix returned {len(df):,} -- paths diverged")

                npairs = (p * (p - 1) // 2 if w is None
                          else sum(min(w, p - 1 - i) for i in range(p)))
                write_s = tot - sc_s
                pl_s = pl_rss = pl_n = None
                if bim is not None:
                    bim.iloc[:p, 1].to_csv("/tmp/_ws_keep.txt", index=False,
                                           header=False)
                    pl_s, pl_rss, pl_n, err = run_plink(
                        a.bfile, "/tmp/_ws_keep.txt", "/tmp/_ws_pl", w, mr2,
                        a.threads)
                    if err:
                        print(f"  plink2 FAILED: {err[:100]}")

                rec = dict(p=p, window=w, min_r2=mr2, pairs=npairs,
                           rows=int(len(df)), total_s=round(tot, 4),
                           scan_s=round(sc_s, 4), write_s=round(write_s, 4),
                           write_frac=round(write_s / tot, 4) if tot else None,
                           ns_per_row=round(write_s / len(df) * 1e9, 1) if len(df) else None,
                           scan_pairs_per_s=round(npairs / sc_s, 1) if sc_s else None,
                           tile_size=B, capacity_retry=retried, tf32=tf32,
                           peak_gib=peak, plink_s=pl_s, plink_rss_gib=pl_rss,
                           plink_rows=pl_n,
                           speedup=round(pl_s / tot, 3) if pl_s else None)
                out["runs"].append(rec)
                json.dump(out, open(a.out, "w"), indent=2)

                ws = f"{w:,}" if w else "all"
                print(f"{p:>8,} {ws:>8s} {mr2:>7.2f} {npairs:>15,} {len(df):>12,} "
                      f"{tot:>8.3f} {sc_s:>8.3f} {write_s:>8.3f} "
                      f"{write_s/tot*100 if tot else 0:>6.1f}% {B:>6,} {peak:>7.3f} | "
                      f"{(f'{pl_s:.3f}' if pl_s else '-'):>8s} "
                      f"{(f'{pl_s/tot:.2f}x' if pl_s else '-'):>7s}"
                      + ("  RETRY" if retried else ""))

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
