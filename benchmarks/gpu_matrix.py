"""Portability + performance probe for cugen.ld across GPU generations.

Self-contained: builds its own seeded .cugen, so every device runs a bit-identical
workload with no download. Emits one JSON blob per device.

    python benchmarks/gpu_matrix.py --out result.json

What it answers:
  * does the NVRTC kernel compile on this architecture at all?
  * does the GPU path agree with the CPU reference on this architecture?
  * does the tile auto-tuner cope with this device's memory (6 GB .. 180 GB)?
  * how long does a fixed workload take, and what is peak pool memory?
"""
import os
import sys

# Make `python benchmarks/<script>.py` work from a source checkout: running a
# script puts ITS directory on sys.path, not the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "tests") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "tests"))

import argparse
import json
import platform
import subprocess
import sys
import time

import numpy as np

from _peak import PeakSampler



def device_info():
    import cupy as cp
    d = cp.cuda.Device()
    free, total = d.mem_info
    cc = d.compute_capability   # CuPy returns e.g. "86", not (8, 6)
    try:
        name = cp.cuda.runtime.getDeviceProperties(d.id)["name"].decode()
    except Exception:
        name = "unknown"
    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        drv = "unknown"
    return {
        "gpu_name": name,
        "compute_capability": (f"{cc[:-1]}.{cc[-1]}" if isinstance(cc, str)
                               else f"{cc[0]}.{cc[1]}"),
        "total_mem_gib": round(total / 2**30, 2),
        "free_mem_gib": round(free / 2**30, 2),
        "driver": drv,
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "python": platform.python_version(),
    }


def build_fixture(path, n_samples, n_variants, seed=20260810):
    from conftest import simulate_haplotypes
    from cugen.write import write_cugen
    dos = simulate_haplotypes(n_samples, n_variants, seed=seed, missing_rate=0.0)
    write_cugen(str(path), dos.T.astype(np.uint8))
    return dos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gpu_matrix.json")
    ap.add_argument("--samples", type=int, default=2504)   # 1KG phase3 size
    ap.add_argument("--p-grid", default="2000,5000,10000,20000")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    rec = {"label": a.label, "ok": False, "stage": "import",
           "n_samples": a.samples, "runs": [], "errors": []}
    try:
        import cupy as cp
        rec.update(device_info())
        rec["stage"] = "fixture"
    except Exception as e:                                   # noqa: BLE001
        rec["errors"].append(f"cupy import/device: {type(e).__name__}: {e}")
        json.dump(rec, open(a.out, "w"), indent=2)
        print(json.dumps(rec, indent=2))
        return

    from cugen.ld import ld_matrix

    try:
        pmax = max(int(x) for x in a.p_grid.split(","))
        build_fixture("/tmp/bench.cugen", a.samples, pmax)
        rec["stage"] = "compile"

        # --- does the kernel compile and agree with the CPU reference? -----
        build_fixture("/tmp/small.cugen", 200, 24)
        t0 = time.perf_counter()
        gpu = ld_matrix("/tmp/small.cugen", backend="gpu", verbose=False)
        rec["kernel_compile_s"] = round(time.perf_counter() - t0, 3)
        cpu = ld_matrix("/tmp/small.cugen", backend="numpy", verbose=False)
        g = gpu.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True)
        c = cpu.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True)
        rec["correctness"] = {
            "pairs": int(len(g)),
            "pair_sets_equal": bool(
                set(zip(g.gidx_a, g.gidx_b)) == set(zip(c.gidx_a, c.gidx_b))),
            "max_abs_err_R": float(np.abs(g["R"] - c["R"]).max()),
            "max_abs_err_D": float(np.abs(g["D"] - c["D"]).max()),
            "max_abs_err_DP": float(np.abs(g["DP"] - c["DP"]).max()),
        }
        rec["stage"] = "bench"

        # --- fixed workload, r+r2 only (the hot path) ----------------------
        pool = cp.get_default_memory_pool()
        for p in [int(x) for x in a.p_grid.split(",")]:
            pool.free_all_blocks()
            try:
                with PeakSampler() as sampler:
                    t0 = time.perf_counter()
                    # max_pairs is a user-facing guard against accidental huge
                    # runs; the probe is deliberate, so opt out. Without this
                    # every device fails at p=20,000 (200M pairs) on the guard,
                    # not on any hardware limit.
                    df = ld_matrix("/tmp/bench.cugen", variant_range=(0, p),
                                   stats=("r", "r2"), min_r2=0.5,
                                   max_pairs=10**15, backend="gpu",
                                   verbose=False)
                    dt = time.perf_counter() - t0
                npairs = p * (p - 1) // 2
                rec["runs"].append({
                    "p": p, "wall_s": round(dt, 4),
                    "peak_pool_gib": sampler.peak_gib,
                    "pairs": npairs,
                    "pairs_per_s": round(npairs / dt, 1),
                    "emitted": int(len(df)),
                })
                print(f"p={p:>7,}  {dt:7.3f}s  peak={sampler.peak_gib:6.3f} GiB"
                      f"  {npairs/dt/1e6:9.1f} Mpair/s")
            except Exception as e:                            # noqa: BLE001
                rec["runs"].append({"p": p, "error": f"{type(e).__name__}: {e}"})
                rec["errors"].append(f"p={p}: {type(e).__name__}: {e}")
                print(f"p={p:>7,}  FAILED {type(e).__name__}: {e}")
            finally:
                pool.free_all_blocks()
        rec["ok"] = bool(rec["correctness"]["pair_sets_equal"]
                         and rec["correctness"]["max_abs_err_R"] < 1e-4
                         and any("wall_s" in r for r in rec["runs"]))
        rec["stage"] = "done"
    except Exception as e:                                    # noqa: BLE001
        import traceback
        rec["errors"].append(f"{rec['stage']}: {type(e).__name__}: {e}")
        rec["traceback"] = traceback.format_exc()[-2000:]

    json.dump(rec, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in rec.items() if k != "traceback"}, indent=2))


if __name__ == "__main__":
    main()
