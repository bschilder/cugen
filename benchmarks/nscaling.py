"""Who wins as SAMPLE COUNT grows -- and is the GPU actually saturated?

Every benchmark so far used 1000 Genomes' n=2504, which is tiny. n is the
GEMM's contraction dimension, so arithmetic intensity scales with it: at small
n the tile is thin and launch/epilogue-bound; at biobank n the same tile does
far more FLOPs for the same output. plink2's bitwise kernel is ~O(p^2 n/64)
with no such transition.

Generates matched PLINK .bed + .cugen at each n so both tools see identical
data, then reports wall time, memory, and sampled GPU utilisation.
"""
import os
import subprocess
import sys
import threading
import time

_ROOT = "/root/cugen"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))

import numpy as np

from _peak import PeakSampler
from cugen.convert import bed2cugen
from cugen.ld import ld_matrix

P = 4000                      # fixed variant count; n is the axis under test
NS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1
                       else "2504,10000,50000,200000").split(",")]
DATA = "/root/nscale"
os.makedirs(DATA, exist_ok=True)


def write_bed(prefix, dos):
    """dos is (n_variants, n_samples) uint8 in {0,1,2}. PLINK1 variant-major.

    PLINK codes: 00 = hom A1, 10 = het, 11 = hom A2, 01 = missing, packed
    little-endian within the byte (sample 0 in the LOW bits) -- the opposite
    order from .cugen, which is why bed2cugen remaps rather than copying.
    """
    nv, ns = dos.shape
    lut = np.array([0b00, 0b10, 0b11], dtype=np.uint8)
    with open(prefix + ".bed", "wb") as f:
        f.write(bytes([0x6C, 0x1B, 0x01]))
        pad = (-ns) % 4
        for v in range(nv):
            codes = lut[dos[v]]
            if pad:
                codes = np.concatenate([codes, np.zeros(pad, np.uint8)])
            c = codes.reshape(-1, 4)
            packed = (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6))
            f.write(packed.astype(np.uint8).tobytes())
    with open(prefix + ".bim", "w") as f:
        for v in range(nv):
            f.write(f"22\tv{v}\t0\t{1000*(v+1)}\tA\tG\n")
    with open(prefix + ".fam", "w") as f:
        for s in range(ns):
            f.write(f"S{s} S{s} 0 0 0 -9\n")


class GpuUtil:
    """Sample SM utilisation and memory-bandwidth utilisation via nvidia-smi."""

    def __init__(self, interval=0.15):
        self.interval, self.sm, self.mem, self._stop = interval, [], [], False

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop:
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,utilization.memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                a, b = out.split(",")
                self.sm.append(float(a)); self.mem.append(float(b))
            except Exception:                                  # noqa: BLE001
                pass
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop = True
        self._t.join(timeout=2)

    def summary(self):
        if not self.sm:
            return "n/a"
        return (f"SM {np.mean(self.sm):4.0f}%/{max(self.sm):3.0f}%peak  "
                f"MEMBW {np.mean(self.mem):4.0f}%/{max(self.mem):3.0f}%peak")


print(f"p = {P:,} variants (fixed), all-pairs, min_r2=0.2, "
      f"plink2 threads={os.cpu_count()}")
from cugen.ld import _resolve_precision
print(f"{'n samples':>10s} | {'plink2 s':>9s} | {'cg fp32':>8s} | {'cg tf32':>8s} "
      f"{'GPU GiB':>8s} | {'tf32 gain':>9s} | {'vs plink2 (min-max)':>21s} | GPU util (tf32)")
print("-" * 118)

for n in NS:
    pre = f"{DATA}/n{n}"
    if not os.path.exists(pre + ".cugen"):
        rng = np.random.default_rng(7)
        latent = rng.random(n)
        dos = np.zeros((P, n), dtype=np.uint8)
        for v in range(P):
            mix = rng.uniform(0.0, 0.9)
            lat = mix * latent + (1 - mix) * rng.random(n)
            af = rng.uniform(0.1, 0.5)
            t1, t2 = np.quantile(lat, [(1 - af) ** 2, 1 - af ** 2])
            dos[v] = (lat > t1).astype(np.uint8) + (lat > t2).astype(np.uint8)
        write_bed(pre, dos)
        bed2cugen(pre + ".bed", pre + ".cugen", verbose=False)
        del dos

    t0 = time.perf_counter()
    r = subprocess.run(
        ["plink2", "--bfile", pre, "--r2-unphased", "allow-ambiguous-allele",
         "cols=chrom,pos,id", "--ld-window", "999999", "--ld-window-kb",
         "999999", "--ld-window-r2", "0.2", "--threads", str(os.cpu_count()),
         "--out", "/tmp/_n", "--silent"], capture_output=True, text=True)
    pl = time.perf_counter() - t0
    if r.returncode:
        print(f"{n:>10,} | plink2 FAILED: {(r.stdout + r.stderr)[-140:]}")
        continue
    import resource
    pl_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 ** 2

    def run_cugen(prec):
        # warm once so cuBLAS handle/plan setup is not charged to the timing
        ld_matrix(pre + ".cugen", stats=("r", "r2"), min_r2=0.99,
                  max_pairs=10 ** 15, output="/tmp/_w.tsv", precision=prec,
                  sign_reference="major", backend="gpu", verbose=False)
        with GpuUtil() as util, PeakSampler() as smp:
            t0 = time.perf_counter()
            ld_matrix(pre + ".cugen", stats=("r", "r2"), min_r2=0.2,
                      max_pairs=10 ** 15, output="/tmp/_n.tsv", precision=prec,
                      sign_reference="major", backend="gpu", verbose=False)
            return time.perf_counter() - t0, smp.peak_gib, util.summary()

    REPS = 3
    # Single runs on a shared cloud host swung 1.9x between sweeps on
    # identical code, so report the median of REPS and the observed spread.
    f32s = [run_cugen("fp32")[0] for _ in range(REPS)]
    t32r = [run_cugen("tf32") for _ in range(REPS)]
    t32s = [x[0] for x in t32r]
    gib = max(x[1] for x in t32r)
    util = t32r[-1][2]
    pls = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        subprocess.run(
            ["plink2", "--bfile", pre, "--r2-unphased", "allow-ambiguous-allele",
             "cols=chrom,pos,id", "--ld-window", "999999", "--ld-window-kb",
             "999999", "--ld-window-r2", "0.2", "--threads", str(os.cpu_count()),
             "--out", "/tmp/_n", "--silent"], capture_output=True, text=True)
        pls.append(time.perf_counter() - t0)
    med = lambda v: sorted(v)[len(v) // 2]
    pl_m, f_m, t_m = med(pls), med(f32s), med(t32s)
    active = _resolve_precision("tf32", n, False)
    print(f"{n:>10,} | {pl_m:>9.2f} | {f_m:>8.2f} | {t_m:>8.2f} {gib:>8.2f} | "
          f"{f_m/t_m:>8.2f}x | {pl_m/t_m:>8.2f}x "
          f"[{min(pls)/max(t32s):.1f}-{max(pls)/min(t32s):.1f}] | {util}"
          + ("" if active else "   [TF32 NOT ACTIVE]"))
