"""Does clumping's verdict change with SAMPLE COUNT?

Every clumping benchmark so far ran at 1000 Genomes' n=2,504, and at that size
cugen LOSES the standard-GWAS configuration to plink2 (0.94 s vs 0.37 s):
with 192 index variants there is not enough arithmetic to amortise moving
170k variants' genotypes onto the device.

n is the GEMM's contraction dimension, so it scales the part cugen
parallelises and none of the part it does not:

    unpack + fp32 planes   O(tile * n)     <- grows
    GEMM                   O(tile * W * n) <- grows
    epilogue threshold     O(tile * W)     <- flat
    parallel MIS           O(edges)        <- flat
    membership + output    O(clumps)       <- flat

plink2's LD kernel also grows with n, but on CPU. The ld_matrix sample sweep
found cugen's advantage going 1.2x -> 70x between n=2,504 and n=500,000, so
the standard-clumping loss is plausibly an artifact of tiny n rather than a
property of the design. This measures that instead of assuming it.

Matched PLINK .bed and .cugen are generated at each n so both tools see
identical genotypes, with p held fixed.

    python benchmarks/clump_nscaling.py --out clump_nscale.json
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "benchmarks") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))

import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd

from _peak import PeakSampler


def make_fixture(n, p, seed, workdir):
    """Matched .bed + .cugen + annotation + sumstats at one sample size.

    Writes the PLINK1 .bed BINARY directly, streaming one LD block at a time.
    The first version wrote a VCF and shelled out to --make-bed, which is
    p * n genotype STRINGS in Python -- 1e10 of them at p=20,000 and
    n=500,000. It never finished, and looked from outside exactly like a hung
    pod. nscaling.py already had the right approach; this borrows it.

    Streaming also bounds peak host memory at O(block * n) rather than
    O(p * n), which would be 10 GB at the top of the grid.
    """
    from cugen.convert import bed2cugen
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(np.arange(1, 20_000_000), size=p, replace=False))
    ids = [f"v{i}" for i in range(p)]
    base = os.path.join(workdir, f"n{n}")

    # PLINK codes: 00 hom A1, 10 het, 11 hom A2, packed little-endian within
    # the byte (sample 0 in the LOW bits) -- the opposite order from .cugen,
    # which is why bed2cugen remaps rather than copying.
    lut = np.array([0b00, 0b10, 0b11], dtype=np.uint8)
    pad = (-n) % 4
    written = 0
    with open(base + ".bed", "wb") as f:
        f.write(bytes([0x6C, 0x1B, 0x01]))
        while written < p:
            blk = int(min(rng.integers(5, 40), p - written))
            bas = rng.integers(0, 2, size=(n, 2)).sum(1)      # the LD block
            for _ in range(blk):
                noise = rng.random(n) < rng.uniform(0.0, 0.5)
                g = np.where(noise, rng.integers(0, 3, n), bas).astype(np.uint8)
                codes = lut[g]
                if pad:
                    codes = np.concatenate([codes, np.zeros(pad, np.uint8)])
                c = codes.reshape(-1, 4)
                f.write((c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4)
                         | (c[:, 3] << 6)).astype(np.uint8).tobytes())
            written += blk
    with open(base + ".bim", "w") as f:
        for v in range(p):
            f.write(f"1\t{ids[v]}\t0\t{pos[v]}\tA\tG\n")
    with open(base + ".fam", "w") as f:
        for s in range(n):
            f.write(f"S{s} S{s} 0 0 0 -9\n")
    bed2cugen(base + ".bed", base + ".cugen", verbose=False)

    ann = pd.DataFrame({"gidx": np.arange(p), "ID": ids, "POS": pos,
                        "CHR": "1"})
    pv = rng.uniform(0, 1, size=p)
    hits = rng.choice(p, size=max(2, p // 1000), replace=False)
    pv[hits] = 10.0 ** (-rng.uniform(5, 20, size=len(hits)))
    ss = os.path.join(workdir, f"n{n}_ss.tsv")
    with open(ss, "w") as f:            # 6 sig figs, matching plink's echo
        f.write("ID\tP\n")
        for i, q in zip(ids, pv):
            f.write(f"{i}\t{q:.6g}\n")
    return base, ann, ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=20000)
    ap.add_argument("--n-grid", default="2504,10000,50000,100000,200000,500000")
    ap.add_argument("--workdir", default="/root/nscale")
    ap.add_argument("--out", default="clump_nscale.json")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)

    from cugen.ld import ld_clump

    grid = [int(x) for x in a.n_grid.split(",")]
    results = []
    warm = False
    for n in grid:
        rec = {"n": n, "p": a.p}
        print(f"\n=== n = {n:,} samples, p = {a.p:,} variants ===", flush=True)
        try:
            t0 = time.perf_counter()
            print(f"  building matched .bed/.cugen ...", flush=True)
            base, ann, ss = make_fixture(n, a.p, 20260812, a.workdir)
            print(f"  fixture ready in {time.perf_counter() - t0:.1f} s",
                  flush=True)
        except Exception as e:                                # noqa: BLE001
            rec["error"] = f"fixture: {type(e).__name__}: {e}"
            print(f"  fixture FAILED: {e}", flush=True)
            results.append(rec)
            json.dump(results, open(a.out, "w"), indent=2)
            continue

        if not warm:      # absorb CUDA context + NVRTC before ANY timing
            t0 = time.perf_counter()
            try:
                ld_clump(base + ".cugen", ss, annotation=ann.head(500),
                         backend="gpu", verbose=False)
                print(f"  warm-up: {time.perf_counter() - t0:.2f} s "
                      f"(not counted)", flush=True)
            except Exception as e:                            # noqa: BLE001
                print(f"  warm-up skipped: {e}", flush=True)
            warm = True

        # Two configurations: the one cugen loses at n=2,504, and the one it
        # wins. If the verdict is really about sample size, the first should
        # cross over and the second should widen.
        for label, p1, r2 in (("standard", 1e-4, 0.5), ("C+T", 1.0, 0.1)):
            try:
                with PeakSampler() as s:
                    t0 = time.perf_counter()
                    cg = ld_clump(base + ".cugen", ss, annotation=ann, p1=p1,
                                  p2=0.01, r2=r2, kb=250, backend="gpu",
                                  verbose=False)
                    dt = time.perf_counter() - t0
                rec[f"{label}_gpu_s"] = round(dt, 3)
                rec[f"{label}_gpu_gib"] = s.peak_gib
                rec[f"{label}_clumps"] = int(len(cg))
            except Exception as e:                            # noqa: BLE001
                rec[f"{label}_gpu_error"] = f"{type(e).__name__}: {e}"
                print(f"  cugen[{label}] FAILED: {e}", flush=True)
                continue

            t0 = time.perf_counter()
            r = subprocess.run(
                ["plink2", "--bfile", base, "--clump", ss, "--clump-unphased",
                 "--clump-p1", str(p1), "--clump-p2", "0.01",
                 "--clump-r2", str(r2), "--clump-kb", "250",
                 "--out", os.path.join(a.workdir, f"pk{n}{label}"), "--silent"],
                capture_output=True, text=True)
            pk = time.perf_counter() - t0
            rec[f"{label}_plink_s"] = round(pk, 3)
            if r.returncode:
                rec[f"{label}_plink_error"] = (r.stdout + r.stderr)[-300:]
            else:
                rec[f"{label}_speedup"] = round(pk / rec[f"{label}_gpu_s"], 2)
            print(f"  {label:9s} cugen {rec[f'{label}_gpu_s']:7.2f} s  "
                  f"plink2 {pk:7.2f} s  "
                  f"{rec.get(f'{label}_speedup', 0):6.2f}x  "
                  f"peak {rec.get(f'{label}_gpu_gib', 0):.2f} GiB", flush=True)

        results.append(rec)
        json.dump(results, open(a.out, "w"), indent=2)
        for ext in (".bed", ".bim", ".fam", ".cugen"):
            try:
                os.remove(base + ext)
            except OSError:
                pass

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
