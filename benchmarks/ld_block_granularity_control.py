"""Same sweep, warmed and run in BOTH orders. The control for trap #1.

Sweep 1 measured max_block_pairs=65,536 at 12.90 s. Sweep 2 measured the
identical configuration at 1.76 s. A control you have already measured moving
by 7.3x means the first row of each sweep is paying kernel compilation, and
any "bigger blocks are faster" conclusion drawn from a single ascending pass
is partly measuring warm-up.
"""
import json, os, sys, time
sys.path.insert(0, "/workspace/repo")
import numpy as np
from cugen import ld as L, ldio

SRC = "/workspace/chr22_dos.cugen"
OUT = "/root/bench2"
os.makedirs(OUT, exist_ok=True)
GRID = [65_536, 262_144, 1_048_576, 4_194_304]


def write_once(val, tag):
    path = os.path.join(OUT, f"{tag}_{val}.cugenld")
    os.system(f"rm -rf {path}")
    t0 = time.perf_counter()
    n = L.ld_matrix(SRC, stats=("r", "r2"), min_r2=0.05, output=path,
                    stream=True, backend="gpu", verbose=False,
                    max_pairs=10**15, ld_max_block_pairs=val)
    wall = time.perf_counter() - t0
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(path) for f in fs)
    return wall, size, n, path


print("=== warm-up write (discarded) ===")
w, _, _, p = write_once(262_144, "warm")
print(f"  discarded: {w:.2f} s"); os.system(f"rm -rf {p}")

print("\n=== ASCENDING, warmed ===")
asc = {}
for v in GRID:
    wall, size, n, path = write_once(v, "asc")
    asc[v] = (wall, size / n)
    print(f"  {v:>9,}  {wall:6.2f} s  {size/n:5.3f} B/pair")
    os.system(f"rm -rf {path}")

print("\n=== DESCENDING, warmed ===")
desc = {}
for v in reversed(GRID):
    wall, size, n, path = write_once(v, "desc")
    desc[v] = (wall, size / n)
    print(f"  {v:>9,}  {wall:6.2f} s  {size/n:5.3f} B/pair")
    os.system(f"rm -rf {path}")

print("\n=== agreement between the two orders ===")
print(f"  {'max_block_pairs':>16} {'asc':>7} {'desc':>7} {'spread':>8}")
for v in GRID:
    a, d = asc[v][0], desc[v][0]
    print(f"  {v:>16,} {a:7.2f} {d:7.2f} {max(a,d)/min(a,d):7.2f}x")

json.dump({"asc": {str(k): v for k, v in asc.items()},
           "desc": {str(k): v for k, v in desc.items()}},
          open("/root/block_sweep2.json", "w"), indent=1)
