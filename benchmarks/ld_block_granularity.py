"""Read-side cost of block granularity -- the number the scan side lacks.

Their finding: bigger blocks are faster AND smaller to WRITE (2.1x wall, 5%
bytes). The cost they could not price is that a block is the unit of both
compression and zone-map skipping, so a coarser block means a threshold or
point query decompresses more to answer the same question.

MAX_BLOCK_PAIRS = 65536 exists precisely because of this: at
block_variants=4096 an uncapped block held 2.5M pairs and variant() cost
360 ms; capping pairs took it to 2.94 ms, a 122x win. So raising the cap to
4,194,304 is undoing the fix that motivated the constant -- the question is
how much.

Source data on the network volume, ALL OUTPUT ON LOCAL DISK: /workspace is
MooseFS and measured 64 MB/s against 2 GB/s local, which is trap #2 in both
handoffs.
"""
import json, os, random, sys, time
sys.path.insert(0, "/workspace/repo")
import numpy as np
from cugen import ld as L, ldio

SRC = "/workspace/chr22_dos.cugen"
OUT = "/root/bench"
os.makedirs(OUT, exist_ok=True)
MIN_R2 = 0.05
random.seed(7)


def timeit(fn, reps=3):
    fn()                                   # warm the page cache and any import
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


def measure(path):
    """Read-side metrics for one written dataset."""
    ds = ldio.open_ld(path) if os.path.isdir(path) else ldio.read_ld(path)
    is_ds = os.path.isdir(path)
    nb = (sum(len(ldio.read_ld(os.path.join(path, s["file"])).blocks)
              for s in ds.shards) if is_ds else len(ds.blocks))

    # a real point lookup: 200 random row variants
    rd = ds
    vs = [int(v) for v in np.random.default_rng(3).integers(0, 51100, 200)]
    def pt():
        for v in vs:
            rd.variant(v)
    t_pt = timeit(pt, reps=1) / len(vs) * 1000.0

    rd.reset_counters()
    for v in vs[:50]:
        rd.variant(v)
    blocks_per_lookup = rd.blocks_read / 50.0

    out = {"n_blocks": nb, "variant_ms": t_pt,
           "blocks_per_lookup": blocks_per_lookup}

    for t in (0.2, 0.5, 0.8):
        rd.reset_counters()
        t0 = time.perf_counter(); n = len(rd.above(min_r2=t)[0])
        wall = time.perf_counter() - t0
        out[f"above_{t}_s"] = wall
        out[f"above_{t}_rows"] = n
        out[f"above_{t}_blocks"] = rd.blocks_read
    out["skip_frac_0.8"] = 1.0 - out["above_0.8_blocks"] / max(nb, 1)

    t0 = time.perf_counter(); nr = len(rd.rows()[0])
    out["full_scan_s"] = time.perf_counter() - t0
    out["rows"] = nr
    return out


def run(label, grid, key, **fixed):
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    rows = []
    for val in grid:
        path = os.path.join(OUT, f"{key}_{val}.cugenld")
        os.system(f"rm -rf {path}")
        kw = dict(fixed); kw[key] = val
        t0 = time.perf_counter()
        # max_pairs counts CANDIDATE pairs, not emitted rows, so all-pairs on
        # 51,100 variants trips the 1e8 default at 1.3e9 candidates while
        # emitting ~1.3e7. This is why genomewide.sh passes 1e15.
        n = L.ld_matrix(SRC, stats=("r", "r2"), min_r2=MIN_R2, output=path,
                        stream=True, backend="gpu", verbose=False,
                        max_pairs=10**15, **{f"ld_{key}": val})
        write_s = time.perf_counter() - t0
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(path) for f in fs) \
            if os.path.isdir(path) else os.path.getsize(path)
        m = measure(path)
        m.update({key: val, "write_s": write_s, "bytes": size,
                  "b_per_pair": size / max(m["rows"], 1), "emitted": n})
        rows.append(m)
        print(f"  {key}={val:>9,}  write {write_s:6.2f}s  {m['b_per_pair']:5.2f} B/pair  "
              f"blocks {m['n_blocks']:>6,}  variant() {m['variant_ms']:7.3f} ms  "
              f"above(.8) {m['above_0.8_s']*1000:8.1f} ms  scan {m['full_scan_s']:6.2f}s")
    return rows


res = {}
res["max_block_pairs"] = run(
    "ALL-PAIRS regime: sweep max_block_pairs (block_variants at default)",
    [65_536, 262_144, 1_048_576, 4_194_304], "max_block_pairs")
res["block_variants"] = run(
    "sweep block_variants (max_block_pairs at default)",
    [4_096, 16_384, 65_536, 262_144], "block_variants")

with open("/root/block_sweep.json", "w") as f:
    json.dump(res, f, indent=1)
print("\nwrote /root/block_sweep.json")
