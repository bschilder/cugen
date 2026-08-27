"""Genome-wide .cugenld write, sourcing and storing on R2.

Adapted from the original HF version. Two things are deliberately unchanged so
that old -> new comparisons stay apples-to-apples:

  * ``sign_reference="major"`` -- the stored r is coded on the MINOR allele,
    while .cugen dosages count ALT. Reconstructing the raw cross-product from a
    stored r therefore needs r * (-1)^(major_i XOR major_j). Getting this wrong
    once produced |r_adj| = 7.7 -- impossible for a correlation -- after a
    complete table claiming 36% of trans LD survived. Wrong by four orders of
    magnitude. The old dataset was written with "major"; keep it.
  * ``min_r2=0.1`` and the same stats tuples.

What IS new: the panel is the union-ascertained one (per-superpop MAF>=1%), so
the output is ~k^2 = 2.68x larger -- 593 GB -> ~1.6 TB for the pooled unphased
arm. Output exceeds any pod disk, so an uploader thread ships each completed
shard to R2 and deletes it locally.

The manifest is the authority on what is safe to upload, never the directory
listing: LDDatasetWriter appends to manifest.json only after a shard is fully
written, and a shard killed mid-write leaves a temp file that must never be
read.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time

sys.path.insert(0, "/root/cugen_ld")
from cugen.io import CugenReader          # noqa: E402
from cugen.ld import ld_matrix            # noqa: E402

ARM = os.environ["ARM"]                              # "ph" | "un"
POP = os.environ.get("POP", "ALL")
MODE = os.environ.get("MODE", "gw")                  # "gw" | "cis"
CHROM = os.environ.get("CHROM")                      # required when MODE=cis
SUF = "_ph" if ARM == "ph" else ""
BUCKET = "smb-data-prod-scratch"

if MODE == "cis":
    # One chromosome at a time, so every emitted pair is within-chromosome by
    # construction rather than by filtering afterwards. The per-chromosome
    # .cugen pulled from R2 is already the right scope; no merge, no window.
    if not CHROM:
        raise SystemExit("MODE=cis needs CHROM")
    PANEL = f"/root/data/chr{CHROM}.cugen"
    OUTD = f"/root/ld_out/{POP}_{ARM}_cis_chr{CHROM}.cugenld"
    DEST = f"cugen/1kg-30x-grch38/ld/{POP}/ld_{ARM}_cis/chr{CHROM}.cugenld"
else:
    PANEL = f"/root/data/chrALL{SUF}.cugen"
    OUTD = f"/root/ld_out/{POP}_{ARM}.cugenld"
    DEST = f"cugen/1kg-30x-grch38/ld/{POP}/ld_{ARM}"
os.makedirs(os.path.dirname(OUTD), exist_ok=True)

R2 = ["--s3-provider", "Cloudflare",
      "--s3-endpoint", os.environ["r2_endpoint"],
      "--s3-access-key-id", os.environ["r2_access_key"],
      "--s3-secret-access-key", os.environ["r2_secret"],
      "--s3-no-check-bucket"]


def push(local, remote_name):
    """Copy one file to R2. Returns True only on a clean exit."""
    cmd = (["rclone"] + R2 + ["copyto", local,
                              f":s3:{BUCKET}/{DEST}/{remote_name}",
                              "--stats-one-line"])
    return subprocess.run(cmd, capture_output=True).returncode == 0


q, done, stats = queue.Queue(), threading.Event(), {"up": 0, "bytes": 0}


def uploader():
    seen = set()
    while not (done.is_set() and q.empty()):
        try:
            q.get(timeout=5)
        except queue.Empty:
            pass
        man = os.path.join(OUTD, "manifest.json")
        if not os.path.exists(man):
            continue
        try:
            shards = json.load(open(man)).get("shards", [])
        except Exception:                                      # noqa: BLE001
            continue                                           # torn mid-write
        for sh in shards:
            fn = sh.get("file") or sh.get("path")
            if not fn or fn in seen:
                continue
            p = os.path.join(OUTD, os.path.basename(fn))
            if not os.path.exists(p):
                continue
            sz = os.path.getsize(p)
            if push(p, os.path.basename(fn)):
                os.remove(p)                     # free disk immediately
                seen.add(fn)
                stats["up"] += 1
                stats["bytes"] += sz
                if stats["up"] % 20 == 0:
                    free = os.statvfs("/root").f_bavail * 4096 / 1e9
                    print(f"    [up] {stats['up']} shards, "
                          f"{stats['bytes']/1e9:.1f} GB, {free:.0f} GB free",
                          flush=True)
            else:
                print(f"    [up] FAILED {fn} -- will retry next sweep",
                      flush=True)
    man = os.path.join(OUTD, "manifest.json")
    if os.path.exists(man) and push(man, "manifest.json"):
        print("    [up] manifest uploaded", flush=True)


threading.Thread(target=uploader, daemon=True).start()


def ticker():
    while not done.is_set():
        time.sleep(30)
        q.put(1)


threading.Thread(target=ticker, daemon=True).start()

r = CugenReader(PANEL)
P, N = r.n_variants, r.n_samples
r.close()
sel = ("r_phased", "r2_phased") if ARM == "ph" else ("r", "r2")
print(f"  panel {PANEL}  n={N:,}  p={P:,}  arm={ARM}  pop={POP}  mode={MODE}",
      flush=True)
print(f"  -> r2:{BUCKET}/{DEST}", flush=True)

# flush_rows sized from ACTUAL free device memory, not a constant. The original
# scan used 200,000,000, which needs ~10.4 GB: 4.0 GB of i/j/r output buffers,
# plus cp.stack([g_j, g_i]) at 3.2 GB and a comparable lexsort workspace inside
# _flush. That fits an 80 GB A100 and OOMs a 24 GB RTX 4090 -- which is exactly
# how the pooled arms died at 24.7 GB allocated of 24.8 GB available.
#
# 52 bytes/row covers all three allocations. Take 15% of free memory for them;
# the row cache, the two plane buffers and the packed residency need the rest.
import cupy as cp                                            # noqa: E402
_free = int(cp.cuda.Device().mem_info[0])
FLUSH_ROWS = max(5_000_000, min(200_000_000, int(0.15 * _free / 52)))
print(f"  free {_free/1e9:.1f} GB -> flush_rows={FLUSH_ROWS:,} "
      f"(~{FLUSH_ROWS*52/1e9:.2f} GB of flush buffers)", flush=True)

t0 = time.perf_counter()
n = ld_matrix(PANEL, variant_range=(0, P), min_r2=0.1, stats=sel,
              sign_reference="major", stream=True, output=OUTD,
              flush_rows=FLUSH_ROWS, max_pairs=10**18, verbose=False)
dt = time.perf_counter() - t0
print(f"\n  ROWS WRITTEN = {n:,} in {dt/3600:.2f} h", flush=True)
done.set()
q.put(1)
time.sleep(20)
print(f"  uploaded {stats['up']} shards / {stats['bytes']/1e9:.1f} GB", flush=True)
print("LDWRITE_DONE", flush=True)
