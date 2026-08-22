"""plink2 LD at biobank sample counts, on 128 real cores.

Writes PLINK .bed directly instead of importing a multi-GB VCF. bed is
variant-major, 2 bits per sample, LSB-first within each byte:
    00 hom A1 | 01 missing | 10 het | 11 hom A2
so ALT dosage 0/1/2 maps to 00/10/11 -- and importantly there is no
missing code in use here, matching the cugen fixture.

Same latent-factor simulation as the cugen sweep so the LD structure (and
therefore the emitted row count, which drives write cost) is comparable.
"""
import argparse, glob, os, subprocess, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, default=4000)
ap.add_argument("--grid", default="2504,50000,100000,250000,500000,1000000")
ap.add_argument("--r2", type=float, default=0.2)
ap.add_argument("--threads", type=int, default=128)
ap.add_argument("--out", default="/root/biobank.json")
a = ap.parse_args()
D = "/root/bbdata"; os.makedirs(D, exist_ok=True)

def simulate_dosage(n, p, seed):
    rng = np.random.default_rng(seed)
    latent = rng.random(n).astype(np.float32)
    dos = np.zeros((p, n), dtype=np.uint8)
    for v in range(p):
        mix = rng.uniform(0.0, 0.95)
        f = rng.uniform(0.05, 0.95)
        s1 = mix * latent + (1 - mix) * rng.random(n).astype(np.float32)
        s2 = mix * latent + (1 - mix) * rng.random(n).astype(np.float32)
        dos[v] = (s1 < f).astype(np.uint8) + (s2 < f).astype(np.uint8)
    return dos

_CODE = np.array([0b00, 0b10, 0b11], dtype=np.uint8)   # dosage 0,1,2

def write_bed(tag, dos, n, p):
    bpv = (n + 3) // 4
    with open(tag + ".bed", "wb") as fh:
        fh.write(bytes([0x6c, 0x1b, 0x01]))
        for v in range(p):
            c = _CODE[dos[v]]
            pad = (-n) % 4
            if pad:
                c = np.concatenate([c, np.zeros(pad, dtype=np.uint8)])
            c = c.reshape(-1, 4)
            packed = (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6))
            fh.write(packed.astype(np.uint8).tobytes())
    with open(tag + ".bim", "w") as fh:
        for v in range(p):
            fh.write(f"1\tv{v}\t0\t{(v + 1) * 1000}\tA\tG\n")
    with open(tag + ".fam", "w") as fh:
        for i in range(n):
            fh.write(f"F{i}\tI{i}\t0\t0\t0\t-9\n")

print(f"  p fixed at {a.p}; plink2 --threads {a.threads} on {os.cpu_count()} CPUs")
print(f"{'n_samples':>10s} {'build_s':>9s} {'plink2_s':>10s} {'rows':>10s} {'GB_bed':>7s}")
print("-" * 54)
res = []
for n in [int(x) for x in a.grid.split(",")]:
    tag = f"{D}/n{n}"
    t0 = time.perf_counter()
    if not os.path.exists(tag + ".bed"):
        dos = simulate_dosage(n, a.p, seed=n)
        write_bed(tag, dos, n, a.p)
        del dos
    tb = time.perf_counter() - t0
    for f in glob.glob("/tmp/bb.*"):
        os.remove(f)
    t0 = time.perf_counter()
    r = subprocess.run(["plink2", "--bfile", tag, "--r2-unphased",
                        "allow-ambiguous-allele", "cols=chrom,pos,id",
                        "--ld-window", "999999", "--ld-window-kb", "999999",
                        "--ld-window-r2", str(a.r2), "--threads", str(a.threads),
                        "--out", "/tmp/bb", "--silent"],
                       capture_output=True, text=True)
    tp = time.perf_counter() - t0
    f = sorted(glob.glob("/tmp/bb.vcor*"))
    rows = (sum(1 for _ in open(f[0])) - 1) if f else None
    gb = os.path.getsize(tag + ".bed") / 1e9
    print(f"{n:>10,} {tb:>9.1f} {tp:>10.3f} "
          f"{(f'{rows:,}' if rows is not None else 'FAIL'):>10s} {gb:>7.2f}")
    if r.returncode:
        print(f"      plink2 rc={r.returncode}: {(r.stdout + r.stderr)[-160:]}")
    res.append(dict(n=n, p=a.p, build_s=tb, plink_s=tp, rows=rows, bed_gb=gb,
                    threads=a.threads))
    import json; json.dump(res, open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
