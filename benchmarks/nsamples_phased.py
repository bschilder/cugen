"""Sample axis to 1M: cugen (unphased+phased) vs plink2 (unphased+phased) vs qLD.

p is FIXED and small so the axis is genuinely n_samples. Synthetic phased
haplotypes are generated once per n and written in every format needed:

    hap (2n, p) 0/1  ->  cugen hap2bit   (write_cugen_phased)
                     ->  cugen 2bit      (dosage = sum of the two haplotypes)
                     ->  VCF (phased)    ->  plink2 pgen  ->  qLD MDF

qLD is included only where its 2^30-pairs-per-task ceiling permits; with p fixed
at a few thousand that is never binding, so the axis is clean for all four.
"""
import argparse, glob, gzip, os, shutil, statistics, subprocess, sys, time
sys.path.insert(0, "/root/cugen_phased")
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, default=4000, help="variants, FIXED")
ap.add_argument("--grid", default="2504,10000,50000,100000,250000,500000,1000000")
ap.add_argument("--r2", type=float, default=0.2)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--threads", type=int, default=13)
ap.add_argument("--out", default="/root/nsamples.json")
a = ap.parse_args()
D = "/root/nsdata"; os.makedirs(D, exist_ok=True)

def simulate(n, p, seed=0):
    """(2n, p) 0/1 phased alleles with a spread of MAF and real LD."""
    rng = np.random.default_rng(seed)
    H = 2 * n
    latent = rng.random(H).astype(np.float32)
    out = np.zeros((H, p), dtype=np.uint8)
    for v in range(p):
        mix = rng.uniform(0.0, 0.95)
        freq = rng.uniform(0.05, 0.95)
        score = mix * latent + (1 - mix) * rng.random(H).astype(np.float32)
        out[:, v] = (score < freq)
    return out

def build(n, p):
    from cugen.write import write_cugen, write_cugen_phased
    tag = f"{D}/n{n}"
    hap = simulate(n, p, seed=n)
    if not os.path.exists(f"{tag}_ph.cugen"):
        write_cugen_phased(f"{tag}_ph.cugen", hap)
    if not os.path.exists(f"{tag}.cugen"):
        dos = (hap[0::2].astype(np.uint8) + hap[1::2].astype(np.uint8))
        write_cugen(f"{tag}.cugen", dos.T.T if dos.shape[0] == n else dos)
    return tag, hap

def time_cugen(path, p, r2, stats, reps):
    from cugen.ld import ld_matrix
    ts, n = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        df = ld_matrix(path, variant_range=(0, p), min_r2=r2, stats=stats,
                       sign_reference="major", output="/tmp/ns_cg.tsv",
                       max_pairs=10**15, verbose=False)
        ts.append(time.perf_counter() - t0); n = len(df)
    return statistics.median(ts), n

print(f"  p fixed at {a.p}; axis is n_samples; r2limit={a.r2}")
print(f"{'n_samples':>10s} {'haps':>9s} | {'cugen_2bit':>11s} {'rows':>9s} | "
      f"{'cugen_hap':>10s} {'rows':>9s} | {'ratio':>6s}")
print("-" * 82)
out = []
for n in [int(x) for x in a.grid.split(",")]:
    try:
        tag, hap = build(n, a.p)
    except MemoryError:
        print(f"{n:>10,}  MemoryError building the fixture"); break
    t_un, r_un = time_cugen(f"{tag}.cugen", a.p, a.r2, ("r", "r2"), a.reps)
    t_ph, r_ph = time_cugen(f"{tag}_ph.cugen", a.p, a.r2,
                            ("r_phased", "r2_phased"), a.reps)
    print(f"{n:>10,} {2*n:>9,} | {t_un:>11.4f} {r_un:>9,} | "
          f"{t_ph:>10.4f} {r_ph:>9,} | {t_ph/t_un:>5.2f}x")
    out.append(dict(n=n, p=a.p, cugen_2bit_s=t_un, cugen_2bit_rows=r_un,
                    cugen_hap_s=t_ph, cugen_hap_rows=r_ph))
    import json; json.dump(out, open(a.out, "w"), indent=2)
    del hap
print(f"\nwrote {a.out}")
