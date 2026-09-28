"""Storage formats and query latency for LD results, measured on a GPU.

Real 1000 Genomes chr22:20-30Mb, MAF >= 0.01 -> 51,100 variants x 3,202 samples,
on one A100-SXM4-80GB. Results in benchmarks/results/STORAGE.md.

Fixture (streamed, not downloaded):
    U=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
    bcftools view -r chr22:20000000-30000000 -v snps -m2 -M2 -Oz -o raw.vcf.gz "$U"
    bcftools index -t raw.vcf.gz
    bcftools view -q 0.01:minor -Q 0.99:minor raw.vcf.gz -Oz -o maf01.vcf.gz
    python -c "from cugen.convert import vcf2cugen; \
               vcf2cugen('maf01.vcf.gz','chr22_dos.cugen')"

Usage:  python benchmarks/ld_storage.py --cugen chr22_dos.cugen --out DIR
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cugen import ld as L, ldio  # noqa: E402


def say(m):
    print(f"\n===== {m} =====", flush=True)


def timeit(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cugen", required=True)
    ap.add_argument("--out", default="/tmp/ld_storage")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}

    say("1. Significance-layer overhead on the fused GPU path")
    P = 20000
    base = None
    res["overhead"] = []
    for label, kw in (
            ("r2 only", dict(stats=("r", "r2"), min_r2=0.2)),
            ("+ chi2, p", dict(stats=("r", "r2", "chi2", "p"), min_r2=0.2)),
            ("+ max_p filter", dict(stats=("r", "r2", "p"), max_p=1e-8)),
            ("+ bonferroni", dict(stats=("r", "r2", "p"),
                                  correction="bonferroni")),
            ("+ BH-FDR", dict(stats=("r", "r2", "p"), correction="fdr"))):
        out = f"{a.out}/o.parquet"
        t = timeit(lambda: L.ld_matrix(a.cugen, variant_range=(0, P),
                                       window=500, output=out, verbose=False,
                                       **kw))
        n = len(L.ld_matrix(a.cugen, variant_range=(0, P), window=500,
                            verbose=False, **kw))
        base = base or t
        print(f"  {label:22s} {t*1000:8.1f} ms  {t/base:5.2f}x   rows={n:,}")
        res["overhead"].append(dict(case=label, sec=t, rows=n, ratio=t / base))

    say("2. Storage formats")
    ref = None
    res["formats"] = []
    for ext in (".tsv", ".feather", ".parquet", ".npz", ".cugenld"):
        out = f"{a.out}/ld{ext}"
        if os.path.exists(out):
            os.remove(out)
        t = timeit(lambda: L.ld_matrix(a.cugen, window=500, min_r2=0.2,
                                       stats=("r", "r2"), output=out,
                                       verbose=False), reps=2)
        sz = os.path.getsize(out)
        n = len(L.ld_matrix(a.cugen, window=500, min_r2=0.2,
                            stats=("r", "r2"), verbose=False))
        ref = ref or sz
        print(f"  {ext:<10} {sz/1e6:8.1f} MB  {sz/n:7.2f} B/pair  "
              f"{ref/sz:5.1f}x  scan+write {t:6.2f}s  rows={n:,}")
        res["formats"].append(dict(ext=ext, bytes=sz, rows=n, per_pair=sz / n,
                                   sec=t, vs_tsv=ref / sz))

    say("3. All-pairs, and the min_r2 lever")
    res["allpairs"] = []
    for min_r2 in (0.2, 0.05, 0.01):
        out = f"{a.out}/all.cugenld"
        if os.path.exists(out):
            os.remove(out)
        t0 = time.perf_counter()
        df = L.ld_matrix(a.cugen, min_r2=min_r2, stats=("r", "r2"),
                         output=out, max_pairs=10**15, verbose=False)
        t = time.perf_counter() - t0
        sz = os.path.getsize(out)
        print(f"  min_r2={min_r2:<5} rows={len(df):>12,}  {t:7.1f}s  "
              f"{sz/1e6:8.1f} MB  {sz/len(df):5.2f} B/pair")
        res["allpairs"].append(dict(min_r2=min_r2, rows=len(df), sec=t,
                                    bytes=sz, per_pair=sz / len(df)))

    say("4. Query latency")
    rd = ldio.read_ld(f"{a.out}/all.cugenld")
    print(f"  {rd.n_pairs:,} pairs, {len(rd.blocks):,} blocks, "
          f"{rd.bytes_per_pair():.2f} B/pair, "
          f"mean {rd.n_pairs/len(rd.blocks):,.0f} pairs/block")
    rd.rows()
    total = rd.blocks_read
    res["queries"] = []
    for t_ in (0.05, 0.2, 0.5, 0.8):
        rd.reset_counters()
        t0 = time.perf_counter()
        n = rd.above(min_r2=t_)[0].size
        el = (time.perf_counter() - t0) * 1000
        print(f"  above(r2>={t_:<5}) {n:>12,} rows {el:>9.1f} ms  "
              f"blocks {rd.blocks_read:>5}/{total} ({rd.blocks_read/total:4.0%})")
        res["queries"].append(dict(min_r2=t_, rows=n, ms=el,
                                   blocks=rd.blocks_read, total=total))
    ts = []
    for v in range(0, 5000, 250):
        t0 = time.perf_counter()
        rd.variant(v)
        ts.append(time.perf_counter() - t0)
    res["variant_ms"] = float(np.median(ts) * 1000)
    print(f"  variant() median over {len(ts)} lookups: "
          f"{res['variant_ms']:.2f} ms")

    json.dump(res, open(f"{a.out}/storage.json", "w"), indent=2)
    print(f"\nwrote {a.out}/storage.json")


if __name__ == "__main__":
    main()
