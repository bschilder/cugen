"""Cost per LD job, from MEASURED times and REAL list prices only."""
A100 = 1.59          # Runpod A100 SXM 80GB, SECURE, /hr
C7A  = 6.56896       # AWS c7a.32xlarge on-demand us-east-1, from the Pricing API
# every number below was measured on the hardware named, same fixture,
# 1000G chr22 MAF>=0.01, 170,949 variants x 2,504 samples, r2 >= 0.2
JOBS = {
 "chr22 all-pairs, p=20,000 (2.000e8 pairs)": [
   ("cugen",       "1x A100",            A100, 0.149,   562857, "genotype r2"),
   ("plink2 @128", "c7a.32xlarge 128c",  C7A,  0.752,   562857, "genotype r2"),
 ],
 "chr22 all-pairs, FULL p=170,949 (1.461e10 pairs)": [
   ("cugen",        "1x A100",           A100, 1.734, 10517635, "genotype r2"),
   ("plink2 @128",  "c7a.32xlarge 128c", C7A, 42.21, 10517635, "genotype r2"),
   ("plink2 @32",   "32-core (Runpod)",  1.76, 89.41, 10517635, "genotype r2"),
   ("qLD-GPU",      "1x A100",           A100, 73.64, 5847458, "haplotype r2"),
   ("qLD-BLIS best","c7a.32xlarge 8thr", C7A, 85.77, None,     "haplotype r2"),
   ("qLD-BLIS @128","c7a.32xlarge 128c", C7A, 370.69, None,    "haplotype r2"),
 ],
 "chr22 PHASED, FULL p=170,949 (1.461e10 pairs)": [
   ("cugen phased", "1x A100",           A100, 1.8038, 5864576, "haplotype r2"),
   ("plink2 --r2-phased", "A100 host 13.6c", A100, 1013.88, 5864576, "haplotype r2"),
 ],
}
for job, rows in JOBS.items():
    print(f"\n=== {job} ===")
    print(f"  {'tool':<16s} {'hardware':<20s} {'$/hr':>7s} {'wall':>10s} "
          f"{'$/job':>11s} {'rows':>10s}  statistic")
    base = None
    for tool, hw, price, wall, rows_n, stat in rows:
        cost = wall * price / 3600
        if base is None: base = cost
        print(f"  {tool:<16s} {hw:<20s} {price:>7.2f} {wall:>9.3f}s "
              f"{cost:>11.3e} {(f'{rows_n:,}' if rows_n else '-'):>10s}  {stat}")
    print(f"  -> cheapest is {rows[0][0]}; ratios vs it:")
    for tool, hw, price, wall, rows_n, stat in rows[1:]:
        print(f"       {tool:<16s} {wall*price/3600/base:>9.1f}x the cost, "
              f"{wall/rows[0][3]:>8.1f}x the time")

print("\n=== break-even: the $/hr at which plink2 matches cugen ===")
for lab, tcg, tpl in (("p=20,000", 0.149, 0.752),
                      ("full chr22 unphased", 1.734, 42.21),
                      ("full chr22 PHASED", 1.8038, 1013.88)):
    print(f"  {lab:<22s} break-even = ${A100*tcg/tpl:.4f}/hr for the CPU box "
          f"(actual: ${C7A:.2f})")
print("\n  A CPU host would have to cost that little to match cugen. No such")
print("  machine exists, so the cost conclusion carries no price assumption.")

print("\n=== genome-wide projection, MAF>=1% (13.7 M variants) ===")
cg_rate = 1.461e10 / 1.734          # measured pairs/s
pl_rate = 1.461e10 / 42.21
q_rate  = 1.461e10 / 73.64
BD = 0.0541
for lab, pairs in (("per-chromosome all-pairs", BD * 13.7e6**2 / 2),
                   ("true cross-chromosome",    13.7e6**2 / 2)):
    print(f"  {lab} ({pairs:.2e} pairs)")
    for nm, rate, price in (("cugen", cg_rate, A100), ("plink2@128", pl_rate, C7A),
                            ("qLD-GPU", q_rate, A100)):
        t = pairs / rate
        print(f"     {nm:<11s} {t/3600:>8.2f} h   ${t*price/3600:>9.2f}")
    print(f"     plink2 cannot actually run either: RSS ~ p^2 passes 1 TiB at")
    print(f"     ~524,000 variants per chromosome.")
    break
