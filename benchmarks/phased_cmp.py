"""Pair-level agreement: cugen r2_phased vs plink2 --r2-phased on phased PGEN."""
import glob, csv
BIM="/root/data/chr22_maf01.bim"; pos=[int(l.split("\t")[3]) for l in open(BIM)]
def cg(p):
    d={}
    with open(p) as fh:
        r=csv.reader(fh,delimiter="\t"); h=next(r)
        ga,gb,i2=h.index("gidx_a"),h.index("gidx_b"),h.index("R2_PHASED")
        for row in r:
            if row:
                a,b=pos[int(row[ga])],pos[int(row[gb])]
                d[(min(a,b),max(a,b))]=float(row[i2])
    return d
def pk(p):
    d={}
    with open(p) as fh:
        h=fh.readline().lstrip("#").rstrip("\n").split("\t")
        ia,ib=h.index("POS_A"),h.index("POS_B"); i2=len(h)-1
        for ln in fh:
            f=ln.rstrip("\n").split("\t")
            if len(f)>i2: a,b=int(f[ia]),int(f[ib]); d[(min(a,b),max(a,b))]=float(f[i2])
    return d
C=cg("/tmp/ph_cg.tsv"); P=pk(sorted(glob.glob("/tmp/ph_pl.vcor*"))[0])
both=set(C)&set(P)
print(f"  cugen={len(C):,}  plink2={len(P):,}  shared={len(both):,}  "
      f"cugen_only={len(set(C)-set(P)):,}  plink_only={len(set(P)-set(C)):,}")
if both:
    diffs=sorted((abs(C[k]-P[k]), k) for k in both)
    bad=[d for d,_ in diffs if d>1e-4]
    print(f"  |dr2|>1e-4: {len(bad):,} ({len(bad)/len(both):.3%})")
    print(f"  max |dr2| = {diffs[-1][0]:.3e}   median = {diffs[len(diffs)//2][0]:.3e}")
    for d,k in diffs[-3:]:
        print(f"    worst {k} cugen={C[k]:.9g} plink={P[k]:.9g} d={d:.2e}")
