# LD test fixtures

`ld_fixture.npy` -- (12 variants, 60 samples) uint8 dosages, 3 = missing.
Variant 11 is monomorphic; variants 4 and 9 carry missing calls; ALT frequency
straddles 0.5 so the "alt" and "major" sign conventions genuinely diverge.

Golden files produced with **PLINK v2.0.0-a.7.1 M1 (4 May 2026)**:

```
plink2 --vcf ld_fixture.vcf --make-pgen --out t
plink2 --pfile t --r-unphased allow-ambiguous-allele \
  cols=chrom,pos,id,ref,alt,maj,nonmaj,freq \
  --ld-window 999999 --ld-window-kb 99999 --ld-window-r2 0 --out u
plink2 --pfile t --r-phased allow-ambiguous-allele \
  cols=chrom,pos,id,ref,alt,maj,nonmaj,freq,d,dprime \
  --ld-window 999999 --ld-window-kb 99999 --ld-window-r2 0 --out p
```

`--ld-window-r2 0` is mandatory: plink2 defaults to 0.2 and would silently drop
most pairs. `--ld-window` has NO default in plink2 (it was 10 in plink 1.9), so
it is set explicitly. `allow-ambiguous-allele` is a MODIFIER on --r*, not a
standalone flag.

The .vcor files carry ~6 significant figures, so compare with rtol=1e-6.
