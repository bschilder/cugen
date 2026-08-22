"""cugen.ld - GPU linkage disequilibrium from packed genotypes.

    ld_matrix(...)    signed r, r^2, signed r^2, D, D', phased variants,
                      and significance: chi2, -log10(p), an exact conditional
                      test, Bonferroni/BH-FDR filtering and inflation control
                      (alias: cg.r2; CLI: cugen ld)
                      Also r2_S / r2_V / r2_VS -- r^2 corrected for population
                      structure and relatedness (estimators only, no p-value)
    ld_prune(...)     prune to approximate linkage equilibrium  (alias:
                      cg.prune). plink2 --indep-pairwise parity. Same greedy
                      algorithm as ld_clump, ranked by allele frequency rather
                      than by p-value -- and with no phase caveat.
    ld_clump(...)     LD-based clumping of association results  (alias:
                      cg.clump). plink2 --clump parity; builds on ld_matrix's
                      r-only path rather than reimplementing LD.

Integer counts, not floating-point correlations
-----------------------------------------------
Every statistic is built from EXACT INTEGER counts over the co-observed
samples, obtained as products of per-variant indicator planes. The plane
values are {0,1,2}, so the O(p^2) work runs on tensor cores and fp32
accumulation is bit-exact while 4*n_samples < 2**24 (n < 4,194,304). Above
that bound the code raises rather than returning quietly-wrong counts. There
is therefore no float screen-then-refine pass here: there is no float error
to correct, and results are identical across GPU generations and precisions.

How much gets counted depends on what you ask for:

  * D and D' need the full 3x3 genotype contingency table, because the
    likelihood is a function of all nine cells. 3 GEMMs per tile on a file
    with no missing calls, 6 with.
  * r, r^2 and signed r^2 need only the cross-product S = sum(g_i g_j) and
    the per-variant moments -- NOT the table. That is 1 GEMM, and it skips
    nine B x B arrays plus the ~8 elementwise passes that derive the outer
    cells. Requesting stats=("r", "r2") therefore selects a materially
    cheaper path, not merely a smaller output.

Both routes consume the same integers, which is why they agree bit-for-bit.

Cost per tile is 3 GEMMs for the full r + D/D' set on a file with no missing
calls -- n21 comes free as n12 transposed, and the marginals collapse to
per-variant constants -- rising to 6 when missingness is present and the
co-observed counts and marginals must themselves be computed pairwise.

PHASE -- read this before trusting D'
-------------------------------------
.cugen stores unphased 2-bit dosages; phase is discarded at conversion. D and
D' are haplotype quantities, so they must be ESTIMATED. We solve the Hill
(1974) likelihood exactly, via the cubic of Gaunt et al. (2007) -- the same
approach plink2 --r2-phased takes (cf. plink-ng cubic_real_roots).

Consequence: on PHASED input (e.g. stock 1000 Genomes VCFs) plink2 uses the
observed haplotypes, and that cannot be reproduced from a .cugen because the
phase is already gone. Measured divergence on identical dosages with phase
toggled is ~8.6x in r^2. Parity with --r2-phased therefore holds on
PHASE-STRIPPED input only:

    bcftools +setGT in.vcf.gz -- -t a -n u | bgzip > unphased.vcf.gz

r and r^2 carry no such caveat; they match --r2-unphased unconditionally.

MISSINGNESS
-----------
Pairwise complete case by default: a sample contributes only if non-missing at
BOTH loci. The header mu_x/sxx are PER-VARIANT non-missing statistics and are
invalid in a pairwise denominator, because variants j and k have different
non-missing sets -- do not "optimise" by reusing them. Likewise this module
does not build on read_to_gpu(), which maps missing -> dosage 0; that is the
failure mode fixed in 85ff1b0 (complete-case association kernel, session 52).
Verified: plink2 --r2-unphased is also pairwise-complete, not mean-imputing.

SIGNIFICANCE
------------
The test of no disequilibrium between two biallelic loci is chi2 = N * r^2 with
1 df -- Park (2019) eq. 1 writes it as 2n D^2 / (pA qA pB qB), the same thing.
N counts GAMETES for the phased statistics and INDIVIDUALS for the composite
ones, which is the factor of two between gametic and composite LD; N_OBS
already carries the right one on every path.

Emitted as -log10(p), never p. p underflows float64 at chi2 ~ 1450 and float32
at chi2 ~ 170, and chi2 = N_OBS * r^2, so at 1000 Genomes size (N_hap = 5008)
float32 dies at r^2 = 0.034 and float64 at r^2 = 0.29. A P column would read as
a flat zero for essentially every linked pair genome-wide.

Filtering by p costs nothing. With no missingness every pair shares one N, so
chi2 is strictly monotone in r^2 and a p-cut IS an r^2-cut: max_p is converted
to min_r2 and handed to the filter the kernel already applies. The number of
tests likewise needs no pass over the data -- it is _count_pairs, closed form
from the row count and the window -- so Bonferroni and BH-FDR are affordable at
genome scale. BH runs in -log10(p) space because at m = 1e14 the thresholds are
around 1e-16 and the p-values are unrepresentable.

exact= adds a two-sided Fisher test on the 2x2 haplotype table (hap2bit only).
This IS the exact permutation p-value, not an approximation to it: permuting
haplotype labels leaves both allele counts fixed, so the permutation null of the
table is the hypergeometric with those margins. Koch (2013) Monte-Carlo sampled
that distribution for ~34,000 CPU-hours. exact='auto' fires only where the
minimum expected cell count is under 5, which bounds the tail sum by
sqrt(5N) ~ 158 terms at 1000 Genomes size -- the pairs that need it are the
pairs where it is cheap.

lambda_gc is off by default and matters on real cohorts. Per-pair p assumes
independent haplotypes, so structure and cryptic relatedness make it
anti-conservative. Measured here on two subpopulations differing by dAF = 0.6
with no gametic LD anywhere in the data: lambda = 920, and all 7,140 pairs look
genome-wide significant on raw p. See the lambda_gc tests.

SIGN
----
r is signed relative to the ALT allele at both variants, consistent with
assoc.py's BETA orientation and directly usable by gpu_susie_rss. plink2 signs
by the MAJOR allele; pass sign_reference="major" for parity. Flipping both
variants leaves r unchanged, so the two conventions differ only when exactly
one variant of a pair has ALT frequency > 0.5.

MEMORY
------
Above the tile size, peak device memory is a function of tile size and sample
count -- NOT of the number of variants. Below it the tile is the whole matrix,
so cost is O(p^2); the two regimes meet where p equals the tile size.

Measured on 1000 Genomes chr22 (2504 samples, A100 80GB): peak held at
~5.6 GiB from p = 20,000 to p = 170,949, i.e. across a 73x growth in pair
count. The plateau height is set by the DEVICE -- the tile auto-tuner takes a
fraction of free VRAM -- so a smaller card plateaus lower, not higher. Whole-
chromosome runs are therefore bounded by wall time, not by memory.

VALIDATION
----------
Against plink2 v2.0.0-a.7.1 (300 samples x 14 variants, ~15% missingness on
two variants, ALT frequency straddling 0.5 so major/ALT orientation actually
diverges on 49 of 91 pairs), max absolute error:

    r  vs UNPHASED_R      4.8e-07        D  vs D        4.7e-07
    r2 vs UNPHASED_R^2    7.2e-07        D' vs DPRIME   5.0e-07
    phased r vs PHASED_R  5.0e-07

which is the 6-significant-figure floor of plink2's text output. The cubic is
additionally cross-checked against an independent multi-start EM and against
brute-force likelihood maximisation.

On REAL data (1000 Genomes chr22, 854,850 pairs) r matches plink2 to 5.3e-07,
but D and D' diverge on 44 pairs (0.005%), and where they diverge they diverge
hard -- 98% of those have opposite signs. Those are tables where the
likelihood has three admissible roots. We take the global maximum, verified
against a dense brute-force scan of the admissible interval; plink2 selects a
different root there. Both implementations claim to maximise a likelihood and
the two formulations look equivalent on inspection, so this is recorded as an
open discrepancy rather than a fault on either side. If you need
plink2-identical D', treat this as a known 0.005% divergence; r and r^2 carry
no such caveat. See test_cubic_picks_the_global_maximum_likelihood_root.

The significance layer has NO plink2 counterpart to compare against: plink2
emits no LD p-values (the author planned {chi-square, df, p-value} columns for
--r2 and never shipped them). It is validated against scipy instead, which is a
genuinely different implementation:

    -log10(p) vs log(2) + log_ndtr(-sqrt(chi2))     < 1e-6 over chi2 in
                                                      [0.5, 1e7]
    exact test vs scipy.stats.fisher_exact          < 1e-4, 40+ pairs
    exact test vs enumerated permutation null       < 1e-12
    BH-FDR vs a textbook rank-walk implementation   exact agreement
    lambda_gc on 1e5 true chi2_1df draws            1.0022
    nAB recovered from float32 r                    0 wrong of 250+ at N=500

Note that scipy.stats.chi2.logsf is NOT usable as an oracle: it computes
log(sf()) and returns inf above chi2 ~ 1450, which is the regime the p-value
helper exists to serve.

Those are simulated panels. On REAL data (1kGP high-coverage phased chr22
20-21Mb, 503 EUR samples, no frequency filter):

    chi2 vs N_hap * plink2's own r^2, 319,600 pairs   9.98e-06 relative,
                                                     plink2's 6-sig-fig floor
    exact test vs scipy.stats.fisher_exact           1.16e-06, 400 real pairs

and the rare-variant tail behaves very differently from a simulated one:
exact='auto' fires on 82% of real pairs against 32% simulated, the asymptotic
test overstates significance on 97.9% of those, and at p < 5e-8 it makes 644
calls where the exact test makes 387 -- 40% false positives. The worst real pair
has r^2 = 1.0000 at N = 1,006 with asymptotic -log10(p) = 220.0 against an exact
3.0, because chi2 = N * r^2 peaks whenever r^2 = 1 no matter how few copies of
the allele produced it. See benchmarks/results/SIGNIFICANCE.md and
tests/data/README.md.

References
----------
Each was checked against the paper before the method went into the code.

Lewontin (1964) Genetics 49(1):49-67            D' and Dmax
    https://pubmed.ncbi.nlm.nih.gov/17248194/
Hill & Robertson (1968) TAG 38(6):226-231       r^2 in finite populations
    https://pubmed.ncbi.nlm.nih.gov/24442307/
Hill (1974) Heredity 33(2):229-239              ML haplotype freqs, unphased
    https://pubmed.ncbi.nlm.nih.gov/4531429/
Weir (1979) Biometrics 35(1):235-254            Burrows' composite LD
    https://pubmed.ncbi.nlm.nih.gov/497335/
Excoffier & Slatkin (1995) MBE 12(5):921-927    EM -- test oracle only
    https://pubmed.ncbi.nlm.nih.gov/7476138/
Gaunt, Rodriguez & Day (2007) BMC Bioinf 8:428  CubeX exact cubic (production)
    https://doi.org/10.1186/1471-2105-8-428
Chang et al. (2015) GigaScience 4:7             PLINK, behavioural reference
    https://pubmed.ncbi.nlm.nih.gov/25722852/
Park (2019) Sci Rep 9:11380                     chi2 = 2n D^2/(pA qA pB qB),
    https://doi.org/10.1038/s41598-019-47832-y  1 df; FDR < 0.05; and the
                                                argument for chi2 over Fisher
                                                on both calibration and cost
Koch, Ristroph & Kirkpatrick (2013)             permutation LRLD -- the method
    PLoS ONE 8:e80754                           the exact test replaces
    https://doi.org/10.1371/journal.pone.0080754
Zaykin, Meng & Weir (2008) Genetics 180:533     n(k-1)(m-1)/(km) R^2 ~ chi2;
    https://pmc.ncbi.nlm.nih.gov/articles/PMC2535703/  composite LD is robust
                                                to single-locus HWE departure
Devlin & Roeder (1999) Biometrics 55(4):997     genomic control; lambda =
    https://pubmed.ncbi.nlm.nih.gov/11315092/   median(chi2)/0.4549364
Yang et al. (2011) AJHG 88(1):76-82             GCTA standardised GRM
    https://doi.org/10.1016/j.ajhg.2010.11.011  (cugen.popstruct.grm)

Implemented as ESTIMATORS ONLY, and the reason for the qualifier matters:

Mangin, Siberchicot, Nicolas, Doligez, This & Cierco-Ayrolles (2012)
    Heredity 108(3):285-291        r^2_S / r^2_V / r^2_VS -- LD corrected for
    https://doi.org/10.1038/hdy.2011.73   population structure and relatedness

Their eqs (1)-(3), cross-checked against the authors' own R implementation
(LDcorSV 1.3.3, CRAN archive), are:

    r^2_S   partial correlation -- the Schur complement of the joint
            covariance of the two loci and the structure matrix S, i.e. the
            residual covariance after regressing both loci on S.
    r^2_V   GLS-centred, V^-1-weighted correlation. With
            F = 1 1' V^-1 / (1' V^-1 1), the paper premultiplies by V^-1/2 and
            forms Sigma^V = (X - FX)' V^-1 (X - FX).
    r^2_VS  the Schur complement of eq (1), taken in the V^-1 metric.

All three reduce to ONE linear map applied to the genotype planes once, after
which an ordinary UNCENTERED r^2 is the answer -- and ld_epilogue_compact
already computes that if it is handed zero sum vectors, since
(n*S - sA*sB)/sqrt(...) collapses to S/sqrt(qA*qB). That is how they are computed here: _corrected_transform builds
one n x n P per dataset and _corrected_r2 takes the Gram of P X. V is only positive SEMI-definite in practice (the paper uses
the Moore-Penrose inverse V^-, and builds a PSD matrix by zeroing negative
eigenvalues of an SVD), so the whitening must come from the eigendecomposition,
V^-1/2 = U Lambda^-1/2 U'; there is no Cholesky factor.

What is NOT implemented is a p-value for them, and that is a statistics
limit rather than a code one. Main text and Supplementary Information (both
read) establish exactly two things:

  Appendix A   Cov(X^l|S, X^m|S) = (1-t)Cov(X^l,X^m|S=0) + t Cov(X^l,X^m|S=1),
               which is 0 for unlinked loci. Unbiasedness, not a distribution.
  Appendix B   in the association model Y = 1u + S beta + X theta + eps, the
               t-statistic at a linked marker is asymptotically Gaussian with
               VARIANCE 1 and expectation sqrt(r^2_S) Esp(t at the causal
               locus). That is the null variance of an ASSOCIATION test that
               carries S as a covariate -- not of the LD measure. Tables 1-3
               add unbiasedness by simulation. The r^2_V / r^2_VS power results
               are asserted to follow "the same steps" and are not proven.

So nothing here gives a null sampling law for the corrected measures, and
chi2 = N * r^2 does not transfer to them: after GLS centring and rank-K
residualisation the effective sample size is not N.

There IS a route the paper does not take, and it is worth naming so nobody
re-derives it from scratch. r^2_S is an ordinary squared PARTIAL correlation, so
classical normal theory gives r^2/(1-r^2) * (N-K-2) ~ F(1, N-K-2) with K the
column rank of the structure matrix. It was rejected here on purpose. That
result needs joint normality, which genotypes do not have, whereas the plain
chi2 = N * r^2 rests on the multinomial structure of the 2x2 table and needs no
such assumption -- and this module already measures the asymptotic test
overstating significance on 97.9% of real rare-variant pairs, where the exact
conditional test is what saves it. Residualising destroys the contingency table,
so the exact test is unavailable and the fallback is WEAKER than the
approximation already known to fail on this data. Shipping it would look like
added rigour and be the opposite. For r^2_V the effective degrees of freedom
after whitening by a rank-deficient V^- is not even clear.

Hence: asking for chi2/p beside a corrected measure raises.

Two further cautions the authors state themselves: which V to use "remains an
open question", and inverting V "drastically slowed down the computation" at
their scale (183 accessions). The n x n eigendecomposition puts r^2_V at
cohort, not biobank, sample sizes. r^2_S is rank-K and has no such limit.

(Main text only; the appendices are in Supplementary Information, unread.)
"""
from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import time as _time

import numpy as np
import pandas as pd

from .io import ENCODING_2BIT, ENCODING_HAP2BIT, read_cugen

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:                                            # noqa: BLE001
    cp = None
    HAS_CUPY = False

# Optional. When present, results never leave the device before being written:
# cudf.Series wraps a CuPy array through __cuda_array_interface__, so building
# the frame is pointer bookkeeping rather than a copy. Measured at 5.2M rows,
# the whole output path goes 3.71 s -> 0.32 s (11.6x), because the old path
# spent most of its time moving data to the host only to hand it to a
# serialiser. Everything still works without it -- this is a fast path, not a
# requirement.
try:
    import cudf
    HAS_CUDF = True
except ImportError:                                            # noqa: BLE001
    cudf = None
    HAS_CUDF = False

__all__ = ["ld_matrix", "ld_clump", "ld_prune", "LDMatrix"]

EPS = 1e-12
# _STATS is the VALIDATION set (every legal name, and the canonical column
# order). _DEFAULT_STATS is what a caller gets when they ask for nothing --
# deliberately dosage-only, so adding a phased statistic here never changes
# the result of an existing call.
_STATS = ("r", "r2", "r2_signed", "d", "dp", "r_phased", "r2_phased",
          "d_phased", "dp_phased", "r2_phased_em",
          "chi2", "p", "p_exact", "chi2_adj", "p_adj",
          "r2_s", "r2_v", "r2_vs")
_DEFAULT_STATS = ("r", "r2", "r2_signed", "d", "dp")
_STAT_COL = {"r": "R", "r2": "R2", "r2_signed": "R2_SIGNED", "d": "D", "dp": "DP",
             "r_phased": "R_PHASED", "r2_phased": "R2_PHASED",
             "d_phased": "D_PHASED", "dp_phased": "DP_PHASED",
             "r2_phased_em": "R2_PHASED_EM",
             "chi2": "CHI2", "p": "NEG_LOG10_P",
             "p_exact": "NEG_LOG10_P_EXACT",
             "chi2_adj": "CHI2_ADJ", "p_adj": "NEG_LOG10_P_ADJ",
             "r2_s": "R2_S", "r2_v": "R2_V", "r2_vs": "R2_VS"}
# The significance pair. Derived from whichever correlation the path computed
# and from N_OBS, so they cost no extra passes over the data. "p" emits
# -log10(p), not p -- see _neglog10_chi2_1df for why p itself is unusable.
_SIG_STATS = frozenset(("chi2", "p", "p_exact", "chi2_adj", "p_adj"))
# median of chi-square with 1 df; the denominator of the genomic-control ratio
_CHI2_1DF_MEDIAN = 0.4549364
# p_exact conditions on the 2x2 HAPLOTYPE table, which dosage data does not
# carry -- unlike chi2/p it is phased-only even though it is not a _PHASED_STATS
# member (those name specific LD estimators; this names a test of any of them).
_EXACT_MODES = ("never", "auto", "always")
# Mangin et al. (2012) bias-corrected r^2. ESTIMATORS ONLY -- the paper derives
# no null sampling distribution for them, so they cannot carry a p-value; see
# the References block at the head of this module.
_CORRECTED_STATS = frozenset(("r2_s", "r2_v", "r2_vs"))
# LDcorSV's Inv.proj.matrix.sdp zeroes eigenvalues below this before inverting.
# The Moore-Penrose inverse is the paper's prescription; the floor is the R
# package's choice and is reproduced here for parity with it.
_PSD_EIGEN_TOL = 1e-5
# Statistics computed from TRUE haplotype counts, which only a HAP2BIT file
# carries. Mixing these with the dosage statistics in one call is refused
# rather than silently served: hap2bit and 2bit share bytes but not meaning
# (see cugen.write), so a caller who asks for "r" on a phased file is asking
# for a number that path cannot produce.
_PHASED_STATS = frozenset(("r_phased", "r2_phased", "d_phased",
                           "dp_phased"))
_FP32_EXACT_MAX_SAMPLES = (1 << 24) // 4      # 4*n must stay below 2**24
_TF32_MIN_CC = 80                             # Ampere; earlier cards have none


class _Tf32:
    """Enable cuBLAS TF32 for a block, restoring the prior math mode after.

    TF32 is normally an accuracy tradeoff -- it truncates inputs to a 10-bit
    mantissa. It is NOT one here: the plane values are exactly {0,1,2}, which
    need two mantissa bits, and TF32 accumulates in fp32 where integer sums
    stay exact to 2**24. Measured on an A100 at n=200,000:
    17.6 -> 110.1 TFLOP/s with max|err| = 0.0 against an fp64 reference.

    fp16 is deliberately NOT offered. Same 10-bit mantissa, but cuBLAS
    accumulates half matmuls in half, so integer exactness ends at 2**11 and
    the accumulator saturates at 65,504 -- measured max|err| = inf at
    n >= 100,000. Silent at small n, catastrophic at biobank scale.
    """

    def __init__(self, enable):
        self.enable = enable
        self._prev = None

    def __enter__(self):
        if self.enable and HAS_CUPY:
            try:
                h = cp.cuda.device.get_cublas_handle()
                self._prev = cp.cuda.cublas.getMathMode(h)
                cp.cuda.cublas.setMathMode(
                    h, cp.cuda.cublas.CUBLAS_TENSOR_OP_MATH)
            except Exception:                                  # noqa: BLE001
                self._prev = None
        return self

    def __exit__(self, *exc):
        if self._prev is not None:
            try:
                cp.cuda.cublas.setMathMode(
                    cp.cuda.device.get_cublas_handle(), self._prev)
            except Exception:                                  # noqa: BLE001
                pass


def _resolve_precision(precision, n_samples, verbose):
    """Decide whether to use TF32; returns True to enable.

    'auto' turns TF32 on wherever the hardware has it, because for this
    workload it is exact -- there is nothing to trade away. Hardware that
    cannot do it falls back to fp32 with a warning rather than failing.
    """
    if precision in ("fp16", "half", "bf16"):
        raise ValueError(
            "precision={!r} is not supported: cuBLAS accumulates half-precision"
            " matmuls in half precision, so the exact integer counts this"
            " module depends on break down above 2**11 and the accumulator"
            " saturates at 65,504 (measured max|err| = inf at n >= 100,000)."
            " Use 'tf32', which is exact here, or 'fp32'.".format(precision))
    if precision not in ("auto", "tf32", "fp32"):
        raise ValueError(
            "precision must be 'auto', 'tf32' or 'fp32'; got {!r}".format(
                precision))
    if precision == "fp32" or not HAS_CUPY:
        return False
    cc = cp.cuda.Device().compute_capability
    cc_num = int(cc) if isinstance(cc, str) else cc[0] * 10 + cc[1]
    if cc_num >= _TF32_MIN_CC:
        return True
    if precision == "tf32":
        import warnings
        warnings.warn(
            "precision='tf32' requested but this GPU is compute capability "
            "{} and TF32 needs {}.0 (Ampere) or newer; falling back to fp32. "
            "Results are identical either way -- only speed differs.".format(
                cc, _TF32_MIN_CC // 10), RuntimeWarning, stacklevel=3)
    elif verbose:
        print("cugen.ld: TF32 unavailable on cc {}, using fp32".format(cc))
    return False


# ---------------------------------------------------------------------------
# Numeric core -- fp64, vectorised, CuPy-free.
# This is the CPU twin of the kernel epilogue: same branches, same guards, so
# a disagreement here is a disagreement there.
# ---------------------------------------------------------------------------
def cubic_real_roots(a, b, c, d):
    """Real roots of a*x^3 + b*x^2 + c*x + d, vectorised.

    Returns (roots, n_real) with roots shaped (..., 3), unused slots NaN.
    Trigonometric form when three real roots exist, Cardano otherwise. This
    mirrors plink-ng's cubic_real_roots so the root SET matches plink2.
    """
    a, b, c, d = (np.asarray(v, dtype=np.float64) for v in (a, b, c, d))
    shape = np.broadcast_shapes(a.shape, b.shape, c.shape, d.shape)
    roots = np.full(shape + (3,), np.nan)
    n_real = np.zeros(shape, dtype=np.int64)

    ok = np.abs(a) > EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(ok, b / a, 0.0)
        q = np.where(ok, c / a, 0.0)
        r = np.where(ok, d / a, 0.0)
        # depressed cubic t^3 + A t + B, with x = t - p/3
        A = q - p * p / 3.0
        B = 2.0 * p ** 3 / 27.0 - p * q / 3.0 + r
        disc = (B / 2.0) ** 2 + (A / 3.0) ** 3

        three = ok & (disc <= 0.0) & (A < -EPS)
        one = ok & ~three

        if three.any():
            At = np.where(three, A, -1.0)
            Bt = np.where(three, B, 0.0)
            m = 2.0 * np.sqrt(-At / 3.0)
            theta = np.arccos(np.clip(3.0 * Bt / (At * m), -1.0, 1.0))
            for k in range(3):
                t = m * np.cos((theta - 2.0 * np.pi * k) / 3.0)
                roots[..., k] = np.where(three, t - p / 3.0, roots[..., k])
            n_real = np.where(three, 3, n_real)

        if one.any():
            Bo = np.where(one, B, 0.0)
            s = np.sqrt(np.where(one, np.maximum(disc, 0.0), 0.0))
            t = np.cbrt(-Bo / 2.0 + s) + np.cbrt(-Bo / 2.0 - s)
            roots[..., 0] = np.where(one, t - p / 3.0, roots[..., 0])
            n_real = np.where(one, 1, n_real)

    return roots, n_real


def _loglik(x, tab, pA, pB):
    """Multinomial log-likelihood of a genotype table given p_AB = x."""
    y, z, w = pA - x, pB - x, 1.0 - pA - pB + x
    bad = (x < -1e-9) | (y < -1e-9) | (z < -1e-9) | (w < -1e-9)
    xe, ye, ze, we = (np.maximum(v, EPS) for v in (x, y, z, w))
    ll = (2 * tab[..., 2, 2] * np.log(xe)
          + tab[..., 2, 1] * np.log(xe * ye)
          + tab[..., 1, 2] * np.log(xe * ze)
          + 2 * tab[..., 2, 0] * np.log(ye)
          + 2 * tab[..., 0, 2] * np.log(ze)
          + tab[..., 1, 1] * np.log(xe * we + ye * ze)
          + tab[..., 1, 0] * np.log(ye * we)
          + tab[..., 0, 1] * np.log(ze * we)
          + 2 * tab[..., 0, 0] * np.log(we))
    return np.where(bad, -np.inf, ll)


def ld_from_counts(counts, dprime_method: str = "phased"):
    """All five statistics from an (..., 3, 3) stack of genotype tables.

    counts[..., i, j] = number of PAIRWISE-CO-OBSERVED samples with ALT dosage
    i at locus A and j at locus B. Degenerate pairs yield NaN.
    """
    tab = np.asarray(counts, dtype=np.float64)
    n = tab.sum(axis=(-2, -1))
    rows, cols = tab.sum(axis=-1), tab.sum(axis=-2)
    dv = np.array([0.0, 1.0, 2.0])
    sA, sB = rows @ dv, cols @ dv
    qA, qB = rows @ (dv * dv), cols @ (dv * dv)
    # S = sum over co-observed of gA*gB, recoverable from the interior cells
    # alone: terms with dosage 0 on either side vanish.
    S = np.einsum("i,...ij,j->...", dv, tab, dv)

    with np.errstate(divide="ignore", invalid="ignore"):
        pA, pB = sA / (2.0 * n), sB / (2.0 * n)
        vA, vB = n * qA - sA * sA, n * qB - sB * sB
        good = (n > 0) & (vA > 0) & (vB > 0)
        # Hill & Robertson (1968): r as the correlation of allele counts.
        # sqrt(vA)*sqrt(vB), not sqrt(vA*vB) -- the product overflows the
        # useful range for no benefit.
        r = np.clip(np.where(good, (n * S - sA * sB) / (np.sqrt(vA) * np.sqrt(vB)),
                             np.nan), -1.0, 1.0)
        qAf, qBf = 1.0 - pA, 1.0 - pB

        # The cubic below already recovers D from estimated haplotype
        # frequencies. r2_phased_em is that D expressed as a correlation --
        # the same quantity plink2 --r2-phased reports on unphased input.
        if dprime_method == "composite":
            # Burrows' composite measure; see Weir (1979). This inverts the
            # haplotypic identity r = D/sqrt(pA qA pB qB) and is NOT a phase
            # estimate. Verified against simulation to recover gametic D under
            # HWE with no factor-of-2 correction.
            D = r * np.sqrt(np.maximum(pA * qAf * pB * qBf, 0.0))
        else:
            # Hill (1974) likelihood, exact roots per Gaunt et al. (2007).
            # E-step note: a COUPLING double heterozygote is AB/ab and yields
            # exactly ONE AB haplotype, not two. That factor is the classic
            # error here; getting it wrong makes EM DECREASE the likelihood.
            cc = 2.0 * tab[..., 2, 2] + tab[..., 2, 1] + tab[..., 1, 2]
            n11 = tab[..., 1, 1]
            a1, a0 = 1.0 - 2.0 * pA - 2.0 * pB, pA * pB
            roots, _ = cubic_real_roots(
                4.0 * n,
                2.0 * n * a1 - 2.0 * cc - n11,
                2.0 * n * a0 - cc * a1 - n11 * (1.0 - pA - pB),
                -cc * a0)
            lo = np.maximum(0.0, pA + pB - 1.0)
            hi = np.minimum(pA, pB)
            best = np.where(good, lo, np.nan)
            best_ll = np.full(n.shape, -np.inf)
            # boundaries are candidates too: the ML can sit on one
            cand = np.concatenate([roots, lo[..., None], hi[..., None]], axis=-1)
            for k in range(cand.shape[-1]):
                x = np.clip(cand[..., k], lo, hi)
                ll = _loglik(x, tab, pA, pB)
                take = good & np.isfinite(x) & (ll > best_ll)
                best = np.where(take, x, best)
                best_ll = np.where(take, ll, best_ll)
            mono = (pA <= EPS) | (pA >= 1 - EPS) | (pB <= EPS) | (pB >= 1 - EPS)
            D = np.where(mono, 0.0, best - pA * pB)

        D = np.where(good, D, np.nan)
        # Lewontin (1964): D normalised by its frequency-dependent maximum.
        # The branch on sign(D) is Lewontin's, not a numerical convenience.
        dmax = np.where(D >= 0, np.minimum(pA * qBf, qAf * pB),
                        np.minimum(pA * pB, qAf * qBf))
        DP = np.where(good, np.where(dmax > EPS,
                                     np.clip(D / dmax, -1.0, 1.0), 0.0), np.nan)

        # r2_phased_em: the SAME D the cubic just solved for, expressed as a
        # correlation. This is what plink2 --r2-phased reports on unphased
        # input -- an EM/likelihood ESTIMATE of phase, not observed phase. For
        # observed phase use a hap2bit file and r2_phased.
        den = pA * qAf * pB * qBf
        r2_em = np.where(good & (den > EPS),
                         np.clip(D * D / np.where(den > EPS, den, 1.0), 0.0, 1.0),
                         np.nan)

    return {"n": n, "pA": pA, "pB": pB, "r": r,
            "r2": np.clip(r * r, 0.0, 1.0),
            "r2_signed": np.clip(r * np.abs(r), -1.0, 1.0),
            "d": D, "dp": DP, "r2_phased_em": r2_em}


def contingency_tables(dosages, pairs):
    """(m, 3, 3) pairwise-complete tables for pairs of rows of `dosages`.

    dosages is (n_variants, n_samples) uint8 with 3 = missing. This is the
    NumPy twin of the GEMM count path; the kernel computes the same integers
    via indicator-plane products.
    """
    g = np.asarray(dosages)
    i = np.asarray(pairs, dtype=np.int64)[:, 0]
    j = np.asarray(pairs, dtype=np.int64)[:, 1]
    out = np.zeros((len(i), 3, 3), dtype=np.int64)
    for a in range(3):
        ia = (g[i] == a)
        for b in range(3):
            # int64 accumulation is mandatory: numpy sums in the INPUT dtype,
            # so counting in uint8 silently overflows past 255 samples.
            out[:, a, b] = np.sum(ia & (g[j] == b), axis=1, dtype=np.int64)
    return out


# ---------------------------------------------------------------------------
# CuPy-free helpers. Everything that is not a kernel lives here so the whole
# decision surface is testable on CPU-only CI.
# ---------------------------------------------------------------------------
def _resolve_gidx(obj) -> Optional[np.ndarray]:
    """Polymorphic gidx resolution, mirroring score._resolve_weights."""
    if obj is None:
        return None
    if isinstance(obj, (str, Path)):
        p = str(obj)
        if p.endswith(".feather"):
            df = pd.read_feather(p)
        elif p.endswith(".npz"):
            return np.asarray(np.load(p)["gidx"], dtype=np.int64)
        else:
            df = pd.read_csv(p, sep="\t" if p.endswith((".tsv", ".tsv.gz")) else ",")
        obj = df
    if isinstance(obj, pd.DataFrame):
        if "gidx" not in obj.columns:
            raise ValueError("variant table needs a 'gidx' column; "
                             f"got {list(obj.columns)}")
        return np.asarray(obj["gidx"], dtype=np.int64)
    return np.asarray(obj, dtype=np.int64)


def _parse_region(region: str) -> Tuple[str, Optional[int], Optional[int]]:
    """'22:1000000-2000000' or '22'. 1-based inclusive, PLINK/samtools style."""
    m = re.fullmatch(r"([^:]+)(?::([\d,]+)-([\d,]+))?", str(region).strip())
    if not m:
        raise ValueError(f"malformed region {region!r}; expected 'CHR' or "
                         f"'CHR:START-END'")
    chrom, s, e = m.group(1), m.group(2), m.group(3)
    if s is None:
        return chrom, None, None
    start, end = int(s.replace(",", "")), int(e.replace(",", ""))
    if end < start:
        raise ValueError(f"region {region!r} has end < start")
    return chrom, start, end


def _pair_bounds(n_rows: int, positions: Optional[np.ndarray],
                 window: Optional[int], window_kb: Optional[float]):
    """Per-row exclusive upper column bound for the upper-triangle scan.

    Both window predicates AND together, matching PLINK. Banding by position
    uses searchsorted so the column extent is O(band), not O(p). Returns
    (starts, hi) so the pair COUNT can be taken without materialising pairs --
    at p = 1.1M the materialised list would be ~6e11 entries.
    """
    if window_kb is not None and positions is None:
        raise ValueError("window_kb requires variant positions; pass annotation=")
    if positions is not None and len(positions):
        if np.any(np.diff(positions) < 0):
            raise ValueError("positions are not non-decreasing along file rows; "
                             "banding would be silently wrong. Sort the .cugen "
                             "or drop window_kb.")
    lo = np.arange(n_rows, dtype=np.int64)
    hi = np.full(n_rows, n_rows, dtype=np.int64)
    if window is not None:
        hi = np.minimum(hi, lo + int(window) + 1)
    if window_kb is not None:
        span = int(round(float(window_kb) * 1000))
        hi = np.minimum(hi, np.searchsorted(positions, positions + span,
                                            side="right").astype(np.int64))
    return lo + 1, hi


def _count_pairs(n_rows: int, positions, window, window_kb) -> int:
    starts, hi = _pair_bounds(n_rows, positions, window, window_kb)
    return int(np.maximum(hi - starts, 0).sum())


def _plan_pairs(n_rows: int, positions: Optional[np.ndarray],
                window: Optional[int], window_kb: Optional[float]):
    """Materialised upper-triangle pair list, for the NumPy reference path."""
    starts, hi = _pair_bounds(n_rows, positions, window, window_kb)
    lo = np.arange(n_rows, dtype=np.int64)
    counts = np.maximum(hi - starts, 0)
    total = int(counts.sum())
    if total == 0:
        return np.zeros((0, 2), dtype=np.int64), 0
    i = np.repeat(lo, counts)
    offs = np.arange(total, dtype=np.int64) - np.repeat(
        np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    j = np.repeat(starts, counts) + offs
    return np.stack([i, j], axis=1), total


# chi2 above which the closed-form tail expansion replaces erfc. Below it erfc
# is exact; above chi2 ~ 1450 erfc underflows to 0 in float64 and -log10 of that
# is +inf, which is the bug this constant exists to avoid. 400 is chosen well
# clear of that ceiling (erfc(sqrt(200)) ~ 1e-88) and high enough that the
# truncated expansion is already accurate to 1e-7 where it takes over -- at a
# cut of 30 the expansion is only good to 2e-4.
_NLP_ASYMPTOTIC_FROM = 400.0
_LN10 = math.log(10.0)
_HALF_LN_PI = 0.5 * math.log(math.pi)


def _erfc_for(xp):
    """erfc for either numpy or cupy arrays. numpy has no erfc of its own."""
    if xp is np:
        from scipy.special import erfc
        return erfc
    from cupyx.scipy.special import erfc
    return erfc


def _neglog10_chi2_1df(chi2, xp=np):
    """``-log10(P(X > chi2))`` for X ~ chi-square with 1 df.

    Returns -log10(p) rather than p because p is not representable. The
    right-tail p-value underflows float64 at chi2 ~ 1450 and float32 at
    chi2 ~ 170 -- and chi2 = N_OBS * r^2, so at 1000 Genomes size
    (N_hap = 5008) float32 dies at r^2 = 0.034 and float64 at r^2 = 0.29. A
    ``P`` column would therefore read as a flat zero for essentially every
    linked pair genome-wide. -log10(p) tops out around 8.7e5 and fits float32
    comfortably.

    Two branches, both evaluated on clamped inputs so neither can produce an
    inf that ``where`` would then have to discard:

      chi2 <= 400:  -log10(erfc(sqrt(chi2/2))), exact.
      chi2 >  400:  the asymptotic expansion of erfc. With z = chi2/2,
                   erfc(sqrt(z)) ~ exp(-z)/sqrt(pi*z) * (1 - 1/(2z) + 3/(4z^2)),
                   so -ln p ~ z + ln(sqrt(pi*z)) - log1p(-1/(2z) + 3/(4z^2)).

    Measured against log(2) + scipy.special.log_ndtr(-sqrt(chi2)), which is
    genuinely log-space: max error 1e-7 in -log10(p) units over chi2 in
    [0.5, 1e7]. Note that scipy.stats.chi2.logsf is NOT a usable oracle here --
    it computes log(sf()) and so returns inf above chi2 ~ 1450, the very regime
    this helper exists to serve. Do not use ``1.0 - erf(...)`` instead -- it
    cancels catastrophically in the tail (see cugen.qc._chi2_p_1df, and the
    same failure documented for the inverse direction in cugen.assoc).
    """
    z = xp.maximum(xp.asarray(chi2, dtype=xp.float64) * 0.5, 0.0)
    cut = _NLP_ASYMPTOTIC_FROM * 0.5
    erfc = _erfc_for(xp)
    small = -xp.log10(erfc(xp.sqrt(xp.minimum(z, cut))))
    zl = xp.maximum(z, cut)
    large = (zl + 0.5 * xp.log(zl) + _HALF_LN_PI
             - xp.log1p(-1.0 / (2.0 * zl) + 3.0 / (4.0 * zl * zl))) / _LN10
    return xp.where(z <= cut, small, large)


def _add_significance(res, stats, xp=np, exact="never"):
    """Attach ``chi2`` and ``p`` to a result dict, in place.

    The test of no disequilibrium between two biallelic loci is

        chi2 = N * r^2,   1 df

    (Park 2019 eq. 1, which writes it as 2n D^2 / (pA qA pB qB); Weir, Genetic
    Data Analysis II.) N is the number of sampled GAMETES for phased data and
    the number of INDIVIDUALS for the composite (unphased) statistic -- the
    factor of two between gametic and composite LD. ``res["n"]`` already
    carries the right one on every path: 2*n_samples for hap2bit input,
    n_samples (or the per-pair co-observed count) for 2bit.

    r^2 is taken from whichever correlation this path computed, matching the
    resolution the caller does for filtering: dosage ``r2`` if present, else
    haplotype ``r2_phased``. A call that asks for r2_phased_em gets the dosage
    r2 and the individual count, which are consistent with each other; the EM
    haplotype estimate is a different estimator and is not tested here.
    """
    if not (_SIG_STATS & set(stats)):
        return res
    for key in ("r2", "r2_phased"):
        if key in res:
            r2 = res[key]
            break
    else:
        r_key = "r" if "r" in res else "r_phased"
        r2 = res[r_key] ** 2
    chi2 = xp.asarray(res["n"], dtype=xp.float64) * r2
    res["chi2"] = chi2
    res["p"] = _neglog10_chi2_1df(chi2, xp)
    if "p_exact" in stats:
        res["p_exact"] = _exact_column(res["nAB"], res["nA"], res["nB"],
                                       res["n"], exact)
    return res


def _exact_column(nAB, nA, nB, n, exact):
    """Fisher exact -log10(p) per pair, NaN where the mode says not to bother.

    NaN rather than falling back to the asymptotic value: the two columns answer
    different questions, and silently mixing them would make NEG_LOG10_P_EXACT
    mean "exact where we felt like it". A NaN says plainly that the asymptotic
    p-value in NEG_LOG10_P is the one to use for that pair.

    Loops over the pairs that need it. Under 'auto' that set is small by
    construction -- see _exact_needed -- and this is the reference path, whose
    contract is correct rather than fast.
    """
    nAB = np.asarray(nAB, dtype=np.int64)
    nA = np.asarray(nA, dtype=np.int64)
    nB = np.asarray(nB, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    out = np.full(nAB.shape, np.nan, dtype=np.float64)
    need = (np.ones(nAB.shape, dtype=bool) if exact == "always"
            else _exact_needed(nA, nB, n))
    for t in np.flatnonzero(need):
        out[t] = _fisher_neglog10p_2x2(nAB[t], nA[t], nB[t], n[t])
    return out


def _fisher_neglog10p_2x2(nAB, nA, nB, N):
    """Two-sided Fisher exact test on the 2x2 haplotype table, as -log10(p).

    This is the EXACT PERMUTATION p-value, not an approximation to it.
    Permuting haplotype labels leaves both variants' allele counts untouched,
    so the permutation null of the table is the hypergeometric distribution
    with those margins fixed, and summing its mass over the tables at least as
    extreme as the observed one is precisely Fisher's exact test. Koch et al.
    (2013) spent ~34,000 CPU-hours Monte-Carlo sampling this distribution.

    Two-sided in the conventional sense: total probability of every table whose
    own probability does not exceed the observed table's. Computed through
    lgamma so the factorials never overflow.

    The cost is one term per admissible table, i.e. min(nA, nB) + 1 terms, and
    the gate in _exact_needed keeps it small where it matters: the exact test is
    only needed when nA*nB/N < 5, and min(nA,nB)^2 <= nA*nB < 5N bounds the
    loop by sqrt(5N) -- about 158 terms at 1000 Genomes size, fewer for rarer
    variants. The pairs that need this are the pairs where it is cheap.
    """
    from scipy.special import gammaln  # noqa: PLC0415
    nAB, nA, nB, N = int(nAB), int(nA), int(nB), int(N)
    lo, hi = max(0, nA + nB - N), min(nA, nB)
    if hi < lo:
        return 0.0
    k = np.arange(lo, hi + 1, dtype=np.float64)
    # log hypergeometric pmf, dropping the k-independent normaliser and
    # restoring it by summing to one.
    logp = (-gammaln(k + 1.0) - gammaln(nA - k + 1.0)
            - gammaln(nB - k + 1.0) - gammaln(N - nA - nB + k + 1.0))
    logp -= logp.max()
    pmf = np.exp(logp)
    pmf /= pmf.sum()
    obs = pmf[nAB - lo]
    # 1 + 1e-7 absorbs the float noise between two mathematically equal
    # tables, which is common with symmetric margins; without it the mirror
    # table is dropped and the p-value comes out roughly half.
    tail = float(pmf[pmf <= obs * (1.0 + 1e-7)].sum())
    tail = min(max(tail, 0.0), 1.0)
    if tail <= 0.0:
        return np.inf
    return -math.log10(tail)


def _exact_needed(nA, nB, N, xp=np):
    """Where the chi-square approximation should not be trusted.

    The minimum expected cell count of the 2x2 table under independence is
    min(nA, N-nA) * min(nB, N-nB) / N. Below 5 -- the classic rule -- the
    asymptotic test is anti-conservative; Park (2019) fig. 1 shows exactly this
    breakdown, and at N=14 with symmetric margins the asymptotic p can be 3x
    too small.
    """
    a = xp.minimum(nA, N - nA)
    b = xp.minimum(nB, N - nB)
    return (a * b / xp.asarray(N, dtype=xp.float64)) < 5.0


def _recover_nab(r, nA, nB, N):
    """Recover the AB haplotype count from r and the two allele counts.

    r = (N*nAB - nA*nB) / sqrt(nA(N-nA) * nB(N-nB)), so nAB follows by
    rearrangement. It exists because the fused GPU path emits r and discards
    the cross-product that produced it -- reconstructing is cheaper than a
    second kernel, and means the exact test has ONE implementation shared by
    the host and device paths.

    Exact in practice despite r being float32: nAB is an integer, and the
    reconstruction error is bounded by sqrt(nA(N-nA)nB(N-nB))/N * eps32 <=
    N/4 * 1.2e-7, which is 1.5e-4 at N=5008 -- three orders below the 0.5
    needed for rounding to land on the right integer. The float64 upcast below
    is insurance for large N rather than a necessity at that scale; the tests
    pass without it today. `rint` is not optional -- truncating instead
    misplaces roughly half the counts.
    """
    r = np.asarray(r, dtype=np.float64)
    nA = np.asarray(nA, dtype=np.float64)
    nB = np.asarray(nB, dtype=np.float64)
    den = np.sqrt(nA * (N - nA) * nB * (N - nB))
    return np.rint((r * den + nA * nB) / float(N)).astype(np.int64)


def _lambda_gc(chi2, separation=None, min_null=100):
    """Genomic-control inflation factor for the LD test statistic.

    lambda = median(chi2) / median(chi2_1df), the genomic-control ratio (Devlin
    & Roeder 1999) applied to an LD test rather than an association test. Above
    1 means the per-pair test is anti-conservative, which is what population
    structure and cryptic relatedness do to LD: they correlate variants that are
    in no gametic disequilibrium at all. Park (2019) fig. 1C/D models the
    effect and Koch (2013) flags it; neither offers a correction, and I could
    not find lambda applied to LD statistics anywhere in the literature.

    Assumes most of the tests it is given are null. For an unwindowed scan that
    is reasonable -- almost every pair on a chromosome is far apart -- but false
    for a tight window, where every pair is expected to be linked. So the median
    is taken over the more DISTANT half of the pairs by index separation
    whenever that leaves at least ``min_null`` of them, keeping the close,
    genuinely-linked pairs out of it.

    Separation in variant index is a deliberately crude proxy for distance. The
    point is to prefer far pairs to near ones, not to model recombination.

    MUST be called before any significance filtering. Filtering selects the
    tail, and the median of a selected tail says nothing about the null.
    """
    chi2 = np.asarray(chi2, dtype=np.float64)
    ok = np.isfinite(chi2)
    if separation is not None and len(separation) == chi2.size:
        sep = np.asarray(separation)[ok]
    else:
        sep = None
    chi2 = chi2[ok]
    if chi2.size == 0:
        return float("nan")
    if sep is not None:
        far = sep >= np.median(sep)
        if far.sum() >= min_null:
            chi2 = chi2[far]
    return float(np.median(chi2) / _CHI2_1DF_MEDIAN)


def _psd_pinv_and_sqrt(V, tol=_PSD_EIGEN_TOL):
    """Moore-Penrose inverse of a PSD matrix, and its symmetric square root.

    Kinship matrices are routinely singular -- and estimators of them routinely
    return something not quite PSD -- so V has no Cholesky factor and V^-1 does
    not exist. Mangin et al. prescribe the Moore-Penrose inverse V^-, "which is
    always defined", and build a PSD matrix by zeroing negative eigenvalues of a
    decomposition. Both come out of one eigendecomposition here.

    Returns ``(V_inv, W)`` with ``W.T @ W == V_inv``, so a quadratic form
    ``x' V^- y`` becomes an ordinary dot product ``(Wx) . (Wy)``.
    """
    w, U = np.linalg.eigh(np.asarray(V, dtype=np.float64))
    dead = w < tol
    inv = np.where(dead, 0.0, 1.0 / np.where(dead, 1.0, w))
    V_inv = (U * inv) @ U.T
    W = (U * np.sqrt(inv)) @ U.T
    return V_inv, W


def _corrected_transform(which, n, kinship=None, structure=None):
    """The one n x n map that turns a corrected r^2 into an ordinary one.

    Every measure in Mangin et al. (2012) is a ratio of entries of a
    covariance-like matrix, and in each case that matrix is the GRAM matrix of a
    linearly transformed genotype vector. So a single P, built once per dataset,
    reduces all three to an UNCENTERED r^2 on ``P x``:

      r2_s   (eq. 1)  P = (I - H_S)(I - 11'/n)
                      centre, then residualise on the structure matrix. The
                      published form is a Schur complement of the joint
                      covariance of the two loci and S, which is the residual
                      covariance after regressing the loci on S -- and since
                      (I - H_S) is idempotent that equals the Gram of the
                      residuals.
      r2_v   (eq. 2)  P = W (I - F),  F = 1 1' V^- / (1' V^- 1)
                      GLS-centre, then whiten. F is the projection onto the GLS
                      mean; W'W = V^-, so x'V^-y becomes (Wx).(Wy).
      r2_vs  (eq. 3)  P = (I - H_Z) W (I - F),  Z_S = W (I - F) S
                      the same Schur complement, taken in the V^- metric.

    H_S and H_Z are hat matrices built with the same pseudo-inverse, so a
    rank-deficient or collinear structure matrix degrades gracefully instead of
    raising.
    """
    eye = np.eye(n)
    if which == "r2_s":
        S = _as_covariate_block(structure, n)
        Sc = S - S.mean(axis=0)
        H = Sc @ _psd_pinv_and_sqrt(Sc.T @ Sc)[0] @ Sc.T
        return (eye - H) @ (eye - np.full((n, n), 1.0 / n))

    V_inv, W = _psd_pinv_and_sqrt(_as_square(kinship, n))
    one = np.ones(n)
    denom = float(one @ V_inv @ one)
    if not np.isfinite(denom) or abs(denom) < 1e-300:
        raise ValueError(
            "1' V^- 1 vanished, so the GLS mean is undefined -- the kinship "
            "matrix has no usable non-null space. Check it is the right matrix "
            "and is positive semi-definite.")
    P = W @ (eye - np.outer(one, one @ V_inv) / denom)
    if which == "r2_v":
        return P
    Zs = P @ _as_covariate_block(structure, n)
    Hz = Zs @ _psd_pinv_and_sqrt(Zs.T @ Zs)[0] @ Zs.T
    return (eye - Hz) @ P


def _as_square(M, n):
    if M is None:
        raise ValueError(
            "r2_v/r2_vs need a kinship matrix; pass kinship= an (n_samples, "
            "n_samples) array of genetic covariances between individuals.")
    A = np.asarray(M, dtype=np.float64)
    if A.shape != (n, n):
        raise ValueError(
            f"kinship shape {A.shape} does not match n_samples={n}; it must be "
            f"({n}, {n}) and in the same sample order as the .cugen file.")
    return A


def _as_covariate_block(M, n):
    if M is None:
        raise ValueError(
            "r2_s/r2_vs need a structure matrix; pass structure= an "
            "(n_samples, K) array -- e.g. K-1 admixture proportions from "
            "STRUCTURE, or leading principal components.")
    A = np.atleast_2d(np.asarray(M, dtype=np.float64))
    if A.shape[0] != n and A.shape[1] == n:
        A = A.T
    if A.shape[0] != n:
        raise ValueError(
            f"structure has {A.shape[0]} rows but n_samples={n}; it must be "
            f"(n_samples, K) and in the same sample order as the .cugen file.")
    return A


def _corrected_r2(dosages, pairs, P):
    """Uncentered r^2 of the transformed variant vectors, for every pair.

    ``dosages`` is (n_variants, n_samples); P is applied on the sample axis, so
    row v becomes ``P x_v``. The Gram matrix is (p, p) -- the same O(p^2) the
    rest of this module lives with -- and every pair reads two diagonal entries
    and one off-diagonal one.
    """
    X = np.asarray(dosages, dtype=np.float64)
    Z = X @ P.T
    G = Z @ Z.T
    d = np.diag(G).copy()
    a, b = pairs[:, 0], pairs[:, 1]
    # LDcorSV returns 0 when a transformed variance underflows; mirror that
    # rather than emitting inf or nan for a variant the transform annihilated.
    floor = 1e-12 * max(float(d.max()), 1.0)
    ok = (d[a] > floor) & (d[b] > floor)
    den = np.where(ok, d[a] * d[b], 1.0)
    return np.where(ok, np.clip(G[a, b] ** 2 / den, 0.0, 1.0), 0.0)


def _bh_threshold_neglog10p(neglog10p, m, alpha):
    """Benjamini-Hochberg, computed entirely in -log10(p) space.

    BH rejects the k smallest p-values for the largest k satisfying
    p_(k) <= k*alpha/m. Taking -log10 of both sides flips the inequality:

        neglog10p_(k) >= log10(m / (k*alpha))

    where neglog10p_(k) is the k-th LARGEST. Working in log space is not a
    convenience here -- at m = 1e14 the Bonferroni-scale thresholds are around
    1e-16 and the observed p-values are unrepresentable, so a direct
    implementation would compare two zeros.

    Returns ``(cut, k)``: the -log10(p) threshold to keep at, and how many
    tests are rejected. ``(inf, 0)`` when nothing is significant.

    The sort makes this O(K log K) in the number of SURVIVORS, not in m -- the
    screen has already discarded everything with a larger p-value, and those
    can never be among the k smallest.
    """
    order = np.sort(np.asarray(neglog10p, dtype=np.float64))[::-1]
    if order.size == 0:
        return np.inf, 0
    k = np.arange(1, order.size + 1, dtype=np.float64)
    ok = order >= np.log10(m / (k * alpha))
    if not ok.any():
        return np.inf, 0
    kmax = int(np.flatnonzero(ok)[-1]) + 1
    return float(order[kmax - 1]), kmax


def _apply_significance_filters(df, *, max_p, correction, alpha, m, screened,
                                verbose):
    """Post-scan p-value filtering. Accepts a pandas or a cudf frame.

    Runs after the scan for two reasons. Under missingness N_OBS varies per
    pair, so the r^2 pre-filter derived from max_p is only a conservative bound
    and the exact cut has to be applied here. And BH-FDR is defined over the
    whole set of p-values, so its threshold is not knowable before the scan.
    """
    if max_p is None and correction is None:
        return df
    col = df["NEG_LOG10_P"]
    vals = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
    vals = np.asarray(vals, dtype=np.float64)

    if correction == "fdr":
        cut, k = _bh_threshold_neglog10p(vals, m, alpha)
        if verbose:
            print(f"cugen.ld: BH-FDR at alpha={alpha} over m={m:,} tests -> "
                  f"{k:,} rejected, -log10(p) cut {cut:.4f}")
        if screened and k == vals.size and k > 0:
            raise ValueError(
                f"BH-FDR rejected every one of the {k:,} pairs that survived "
                f"the screen, so the true threshold may lie below it and real "
                f"discoveries were discarded before FDR ran. Loosen or drop "
                f"the min_r2/max_p screen and re-run.")
    else:
        # max_p, exact. -log10 so the comparison never meets an underflowed p.
        cut = -math.log10(max_p)
    keep = vals >= cut - 1e-12
    if keep.all():
        return df
    if hasattr(df, "iloc"):
        return df.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    return df[keep]


def _empty_pairs(stats: Sequence[str]) -> pd.DataFrame:
    """Zero rows, full schema, correct dtypes. Callers break on bare
    pd.DataFrame() -- this is a real bug class, not a nicety."""
    cols = {"CHR_A": np.int32, "POS_A": np.int64, "ID_A": object, "MAF_A": np.float32,
            "CHR_B": np.int32, "POS_B": np.int64, "ID_B": object, "MAF_B": np.float32,
            "N_OBS": np.int32}
    for s in _STATS:
        if s in stats:
            cols[_STAT_COL[s]] = np.float32
    cols["gidx_a"] = np.int64
    cols["gidx_b"] = np.int64
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()})


class _ChunkWriter:
    """Appends successive frames to one output file.

    Streaming exists because the result does not fit in device memory; it
    must not then be accumulated in host memory either, so each chunk is
    written and dropped. Parquet gets one row group per flush via
    ParquetWriter, which keeps the file open and the schema fixed.
    """

    def __init__(self, path: str):
        self.path = str(path)
        self.rows = 0
        self._pq = None

    def write(self, df) -> None:
        if len(df) == 0:
            return
        if self.path.endswith(".parquet"):
            import pyarrow as pa
            import pyarrow.parquet as pq
            tbl = (df.to_arrow() if hasattr(df, "to_arrow")
                   else pa.Table.from_pandas(df, preserve_index=False))
            if self._pq is None:
                self._pq = pq.ParquetWriter(self.path, tbl.schema)
            self._pq.write_table(tbl)
        else:
            sep = "\t" if self.path.endswith((".tsv", ".tsv.gz")) else ","
            gz = self.path.endswith(".gz")
            if hasattr(df, "to_pandas") and not gz:
                # Keep libcudf's writer in the loop. Going through
                # to_pandas().to_csv() per chunk cost 62.9 s to serialise
                # 10,517,635 rows against 2.4 s for the whole buffered run --
                # 26x, which would have made every streamed timing a
                # measurement of pandas rather than of cugen. libcudf cannot
                # append, so it writes a part file and the bytes are
                # concatenated.
                part = f"{self.path}.part"
                df.to_csv(part, index=False, sep=sep, header=(self.rows == 0))
                with open(part, "rb") as src, \
                        open(self.path, "wb" if self.rows == 0 else "ab") as dst:
                    shutil.copyfileobj(src, dst, 1 << 22)
                os.remove(part)
            else:
                pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
                pdf.to_csv(self.path, sep=sep, index=False,
                           header=(self.rows == 0),
                           mode="w" if self.rows == 0 else "a",
                           compression="gzip" if gz else None)
        self.rows += len(df)

    def close(self) -> None:
        if self._pq is not None:
            self._pq.close()
            self._pq = None


def _write_df(df: pd.DataFrame, path: str) -> None:
    """Same output conventions as qc._write_df, but via pyarrow when possible.

    pandas.to_csv is the single largest cost in a large LD run -- 7.3 s to
    write 1.4M rows, against 0.22 s for the entire GPU scan that produced
    them. pyarrow's writer is C-speed and pyarrow is already a hard
    dependency, so there is no reason to pay for Python-side formatting.
    Falls back to pandas for gzip and for anything pyarrow declines.
    """
    if HAS_CUDF and isinstance(df, cudf.DataFrame):
        # already on the device -- write straight from there
        if path.endswith(".parquet"):
            df.to_parquet(path)
        else:
            df.to_csv(path, index=False,
                      sep="\t" if path.endswith((".tsv", ".tsv.gz")) else ",")
        return
    if path.endswith(".feather"):
        df.to_feather(path)
        return
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
        return
    sep = "\t" if path.endswith((".tsv", ".tsv.gz")) else ","
    if not path.endswith(".gz"):
        try:
            import pyarrow as pa
            from pyarrow import csv as pacsv
            pacsv.write_csv(
                pa.Table.from_pandas(df, preserve_index=False), path,
                pacsv.WriteOptions(include_header=True,
                                   delimiter=sep, quoting_style="none"))
            return
        except Exception:                                      # noqa: BLE001
            pass                                               # fall through
    df.to_csv(path, sep=sep, index=False,
              compression="gzip" if path.endswith(".gz") else None)


def _merge_annotation(df: pd.DataFrame, ann: Optional[pd.DataFrame],
                      chrom_fallback: int) -> pd.DataFrame:
    """Left-join CHR/POS/ID for both sides on gidx, gwas-style placeholders."""
    for side in ("a", "b"):
        S = side.upper()
        if ann is None:
            df[f"CHR_{S}"] = np.int32(chrom_fallback)
            df[f"POS_{S}"] = np.int64(0)
            df[f"ID_{S}"] = "."
        else:
            sub = ann[["gidx", "CHR", "POS", "ID"]].rename(
                columns={"CHR": f"CHR_{S}", "POS": f"POS_{S}", "ID": f"ID_{S}",
                         "gidx": f"gidx_{side}"})
            df = df.merge(sub, on=f"gidx_{side}", how="left")
            df[f"ID_{S}"] = df[f"ID_{S}"].fillna(".")
    return df


def _finalize(df: pd.DataFrame, stats: Sequence[str]) -> pd.DataFrame:
    """Coerce to the canonical schema, skipping columns already correct.

    An unconditional astype copies every column; at 5M rows that is real
    time for nothing when most columns were built at the right dtype.
    """
    tmpl = _empty_pairs(stats)
    for c, dt in tmpl.dtypes.items():
        if c in df.columns and df[c].dtype != dt:
            try:
                df[c] = df[c].astype(dt, copy=False)
            except (ValueError, TypeError):
                pass
    return df[list(tmpl.columns)]


# ---------------------------------------------------------------------------
# Clumping. The expensive half is embarrassingly parallel and the greedy half
# is nearly free, so they are separated rather than interleaved -- see the
# ld_clump docstring for why that is the whole design.
# ---------------------------------------------------------------------------
_CLUMP_BINS = (0.0001, 0.001, 0.01, 0.05)     # plink2 --clump-bins default

# Candidate fraction above which the BANDED scan is used instead of the
# rectangular one. Module-level so tests can force either shape: the committed
# clump fixture is 74-100% candidates, so left to itself it would exercise the
# banded path only and the rectangular kernel would ship untested.
_CLUMP_DENSE_FRAC = 0.5


def _prune_ids(annotation, gidx) -> np.ndarray:
    """IDs for the kept/pruned lists, or '.' when no annotation was given.

    Pruning needs no annotation at all when the window is in variant counts,
    so the ID column is a convenience rather than a requirement.
    """
    if annotation is None:
        return np.full(len(gidx), ".", dtype=object)
    ann = (annotation if isinstance(annotation, pd.DataFrame)
           else pd.read_csv(str(annotation), sep=None, engine="python"))
    ann = ann.rename(columns={c: str(c).lstrip("#") for c in ann.columns})
    if "ID" not in ann.columns or "gidx" not in ann.columns:
        return np.full(len(gidx), ".", dtype=object)
    lut = pd.Series(ann["ID"].astype(str).to_numpy(),
                    index=np.asarray(ann["gidx"]))
    return lut.reindex(gidx).fillna(".").to_numpy()


def ld_prune(
    cugen: Union[str, Path],
    *,
    window: Optional[int] = None,
    window_kb: Optional[float] = None,
    r2: float = 0.5,
    variants=None,
    maf_min: float = 0.0,
    annotation=None,
    output: Optional[Union[str, Path]] = None,
    backend: str = "auto",
    precision: str = "auto",
    tile_size: Optional[int] = None,
    max_pairs: int = 100_000_000,
    device: int = 0,
    verbose: bool = True,
):
    """Prune to variants in approximate linkage equilibrium (alias ``cg.prune``).

    Same job as plink2 ``--indep-pairwise``. Returns ``(keep, drop)``, two
    frames of ``gidx``/``ID`` mirroring ``.prune.in`` / ``.prune.out``.

    NOT byte-identical to plink, deliberately. This is the one place in
    cugen.ld that departs from it, so the reasoning is spelled out.

    **What both guarantee:** no two retained variants exceed the r^2
    threshold. That is the property pruning exists to provide, and both
    deliver it.

    **Where they differ:** plink's scan is sequential and never reconsiders. A
    variant dropped for conflicting with X stays dropped even when X is itself
    dropped later, so plink's output is a valid but NOT MAXIMAL independent
    set -- measured on a 400-variant fixture, 17 of the variants it discarded
    at r2=0.5 could be added straight back without breaking the guarantee.
    This computes a maximal independent set instead, retaining **7.5% more
    variants at r2=0.5 and 9.2% more at r2=0.2** under the identical
    constraint (max r2 among retained: 0.4996 in both).

    Keeping more variants at the same LD ceiling is strictly better for what
    pruning feeds -- PCA, relatedness and GRM construction all lose
    information when variants are discarded unnecessarily. It does mean a
    pruned set from here will not match plink variant-for-variant, which
    matters when reproducing someone else's pipeline. Exact emulation is
    implementable (plink's sequential scan is O(edges) on the host) and is an
    open question for the maintainer rather than something to guess at.

    Pruning and clumping are the SAME algorithm with different priorities --
    both are greedy maximal-independent-set selection under an r^2 constraint.
    Clumping ranks by p-value and answers "which variant leads this locus";
    pruning uses no association statistics at all, ranks by allele frequency,
    and answers "give me a non-redundant variant set". So this reuses
    :func:`clump_core` unchanged; only the ranking differs.

    **Higher MAF wins**, measured rather than assumed: on a fixture where file
    order and MAF order disagree, plink2 v2.0.0-a.7.1 kept the higher-MAF
    variant under BOTH orderings, so the rule is frequency and not position.
    Ties break on gidx, so two runs on one dataset cannot disagree.

    Unlike ``--clump`` there is no phase caveat: ``--indep-pairwise`` is
    defined on unphased hardcall r^2, which is exactly what .cugen stores and
    what :func:`ld_matrix` already matches plink2 on to 5.3e-07.

    Parameters
    ----------
    window
        Window in VARIANT COUNT (plink's first argument).
    window_kb
        Window in kb instead (plink's ``'kb'`` modifier); needs ``annotation``.
        Give one of ``window`` / ``window_kb``.
    r2
        Prune a variant whose r^2 with a retained variant exceeds this.
    maf_min
        Drop variants below this MAF first, from the header (no decode).

    Notes
    -----
    plink's step size is fixed at 1 here. A larger step makes the result depend
    on where window boundaries happen to land; step 1 is the only
    boundary-independent setting, and plink itself requires it whenever the
    window is expressed in kb.

    References
    ----------
    Purcell et al. (2007) AJHG 81:559-575     PLINK, --indep-pairwise
    """
    if window is None and window_kb is None:
        raise ValueError(
            "give window= (variant count) or window_kb=; plink's "
            "--indep-pairwise always takes a window, and so does this.")
    if not 0.0 <= r2 <= 1.0:
        raise ValueError(f"r2 must be in [0, 1], got {r2}")

    pairs = ld_matrix(
        cugen, variants=variants, annotation=annotation, maf_min=maf_min,
        window=window, window_kb=window_kb, min_r2=r2, stats=("r2",),
        output_format="pairs", backend=backend, precision=precision,
        tile_size=tile_size, max_pairs=max_pairs, device=device, verbose=False)
    pairs = pairs.to_pandas() if hasattr(pairs, "to_pandas") else pairs

    reader = read_cugen(str(cugen))
    gidx_all = np.asarray(reader.gidx)
    maf_all = np.asarray(reader.maf, dtype=np.float32)
    rows = np.arange(len(gidx_all))
    if variants is not None:
        rows = rows[np.isin(gidx_all, _resolve_gidx(variants))]
    if maf_min > 0.0:
        rows = rows[maf_all[rows] >= maf_min]

    g, maf = gidx_all[rows], maf_all[rows]
    n = len(g)
    # Priority: MAF DESCENDING (measured above), gidx ascending to break ties.
    order = np.lexsort((g, -maf.astype(np.float64)))
    rank = np.empty(n, dtype=np.int32)
    rank[order] = np.arange(n, dtype=np.int32)

    row_of = pd.Series(np.arange(n), index=g)
    if len(pairs):
        eu = row_of.reindex(pairs["gidx_a"].to_numpy()).to_numpy()
        ev = row_of.reindex(pairs["gidx_b"].to_numpy()).to_numpy()
        ok = np.isfinite(eu) & np.isfinite(ev)
        eu, ev = eu[ok].astype(np.int32), ev[ok].astype(np.int32)
    else:
        eu = ev = np.empty(0, dtype=np.int32)

    # Every variant is a candidate: there is no significance threshold to
    # clear, which is the entire difference from clumping.
    is_keep, _owner, rounds = clump_core(eu, ev, rank, np.ones(n, dtype=bool),
                                         allow_overlap=False, xp=np)
    ids = _prune_ids(annotation, g)
    keep = pd.DataFrame({"gidx": g[is_keep], "ID": ids[is_keep]})
    drop = pd.DataFrame({"gidx": g[~is_keep], "ID": ids[~is_keep]})
    if verbose:
        print(f"[prune] {n:,} variants, {len(eu):,} pairs at r2 >= {r2:g} -> "
              f"{len(keep):,} kept, {len(drop):,} pruned "
              f"({rounds} parallel round(s))")
    if output is not None:
        _write_df(keep, f"{output}.prune.in")
        _write_df(drop, f"{output}.prune.out")
    return keep, drop


def _load_sumstats(obj, id_field: Sequence[str], p_field: Sequence[str],
                   log10: bool) -> pd.DataFrame:
    """Read association results down to two columns: ID and P.

    Field-name SEARCH ORDER rather than a fixed name, because the same
    quantity is called ID by plink2, SNP by plink1.9, and something else again
    by most published sumstats. plink2 solves this with --clump-id-field; we
    mirror it, earlier names winning, matching its documented precedence.
    """
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        df = pd.read_csv(str(obj), sep=None, engine="python")
    df = df.rename(columns={c: str(c).lstrip("#") for c in df.columns})

    def pick(cands, what):
        for c in cands:
            if c in df.columns:
                return c
        raise ValueError(
            f"no {what} column in sumstats: looked for {list(cands)}, found "
            f"{list(df.columns)[:12]}. Pass {what}_field= to override.")

    idc, pc = pick(id_field, "id"), pick(p_field, "p")
    out = pd.DataFrame({"ID": df[idc].astype(str),
                        "P": pd.to_numeric(df[pc], errors="coerce")})
    if log10:
        # plink2 --clump-log10: the column holds -log10(p). Convert once here
        # so every threshold downstream stays in ordinary p-space; carrying
        # two conventions with inverted comparisons invites a sign bug.
        out["P"] = np.power(10.0, -out["P"].to_numpy(dtype=float))
    out = out[np.isfinite(out["P"].to_numpy(dtype=float))]
    return out.drop_duplicates(subset="ID", keep="first").reset_index(drop=True)


def _greedy_clump(order: np.ndarray, pvals: np.ndarray, neighbours: dict,
                  p1: float, allow_overlap: bool = False):
    """Sequential greedy clumping -- now the ORACLE, not the production path.

    :func:`clump_core` replaced this: the same answer computed array-parallel,
    on the device. This is kept because a second, independently written
    implementation is the strongest correctness evidence available, and
    `test_parallel_core_equals_sequential_greedy` checks the two agree on 400
    random windowed graphs across both overlap modes. Same role the EM solver
    plays for the D' cubic.

    Pure NumPy and GPU-free by construction, so the logic that decides the
    answer stays testable anywhere.

    ``order`` is row indices sorted by ascending p; ``neighbours[i]`` lists the
    rows whose r^2 with ``i`` cleared the threshold inside the kb window.
    Returns ``[(index_row, [member_rows...]), ...]``.

    This loop is O(edges), not O(p^2), because every r^2 it consults was
    computed once, up front. plink recomputes LD per index variant -- the right
    call when a pair is expensive and memory is precious, the wrong one when
    the whole neighbourhood is a single batched GEMM.

    NOTE p2 is deliberately absent. It does NOT gate membership: measured on a
    400-variant fixture, plink2 v2.0.0-a.7.1 reports an IDENTICAL clump count
    (182) and identical TOTAL (149) at p2=0.01 and p2=1.0, with only the SP2
    column changing. p2 is a DISPLAY threshold -- it selects which members are
    listed -- so it is applied when the frame is built, not here. Gating
    membership on it under-counts TOTAL and zeroes the NONSIG bin, which is
    exactly the bug this signature now makes impossible.

    ``allow_overlap`` governs membership only. A variant absorbed into a clump
    is barred from indexing one of its own either way, even if its p clears p1.
    Also measured rather than inferred ("let non-index variants join multiple
    clumps" is ambiguous): on a fixture with rsA=1e-9 and rsC=1e-5 at r^2=0.94,
    plink2 emits byte-identical output with and without
    --clump-allow-overlap, and rsC never indexes despite clearing p1.
    """
    can_index = np.ones(len(pvals), dtype=bool)   # not yet absorbed by a clump
    can_join = np.ones(len(pvals), dtype=bool)    # not yet consumed as a member
    clumps = []
    for i in order:
        if pvals[i] > p1:
            break                       # order is sorted: no candidates left
        if not can_index[i]:
            continue                    # absorbed by a more significant index
        members = [j for j in neighbours.get(int(i), ())
                   if j != i and can_join[j]]
        # Position order, not p order. plink2's SP2 lists members by
        # coordinate; rows here are gidx-sorted, so the row index IS position
        # order, and matching it lets the two outputs be diffed directly.
        members.sort()
        clumps.append((int(i), [int(j) for j in members]))
        can_index[i] = can_join[i] = False
        can_index[members] = False                # never index, either way
        if not allow_overlap:
            can_join[members] = False
    return clumps


# ---------------------------------------------------------------------------
# Array-parallel clumping. The greedy loop LOOKS irreducibly serial and is not.
# Three facts make the whole thing data-parallel, each checked against the
# sequential implementation before this was written (see test_ld_clump.py):
#
#   1. A candidate with no lower-p UNASSIGNED candidate neighbour can never be
#      absorbed, so it is definitely an index. Selecting all such vertices at
#      once and repeating computes the lexicographically-first maximal
#      independent set -- exactly what sequential greedy computes. Measured at
#      2.0 rounds on average and 4 at worst, against O(p) sequential steps.
#   2. Membership is an ARGMIN, not a sequence: under no-overlap a variant
#      joins the lowest-p index adjacent to it, because that is the first index
#      the sequential loop would have reached.
#   3. TOTAL and the bins are segmented counts.
#
# Every phase is a reduction over the edge set, so nothing here builds a
# host-side adjacency structure and nothing is O(p^2).
#
# KERNEL SOURCE MUST STAY PURE ASCII (cf. 34b4a59).
# ---------------------------------------------------------------------------
_CLUMP_MODULE = None

_CLUMP_SRC = r'''
/* Each undirected edge is stored ONCE; every kernel handles both endpoints.
   That halves memory against a symmetric CSR and needs no sort. */

extern "C" __global__
void clump_nbr_min(const int* eu, const int* ev, const int* rank,
                   const unsigned char* alive, int* out, long long n_edges)
{
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int a = eu[e], b = ev[e];
    if (!alive[a] || !alive[b]) return;
    atomicMin(&out[a], rank[b]);
    atomicMin(&out[b], rank[a]);
}

extern "C" __global__
void clump_claim(const int* eu, const int* ev,
                 const unsigned char* picked, unsigned char* claimed,
                 long long n_edges)
{
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int a = eu[e], b = ev[e];
    if (picked[a]) claimed[b] = 1;
    if (picked[b]) claimed[a] = 1;
}

/* For every non-index vertex, the rank of the lowest-ranked adjacent INDEX.
   That one atomicMin IS the whole no-overlap membership rule. */
extern "C" __global__
void clump_assign(const int* eu, const int* ev, const int* rank,
                  const unsigned char* is_index, int* best, long long n_edges)
{
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int a = eu[e], b = ev[e];
    if (is_index[a] && !is_index[b]) atomicMin(&best[b], rank[a]);
    if (is_index[b] && !is_index[a]) atomicMin(&best[a], rank[b]);
}

/* Per-variant sums read straight from the 2-bit packed bytes.

   The moments pre-pass called _build_dosage, which materialises THREE fp32
   planes (G, G2, M) of chunk x n_samples purely to produce two per-variant
   sums. At n = 500,000 with p = 20,000 that is ~120 GB of plane writes to
   compute 160 KB of output, and it dominated standard-GWAS clumping.

   Accumulation is in INTEGERS, so this stays bit-exact rather than merely
   close: sum(g) <= 2n and sum(g*g) <= 4n, both exactly representable in fp32
   while 4n < 2^24 -- the bound the rest of the module already relies on. The
   header's mu_x/sxx would be cheaper still, but sxx is stored as float32 and
   reconstructing sum(g*g) from it would inject rounding into quantities this
   module guarantees are exact.

   One block per variant; threads stride over samples and reduce in shared
   memory. */
extern "C" __global__
void variant_moments(const unsigned char* __restrict__ packed,
                     float* __restrict__ s_out, float* __restrict__ q_out,
                     const long long n_samples, const long long n_variants,
                     const long long bytes_per_variant)
{
    long long v = (long long)blockIdx.x;
    if (v >= n_variants) return;
    const unsigned char* row = packed + v * bytes_per_variant;

    long long s = 0, q = 0;
    for (long long i = (long long)threadIdx.x; i < n_samples;
         i += (long long)blockDim.x) {
        unsigned char byte = row[i >> 2];
        int code = (byte >> (6 - 2 * (i & 3))) & 3;   /* big-endian in byte */
        int g = (code == 3) ? 0 : code;               /* missing -> 0       */
        s += g;
        q += g * g;
    }

    __shared__ long long sh_s[256];
    __shared__ long long sh_q[256];
    sh_s[threadIdx.x] = s;
    sh_q[threadIdx.x] = q;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            sh_s[threadIdx.x] += sh_s[threadIdx.x + stride];
            sh_q[threadIdx.x] += sh_q[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        s_out[v] = (float)sh_s[0];
        q_out[v] = (float)sh_q[0];
    }
}

/* RECTANGULAR epilogue: index CANDIDATES against their windows, instead of
   every relevant variant against every other.

   The banded scan computes all-pairs LD across the whole relevant set. That
   is the right shape when nearly every variant is a candidate (p1 = 1, the
   polygenic-score case) and badly wrong when few are. Measured on real chr22
   at standard thresholds -- 168 candidates out of 170,949 variants -- the
   banded scan did roughly 500x more pair evaluations than the answer needs,
   and lost to plink2 by 23x. plink computes LD lazily around each index
   variant; this is that, done in parallel.

   Candidates are scattered, so side A is an explicit row list rather than a
   base offset. A pair between two candidates is emitted once, from the lower
   row, so the edge list carries no duplicates. */
extern "C" __global__
void clump_epilogue_rect(
    const float* __restrict__ S,         /* (bi x bj) cross products      */
    const float* __restrict__ sA, const float* __restrict__ sB,
    const float* __restrict__ qA, const float* __restrict__ qB,
    const float nsamp,
    const long long* __restrict__ rowsA, /* (bi,) global row per candidate */
    const long long j0, const long long bi, const long long bj,
    const long long* __restrict__ pos,   /* (p,) positions                */
    const long long span,                /* kb window, in bp              */
    const unsigned char* __restrict__ is_cand,
    const float min_r2,
    long long* __restrict__ out_i, long long* __restrict__ out_j,
    float* __restrict__ out_r,
    unsigned long long* __restrict__ counter, const long long capacity)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= bi * bj) return;
    long long a = t / bj;
    long long b = t - a * bj;
    long long gi = rowsA[a];
    long long gj = j0 + b;
    if (gi == gj) return;
    if (is_cand[gj] && gj < gi) return;   /* cand-cand: emit once only    */

    long long d = pos[gi] - pos[gj];
    if (d < 0) d = -d;
    if (d > span) return;                 /* exact bp window              */

    double n  = (double)nsamp;
    double sa = (double)sA[a], sb = (double)sB[b];
    double qa = (double)qA[a], qb = (double)qB[b];
    double vA = n * qa - sa * sa;
    double vB = n * qb - sb * sb;
    if (!(vA > 0.0) || !(vB > 0.0)) return;    /* monomorphic             */

    double r = (n * (double)S[t] - sa * sb) / (sqrt(vA) * sqrt(vB));
    if (r >  1.0) r =  1.0;
    if (r < -1.0) r = -1.0;
    if (min_r2 > 0.0f && (float)(r * r) < min_r2) return;

    unsigned long long slot = atomicAdd(counter, 1ULL);
    if (slot < (unsigned long long)capacity) {
        out_i[slot] = gi;
        out_j[slot] = gj;
        out_r[slot] = (float)r;
    }
}
'''

assert _CLUMP_SRC.isascii(), "kernel source must be pure ASCII (cf. 34b4a59)"

_INT_MAX = int(np.iinfo(np.int32).max)


def _get_clump_module():
    global _CLUMP_MODULE
    if _CLUMP_MODULE is None and HAS_CUPY:
        _CLUMP_MODULE = cp.RawModule(code=_CLUMP_SRC)
    return _CLUMP_MODULE


def clump_core(eu, ev, rank, cand, allow_overlap=False, xp=np):
    """Index selection and member assignment, array-parallel, CPU or GPU.

    ``eu``/``ev`` hold the two endpoints of each undirected edge (each pair
    once). ``rank`` is a UNIQUE per-vertex p-value rank (lower = more
    significant), so ties are already broken by the caller. ``cand`` marks
    index candidates (p <= p1).

    Returns ``(is_index, owner, rounds)``; ``owner[v]`` is the rank of the
    index claiming v, or ``_INT_MAX`` for none. Under ``allow_overlap`` the
    owner array is not used -- membership then comes straight off the edges.

    The identical code runs under NumPy and CuPy: ``np.minimum.at`` is the CPU
    analogue of ``atomicMin``. So the CPU path is a genuine reference for the
    GPU one rather than a second, differently-shaped algorithm.
    """
    n = int(len(rank))
    gpu = xp is not np
    nE = int(len(eu))
    if gpu:
        mod = _get_clump_module()
        k_min = mod.get_function("clump_nbr_min")
        k_claim = mod.get_function("clump_claim")
        k_assign = mod.get_function("clump_assign")
        grid = ((nE + 255) // 256 or 1,)

    alive = xp.asarray(cand).astype(xp.bool_).copy()
    is_index = xp.zeros(n, dtype=xp.bool_)
    rounds = 0
    while bool(alive.any()):
        rounds += 1
        nbr = xp.full(n, _INT_MAX, dtype=xp.int32)
        if nE:
            if gpu:
                k_min(grid, (256,), (eu, ev, rank,
                                     alive.view(xp.uint8), nbr, np.int64(nE)))
            else:
                m = alive[eu] & alive[ev]
                np.minimum.at(nbr, eu[m], rank[ev[m]])
                np.minimum.at(nbr, ev[m], rank[eu[m]])
        picked = alive & (rank < nbr)
        if not bool(picked.any()):
            # Impossible: the globally lowest-ranked alive vertex always
            # qualifies. Guard anyway -- spinning forever on a GPU is a worse
            # failure than an exception.
            raise RuntimeError("clump MIS made no progress; edge list corrupt")
        is_index |= picked
        alive &= ~picked
        if nE:
            claimed = xp.zeros(n, dtype=xp.uint8)
            if gpu:
                k_claim(grid, (256,), (eu, ev, picked.view(xp.uint8),
                                       claimed, np.int64(nE)))
            else:
                claimed[ev[picked[eu]]] = 1
                claimed[eu[picked[ev]]] = 1
            alive &= ~claimed.astype(xp.bool_)

    owner = xp.full(n, _INT_MAX, dtype=xp.int32)
    if nE and not allow_overlap:
        if gpu:
            k_assign(grid, (256,), (eu, ev, rank, is_index.view(xp.uint8),
                                    owner, np.int64(nE)))
        else:
            ia, ib = is_index[eu], is_index[ev]
            m = ia & ~ib
            np.minimum.at(owner, ev[m], rank[eu[m]])
            m = ib & ~ia
            np.minimum.at(owner, eu[m], rank[ev[m]])
    return is_index, owner, rounds


def membership_pairs(is_index, owner, rank, eu, ev, allow_overlap, xp=np):
    """(index_row, member_row) pairs, computed wherever the arrays live.

    Runs on the device under CuPy, so the only thing that ever crosses to the
    host is the membership list itself -- which is the output, and is far
    smaller than the edge list it was derived from. Under no-overlap it does
    not touch the edges at all: ``owner`` already holds the answer in O(p).
    """
    if allow_overlap:
        m1 = is_index[eu] & ~is_index[ev]
        m2 = is_index[ev] & ~is_index[eu]
        return (xp.concatenate([eu[m1], ev[m2]]),
                xp.concatenate([ev[m1], eu[m2]]))
    n = int(len(rank))
    rank_to_row = xp.full(n + 1, -1, dtype=xp.int64)
    idx_rows = xp.flatnonzero(is_index)
    rank_to_row[rank[idx_rows]] = idx_rows
    b = xp.flatnonzero((owner < _INT_MAX) & ~is_index)
    return rank_to_row[owner[b]], b


def _clumps_frame_vec(is_index, rank, a, b, rel, bins, p2, sp2=True):
    """Build the clump table straight from (index, member) arrays.

    No Python-level member lists anywhere. :func:`_clumps_from_pairs` boxes
    every membership into a Python int inside a list, which is harmless at test
    sizes and cost an OOM kill on real chr22 -- a 250 kb window at r2 >= 0.5
    produces millions of memberships, and a few million boxed ints in lists is
    gigabytes.

    TOTAL and the bins are ``bincount``s over the membership arrays. Only SP2
    needs actual identifiers, and only for the ``p <= p2`` subset, so that is
    the one place strings get built -- with a loop over CLUMPS, not members.
    """
    ids = rel["ID"].to_numpy()
    pos = rel["POS"].to_numpy()
    gidx = rel["gidx"].to_numpy()
    pv = rel["P"].to_numpy(dtype=float)
    chrom = (rel["CHR"].to_numpy() if "CHR" in rel.columns
             else np.full(len(rel), "."))

    is_index = np.asarray(is_index)
    rank = np.asarray(rank)
    idx_rows = np.flatnonzero(is_index)
    if not len(idx_rows):
        return _empty_clumps(bins)
    order = idx_rows[np.argsort(rank[idx_rows])]     # plink's output order
    k = len(order)
    slot = np.full(len(rel), -1, dtype=np.int64)
    slot[order] = np.arange(k)

    a = np.asarray(a)
    b = np.asarray(b)
    sa = slot[a] if len(a) else np.empty(0, dtype=np.int64)
    total = (np.bincount(sa, minlength=k) if len(a)
             else np.zeros(k, dtype=np.int64))

    asc = sorted(bins)
    bin_cols = _bin_columns(bins)
    counts = np.zeros((k, len(bin_cols)), dtype=np.int64)
    if len(a):
        mp = pv[b]
        masks = [mp > asc[-1]] if asc else [np.ones(len(b), bool)]
        for i in range(len(asc) - 1, -1, -1):
            lo = -np.inf if i == 0 else asc[i - 1]
            masks.append((mp > lo) & (mp <= asc[i]))
        for c, m in enumerate(masks):
            if m.any():
                counts[:, c] = np.bincount(sa[m], minlength=k)

    sp2_col = np.full(k, ".", dtype=object)
    if sp2 and len(a):
        keep = pv[b] <= p2
        if keep.any():
            ka, kb = sa[keep], b[keep]
            o = np.lexsort((kb, ka))            # by clump, then position
            ka, kb = ka[o], kb[o]
            starts = np.flatnonzero(np.r_[True, ka[1:] != ka[:-1]])
            ends = np.r_[starts[1:], len(ka)]
            sid = ids[kb].astype(str)
            for s, e in zip(starts, ends):
                sp2_col[ka[s]] = ",".join(sid[s:e])

    out = pd.DataFrame({
        "CHR": chrom[order], "POS": pos[order].astype("int64"),
        "ID": ids[order], "P": pv[order], "TOTAL": total})
    for c, name in enumerate(bin_cols):
        out[name] = counts[:, c]
    out["SP2"] = sp2_col
    out["gidx"] = gidx[order].astype("int64")
    tmpl = _empty_clumps(bins)
    return out[list(tmpl.columns)].astype(
        {kk: v for kk, v in tmpl.dtypes.items() if kk in out.columns})


def _clumps_from_pairs(is_index, rank, a, b):
    """Group host-side (index, member) pairs into plink's clump order.

    Retained for the tests, which compare against the sequential oracle's list
    form. Production goes through :func:`_clumps_frame_vec` -- see its docstring
    for why this shape does not survive real data.
    """
    is_index = np.asarray(is_index)
    rank = np.asarray(rank)
    idx_rows = np.flatnonzero(is_index)
    members = {int(i): [] for i in idx_rows}
    if len(a):
        o = np.argsort(np.asarray(a), kind="stable")
        aa, bb = np.asarray(a)[o], np.asarray(b)[o]
        edges = np.flatnonzero(np.r_[True, aa[1:] != aa[:-1]])
        for s, e in zip(edges, np.r_[edges[1:], len(aa)]):
            members[int(aa[s])] = sorted(int(x) for x in bb[s:e])
    order = idx_rows[np.argsort(rank[idx_rows])]
    return [(int(i), members[int(i)]) for i in order]


def _variant_moments(packed, p, ns, bpv):
    """Per-variant sum(g) and sum(g*g), straight from packed bytes.

    Replaces a pre-pass that built three fp32 planes per chunk to produce two
    per-variant sums. Integer accumulation keeps it bit-exact; the caller must
    already have established the file has no missing calls (both fused paths
    require that), since missing is folded to dosage 0 here.
    """
    s_v = cp.empty(p, dtype=cp.float32)
    q_v = cp.empty(p, dtype=cp.float32)
    _get_clump_module().get_function("variant_moments")(
        (int(p),), (256,),
        (packed, s_v, q_v, np.int64(ns), np.int64(p), np.int64(bpv)))
    return s_v, q_v


def _contiguous_runs(rows):
    """Split a strictly increasing row index array into contiguous runs.

    A clumping relevant-set is a union of kb windows, so it is a handful of
    long runs rather than scattered singletons -- which is exactly what makes
    ranged reads worthwhile instead of one whole-file read plus a gather.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return []
    brk = np.flatnonzero(np.diff(rows) != 1)
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk + 1, len(rows)]
    return [(int(rows[s]), int(rows[e - 1]) + 1) for s, e in zip(starts, ends)]


def _load_packed_rows(reader, rows, bpv, verbose=False,
                      chunk_bytes=(192 << 20)):
    """Device array of just the packed rows needed, in ``rows`` order.

    Profiling put 96% of standard-GWAS clumping here -- 5.37 s of a 5.62 s run
    at n=500,000, against 0.05 s for the GEMM scan the previous two
    optimisation passes had targeted. The old version read the WHOLE file,
    copied all of it to the device, then fancy-indexed down to the subset: ~2x
    the bytes it needed, plus a second full-size allocation to do it in.

    Now it reads only the byte ranges the rows cover, straight into their
    slots. An identity selection skips the copy entirely, which is the C+T
    case where every variant is a candidate.
    """
    rows = np.asarray(rows, dtype=np.int64)
    p, nv = len(rows), int(reader.n_variants)
    runs = _contiguous_runs(rows)
    total = sum(hi - lo for lo, hi in runs)
    if verbose:
        print(f"[clump] reading {p:,}/{nv:,} rows in {len(runs):,} run(s) = "
              f"{total * bpv / 2**30:.2f} GiB (whole file is "
              f"{nv * bpv / 2**30:.2f} GiB)")

    # HOST memory is the binding constraint here, not device memory. A RunPod
    # A100 pod was observed with 2 GB of host RAM against 80 GB of GPU:
    # read_packed_bytes copies into host memory, so pulling a 2.33 GiB
    # chromosome in one call raises MemoryError on the host while the card sits
    # empty. Each run is therefore fetched in bounded pieces and pushed to the
    # device as it goes, so peak host use is one chunk regardless of file size.
    chunk_rows = max(1, min(p, chunk_bytes // max(bpv, 1)))
    out = cp.empty((p, bpv), dtype=cp.uint8)
    at = 0
    for lo, hi in runs:
        for c0 in range(lo, hi, chunk_rows):
            c1 = min(c0 + chunk_rows, hi)
            buf = np.frombuffer(reader.read_packed_bytes(c0, c1),
                                dtype=np.uint8)
            k = c1 - c0
            out[at:at + k] = cp.asarray(buf).reshape(k, bpv)
            at += k
            del buf
    return out


def _plan_cand_tiles(cand, pos, span, max_cands, row_budget):
    """Group candidates into tiles whose window UNION stays bounded.

    Tiling by count alone is pessimal exactly where the rectangular scan is
    meant to win. Standard-GWAS candidates are scattered, so ~20 of them
    across 20 Mb land in a single tile whose union window is the whole
    chromosome -- scanning ~400,000 pairs to answer a question needing
    ~10,000, and building fp32 planes over every row on the way. Breaking a
    tile once its union exceeds ``row_budget`` keeps the scanned area
    proportional to the work rather than to how far apart the hits happen to
    be.
    """
    tiles = []
    i = 0
    while i < len(cand):
        lo = int(np.searchsorted(pos, pos[cand[i]] - span, "left"))
        j = i + 1
        while j < len(cand) and (j - i) < max_cands:
            hi_try = int(np.searchsorted(pos, pos[cand[j]] + span, "right"))
            if hi_try - lo > row_budget:
                break
            j += 1
        tiles.append(cand[i:j])
        i = j
    return tiles


def _clump_edges_rect_gpu(reader, rows, positions, cand_mask, kb, r2_thresh,
                          tf32, verbose, cand_tile=256, nbr_tile=None):
    """Edges between index CANDIDATES and their kb windows. Device-resident.

    Complexity is O(n_candidates * window) rather than O(p * window), which is
    the difference between doing the work the answer needs and doing the whole
    band. On real chr22 at standard thresholds that is 168 candidates instead
    of 170,949 variants.

    Two quantities are sized from the data rather than fixed, both of which
    were constants that misbehaved at biobank sample counts:

    * ``nbr_tile`` comes from a memory budget. It was 8,192 rows regardless of
      n, and _build_g materialises a tile x n_samples fp32 plane -- 16.4 GB at
      n = 500,000, which accounted for the entire observed peak.
    * candidate tiles are bounded by their window UNION rather than by count
      alone; see :func:`_plan_cand_tiles`.
    """
    ns = int(reader.n_samples)
    bpv = int(reader.bytes_per_variant)
    p = len(rows)
    span = int(round(kb * 1000))
    pos = np.asarray(positions, dtype=np.int64)
    cand = np.flatnonzero(np.asarray(cand_mask))
    if not len(cand):
        z = cp.empty(0, dtype=cp.int32)
        return z, z

    _t0_read = _time.perf_counter()
    packed = _load_packed_rows(reader, rows, bpv, verbose)
    cp.cuda.Stream.null.synchronize()
    _t_read = _time.perf_counter() - _t0_read

    # Per-variant moments once, streamed -- identical to the banded scan.
    # Genotype-only on purpose: ns, _build_g and the kb-window logic below all
    # assume dosages, so there is no phased variant of this scan to select.
    # d5de783 copied a `if phased` branch in here from the fused scan, where
    # `phased` is a parameter; here it is not bound, so every GPU rectangular
    # clump raised NameError.
    _t0 = _time.perf_counter()
    s_v, q_v = _variant_moments(packed, p, ns, bpv)
    cp.cuda.Stream.null.synchronize()
    _t_mom = _time.perf_counter() - _t0

    pos_d = cp.asarray(pos)
    is_cand_d = cp.zeros(p, dtype=cp.uint8)
    is_cand_d[cp.asarray(cand)] = 1
    kern = _get_clump_module().get_function("clump_epilogue_rect")

    # Bound the work up front so the capacity guess is informed rather than
    # optimistic: sum of each candidate tile's window width.
    lo_all = np.searchsorted(pos, pos[cand] - span, side="left")
    hi_all = np.searchsorted(pos, pos[cand] + span, side="right")
    planned = int((hi_all - lo_all).sum())

    # Size the neighbour tile from a MEMORY budget: _build_g materialises a
    # tile x n_samples fp32 plane, so a constant row count silently becomes
    # 16.4 GB at n = 500,000. Cap the plane near 1 GiB.
    if nbr_tile is None:
        nbr_tile = int(max(256, min(8192, (1 << 30) // max(ns * 4, 1))))
    # Bound a candidate tile's window union too, so scattered candidates do
    # not drag one tile across the whole chromosome.
    row_budget = max(int(4 * np.median(hi_all - lo_all)), 4 * nbr_tile, 4096)
    tiles = _plan_cand_tiles(cand, pos, span, cand_tile, row_budget)
    scanned = sum(len(t) * (int(np.searchsorted(pos, pos[t[-1]] + span, "right"))
                            - int(np.searchsorted(pos, pos[t[0]] - span, "left")))
                  for t in tiles)
    if verbose:
        print(f"[clump] rectangular scan: {len(cand):,} candidates in "
              f"{len(tiles):,} tile(s), nbr_tile={nbr_tile:,} rows "
              f"({nbr_tile * ns * 4 / 2**30:.2f} GiB plane); "
              f"{scanned:,} pair evaluations "
              f"(ideal {planned:,}, banded ~{p * (hi_all - lo_all).max():,})")

    def run(capacity):
        out_i = cp.empty(capacity, dtype=cp.int64)
        out_j = cp.empty(capacity, dtype=cp.int64)
        out_r = cp.empty(capacity, dtype=cp.float32)
        counter = cp.zeros(1, dtype=cp.uint64)
        with _Tf32(tf32):
            # One reusable neighbour plane for the whole scan. A fresh
            # tile x n allocation per neighbour block costs about as much as
            # the unpack kernel itself -- the same finding that put buffer
            # reuse into the banded scan.
            bufB = cp.empty((nbr_tile, ns), dtype=cp.float32)
            for cs in tiles:
                rows_d = cp.asarray(cs.astype(np.int64))
                # gather is tiny: |tile| x bytes_per_variant
                Ga = _build_g(packed[cp.asarray(cs)], 0, len(cs), ns, bpv)
                sa, qa = s_v[cp.asarray(cs)], q_v[cp.asarray(cs)]
                lo = int(np.searchsorted(pos, pos[cs[0]] - span, "left"))
                hi = int(np.searchsorted(pos, pos[cs[-1]] + span, "right"))
                for j0 in range(lo, hi, nbr_tile):
                    j1 = min(j0 + nbr_tile, hi)
                    Gb = _build_g(packed, j0, j1, ns, bpv,
                                  out=bufB[:j1 - j0])
                    S = Ga @ Gb.T
                    bi, bj = len(cs), j1 - j0
                    nthread = bi * bj
                    kern(((nthread + 255) // 256,), (256,),
                         (S, sa, s_v[j0:j1], qa, q_v[j0:j1], np.float32(ns),
                          rows_d, np.int64(j0), np.int64(bi), np.int64(bj),
                          pos_d, np.int64(span), is_cand_d,
                          np.float32(r2_thresh), out_i, out_j, out_r,
                          counter, np.int64(capacity)))
                    del S
                del Ga
        del bufB
        return out_i, out_j, out_r, int(counter[0])

    cap = max(1 << 16, min(int(50e6), planned // 4 + 1024))
    _t0 = _time.perf_counter()
    oi, oj, _orr, found = run(cap)
    cp.cuda.Stream.null.synchronize()
    if verbose:
        print(f"[clump] rect timing: read+gather {_t_read:.2f}s  "
              f"moments {_t_mom:.2f}s  scan "
              f"{_time.perf_counter() - _t0:.2f}s  "
              f"(packed {packed.nbytes / 2**30:.2f} GiB)")
    if found > cap:
        if verbose:
            print(f"[clump] buffer held {cap:,}, {found:,} survived -- "
                  f"re-running the epilogue at exact size")
        del oi, oj, _orr
        cp.get_default_memory_pool().free_all_blocks()
        oi, oj, _orr, found = run(found)
    del packed
    cp.get_default_memory_pool().free_all_blocks()
    return oi[:found].astype(cp.int32), oj[:found].astype(cp.int32)


def _clump_edges_gpu(reader, rows, positions, kb, r2_thresh, tile_size,
                     tf32, verbose):
    """Device-resident edge list for the clump graph. Never reaches the host.

    The fused epilogue kernel takes an INDEX window, not a bp one, and
    ld_matrix disables the fused path entirely when window_kb is set. Rather
    than lose it, convert: on sorted positions the kb window is contained in an
    index window of width ``max_i (hi(i) - i)``, so scan with that -- a
    superset -- and drop the surplus with an exact bp test on the device. The
    surplus is bounded by how uneven the local variant density is, which is
    reported so a pathological file is visible rather than merely slow.
    """
    span = int(round(kb * 1000))
    pos = np.asarray(positions, dtype=np.int64)
    hi = np.searchsorted(pos, pos + span, side="right")
    win = int(np.max(hi - np.arange(len(pos)) - 1)) if len(pos) else 1
    win = max(1, win)
    if verbose:
        mean_span = float(np.mean(hi - np.arange(len(pos)) - 1)) if len(pos) else 0
        print(f"[clump] {kb:g} kb -> index window {win:,} "
              f"(mean {mean_span:,.0f}; superset factor "
              f"{win / max(mean_span, 1):.1f}x)")
    ii, jj, _rr = _scan_gpu_fused(reader, rows, win, r2_thresh,
                                  tile_size=tile_size, verbose=verbose,
                                  tf32=tf32)
    if ii.size:
        pos_d = cp.asarray(pos)
        keep = cp.abs(pos_d[jj] - pos_d[ii]) <= span
        ii, jj = ii[keep], jj[keep]
    return ii.astype(cp.int32), jj.astype(cp.int32)


def _bin_columns(bins: Sequence[float]):
    """plink2's bin column NAMES, in its order: NONSIG then descending bounds.

    Verified against plink2 v2.0.0-a.7.1 output rather than assumed: with the
    default boundaries it writes `NONSIG S0.05 S0.01 S0.001 S0.0001`, i.e.
    least significant band first. Generating the names from the boundaries
    keeps a non-default --clump-bins diffable too.
    """
    return ["NONSIG"] + [f"S{b:g}" for b in sorted(bins, reverse=True)]


def _clump_bin_counts(member_p: np.ndarray, bins: Sequence[float]):
    """Member counts per p-value band, aligned to :func:`_bin_columns`."""
    asc = sorted(bins)
    counts = [int((member_p > asc[-1]).sum()) if asc else len(member_p)]
    for k in range(len(asc) - 1, -1, -1):
        lo = -np.inf if k == 0 else asc[k - 1]
        counts.append(int(((member_p > lo) & (member_p <= asc[k])).sum()))
    return counts


def _empty_clumps(bins: Sequence[float] = _CLUMP_BINS) -> pd.DataFrame:
    """Zero rows, full schema, correct dtypes -- same contract as _empty_pairs.

    An empty result must not be a bare DataFrame(): downstream code doing
    df['TOTAL'].sum() should get 0 rather than a KeyError, and concatenating
    an empty result with a non-empty one must not silently widen dtypes.
    """
    cols = {"CHR": object, "POS": "int64", "ID": object, "P": "float64",
            "TOTAL": "int64"}
    cols.update({c: "int64" for c in _bin_columns(bins)})
    cols.update({"SP2": object, "gidx": "int64"})
    return pd.DataFrame({c: pd.Series([], dtype=d) for c, d in cols.items()})


def _clumps_to_frame(clumps, rel: pd.DataFrame, bins, p2: float) -> pd.DataFrame:
    """Render clumps in plink2's .clumps layout.

    TOTAL and the bins count EVERY member; SP2 lists only those at p <= p2.
    That asymmetry is plink2's, verified rather than assumed -- see the note in
    _greedy_clump. Deriving both from one member list keeps them consistent by
    construction, so TOTAL can never disagree with the bins.
    """
    if not clumps:
        return _empty_clumps(bins)
    ids, pos = rel["ID"].to_numpy(), rel["POS"].to_numpy()
    gidx, pv = rel["gidx"].to_numpy(), rel["P"].to_numpy(dtype=float)
    chrom = (rel["CHR"].to_numpy() if "CHR" in rel.columns
             else np.full(len(rel), "."))
    bin_cols = _bin_columns(bins)
    rows = []
    for i, m in clumps:
        mp = pv[m] if m else np.empty(0)
        listed = [j for j in m if pv[j] <= p2]
        row = {"CHR": chrom[i], "POS": int(pos[i]), "ID": ids[i],
               "P": float(pv[i]), "TOTAL": len(m)}
        row.update(dict(zip(bin_cols, _clump_bin_counts(mp, bins))))
        # '.' for an empty list, and bare IDs, both matching plink2's .clumps
        # output so the two files can be diffed directly.
        row["SP2"] = ",".join(str(ids[j]) for j in listed) or "."
        row["gidx"] = int(gidx[i])
        rows.append(row)
    out = pd.DataFrame(rows)
    tmpl = _empty_clumps(bins)
    return out[list(tmpl.columns)].astype(
        {k: v for k, v in tmpl.dtypes.items() if k in out.columns})


def ld_clump(
    cugen: Union[str, Path],
    sumstats,
    *,
    annotation=None,
    p1: float = 1e-4,
    p2: float = 0.01,
    r2: float = 0.5,
    kb: float = 250.0,
    allow_overlap: bool = False,
    id_field: Sequence[str] = ("ID", "SNP"),
    p_field: Sequence[str] = ("P", "PVAL", "P_VALUE", "p_value"),
    log10: bool = False,
    bins: Sequence[float] = _CLUMP_BINS,
    sp2: bool = True,
    output: Optional[Union[str, Path]] = None,
    backend: str = "auto",
    precision: str = "auto",
    tile_size: Optional[int] = None,
    max_pairs: int = 100_000_000,
    device: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """LD-based clumping of association results (alias: ``cg.clump``).

    plink2 ``--clump`` parity, with its defaults: ``p1=1e-4``, ``p2=0.01``,
    ``r2=0.5``, ``kb=250`` (a RADIUS, as in plink).

    Two structural choices carry the whole design, and both come from what the
    LD work measured rather than from theory:

    **1. Only variants with p <= p2 can participate at all.** A member must
    clear p2 by definition, and an index must clear p1 (<= p2 in any sensible
    configuration), so every other variant is irrelevant BEFORE a genotype is
    read. On a typical GWAS that is ~1.1M variants down to a few thousand --
    a reduction of two to three orders of magnitude in the O(p^2) term. This
    is not an approximation, it is the definition of the problem.

    **2. The parallel and serial halves are separated.** Clumping is greedy and
    therefore sequential, but the r^2 values it consults never change while it
    runs. So they are ALL computed first, in one batched windowed scan
    (:func:`ld_matrix` with ``stats=("r2",)``, which selects the 1-GEMM path
    and skips the 3x3 table entirely), and the greedy loop then walks a small
    edge list. plink recomputes LD per index variant; that is right when a pair
    is expensive and wrong when a whole neighbourhood is one GEMM over a tile.

    Everything the LD path learned therefore applies here for free: the r-only
    single GEMM, the fused epilogue kernel with atomic compaction (so only
    pairs above ``r2`` ever leave the device), TF32 where the hardware has it,
    and the cuDF device write path.

    PHASE -- read this before comparing against plink
    -------------------------------------------------
    plink2 ``--clump`` uses **phased** r^2 by default and .cugen discards
    phase, so compare against ``--clump-unphased``. On phased input (e.g. stock
    1000 Genomes VCFs) the two cannot agree, for exactly the reason set out in
    the module docstring. Our r^2 matches ``--r2-unphased`` exactly.

    Parameters
    ----------
    cugen
        Path to a ``.cugen`` file.
    sumstats
        Association results: a DataFrame, or a path to a TSV/CSV. Needs a
        variant-ID column and a p-value column -- see ``id_field``/``p_field``.
    annotation
        Table with ``gidx``, ``ID`` and ``POS`` (``CHR`` optional). REQUIRED:
        sumstats identify variants by ID and ``kb`` needs coordinates, and a
        .cugen stores neither.
    p1, p2
        Index-variant and member p-value ceilings.
    r2, kb
        LD and distance thresholds; ``kb`` is a radius.
    allow_overlap
        Let a non-index variant join more than one clump (plink2
        ``--clump-allow-overlap``). Default ``False``, matching plink.
    log10
        ``sumstats`` p-values are -log10(p) (plink2 ``--clump-log10``).
    sp2
        Build the ``SP2`` member-ID column. It is the only output whose size
        grows with the number of MEMBERSHIPS rather than the number of clumps,
        so setting it False makes the whole pipeline O(clumps) on the host.
        Worth it when you only need the index variants.

    Returns
    -------
    One row per clump: ``CHR POS ID P TOTAL BINS SP2 gidx``. ``SP2`` lists the
    members, ``TOTAL`` counts them, ``BINS`` bands them by p.

    References
    ----------
    Purcell et al. (2007) AJHG 81:559-575     PLINK, --clump
    Chang et al. (2015) GigaScience 4:7       PLINK 2, the parity reference
    """
    if annotation is None:
        raise ValueError(
            "ld_clump needs annotation= with gidx, ID and POS: sumstats "
            "identify variants by ID and the kb window needs coordinates, "
            "neither of which a .cugen stores.")
    if not 0.0 <= r2 <= 1.0:
        raise ValueError(f"r2 must be in [0, 1], got {r2}")
    # There is deliberately NO p2 >= p1 guard. An earlier version raised on
    # p2 < p1, reasoning that no variant could be a member without also being
    # an index candidate -- which follows only from the false premise that p2
    # gates membership. It does not (see _greedy_clump), and p1=1 with p2=0.01
    # is the standard clumping-and-thresholding configuration for polygenic
    # scores: index on everything, list only the significant members. plink2
    # accepts it, so that guard rejected the single most valuable use case.

    ann = (annotation if isinstance(annotation, pd.DataFrame)
           else pd.read_csv(str(annotation), sep=None, engine="python"))
    ann = ann.rename(columns={c: str(c).lstrip("#") for c in ann.columns})
    for need in ("gidx", "ID", "POS"):
        if need not in ann.columns:
            raise ValueError(f"annotation lacks a {need!r} column; it has "
                             f"{list(ann.columns)[:12]}")

    keep = ["gidx", "ID", "POS"] + (["CHR"] if "CHR" in ann.columns else [])
    merged = _load_sumstats(sumstats, id_field, p_field, log10) \
        .merge(ann[keep], on="ID", how="inner")
    if merged.empty:
        raise ValueError(
            "no sumstats ID matched the annotation. Check that both use the "
            "same variant naming -- plink's --make-bed rewrites every ID to "
            "'.' unless --set-all-var-ids was given, which silently produces "
            "exactly this.")

    # --- the reduction that makes the whole thing cheap --------------------
    # A member must lie within kb of an INDEX CANDIDATE and clear r^2. So the
    # only variants that can participate are the candidates (p <= p1) and
    # whatever falls inside one of their windows; everything else cannot enter
    # a clump by any route and is dropped before a genotype is read.
    #
    # NOT a p2 filter. p2 is presentational -- it selects which members appear
    # in SP2 and changes nothing else -- so screening on it would silently
    # under-count TOTAL and empty the NONSIG bin. Measured, not assumed.
    merged = merged.sort_values("gidx").reset_index(drop=True)
    n_all = len(merged)
    pos_all = merged["POS"].to_numpy(dtype=np.int64)
    is_cand = merged["P"].to_numpy(dtype=float) <= p1
    if not is_cand.any():
        if verbose:
            print(f"[clump] no variant reached p1={p1:g}; nothing to clump")
        return _empty_clumps(bins)

    span = int(round(kb * 1000))
    cand_pos = pos_all[is_cand]
    # Vectorised union of the candidate windows: for each candidate the rows
    # whose position is within +/-span. searchsorted on the (sorted) positions
    # turns that into two index bounds per candidate.
    lo = np.searchsorted(pos_all, cand_pos - span, side="left")
    hi = np.searchsorted(pos_all, cand_pos + span, side="right")
    keep_mask = np.zeros(n_all, dtype=bool)
    for a, b in zip(lo, hi):
        keep_mask[a:b] = True
    rel = merged[keep_mask].reset_index(drop=True)
    if verbose:
        print(f"[clump] {n_all:,} variants with p-values -> {len(rel):,} "
              f"within {kb:g} kb of one of {int(is_cand.sum()):,} index "
              f"candidates ({100 * len(rel) / max(n_all, 1):.1f}%); LD is "
              f"computed on those only")

    pv = rel["P"].to_numpy(dtype=float)
    rel_gidx = rel["gidx"].to_numpy()
    rel_pos = rel["POS"].to_numpy(dtype=np.int64)
    # gidx breaks p ties deterministically: without it the assignment would
    # depend on row order and two runs on the same data could disagree.
    order = np.lexsort((rel_gidx, pv))
    rank = np.empty(len(rel), dtype=np.int32)
    rank[order] = np.arange(len(rel), dtype=np.int32)
    cand = pv <= p1

    want_gpu = backend == "gpu" or (backend == "auto" and HAS_CUPY)
    reader = read_cugen(str(cugen))
    if backend == "gpu" and not HAS_CUPY:
        raise RuntimeError("backend='gpu' but CuPy is not available")
    # The FUSED path additionally needs a file with no missing calls. When it
    # cannot be used we still want the GPU -- the previous version dropped
    # straight to backend="numpy" here, so a .cugen carrying the HAS_MISSING
    # flag silently ran the O(p * window * n) NumPy reference even under
    # backend="gpu". On chr22 that is ~1e12 operations and an OOM kill, and it
    # looked like a memory bug in the device path rather than what it was:
    # the device path never ran.
    on_device = want_gpu and HAS_CUPY and not reader.has_missing
    if verbose:
        path = ("fused device kernels" if on_device else
                "gpu, non-fused (file has missing calls)" if want_gpu else
                "numpy reference (CPU)")
        print(f"[clump] path: {path}")

    if on_device:
        # --- everything below runs on the device -------------------------
        cp.cuda.Device(device).use()
        # gidx -> file row. NOT searchsorted: a .cugen carrying a gidx map is
        # under no obligation to have it sorted, and searchsorted would then
        # return confidently wrong rows rather than failing. An explicit lookup
        # errors on a gidx the file does not contain.
        lut = pd.Series(np.arange(int(reader.n_variants), dtype=np.int64),
                        index=np.asarray(reader.gidx))
        rows = lut.reindex(rel_gidx).to_numpy()
        if np.isnan(rows).any():
            missing = rel_gidx[np.isnan(rows)][:5]
            raise ValueError(
                f"{int(np.isnan(rows).sum())} annotation gidx values are not "
                f"in {cugen} (first few: {list(missing)}). The annotation and "
                "the .cugen disagree about which variants exist.")
        rows = rows.astype(np.int64)
        # The banded scan assumes rows ascend with position; rel is gidx-sorted
        # and a .cugen is position-ordered, so this holds -- but silently wrong
        # banding is the worst failure mode available, so check rather than
        # assume.
        if rows.size > 1 and not bool((np.diff(rows) > 0).all()):
            raise ValueError(
                "selected rows are not strictly increasing in file order; "
                "the kb window would be banded against the wrong neighbours. "
                "Is the .cugen sorted by position?")
        tf32 = _resolve_precision(precision, int(reader.n_samples), False)
        # Pick the scan SHAPE by candidate density. The banded scan evaluates
        # every pair in the band; the rectangular one evaluates only
        # candidates against their windows. Banded wins when almost everything
        # is a candidate (its tiles are dense and it reuses plane buffers);
        # rectangular wins by orders of magnitude when few are, which is the
        # standard-GWAS case that lost to plink2 by 23x.
        dense = cand.mean() > _CLUMP_DENSE_FRAC
        if verbose:
            print(f"[clump] {int(cand.sum()):,}/{len(cand):,} candidates "
                  f"({100 * cand.mean():.2f}%) -> "
                  f"{'banded' if dense else 'rectangular'} scan")
        with _Tf32(tf32):
            if dense:
                eu, ev = _clump_edges_gpu(reader, rows, rel_pos, kb, r2,
                                          tile_size, tf32, verbose)
            else:
                eu, ev = _clump_edges_rect_gpu(reader, rows, rel_pos, cand,
                                               kb, r2, tf32, verbose)
        n_edges = int(eu.size)
        rank_d, cand_d = cp.asarray(rank), cp.asarray(cand)
        is_index_d, owner_d, rounds = clump_core(eu, ev, rank_d, cand_d,
                                                 allow_overlap, xp=cp)
        a_d, b_d = membership_pairs(is_index_d, owner_d, rank_d, eu, ev,
                                    allow_overlap, xp=cp)
        # Only the membership list crosses to the host -- the edges never do.
        is_index = cp.asnumpy(is_index_d)
        a, b = cp.asnumpy(a_d), cp.asnumpy(b_d)
        del eu, ev, rank_d, cand_d, owner_d, a_d, b_d
        cp.get_default_memory_pool().free_all_blocks()
        if verbose:
            print(f"[clump] {n_edges:,} edges at r2 >= {r2:g}; "
                  f"index selection converged in {rounds} parallel round(s); "
                  f"{len(a):,} memberships returned to host")
    else:
        # --- non-fused: same algorithm, edges via the general scan ---------
        # backend is passed THROUGH, not forced to numpy. Hardcoding numpy
        # here is what turned "this file has missing calls" into "run the CPU
        # reference on a whole chromosome".
        pairs = ld_matrix(
            cugen, variants=rel_gidx, annotation=ann, window_kb=kb, min_r2=r2,
            stats=("r2",), output_format="pairs", backend=backend,
            precision=precision, tile_size=tile_size, device=device,
            max_pairs=max_pairs, verbose=False)
        pairs = pairs.to_pandas() if hasattr(pairs, "to_pandas") else pairs
        row_of = pd.Series(np.arange(len(rel)), index=rel_gidx)
        if len(pairs):
            eu = row_of.reindex(pairs["gidx_a"].to_numpy()).to_numpy()
            ev = row_of.reindex(pairs["gidx_b"].to_numpy()).to_numpy()
            ok = np.isfinite(eu) & np.isfinite(ev)
            eu = eu[ok].astype(np.int32)
            ev = ev[ok].astype(np.int32)
        else:
            eu = ev = np.empty(0, dtype=np.int32)
        if verbose:
            print(f"[clump] {len(eu):,} edges at r2 >= {r2:g} within "
                  f"{kb:g} kb (cpu reference path)")
        is_index, owner, rounds = clump_core(eu, ev, rank, cand,
                                             allow_overlap, xp=np)
        a, b = membership_pairs(is_index, owner, rank, eu, ev,
                                allow_overlap, xp=np)

    out = _clumps_frame_vec(is_index, rank, a, b, rel, bins, p2, sp2)
    if verbose:
        print(f"[clump] {len(out):,} clumps covering "
              f"{int(out['TOTAL'].sum()) + len(out):,} variants")
    if output is not None:
        _write_df(out, str(output))
    return out


# ---------------------------------------------------------------------------
# GPU: one small kernel to build indicator planes; all O(p^2) work is cuBLAS.
# Keeping hand-written CUDA to a minimum is deliberate -- 34b4a59 was an NVRTC
# crash from a single non-ASCII character, and that class of failure cannot be
# reproduced without a GPU. KERNEL SOURCE MUST STAY PURE ASCII.
# ---------------------------------------------------------------------------
_LD_PLANES_KERNEL = None

_LD_PLANES_SRC = r'''
extern "C" __global__
void build_ld_planes(const unsigned char* packed,
                     float* M, float* I1, float* I2,
                     const long long n_samples,
                     const long long n_variants,
                     const long long bytes_per_variant)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n_variants * n_samples;
    if (idx >= total) return;
    long long v = idx / n_samples;
    long long s = idx - v * n_samples;
    /* 2-bit codes are big-endian within the byte: sample 0 is the HIGH pair.
       Loop bound is n_samples so the zero-padded tail of the last byte, which
       would decode as dosage 0, can never leak in as a real sample. */
    unsigned char byte = packed[v * bytes_per_variant + (s >> 2)];
    int code = (byte >> (6 - 2 * (s & 3))) & 3;
    M[idx]  = (code != 3) ? 1.0f : 0.0f;   /* non-missing indicator */
    I1[idx] = (code == 1) ? 1.0f : 0.0f;
    I2[idx] = (code == 2) ? 1.0f : 0.0f;
}
'''

assert _LD_PLANES_SRC.isascii(), "kernel source must be pure ASCII (cf. 34b4a59)"

_LD_DOSAGE_KERNEL = None

# Fast path for r / r^2 / signed r^2. Those need only S = sum(g_i g_j) and the
# per-variant moments -- NOT the 3x3 contingency table. Building the full table
# costs nine B x B arrays plus ~8 elementwise passes to derive the outer cells,
# i.e. roughly 9x the memory traffic, to produce something one GEMM already
# gives. D and D' still need the table; r does not.
_LD_DOSAGE_SRC = r'''
extern "C" __global__
void build_dosage_planes(const unsigned char* packed,
                         float* G, float* G2, float* M,
                         const long long n_samples,
                         const long long n_variants,
                         const long long bytes_per_variant)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n_variants * n_samples;
    if (idx >= total) return;
    long long v = idx / n_samples;
    long long s = idx - v * n_samples;
    unsigned char byte = packed[v * bytes_per_variant + (s >> 2)];
    int code = (byte >> (6 - 2 * (s & 3))) & 3;
    float g = (code == 3) ? 0.0f : (float)code;   /* missing contributes 0 */
    G[idx]  = g;
    G2[idx] = g * g;
    M[idx]  = (code != 3) ? 1.0f : 0.0f;
}
'''

assert _LD_DOSAGE_SRC.isascii(), "kernel source must be pure ASCII (cf. 34b4a59)"


def _get_planes_kernel():
    global _LD_PLANES_KERNEL
    if _LD_PLANES_KERNEL is None and HAS_CUPY:
        _LD_PLANES_KERNEL = cp.RawKernel(_LD_PLANES_SRC, "build_ld_planes")
    return _LD_PLANES_KERNEL


_LD_EPILOGUE_KERNEL = None

# One fused kernel replacing ~15 CuPy elementwise launches, six B x B
# temporaries (vA, vB, ok, den, r, keep) and a blocking cp.nonzero per tile.
# Measured SM utilisation before this was 1-4%: the device was idle BETWEEN
# launches, not short of work. Survivors append straight into a global buffer
# via an atomic counter, so there is no per-tile sync and no per-tile
# concatenate either.
_LD_EPILOGUE_SRC = r'''
extern "C" __global__
void ld_epilogue_compact(
    const float* __restrict__ S,       /* (bi x bj) cross products     */
    const float* __restrict__ sA,      /* (bi,) per-variant sums       */
    const float* __restrict__ sB,      /* (bj,)                        */
    const float* __restrict__ qA,      /* (bi,) per-variant sum of sq  */
    const float* __restrict__ qB,      /* (bj,)                        */
    const float  nsamp,
    const long long i0, const long long j0,
    const long long bi, const long long bj,
    const long long window,            /* <= 0 means no index window   */
    const float min_r2,
    long long* __restrict__ out_i,
    long long* __restrict__ out_j,
    float* __restrict__ out_r,
    unsigned long long* __restrict__ counter,
    const long long capacity)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = bi * bj;
    if (t >= total) return;
    long long a = t / bj;
    long long b = t - a * bj;
    long long gi = i0 + a;
    long long gj = j0 + b;
    if (gj <= gi) return;                      /* upper triangle only  */
    if (window > 0 && (gj - gi) > window) return;

    double n  = (double)nsamp;
    double sa = (double)sA[a], sb = (double)sB[b];
    double qa = (double)qA[a], qb = (double)qB[b];
    double vA = n * qa - sa * sa;
    double vB = n * qb - sb * sb;
    if (!(vA > 0.0) || !(vB > 0.0)) return;    /* monomorphic          */

    double r = (n * (double)S[t] - sa * sb) / (sqrt(vA) * sqrt(vB));
    if (r >  1.0) r =  1.0;
    if (r < -1.0) r = -1.0;
    if (min_r2 > 0.0f && (float)(r * r) < min_r2) return;

    /* the counter always advances, so its final value is the true survivor
       count even when the buffer was too small -- the caller can resize and
       re-run only the epilogue, repeating no GEMM work. */
    unsigned long long slot = atomicAdd(counter, 1ULL);
    if (slot < (unsigned long long)capacity) {
        out_i[slot] = gi;
        out_j[slot] = gj;
        out_r[slot] = (float)r;
    }
}
'''

assert _LD_EPILOGUE_SRC.isascii(), "kernel source must be pure ASCII (cf. 34b4a59)"


def _get_epilogue_kernel():
    global _LD_EPILOGUE_KERNEL
    if _LD_EPILOGUE_KERNEL is None and HAS_CUPY:
        _LD_EPILOGUE_KERNEL = cp.RawKernel(_LD_EPILOGUE_SRC,
                                           "ld_epilogue_compact")
    return _LD_EPILOGUE_KERNEL


_LD_G_ONLY_KERNEL = None

# The fused scan consumes only G, but _build_dosage writes G, G2 and M -- three
# planes of B x n x 4 bytes when one is used. At n = 1e6 that is 12 GB written
# per tile to consume 4 GB, and plane construction is what bounds the run once
# n passes ~200k. s_v/q_v still need G2, but they are computed once in the
# streaming pre-pass, not per tile.
_LD_G_ONLY_SRC = r'''
extern "C" __global__
void build_g_plane(const unsigned char* packed, float* G,
                   const long long n_samples,
                   const long long n_variants,
                   const long long bytes_per_variant)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n_variants * n_samples;
    if (idx >= total) return;
    long long v = idx / n_samples;
    long long s = idx - v * n_samples;
    unsigned char byte = packed[v * bytes_per_variant + (s >> 2)];
    int code = (byte >> (6 - 2 * (s & 3))) & 3;
    G[idx] = (code == 3) ? 0.0f : (float)code;
}
'''

assert _LD_G_ONLY_SRC.isascii(), "kernel source must be pure ASCII (cf. 34b4a59)"

# hap2bit stores haplotype j in byte j>>3 at bit 7-(j&7) (see cugen/write.py).
# For j = 2i and j = 2i+1 those are the high and low bits of sample i's 2-bit
# field -- the same bytes as the dosage plane, read one bit at a time.
_LD_H_ONLY_SRC = r'''
extern "C" __global__
void build_h_plane(const unsigned char* packed, float* H,
                   const long long n_haps,
                   const long long n_variants,
                   const long long bytes_per_variant)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n_variants * n_haps;
    if (idx >= total) return;
    long long v = idx / n_haps;
    long long j = idx - v * n_haps;
    unsigned char byte = packed[v * bytes_per_variant + (j >> 3)];
    H[idx] = (float)((byte >> (7 - (j & 7))) & 1);
}

extern "C" __global__
void hap_moments(const unsigned char* packed, float* s_v, float* q_v,
                 const long long n_haps,
                 const long long n_variants,
                 const long long bytes_per_variant)
{
    long long v = blockIdx.x;
    if (v >= n_variants) return;
    __shared__ long long acc[256];
    long long local = 0;
    for (long long j = threadIdx.x; j < n_haps; j += blockDim.x) {
        unsigned char byte = packed[v * bytes_per_variant + (j >> 3)];
        local += (byte >> (7 - (j & 7))) & 1;
    }
    acc[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) acc[threadIdx.x] += acc[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // 0/1 alleles: sum(x*x) == sum(x), which is why the dosage epilogue
        // works unchanged on a haplotype plane.
        s_v[v] = (float)acc[0];
        q_v[v] = (float)acc[0];
    }
}
'''

assert _LD_H_ONLY_SRC.isascii(), "kernel source must be pure ASCII (cf. 34b4a59)"
_LD_H_ONLY_KERNEL = None
_LD_H_MOMENTS_KERNEL = None


def _get_h_only_kernel():
    global _LD_H_ONLY_KERNEL
    if _LD_H_ONLY_KERNEL is None and HAS_CUPY:
        _LD_H_ONLY_KERNEL = cp.RawKernel(_LD_H_ONLY_SRC, "build_h_plane")
    return _LD_H_ONLY_KERNEL


def _get_h_moments_kernel():
    global _LD_H_MOMENTS_KERNEL
    if _LD_H_MOMENTS_KERNEL is None and HAS_CUPY:
        _LD_H_MOMENTS_KERNEL = cp.RawKernel(_LD_H_ONLY_SRC, "hap_moments")
    return _LD_H_MOMENTS_KERNEL


def _build_h(packed2d, lo, hi, n_haps, bytes_per_variant, out=None):
    """Haplotype plane (0/1) for the CONTIGUOUS row range [lo, hi).

    The dosage twin of this is _build_g; the same slice-not-fancy-index rule
    applies and for the same measured reason.
    """
    b, nh = int(hi - lo), int(n_haps)
    blk = packed2d[lo:hi].ravel()                  # view, not a copy
    H = out if out is not None else cp.empty((b, nh), dtype=cp.float32)
    tpb = 256
    total = b * nh
    _get_h_only_kernel()(((total + tpb - 1) // tpb,), (tpb,),
                         (blk, H, np.int64(nh), np.int64(b),
                          np.int64(bytes_per_variant)))
    return H[:b] if out is not None else H


def _hap_moments(packed, p, nh, bpv):
    """Per-variant allele count; q_v == s_v because the alleles are 0/1."""
    s_v = cp.empty(p, dtype=cp.float32)
    q_v = cp.empty(p, dtype=cp.float32)
    _get_h_moments_kernel()((int(p),), (256,),
                            (packed, s_v, q_v, np.int64(nh), np.int64(p),
                             np.int64(bpv)))
    return s_v, q_v


def _get_g_only_kernel():
    global _LD_G_ONLY_KERNEL
    if _LD_G_ONLY_KERNEL is None and HAS_CUPY:
        _LD_G_ONLY_KERNEL = cp.RawKernel(_LD_G_ONLY_SRC, "build_g_plane")
    return _LD_G_ONLY_KERNEL


def _build_g(packed2d, lo, hi, n_samples, bytes_per_variant, out=None):
    """Dosage plane for the CONTIGUOUS row range [lo, hi).

    Takes a range rather than an index array on purpose. The rows are always
    contiguous here, but packed2d[cp.arange(lo, hi)] is a fancy-index gather
    that copies the whole packed slice -- measured at 40.6 ms against 18.3 ms
    for the unpack kernel itself, which already runs at 99% of the pure-fill
    bandwidth floor. A slice is a zero-copy view.

    `out` reuses a caller-owned buffer; allocating a fresh B x n plane per
    tile was costing about as much again as the kernel.
    """
    b, ns = int(hi - lo), int(n_samples)
    blk = packed2d[lo:hi].ravel()                  # view, not a copy
    G = out if out is not None else cp.empty((b, ns), dtype=cp.float32)
    tpb = 256
    total = b * ns
    _get_g_only_kernel()(((total + tpb - 1) // tpb,), (tpb,),
                         (blk, G, np.int64(ns), np.int64(b),
                          np.int64(bytes_per_variant)))
    return G[:b] if out is not None else G


def _get_dosage_kernel():
    global _LD_DOSAGE_KERNEL
    if _LD_DOSAGE_KERNEL is None and HAS_CUPY:
        _LD_DOSAGE_KERNEL = cp.RawKernel(_LD_DOSAGE_SRC, "build_dosage_planes")
    return _LD_DOSAGE_KERNEL


def _build_dosage(packed2d, rows, n_samples, bytes_per_variant):
    """(G, G2, M) planes for a block of variant rows."""
    blk = packed2d[rows].ravel()
    b, ns = int(len(rows)), int(n_samples)
    G = cp.empty((b, ns), dtype=cp.float32)
    G2 = cp.empty((b, ns), dtype=cp.float32)
    M = cp.empty((b, ns), dtype=cp.float32)
    tpb = 256
    total = b * ns
    _get_dosage_kernel()(((total + tpb - 1) // tpb,), (tpb,),
                         (blk, G, G2, M, np.int64(ns), np.int64(b),
                          np.int64(bytes_per_variant)))
    return G, G2, M


def _r_block(pl_a, pl_b, n_samples, has_missing):
    """Pairwise-complete r for one tile, without materialising the 3x3 table.

    1 GEMM when the file has no missing calls, 4 when it does (S, n, and the
    two co-observed moment products; their B-side twins come free by
    transpose). Returns (r, n_obs).
    """
    Ga, G2a, Ma = pl_a
    Gb, G2b, Mb = pl_b
    ns = float(n_samples)
    S = Ga @ Gb.T
    if has_missing:
        n = Ma @ Mb.T
        sA = Ga @ Mb.T            # sum of g_i over samples co-observed with j
        sB = (Ma @ Gb.T)
        qA = G2a @ Mb.T
        qB = (Ma @ G2b.T)
    else:
        n = cp.float32(ns)
        sA = cp.broadcast_to(Ga.sum(axis=1)[:, None], S.shape)
        sB = cp.broadcast_to(Gb.sum(axis=1)[None, :], S.shape)
        qA = cp.broadcast_to(G2a.sum(axis=1)[:, None], S.shape)
        qB = cp.broadcast_to(G2b.sum(axis=1)[None, :], S.shape)
    vA = n * qA - sA * sA
    vB = n * qB - sB * sB
    ok = (vA > 0) & (vB > 0)
    if has_missing:
        ok &= n > 0
    den = cp.sqrt(cp.where(ok, vA, 1.0)) * cp.sqrt(cp.where(ok, vB, 1.0))
    r = cp.where(ok, (n * S - sA * sB) / den, cp.nan)
    # With no missingness every pair sees every sample, so return the scalar
    # rather than materialising a B x B array of a constant.
    return cp.clip(r, -1.0, 1.0), (n if has_missing else float(ns))


def _build_planes(packed2d, rows, n_samples, bytes_per_variant):
    """Indicator planes (M, I1, I2) for a block of variant rows, on device."""
    blk = packed2d[rows].ravel()
    b = int(len(rows))
    ns = int(n_samples)
    M = cp.empty((b, ns), dtype=cp.float32)
    I1 = cp.empty((b, ns), dtype=cp.float32)
    I2 = cp.empty((b, ns), dtype=cp.float32)
    tpb = 256
    total = b * ns
    _get_planes_kernel()(((total + tpb - 1) // tpb,), (tpb,),
                         (blk, M, I1, I2, np.int64(ns), np.int64(b),
                          np.int64(bytes_per_variant)))
    return M, I1, I2


def _counts_block(pl_a, pl_b, n_samples, has_missing):
    """(3, 3, ba, bb) exact integer counts for one row-block x col-block tile.

    3 GEMMs when the file has no missing calls (n21 = n12^T and the marginals
    are per-variant constants), 6 when it does.
    """
    Ma, I1a, I2a = pl_a
    Mb, I1b, I2b = pl_b
    ba, bb = I1a.shape[0], I1b.shape[0]
    ns = int(n_samples)

    n11 = I1a @ I1b.T
    n22 = I2a @ I2b.T
    n12 = I1a @ I2b.T
    n21 = I2a @ I1b.T
    if has_missing:
        nn = Ma @ Mb.T
        rr1 = I1a @ Mb.T
        rr2 = I2a @ Mb.T
        cc1 = Ma @ I1b.T
        cc2 = Ma @ I2b.T
    else:
        nn = cp.full((ba, bb), float(ns), dtype=cp.float32)
        rr1 = cp.broadcast_to(I1a.sum(axis=1)[:, None], (ba, bb))
        rr2 = cp.broadcast_to(I2a.sum(axis=1)[:, None], (ba, bb))
        cc1 = cp.broadcast_to(I1b.sum(axis=1)[None, :], (ba, bb))
        cc2 = cp.broadcast_to(I2b.sum(axis=1)[None, :], (ba, bb))

    out = cp.empty((3, 3, ba, bb), dtype=cp.float32)
    out[1, 1], out[1, 2], out[2, 1], out[2, 2] = n11, n12, n21, n22
    out[1, 0] = rr1 - n11 - n12
    out[2, 0] = rr2 - n21 - n22
    out[0, 1] = cc1 - n11 - n21
    out[0, 2] = cc2 - n12 - n22
    out[0, 0] = nn - (out[1, 1] + out[1, 2] + out[2, 1] + out[2, 2]
                      + out[1, 0] + out[2, 0] + out[0, 1] + out[0, 2])
    return out


def _r_from_counts_gpu(c):
    """Cheap on-device r, used only to threshold before the host transfer."""
    n = c.sum(axis=(0, 1))
    sA = c[1].sum(axis=0) + 2.0 * c[2].sum(axis=0)
    sB = c[:, 1].sum(axis=0) + 2.0 * c[:, 2].sum(axis=0)
    qA = c[1].sum(axis=0) + 4.0 * c[2].sum(axis=0)
    qB = c[:, 1].sum(axis=0) + 4.0 * c[:, 2].sum(axis=0)
    S = c[1, 1] + 2.0 * c[1, 2] + 2.0 * c[2, 1] + 4.0 * c[2, 2]
    vA, vB = n * qA - sA * sA, n * qB - sB * sB
    ok = (n > 0) & (vA > 0) & (vB > 0)
    den = cp.sqrt(cp.where(ok, vA, 1.0)) * cp.sqrt(cp.where(ok, vB, 1.0))
    r = cp.where(ok, (n * S - sA * sB) / den, cp.nan)
    return cp.clip(r, -1.0, 1.0), n


def _tile_size_for(n_samples: int, budget_frac: float = 0.35,
                   window: Optional[int] = None, fused: bool = False) -> int:
    """Pick B so planes + counts fit free device memory.

    planes  = 2 * 3 * B * n_samples * 4
    counts  = 9 * B * B * 4
    Peak is a function of B and n_samples only -- NOT of the variant count.

    When a narrow `window` is in play, a large B is actively harmful: a tile
    evaluates B x (B + window) cells but only B * window of them can survive
    the band, so useful work is window / (B + window). At B = 8192 and
    window = 500 that is 5.8% -- which is what made the windowed path ~19x
    slower than plink2. Sizing B to the window lifts it to ~50%, at the cost
    of smaller (but still tensor-core-sized) GEMMs.
    """
    free, _total = cp.cuda.Device().mem_info
    budget = budget_frac * free
    # Coefficients must match what the chosen path actually allocates.
    #   fused r-only : 2 plane buffers (2*4) + the S tile (4)
    #   table path   : 2 blocks x 3 planes (24) + 9 count arrays (36)
    # Modelling the table path's footprint while running the fused path
    # under-sizes B by ~3x, which costs both per-tile overhead and GEMM
    # efficiency.
    a, b = (4.0, 8.0 * n_samples) if fused else (36.0, 24.0 * n_samples)
    B = int((-b + (b * b + 4 * a * budget) ** 0.5) / (2 * a))
    # The old 8192 ceiling was set when a tile cost 9 count arrays. With one
    # plane and one S tile, larger blocks fit easily at small n -- which is
    # exactly where the run is overhead-bound and wants fewer, bigger GEMMs.
    ceiling = 32768 if fused else 8192
    B = max(256, min(ceiling, (B // 256) * 256))
    if window is not None:
        # round the window up to a multiple of 256, and never grow B
        wb = max(256, ((int(window) + 255) // 256) * 256)
        B = min(B, wb)
    return B


@dataclass(frozen=True)
class LDMatrix:
    """Dense LD result for one contiguous variant selection on one chromosome."""
    r: Optional[np.ndarray]
    r2: Optional[np.ndarray]
    r2_signed: Optional[np.ndarray]
    d: Optional[np.ndarray]
    d_prime: Optional[np.ndarray]
    n_obs: np.ndarray
    gidx: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    maf: np.ndarray
    n_samples: int
    dprime_method: str
    missing: str

    def __repr__(self) -> str:
        p = len(self.gidx)
        return (f"<LDMatrix p={p} n_samples={self.n_samples} "
                f"dprime={self.dprime_method} missing={self.missing}>")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _r_only_result(r_arr, n_arr, reader, rows, pairs_local):
    """Shape the r-only fast path into the same dict ld_from_counts returns.

    D/D' are NaN here by construction -- this path is only taken when neither
    was requested. pA/pB come from the header's per-variant mean dosage, which
    is the right quantity for allele ORIENTATION (a per-variant property, and
    what plink2 orients on) even though it is not the per-pair co-observed
    frequency.
    """
    r = np.asarray(r_arr, dtype=np.float64)
    af = np.asarray(reader.mu_x, dtype=np.float64)[rows] / 2.0
    # No NaN d/dp arrays: this path is only taken when neither was requested,
    # and allocating two full-length arrays of NaN to immediately drop them
    # costs real time at 5M rows.
    return {"n": np.asarray(n_arr, dtype=np.float64),
            "pA": af[pairs_local[:, 0]], "pB": af[pairs_local[:, 1]],
            "r": r, "r2": np.clip(r * r, 0.0, 1.0),
            "r2_signed": np.clip(r * np.abs(r), -1.0, 1.0)}


def _scan_gpu_fused(reader, rows, window, min_r2, tile_size=None,
                    count_only=False, on_flush=None, flush_rows=None,
                    verbose=False, tf32=False, phased=False):
    """Fused scan: one kernel per tile, one output buffer, no per-tile sync.

    Only for the clean r-only case (no missingness, no bp window). Everything
    else falls back to _scan_gpu. Returns device arrays (idx_i, idx_j, r).
    """
    # A phased file contributes 2 haplotype columns per sample; everything
    # downstream (tile planner, GEMM, epilogue) is a function of that width
    # only, so the phased path differs solely in ns and the plane builder.
    ns = 2 * int(reader.n_samples) if phased else int(reader.n_samples)
    build_plane = _build_h if phased else _build_g
    bpv = int(reader.bytes_per_variant)
    p = len(rows)
    B = int(tile_size) if tile_size else _tile_size_for(
        ns, window=window, fused=True)
    # Never size a tile larger than the problem. The planner bounds B by
    # MEMORY, not by p, so at p=4,000 with n=100,000 it happily picked
    # B=31,744 and then allocated a 31,744 x 100,000 plane buffer to hold
    # 4,000 rows -- ~8x over-allocation, invisible whenever p >> B.
    B = max(256, min(B, p))
    packed = _load_packed_rows(reader, rows, bpv, verbose)

    # Per-variant moments straight from the packed bytes. This previously
    # streamed chunks through _build_dosage, which materialises G, G2 and M --
    # three fp32 planes -- to produce two per-variant sums. Chunking bounded
    # the PEAK but not the total work: ~120 GB of plane writes at p=20,000,
    # n=500,000 to compute 160 KB of output, and it dominated the run at large
    # n. The kernel reads the 2-bit data once, writes only the sums, and stays
    # bit-exact by accumulating in integers.
    s_v, q_v = (_hap_moments(packed, p, ns, bpv) if phased
                else _variant_moments(packed, p, ns, bpv))

    def run(capacity, on_flush=None, tile_max=0):
        """Scan every tile. With on_flush, drain the buffer at tile
        boundaries so it can never overflow; without it, fill one buffer."""
        flushed = 0
        # Set the cuBLAS math mode ONCE for the whole scan. Doing it per GEMM
        # cost two cuBLAS calls per tile, which showed up as TF32 being
        # *slower* than fp32 at large n where tiles are many.
        out_i = cp.empty(capacity, dtype=cp.int64)
        out_j = cp.empty(capacity, dtype=cp.int64)
        out_r = cp.empty(capacity, dtype=cp.float32)
        counter = cp.zeros(1, dtype=cp.uint64)
        kern = _get_epilogue_kernel()
        # Two reusable plane buffers for the whole scan. A fresh B x n
        # allocation per tile cost roughly as much as the unpack kernel.
        bufA = cp.empty((B, ns), dtype=cp.float32)
        bufB = cp.empty((B, ns), dtype=cp.float32)
        with _Tf32(tf32):
            for i0 in range(0, p, B):
                i1 = min(i0 + B, p)
                hi = p if window is None else min(p, i1 - 1 + int(window) + 1)
                Ga = build_plane(packed, i0, i1, ns, bpv, out=bufA)
                for j0 in range(i0, hi, B):
                    j1 = min(j0 + B, hi)
                    if j0 == i0:
                        Gb = Ga
                    else:
                        Gb = build_plane(packed, j0, j1, ns, bpv, out=bufB)
                    S = Ga @ Gb.T
                    bi, bj = i1 - i0, j1 - j0
                    nthread = bi * bj
                    # Reserving B*B up front is unaffordable: B reaches the
                    # planner's 32,768 ceiling at small n, so 2*B*B is 2.1e9
                    # rows (~43 GB) -- worse than the buffer it replaced.
                    # Instead keep the buffer at most half full going in, and
                    # handle the rare tile that still overflows below.
                    held = 0
                    if on_flush is not None:
                        held = int(counter[0])
                        if held > capacity // 2:
                            on_flush(out_i[:held], out_j[:held], out_r[:held])
                            flushed += held
                            counter.fill(0)
                            held = 0
                    kern(((nthread + 255) // 256,), (256,),
                         (S, s_v[i0:i1], s_v[j0:j1], q_v[i0:i1], q_v[j0:j1],
                          np.float32(ns), np.int64(i0), np.int64(j0),
                          np.int64(bi), np.int64(bj),
                          np.int64(window if window else 0), np.float32(min_r2),
                          out_i, out_j, out_r, counter, np.int64(capacity)))
                    if on_flush is not None:
                        after = int(counter[0])
                        if after > capacity:
                            # This tile alone overflowed a half-empty buffer.
                            # atomicAdd is unconditional, so `after` is exact
                            # even though writes past capacity were dropped --
                            # and S is still live, so only the epilogue is
                            # re-run, never the GEMM. That is the difference
                            # between retrying one tile and retrying O(p^2).
                            need = after - held
                            if held:
                                on_flush(out_i[:held], out_j[:held],
                                         out_r[:held])
                                flushed += held
                            counter.fill(0)
                            t_i = cp.empty(need, dtype=cp.int64)
                            t_j = cp.empty(need, dtype=cp.int64)
                            t_r = cp.empty(need, dtype=cp.float32)
                            kern(((nthread + 255) // 256,), (256,),
                                 (S, s_v[i0:i1], s_v[j0:j1], q_v[i0:i1],
                                  q_v[j0:j1], np.float32(ns), np.int64(i0),
                                  np.int64(j0), np.int64(bi), np.int64(bj),
                                  np.int64(window if window else 0),
                                  np.float32(min_r2), t_i, t_j, t_r, counter,
                                  np.int64(need)))
                            got = int(counter[0])
                            on_flush(t_i[:got], t_j[:got], t_r[:got])
                            flushed += got
                            del t_i, t_j, t_r
                            counter.fill(0)
                    del S
        del bufA, bufB
        if on_flush is not None:
            held = int(counter[0])
            if held:
                on_flush(out_i[:held], out_j[:held], out_r[:held])
                flushed += held
            return None, None, None, flushed
        return out_i, out_j, out_r, int(counter[0])

    if on_flush is not None:
        # A tile emits at most B*B rows, so a buffer of at least 2*B*B drained
        # whenever the count comes within B*B of capacity can never overflow.
        # Overflow becomes structurally impossible rather than merely
        # unlikely, which is what retires the re-run-everything retry.
        # Sized from a memory budget, NOT from B: 8.4e6 rows is ~168 MB at
        # 20 B/row and is independent of both p and the tile size. An
        # oversized tile is handled by the exact epilogue retry above.
        cap = max(1 << 16, int(flush_rows) if flush_rows else (1 << 23))
        _a, _b, _c, total = run(cap, on_flush=on_flush, tile_max=0)
        del packed
        cp.get_default_memory_pool().free_all_blocks()
        return int(total)

    # count_only: the kernel's atomicAdd is unconditional and it only writes
    # when slot < capacity, so a zero-capacity run does every GEMM, writes
    # nothing, and still counts exactly. That is the only way to measure
    # genome-scale compute today -- chr22's full variant set alone emits
    # 187,252,868 rows, and genome-wide extrapolates to ~630 GB of output
    # arrays, far past both this buffer and the card.
    if count_only:
        _i, _j, _r, found = run(0)
        del _i, _j, _r, packed
        cp.get_default_memory_pool().free_all_blocks()
        return int(found)

    # optimistic capacity, then one exact retry if it overflowed
    cap = min(int(200e6), max(1 << 20, p * (int(window) if window else 4096)))
    out_i, out_j, out_r, found = run(cap)
    if found > cap:
        if verbose:
            print(f"cugen.ld: buffer held {cap:,}, {found:,} survived -- "
                  f"re-running the epilogue at exact size")
        del out_i, out_j, out_r
        cp.get_default_memory_pool().free_all_blocks()
        out_i, out_j, out_r, found = run(found)
    del packed
    cp.get_default_memory_pool().free_all_blocks()
    return out_i[:found], out_j[:found], out_r[:found]


def _assemble_device(pairs_dev, payload, reader, rows, stats, sign_reference,
                     path, verbose, n_planned=0):
    """Build a cudf.DataFrame straight from device arrays -- no host copy.

    Only reachable for the r-family stats with no annotation, which is the
    case where the output is large enough for the transfer to dominate.
    """
    r_dev, n_dev = payload
    rows_dev = cp.asarray(rows)
    ia = rows_dev[pairs_dev[:, 0]]
    ib = rows_dev[pairs_dev[:, 1]]
    gidx_dev = cp.asarray(np.asarray(reader.gidx, dtype=np.int64))
    maf_dev = cp.asarray(np.asarray(reader.maf, dtype=np.float32))

    r = r_dev
    if sign_reference == "major":
        af = cp.asarray(np.asarray(reader.mu_x, dtype=np.float32)) / 2.0
        flip = (af[ia] > 0.5) ^ (af[ib] > 0.5)
        r = cp.where(flip, -r, r)
    r2 = cp.clip(r * r, 0.0, 1.0)

    chrom = 0
    m = re.search(r"chr(\d+)", os.path.basename(path))
    if m:
        chrom = int(m.group(1))

    cols = {"CHR_A": cp.full(r.size, chrom, dtype=cp.int32),
            "POS_A": cp.zeros(r.size, dtype=cp.int64)}
    g = cudf.DataFrame(cols)
    g["ID_A"] = "."
    g["MAF_A"] = maf_dev[ia]
    g["CHR_B"] = cp.full(r.size, chrom, dtype=cp.int32)
    g["POS_B"] = cp.zeros(r.size, dtype=cp.int64)
    g["ID_B"] = "."
    g["MAF_B"] = maf_dev[ib]
    g["N_OBS"] = n_dev.astype(cp.int32)
    if "r" in stats:
        g["R"] = r.astype(cp.float32)
    if "r2" in stats:
        g["R2"] = r2.astype(cp.float32)
    if "r2_signed" in stats:
        g["R2_SIGNED"] = cp.clip(r * cp.abs(r), -1.0, 1.0).astype(cp.float32)
    # the fused epilogue returns one correlation; which column it IS depends on
    # whether the plane it ran over was dosages or haplotypes
    if "r_phased" in stats:
        g["R_PHASED"] = r.astype(cp.float32)
    if "r2_phased" in stats:
        g["R2_PHASED"] = r2.astype(cp.float32)
    # chi2 = N_OBS * r2, 1 df. n_dev already holds haplotypes for a phased scan
    # and individuals for a dosage one (see ld_matrix, where it is built).
    if _SIG_STATS & set(stats):
        chi2 = n_dev.astype(cp.float64) * r2
        if "chi2" in stats:
            g["CHI2"] = chi2.astype(cp.float32)
        if "p" in stats:
            g["NEG_LOG10_P"] = _neglog10_chi2_1df(chi2, cp).astype(cp.float32)
    g["gidx_a"] = gidx_dev[ia]
    g["gidx_b"] = gidx_dev[ib]
    return g


def _haplotypes_numpy(reader) -> np.ndarray:
    """(n_variants, 2*n_samples) uint8 0/1 alleles, no CuPy required."""
    from .write import unpack_hap2bit
    packed = np.frombuffer(reader.read_packed_bytes(), dtype=np.uint8)
    bpv = int(reader.bytes_per_variant)
    p, H = int(reader.n_variants), 2 * int(reader.n_samples)
    return np.stack([unpack_hap2bit(packed[v * bpv:(v + 1) * bpv], H)
                     for v in range(p)])


def phased_from_haplotypes(hap, pairs):
    """r and r^2 from 0/1 haplotype rows -- no EM, no cubic.

    `hap` is (n_variants, H) of 0/1; `pairs` is (n_pairs, 2) of row indices.

    For 0/1 indicators sum(x^2) == sum(x), so the Hill & Robertson allele-count
    correlation collapses to the haplotypic identity exactly:

        r = (H*nAB - nA*nB) / sqrt(nA(H-nA) * nB(H-nB))
          = D / sqrt(pA qA pB qB),   D = nAB/H - pA pB

    A HAP2BIT file cannot encode missingness, so every pair is complete and
    n == H unconditionally.
    """
    h = np.asarray(hap)
    H = float(h.shape[1])
    x = h[pairs[:, 0]].astype(np.float64)
    y = h[pairs[:, 1]].astype(np.float64)
    nA, nB = x.sum(axis=1), y.sum(axis=1)
    nAB = (x * y).sum(axis=1)
    pA, pB = nA / H, nB / H
    vA, vB = nA * (H - nA), nB * (H - nB)
    with np.errstate(divide="ignore", invalid="ignore"):
        good = (vA > 0) & (vB > 0)
        r = np.clip(np.where(good, (H * nAB - nA * nB)
                             / (np.sqrt(vA) * np.sqrt(vB)), np.nan), -1.0, 1.0)
        D = nAB / H - pA * pB
        qA, qB = 1.0 - pA, 1.0 - pB
        # Lewontin (1964): normalise by the largest |D| the margins permit.
        # The branch is on the SIGN of D, not on which product is smaller.
        dmax = np.where(D > 0, np.minimum(pA * qB, qA * pB),
                        np.minimum(pA * pB, qA * qB))
        dp = np.where(good & (dmax > 0), D / dmax, np.nan)
    return {"r_phased": r, "r2_phased": r * r, "d_phased": np.where(good, D, np.nan),
            "dp_phased": dp, "pA": pA, "pB": pB,
            "n": np.full(pairs.shape[0], H),
            # the 2x2 haplotype table, for the exact conditional test
            "nAB": nAB, "nA": nA, "nB": nB}


def _dosages_numpy(reader) -> np.ndarray:
    """(n_variants, n_samples) uint8 with 3 = missing, no CuPy required."""
    from .write import unpack_2bit
    packed = np.frombuffer(reader.read_packed_bytes(), dtype=np.uint8)
    bpv = int(reader.bytes_per_variant)
    p, ns = int(reader.n_variants), int(reader.n_samples)
    return np.stack([unpack_2bit(packed[v * bpv:(v + 1) * bpv], ns)
                     for v in range(p)])


def _scan_gpu(reader, rows, window, window_kb, positions, min_r2, min_obs,
              tile_size=None, verbose=False, need_table=True,
              keep_device=False):
    """Tiled upper-triangle scan. Returns (pairs_local, tables) for survivors.

    Peak device memory is a function of tile size and sample count only, never
    of the number of variants: each (row-block, col-block) tile allocates its
    own planes and counts and frees them before the next.
    """
    ns = int(reader.n_samples)
    if ns > _FP32_EXACT_MAX_SAMPLES:
        raise ValueError(
            f"n_samples={ns:,} exceeds {_FP32_EXACT_MAX_SAMPLES:,}, above which "
            f"fp32 accumulation of 4*n is no longer exact. Refusing rather than "
            f"returning quietly-wrong counts.")
    bpv = int(reader.bytes_per_variant)
    p = len(rows)
    has_missing = bool(reader.has_missing)
    B = int(tile_size) if tile_size else _tile_size_for(ns, window=window)
    build = _build_planes if need_table else _build_dosage

    packed = _load_packed_rows(reader, rows, bpv, verbose)

    out_pairs, out_tabs, out_r, out_n = [], [], [], []
    for i0 in range(0, p, B):
        i1 = min(i0 + B, p)
        pl_a = build(packed, cp.arange(i0, i1), ns, bpv)
        # column extent: upper triangle, bounded by whichever windows are set
        hi = p
        if window is not None:
            hi = min(hi, i1 - 1 + int(window) + 1)
        if window_kb is not None:
            hi = min(hi, int(np.searchsorted(
                positions, positions[i1 - 1] + int(round(window_kb * 1000)),
                side="right")))
        for j0 in range(i0, hi, B):
            j1 = min(j0 + B, hi)
            pl_b = pl_a if j0 == i0 else build(
                packed, cp.arange(j0, j1), ns, bpv)
            if need_table:
                c = _counts_block(pl_a, pl_b, ns, has_missing)
                r, n = _r_from_counts_gpu(c)
            else:
                c = None
                r, n = _r_block(pl_a, pl_b, ns, has_missing)

            ii = cp.arange(i0, i1)[:, None]
            jj = cp.arange(j0, j1)[None, :]
            keep = (jj > ii) & cp.isfinite(r) & (n >= min_obs)
            if window is not None:
                keep &= (jj - ii) <= int(window)
            if window_kb is not None:
                pos = cp.asarray(positions)
                keep &= (pos[jj] - pos[ii]) <= int(round(window_kb * 1000))
            if min_r2 > 0:
                keep &= (r * r) >= min_r2

            idx = cp.nonzero(keep)
            if idx[0].size:
                if need_table:
                    tabs = cp.empty((idx[0].size, 3, 3), dtype=cp.float32)
                    for a in range(3):
                        for b in range(3):
                            tabs[:, a, b] = c[a, b][idx]
                    out_tabs.append(cp.asnumpy(tabs))
                elif keep_device:
                    # stay on the device: the survivors are already here, and
                    # cudf can wrap them without a copy
                    out_r.append(r[idx])
                    out_n.append(n[idx] if has_missing else
                                 cp.full(idx[0].size, float(ns), dtype=cp.float32))
                else:
                    out_r.append(cp.asnumpy(r[idx]))
                    out_n.append(cp.asnumpy(
                        n[idx] if has_missing
                        else cp.full(idx[0].size, float(ns), dtype=cp.float32)))
                if keep_device and not need_table:
                    out_pairs.append(cp.stack([idx[0] + i0, idx[1] + j0], axis=1))
                else:
                    out_pairs.append(np.stack([cp.asnumpy(idx[0]) + i0,
                                               cp.asnumpy(idx[1]) + j0], axis=1))
            del c, r, n, keep, idx
            if pl_b is not pl_a:
                del pl_b
        del pl_a
        cp.get_default_memory_pool().free_all_blocks()

    del packed
    cp.get_default_memory_pool().free_all_blocks()
    if not out_pairs:
        empty = np.zeros((0, 2), dtype=np.int64)
        return (empty, np.zeros((0, 3, 3)) if need_table
                else (np.zeros(0), np.zeros(0)))
    if keep_device and not need_table:
        return (cp.concatenate(out_pairs),
                (cp.concatenate(out_r), cp.concatenate(out_n)))
    pairs = np.concatenate(out_pairs)
    if need_table:
        return pairs, np.concatenate(out_tabs)
    return pairs, (np.concatenate(out_r), np.concatenate(out_n))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ld_matrix(
    cugen: Union[str, Path],
    *,
    variant_range: Optional[Tuple[int, int]] = None,
    variants=None,
    region: Optional[str] = None,
    maf_min: float = 0.0,
    window: Optional[int] = None,
    window_kb: Optional[float] = None,
    min_r2: float = 0.0,
    max_p: Optional[float] = None,
    correction: Optional[str] = None,
    alpha: float = 0.05,
    exact: str = "never",
    lambda_gc: bool = False,
    kinship=None,
    structure=None,
    stats: Sequence[str] = _DEFAULT_STATS,
    dprime_method: str = "phased",
    sign_reference: str = "alt",
    missing: str = "pairwise",
    min_obs: int = 2,
    output_format: str = "pairs",
    annotation=None,
    output: Optional[Union[str, Path]] = None,
    backend: str = "auto",
    tile_size: Optional[int] = None,
    precision: str = "auto",
    max_pairs: int = 100_000_000,
    device: int = 0,
    count_only: bool = False,
    stream: bool = False,
    flush_rows: Optional[int] = None,
    verbose: bool = True,
):
    """Pairwise LD (r, r^2, signed r^2, D, D') from a .cugen file.

    See the module docstring for the phase caveat on D/D', the pairwise
    complete-case missingness rule, and plink2 parity.

    Parameters
    ----------
    cugen
        Path to a single ``.cugen`` file.
    variant_range
        ``(start, end)`` half-open LOCAL ROW indices into the file. Note these
        are row indices, not gidx: gidx is global, rows are local.
    variants
        Restrict to these gidx values. Sequence, ndarray, DataFrame with a
        ``gidx`` column, or a path to feather/TSV/CSV/NPZ.
    region
        ``'22:1000000-2000000'``, 1-based inclusive. Requires ``annotation``,
        since a .cugen carries no coordinates.
    maf_min
        Drop variants below this MAF, read from the header (no decode).
    window, window_kb
        Max separation in variant-index distance and in kb respectively. Both
        may be given; a pair must satisfy both, matching PLINK. ``window_kb``
        requires ``annotation``. Unlike PLINK these default to NO filter --
        silently truncating a result set is worse than an explicit error, and
        ``max_pairs`` catches the runaway case.
    min_r2
        Drop pairs below this r^2. PLINK's ``--ld-window-r2``.
    max_p
        Drop pairs whose p-value exceeds this. Requires ``'p'`` in ``stats``,
        and is mutually exclusive with ``min_r2``: with N constant across pairs
        the two are the same filter, so this is converted to the equivalent
        ``min_r2`` and costs nothing.
    correction
        ``None`` (default), ``'bonferroni'``, or ``'fdr'`` (Benjamini-Hochberg).
        Derives its own threshold from ``alpha`` and the number of tests, which
        is the pair count and is known in closed form before the scan. Requires
        ``'p'`` in ``stats``.
    alpha
        Family-wise error rate for ``'bonferroni'``, or the false discovery
        rate for ``'fdr'``. Default 0.05.
    exact
        ``'never'`` (default), ``'auto'`` or ``'always'``. Adds
        ``NEG_LOG10_P_EXACT``, a two-sided Fisher exact test on the 2x2
        haplotype table -- which IS the exact permutation p-value, since
        permuting haplotype labels leaves both allele counts fixed and so draws
        from the hypergeometric with those margins. hap2bit input only.
        ``'auto'`` computes it where the minimum expected cell count falls below
        5 and leaves NaN elsewhere, meaning the asymptotic p-value is the one to
        use for that pair. Selecting ``'p_exact'`` in ``stats`` implies
        ``'auto'``. Forces the reference path, as ``d``/``dp`` already do.
    lambda_gc
        Estimate the genomic-control inflation factor and add ``CHI2_ADJ`` /
        ``NEG_LOG10_P_ADJ`` beside the raw columns, with lambda itself in
        ``df.attrs['lambda_gc']``. Off by default. Per-pair p-values assume
        independent haplotypes, so population structure and cryptic relatedness
        make them anti-conservative: on two subpopulations differing by
        dAF = 0.6, lambda reaches ~920 and every pair looks genome-wide
        significant despite no gametic LD existing at all. Estimated over the
        more distant half of pairs and always before any filtering, since
        filtering selects the tail. Forces the reference path.
    kinship, structure
        Matrices for the bias-corrected measures ``r2_s`` / ``r2_v`` /
        ``r2_vs`` (Mangin et al. 2012): ``kinship`` is (n_samples, n_samples)
        genetic covariances between individuals, ``structure`` is
        (n_samples, K) -- admixture proportions or leading PCs. Both must be in
        the .cugen file's sample order. Dosage input only, and they force the
        reference path. These are ESTIMATORS: the paper derives no null
        distribution for them, so requesting ``chi2``/``p`` alongside raises
        rather than inventing a degrees-of-freedom.
    stats
        Any of ``r, r2, r2_signed, d, dp`` (dosage), ``r_phased, r2_phased,
        d_phased, dp_phased`` (true phase, hap2bit input), ``r2_phased_em`` (EM
        haplotype estimate from unphased input), or ``chi2, p`` (significance;
        valid on either encoding). Dropping ``d``/``dp`` skips the cubic
        entirely. Defaults to the dosage set, so adding a statistic here never
        changes an existing call's result.
    dprime_method
        ``'phased'`` (exact cubic, plink2 ``--r2-phased`` parity) or
        ``'composite'`` (Burrows' closed form; not a phase estimate).
    sign_reference
        ``'alt'`` (default, cugen-internal) or ``'major'`` (plink2 parity).
    missing
        ``'pairwise'`` complete-case (default). Only this is implemented.
    min_obs
        Drop pairs with fewer co-observed samples than this.
    output_format
        ``'pairs'`` -> DataFrame, or ``'matrix'`` -> :class:`LDMatrix`.
    backend
        ``'auto'``, ``'gpu'``, or ``'numpy'`` (reference path, no CuPy).

    Returns
    -------
    DataFrame with columns
    ``CHR_A POS_A ID_A MAF_A CHR_B POS_B ID_B MAF_B N_OBS [R R2 R2_SIGNED D DP
    R_PHASED R2_PHASED D_PHASED DP_PHASED R2_PHASED_EM CHI2 NEG_LOG10_P]
    gidx_a gidx_b``, or an :class:`LDMatrix`. Which statistic columns appear is
    exactly what ``stats`` asked for. ``NEG_LOG10_P`` carries -log10(p), not p:
    p itself underflows float32 at chi2 = 170, i.e. r^2 = 0.034 at 1000 Genomes
    sample size.
    """
    # ---- validation, all before any allocation --------------------------
    bad = [s for s in stats if s not in _STATS]
    if bad:
        raise ValueError(f"unknown stats {bad}; valid are {list(_STATS)}")
    if output_format not in ("pairs", "matrix"):
        raise ValueError(f"output_format must be 'pairs' or 'matrix', got {output_format!r}")
    if dprime_method not in ("phased", "composite"):
        raise ValueError(f"dprime_method must be 'phased' or 'composite', got {dprime_method!r}")
    if sign_reference not in ("alt", "major"):
        raise ValueError(f"sign_reference must be 'alt' or 'major', got {sign_reference!r}")
    if missing != "pairwise":
        raise NotImplementedError(
            f"missing={missing!r} is not implemented; only 'pairwise' "
            f"(complete-case) is available.")
    if backend not in ("auto", "gpu", "numpy"):
        raise ValueError(f"backend must be 'auto', 'gpu' or 'numpy', got {backend!r}")
    if correction is not None and correction not in ("bonferroni", "fdr"):
        raise ValueError(
            f"correction must be None, 'bonferroni' or 'fdr', got {correction!r}")
    if max_p is not None:
        if not (0.0 < float(max_p) <= 1.0):
            raise ValueError(f"max_p must be in (0, 1], got {max_p!r}")
        if min_r2 > 0:
            raise ValueError(
                "pass max_p or min_r2, not both -- with N constant they are the "
                "same filter, and two thresholds would silently take the "
                "stricter one.")
        if correction is not None:
            raise ValueError(
                f"pass max_p or correction={correction!r}, not both: a "
                f"correction derives its own threshold.")
    if (max_p is not None or correction is not None) and "p" not in stats:
        raise ValueError(
            "max_p/correction filter on the p-value, so 'p' must be in stats. "
            "Add it: stats=(..., 'p').")
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
    if lambda_gc and "p" not in stats:
        raise ValueError(
            "lambda_gc adjusts the p-value, so 'p' must be in stats. "
            "Add it: stats=(..., 'p').")
    want_corrected = [x for x in stats if x in _CORRECTED_STATS]
    if want_corrected and (_SIG_STATS & set(stats)):
        raise ValueError(
            f"{want_corrected} are bias-corrected ESTIMATORS with no null "
            f"distribution -- Mangin et al. (2012) prove them unbiased for "
            f"unlinked loci and link them to association power, but derive no "
            f"null sampling law, so chi2 = N * r^2 does not transfer to them. "
            f"Emitting a p-value beside them would mean inventing a "
            f"degrees-of-freedom. Request them on their own.")
    if exact not in _EXACT_MODES:
        raise ValueError(f"exact must be one of {_EXACT_MODES}, got {exact!r}")
    # exact= and stats=('p_exact',) are two doors to the same room: either one
    # implies the other, and asking via stats defaults the mode to 'auto'.
    stats = tuple(stats)
    if exact != "never" and "p_exact" not in stats:
        stats = stats + ("p_exact",)
    if lambda_gc:
        stats = stats + tuple(c for c in ("chi2_adj", "p_adj")
                              if c not in stats)
    elif exact == "never" and "p_exact" in stats:
        exact = "auto"

    path = str(cugen)
    if os.path.isdir(path):
        raise NotImplementedError(
            "directory input is not implemented yet; pass a single .cugen file")
    if not path.endswith(".cugen"):
        raise ValueError(
            f"{path!r} is not a .cugen file. cugen computes on .cugen only; "
            f"convert first, e.g.  cg.convert.vcf2cugen(in_vcf, out_cugen)  or "
            f"cugen-convert vcf in.vcf.gz out.cugen  (also pgen2cugen / "
            f"bed2cugen).")

    ann = None
    if annotation is not None:
        ann = (annotation if isinstance(annotation, pd.DataFrame)
               else pd.read_feather(str(annotation)))
    if (region is not None or window_kb is not None) and ann is None:
        raise ValueError("region= and window_kb= need coordinates; pass annotation=")
    if region is not None and variant_range is not None:
        raise ValueError("pass at most one of region= and variant_range=")

    use_gpu = HAS_CUPY if backend == "auto" else (backend == "gpu")
    if backend == "gpu" and not HAS_CUPY:
        raise RuntimeError("CuPy not available")
    if backend == "auto" and not HAS_CUPY:
        raise RuntimeError(
            "CuPy not available. Pass backend='numpy' for the CPU reference "
            "path (correct but not fast).")

    reader = read_cugen(path, device=device)
    want_phased = [s for s in stats if s in _PHASED_STATS]
    # chi2/p are ENCODING-NEUTRAL: the test is N_OBS * r^2 either way, and the
    # encoding only decides whether N_OBS counts gametes or individuals. They
    # must not fall into want_dosage, or asking for them on a hap2bit file
    # trips the cross-guard below.
    want_dosage = [s for s in stats
                   if s not in _PHASED_STATS and s not in _SIG_STATS]
    enc = int(reader.encoding)
    if enc == ENCODING_HAP2BIT:
        if want_dosage:
            raise ValueError(
                f"{path} is phased (hap2bit); {want_dosage} are dosage "
                f"statistics. hap2bit and 2bit share bytes but not meaning, so "
                f"serving them here would return plausible wrong numbers. Ask "
                f"for {sorted(_PHASED_STATS)} instead.")
    elif enc != ENCODING_2BIT:
        raise NotImplementedError(
            f"cugen.ld requires 2-bit encoding (encoding={ENCODING_2BIT}); this "
            f"file has encoding={enc}. Re-convert with cg.convert.")
    elif want_phased:
        raise ValueError(
            f"{want_phased} need TRUE phase, which a 2bit file does not carry. "
            f"Convert a phased VCF with cg.convert.vcf2cugenh().")
    if enc == ENCODING_HAP2BIT and want_corrected:
        raise ValueError(
            f"{want_corrected} are defined on GENOTYPES -- Mangin et al. build "
            f"the correction from a covariance matrix between individuals, and "
            f"the kinship/structure matrices are indexed by individual, not by "
            f"haplotype. Use a 2bit (dosage) file.")
    if enc != ENCODING_HAP2BIT and "p_exact" in stats:
        raise ValueError(
            "the exact conditional test conditions on the 2x2 HAPLOTYPE table, "
            "which needs TRUE phase and a 2bit file does not carry -- there is "
            "nothing to condition on, so serving it would return a plausible "
            "wrong number. Convert a phased VCF with cg.convert.vcf2cugenh(), "
            "or drop exact=/'p_exact'.")

    p_all = int(reader.n_variants)
    gidx_all = np.asarray(reader.gidx, dtype=np.int64)
    maf_all = np.asarray(reader.maf, dtype=np.float32)

    # ---- row selection ---------------------------------------------------
    rows = np.arange(p_all, dtype=np.int64)
    if variant_range is not None:
        s, e = variant_range
        rows = rows[max(0, int(s)):min(p_all, int(e))]
    if variants is not None:
        want = set(_resolve_gidx(variants).tolist())
        rows = rows[np.isin(gidx_all[rows], list(want))]
    if region is not None:
        chrom, start, end = _parse_region(region)
        sub = ann[ann["CHR"].astype(str) == str(chrom)]
        if start is not None:
            sub = sub[(sub["POS"] >= start) & (sub["POS"] <= end)]
        rows = rows[np.isin(gidx_all[rows], np.asarray(sub["gidx"], dtype=np.int64))]
    if maf_min > 0:
        rows = rows[maf_all[rows] >= maf_min]
    if rows.size == 0:
        return _empty_pairs(stats)

    positions = None
    if ann is not None:
        pos_map = dict(zip(np.asarray(ann["gidx"], dtype=np.int64),
                           np.asarray(ann["POS"], dtype=np.int64)))
        positions = np.array([pos_map.get(int(g), 0) for g in gidx_all[rows]],
                             dtype=np.int64)

    n_pairs = _count_pairs(len(rows), positions, window, window_kb)
    if n_pairs > max_pairs and not count_only:
        raise ValueError(
            f"plan would emit {n_pairs:,} pairs, above max_pairs={max_pairs:,}. "
            f"Narrow it with window= or window_kb=, raise min_r2, or raise "
            f"max_pairs if you really want this.")
    if n_pairs == 0:
        return _empty_pairs(stats)

    # ---- significance thresholds ----------------------------------------
    # n_pairs IS the number of tests. It comes from _count_pairs, in closed
    # form from the row count and the window, so m is known without touching
    # the data -- which is what makes correction affordable at genome scale.
    m_tests = n_pairs
    if correction == "bonferroni":
        max_p = alpha / m_tests
    if max_p is not None:
        # With no missingness every pair shares one N, so chi2 = N*r^2 is
        # strictly monotone in r^2 and the p-cut IS an r^2-cut: hand it to the
        # filter the kernel already applies and the test costs nothing. With
        # missingness N varies per pair and N <= n_eff, so the same conversion
        # is a CONSERVATIVE pre-filter (a pair below it cannot reach the cut at
        # any smaller N) and _apply_significance_filters applies the exact one.
        from scipy.stats import chi2 as _chi2_dist  # noqa: PLC0415
        n_eff = (2 * int(reader.n_samples) if want_phased
                 else int(reader.n_samples))
        min_r2 = float(_chi2_dist.isf(max_p, 1)) / n_eff
        if verbose:
            print(f"cugen.ld: max_p={max_p:.3g} over m={m_tests:,} tests -> "
                  f"min_r2={min_r2:.6g} at N={n_eff}")
    _screened = min_r2 > 0
    _sig_kw = dict(max_p=max_p, correction=correction, alpha=alpha,
                   m=m_tests, screened=_screened, verbose=verbose)
    # FDR needs every p-value to find its threshold, so it must not pre-screen.
    if correction == "fdr":
        _sig_kw["max_p"] = None

    # ---- counts ----------------------------------------------------------
    # D and D' need the 3x3 table; r-family statistics do not, and skipping it
    # avoids ~9x the memory traffic per tile.
    need_table = bool({"d", "dp"} & set(stats))
    if want_corrected and use_gpu:
        # The cost here is one n x n eigendecomposition plus a dense transform
        # of the genotype matrix, neither of which the GPU scan is shaped for,
        # and the paper's own scale is hundreds of individuals. Take the
        # reference path and say so rather than pretending otherwise.
        if verbose:
            print(f"cugen.ld: {want_corrected} take the reference path "
                  f"(dense sample-axis transform); backend={backend!r} ignored")
        use_gpu = False

    use_tf32 = _resolve_precision(precision, int(reader.n_samples), verbose)

    # cuDF fast path: r-family stats only, and only when we are writing a file.
    # The survivors are already in device memory, so keeping them there and
    # letting cudf wrap them avoids a host round trip we would immediately undo.
    # The cuDF/output preconditions exist to avoid a host round trip when
    # SERIALISING. count_only serialises nothing, so they do not apply.
    #
    # p_exact is excluded for the same reason need_table is: it needs the 2x2
    # table, and the fused epilogue emits only r. nAB IS recoverable from r via
    # _recover_nab, but the Fisher tail sum itself is a host loop over scipy's
    # gammaln, so keeping the survivors on the device would buy nothing and
    # would leave a second implementation of the same test to keep in step.
    # Under exact='auto' the loop runs on a small subset by construction.
    on_device = (use_gpu and not need_table and "p_exact" not in stats
                 and not lambda_gc and annotation is None
                 and (count_only or (HAS_CUDF and output is not None)))
    # The fused single-kernel scan handles the clean r-only case: no
    # missingness, no bp window, no D/D'. That is the hot path, and it is
    # where the device was sitting at 1-4% SM utilisation.
    # A hap2bit file cannot encode missingness at all, so the no-missing
    # precondition is structural there rather than a property of this file.
    fused_ok_phased = (not want_phased
                       or set(want_phased) <= {"r_phased", "r2_phased"})
    fused = (on_device and not reader.has_missing and window_kb is None
             and min_obs <= reader.n_samples and fused_ok_phased)

    if stream and output is None:
        raise ValueError(
            "stream=True writes the result incrementally and returns a row "
            "count, so it needs output=. Streaming to nowhere is a no-op.")
    if stream and not (use_gpu and fused):
        raise ValueError(
            "stream=True needs the fused GPU scan; it is the only path whose "
            "output is produced tile by tile.")
    if count_only and not (use_gpu and fused):
        raise ValueError(
            "count_only needs the fused GPU scan, which is the only path with "
            "a survivor counter. It is unavailable here because "
            f"{'D/D-prime were requested' if need_table else ''}"
            f"{'the file has missing calls' if reader.has_missing else ''}"
            f"{'window_kb was set' if window_kb is not None else ''}"
            f"{'no GPU is available' if not use_gpu else ''}"
            f"{'an annotation was passed' if annotation is not None else ''}"
            ". Returning a DataFrame instead of a count would be worse: the "
            "caller would treat a frame as a number.")

    if use_gpu and fused:
        cp.cuda.Device(device).use()
        if count_only:
            return _scan_gpu_fused(reader, rows, window, min_r2,
                                   tile_size=tile_size, verbose=verbose,
                                   tf32=use_tf32, phased=bool(want_phased),
                                   count_only=True)
        if stream:
            n_obs_s = (2 * int(reader.n_samples) if want_phased
                       else int(reader.n_samples))
            writer = _ChunkWriter(str(output))

            def _flush(ii_d, jj_d, rr_d):
                g = _assemble_device(
                    cp.stack([ii_d, jj_d], axis=1),
                    (rr_d, cp.full(ii_d.size, float(n_obs_s),
                                   dtype=cp.float32)),
                    reader, rows, stats, sign_reference, path, False,
                    n_planned=n_pairs)
                writer.write(g)

            try:
                total = _scan_gpu_fused(
                    reader, rows, window, min_r2, tile_size=tile_size,
                    verbose=verbose, tf32=use_tf32,
                    phased=bool(want_phased), on_flush=_flush,
                    flush_rows=flush_rows)
            finally:
                writer.close()
            if verbose:
                print(f"cugen.ld: streamed {total:,} rows to {output}")
            return int(total)
        ii, jj, rr = _scan_gpu_fused(reader, rows, window, min_r2,
                                     tile_size=tile_size, verbose=verbose,
                                     tf32=use_tf32, phased=bool(want_phased))
        if ii.size == 0:
            return _empty_pairs(stats)
        pairs_local = cp.stack([ii, jj], axis=1)
        n_obs = (2 * int(reader.n_samples) if want_phased
                 else int(reader.n_samples))
        n_dev = cp.full(ii.size, float(n_obs), dtype=cp.float32)
        df = _assemble_device(pairs_local, (rr, n_dev), reader, rows, stats,
                              sign_reference, path, verbose, n_planned=n_pairs)
        df = _apply_significance_filters(df, **_sig_kw)
        _write_df(df, str(output))
        if verbose:
            print(f"cugen.ld: {len(rows):,} variants -> {n_pairs:,} pairs "
                  f"planned, {len(df):,} emitted  (gpu, fused kernel)")
        return df

    if use_gpu:
        cp.cuda.Device(device).use()
        pairs_local, payload = _scan_gpu(
            reader, rows, window, window_kb, positions, min_r2, min_obs,
            tile_size=tile_size, verbose=verbose, need_table=need_table,
            keep_device=on_device)
        if len(pairs_local) == 0:
            return _empty_pairs(stats)
        if on_device:
            df = _assemble_device(pairs_local, payload, reader, rows, stats,
                                  sign_reference, path, verbose, n_planned=n_pairs)
            df = _apply_significance_filters(df, **_sig_kw)
            _write_df(df, str(output))
            if verbose:
                print(f"cugen.ld: {len(rows):,} variants -> {n_pairs:,} pairs "
                      f"planned, {len(df):,} emitted  (gpu, cudf device write)")
            return df
        if need_table:
            res = ld_from_counts(payload, dprime_method=dprime_method)
        else:
            r_arr, n_arr = payload
            res = _r_only_result(r_arr, n_arr, reader, rows, pairs_local)
    else:
        pairs_local, _ = _plan_pairs(len(rows), positions, window, window_kb)
        if want_phased:
            hap = _haplotypes_numpy(reader)[rows]
            res = phased_from_haplotypes(hap, pairs_local)
        else:
            dos = _dosages_numpy(reader)[rows]
            tables = contingency_tables(dos, pairs_local)
            res = ld_from_counts(tables, dprime_method=dprime_method)
            if want_corrected:
                # Missing calls are mean-imputed here. The transform is a dense
                # operation on the sample axis, so it cannot be done pairwise
                # complete-case the way the counts above are; LDcorSV drops
                # incomplete rows instead, which would change the sample set
                # from pair to pair and with it the kinship matrix.
                d = np.asarray(dos, dtype=np.float64)
                miss = d == 3
                if miss.any():
                    mu = np.where(miss, np.nan, d)
                    col = np.nanmean(mu, axis=1)
                    d = np.where(miss, col[:, None], d)
                for which in want_corrected:
                    P = _corrected_transform(which, int(reader.n_samples),
                                             kinship=kinship,
                                             structure=structure)
                    res[which] = _corrected_r2(d, pairs_local, P)

    # ---- orientation -----------------------------------------------------
    # Flipping BOTH variants leaves r unchanged; flipping exactly one negates
    # r, D and D'. plink2 orients by the major allele (Chang et al. 2015).
    if sign_reference == "major":
        flip = (res["pA"] > 0.5) ^ (res["pB"] > 0.5)
        for k in ("r", "r2_signed", "d", "dp", "r_phased", "d_phased",
                  "dp_phased"):
            if k in res:
                res[k] = np.where(flip, -res[k], res[k])

    # the phased path produces r_phased and no dosage r; filter on whichever
    # this call actually computed rather than assuming "r" is present
    r_key = "r" if "r" in res else "r_phased"
    r2_key = "r2" if "r2" in res else "r2_phased"
    _add_significance(res, stats, exact=exact)
    lam = None
    if lambda_gc:
        # Before `keep` is applied: filtering selects the tail, and the median
        # of a selected tail carries no information about the null.
        lam = _lambda_gc(res["chi2"],
                         pairs_local[:, 1] - pairs_local[:, 0])
        res["chi2_adj"] = res["chi2"] / lam
        res["p_adj"] = _neglog10_chi2_1df(res["chi2_adj"])
        if verbose:
            print(f"cugen.ld: lambda_gc = {lam:.4f} over "
                  f"{len(res['chi2']):,} pairs")
    keep = np.isfinite(res[r_key]) & (res["n"] >= min_obs)
    if min_r2 > 0:
        keep &= res[r2_key] >= min_r2
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return _empty_pairs(stats)

    ga = gidx_all[rows[pairs_local[idx, 0]]]
    gb = gidx_all[rows[pairs_local[idx, 1]]]
    df = pd.DataFrame({
        "gidx_a": ga, "gidx_b": gb,
        "MAF_A": maf_all[rows[pairs_local[idx, 0]]],
        "MAF_B": maf_all[rows[pairs_local[idx, 1]]],
        "N_OBS": res["n"][idx].astype(np.int32),
    })
    for s in _STATS:
        if s in stats:
            df[_STAT_COL[s]] = res[s][idx]

    chrom_fallback = 0
    m = re.search(r"chr(\d+)", os.path.basename(path))
    if m:
        chrom_fallback = int(m.group(1))
    df = _merge_annotation(df, ann, chrom_fallback)
    df = _finalize(df, stats)
    df = _apply_significance_filters(df, **_sig_kw)

    if verbose:
        print(f"cugen.ld: {len(rows):,} variants -> {n_pairs:,} pairs planned, "
              f"{len(df):,} emitted  (backend={'gpu' if use_gpu else 'numpy'}, "
              f"dprime={dprime_method}, sign={sign_reference})")
    if lam is not None:
        # last, because iloc/astype/concat do not reliably carry attrs
        df.attrs["lambda_gc"] = lam
    if output is not None:
        _write_df(df, str(output))
    return df
