"""cugen._genmap - genetic maps and the base-pair -> centiMorgan conversion.

    GeneticMap.from_plink(path)      PLINK-format map, linear interpolation
    GeneticMap.constant_rate(1.0)    1 cM/Mb, Beagle's behaviour with no map
    gmap.cm(positions)               interpolated genetic position, cM
    gmap.morgans(positions)          the same, in MORGANS

UNITS ARE THE WHOLE POINT OF THIS MODULE
-----------------------------------------
The Li and Stephens transition probability is

    tau_m = 1 - exp(-4 * Ne * d_m / |H|)

and d_m there is in MORGANS, not centiMorgans (Browning et al. 2018, AJHG
103(3):338-348). Maps on disk are in cM. A factor of 100 in d_m moves tau_m by
orders of magnitude, and the result is not an error or a NaN -- it is an HMM
that switches reference haplotypes far too eagerly or barely at all, producing
imputed dosages that look entirely reasonable and are wrong.

So the conversion lives in exactly one place, `morgans()`, and callers building
transition probabilities are expected to use it rather than dividing by 100 at
the call site.

PLINK MAP FORMAT
----------------
Four whitespace-separated columns, no header:

    CHROM   ID   CM   POS
    22      .    0.0  16050408

which is the layout of the HapMap maps Beagle distributes at
bochet.gcc.biostat.washington.edu/beagle/genetic_maps/. Some maps in the wild
put cM and POS the other way round; a map whose "cM" column is not
non-decreasing is rejected rather than silently used, because a swapped map
still interpolates and still returns plausible numbers.

POSITIONS OUTSIDE THE MAP
-------------------------
Genetic maps begin at the first typed marker and end at the last, so real
chromosomes always carry variants outside that range -- telomeric and
centromeric regions especially. Three behaviours are available and the default
is `extrapolate`:

    extrapolate  continue at the rate of the terminal map segment (default)
    clamp        assign the boundary cM to everything beyond it
    raise        refuse

`clamp` is available but is the dangerous one: it implies ZERO recombination
across what may be megabases, so every variant past the map edge appears
perfectly linked to the edge and to each other. It is offered only because some
pipelines expect it; it is never the default.

Beagle's own out-of-range behaviour has not been confirmed against the binary
(it needs Java, and this was written on a machine without it). Until it is,
`extrapolate` is a considered default, not a claim of parity.
"""
import os

import numpy as np

CM_PER_MORGAN = 100.0
DEFAULT_RATE_CM_PER_MB = 1.0        # Beagle's assumption when no map is given

_OUT_OF_RANGE = ("extrapolate", "clamp", "raise")


class GeneticMap:
    """Interpolates genetic position (cM) at arbitrary base-pair positions."""

    def __init__(self, positions, cm, chrom=None, out_of_range="extrapolate"):
        if out_of_range not in _OUT_OF_RANGE:
            raise ValueError(
                f"out_of_range must be one of {_OUT_OF_RANGE}, got {out_of_range!r}")
        p = np.asarray(positions, dtype=np.int64)
        c = np.asarray(cm, dtype=np.float64)
        if p.ndim != 1 or c.ndim != 1:
            raise ValueError("positions and cm must be 1-D")
        if p.size != c.size:
            raise ValueError(f"{p.size} positions but {c.size} cM values")
        if p.size < 2:
            raise ValueError(
                f"a genetic map needs at least 2 points to interpolate, got "
                f"{p.size}. For a rate assumption use "
                f"GeneticMap.constant_rate().")

        order = np.argsort(p, kind="stable")
        p, c = p[order], c[order]

        if np.any(np.diff(p) <= 0):
            dup = int(p[np.flatnonzero(np.diff(p) <= 0)[0]])
            raise ValueError(
                f"map positions must be strictly increasing; position {dup} "
                f"repeats or decreases. Duplicate positions make the slope "
                f"between them infinite.")
        if np.any(np.diff(c) < 0):
            i = int(np.flatnonzero(np.diff(c) < 0)[0])
            raise ValueError(
                f"cM must be non-decreasing with position; it falls from "
                f"{c[i]:.6f} to {c[i+1]:.6f} between bp {p[i]} and {p[i+1]}. "
                f"This is the signature of a map with the cM and POS columns "
                f"swapped, which would otherwise interpolate happily.")

        self.positions = p
        self.cm_values = c
        self.chrom = None if chrom is None else str(chrom)
        self.out_of_range = out_of_range

    @classmethod
    def from_plink(cls, path, chrom=None, out_of_range="extrapolate"):
        """Read a PLINK-format map (CHROM ID CM POS), optionally one chromosome."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"genetic map not found: {path}")
        chroms, cms, poss = [], [], []
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                s = line.split()
                if not s or s[0].startswith("#"):
                    continue
                if len(s) < 4:
                    raise ValueError(
                        f"{path}:{lineno}: expected 4 columns "
                        f"(CHROM ID CM POS), got {len(s)}: {line.strip()!r}")
                try:
                    cm_v, pos_v = float(s[2]), int(s[3])
                except ValueError as e:
                    raise ValueError(
                        f"{path}:{lineno}: could not parse CM={s[2]!r} "
                        f"POS={s[3]!r} -- is this really a PLINK map?") from e
                chroms.append(s[0])
                cms.append(cm_v)
                poss.append(pos_v)

        if not poss:
            raise ValueError(f"{path}: no map rows found")

        chroms = np.asarray(chroms)
        if chrom is not None:
            want = str(chrom)
            keep = (chroms == want) | (chroms == f"chr{want}") | \
                   (np.char.replace(chroms.astype(str), "chr", "") == want)
            if not keep.any():
                raise ValueError(
                    f"{path}: no rows for chromosome {want!r}; the file has "
                    f"{sorted(set(chroms.tolist()))[:8]}")
            poss = np.asarray(poss)[keep]
            cms = np.asarray(cms)[keep]
        else:
            uniq = sorted(set(chroms.tolist()))
            if len(uniq) > 1:
                raise ValueError(
                    f"{path} covers {len(uniq)} chromosomes {uniq[:8]}; pass "
                    f"chrom= to select one. Interpolating across a chromosome "
                    f"boundary would treat the join as a genetic distance.")
        return cls(poss, cms, chrom=chrom, out_of_range=out_of_range)

    @classmethod
    def constant_rate(cls, rate_cm_per_mb=DEFAULT_RATE_CM_PER_MB, chrom=None):
        """A uniform map, which is what Beagle assumes when given no map file."""
        if rate_cm_per_mb <= 0:
            raise ValueError(f"rate must be positive, got {rate_cm_per_mb}")
        obj = cls.__new__(cls)
        obj.positions = None
        obj.cm_values = None
        obj.chrom = None if chrom is None else str(chrom)
        obj.out_of_range = "extrapolate"
        obj.rate_cm_per_mb = float(rate_cm_per_mb)
        return obj

    @property
    def is_constant_rate(self):
        return getattr(self, "rate_cm_per_mb", None) is not None

    def cm(self, positions):
        """Genetic position in centiMorgans at each base-pair position."""
        q = np.asarray(positions, dtype=np.float64)
        if self.is_constant_rate:
            return q * 1e-6 * self.rate_cm_per_mb

        p = self.positions.astype(np.float64)
        c = self.cm_values

        below = q < p[0]
        above = q > p[-1]
        if (below.any() or above.any()) and self.out_of_range == "raise":
            bad = q[below | above]
            raise ValueError(
                f"{bad.size} position(s) outside the map range "
                f"[{int(p[0])}, {int(p[-1])}], e.g. {int(bad[0])}. Pass "
                f"out_of_range='extrapolate' (the default) or 'clamp'.")

        out = np.interp(q, p, c)          # np.interp clamps outside the range

        if self.out_of_range == "extrapolate":
            # np.interp has already clamped; replace the clamped values with a
            # linear continuation at the terminal segment's rate. Clamping would
            # imply zero recombination beyond the map, making every variant past
            # the edge look perfectly linked.
            if below.any():
                rate = (c[1] - c[0]) / (p[1] - p[0])
                out[below] = c[0] + (q[below] - p[0]) * rate
            if above.any():
                rate = (c[-1] - c[-2]) / (p[-1] - p[-2])
                out[above] = c[-1] + (q[above] - p[-1]) * rate
        return out

    def morgans(self, positions):
        """Genetic position in MORGANS -- the unit the Li-Stephens tau needs."""
        return self.cm(positions) / CM_PER_MORGAN

    def __repr__(self):
        if self.is_constant_rate:
            return (f"GeneticMap(constant {self.rate_cm_per_mb} cM/Mb"
                    f"{'' if self.chrom is None else f', chrom={self.chrom}'})")
        return (f"GeneticMap({self.positions.size:,} points, "
                f"{self.positions[0]:,}-{self.positions[-1]:,} bp, "
                f"{self.cm_values[-1] - self.cm_values[0]:.2f} cM"
                f"{'' if self.chrom is None else f', chrom={self.chrom}'})")


def resolve_map(map_arg, chrom=None):
    """Accept a path, a GeneticMap, or None -> Beagle's 1 cM/Mb assumption."""
    if map_arg is None:
        return GeneticMap.constant_rate(chrom=chrom)
    if isinstance(map_arg, GeneticMap):
        return map_arg
    return GeneticMap.from_plink(map_arg, chrom=chrom)
