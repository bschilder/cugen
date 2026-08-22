"""Mutation sweep: each entry is a plausible wrong implementation.

If the suite still passes, whatever test was supposed to cover that behaviour
is decorative. This is the check that caught three dud tests in the LD work
and one here (the l*eps rule, which was computed inline and never asserted).

Deliberately NOT included: removing `tau[0] = 0.0` from transition_tau. That is
an equivalent mutant -- d[0] is already 0, so tau[0] is already 0 and the line
cannot change any output. Listing it would make the sweep permanently red for a
behaviour that does not exist.
"""
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/Users/bschilder/code/cugen")

MUTATIONS = [
    # ---- LD result storage (.cugenld) ----
    # Deliberately NOT included: replacing os.replace() with copy+remove.
    # Both complete identically in the absence of a crash, so it is an
    # equivalent mutant -- atomicity is only observable under fault
    # injection, and listing it would keep the sweep permanently red.
    ('resume trusts a manifest entry whose file is gone', 'cugen/ldio.py',
     """                            if os.path.exists(os.path.join(self.path,""",
     """                            if True or os.path.exists(os.path.join(self.path,""", "tests/test_ldio.py"),
    ('resume ignores a params mismatch', 'cugen/ldio.py',
     """            if man.get(\"params\") != self.params:""",
     """            if False:""", "tests/test_ldio.py"),
    ('manifest shard skip ignores max_abs_r', 'cugen/ldio.py',
     """            return sh[\"max_abs_r\"] ** 2 >= min_r2""",
     """            return True""", "tests/test_ldio.py"),
    ('variant() opens every shard', 'cugen/ldio.py',
     """            if not (sh[\"min_i\"] <= v <= sh[\"max_i\"]):""",
     """            if False:""", "tests/test_ldio.py"),
    ('block pair cap is ignored', 'cugen/ldio.py',
     """                if ti.size > self.max_block_pairs:""",
     """                if False:""", "tests/test_ldio.py"),
    ('r quantisation truncates instead of rounding', 'cugen/ldio.py',
     """    return np.clip(np.rint(a * scale), -lim, lim).astype(_ENC_DTYPE[encoding])""",
     """    return np.clip(np.trunc(a * scale), -lim, lim).astype(_ENC_DTYPE[encoding])""", "tests/test_ldio.py"),
    ('delta coding does not reset at row boundaries', 'cugen/ldio.py',
     """        d[rs] = 0                                # reset at each row start""",
     """        pass""", "tests/test_ldio.py"),
    ('delta decode drops the running sum', 'cugen/ldio.py',
     """            c = np.cumsum(d.astype(np.int64))""",
     """            c = d.astype(np.int64)""", "tests/test_ldio.py"),
    ('zone map ignores max_abs_r and never skips', 'cugen/ldio.py',
     """            if t is not None and t > 0.0 and b[\"max_abs_r\"] ** 2 < t:""",
     """            if False:""", "tests/test_ldio.py"),
    ('reader answers a too-loose threshold instead of raising', 'cugen/ldio.py',
     """        if t < stored - 1e-12:""",
     """        if False:""", "tests/test_ldio.py"),
    ('rows() re-applies the stored cut to quantised r', 'cugen/ldio.py',
     """        if min_r2 is None:
            return None""",
     """        if min_r2 is None:
            return stored""", "tests/test_ldio.py"),
    ('dense() fills a thresholded file with zeros', 'cugen/ldio.py',
     """        if stored > 0.0:""",
     """        if False:""", "tests/test_ldio.py"),
    # ---- GRM ----
    ('GRM standardises by p(1-p), not 2p(1-p)', 'cugen/popstruct.py',
     """        z = xp.where(obs, (x - two_p) / xp.sqrt(two_p * (1.0 - freq[:, None])),""",
     """        z = xp.where(obs, (x - two_p) / xp.sqrt(freq[:, None] * (1.0 - freq[:, None])),""", "tests/test_popstruct.py"),
    ('GRM allele freq forgets the factor of two', 'cugen/popstruct.py',
     """            freq = x.sum(axis=1) / (2.0 * xp.maximum(n_obs, 1))""",
     """            freq = x.sum(axis=1) / xp.maximum(n_obs, 1)""", "tests/test_popstruct.py"),
    ('GRM 2-bit unpack loses the big-endian order', 'cugen/popstruct.py',
     """_SHIFTS = np.array([6, 4, 2, 0], dtype=np.uint8)""",
     """_SHIFTS = np.array([0, 2, 4, 6], dtype=np.uint8)""", "tests/test_popstruct.py"),
    # ---- Mangin corrected r^2 ----
    ('r2_v skips the GLS centring', 'cugen/ld.py',
     """    P = W @ (eye - np.outer(one, one @ V_inv) / denom)""",
     """    P = W @ eye""", "tests/test_ld_corrected.py"),
    ('r2_v centres on the ordinary mean, not the GLS one', 'cugen/ld.py',
     """    P = W @ (eye - np.outer(one, one @ V_inv) / denom)""",
     """    P = W @ (eye - np.full((n, n), 1.0 / n))""", "tests/test_ld_corrected.py"),
    ('r2_v whitens with V instead of its inverse root', 'cugen/ld.py',
     """    W = (U * np.sqrt(inv)) @ U.T""",
     """    W = (U * np.sqrt(np.abs(w))) @ U.T""", "tests/test_ld_corrected.py"),
    ('r2_s forgets to residualise on the structure', 'cugen/ld.py',
     """        return (eye - H) @ (eye - np.full((n, n), 1.0 / n))""",
     """        return (eye - np.full((n, n), 1.0 / n))""", "tests/test_ld_corrected.py"),
    ('r2_vs drops the Schur complement', 'cugen/ld.py',
     """    return (eye - Hz) @ P""",
     """    return P""", "tests/test_ld_corrected.py"),
    ('pseudo-inverse keeps the null eigenvalues', 'cugen/ld.py',
     """    inv = np.where(dead, 0.0, 1.0 / np.where(dead, 1.0, w))""",
     """    inv = 1.0 / np.where(np.abs(w) < 1e-300, 1e-300, w)""", "tests/test_ld_corrected.py"),
    # ---- LD significance ----
    ('exact test just returns the asymptotic value', 'cugen/ld.py',
     """    for t in np.flatnonzero(need):
        out[t] = _fisher_neglog10p_2x2(nAB[t], nA[t], nB[t], n[t])""",
     """    for t in np.flatnonzero(need):
        out[t] = _neglog10_chi2_1df(np.array([float(n[t]) * 0.5]), np)[0]""", "tests/test_ld_significance.py"),
    ('nAB reconstruction truncates instead of rounding', 'cugen/ld.py',
     """    return np.rint((r * den + nA * nB) / float(N)).astype(np.int64)""",
     """    return np.floor((r * den + nA * nB) / float(N)).astype(np.int64)""", "tests/test_ld_significance.py"),
    ('exact test gate never fires', 'cugen/ld.py',
     """    return (a * b / xp.asarray(N, dtype=xp.float64)) < 5.0""",
     """    return (a * b / xp.asarray(N, dtype=xp.float64)) < 0.0""", "tests/test_ld_significance.py"),
    ('Fisher two-sided keeps only the observed table', 'cugen/ld.py',
     """    tail = float(pmf[pmf <= obs * (1.0 + 1e-7)].sum())""",
     """    tail = float(obs)""", "tests/test_ld_significance.py"),
    ('chi2 counts individuals, not gametes', 'cugen/ld.py',
     """    n_eff = (2 * int(reader.n_samples) if want_phased
                 else int(reader.n_samples))""",
     """    n_eff = int(reader.n_samples)""", "tests/test_ld_significance.py"),
    # The device emit builds its own n_obs, but that path needs cuDF, so a
    # mutation there is unreachable on a CPU box and would leave the sweep
    # permanently red. phased_from_haplotypes is the CPU-side equivalent.
    ('phased N counts individuals, not haplotypes', 'cugen/ld.py',
     """            "n": np.full(pairs.shape[0], H),""",
     """            "n": np.full(pairs.shape[0], H / 2.0),""", "tests/test_ld_significance.py"),
    ('asymptotic p-value branch taken far too early', 'cugen/ld.py',
     """_NLP_ASYMPTOTIC_FROM = 400.0""",
     """_NLP_ASYMPTOTIC_FROM = 30.0""", "tests/test_ld_significance.py"),
    ('erfc branch used everywhere, no asymptotic tail', 'cugen/ld.py',
     """    return xp.where(z <= cut, small, large)""",
     """    return small""", "tests/test_ld_significance.py"),
    ('BH inequality flipped', 'cugen/ld.py',
     """    ok = order >= np.log10(m / (k * alpha))""",
     """    ok = order <= np.log10(m / (k * alpha))""", "tests/test_ld_significance.py"),
    ('BH uses the survivor count as m, not the test count', 'cugen/ld.py',
     """        cut, k = _bh_threshold_neglog10p(vals, m, alpha)""",
     """        cut, k = _bh_threshold_neglog10p(vals, vals.size, alpha)""", "tests/test_ld_significance.py"),
    ('Bonferroni forgets to divide by the test count', 'cugen/ld.py',
     """        max_p = alpha / m_tests""",
     """        max_p = alpha""", "tests/test_ld_significance.py"),
    ("aggregate pos = mean(all) not mean(first,last)", "cugen/_impute_core.py",
     "agg_cm = 0.5 * (cm[starts] + cm[stops - 1])",
     "agg_cm = np.array([cm[a:b].mean() for a, b in zip(starts, stops)])"),
    ("tau given cM instead of Morgans", "cugen/_impute_core.py",
     "tau = transition_tau(agg_cm / 100.0, ne, n_states_eff)",
     "tau = transition_tau(agg_cm, ne, n_states_eff)"),
    ("tau uses panel size, not the state count", "cugen/_impute_core.py",
     "tau = transition_tau(agg_cm / 100.0, ne, n_states_eff)",
     "tau = transition_tau(agg_cm / 100.0, ne, K)"),
    ("state selection ignores IBS run length", "cugen/_impute_core.py",
     "score = best[:, t].astype(np.int64) * (K + 1) - np.arange(K)",
     "score = -np.arange(K).astype(np.int64)"),
    ("aggregate mismatch = eps, not l*eps", "cugen/_impute_core.py",
     "mism = aggregate_mismatch(starts, stops, err)",
     "mism = np.full(starts.size, err)"),
    ("l*eps cap removed", "cugen/_impute_core.py",
     "return np.minimum(l * float(err), 0.5)", "return l * float(err)"),
    ("forward pass skips normalisation", "cugen/_impute_core.py",
     "        a /= a.sum(axis=0, keepdims=True)\n        alpha[c] = a",
     "        alpha[c] = a"),
    ("carriers store majority not minority", "cugen/_impute_core.py",
     "major = (ones * 2 > K).astype(np.uint8)",
     "major = np.zeros(M, dtype=np.uint8)"),
    ("phased dosage map uses the unphased meaning", "cugen/write.py",
     "return (h[0::2] + h[1::2]).astype(np.uint8)",
     "return ((h[0::2] << 1) | h[1::2]).astype(np.uint8)"),
    ("read_to_numpy phased guard dropped", "cugen/io.py",
     '        self._require_unphased("read_to_numpy")', "        pass"),
    ("pinned read_to_gpu guard dropped", "cugen/io.py",
     '        self._require_unphased("read_to_gpu")     # overrides the base method,',
     "        # overrides the base method,"),
    ("window overlap switch at start not midpoint", "cugen/impute.py",
     "bounds[w][0] + overlap / 2.0", "bounds[w][0]"),
    ("map extrapolation silently clamps", "cugen/_genmap.py",
     'if self.out_of_range == "extrapolate":', "if False:"),
    ("non-monotonic cM accepted (swapped columns)", "cugen/_genmap.py",
     "if np.any(np.diff(c) < 0):", "if False:"),
    ("sparse dose forgets the major-allele complement", "cugen/_impute_core.py",
     "        if major[m] == 1:\n            sl = 1.0 - sl\n            sr = 1.0 - sr",
     "        pass"),
]


# Each mutation names the test file that is SUPPOSED to catch it. Running the
# whole suite per mutation would be correct but slow; running one fixed file --
# which this did, always tests/test_impute.py -- silently gives every mutation
# outside that file a free pass. Both failure modes are worse than this.
DEFAULT_TARGET = "tests/test_impute.py"


def run(target=DEFAULT_TARGET):
    # PYTHONDONTWRITEBYTECODE is load-bearing, not tidiness. Several mutations
    # here swap a string for one of the SAME LENGTH (e.g. [6, 4, 2, 0] ->
    # [0, 2, 4, 6]). Restoring the original then leaves a file with identical
    # size and a near-identical mtime, so CPython happily reuses the .pyc it
    # compiled from the MUTATED source -- and the next ordinary `pytest` run
    # reports failures that are not in the working tree. That cost real
    # debugging time once; do not remove this.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([".venv/bin/python", "-m", "pytest", target,
                        "-q", "--no-header", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    return (r.stdout + r.stderr).strip().splitlines()[-1]


def _drop_pycache():
    for d in ROOT.rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)


caught = missed = 0
for entry in MUTATIONS:
    desc, rel, frm, to = entry[:4]
    target = entry[4] if len(entry) > 4 else DEFAULT_TARGET
    f = ROOT / rel
    bak = f.read_text()
    if frm not in bak:
        print(f"  {desc:48s} !! PATTERN NOT FOUND")
        continue
    f.write_text(bak.replace(frm, to, 1))
    _drop_pycache()
    line = run(target)
    f.write_text(bak)
    _drop_pycache()
    ok = "failed" in line
    caught += ok
    missed += not ok
    print(f"  {'caught ' if ok else 'MISSED '} {desc:48s} {line}")

print(f"\n{caught} caught, {missed} missed")
print("baseline:", run())
sys.exit(1 if missed else 0)
