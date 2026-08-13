"""Mutation sweep: each entry is a plausible wrong implementation.

If the suite still passes, whatever test was supposed to cover that behaviour
is decorative. This is the check that caught three dud tests in the LD work
and one here (the l*eps rule, which was computed inline and never asserted).

Deliberately NOT included: removing `tau[0] = 0.0` from transition_tau. That is
an equivalent mutant -- d[0] is already 0, so tau[0] is already 0 and the line
cannot change any output. Listing it would make the sweep permanently red for a
behaviour that does not exist.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/Users/bschilder/code/cugen")

MUTATIONS = [
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


def run():
    r = subprocess.run([".venv/bin/python", "-m", "pytest", "tests/test_impute.py",
                        "-q", "--no-header", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip().splitlines()[-1]


caught = missed = 0
for desc, rel, frm, to in MUTATIONS:
    f = ROOT / rel
    bak = f.read_text()
    if frm not in bak:
        print(f"  {desc:48s} !! PATTERN NOT FOUND")
        continue
    f.write_text(bak.replace(frm, to, 1))
    line = run()
    f.write_text(bak)
    ok = "failed" in line
    caught += ok
    missed += not ok
    print(f"  {'caught ' if ok else 'MISSED '} {desc:48s} {line}")

print(f"\n{caught} caught, {missed} missed")
print("baseline:", run())
sys.exit(1 if missed else 0)
