"""The `cugen ld` subcommand.

The first LD entry point on the CLI, and the first subcommand in this repo with
real numeric output, so these tests drive main(argv=[...]) directly and check
the written file rather than only that the process exited.
"""
import numpy as np
import pandas as pd
import pytest

from cugen.cli import main

CPU = ["--backend", "numpy"]


@pytest.fixture
def panel(tmp_path):
    """12 variants x 200 samples with a spread of LD, written as chr22.cugen."""
    from cugen.write import write_cugen
    rng = np.random.default_rng(4)
    n, p_var = 200, 12
    base = rng.integers(0, 3, size=n).astype(np.uint8)
    dos = np.empty((n, p_var), dtype=np.uint8)
    for v in range(p_var):
        q = 0.05 + 0.9 * (v / (p_var - 1))
        noisy = rng.integers(0, 3, size=n).astype(np.uint8)
        dos[:, v] = np.where(rng.random(n) < q, noisy, base)
    path = tmp_path / "chr22.cugen"
    write_cugen(str(path), dos)
    return str(path)


def test_ld_writes_a_table_with_the_requested_statistics(panel, tmp_path):
    out = tmp_path / "ld.tsv"
    main(["ld", panel, "--out", str(out), "--stats", "r2,chi2,p"] + CPU)
    df = pd.read_csv(out, sep="\t")
    assert len(df) == 66
    for col in ("R2", "CHI2", "NEG_LOG10_P", "N_OBS"):
        assert col in df.columns, f"missing {col}"
    np.testing.assert_allclose(df["CHI2"], df["N_OBS"] * df["R2"], rtol=1e-5)


def test_ld_max_p_filters_the_output(panel, tmp_path):
    every = tmp_path / "all.tsv"
    some = tmp_path / "some.tsv"
    main(["ld", panel, "--out", str(every), "--stats", "r2,p"] + CPU)
    main(["ld", panel, "--out", str(some), "--stats", "r2,p",
          "--max-p", "1e-6"] + CPU)
    a = pd.read_csv(every, sep="\t")
    b = pd.read_csv(some, sep="\t")
    assert 0 < len(b) < len(a)
    assert (b["NEG_LOG10_P"] >= 6.0 - 1e-6).all()


def test_ld_fdr_correction_runs_and_reports(panel, tmp_path, capsys):
    out = tmp_path / "fdr.tsv"
    main(["ld", panel, "--out", str(out), "--stats", "r2,p",
          "--correction", "fdr", "--alpha", "0.01"] + CPU)
    assert "BH-FDR" in capsys.readouterr().out
    assert len(pd.read_csv(out, sep="\t")) > 0


def test_ld_defaults_to_r2_and_p(panel, tmp_path):
    """The subcommand exists to filter LD by significance, so p must be there
    without being asked for -- unlike the Python API, whose default set is
    frozen for backwards compatibility."""
    out = tmp_path / "d.tsv"
    main(["ld", panel, "--out", str(out)] + CPU)
    df = pd.read_csv(out, sep="\t")
    assert "R2" in df.columns and "NEG_LOG10_P" in df.columns


def test_ld_window_restricts_pairs(panel, tmp_path):
    out = tmp_path / "w.tsv"
    main(["ld", panel, "--out", str(out), "--window", "2"] + CPU)
    df = pd.read_csv(out, sep="\t")
    assert len(df) == 21          # 12 variants, |i-j| <= 2
    assert (abs(df["gidx_b"] - df["gidx_a"]) <= 2).all()


def test_ld_rejects_max_p_with_min_r2(panel, tmp_path):
    with pytest.raises(ValueError, match="max_p.*min_r2|min_r2.*max_p"):
        main(["ld", panel, "--out", str(tmp_path / "x.tsv"),
              "--max-p", "1e-3", "--min-r2", "0.2"] + CPU)


def test_ld_help_works_without_a_gpu():
    """`cugen ld --help` must print usage and exit cleanly.

    An earlier version of this asserted that cupy stays out of sys.modules. That
    passed only on a machine with no cupy installed, and failed the first time it
    ran on a GPU box -- because cugen/__init__.py imports its submodules eagerly,
    so `import cugen.cli` pulls in cupy wherever cupy exists. The real guarantee
    is the one below: the CuPy imports are individually guarded, so --help works
    with no GPU stack present at all. Asserting the absence of cupy tested the
    test environment, not the code.
    """
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-c",
         "import cugen.cli as c\n"
         "try: c.main(['ld','--help'])\n"
         "except SystemExit: pass\n"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--max-p" in r.stdout and "--correction" in r.stdout, r.stdout


def test_ld_lambda_gc_flag_reports_the_inflation_factor(panel, tmp_path, capsys):
    out = tmp_path / "lam.tsv"
    main(["ld", panel, "--out", str(out), "--stats", "r2,p",
          "--lambda-gc"] + CPU)
    assert "lambda_gc" in capsys.readouterr().out
    df = pd.read_csv(out, sep="\t")
    assert "NEG_LOG10_P_ADJ" in df.columns
