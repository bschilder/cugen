"""The manifest must record which sign convention r was written with.

`.cugenld` stores r signed either on the ALT allele (ld_matrix's default) or on
the MINOR allele (`sign_reference="major"`). `.cugen` dosages count ALT. So
reconstructing a raw cross-product from a stored r needs
`r * (-1)^(major_i XOR major_j)` for a "major" file and no flip for an "alt"
file -- and applying the correction that is not needed is as wrong as omitting
one that is.

That ambiguity has already cost a wrong answer in this project: it produced
|r_adj| = 7.7, impossible for a correlation, after a complete table claiming 36%
of trans LD survived. Wrong by four orders of magnitude. A file that does not
record its own convention makes the mistake available to every future reader.
"""
from cugen.ldio import _PARAM_KEYS


def test_sign_reference_is_a_recorded_parameter():
    assert "sign_reference" in _PARAM_KEYS, (
        "a .cugenld cannot be interpreted without knowing whether r is signed "
        "on ALT or on the minor allele")


def test_the_recorded_params_still_cover_the_test_space_and_retention():
    # Guard against the key being added by dropping something else.
    for k in ("maf_min", "window", "min_r2", "scope", "adjust"):
        assert k in _PARAM_KEYS, k


def test_a_writer_round_trips_the_sign_convention(tmp_path):
    """Absent must stay distinguishable from 'alt'. Datasets written before this
    key existed have it unrecorded, and silently reporting 'alt' for them would
    assert a convention nobody checked."""
    import numpy as np
    from cugen.ldio import LDDatasetWriter, LDDatasetReader

    out = tmp_path / "d.cugenld"
    w = LDDatasetWriter(str(out), params={"min_r2": 0.1,
                                         "sign_reference": "major"})
    i = np.array([0, 0, 1], dtype=np.int64)
    j = np.array([1, 2, 2], dtype=np.int64)
    r = np.array([0.5, -0.4, 0.9], dtype=np.float32)
    w.write_shard((0, 0), i, j, r)
    w.close()
    assert LDDatasetReader(str(out)).params["sign_reference"] == "major"


def test_an_unrecorded_convention_reads_back_as_none(tmp_path):
    import numpy as np
    from cugen.ldio import LDDatasetWriter, LDDatasetReader

    out = tmp_path / "e.cugenld"
    w = LDDatasetWriter(str(out), params={"min_r2": 0.1})
    w.write_shard((0, 0), np.array([0], dtype=np.int64),
                  np.array([1], dtype=np.int64),
                  np.array([0.5], dtype=np.float32))
    w.close()
    assert LDDatasetReader(str(out)).params["sign_reference"] is None
