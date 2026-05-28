# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GlobalKeys dataclass and its sub-dataclasses."""

import json
from pathlib import Path

import pytest
import yaml

from felis.configs import GKAB
from felis.configs import GKBoresch
from felis.configs import GKDir
from felis.configs import GKFilename
from felis.configs import GKIntegrator
from felis.configs import GKOpenMM
from felis.configs import GKPosres
from felis.configs import GlobalKeys

# ---------------------------------------------------------------------------
# 1. Default construction
# ---------------------------------------------------------------------------


def test_default_construction_types():
    """Verify all 7 sub-dataclasses are initialized with correct types."""
    gk = GlobalKeys()

    assert isinstance(gk.dir, GKDir)
    assert isinstance(gk.filename, GKFilename)
    assert isinstance(gk.integrator, GKIntegrator)
    assert isinstance(gk.posres, GKPosres)
    assert isinstance(gk.boresch, GKBoresch)
    assert isinstance(gk.ab, GKAB)
    assert isinstance(gk.openmm, GKOpenMM)


def test_default_construction_values():
    """Verify key default values on a freshly constructed GlobalKeys."""
    gk = GlobalKeys()

    # integrator defaults
    assert gk.integrator.name == "LangevinMiddleIntegrator"
    assert gk.integrator.dt_ps == 0.002
    assert gk.integrator.constraint_tol == 1e-8
    assert gk.integrator.friction_1_ps == 0.5
    assert gk.integrator.nstep_per_snapshot == 500
    assert gk.integrator.nsnapshots == 1
    assert gk.integrator.targetT_K == 298.15
    assert gk.integrator.targetP_bar == 1.01325
    assert gk.integrator.npt == 0
    assert gk.integrator.npt_mc_freq == 25

    # openmm defaults
    assert gk.openmm.platform == "CUDA"
    assert gk.openmm.precision == "mixed"
    assert gk.openmm.params_ecosystem == "gromacs"
    assert gk.openmm.checkpoint_interval == 50

    # posres defaults
    assert gk.posres.atoms is None
    assert gk.posres.k_kcal == 25.0
    assert gk.posres.tol_angstrom == 0.5

    # boresch defaults
    assert gk.boresch.ligatoms is None
    assert gk.boresch.proatoms is None
    assert gk.boresch.r_theta_phi is None
    assert gk.boresch.alpha_beta_gamma is None
    assert gk.boresch.k_r_a_dih_kcal is None

    # ab defaults
    assert gk.ab.ligatoms is None
    assert gk.ab.vlam == 1.0
    assert gk.ab.elam == 1.0
    assert gk.ab.reslam == 0.0
    assert gk.ab.lam_list is None
    assert gk.ab.ilam == 0

    # dir defaults (outbase resolves to an absolute path)
    assert gk.dir.outbase is not None
    assert Path(gk.dir.outbase).is_absolute()
    assert gk.dir.trj is not None

    # filename defaults
    assert gk.filename.stem is None
    assert gk.filename.sys is None
    assert gk.filename.crd is None
    assert gk.filename.monomer is None
    assert gk.filename.atom_ids is None


# ---------------------------------------------------------------------------
# 2. update_by_tkv
# ---------------------------------------------------------------------------


def test_update_by_tkv_int():
    """Update an integer field via tkv."""
    gk = GlobalKeys()
    gk.update_by_tkv("i:integrator.nsnapshots:100")
    assert gk.integrator.nsnapshots == 100
    assert isinstance(gk.integrator.nsnapshots, int)


def test_update_by_tkv_float():
    """Update a float field via tkv."""
    gk = GlobalKeys()
    gk.update_by_tkv("f:integrator.targetT_K:310.0")
    assert gk.integrator.targetT_K == 310.0
    assert isinstance(gk.integrator.targetT_K, float)


def test_update_by_tkv_string():
    """Update a string field via tkv."""
    gk = GlobalKeys()
    gk.update_by_tkv("s:openmm.platform:CPU")
    assert gk.openmm.platform == "CPU"


def test_update_by_tkv_none_value():
    """An empty value after the last colon should set the field to None."""
    gk = GlobalKeys()
    gk.update_by_tkv("s:filename.stem:")
    assert gk.filename.stem is None


def test_update_by_tkv_case_insensitive():
    """Field names in tkv should be case-insensitive."""
    gk = GlobalKeys()
    gk.update_by_tkv("f:INTEGRATOR.TARGETT_K:310.0")
    assert gk.integrator.targetT_K == 310.0


# ---------------------------------------------------------------------------
# 3. update_by_comma_sep_tkv
# ---------------------------------------------------------------------------


def test_update_by_comma_sep_tkv_multiple():
    """Multiple tkv entries separated by commas should all be applied."""
    gk = GlobalKeys()
    gk.update_by_comma_sep_tkv("f:integrator.targetT_K:310.0,s:openmm.platform:CPU")
    assert gk.integrator.targetT_K == 310.0
    assert gk.openmm.platform == "CPU"


def test_update_by_comma_sep_tkv_empty_entries():
    """Empty entries between commas should be silently ignored."""
    gk = GlobalKeys()
    gk.update_by_comma_sep_tkv("f:integrator.targetT_K:310.0,,s:openmm.platform:CPU,")
    assert gk.integrator.targetT_K == 310.0
    assert gk.openmm.platform == "CPU"


def test_update_by_comma_sep_tkv_single():
    """A single tkv entry without commas should work."""
    gk = GlobalKeys()
    gk.update_by_comma_sep_tkv("i:integrator.nsnapshots:200")
    assert gk.integrator.nsnapshots == 200


# ---------------------------------------------------------------------------
# 4. update_by_cfg
# ---------------------------------------------------------------------------


def test_update_by_cfg_yaml(tmp_path: Path):
    """Load configuration from a YAML file."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.dump({
            "integrator": {
                "targetT_K": 310.0,
                "nsnapshots": 500,
            },
            "openmm": {
                "platform": "CPU",
                "precision": "double",
            },
        }))

    gk = GlobalKeys()
    gk.update_by_cfg(str(cfg_path))

    assert gk.integrator.targetT_K == 310.0
    assert gk.integrator.nsnapshots == 500
    assert gk.openmm.platform == "CPU"
    assert gk.openmm.precision == "double"


def test_update_by_cfg_json(tmp_path: Path):
    """Load configuration from a JSON file."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({
            "integrator": {
                "targetT_K": 300.0,
                "dt_ps": 0.001,
            },
            "openmm": {
                "platform": "Reference",
            },
        }))

    gk = GlobalKeys()
    gk.update_by_cfg(str(cfg_path))

    assert gk.integrator.targetT_K == 300.0
    assert gk.integrator.dt_ps == 0.001
    assert gk.openmm.platform == "Reference"


def test_update_by_cfg_yaml_case_insensitive(tmp_path: Path):
    """Section and field names in YAML should be case-insensitive."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.dump({
            "INTEGRATOR": {
                "TARGETT_K": 320.0,
                "Nsnapshots": 250,
            },
            "OPENMM": {
                "Platform": "CPU",
            },
        }))

    gk = GlobalKeys()
    gk.update_by_cfg(str(cfg_path))

    assert gk.integrator.targetT_K == 320.0
    assert gk.integrator.nsnapshots == 250
    assert gk.openmm.platform == "CPU"


# ---------------------------------------------------------------------------
# 5. check() method
# ---------------------------------------------------------------------------


def test_check_default():
    """check() should run without error on a default GlobalKeys."""
    gk = GlobalKeys()
    gk.check()  # should not raise


def test_check_after_updates():
    """check() should run without error after various updates."""
    gk = GlobalKeys()
    gk.update_by_tkv("f:integrator.targetT_K:310.0")
    gk.update_by_tkv("s:openmm.platform:CPU")
    gk.check()  # should not raise


# ---------------------------------------------------------------------------
# 6. __str__ method
# ---------------------------------------------------------------------------


def test_str_non_empty():
    """__str__ should return a non-empty string."""
    gk = GlobalKeys()
    s = str(gk)
    assert isinstance(s, str)
    assert len(s) > 0


def test_str_contains_sub_dataclass_names():
    """The string representation should mention sub-dataclass field names."""
    gk = GlobalKeys()
    s = str(gk)
    # The __str__ iterates over fields(GlobalKeys) and calls str() on each
    # sub-dataclass, so the output should be non-empty and contain content
    # from the sub-dataclasses.
    assert "GKDir" in s or "GKFilename" in s or "GKIntegrator" in s


# ---------------------------------------------------------------------------
# 7. Error cases
# ---------------------------------------------------------------------------


def test_error_invalid_tkv_type_char():
    """An invalid type character in tkv should raise AssertionError."""
    gk = GlobalKeys()
    with pytest.raises(AssertionError):
        gk.update_by_tkv("x:integrator.nsnapshots:100")


def test_error_invalid_tkv_format():
    """A tkv string without enough colons should raise ValueError."""
    gk = GlobalKeys()
    with pytest.raises(ValueError, match="invalid tkv"):
        gk.update_by_tkv("invalid")


def test_error_invalid_tkv_single_colon():
    """A tkv string with only one colon should raise AssertionError
    (because the key part won't split into two parts)."""
    gk = GlobalKeys()
    with pytest.raises((AssertionError, ValueError)):
        gk.update_by_tkv("i:onlyonecolon")


def test_error_invalid_cfg_section_name(tmp_path: Path):
    """A YAML config with a nonexistent section name should raise AssertionError."""
    cfg_path = tmp_path / "bad_config.yaml"
    cfg_path.write_text(yaml.dump({
        "nonexistent_section": {
            "some_key": "some_value",
        },
    }))

    gk = GlobalKeys()
    with pytest.raises(AssertionError):
        gk.update_by_cfg(str(cfg_path))


def test_error_invalid_cfg_field_name(tmp_path: Path):
    """A YAML config with a valid section but invalid field name should raise AssertionError."""
    cfg_path = tmp_path / "bad_field.yaml"
    cfg_path.write_text(yaml.dump({
        "integrator": {
            "nonexistent_field": 123,
        },
    }))

    gk = GlobalKeys()
    with pytest.raises(AssertionError):
        gk.update_by_cfg(str(cfg_path))
