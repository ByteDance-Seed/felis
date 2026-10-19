# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
from pathlib import Path

import numpy as np
import pytest

from felis.protocols.abfe.config_types import ABFEInputConfig
from felis.protocols.abfe_v2.lambda_config import (
    CHARGED_WATER_LEG_WINDOWS,
    NEUTRAL_WATER_LEG_WINDOWS,
    get_water_leg_window_count,
    resample_water_leg_schedule,
)
from felis.protocols.abfe_v2.lambda_config import read_lambda_schedule
from felis.protocols.abfe_v2.lambda_config import select_lambda_schedule

SYSA_OPT_LAM_JSON = "sysA_opt_lam.json"
SYSB_OPT_LAM_JSON = "sysB_opt_lam.json"


def _write_lam_json(path: Path, schedule: list[list[float]]) -> None:
    path.write_text(json.dumps({"ab": {"lam_list": schedule}}))


def test_abfe_v2_sampling_defaults_are_per_leg():
    acfg = ABFEInputConfig()

    assert acfg.md_sol_nsnapshots == 10
    assert acfg.md_pro_nsnapshots == 10
    assert acfg.md_boresch_nsnapshots == 300
    assert acfg.md_boresch_traj_interval == 2500
    assert not hasattr(acfg, "md_nsnapshots")
    assert not hasattr(acfg, "md_eq_nsnapshots")


def test_abfe_v2_rejects_removed_shared_snapshot_field(tmp_path):
    cfg_file = tmp_path / "abfe.yaml"
    cfg_file.write_text("md_nsnapshots: 10\n")

    with pytest.raises(ValueError, match="Unknown key md_nsnapshots"):
        ABFEInputConfig.from_file(str(cfg_file))


def test_abfe_v2_rejects_removed_equilibration_snapshot_field(tmp_path):
    cfg_file = tmp_path / "abfe.yaml"
    cfg_file.write_text("md_eq_nsnapshots: 300\n")

    with pytest.raises(ValueError, match="Unknown key md_eq_nsnapshots"):
        ABFEInputConfig.from_file(str(cfg_file))


def test_abfe_v2_manual_lambda_config_paths_are_accepted(tmp_path, monkeypatch):
    cfg_file = tmp_path / "abfe.yaml"
    cfg_file.write_text(
        """
wat_lam_json: water_lam.json
pro_lam_json: protein_lam.json
outdir: out
"""
    )

    monkeypatch.chdir(tmp_path)
    acfg = ABFEInputConfig.from_file(str(cfg_file))
    acfg.check()

    assert acfg.wat_lam_json == str(tmp_path / "water_lam.json")
    assert acfg.pro_lam_json == str(tmp_path / "protein_lam.json")


def test_read_lambda_schedule_accepts_nested_and_top_level_lists(tmp_path):
    schedule = [
        [1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 1, 1],
    ]
    nested_path = tmp_path / "nested.json"
    list_path = tmp_path / "list.json"
    _write_lam_json(nested_path, schedule)
    list_path.write_text(json.dumps(schedule))

    assert read_lambda_schedule(nested_path) == [[float(value) for value in row] for row in schedule]
    assert read_lambda_schedule(list_path) == [[float(value) for value in row] for row in schedule]


def test_read_lambda_schedule_rejects_non_abfe_v2_rows(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"ab": {"lam_list": [[1, 0, 0]]}}))

    with pytest.raises(ValueError, match="list of 7 values"):
        read_lambda_schedule(bad_path)


def test_water_leg_window_count_depends_on_formal_charge():
    assert get_water_leg_window_count(0) == NEUTRAL_WATER_LEG_WINDOWS == 24
    assert get_water_leg_window_count(1) == CHARGED_WATER_LEG_WINDOWS == 32
    assert get_water_leg_window_count(-2) == CHARGED_WATER_LEG_WINDOWS


@pytest.mark.parametrize(
    ("ligand_charge", "expected_windows"),
    [(0, 24), (1, 32), (-1, 32)],
)
def test_resample_water_leg_schedule_enforces_fixed_count(
    ligand_charge,
    expected_windows,
):
    schedule = np.array([
        [1.0, 0.0, 0.0, 1.0, 0.10, 1.0, 0.0],
        [0.7, 0.4, 0.2, 1.0, 0.25, 0.9, 0.0],
        [0.0, 1.0, 0.0, 0.7, 0.55, 0.5, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.85, 0.0, 0.0],
    ])

    resampled = resample_water_leg_schedule(schedule, ligand_charge)

    assert resampled.shape == (expected_windows, 7)
    np.testing.assert_allclose(resampled[0], [
        1.0, 0.0, 0.0, 1.0, 0.10, 1.0, 0.0
    ])
    np.testing.assert_allclose(resampled[-1], [
        0.0, 1.0, 0.0, 0.0, 0.85, 0.0, 0.0
    ])
    np.testing.assert_allclose(resampled[:, 1], 1.0 - resampled[:, 0])
    np.testing.assert_allclose(resampled[:, 2], 0.0)
    assert not np.allclose(resampled[:, 4], 1.0 - resampled[:, 3])


def test_configured_water_lambdas_override_optimized_file(tmp_path):
    configured_schedule = [[1, 0, 0, 1, 0, 1, 0]]
    optimized_schedule = [[0, 1, 0, 0, 1, 1, 1]]
    configured_path = tmp_path / "configured_wat.json"
    prepare_dir = tmp_path / "prepare"
    prepare_dir.mkdir()
    _write_lam_json(configured_path, configured_schedule)
    _write_lam_json(prepare_dir / SYSA_OPT_LAM_JSON, optimized_schedule)

    assert select_lambda_schedule(
        configured_path=configured_path,
        optimized_path=prepare_dir / SYSA_OPT_LAM_JSON,
        recipe_lambdas_fn=lambda: [[9, 9, 9, 9, 9, 9, 9]],
        schedule_name="water-leg",
    ) == configured_schedule


def test_configured_protein_lambdas_override_optimized_file(tmp_path):
    configured_schedule = [[0, 1, 0, 0, 1, 1, 1]]
    optimized_schedule = [[1, 0, 0, 1, 0, 1, 1]]
    configured_path = tmp_path / "configured_pro.json"
    prepare_dir = tmp_path / "prepare"
    prepare_dir.mkdir()
    _write_lam_json(configured_path, configured_schedule)
    _write_lam_json(prepare_dir / SYSB_OPT_LAM_JSON, optimized_schedule)

    assert select_lambda_schedule(
        configured_path=configured_path,
        optimized_path=prepare_dir / SYSB_OPT_LAM_JSON,
        recipe_lambdas_fn=lambda: [[9, 9, 9, 9, 9, 9, 9]],
        schedule_name="protein-leg",
    ) == configured_schedule


def test_optimized_lambdas_override_recipe_when_config_is_missing(tmp_path):
    optimized_schedule = [[0, 1, 0, 0, 1, 1, 1]]
    optimized_path = tmp_path / SYSA_OPT_LAM_JSON
    _write_lam_json(optimized_path, optimized_schedule)

    assert select_lambda_schedule(
        configured_path=None,
        optimized_path=optimized_path,
        recipe_lambdas_fn=lambda: [[9, 9, 9, 9, 9, 9, 9]],
        schedule_name="water-leg",
    ) == optimized_schedule


def test_recipe_lambdas_are_used_when_no_json_is_configured(tmp_path):
    recipe_schedule = [[9, 9, 9, 9, 9, 9, 9]]

    assert select_lambda_schedule(
        configured_path=None,
        optimized_path=tmp_path / SYSA_OPT_LAM_JSON,
        recipe_lambdas_fn=lambda: recipe_schedule,
        schedule_name="water-leg",
    ) == recipe_schedule
