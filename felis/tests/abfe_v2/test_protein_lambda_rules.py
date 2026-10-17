# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import pytest

from felis.protocols.abfe_v2.lambda_config import resample_water_leg_schedule
from felis.protocols.abfe_v2.protein_lambda_rules import (
    freeze_stage_lambda_counts,
    infer_stage_boundaries,
    require_full_restraint_endpoint,
    set_restraint_lambda_count,
    validate_protein_lambda_schedule,
)
from felis.protocols.abfe_v2.recipes import (
    get_charged_water_leg_lambdas,
    get_neutral_water_leg_lambdas,
)


def test_full_restraint_endpoint_validation_accepts_only_one():
    require_full_restraint_endpoint(1.0)
    valid_schedule = np.zeros((2, 7), dtype=np.float64)
    valid_schedule[-1, 6] = 1.0
    validate_protein_lambda_schedule(valid_schedule)

    for endpoint in (0.5, np.nan, np.inf):
        with pytest.raises(ValueError, match="analytical Boresch correction.*full restraint endpoint"):
            require_full_restraint_endpoint(endpoint)


def test_set_restraint_lambda_count_uses_exact_window_count():
    schedule = np.ones((10, 7), dtype=np.float64)
    schedule[:, 6] = [0.0, 0.2, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    original_non_restraint = schedule[:, :6].copy()

    out = set_restraint_lambda_count(schedule, n_restraint_lambdas=6)

    np.testing.assert_allclose(out[:, :6], original_non_restraint)
    assert np.all(out[:5, 6] < 1.0)
    assert out[5, 6] == pytest.approx(1.0)
    assert np.all(out[5:, 6] == pytest.approx(1.0))
    assert np.all(np.diff(out[:6, 6]) >= 0.0)


def test_set_restraint_lambda_count_moves_early_plateau_back_to_target_index():
    schedule = np.ones((10, 7), dtype=np.float64)
    schedule[:, 6] = [0.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    out = set_restraint_lambda_count(schedule, n_restraint_lambdas=6)

    assert np.all(out[:5, 6] < 1.0)
    assert out[5, 6] == pytest.approx(1.0)
    assert np.all(out[5:, 6] == pytest.approx(1.0))


def test_set_restraint_lambda_count_validates_window_count():
    schedule = np.ones((4, 7), dtype=np.float64)

    with pytest.raises(ValueError, match="at least 2"):
        set_restraint_lambda_count(schedule, n_restraint_lambdas=1)

    with pytest.raises(ValueError, match="cannot exceed"):
        set_restraint_lambda_count(schedule, n_restraint_lambdas=5)


def _make_three_stage_schedule(
    n_replicas: int,
    rest_end_idx: int,
    charge_end_idx: int,
) -> np.ndarray:
    schedule = np.ones((n_replicas, 7), dtype=np.float64)
    schedule[:, 2] = 0.0
    schedule[:, 6] = 1.0
    schedule[:rest_end_idx + 1, 6] = np.linspace(0.0, 1.0, rest_end_idx + 1)
    schedule[:charge_end_idx + 1, 0] = np.linspace(1.0, 0.0, charge_end_idx + 1)
    schedule[charge_end_idx:, 0] = 0.0
    schedule[:, 1] = 1.0 - schedule[:, 0]
    schedule[:charge_end_idx + 1, 3] = 1.0
    schedule[charge_end_idx:, 3] = np.linspace(1.0, 0.0, n_replicas - charge_end_idx)
    schedule[:, 4] = 1.0 - schedule[:, 3]
    return schedule


def test_freeze_stage_lambda_counts_restores_reference_boundaries():
    reference = _make_three_stage_schedule(
        n_replicas=12,
        rest_end_idx=5,
        charge_end_idx=8,
    )
    candidate = _make_three_stage_schedule(
        n_replicas=12,
        rest_end_idx=2,
        charge_end_idx=6,
    )

    out = freeze_stage_lambda_counts(candidate, reference)

    assert infer_stage_boundaries(out) == infer_stage_boundaries(reference)
    assert out[5, 6] == pytest.approx(1.0)
    assert np.all(out[:5, 6] < 1.0)
    assert np.all(out[8:, 0] == pytest.approx(0.0))
    assert np.all(out[:8, 0] > 0.0)
    np.testing.assert_allclose(out[:, 1], 1.0 - out[:, 0])
    np.testing.assert_allclose(out[:, 2], 0.0)


def test_freeze_stage_lambda_counts_validates_shape():
    schedule = np.ones((5, 7), dtype=np.float64)
    reference = np.ones((6, 7), dtype=np.float64)

    with pytest.raises(ValueError, match="same shape"):
        freeze_stage_lambda_counts(schedule, reference)


@pytest.mark.parametrize(
    ("recipe_fn", "ligand_charge", "expected_windows"),
    [
        (get_neutral_water_leg_lambdas, 0, 24),
        (get_charged_water_leg_lambdas, 1, 32),
    ],
)
def test_freeze_stage_counts_supports_water_leg_without_restraint_phase(
    recipe_fn,
    ligand_charge,
    expected_windows,
):
    schedule = resample_water_leg_schedule(recipe_fn(), ligand_charge)

    frozen = freeze_stage_lambda_counts(schedule, schedule)

    assert frozen.shape == (expected_windows, 7)
    np.testing.assert_allclose(frozen, schedule)
