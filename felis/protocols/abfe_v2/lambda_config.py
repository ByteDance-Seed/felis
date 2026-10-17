# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)

NEUTRAL_WATER_LEG_WINDOWS = 24
CHARGED_WATER_LEG_WINDOWS = 32
N_LAMBDA_COLUMNS = 7
COL_A_CHARGE = 0
COL_B_CHARGE = 1
COL_C_CHARGE = 2


def get_water_leg_window_count(ligand_charge: int) -> int:
    """Return the fixed optimization window count for a ligand charge."""
    if ligand_charge == 0:
        return NEUTRAL_WATER_LEG_WINDOWS
    return CHARGED_WATER_LEG_WINDOWS


def resample_lambda_schedule(
    schedule: list[list[float]] | np.ndarray,
    n_windows: int,
) -> np.ndarray:
    """Resample an ABFE v2 path to an exact number of lambda windows.

    PCHIP preserves the shape of every lambda column while repositioning the
    fixed number of windows on an evenly spaced path coordinate. Endpoints are
    pinned, the charge complement is restored, and ``lambdaB_vdw`` remains
    independent of ``lambdaA_vdw``.
    """
    array = np.asarray(schedule, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != N_LAMBDA_COLUMNS:
        raise ValueError(
            f"schedule must have shape (N, {N_LAMBDA_COLUMNS}); "
            f"got {array.shape}"
        )
    if array.shape[0] < 2:
        raise ValueError("schedule must contain at least 2 windows")
    if n_windows < 2:
        raise ValueError(f"n_windows must be at least 2; got {n_windows}")

    source_s = np.linspace(0.0, 1.0, array.shape[0])
    target_s = np.linspace(0.0, 1.0, n_windows)
    resampled = np.empty((n_windows, N_LAMBDA_COLUMNS), dtype=np.float64)
    for column in range(N_LAMBDA_COLUMNS):
        interpolator = PchipInterpolator(source_s, array[:, column])
        resampled[:, column] = interpolator(target_s)

    resampled[0, :] = array[0, :]
    resampled[-1, :] = array[-1, :]
    resampled[:, COL_B_CHARGE] = 1.0 - resampled[:, COL_A_CHARGE]
    resampled[:, COL_C_CHARGE] = 0.0
    return resampled


def resample_water_leg_schedule(
    schedule: list[list[float]] | np.ndarray,
    ligand_charge: int,
) -> np.ndarray:
    """Resample a water-leg path to its charge-dependent fixed count."""
    return resample_lambda_schedule(
        schedule,
        get_water_leg_window_count(ligand_charge),
    )


def read_lambda_schedule(path: str | Path) -> list[list[float]]:
    """Read an ABFE v2 lambda schedule from JSON config."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("ab", {}).get("lam_list")
    if not isinstance(data, list):
        raise ValueError(f"lambda schedule file {path} must contain a list or ab.lam_list")

    schedule = []
    for idx, row in enumerate(data):
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError(f"lambda schedule row {idx} in {path} must be a list of 7 values")
        schedule.append([float(value) for value in row])
    return schedule


def select_lambda_schedule(
    configured_path: str | Path | None,
    optimized_path: str | Path,
    recipe_lambdas_fn,
    schedule_name: str,
) -> list[list[float]]:
    """Select configured, optimized, or recipe lambda schedule in priority order."""
    if configured_path is not None:
        logger.info(f"Using configured {schedule_name} lambda schedule: {configured_path}")
        return read_lambda_schedule(configured_path)
    if Path(optimized_path).is_file():
        logger.info(f"Using optimized lambda schedule: {optimized_path}")
        return read_lambda_schedule(optimized_path)

    logger.info("Configured or optimized lambda schedule not found; using default recipe")
    return recipe_lambdas_fn()
