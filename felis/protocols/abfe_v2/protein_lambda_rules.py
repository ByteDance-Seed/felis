# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np

COL_A_CHARGE = 0
COL_B_CHARGE = 1
COL_C_CHARGE = 2
COL_RESTRAINT = 6
FULL_RESTRAINT_ENDPOINT = 1.0


def require_full_restraint_endpoint(
    endpoint: float,
    source: str = "restraint_lambda_end",
) -> float:
    """Reject endpoints that are incompatible with the Boresch correction."""
    try:
        endpoint = float(endpoint)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a numeric value; got {endpoint!r}") from exc
    if not np.isfinite(endpoint) or endpoint != FULL_RESTRAINT_ENDPOINT:
        raise ValueError(
            f"{source} must be 1.0 because the analytical Boresch correction "
            f"only supports a full restraint endpoint; got {endpoint}"
        )
    return endpoint


def validate_protein_lambda_schedule(
    schedule: list[list[float]] | np.ndarray,
    source: str = "protein-leg lambda schedule",
) -> None:
    """Require a protein-leg schedule to end with full Boresch restraints."""
    array = np.asarray(schedule, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 7:
        raise ValueError(f"{source} must have shape (N, 7) with N >= 1; got {array.shape}")
    require_full_restraint_endpoint(
        array[-1, COL_RESTRAINT],
        source=f"{source} restraint endpoint",
    )


def infer_stage_boundaries(schedule: np.ndarray, charge_zero_tol: float = 1e-6) -> tuple[int, int]:
    """Infer restraint-end and charge-end indices from an ABFE v2 schedule."""
    schedule = np.asarray(schedule, dtype=np.float64)
    if schedule.ndim != 2 or schedule.shape[1] != 7:
        raise ValueError(f"schedule must have shape (N, 7); got {schedule.shape}")

    restraint = schedule[:, COL_RESTRAINT]
    rest_max = float(restraint.max())
    rest_end_idx = int(np.where(restraint >= rest_max - 1e-9)[0][0])

    a_charge = schedule[:, COL_A_CHARGE]
    charge_zero_idx = np.where(a_charge <= charge_zero_tol)[0]
    charge_end_idx = int(charge_zero_idx[0]) if len(charge_zero_idx) > 0 else schedule.shape[0] - 1
    if charge_end_idx < rest_end_idx:
        charge_end_idx = rest_end_idx
    return rest_end_idx, charge_end_idx


def _stage_axis(
    n_replicas: int,
    rest_end_idx: int,
    charge_end_idx: int,
    target_rest_end_idx: int,
    target_charge_end_idx: int,
) -> np.ndarray:
    """Map indices from their current phase boundaries onto target boundaries."""
    axis = np.zeros(n_replicas, dtype=np.float64)
    last_idx = n_replicas - 1

    if rest_end_idx > 0:
        axis[:rest_end_idx + 1] = np.linspace(0.0, float(target_rest_end_idx), rest_end_idx + 1)
    else:
        axis[0] = 0.0

    if charge_end_idx > rest_end_idx:
        axis[rest_end_idx:charge_end_idx + 1] = np.linspace(
            float(target_rest_end_idx),
            float(target_charge_end_idx),
            charge_end_idx - rest_end_idx + 1,
        )
    else:
        axis[rest_end_idx] = float(target_charge_end_idx)

    if last_idx > charge_end_idx:
        axis[charge_end_idx:] = np.linspace(
            float(target_charge_end_idx),
            float(last_idx),
            last_idx - charge_end_idx + 1,
        )
    else:
        axis[last_idx] = float(last_idx)
    return axis


def freeze_stage_lambda_counts(
    schedule: np.ndarray,
    reference_schedule: np.ndarray,
    charge_zero_tol: float = 1e-6,
) -> np.ndarray:
    """Keep restraint, charge, and vdW stage window counts fixed.

    The optimizer may redistribute the progress coordinate so phase boundaries
    land at different replica indices. This transform reprojects a candidate
    schedule onto the phase boundaries inferred from ``reference_schedule``.
    """
    schedule = np.asarray(schedule, dtype=np.float64)
    reference_schedule = np.asarray(reference_schedule, dtype=np.float64)
    if schedule.shape != reference_schedule.shape:
        raise ValueError(
            f"schedule and reference_schedule must have the same shape; got {schedule.shape} and "
            f"{reference_schedule.shape}"
        )
    if schedule.ndim != 2 or schedule.shape[1] != 7:
        raise ValueError(f"schedule must have shape (N, 7); got {schedule.shape}")

    target_rest_end, target_charge_end = infer_stage_boundaries(reference_schedule, charge_zero_tol)
    current_rest_end, current_charge_end = infer_stage_boundaries(schedule, charge_zero_tol)
    n_replicas = schedule.shape[0]
    source_axis = _stage_axis(
        n_replicas,
        current_rest_end,
        current_charge_end,
        target_rest_end,
        target_charge_end,
    )
    target_axis = np.arange(n_replicas, dtype=np.float64)

    frozen = np.empty_like(schedule)
    for col in range(schedule.shape[1]):
        frozen[:, col] = np.interp(target_axis, source_axis, schedule[:, col])

    if target_rest_end > 0:
        frozen = set_restraint_lambda_count(frozen, target_rest_end + 1)
    else:
        # Water-leg schedules have no restraint turn-on phase: the restraint
        # column is constant and its inferred stage ends at index zero.
        frozen[:, COL_RESTRAINT] = reference_schedule[:, COL_RESTRAINT]
    frozen[target_charge_end:, COL_A_CHARGE] = 0.0
    if target_charge_end > 0:
        frozen[:target_charge_end, COL_A_CHARGE] = np.maximum(
            frozen[:target_charge_end, COL_A_CHARGE],
            charge_zero_tol * 10.0,
        )
    frozen[:, COL_A_CHARGE] = np.clip(frozen[:, COL_A_CHARGE], 0.0, 1.0)
    frozen[:, COL_B_CHARGE] = 1.0 - frozen[:, COL_A_CHARGE]
    frozen[:, COL_C_CHARGE] = 0.0
    frozen[:, 3:7] = np.clip(frozen[:, 3:7], 0.0, 1.0)
    frozen[0, :] = reference_schedule[0, :]
    frozen[-1, :] = reference_schedule[-1, :]
    return frozen


def set_restraint_lambda_count(schedule: np.ndarray, n_restraint_lambdas: int = 6) -> np.ndarray:
    """Return a schedule whose restraint turn-on spans exactly N windows."""
    if n_restraint_lambdas < 2:
        raise ValueError(f"n_restraint_lambdas must be at least 2; got {n_restraint_lambdas}")

    schedule = np.asarray(schedule, dtype=np.float64).copy()
    if schedule.ndim != 2 or schedule.shape[1] != 7:
        raise ValueError(f"schedule must have shape (N, 7); got {schedule.shape}")
    if n_restraint_lambdas > schedule.shape[0]:
        raise ValueError(
            f"n_restraint_lambdas ({n_restraint_lambdas}) cannot exceed schedule length ({schedule.shape[0]})"
        )

    restraint = schedule[:, 6]
    rest_max = float(restraint.max())
    plateau_idx = int(np.where(restraint >= rest_max - 1e-9)[0][0])
    if plateau_idx == n_restraint_lambdas - 1:
        return schedule

    old_x = np.linspace(0.0, 1.0, plateau_idx + 1)
    new_x = np.linspace(0.0, 1.0, n_restraint_lambdas)
    schedule[:n_restraint_lambdas, 6] = np.interp(new_x, old_x, restraint[:plateau_idx + 1])
    schedule[n_restraint_lambdas - 1:, 6] = rest_max
    return schedule
