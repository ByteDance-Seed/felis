# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Lambda-schedule optimizer for ABFE v2.

The schedule is a 2-D array of shape ``(N_replica, 7)`` whose columns are::

    [lambdaA_charge, lambdaB_charge, lambdaC_charge,
     lambdaA_vdw,    lambdaB_vdw,    torsionLambda,    restraintLambda]

Constraints enforced by this module:

* ``lambdaB_charge = 1 - lambdaA_charge`` (always enforced, in both
  parametric and path-preserving modes).
* ``lambdaB_vdw    = 1 - lambdaA_vdw`` (parametric mode only; in
  path-preserving mode ``lambdaB_vdw`` is taken from the initial
  schedule and is *not* constrained to ``1 - lambdaA_vdw``).
* ``lambdaC_charge = 0``
* ``torsionLambda`` and ``restraintLambda`` are taken from the initial
  schedule and left untouched.
* ``lambdaA_vdw`` only starts decaying after ``lambdaA_charge`` has
  reached zero (two-phase schedule with a split point ``t_split``).
* ``lambdaA_charge`` and ``lambdaA_vdw`` follow normalized sigmoid
  decays on their respective phases, anchored to hit exactly 1 and 0
  at the phase endpoints.

Optimization strategy: introduce a 1-D progress coordinate
``s`` with one value per replica, ``s_0 = 0`` and ``s_{N-1} = 1``, and
build the schedule from ``s``. The default optimization target is a uniform
adjacent-pair exchange-acceptance profile: after each HREMD iteration, the
grid is reshaped from acceptance-derived thermodynamic distances so replicas
are concentrated at low-acceptance bottlenecks. A hard acceptance floor is
enforced, and the number of replicas is held fixed (no adaptive
insertion/removal of windows). Trebst-Katzgraber replica-flow diagnostics are
still measured and can be used as an explicit alternative objective.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)


# Column indices in the ``(N, 7)`` schedule.
COL_A_CHARGE = 0
COL_B_CHARGE = 1
COL_C_CHARGE = 2
COL_A_VDW = 3
COL_B_VDW = 4
COL_TORSION = 5
COL_RESTRAINT = 6
N_COLUMNS = 7
DEFAULT_MIN_ACCEPTANCE_RATIO = 0.2


# A run_hremd callable takes a full ``(N, 7)`` lambda schedule and returns
# the energy records list as produced by ``read_records`` on the HREMD
# output ``records.sqlite``.
RunHremdFn = Callable[[np.ndarray], list]


@dataclass
class SigmoidShape:
    """Parameters of a normalized sigmoid decay on ``[0, 1]``.

    The unnormalized sigmoid is ``raw(u) = 1 / (1 + exp(k * (u - x0)))``;
    after normalization the decay hits exactly ``1`` at ``u = 0`` and
    exactly ``0`` at ``u = 1``.

    * ``k`` controls steepness (larger ``k`` -> sharper transition).
    * ``x0`` controls the inflection location, in ``[0, 1]``.
    """

    k: float = 8.0
    x0: float = 0.5


@dataclass
class LambdaScheduleParams:
    """Static parameters describing the parametric schedule.

    ``torsion`` and ``restraint`` are length-``N`` arrays copied from the
    initial schedule and reused verbatim across iterations.

    When ``initial_schedule`` is provided, the thermodynamic path is preserved
    from that schedule (including ``lambdaB`` values are not forced to
    ``1 - lambdaA``), and replicas are repositioned along this path by
    interpolating from the initial schedule.
    """

    t_split: float
    charge_shape: SigmoidShape
    vdw_shape: SigmoidShape
    torsion: np.ndarray
    restraint: np.ndarray
    initial_s: Optional[np.ndarray] = None
    initial_schedule: Optional[np.ndarray] = None
    # Path-shape controls, only meaningful in path-preserving mode.
    # ``vdwb_gamma`` is a scalar curvature knob for the free lambdaB_vdw
    # decay curve (1.0 == identity). ``canonical_t_split`` (``c0``) is the
    # charge->vdW phase boundary location in the initial schedule's ``s``
    # space; it anchors the ``t_split`` s-warp so that ``t_split == c0``
    # reproduces the un-warped path.
    vdwb_gamma: float = 1.0
    canonical_t_split: Optional[float] = None

    @property
    def n_replicas(self) -> int:
        return len(self.torsion)

    @property
    def use_path_interpolation(self) -> bool:
        """True when we should interpolate from the initial schedule path
        instead of using the sigmoid parametric form.
        """
        return self.initial_schedule is not None


@dataclass
class IterationDiagnostics:
    """Per-iteration record kept by the optimizer.

    The default optimization target is a uniform adjacent-pair exchange
    acceptance profile, subject to a hard minimum acceptance constraint.
    Flow diagnostics are retained for validation and the optional flow target.
    """

    iteration: int
    s: np.ndarray
    schedule: np.ndarray  # shape (N, 7)
    flow_profile: Optional[np.ndarray] = None
    flow_drops: Optional[np.ndarray] = None
    flow_drop_range: Optional[float] = None
    flow_drop_std: Optional[float] = None
    flow_drop_mean: Optional[float] = None
    flow_linearity_error: Optional[float] = None
    round_trip_rate: Optional[float] = None
    acceptance_ratios: Optional[np.ndarray] = None
    acceptance_mean: Optional[float] = None
    acceptance_std: Optional[float] = None
    acceptance_range: Optional[float] = None
    min_acceptance_ratio: Optional[float] = None
    acceptance_feasible: Optional[bool] = None
    placement_metric: Optional[str] = None
    diffusivity: Optional[np.ndarray] = None
    proposed_s: Optional[np.ndarray] = None
    # Joint path-shape optimization state for path-preserving flow mode.
    t_split: Optional[float] = None
    vdwb_gamma: Optional[float] = None
    rtr_rel_improvement: Optional[float] = None
    charge_phase_flow: Optional[float] = None
    vdw_phase_flow: Optional[float] = None
    phase_flow_imbalance: Optional[float] = None
    accepted_nudge: Optional[str] = None


@dataclass
class OptimizationResult:
    """Final return value of :func:`optimize_lambda_schedule`."""

    schedule: np.ndarray  # shape (N, 7)
    s: np.ndarray
    params: LambdaScheduleParams
    converged: bool
    n_iterations: int
    selected_iteration: int
    acceptance_threshold: float
    history: List[IterationDiagnostics] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sigmoid utilities
# ---------------------------------------------------------------------------


def _normalized_sigmoid_decay(x: np.ndarray, shape: SigmoidShape) -> np.ndarray:
    """Sigmoid decay from 1 (at x=0) to 0 (at x=1), anchored exactly."""
    x = np.asarray(x, dtype=np.float64)
    k, x0 = shape.k, shape.x0
    raw = 1.0 / (1.0 + np.exp(k * (x - x0)))
    raw_left = 1.0 / (1.0 + np.exp(k * (0.0 - x0)))
    raw_right = 1.0 / (1.0 + np.exp(k * (1.0 - x0)))
    denom = raw_left - raw_right
    if denom <= 0:
        raise ValueError(
            "Sigmoid is not strictly decreasing on [0, 1]; check k > 0."
        )
    return (raw - raw_right) / denom


def _a_charge_a_vdw_from_s(
    s: np.ndarray, params: LambdaScheduleParams
) -> tuple:
    """Map progress ``s`` to ``(A_charge, A_vdw)`` arrays.

    * For ``s <= t_split``: ``A_charge`` follows a sigmoid decay from 1
      to 0 on the rescaled coordinate ``s / t_split``; ``A_vdw = 1``.
    * For ``s >= t_split``: ``A_charge = 0``; ``A_vdw`` follows a sigmoid
      decay from 1 to 0 on ``(s - t_split) / (1 - t_split)``.
    """
    s = np.asarray(s, dtype=np.float64)
    t_split = params.t_split
    if not 0.0 < t_split < 1.0:
        raise ValueError(f"t_split must be in (0, 1); got {t_split}")

    a_charge = np.empty_like(s)
    a_vdw = np.empty_like(s)

    in_charge = s <= t_split
    in_vdw = ~in_charge

    if np.any(in_charge):
        u = s[in_charge] / t_split
        a_charge[in_charge] = _normalized_sigmoid_decay(u, params.charge_shape)
        a_vdw[in_charge] = 1.0

    if np.any(in_vdw):
        u = (s[in_vdw] - t_split) / (1.0 - t_split)
        a_charge[in_vdw] = 0.0
        a_vdw[in_vdw] = _normalized_sigmoid_decay(u, params.vdw_shape)

    # Hard pin endpoints to remove tiny numerical residues.
    if s[0] <= 0.0:
        a_charge[0] = 1.0
        a_vdw[0] = 1.0
    if s[-1] >= 1.0:
        a_charge[-1] = 0.0
        a_vdw[-1] = 0.0

    return a_charge, a_vdw


def _apply_vdwb_gamma(vdwb: np.ndarray, gamma: float) -> np.ndarray:
    """Reshape a monotone ``lambdaB_vdw`` curve with a scalar curvature knob.

    The curve is normalized to ``[0, 1]`` (with its own endpoints mapped to
    1 and 0), raised to the power ``gamma``, and denormalized. Because
    ``u -> u ** gamma`` is monotone on ``[0, 1]`` for any ``gamma > 0`` and
    fixes ``0`` and ``1``, the transform preserves monotonicity and pins both
    endpoints exactly. ``gamma == 1`` is the identity; ``gamma > 1`` makes the
    normalized curve decay faster early, ``gamma < 1`` slower. Direction is
    inferred from the endpoints, so it works whether ``vdwb`` rises or falls.
    """
    vdwb = np.asarray(vdwb, dtype=np.float64)
    if gamma <= 0.0:
        raise ValueError(f"vdwb gamma must be > 0; got {gamma}")
    a = vdwb[0]
    b = vdwb[-1]
    if abs(a - b) < 1e-12 or abs(gamma - 1.0) < 1e-12:
        return vdwb.copy()
    u = (vdwb - b) / (a - b)
    u = np.clip(u, 0.0, 1.0)
    u_g = u ** gamma
    return b + (a - b) * u_g


def _warp_s_for_t_split(
    s: np.ndarray, t_split: float, c0: float, min_gap: float = 1e-3
) -> np.ndarray:
    """Monotone two-segment reparameterization moving the phase boundary.

    Returns the canonical-``s`` coordinates at which the initial schedule
    should be interpolated so that the charge->vdW phase boundary (fixed at
    ``c0`` in the initial path) is presented at progress ``t_split`` in the
    new grid. The map ``W`` is continuous, strictly increasing, and satisfies
    ``W(0) = 0``, ``W(1) = 1``, ``W(t_split) = c0``. It is the identity when
    ``t_split == c0``. Increasing ``t_split`` allocates more replicas to the
    charge phase, decreasing it favors the vdW phase.
    """
    s = np.asarray(s, dtype=np.float64)
    t_split = float(np.clip(t_split, min_gap, 1.0 - min_gap))
    c0 = float(np.clip(c0, min_gap, 1.0 - min_gap))
    warped = np.empty_like(s)
    lower = s <= t_split
    warped[lower] = c0 * (s[lower] / t_split)
    upper = ~lower
    warped[upper] = c0 + (1.0 - c0) * (s[upper] - t_split) / (1.0 - t_split)
    warped[0] = s[0]
    warped[-1] = s[-1]
    return warped


def build_schedule(
    s: np.ndarray, params: LambdaScheduleParams
) -> np.ndarray:
    """Build the full ``(N, 7)`` schedule from ``s`` and parameters.

    When ``params.use_path_interpolation`` is True (i.e. an initial schedule
    was provided), lambda values are interpolated from the initial schedule
    along the ``s`` coordinate, preserving the original thermodynamic path.
    The charge B coordinate is re-derived as ``lambdaB_charge = 1 -
    lambdaA_charge`` (a strict complementary constraint), while
    ``lambdaB_vdw`` is left as interpolated from the path and is *not*
    constrained to ``1 - lambdaA_vdw``.

    Otherwise, the sigmoid parametric form is used, which forces both
    ``lambdaB_charge = 1 - lambdaA_charge`` and ``lambdaB_vdw = 1 - lambdaA_vdw``.
    """
    s = np.asarray(s, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError("s must be 1-D")
    if len(s) != params.n_replicas:
        raise ValueError(
            f"len(s) ({len(s)}) must equal params.n_replicas "
            f"({params.n_replicas})"
        )

    if params.use_path_interpolation:
        # Interpolate all lambda columns from the initial schedule along s.
        # This preserves the original thermodynamic path exactly. When a
        # canonical phase boundary ``c0`` is known, ``s`` is first warped so
        # that ``params.t_split`` controls where the charge->vdW boundary
        # falls; ``t_split == c0`` gives the identity (un-warped) path.
        s_init = params.initial_s
        schedule_init = params.initial_schedule
        if params.canonical_t_split is not None:
            s_interp = _warp_s_for_t_split(
                s, params.t_split, params.canonical_t_split
            )
        else:
            s_interp = s
        schedule = np.zeros((params.n_replicas, N_COLUMNS), dtype=np.float64)
        for col in range(N_COLUMNS):
            interpolator = PchipInterpolator(s_init, schedule_init[:, col])
            schedule[:, col] = interpolator(s_interp)
        # Hard pin endpoints to match initial schedule exactly
        schedule[0, :] = schedule_init[0, :]
        schedule[-1, :] = schedule_init[-1, :]
        # Enforce qB = 1 - qA (strict complementary charge constraint);
        # lambdaB_vdw is intentionally left as interpolated (not forced to
        # 1 - lambdaA_vdw).
        schedule[:, COL_B_CHARGE] = 1.0 - schedule[:, COL_A_CHARGE]
        # Ensure C_charge stays 0
        schedule[:, COL_C_CHARGE] = 0.0
        # Scalar curvature reshape of the free lambdaB_vdw curve.
        schedule[:, COL_B_VDW] = _apply_vdwb_gamma(
            schedule[:, COL_B_VDW], params.vdwb_gamma
        )
        # Re-pin endpoints (gamma reshape keeps them, but guard rounding).
        schedule[0, :] = schedule_init[0, :]
        schedule[-1, :] = schedule_init[-1, :]
        # Re-apply strict charge constraints at the endpoints in case the initial
        # schedule had tiny numerical drift.
        schedule[[0, -1], COL_B_CHARGE] = 1.0 - schedule[[0, -1], COL_A_CHARGE]
        schedule[[0, -1], COL_C_CHARGE] = 0.0
        return schedule

    # Original parametric path: sigmoid + lambdaB = 1 - lambdaA
    a_charge, a_vdw = _a_charge_a_vdw_from_s(s, params)
    schedule = np.zeros((params.n_replicas, N_COLUMNS), dtype=np.float64)
    schedule[:, COL_A_CHARGE] = a_charge
    schedule[:, COL_B_CHARGE] = 1.0 - a_charge
    schedule[:, COL_C_CHARGE] = 0.0
    schedule[:, COL_A_VDW] = a_vdw
    schedule[:, COL_B_VDW] = 1.0 - a_vdw
    schedule[:, COL_TORSION] = params.torsion
    schedule[:, COL_RESTRAINT] = params.restraint
    return schedule


def infer_t_split_from_schedule(
    schedule: np.ndarray, charge_zero_tol: float = 1e-6
) -> float:
    """Estimate ``t_split`` as the fractional index where A_charge first hits 0.

    Falls back to ``0.5`` if A_charge never reaches zero or if the
    crossing index would land at the very last replica.
    """
    a_charge = np.asarray(schedule)[:, COL_A_CHARGE]
    n = len(a_charge)
    zero_idx = np.where(a_charge <= charge_zero_tol)[0]
    if len(zero_idx) == 0 or zero_idx[0] == 0 or zero_idx[0] == n - 1:
        return 0.5
    return float(zero_idx[0]) / float(n - 1)


# ---------------------------------------------------------------------------
# Flow helpers
# ---------------------------------------------------------------------------


def _compute_metrics_for_schedule(
    schedule: np.ndarray,
    run_hremd_fn: RunHremdFn,
    flow_linearity_norm: str = "l2",
    flow_smooth: bool = True,
):
    """Run a short HREMD and return flow diagnostics.

    Returns ``(flow_profile, flow_drops, flow_linearity_error,
    round_trip_rate, acceptance_ratios)``.

    ``flow_profile`` and ``flow_linearity_error`` are computed on the *raw*
    (unsmoothed) flow so the linearity diagnostic stays honest. ``flow_drops``
    are computed on the monotone-smoothed flow when ``flow_smooth`` is True, so
    the feedback-optimized placement is driven by a denoised ``df``.
    """
    from felis.protocols.abfe_v2.utilities import (
        get_acceptance_ratios_from_records,
        get_flow_drops_from_records,
        get_flow_linearity_error,
        get_flow_profile_from_records,
        get_round_trip_rate_from_records,
    )

    energy_records = run_hremd_fn(schedule)
    flow_profile = get_flow_profile_from_records(energy_records)
    flow_drops = get_flow_drops_from_records(energy_records, smooth=flow_smooth)
    flow_linearity_error = get_flow_linearity_error(
        energy_records, norm=flow_linearity_norm
    )
    round_trip_rate = get_round_trip_rate_from_records(energy_records)
    acceptance_ratios = get_acceptance_ratios_from_records(energy_records)
    return (
        flow_profile,
        flow_drops,
        flow_linearity_error,
        round_trip_rate,
        acceptance_ratios,
    )


# ---------------------------------------------------------------------------
# 1-D progress-coordinate reschedule (Option A)
# ---------------------------------------------------------------------------


def diffusivity_profile(
    s: np.ndarray, flow_drops: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Per-pair local diffusivity of the replica walk along ``s``.

    Derived exactly as in Katzgraber-Trebst-Huse-Troyer (cond-mat/0602085),
    transposed from temperature ``T`` to our 1-D progress coordinate ``s``.
    Their steady-state diffusion of the up-walker fraction ``f`` gives a
    constant current ``j = D(s) * eta(s) * (df/ds)``, where ``eta(s)`` is the
    replica density. On the current grid the density of one replica per
    interval is ``eta_i ~ 1 / (N * Delta s_i)``, and using the per-pair drop
    ``Delta f_i`` as the discrete ``df`` over interval ``i`` yields (their
    Eq. 6, ``D ~ Delta T / (df/dT)``)::

        D_i  ~  Delta s_i / Delta f_i

    Larger ``D_i`` means the walk moves easily across boundary ``i``; small
    ``D_i`` (large drop over a wide ``s`` interval) marks a transport
    bottleneck. The feedback-optimized density ``eta_opt ~ 1 / sqrt(D)`` then
    reproduces the ``sqrt(Delta f)`` placement cost used in
    :func:`reschedule_s`.

    Args:
        s: Strictly increasing progress coordinate of length ``N``.
        flow_drops: Per-pair (length ``N-1``) non-negative flow drops.
        normalize: If True, scale so the mean diffusivity is 1 (only relative
            values are meaningful; the current ``j`` and constants are
            unknown).

    Returns:
        Length ``N-1`` array of per-pair diffusivities (boundary ``i`` between
        replica ``i`` and ``i+1``).
    """
    s = np.asarray(s, dtype=np.float64)
    flow_drops = np.asarray(flow_drops, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError("s must be 1-D")
    if len(flow_drops) != len(s) - 1:
        raise ValueError(
            "len(flow_drops) must equal len(s) - 1; "
            f"got {len(flow_drops)} and {len(s)}"
        )
    ds = np.diff(s)
    diffusivity = ds / flow_drops
    if normalize:
        mean = float(np.mean(diffusivity))
        if mean > 0:
            diffusivity = diffusivity / mean
    return diffusivity


def reschedule_s(
    s: np.ndarray,
    neighbor_metrics: np.ndarray,
    n_new: Optional[int] = None,
    min_gap: float = 1e-3,
    flow_drop_floor_frac: float = 0.25,
) -> np.ndarray:
    """Resample ``s`` using feedback-optimized flow drops.

    ``neighbor_metrics`` are per-pair Trebst-Katzgraber flow drops
    ``Delta f_i``. The placement cost is ``sqrt(Delta f_i)`` so replica
    density follows ``sqrt(|df/ds|)``. Small drops are floored at
    ``flow_drop_floor_frac / (N-1)`` so near-linear regions keep some
    replica density.
    """
    s = np.asarray(s, dtype=np.float64)
    metrics = np.asarray(neighbor_metrics, dtype=np.float64)

    if s.ndim != 1:
        raise ValueError("s must be 1-D")
    if metrics.ndim != 1:
        raise ValueError("neighbor_metrics must be 1-D")
    if len(metrics) != len(s) - 1:
        raise ValueError(
            "len(neighbor_metrics) must equal len(s) - 1; "
            f"got {len(metrics)} and {len(s)}"
        )
    if not np.all(np.diff(s) > 0):
        raise ValueError("s must be strictly increasing")
    if flow_drop_floor_frac <= 0.0:
        raise ValueError(
            f"flow_drop_floor_frac must be > 0; got {flow_drop_floor_frac}"
        )

    if n_new is None:
        n_new = len(s)
    if n_new < 2:
        raise ValueError("n_new must be at least 2")

    ideal_drop = 1.0 / (len(s) - 1)
    flow_floor = flow_drop_floor_frac * ideal_drop
    m_clipped = np.clip(metrics, flow_floor, None)
    d = np.sqrt(m_clipped)
    s_cum = np.concatenate([[0.0], np.cumsum(d)])

    s_of_cum = PchipInterpolator(s_cum, s)
    cum_new = np.linspace(0.0, s_cum[-1], n_new)
    s_new = s_of_cum(cum_new)

    s_new[0] = s[0]
    s_new[-1] = s[-1]
    s_new = np.sort(s_new)

    # Enforce a minimum gap between consecutive points.
    for i in range(1, len(s_new) - 1):
        if s_new[i] - s_new[i - 1] < min_gap:
            s_new[i] = s_new[i - 1] + min_gap
    for i in range(len(s_new) - 2, 0, -1):
        if s_new[i + 1] - s_new[i] < min_gap:
            s_new[i] = s_new[i + 1] - min_gap

    if not np.all(np.diff(s_new) > 0):
        raise RuntimeError(
            "reschedule_s produced a non-monotonic schedule; "
            f"got: {s_new}"
        )

    return s_new


def _acceptance_reschedule_metrics(
    acceptance_ratios: np.ndarray,
) -> np.ndarray:
    """Convert acceptance ratios to normalized thermodynamic distances.

    ``-log(p_i)`` grows as the adjacent-pair acceptance ``p_i`` decreases.
    Normalizing the distances to sum to one makes them compatible with the
    flow-drop scaling expected by :func:`reschedule_s`. This keeps the replica
    count fixed while moving density toward low-acceptance pairs, including
    bottlenecks at either endpoint.
    """
    ratios = np.asarray(acceptance_ratios, dtype=np.float64)
    if ratios.ndim != 1 or len(ratios) == 0:
        raise ValueError("acceptance_ratios must be a non-empty 1-D array")
    if not np.all(np.isfinite(ratios)):
        raise ValueError("acceptance_ratios must contain only finite values")
    if np.any((ratios < 0.0) | (ratios > 1.0)):
        raise ValueError("acceptance_ratios must be in [0, 1]")

    clipped = np.clip(ratios, np.finfo(np.float64).tiny, 1.0)
    distances = -np.log(clipped)
    total = float(distances.sum())
    if total <= 0.0:
        return np.full(len(ratios), 1.0 / len(ratios), dtype=np.float64)
    return distances / total


def _acceptance_profile_score(
    acceptance_ratios: np.ndarray,
) -> tuple:
    """Return a lexicographic score for the acceptance target.

    A balanced replica ladder has similar exchange acceptance at every
    adjacent pair. The leading term therefore minimizes the pairwise
    acceptance standard deviation. For equally balanced profiles, favor the
    schedule with the larger worst-pair acceptance, followed by the larger
    mean acceptance. The hard floor is handled separately, so callers should
    only compare feasible schedules with this score.
    """
    ratios = np.asarray(acceptance_ratios, dtype=np.float64)
    return (
        -float(ratios.std()),
        float(ratios.min()),
        float(ratios.mean()),
    )


def _propose_path_shape_nudge(
    params: LambdaScheduleParams,
    s: np.ndarray,
    measured_t_split: float,
    flow_drops: np.ndarray,
    phase_flow_imbalance: float,
    charge_phase_flow: float,
    vdw_phase_flow: float,
    baseline_rtr: float,
    t_split_step: float,
    vdwb_gamma_step: float,
    imbalance_threshold: float,
    min_gap: float,
) -> Optional[dict]:
    """Apply one greedy path-shape nudge in place and return a pending record.

    Chooses a single knob to move for the *next* iteration based on the
    measured flow drops:

    * If the phase-flow imbalance between the charge and vdW phases is large,
      move ``t_split`` toward the heavier phase (a positive imbalance means the
      charge phase carries more flow drop, so ``t_split`` is increased to give
      it more replicas).
    * Otherwise, if the dominant bottleneck sits in the vdW phase, reshape the
      free ``lambdaB_vdw`` curve via a multiplicative ``vdwb_gamma`` nudge
      (increase gamma when the bottleneck sits early in the vdW decay,
      decrease when late).

    ``s`` and ``measured_t_split`` must describe the grid and phase boundary
    used to produce ``flow_drops``. The chosen knob is mutated on ``params``
    and a record describing the previous value plus the baseline round-trip
    rate is returned so the caller can accept or revert it after the next
    measurement. Returns ``None`` when no sensible nudge is available.
    """
    total_flow = charge_phase_flow + vdw_phase_flow
    if total_flow <= 0.0:
        return None
    norm_imbalance = phase_flow_imbalance / total_flow

    if abs(norm_imbalance) >= imbalance_threshold:
        # Rebalance the phases by moving the boundary toward the heavier phase.
        old_value = float(params.t_split)
        direction = 1.0 if norm_imbalance > 0.0 else -1.0
        new_value = float(
            np.clip(old_value + direction * t_split_step,
                    min_gap, 1.0 - min_gap)
        )
        if abs(new_value - old_value) < 1e-9:
            return None
        params.t_split = new_value
        return {"knob": "t_split", "old_value": old_value,
                "baseline_rtr": baseline_rtr}

    # Otherwise target the vdW-phase bottleneck via the gamma curvature knob.
    charge_mask = s[:-1] < measured_t_split
    vdw_idx = np.where(~charge_mask)[0]
    if len(vdw_idx) == 0:
        return None
    vdw_drops = flow_drops[vdw_idx]
    bottleneck_local = int(np.argmax(vdw_drops))
    # Fractional position of the bottleneck within the vdW phase.
    frac = bottleneck_local / max(len(vdw_idx) - 1, 1)
    # Early bottleneck (frac < 0.5) -> increase gamma (faster early decay);
    # late bottleneck -> decrease gamma.
    sign = 1.0 if frac < 0.5 else -1.0
    old_value = float(params.vdwb_gamma)
    new_value = float(old_value * np.exp(sign * vdwb_gamma_step))
    new_value = float(np.clip(new_value, 0.1, 10.0))
    if abs(new_value - old_value) < 1e-9:
        return None
    params.vdwb_gamma = new_value
    return {"knob": "vdwb_gamma", "old_value": old_value,
            "baseline_rtr": baseline_rtr}


# ---------------------------------------------------------------------------
# Top-level optimizer
# ---------------------------------------------------------------------------


def optimize_lambda_schedule(
    initial_schedule,
    run_hremd_fn: RunHremdFn,
    t_split: Optional[float] = None,
    charge_shape: Optional[SigmoidShape] = None,
    vdw_shape: Optional[SigmoidShape] = None,
    initial_s: Optional[np.ndarray] = None,
    use_path_from_initial_schedule: bool = False,
    flow_linearity_norm: str = "l2",
    flow_smooth: bool = True,
    flow_drop_floor_frac: float = 0.25,
    min_acceptance_ratio: float = DEFAULT_MIN_ACCEPTANCE_RATIO,
    optimization_target: str = "acceptance",
    acceptance_plateau_tol: float = 0.01,
    rtr_plateau_tol: float = 0.05,
    patience: int = 2,
    optimize_path_shape: bool = True,
    t_split_step: float = 0.05,
    vdwb_gamma_step: float = 0.15,
    max_iter: int = 8,
    eta: float = 0.5,
    min_gap: float = 1e-3,
    schedule_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> OptimizationResult:
    """Optimize an ABFE v2 ``(N, 7)`` lambda schedule.

    With the default ``optimization_target="acceptance"``, the optimizer
    seeks an equally distributed adjacent-pair exchange acceptance profile,
    subject to the hard minimum ``min_acceptance_ratio``. Replica placement is
    driven by ``-log(acceptance_ratio)``, which allocates more density to the
    difficult parts of the thermodynamic path. The legacy
    ``optimization_target="flow"`` retains Trebst-Katzgraber flow-driven
    placement and round-trip-rate selection.

    **Two modes of operation:**

    1. **Parametric sigmoid mode** (default): ``lambdaA_charge`` and
       ``lambdaA_vdw`` are constructed from a 1-D progress coordinate ``s``
       through normalized sigmoid decays. ``lambdaB`` is forced to
       ``1 - lambdaA``. Only ``torsionLambda`` and ``restraintLambda`` are
       copied from ``initial_schedule``.

    2. **Path-preserving mode** (``use_path_from_initial_schedule=True``):
       The full thermodynamic path is preserved from ``initial_schedule``
       (including any ``lambdaB != 1 - lambdaA`` relationship). Replica
       positions along the path are optimized by interpolating from the
       initial schedule. This is the correct mode to use when loading a
       schedule from ``--initial-lam-json``.

    Per iteration ``k``:

    1. Build the ``(N, 7)`` schedule from the current ``s``.
    2. Run a (short) HREMD via ``run_hremd_fn`` and compute flow metrics.
    3. Check that every adjacent-pair acceptance ratio is strictly greater
       than ``min_acceptance_ratio``.
    4. Stop when the acceptance constraint is satisfied and the chosen
       objective plateaus.
    5. Propose a new ``s`` via :func:`reschedule_s`, using acceptance-derived
       distances for the acceptance target or flow drops for the flow target
       after the acceptance constraint is satisfied.
    6. Damped update: ``s <- (1 - eta) * s + eta * proposed``, with
       endpoints pinned to ``[0, 1]`` and a ``min_gap`` floor.

    Parameters
    ----------
    initial_schedule:
        Array-like of shape ``(N, 7)``. In parametric mode, only
        ``torsionLambda`` and ``restraintLambda`` columns are used. In
        path-preserving mode, the full schedule defines the thermodynamic
        path.
    run_hremd_fn:
        Callable taking the full ``(N, 7)`` schedule and returning the
        HREMD energy records list.
    t_split:
        Phase-split point in ``s`` space. Inferred from
        ``initial_schedule`` if ``None``.
    charge_shape, vdw_shape:
        Sigmoid shape parameters for each phase (parametric mode only).
        Default to ``SigmoidShape(k=8.0, x0=0.5)``.
    initial_s:
        Optional initial progress coordinate of length ``N`` with
        ``s[0] = 0``, ``s[-1] = 1``. In path-preserving mode, defaults to
        ``linspace(0, 1, N)`` so that the first iteration uses the exact
        ``initial_schedule``.
    use_path_from_initial_schedule:
        If True, preserve the full thermodynamic path from
        ``initial_schedule`` and optimize only replica positions along
        this path. Use this when loading a schedule from a JSON file.
    flow_linearity_norm:
        Norm used to measure the flow profile's deviation from the ideal
        linear profile ``f_ideal(H) = 1 - H/(N-1)``.
        ``"l2"`` (default) is the root-mean-square deviation (robust to a
        single noisy pair on short runs); ``"linf"`` is the maximum absolute
        deviation (stricter, every window must be close to the line). This is
        recorded as a diagnostic (and used as a tiebreak in best-iteration
        selection); it does not gate convergence.
    flow_smooth:
        If True (default), the per-pair flow drops that drive the
        feedback-optimized replica placement are computed from a
        monotone-regularized (isotonic, decreasing) flow profile. This
        denoises ``df`` on short runs, mirroring the linear-regression fit of
        ``df/dT`` in cond-mat/0602085. The linearity-error diagnostic always
        uses the raw (unsmoothed) profile regardless of this flag.
    flow_drop_floor_frac:
        Lower bound, as a fraction of the ideal uniform drop ``1/(N-1)``,
        applied to the per-pair flow drops before the ``sqrt`` placement cost.
        This guarantees near-linear regions keep a minimum replica density so
        their BAR overlap does not collapse; smaller values approach the
        unconstrained feedback optimum (stronger crowding at the bottleneck,
        more risk of low-overlap pairs). The densest-to-sparsest spacing ratio
        is bounded near
        ``sqrt(Delta f_max / (flow_drop_floor_frac / (N-1)))``. Default 0.25.
    min_acceptance_ratio:
        Hard lower bound for every measured adjacent-pair exchange acceptance
        ratio. Feasibility requires ``acceptance > min_acceptance_ratio`` for
        every pair; equality fails. The protocol minimum is 0.2, so callers
        may request a stricter threshold but may not lower it.
    optimization_target:
        ``"acceptance"`` (default) equalizes adjacent-pair acceptance
        ratios. ``"flow"`` retains the former Trebst-Katzgraber objective.
    acceptance_plateau_tol:
        Absolute tolerance for improvement in acceptance standard deviation.
        Once a feasible acceptance-target run improves by less than this
        amount for ``patience`` consecutive iterations, it converges.
    rtr_plateau_tol:
        Relative round-trip-rate improvement threshold.
        When the relative improvement of the measured round-trip rate over the
        best-so-far stays below this value for ``patience`` consecutive
        iterations, the optimization is considered converged.
    patience:
        Number of consecutive low-improvement iterations required to declare a
        round-trip-rate plateau.
    optimize_path_shape:
        If True (default), jointly optimize the path shape (``t_split`` and
        the ``lambdaB_vdw`` curvature ``vdwb_gamma``) alongside ``s``. Only
        engages when ``use_path_from_initial_schedule=True``; a no-op otherwise.
    t_split_step:
        Greedy nudge size (in ``s`` space) applied to ``t_split`` per outer
        iteration during joint path-shape optimization.
    vdwb_gamma_step:
        Multiplicative (log-space) nudge size applied to ``vdwb_gamma`` per
        outer iteration during joint path-shape optimization.
    max_iter:
        Maximum number of outer-loop iterations.
    eta:
        Damping factor in ``(0, 1]``.
    min_gap:
        Forwarded to :func:`reschedule_s`.
    schedule_transform:
        Optional callable applied to every built schedule before measurement.
        Use this to enforce protocol-specific constraints that must remain
        fixed across all optimized iterations.

    Returns
    -------
    OptimizationResult
        The best feasible *measured* ``(N, 7)`` schedule and its ``s``. The
        acceptance target selects the most uniform acceptance profile, then
        the larger minimum and mean acceptance. Raises ``RuntimeError`` if no
        measured iteration satisfies the acceptance constraint.
    """
    initial_schedule = np.asarray(initial_schedule, dtype=np.float64)
    if initial_schedule.ndim != 2 or initial_schedule.shape[1] != N_COLUMNS:
        raise ValueError(
            "initial_schedule must have shape (N, 7); got "
            f"{initial_schedule.shape}"
        )
    n_replicas = initial_schedule.shape[0]
    if n_replicas < 3:
        raise ValueError("Need at least 3 replicas for a two-phase schedule")

    if not 0.0 < eta <= 1.0:
        raise ValueError(f"eta must be in (0, 1], got {eta}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    if flow_linearity_norm not in ("l2", "linf"):
        raise ValueError(
            f"flow_linearity_norm must be 'l2' or 'linf'; got "
            f"{flow_linearity_norm!r}"
        )
    if patience < 1:
        raise ValueError(f"patience must be at least 1, got {patience}")
    if not DEFAULT_MIN_ACCEPTANCE_RATIO <= min_acceptance_ratio < 1.0:
        raise ValueError(
            "min_acceptance_ratio must be in "
            f"[{DEFAULT_MIN_ACCEPTANCE_RATIO}, 1); got "
            f"{min_acceptance_ratio}"
        )
    if optimization_target not in ("acceptance", "flow"):
        raise ValueError(
            "optimization_target must be 'acceptance' or 'flow'; got "
            f"{optimization_target!r}"
        )
    if acceptance_plateau_tol < 0.0:
        raise ValueError(
            "acceptance_plateau_tol must be >= 0; got "
            f"{acceptance_plateau_tol}"
        )
    if t_split_step <= 0.0:
        raise ValueError(f"t_split_step must be > 0, got {t_split_step}")
    if vdwb_gamma_step <= 0.0:
        raise ValueError(f"vdwb_gamma_step must be > 0, got {vdwb_gamma_step}")

    if t_split is None:
        t_split = infer_t_split_from_schedule(initial_schedule)
    if charge_shape is None:
        charge_shape = SigmoidShape()
    if vdw_shape is None:
        vdw_shape = SigmoidShape()

    # In path-preserving mode, store the initial schedule for interpolation
    init_s_for_params = None
    init_schedule_for_params = None
    canonical_t_split = None
    if use_path_from_initial_schedule:
        if initial_s is None:
            # Default to evenly spaced s so iteration 0 uses the exact
            # initial_schedule values (interpolation at linspace points)
            init_s_for_params = np.linspace(0.0, 1.0, n_replicas, dtype=np.float64)
        else:
            init_s_for_params = np.asarray(initial_s, dtype=np.float64).copy()
        init_schedule_for_params = initial_schedule.copy()
        # Anchor the t_split s-warp at the initial path's phase boundary so
        # that t_split == canonical_t_split reproduces the un-warped path.
        canonical_t_split = infer_t_split_from_schedule(initial_schedule)

    params = LambdaScheduleParams(
        t_split=t_split,
        charge_shape=charge_shape,
        vdw_shape=vdw_shape,
        torsion=initial_schedule[:, COL_TORSION].copy(),
        restraint=initial_schedule[:, COL_RESTRAINT].copy(),
        initial_s=init_s_for_params,
        initial_schedule=init_schedule_for_params,
        vdwb_gamma=1.0,
        canonical_t_split=canonical_t_split,
    )

    # The existing path-shape nudges are driven by flow drops and round-trip
    # rate, so only retain them for the explicit flow objective. Acceptance
    # optimization changes replica positions only; this keeps a measured
    # acceptance improvement from being confounded with a simultaneous path
    # shape change.
    joint_active = (
        optimize_path_shape
        and use_path_from_initial_schedule
        and optimization_target == "flow"
    )

    if initial_s is None:
        s = np.linspace(0.0, 1.0, n_replicas, dtype=np.float64)
    else:
        s = np.asarray(initial_s, dtype=np.float64).copy()
        if s.ndim != 1 or len(s) != n_replicas:
            raise ValueError(
                f"initial_s must be 1-D of length {n_replicas}; got "
                f"shape {s.shape}"
            )
        if not np.all(np.diff(s) > 0):
            raise ValueError("initial_s must be strictly increasing")
        if s[0] != 0.0 or s[-1] != 1.0:
            raise ValueError("initial_s must start at 0 and end at 1")

    history: List[IterationDiagnostics] = []
    converged = False

    # Objective plateau tracking.
    best_rtr = -np.inf
    best_acceptance_std = np.inf
    plateau_count = 0
    rtr_eps = 1e-9
    # Greedy path-shape nudge bookkeeping. ``baseline_rtr`` is the measured
    # round-trip rate of the currently accepted path-shape configuration;
    # ``pending`` holds the knob nudged for the *next* iteration so it can be
    # accepted or reverted once measured.
    baseline_rtr = -np.inf
    pending = None
    accept_margin = 0.02
    imbalance_threshold = 0.15

    for k in range(max_iter):
        schedule = build_schedule(s, params)
        if schedule_transform is not None:
            schedule = np.asarray(schedule_transform(schedule), dtype=np.float64)
            if schedule.shape != (n_replicas, N_COLUMNS):
                raise ValueError(
                    "schedule_transform must return shape "
                    f"({n_replicas}, {N_COLUMNS}); got {schedule.shape}"
                )
        used_t_split = float(params.t_split)
        used_vdwb_gamma = float(params.vdwb_gamma)
        (
            flow_profile,
            flow_drops,
            flow_linearity_error,
            round_trip_rate,
            acceptance_ratios,
        ) = _compute_metrics_for_schedule(
            schedule,
            run_hremd_fn,
            flow_linearity_norm=flow_linearity_norm,
            flow_smooth=flow_smooth,
        )
        acceptance_ratios = np.asarray(
            acceptance_ratios, dtype=np.float64
        )
        if acceptance_ratios.shape != (n_replicas - 1,):
            raise ValueError(
                "acceptance ratios must have shape "
                f"({n_replicas - 1},); got {acceptance_ratios.shape}"
            )
        if not np.all(np.isfinite(acceptance_ratios)):
            raise ValueError(
                "acceptance ratios must contain only finite values"
            )
        if np.any(
            (acceptance_ratios < 0.0) | (acceptance_ratios > 1.0)
        ):
            raise ValueError("acceptance ratios must be in [0, 1]")
        measured_min_acceptance = float(acceptance_ratios.min())
        acceptance_mean = float(acceptance_ratios.mean())
        acceptance_std = float(acceptance_ratios.std())
        acceptance_range = float(np.ptp(acceptance_ratios))
        acceptance_feasible = bool(
            np.all(acceptance_ratios > min_acceptance_ratio)
        )
        diffusivity = diffusivity_profile(s, flow_drops)

        # Resolve a pending path-shape nudge from the previous iteration:
        # keep it only if it improved the measured round-trip rate, otherwise
        # revert the knob (graceful degradation).
        accepted_nudge = None
        if pending is not None:
            if (
                acceptance_feasible
                and round_trip_rate
                > pending["baseline_rtr"] * (1.0 + accept_margin)
            ):
                accepted_nudge = pending["knob"]
                baseline_rtr = round_trip_rate
            else:
                setattr(params, pending["knob"], pending["old_value"])
                accepted_nudge = "reverted"
            pending = None
        elif acceptance_feasible:
            baseline_rtr = max(baseline_rtr, round_trip_rate)

        # Phase-split flow (charge vs vdW) using the boundary in s-space.
        charge_mask = s[:-1] < used_t_split
        charge_phase_flow = float(flow_drops[charge_mask].sum())
        vdw_phase_flow = float(flow_drops[~charge_mask].sum())
        phase_flow_imbalance = charge_phase_flow - vdw_phase_flow

        f_min = float(flow_drops.min())
        f_max = float(flow_drops.max())
        f_mean = float(flow_drops.mean())
        f_std = float(flow_drops.std())
        f_range = f_max - f_min

        logger.info(
            "iter=%d  N=%d  flow_drop[max=%.4f mean=%.4f range=%.4f]  "
            "flow_lin_err[%s]=%.4f  round_trip_rate=%.4e  "
            "acceptance[min=%.4f mean=%.4f std=%.4f threshold=>%.4f "
            "feasible=%s]  "
            "t_split=%.4f  vdwb_gamma=%.4f  nudge=%s  "
            "diffusivity[min=%.3f max=%.3f]",
            k, n_replicas, f_max, f_mean, f_range,
            flow_linearity_norm, flow_linearity_error, round_trip_rate,
            measured_min_acceptance, acceptance_mean, acceptance_std,
            min_acceptance_ratio,
            acceptance_feasible,
            used_t_split, used_vdwb_gamma, accepted_nudge,
            float(diffusivity.min()), float(diffusivity.max()),
        )

        diagnostics = IterationDiagnostics(
            iteration=k,
            s=s.copy(),
            schedule=schedule.copy(),
            flow_profile=flow_profile.copy(),
            flow_drops=flow_drops.copy(),
            flow_drop_range=f_range,
            flow_drop_std=f_std,
            flow_drop_mean=f_mean,
            flow_linearity_error=flow_linearity_error,
            round_trip_rate=round_trip_rate,
            acceptance_ratios=acceptance_ratios.copy(),
            acceptance_mean=acceptance_mean,
            acceptance_std=acceptance_std,
            acceptance_range=acceptance_range,
            min_acceptance_ratio=measured_min_acceptance,
            acceptance_feasible=acceptance_feasible,
            diffusivity=diffusivity.copy(),
            t_split=used_t_split,
            vdwb_gamma=used_vdwb_gamma,
            charge_phase_flow=charge_phase_flow,
            vdw_phase_flow=vdw_phase_flow,
            phase_flow_imbalance=phase_flow_imbalance,
            accepted_nudge=accepted_nudge,
        )

        if acceptance_feasible:
            if optimization_target == "acceptance":
                std_improvement = best_acceptance_std - acceptance_std
                if std_improvement < acceptance_plateau_tol:
                    plateau_count += 1
                else:
                    plateau_count = 0
                best_acceptance_std = min(best_acceptance_std, acceptance_std)
            else:
                rel_impr = (
                    (round_trip_rate - best_rtr) / max(best_rtr, rtr_eps)
                )
                diagnostics.rtr_rel_improvement = float(rel_impr)
                if rel_impr < rtr_plateau_tol:
                    plateau_count += 1
                else:
                    plateau_count = 0
                if round_trip_rate > best_rtr:
                    best_rtr = round_trip_rate
        else:
            plateau_count = 0
        converged = acceptance_feasible and plateau_count >= patience

        if converged:
            history.append(diagnostics)
            if optimization_target == "acceptance":
                logger.info(
                    "Converged at iter=%d: acceptance-std plateau "
                    "(%d iters below acceptance_plateau_tol=%.4f), "
                    "std=%.4f, min_acceptance=%.4f > %.4f",
                    k, plateau_count, acceptance_plateau_tol,
                    acceptance_std, measured_min_acceptance,
                    min_acceptance_ratio,
                )
            else:
                logger.info(
                    "Converged at iter=%d: round-trip-rate plateau "
                    "(%d iters below rtr_plateau_tol=%.4f), rtr=%.4e, "
                    "min_acceptance=%.4f > %.4f",
                    k, plateau_count, rtr_plateau_tol, round_trip_rate,
                    measured_min_acceptance, min_acceptance_ratio,
                )
            break

        if optimization_target == "flow" and acceptance_feasible:
            placement_metrics = flow_drops
            diagnostics.placement_metric = "flow"
        else:
            placement_metrics = _acceptance_reschedule_metrics(
                acceptance_ratios
            )
            diagnostics.placement_metric = "acceptance"
        proposed_s = reschedule_s(
            s,
            placement_metrics,
            n_new=n_replicas,
            min_gap=min_gap,
            flow_drop_floor_frac=flow_drop_floor_frac,
        )
        diagnostics.proposed_s = proposed_s.copy()
        history.append(diagnostics)

        new_s = (1.0 - eta) * s + eta * proposed_s
        new_s = np.clip(new_s, 0.0, 1.0)
        new_s[0] = 0.0
        new_s[-1] = 1.0
        new_s = np.sort(new_s)

        for i in range(1, len(new_s) - 1):
            if new_s[i] - new_s[i - 1] < min_gap:
                new_s[i] = new_s[i - 1] + min_gap
        for i in range(len(new_s) - 2, 0, -1):
            if new_s[i + 1] - new_s[i] < min_gap:
                new_s[i] = new_s[i + 1] - min_gap

        if not np.all(np.diff(new_s) > 0):
            raise RuntimeError(
                "Damped s update produced a non-monotonic sequence at "
                f"iter={k}: {new_s}"
            )

        # Joint path-shape nudge: pick at most ONE knob to move for the next
        # iteration, driven by the measured flow. Classify those measurements
        # on the current grid; ``new_s`` is only the next iteration's grid.
        # Skip on the last iteration (a nudge there would never be measured).
        # The nudge is provisional and accepted/reverted at the top of the next
        # iteration.
        if joint_active and acceptance_feasible and k < max_iter - 1:
            pending = _propose_path_shape_nudge(
                params=params,
                s=s,
                measured_t_split=used_t_split,
                flow_drops=flow_drops,
                phase_flow_imbalance=phase_flow_imbalance,
                charge_phase_flow=charge_phase_flow,
                vdw_phase_flow=vdw_phase_flow,
                baseline_rtr=baseline_rtr,
                t_split_step=t_split_step,
                vdwb_gamma_step=vdwb_gamma_step,
                imbalance_threshold=imbalance_threshold,
                min_gap=min_gap,
            )

        s = new_s

    # Select the best feasible schedule among all *measured* iterations.
    # The acceptance objective minimizes pairwise acceptance spread, then
    # prefers larger worst-pair and mean acceptance. The legacy flow objective
    # maximizes round-trip rate, with flow-linearity error as a tie-break.
    #
    # Returning a *measured* iteration avoids handing back the final
    # reparameterized ``s``, whose schedule was proposed after the last HREMD
    # and therefore never measured.
    feasible_history = [
        diagnostics
        for diagnostics in history
        if diagnostics.acceptance_feasible
    ]
    if not feasible_history:
        best_infeasible = max(
            history,
            key=lambda diagnostics: float(
                diagnostics.min_acceptance_ratio
            ),
        )
        raise RuntimeError(
            "No measured lambda schedule satisfied the hard acceptance "
            f"constraint: every adjacent-pair ratio must be > "
            f"{min_acceptance_ratio:.4f}. Best measured minimum was "
            f"{best_infeasible.min_acceptance_ratio:.4f} at iteration "
            f"{best_infeasible.iteration}. Increase max_iter or improve the "
            "initial schedule before using an optimized result."
        )

    if optimization_target == "acceptance":
        best = max(
            feasible_history,
            key=lambda d: _acceptance_profile_score(d.acceptance_ratios),
        )
        logger.info(
            "selected best iter=%d: acceptance[std=%.4f mean=%.4f "
            "min=%.4f > %.4f]  flow_linearity_error[%s]=%.4f  "
            "round_trip_rate=%.4e",
            best.iteration,
            float(best.acceptance_std),
            float(best.acceptance_mean),
            float(best.min_acceptance_ratio),
            min_acceptance_ratio,
            flow_linearity_norm,
            float(best.flow_linearity_error),
            float(best.round_trip_rate),
        )
    else:
        best = max(
            feasible_history,
            key=lambda d: (
                float(d.round_trip_rate),
                -float(d.flow_linearity_error),
            ),
        )
        logger.info(
            "selected best iter=%d: round_trip_rate=%.4e  "
            "flow_linearity_error[%s]=%.4f  min_acceptance=%.4f > %.4f  "
            "t_split=%.4f  vdwb_gamma=%.4f",
            best.iteration,
            float(best.round_trip_rate),
            flow_linearity_norm,
            float(best.flow_linearity_error),
            float(best.min_acceptance_ratio),
            min_acceptance_ratio,
            float(best.t_split),
            float(best.vdwb_gamma),
        )

    # Reflect the best *measured* path-shape configuration in the returned
    # params (rather than the last provisional nudge).
    if best.t_split is not None:
        params.t_split = float(best.t_split)
    if best.vdwb_gamma is not None:
        params.vdwb_gamma = float(best.vdwb_gamma)
    return OptimizationResult(
        schedule=best.schedule.copy(),
        s=best.s.copy(),
        params=params,
        converged=converged,
        n_iterations=len(history),
        selected_iteration=best.iteration,
        acceptance_threshold=float(min_acceptance_ratio),
        history=history,
    )
