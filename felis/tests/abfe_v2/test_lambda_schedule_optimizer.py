# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Smoke tests for the ABFE v2 lambda-schedule optimizer refactor.

These tests exercise the pure path-shape math, backward compatibility of
``build_schedule`` in path-preserving mode, and the flow-target control flow
(round-trip-rate plateau gating and the greedy path-shape nudge accept/revert)
without running any real HREMD. The expensive ``run_hremd_fn`` is replaced by
a sentinel and the metric extraction is monkeypatched to deterministic values.
"""

import numpy as np
import pytest

from felis.protocols.abfe_v2 import lambda_schedule_optimizer as lso
from felis.protocols.abfe_v2.lambda_schedule_optimizer import (
    LambdaScheduleParams,
    SigmoidShape,
    _acceptance_reschedule_metrics,
    _apply_vdwb_gamma,
    _warp_s_for_t_split,
    build_schedule,
    optimize_lambda_schedule,
    reschedule_s,
)
from felis.protocols.abfe_v2.protein_lambda_rules import (
    freeze_stage_lambda_counts,
    infer_stage_boundaries,
)
from felis.protocols.abfe_v2.utilities import get_round_trip_rate_from_records


def _make_initial_schedule(n=12):
    """Build a simple 3-phase (N, 7) schedule with a charge->vdW split."""
    s = np.linspace(0.0, 1.0, n)
    a_charge = np.clip(1.0 - 2.0 * s, 0.0, 1.0)          # hits 0 at s=0.5
    a_vdw = np.clip(1.0 - 2.0 * (s - 0.5), 0.0, 1.0)     # decays after 0.5
    b_charge = 1.0 - a_charge
    b_vdw = 1.0 - a_vdw
    c_charge = np.zeros(n)
    torsion = np.ones(n)
    restraint = np.ones(n)
    return np.column_stack(
        [a_charge, b_charge, c_charge, a_vdw, b_vdw, torsion, restraint]
    )


# ---------------------------------------------------------------------------
# Pure path-shape math
# ---------------------------------------------------------------------------


def test_warp_s_endpoints_and_boundary():
    s = np.linspace(0.0, 1.0, 21)
    c0, t_split = 0.5, 0.7
    w = _warp_s_for_t_split(s, t_split, c0)
    assert w[0] == pytest.approx(0.0)
    assert w[-1] == pytest.approx(1.0)
    # W(t_split) == c0 (t_split coincides with a grid point here).
    idx = int(np.argmin(np.abs(s - t_split)))
    assert w[idx] == pytest.approx(c0, abs=1e-9)
    assert np.all(np.diff(w) > 0)  # strictly increasing


def test_warp_s_identity_when_tsplit_equals_c0():
    s = np.linspace(0.0, 1.0, 15)
    w = _warp_s_for_t_split(s, 0.4, 0.4)
    np.testing.assert_allclose(w, s, atol=1e-12)


def test_apply_vdwb_gamma_endpoints_and_monotone():
    vdwb = np.linspace(1.0, 0.0, 11)
    out = _apply_vdwb_gamma(vdwb, 2.5)
    assert out[0] == pytest.approx(vdwb[0])
    assert out[-1] == pytest.approx(vdwb[-1])
    assert np.all(np.diff(out) <= 1e-12)  # still decreasing
    # gamma>1 pushes the normalized curve below the linear ramp.
    assert np.all(out[1:-1] <= vdwb[1:-1] + 1e-12)


def test_apply_vdwb_gamma_identity_at_one():
    vdwb = np.linspace(1.0, 0.2, 9)
    np.testing.assert_allclose(_apply_vdwb_gamma(vdwb, 1.0), vdwb, atol=1e-12)


# ---------------------------------------------------------------------------
# build_schedule backward compatibility
# ---------------------------------------------------------------------------


def test_build_schedule_path_mode_identity_backward_compat():
    """t_split=c0 and gamma=1 reproduce the plain interpolation-at-s path."""
    init = _make_initial_schedule(12)
    n = init.shape[0]
    s_grid = np.linspace(0.0, 1.0, n)
    c0 = lso.infer_t_split_from_schedule(init)

    params = LambdaScheduleParams(
        t_split=c0,
        charge_shape=SigmoidShape(),
        vdw_shape=SigmoidShape(),
        torsion=init[:, lso.COL_TORSION].copy(),
        restraint=init[:, lso.COL_RESTRAINT].copy(),
        initial_s=s_grid.copy(),
        initial_schedule=init.copy(),
        vdwb_gamma=1.0,
        canonical_t_split=c0,
    )
    sched = build_schedule(s_grid, params)
    # With s == initial_s, identity warp, gamma=1 -> exact initial schedule
    # (B_charge re-derived, C_charge zeroed, which init already satisfies).
    np.testing.assert_allclose(sched, init, atol=1e-9)


def test_schedule_transform_freezes_stage_counts_during_optimization(monkeypatch):
    """A protocol transform can keep all stage boundaries fixed as s changes."""
    init = _make_initial_schedule(10)
    init[:, lso.COL_RESTRAINT] = [
        0.0, 0.2, 0.5, 0.8, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0
    ]
    n = init.shape[0]
    measured_boundaries = []
    reference_boundaries = infer_stage_boundaries(init)

    def fake_metrics(schedule,
                     run_hremd_fn,
                     flow_linearity_norm="l2",
                     flow_smooth=True):
        run_hremd_fn(schedule)
        measured_boundaries.append(infer_stage_boundaries(schedule))
        flow_profile = 1.0 - np.arange(n) / (n - 1)
        flow_drops = np.full(n - 1, 1.0 / (n - 1))
        acceptance_ratios = np.full(n - 1, 0.5)
        return flow_profile, flow_drops, 0.3, 1.0, acceptance_ratios

    monkeypatch.setattr(lso, "_compute_metrics_for_schedule", fake_metrics)
    result = optimize_lambda_schedule(
        initial_schedule=init,
        run_hremd_fn=lambda sched: None,
        use_path_from_initial_schedule=True,
        max_iter=3,
        eta=1.0,
        schedule_transform=lambda schedule: freeze_stage_lambda_counts(
            schedule, init
        ),
    )

    assert measured_boundaries == [reference_boundaries] * 3
    assert infer_stage_boundaries(result.schedule) == reference_boundaries


# ---------------------------------------------------------------------------
# Flow-target control flow with a fake HREMD
# ---------------------------------------------------------------------------


class _FakeMetrics:
    """Deterministic metric provider keyed by call index.

    ``rtr_sequence`` gives the round-trip rate returned on each successive
    call to ``_compute_metrics_for_schedule``. All other metrics are held
    constant/benign so only the gating and nudge logic is under test.
    """

    def __init__(self,
                 rtr_sequence,
                 n,
                 flow_drops=None,
                 acceptance_sequence=None):
        self.rtr_sequence = rtr_sequence
        self.n = n
        self.flow_drops = flow_drops
        self.acceptance_sequence = acceptance_sequence
        self.calls = 0

    def __call__(self,
                 schedule,
                 run_hremd_fn,
                 flow_linearity_norm="l2",
                 flow_smooth=True):
        run_hremd_fn(schedule)  # exercise the sentinel
        i = min(self.calls, len(self.rtr_sequence) - 1)
        rtr = float(self.rtr_sequence[i])
        self.calls += 1
        n = self.n
        flow_profile = 1.0 - np.arange(n) / (n - 1)
        if self.flow_drops is not None:
            flow_drops = np.asarray(self.flow_drops, dtype=np.float64)
        else:
            flow_drops = np.full(n - 1, 1.0 / (n - 1))
        if self.acceptance_sequence is None:
            acceptance_ratios = np.full(n - 1, 0.5)
        else:
            acceptance_i = min(i, len(self.acceptance_sequence) - 1)
            acceptance_ratios = np.asarray(
                self.acceptance_sequence[acceptance_i], dtype=np.float64
            )
        flow_linearity_error = 0.3
        return (
            flow_profile,
            flow_drops,
            flow_linearity_error,
            rtr,
            acceptance_ratios,
        )


def test_flow_plateau_convergence(monkeypatch):
    """rtr improves then flattens -> converge after `patience` flat iters."""
    init = _make_initial_schedule(10)
    n = init.shape[0]
    # rtr: 1.0, 2.0 (big jump), then flat 2.0, 2.0 -> two plateau iters.
    fake = _FakeMetrics([1.0, 2.0, 2.0, 2.0, 2.0], n)
    monkeypatch.setattr(lso, "_compute_metrics_for_schedule", fake)

    result = optimize_lambda_schedule(
        initial_schedule=init,
        run_hremd_fn=lambda sched: None,
        use_path_from_initial_schedule=True,
        optimize_path_shape=False,  # isolate gating
        optimization_target="flow",
        rtr_plateau_tol=0.05,
        patience=2,
        max_iter=8,
    )
    assert result.converged
    # iter0 rtr=1, iter1 rtr=2 (improve, reset), iter2 flat (count1),
    # iter3 flat (count2 -> converge). So 4 measured iterations.
    assert result.n_iterations == 4


def test_flow_zero_rtr_graceful(monkeypatch):
    """All-zero rtr short run converges gracefully via the plateau gate."""
    init = _make_initial_schedule(8)
    n = init.shape[0]
    fake = _FakeMetrics([0.0, 0.0, 0.0, 0.0], n)
    monkeypatch.setattr(lso, "_compute_metrics_for_schedule", fake)

    result = optimize_lambda_schedule(
        initial_schedule=init,
        run_hremd_fn=lambda sched: None,
        use_path_from_initial_schedule=True,
        optimize_path_shape=False,
        optimization_target="flow",
        patience=2,
        max_iter=6,
    )
    assert result.converged
    # iter0 registers +inf improvement from the -inf baseline (plateau=0),
    # then iter1 and iter2 are flat (plateau reaches patience=2) -> 3 iters.
    assert result.n_iterations == 3


def test_flow_nudge_revert_when_not_improving(monkeypatch):
    """A t_split nudge that does not improve rtr is reverted."""
    init = _make_initial_schedule(10)
    n = init.shape[0]
    # Concentrate the flow drops in the charge phase (first pairs) so the
    # normalized phase imbalance exceeds the threshold and a t_split nudge is
    # proposed. rtr never improves, so every nudge must be reverted.
    heavy = np.array([0.9 / 5] * 5 + [0.1 / 4] * 4, dtype=np.float64)
    fake = _FakeMetrics([1.0, 1.0, 1.0, 1.0], n, flow_drops=heavy)
    monkeypatch.setattr(lso, "_compute_metrics_for_schedule", fake)

    result = optimize_lambda_schedule(
        initial_schedule=init,
        run_hremd_fn=lambda sched: None,
        use_path_from_initial_schedule=True,
        optimize_path_shape=True,
        optimization_target="flow",
        patience=10,       # don't converge on plateau; run full max_iter
        max_iter=4,
    )
    reverts = [d.accepted_nudge for d in result.history]
    # A nudge was proposed and, since rtr never improved, it was reverted.
    assert "reverted" in reverts
    # t_split must return to its canonical value (all nudges reverted) and the
    # returned params reflect a measured iteration.
    assert result.params.t_split == pytest.approx(
        result.params.canonical_t_split
    )
    assert result.params.vdwb_gamma == pytest.approx(1.0)


def _two_replica_swap_record():
    return {
        "old_replica_ids": [0, 1],
        "worker_ids": [0, 1],
        "new_replica_ids": [1, 0],
    }


def test_round_trip_does_not_count_initial_top_to_bottom_traversal():
    records = [_two_replica_swap_record()]

    assert get_round_trip_rate_from_records(records) == 0.0


def test_round_trip_counts_only_bottom_top_bottom_cycle():
    records = [_two_replica_swap_record(), _two_replica_swap_record()]

    assert get_round_trip_rate_from_records(records) == pytest.approx(1.0 / 3.0)


def test_path_shape_nudge_uses_grid_that_produced_flow(monkeypatch):
    init = _make_initial_schedule(7)
    measured_s = np.linspace(0.0, 1.0, 7)
    next_s = np.array([0.0, 0.16515, 0.33030, 0.49491, 0.67809, 0.81821, 1.0])
    flow_drops = np.array([1 / 6, 1 / 6, 1 / 6, 0.1, 0.3, 0.1])
    acceptance = np.full(6, 0.5)
    captured_s = []

    def fake_metrics(*_args, **_kwargs):
        return (
            np.linspace(1.0, 0.0, 7),
            flow_drops,
            0.0,
            1.0,
            acceptance,
        )

    def capture_nudge(*, s, **_kwargs):
        captured_s.append(s.copy())
        return None

    monkeypatch.setattr(lso, "_compute_metrics_for_schedule", fake_metrics)
    monkeypatch.setattr(lso, "reschedule_s", lambda *_args, **_kwargs: next_s)
    monkeypatch.setattr(lso, "_propose_path_shape_nudge", capture_nudge)

    optimize_lambda_schedule(
        initial_schedule=init,
        initial_s=measured_s,
        run_hremd_fn=lambda _schedule: None,
        use_path_from_initial_schedule=True,
        optimize_path_shape=True,
        optimization_target="flow",
        eta=1.0,
        patience=10,
        max_iter=2,
    )

    assert len(captured_s) == 1
    np.testing.assert_allclose(captured_s[0], measured_s)
    assert not np.allclose(captured_s[0], next_s)


def test_path_shape_nudge_keeps_measurement_boundary_after_pending_revert(
    monkeypatch,
):
    init = _make_initial_schedule(7)
    n = len(init)
    flow_drops = np.full(n - 1, 1.0 / (n - 1))
    acceptance = np.full(n - 1, 0.5)
    captured_boundaries = []

    def fake_metrics(*_args, **_kwargs):
        return (
            np.linspace(1.0, 0.0, n),
            flow_drops,
            0.0,
            1.0,
            acceptance,
        )

    def fake_nudge(*, params, measured_t_split, **_kwargs):
        captured_boundaries.append((measured_t_split, params.t_split))
        if len(captured_boundaries) == 1:
            old_value = params.t_split
            params.t_split = old_value + 0.1
            return {
                "knob": "t_split",
                "old_value": old_value,
                "baseline_rtr": 1.0,
            }
        return None

    monkeypatch.setattr(lso, "_compute_metrics_for_schedule", fake_metrics)
    monkeypatch.setattr(lso, "_propose_path_shape_nudge", fake_nudge)

    optimize_lambda_schedule(
        initial_schedule=init,
        run_hremd_fn=lambda _schedule: None,
        use_path_from_initial_schedule=True,
        optimize_path_shape=True,
        optimization_target="flow",
        patience=10,
        max_iter=3,
    )

    initial_t_split = lso.infer_t_split_from_schedule(init)
    assert captured_boundaries[0] == pytest.approx(
        (initial_t_split, initial_t_split)
    )
    assert captured_boundaries[1] == pytest.approx(
        (initial_t_split + 0.1, initial_t_split)
    )


# ---------------------------------------------------------------------------
# Hard acceptance constraint
# ---------------------------------------------------------------------------


def test_acceptance_redistribution_targets_both_edge_pairs():
    """Poor first/last acceptance pulls the neighboring windows outward."""
    s = np.linspace(0.0, 1.0, 8)

    poor_first = np.full(len(s) - 1, 0.8)
    poor_first[0] = 0.05
    first_proposal = reschedule_s(
        s,
        _acceptance_reschedule_metrics(poor_first),
        flow_drop_floor_frac=0.25,
    )

    poor_last = np.full(len(s) - 1, 0.8)
    poor_last[-1] = 0.05
    last_proposal = reschedule_s(
        s,
        _acceptance_reschedule_metrics(poor_last),
        flow_drop_floor_frac=0.25,
    )

    assert first_proposal[1] < s[1]
    assert last_proposal[-2] > s[-2]


def test_acceptance_target_selects_flattest_feasible_profile(monkeypatch):
    """Acceptance target outranks round-trip rate after the floor passes."""
    init = _make_initial_schedule(8)
    n = init.shape[0]
    uneven = np.array([0.30, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90])
    flat = np.array([0.55, 0.60, 0.65, 0.60, 0.55, 0.60, 0.65])
    fake = _FakeMetrics(
        [100.0, 1.0],
        n,
        acceptance_sequence=[uneven, flat],
    )
    monkeypatch.setattr(lso, "_compute_metrics_for_schedule", fake)

    result = optimize_lambda_schedule(
        initial_schedule=init,
        run_hremd_fn=lambda sched: None,
        use_path_from_initial_schedule=True,
        optimize_path_shape=False,
        max_iter=2,
        patience=10,
    )

    assert result.history[0].placement_metric == "acceptance"
    assert result.history[1].placement_metric == "acceptance"
    assert result.history[1].acceptance_std < result.history[0].acceptance_std
    assert result.selected_iteration == 1


def test_acceptance_equal_to_threshold_is_infeasible_then_recovers(
    monkeypatch,
):
    """Equality fails the strict floor; a later >0.2 schedule can be selected."""
    init = _make_initial_schedule(8)
    n = init.shape[0]
    at_threshold = np.full(n - 1, 0.5)
    at_threshold[0] = 0.2
    above_threshold = np.full(n - 1, 0.21)
    fake = _FakeMetrics(
        [10.0, 1.0],
        n,
        acceptance_sequence=[at_threshold, above_threshold],
    )
    monkeypatch.setattr(lso, "_compute_metrics_for_schedule", fake)

    result = optimize_lambda_schedule(
        initial_schedule=init,
        run_hremd_fn=lambda sched: None,
        use_path_from_initial_schedule=True,
        optimize_path_shape=False,
        patience=10,
        max_iter=2,
    )

    assert not result.history[0].acceptance_feasible
    assert result.history[0].placement_metric == "acceptance"
    assert result.history[1].acceptance_feasible
    assert result.selected_iteration == 1
    assert result.history[1].min_acceptance_ratio > 0.2


def test_infeasible_schedule_cannot_win_on_round_trip_rate(monkeypatch):
    """A high-flow schedule below the floor cannot outrank a feasible one."""
    init = _make_initial_schedule(8)
    n = init.shape[0]
    feasible = np.full(n - 1, 0.3)
    infeasible = np.full(n - 1, 0.3)
    infeasible[-1] = 0.1
    fake = _FakeMetrics(
        [1.0, 100.0],
        n,
        acceptance_sequence=[feasible, infeasible],
    )
    monkeypatch.setattr(lso, "_compute_metrics_for_schedule", fake)

    result = optimize_lambda_schedule(
        initial_schedule=init,
        run_hremd_fn=lambda sched: None,
        use_path_from_initial_schedule=True,
        optimize_path_shape=False,
        patience=10,
        max_iter=2,
    )

    assert result.selected_iteration == 0
    assert result.history[0].acceptance_feasible
    assert not result.history[1].acceptance_feasible


def test_optimizer_fails_when_no_schedule_satisfies_acceptance(monkeypatch):
    """Do not return or write a flow-optimal schedule that violates the floor."""
    init = _make_initial_schedule(8)
    n = init.shape[0]
    below_threshold = np.full(n - 1, 0.2)
    fake = _FakeMetrics(
        [1.0, 2.0],
        n,
        acceptance_sequence=[below_threshold],
    )
    monkeypatch.setattr(lso, "_compute_metrics_for_schedule", fake)

    with pytest.raises(RuntimeError, match="every adjacent-pair ratio"):
        optimize_lambda_schedule(
            initial_schedule=init,
            run_hremd_fn=lambda sched: None,
            use_path_from_initial_schedule=True,
            optimize_path_shape=False,
            patience=10,
            max_iter=2,
        )


def test_acceptance_threshold_cannot_be_weakened():
    """The protocol floor remains at 0.2 even when called programmatically."""
    init = _make_initial_schedule(8)

    with pytest.raises(ValueError, match=r"\[0.2, 1\)"):
        optimize_lambda_schedule(
            initial_schedule=init,
            run_hremd_fn=lambda sched: None,
            min_acceptance_ratio=0.19,
            max_iter=1,
        )

