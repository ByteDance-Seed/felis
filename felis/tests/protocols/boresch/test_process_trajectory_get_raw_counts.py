# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np


class _DummyFrame:

    def __init__(self, positions: np.ndarray, dimensions: np.ndarray):
        self.positions = positions
        self.dimensions = dimensions


class _DummyUniverse:

    def __init__(self, frames: list[_DummyFrame]):
        self.trajectory = frames


def test_process_trajectory_and_get_raw_counts_converts_units(monkeypatch):
    import felis.protocols.boresch.process_trajectory_get_raw_counts as mod

    # 3 frames; contents are irrelevant because we stub out distance/angle functions.
    positions = np.zeros((10, 3), dtype=float)
    dims = np.array([10, 10, 10, 90, 90, 90], dtype=float)
    u = _DummyUniverse([_DummyFrame(positions, dims) for _ in range(3)])

    monkeypatch.setattr(mod, "get_mda_universe", lambda _gk, _trj: u)
    # Bonds in Angstrom -> nm via 0.1
    monkeypatch.setattr(mod, "calc_bonds", lambda _a, _b, _dims: 10.0)
    # Angles/dihedrals in radians -> degrees via 180/pi
    monkeypatch.setattr(mod, "calc_angles", lambda _a, _b, _c: np.pi / 2)
    monkeypatch.setattr(mod, "calc_dihedrals", lambda _a, _b, _c, _d: -np.pi)

    out = mod.process_trajectory_and_get_raw_counts("top", "traj", (0, 1, 2), (3, 4, 5))
    assert set(out.keys()) == {"r", "theta", "phi", "alpha", "beta", "gamma"}
    assert np.allclose(out["r"], np.array([1.0, 1.0, 1.0]))
    assert np.allclose(out["theta"], np.array([90.0, 90.0, 90.0]))
    assert np.allclose(out["alpha"], np.array([90.0, 90.0, 90.0]))
    assert np.allclose(out["phi"], np.array([-180.0, -180.0, -180.0]))
    assert np.allclose(out["beta"], np.array([-180.0, -180.0, -180.0]))
    assert np.allclose(out["gamma"], np.array([-180.0, -180.0, -180.0]))
