# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np


class _Frame:

    def __init__(self, positions: np.ndarray):
        self.positions = positions
        self.dimensions = np.array([10, 10, 10, 90, 90, 90], dtype=float)


class _Universe:

    def __init__(self, frames: list[_Frame]):
        self.trajectory = frames


def test_filter_frames_based_on_angle_filters_and_mean(monkeypatch):
    import felis.protocols.boresch.anchor_geometry_filters as mod

    # Encode desired degrees in positions[ia0][0], then fake calc_angles reads it.
    def fake_calc_angles(pa0, _pa1, _pa2):
        return np.deg2rad(pa0[0])

    monkeypatch.setattr(mod, "calc_angles", fake_calc_angles)

    ia0, ia1, ia2 = 1, 2, 3
    frames = []
    for deg in [30.0, 90.0, 150.0]:
        pos = np.zeros((10, 3), dtype=float)
        pos[ia0] = np.array([deg, 0.0, 0.0])
        frames.append(_Frame(pos))
    u = _Universe(frames)

    kept, mean_absdiff = mod.filter_frames_based_on_angle(u, [0, 1, 2], (ia0, ia1, ia2), (45.0, 135.0))
    assert kept == [1]
    assert mean_absdiff == np.mean([60.0, 0.0, 60.0])
