# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import MDAnalysis
import numpy as np
from MDAnalysis.lib.distances import calc_angles


def filter_frames_based_on_angle(u: MDAnalysis.Universe, frames: list[int], angle_atoms: tuple[int, int, int],
                                 valid_range: tuple[float, float]) -> tuple[list[int], float]:
    radian_to_degree = 180. / np.pi
    ia0, ia1, ia2 = angle_atoms
    mask_list: list[int] = []
    angle_absdiff_list: list[float] = []
    for ifr in frames:
        t = u.trajectory[int(ifr)]
        pos = t.positions
        pa0, pa1, pa2 = pos[ia0], pos[ia1], pos[ia2]
        ans = calc_angles(pa0, pa1, pa2) * radian_to_degree
        angle_absdiff_list.append(abs(ans - 90.0))
        imask = 1 if (valid_range[0] <= ans) and (ans <= valid_range[1]) else 0
        mask_list.append(imask)
    result_frames = np.array(frames)[np.array(mask_list) == 1]
    return result_frames.tolist(), float(np.mean(angle_absdiff_list))
