# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from MDAnalysis.lib.distances import calc_angles
from MDAnalysis.lib.distances import calc_bonds
from MDAnalysis.lib.distances import calc_dihedrals
import numpy as np

from felis.configs import GlobalKeys
from felis.external.mda_tools import get_mda_universe


def process_trajectory_and_get_raw_counts(sys_top: str, trj_dcd: str, p123: tuple[int, int, int],
                                          l123: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """Process trajectory and collect raw Boresch geometry series.

    This function keeps the original implementation semantics from
    `main_boresch_restraints.generate_boresch_restraints`.

    Args:
        sys_top: Topology file path.
        trj_dcd: Trajectory file path.
        p123: Protein anchor atom indices (p1, p2, p3).
        l123: Ligand anchor atom indices (l1, l2, l3).

    Returns:
        Mapping with keys: r, theta, phi, alpha, beta, gamma (all as numpy arrays).
    """
    p1, p2, p3 = p123
    l1, l2, l3 = l123

    radian_to_degree = 180. / np.pi
    angstrom_to_nm = 0.1

    r: list[float] = []
    theta: list[float] = []
    phi: list[float] = []
    alpha: list[float] = []
    beta: list[float] = []
    gamma: list[float] = []

    gk = GlobalKeys()
    gk.filename.sys = sys_top
    u = get_mda_universe(gk, trj_dcd)

    for t in u.trajectory:
        pos = t.positions
        dims = t.dimensions
        ap1 = pos[p1]
        ap2 = pos[p2]
        ap3 = pos[p3]
        al1 = pos[l1]
        al2 = pos[l2]
        al3 = pos[l3]

        r.append(calc_bonds(al1, ap1, dims) * angstrom_to_nm)
        theta.append(calc_angles(al1, ap1, ap2) * radian_to_degree)
        phi.append(calc_dihedrals(al1, ap1, ap2, ap3) * radian_to_degree)

        alpha.append(calc_angles(ap1, al1, al2) * radian_to_degree)
        beta.append(calc_dihedrals(ap2, ap1, al1, al2) * radian_to_degree)
        gamma.append(calc_dihedrals(ap1, al1, al2, al3) * radian_to_degree)

    return {
        "r": np.array(r),
        "theta": np.array(theta),
        "phi": np.array(phi),
        "alpha": np.array(alpha),
        "beta": np.array(beta),
        "gamma": np.array(gamma),
    }
