# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from math import cos
from math import pi
from math import sin

from openmm.app import DCDFile

from felis.configs import GlobalKeys
from felis.utils.omm.omm_system import get_system_topology
from felis.utils.omm.omm_tools import extract_state_from_netdcd

logger = logging.getLogger(__name__)


def trj_nc2dcd(nc: str, state_index: list[int], discard: int, sys_pdb: str, sys_top: str, trj_dcd: str,
               nstep_per_snapshot: int) -> None:
    gk = GlobalKeys()
    gk.filename.crd = sys_pdb
    gk.filename.sys = sys_top
    gk.integrator.nstep_per_snapshot = nstep_per_snapshot
    if gk.openmm.checkpoint_interval != 1:
        gk.openmm.checkpoint_interval = 1
        logger.info("openmm.checkpoint_interval set to 1 by trj_nc2dcd")

    _osys, otop = get_system_topology(gk)
    dcd_handle = open(trj_dcd, "wb")
    dcdfile = DCDFile(dcd_handle, otop, gk.integrator.dt_ps * gk.integrator.nstep_per_snapshot)

    for idx in state_index:
        coords, pbcs = extract_state_from_netdcd(nc, idx, gk.openmm.checkpoint_interval)
        assert len(coords) > discard, f"len(coords)={len(coords)}, discard={discard}"
        count = 0
        for pos, pbc in zip(coords, pbcs):
            if count >= discard:
                dcdfile.writeModel(positions=pos, periodicBoxVectors=pbc)
            count += 1
    logger.info(f"DCD file written to {trj_dcd}")


def convert_lattice6_to_vec3x3(a: float,
                               b: float,
                               c: float,
                               al: float,
                               be: float,
                               ga: float,
                               deg: bool = True) -> list[list[float]]:
    eps = 1e-6
    deg_to_radian = pi / 180.0
    pi_2 = pi * 0.5

    if deg:
        al = al * deg_to_radian
        be = be * deg_to_radian
        ga = ga * deg_to_radian

    if abs(al - pi_2) < eps:
        cos_alpha = 0.0
    else:
        cos_alpha = cos(al)
    if abs(be - pi_2) < eps:
        cos_beta = 0.0
    else:
        cos_beta = cos(be)
    if abs(ga - pi_2) < eps:
        cos_gamma = 0.0
        sin_gamma = 1.0
    else:
        cos_gamma = cos(ga)
        sin_gamma = sin(ga)

    vec1 = [a, 0.0, 0.0]
    bx = b * cos_gamma
    by = b * sin_gamma
    vec2 = [bx, by, 0.0]
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = c / sin_gamma * (2 * cos_alpha * cos_beta * cos_gamma + sin_gamma**2 - cos_alpha**2 - cos_beta**2)**0.5
    vec3 = [cx, cy, cz]
    return [vec1, vec2, vec3]


def convert_lattice6_to_volume(a, b, c, alpha, beta, gamma):
    deg_to_radian = pi / 180.0
    cos_al = cos(deg_to_radian * alpha)
    cos_be = cos(deg_to_radian * beta)
    cos_ga = cos(deg_to_radian * gamma)

    fac2 = abs(1. - cos_al**2 - cos_be**2 - cos_ga**2 + 2. * cos_al * cos_be * cos_ga)
    vol = a * b * c * fac2**0.5
    return vol
