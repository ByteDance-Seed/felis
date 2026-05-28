# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

from MDAnalysis import Universe
import numpy as np

from felis.configs import GlobalKeys
from felis.utils.topfile_tools import get_top_system_symbols

logger = logging.getLogger(__name__)


def get_mda_universe(gk: GlobalKeys, trj: str) -> Universe:
    if gk.openmm.params_ecosystem == "gromacs":
        gmxtop = gk.filename.sys
        u = Universe(gmxtop, trj, topology_format="ITP")
        sys_symbols = get_top_system_symbols(gmxtop)
        np_symbols = np.array([elem.upper() for elem in sys_symbols], dtype=object)  # must be upper case!
        u.add_TopologyAttr("elements", np_symbols)
        return u
    else:
        raise NotImplementedError(gk.openmm.params_ecosystem)


def get_trj_nsnapshots(gk: GlobalKeys, trj: str) -> int:
    u = get_mda_universe(gk, trj)
    return len(u.trajectory)
