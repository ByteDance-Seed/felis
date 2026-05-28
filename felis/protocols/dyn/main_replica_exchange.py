# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

from felis.configs import GlobalKeys
from felis.utils.omm.omm_system import get_initial_pos_pbc
from felis.utils.omm.omm_tools import get_list_of_thermodynamic_states
from felis.utils.omm.omm_tools import get_replica_exchange_sampler

logger = logging.getLogger(__name__)


def mainfunc(gk_list: list[GlobalKeys]) -> None:
    for i, gk in enumerate(gk_list):
        logger.info(f"Thermodynamic State {i}{gk}")

    gk = gk_list[0]
    pos, pbc = get_initial_pos_pbc(gk)
    thermo_states = get_list_of_thermodynamic_states(gk_list)
    simu = get_replica_exchange_sampler(thermo_states, pos, pbc, gk)

    simu.run()
