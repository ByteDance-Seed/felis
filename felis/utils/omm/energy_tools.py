# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

import openmm.unit as unit
from openmm import Context, Force, State
from openmm.app import Simulation

from felis.utils.omm.omm_system import get_force_from_system_by_name

logger = logging.getLogger(__name__)


def get_energies_in_kcal_mol(forces: list[Force], octx: Context, pos, pbc=None, get_forces=False) -> dict[str, float]:
    octx.setPositions(pos)
    if pbc is not None:
        octx.setPeriodicBoxVectors(*pbc)
    d = dict()
    for f in forces:
        name = f.getName()
        state: State = octx.getState(getEnergy=True, getForces=get_forces, groups=1 << f.getForceGroup())
        d[name] = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        if get_forces:
            fvalue = state.getForces(asNumpy=True).value_in_unit(unit.kilocalorie_per_mole / unit.nanometer)
            d[name + "_f"] = fvalue
    return d


def get_energies_in_kcal_mol_by_names(names: list[str],
                                      osim: Simulation,
                                      pos,
                                      pbc=None,
                                      get_forces=False) -> dict[str, float]:
    osys = osim.system
    octx = osim.context
    forces = [get_force_from_system_by_name(osys, name) for name in names]
    return get_energies_in_kcal_mol(forces, octx, pos, pbc, get_forces=get_forces)


def get_potential_energy_in_kcal_mol(osim: Simulation, pos, pbc=None) -> float:
    octx = osim.context
    octx.setPositions(pos)
    if pbc is not None:
        octx.setPeriodicBoxVectors(*pbc)
    state: State = octx.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
