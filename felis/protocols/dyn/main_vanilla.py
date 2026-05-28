# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from pathlib import Path

from openmm.app import CheckpointReporter
from openmm.app import DCDReporter
from openmm.app import StateDataReporter
import openmm.unit as unit

from felis.configs import GlobalKeys
from felis.configs import IntegratorNameOption
from felis.configs import MinimizeRelaxOption
from felis.external.mda_tools import get_trj_nsnapshots
from felis.utils.omm.omm_reporter import MinimizerReporter
from felis.utils.omm.omm_system import get_simulation
from felis.utils.omm.omm_system import write_pdb

logger = logging.getLogger(__name__)


def mainfunc(gk: GlobalKeys):
    logger.info(f"Settings:{gk}")

    osim = get_simulation(gk)
    dir_trj = Path(gk.dir.trj)
    dir_trj.mkdir(parents=True, exist_ok=True)
    filename_stem = gk.filename.stem
    newcsv = dir_trj / (filename_stem + ".csv")
    newdcd = dir_trj / (filename_stem + ".dcd")
    newpdb = dir_trj / (filename_stem + ".pdb")
    newchk = dir_trj / (filename_stem + ".chk")

    nstep1 = gk.integrator.nstep_per_snapshot
    ndump = gk.integrator.nsnapshots
    nsteps = nstep1 * ndump

    new_em_csv = dir_trj / (filename_stem + f".em{gk.integrator.minimize}.csv")
    tol = 1.0 * unit.kilojoules_per_mole / unit.nanometer
    if MinimizeRelaxOption(gk.integrator.minimize) == MinimizeRelaxOption.l_bfgs:
        minreport = MinimizerReporter(logger)
        logger.info(f"Begin minimization maxIterations {nsteps} tol {tol}")
        osim.minimizeEnergy(tolerance=tol, maxIterations=nsteps, reporter=minreport)
        logger.info("End minimization")
    elif MinimizeRelaxOption(gk.integrator.minimize) == MinimizeRelaxOption.heating:
        # minimize 1000 steps
        osim.minimizeEnergy(tolerance=tol, maxIterations=1000)
        targetT = gk.integrator.targetT_K
        # 10 Kelvin, 20 Kelvin, ..., 300 Kelvin
        # 10000 steps for each temperature
        n_em_steps = 10000
        new_kelvin = 0.0
        while new_kelvin < targetT:
            new_kelvin += 10.0
            osim.integrator.setTemperature(new_kelvin)
            osim.context.setVelocitiesToTemperature(new_kelvin)
            osim.step(n_em_steps)
            logger.info(f"Equilibrated {n_em_steps} steps at {new_kelvin} Kelvin")
    elif MinimizeRelaxOption(gk.integrator.minimize) == MinimizeRelaxOption.brownian:
        gk_min3 = copy.deepcopy(gk)
        gk_min3.integrator.name = IntegratorNameOption.BrownianIntegrator.value
        osim_min3 = get_simulation(gk_min3)
        osim_min3.reporters.append(
            StateDataReporter(open(new_em_csv, "w"),
                              reportInterval=nstep1,
                              step=True,
                              time=True,
                              potentialEnergy=True,
                              kineticEnergy=True,
                              totalEnergy=True,
                              temperature=True,
                              volume=True,
                              elapsedTime=True,
                              separator=","))
        em3_step_count_min = 100000
        ndump3_min = (em3_step_count_min + nstep1 - 1) // nstep1
        ndump3 = max(ndump, ndump3_min)
        logger.info(f"Begin minimization/relaxation using BrownianIntegrator with {ndump3} iterations")
        for i in range(ndump3):
            osim_min3.step(nstep1)
            logger.info(f"Iterations completed {i}")
        logger.info("End minimization")
        em3_state = osim_min3.context.getState(positions=True, velocities=False, enforcePeriodicBox=True)
        osim.context.setPositions(em3_state.getPositions())
        osim.context.setPeriodicBoxVectors(*em3_state.getPeriodicBoxVectors())
    elif MinimizeRelaxOption(gk.integrator.minimize) == MinimizeRelaxOption.fire2:
        gk_min4 = copy.deepcopy(gk)
        gk_min4.integrator.name = IntegratorNameOption.FIRE2.value
        osim_min4 = get_simulation(gk_min4)
        osim_min4.reporters.append(
            StateDataReporter(open(new_em_csv, "w"),
                              reportInterval=nstep1,
                              step=True,
                              time=True,
                              potentialEnergy=True,
                              kineticEnergy=True,
                              totalEnergy=True,
                              temperature=True,
                              volume=True,
                              elapsedTime=True,
                              separator=","))
        em4_step_count_min = 100000
        ndump4_min = (em4_step_count_min + nstep1 - 1) // nstep1
        ndump4 = max(ndump, ndump4_min)
        for i in range(ndump4):
            osim_min4.step(nstep1)
            logger.info(f"Iterations completed {i}")
        logger.info("End minimization")
        em4_state = osim_min4.context.getState(positions=True, velocities=False, enforcePeriodicBox=True)
        osim.context.setPositions(em4_state.getPositions())
        osim.context.setPeriodicBoxVectors(*em4_state.getPeriodicBoxVectors())
    if MinimizeRelaxOption(gk.integrator.minimize) != MinimizeRelaxOption.none:
        newpos = osim.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
        newpbc = osim.context.getState().getPeriodicBoxVectors()
        write_pdb(osim.topology, newpos, newpbc, newpdb)
        return

    logger.info(f"Plan to sample {ndump} iterations")
    append_dcd: bool = False
    if Path(newdcd).exists() and Path(newchk).exists():
        nsampled = get_trj_nsnapshots(gk, newdcd)
        nsteps = nsteps - nsampled * nstep1
        osim.loadCheckpoint(open(newchk, "rb"))
        append_dcd = True
        logger.info(f"Already sampled {nsampled} iterations")
        if nsteps > 0:
            logger.info(f"Extending simulations from {newchk}")
        else:
            logger.info("No need to extend simulations")
    else:
        logger.info("Already sampled 0 iterations")
        logger.info("Starting new simulations")

    if nsteps > 0:
        osim.reporters.append(
            StateDataReporter(open(newcsv, "w"),
                              reportInterval=nstep1,
                              step=True,
                              time=True,
                              potentialEnergy=True,
                              kineticEnergy=True,
                              totalEnergy=True,
                              temperature=True,
                              volume=True,
                              elapsedTime=True,
                              separator=","))
        osim.reporters.append(CheckpointReporter(open(newchk, "wb"), reportInterval=nstep1))
        osim.reporters.append(DCDReporter(newdcd, reportInterval=nstep1, append=append_dcd, enforcePeriodicBox=False))
        osim.step(nsteps)
        newpos = osim.context.getState(getPositions=True).getPositions()
        newpbc = osim.context.getState().getPeriodicBoxVectors()
        write_pdb(osim.topology, newpos, newpbc, newpdb)
    logger.info("Finished")
