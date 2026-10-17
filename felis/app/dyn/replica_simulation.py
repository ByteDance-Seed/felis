# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import argparse
import copy
from felis.utils import mkdir_and_get_filepath
from felis.configs.global_keys import GlobalKeys
from felis.protocols.dyn.replica import Replica
from felis.utils.omm.omm_system import get_system_topology
from felis.utils.omm.omm_system import get_initial_pos_pbc
from felis.utils.omm.omm_system import get_integrator
from felis.utils.setup_logger import setup_logger
from openmm import Platform

DEFAULT_LAMBDA_SCHEDULES = {
    "lambdaA_charge": [1.0],
    "lambdaB_charge": [0.0],
    "lambdaC_charge": [0.0],
    "lambdaA_vdw": [1.0],
    "lambdaB_vdw": [0.0],
    "torsionLambda": [1.0],
    "reslam": [0.0],
    "rest2_scale": [1.0],
}

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, nargs="*", default=[])
    ap.add_argument("--tkv", type=str, nargs="*", default=[])
    ap.add_argument("--rextkv", type=str, nargs="*", default=[])
    return ap


def mainfunc(argv: list[str]) -> None:
    args = _build_parser().parse_args(argv)

    gk0 = GlobalKeys()
    for icfg in args.cfg:
        gk0.update_by_cfg(icfg)
    for itkv in args.tkv:
        gk0.update_by_tkv(itkv)

    gk_list: list[GlobalKeys] = []
    for irextkv in args.rextkv:
        igk = copy.deepcopy(gk0)
        igk.update_by_comma_sep_tkv(irextkv)
        igk.check()
        gk_list.append(igk)

    if not gk_list:
        gk0.check()
        gk_list.append(gk0)

    for idx, igk in enumerate(gk_list):
        igk.integrator.randomseed += idx
        igk.check()

    gk = gk_list[0]
    newlog = mkdir_and_get_filepath(basedir=None, dirname=gk.dir.trj, filename=gk.filename.stem, ext=".dynlog")
    logger = setup_logger(newlog)

    osys, otop = get_system_topology(gk0, enable_perturb_nonbonded=True, customLambdaSchedules=DEFAULT_LAMBDA_SCHEDULES)
    pos0, box0 = get_initial_pos_pbc(gk0)
    traj_record_interval = gk.integrator.nstep_per_snapshot
    traj_dt_ps = gk.integrator.dt_ps
    is_boresch_restraint = False


    replica_0 = Replica(
        system=osys,
        topology=otop,
        is_boresch_restraint=is_boresch_restraint,
        replica_id=0,
        lambda_schedules = DEFAULT_LAMBDA_SCHEDULES,
        pressure=gk0.integrator.targetP_bar,
        temperature=gk0.integrator.targetT_K,
        initial_positions=pos0,
        initial_box_vectors=box0,
        velocity_temperature_K=gk0.integrator.targetT_K,
        velocity_seed=gk0.integrator.randomseed,
    )
    replica_0.set_integrator(get_integrator(gk0))
    replica_0.set_num_replicas(1) # This is a single replica simulation.
    platform = Platform.getPlatformByName("CUDA")
    platform_properties = {"Precision": "mixed", "DeviceIndex": "0"}
    replica_0.set_platform(platform, platform_properties)
    replica_0.init_context()

    state = replica_0.context.getState(energy=True, positions=True, velocities=True)
    logger.info(f"Boresch NPT Init Potential Energy: {state.getPotentialEnergy()}")

    traj_dcd_file = os.path.join(gk.dir.trj, "sysB_boresch_npt.dcd")
    replica_0.open_trajectory(traj_dcd_file, traj_dt_ps)
    
    # TODO: add reporter to record energy in boresch npt
    for _ in range(gk.integrator.nsnapshots):
        replica_0.step(traj_record_interval)
        replica_0.write_trajectory()
    replica_0.close_trajectory()

if __name__ == "__main__":
    mainfunc(None)
