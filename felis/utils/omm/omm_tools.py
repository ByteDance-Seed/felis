# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import openmm.unit as unit
from openmmtools import mcmc
from openmmtools import states
from openmmtools.multistate import MultiStateReporter
from openmmtools.multistate import MultiStateSamplerAnalyzer
from openmmtools.multistate import ReplicaExchangeAnalyzer
from openmmtools.multistate import ReplicaExchangeSampler
from openmmtools.states import SamplerState

from felis.configs import GlobalKeys
from felis.utils.mpi_tools import mpicomm
from felis.utils.omm.omm_system import get_system_topology

logger = logging.getLogger(__name__)

mpi_nproc, mpi_rank = mpicomm.mpi_nproc, mpicomm.mpi_rank
_NP_VALUE = int(os.environ.get("NP_VALUE", "-1"))
if _NP_VALUE < 0:
    _NP_VALUE = mpi_nproc

ReplicaExchangeSampler._global_citation_silence = True
MPS_N_CONTEXT_LIMIT = 48


def get_list_of_thermodynamic_states(gks: list[GlobalKeys]) -> list[states.ThermodynamicState]:

    def _build(gk):
        osys, _otop = get_system_topology(gk)
        return osys, gk.integrator.targetT_K

    max_workers = min(8, os.cpu_count() or 8)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        slist = list(ex.map(_build, gks))
    return [states.ThermodynamicState(system=s, temperature=t * unit.kelvin) for (s, t) in slist]


def _get_new_state_reporter(storage: str, gk: GlobalKeys) -> MultiStateReporter:
    return MultiStateReporter(storage, open_mode=None, checkpoint_interval=gk.openmm.checkpoint_interval)


def _get_state_reporter(storage: str, checkpoint_interval: int) -> Optional[MultiStateReporter]:
    if not Path(storage).is_file():
        return None
    return MultiStateReporter(storage, open_mode="r", checkpoint_interval=checkpoint_interval)


def _get_storage_name(gk: GlobalKeys) -> str:
    return str(Path(gk.dir.trj) / f"{gk.filename.stem}.nc")


def get_replica_exchange_sampler(thermo_states: list[states.ThermodynamicState], pos, box,
                                 gk: GlobalKeys) -> ReplicaExchangeSampler:
    itg = gk.integrator
    assert _NP_VALUE * len(thermo_states) <= MPS_N_CONTEXT_LIMIT
    assert itg.name.startswith("Langevin")
    move = mcmc.LangevinDynamicsMove(itg.dt_ps * unit.picoseconds,
                                     itg.friction_1_ps / unit.picoseconds,
                                     itg.nstep_per_snapshot,
                                     constraint_tolerance=itg.constraint_tol)
    storage = _get_storage_name(gk)
    create_done_flag = Path(os.path.splitext(storage)[0] + ".create_done")
    if create_done_flag.exists():
        try:
            simu = ReplicaExchangeSampler.from_storage(storage)
            logger.info("Resume ReplicaExchangeSampler")
            return simu
        except RuntimeError as e:
            if not "NetCDF" in str(e):
                raise
            # for "NetCDF: HDF error", delete all nc files and re-create the simulation
            if mpi_rank == 0:
                create_done_flag.unlink()

    if mpi_rank == 0:
        # clean up broken netcdf files
        if Path(storage).exists():
            Path(storage).unlink()
        checkpoint_storage = Path(os.path.splitext(storage)[0] + "_checkpoint.nc")
        if checkpoint_storage.exists():
            checkpoint_storage.unlink()
    mpicomm.barrier()
    new_reporter = _get_new_state_reporter(storage, gk)

    niter = gk.integrator.nsnapshots
    sampler_states = [states.SamplerState(positions=pos, velocities=None, box_vectors=box) for _ in thermo_states]
    # mixing_scheme = "swap-neighbors"
    mixing_scheme = "swap-all"
    simu = ReplicaExchangeSampler(mcmc_moves=move,
                                  number_of_iterations=niter,
                                  replica_mixing_scheme=mixing_scheme,
                                  online_analysis_interval=None)
    logger.info(f"Use ReplicaExchangeSampler {mixing_scheme}")
    logger.info(f"NStates {len(thermo_states)} NProcesses {mpi_nproc} NP_VALUE {_NP_VALUE}")
    simu.create(thermo_states, sampler_states=sampler_states, storage=new_reporter)

    if mpi_rank == 0:
        with open(create_done_flag, "w") as f:
            # avoid empty file
            f.write("create_done")
    return simu


def get_replica_exchange_analyzer(nc_file: Optional[str], gk: Optional[GlobalKeys]) -> ReplicaExchangeAnalyzer:
    if nc_file is None:
        storage = _get_storage_name(gk)
    else:
        storage = nc_file
    assert not (nc_file is None and gk is None)
    if gk is None:
        gk = GlobalKeys()
    reporter = _get_state_reporter(storage, gk.openmm.checkpoint_interval)
    return ReplicaExchangeAnalyzer(reporter, analysis_kwargs={"solver_protocol": "robust"})


def get_n_iterations(gk: GlobalKeys) -> tuple[int, int]:

    def impl_rank0(gk: GlobalKeys) -> tuple[int, int]:
        storage = _get_storage_name(gk)
        ntotal = gk.integrator.nsnapshots
        logger.info(f"Plan to sample {ntotal} iterations")
        if Path(storage).is_file():
            al = get_replica_exchange_analyzer(storage, gk)
            niter = al.n_iterations
            logger.info(f"Already sampled {niter} iterations")
            if ntotal > niter:
                return ntotal - niter, ntotal
            else:
                return 0, ntotal
        else:
            logger.info("Already sampled 0 iterations")
            return ntotal, ntotal

    extra_niter, ntotal = None, None
    if mpi_rank == 0:
        extra_niter, ntotal = impl_rank0(gk)
    extra_niter = mpicomm.bcast(extra_niter, root=0)
    ntotal = mpicomm.bcast(ntotal, root=0)
    return extra_niter, ntotal


def extract_state_from_netdcd(nc_file: str, target_state: int, checkpoint_interval: int) -> tuple[list, list]:
    reporter = MultiStateReporter(nc_file, open_mode="r", checkpoint_interval=checkpoint_interval)
    alyz = MultiStateSamplerAnalyzer(reporter)
    _, _, _, replica_index = alyz.read_energies()
    n_states, n_iters = replica_index.shape

    assert target_state < n_states
    mask = (replica_index == target_state)
    ireplica_list = []
    for i in range(n_iters):
        imask = np.array(list(range(n_states)))[mask[:, i]]
        assert len(imask) == 1
        ireplica_list.append(imask[0])
    logger.info(f"Total number of states {n_states}")
    logger.info(f"Total number of iterations {n_iters}")
    logger.info(f"Replica index {replica_index}")
    logger.info(f"State {target_state} can be found in replicas {ireplica_list}")

    coords, pbcvecs = [], []
    for i in range(n_iters):
        if i % 200 == 0:
            logger.info(f"Processing iteration {i}")

        state_list = reporter.read_sampler_states(i)
        if not state_list:
            # if checkpoint_interval > 1, valid iter_count is not continuous
            continue
        sample_state: SamplerState = state_list[ireplica_list[i]]

        pos_quantity = sample_state.positions
        pbcvecs_quantity = sample_state.box_vectors
        coords.append(pos_quantity)
        pbcvecs.append(pbcvecs_quantity)

    return coords, pbcvecs


def get_mbar_results(nc_file: str, gk: GlobalKeys):
    alyz = get_replica_exchange_analyzer(nc_file=nc_file, gk=gk)
    kT_kJ_mol = alyz.kT / unit.kilojoule_per_mole
    fe, std_fe = alyz.get_free_energy()
    matrix, eigenval, ineff = alyz.generate_mixing_statistics()
    n_states = alyz.n_states
    n_discard = alyz.n_equilibration_iterations
    ematx, _umatx, _nbs, ridx = alyz.read_energies()
    logger.info(f"nc file: {nc_file} n_discard {n_discard}")

    return {
        "kT_kJ_mol": kT_kJ_mol,
        "fe": fe,
        "std_fe": std_fe,
        "n_states": n_states,
        "n_discard": n_discard,
        "energy_matrix": ematx,
        "replica_index": ridx,
        "mixing_matrix": matrix,
        "mixing_matrix_eigenvalues": eigenval,
        "mixing_matrix_inefficiency": ineff,
    }
