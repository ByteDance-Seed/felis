# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
from pathlib import Path

from felis.configs.global_keys import GlobalKeys
from felis.protocols.dyn.hremd import run_hremd
from felis.protocols.dyn.replica import get_list_of_replicas
from felis.utils import get_cuda_device_indices

logger = logging.getLogger(__name__)


def mainfunc(gk_list: list[GlobalKeys]) -> None:
    for i, gk in enumerate(gk_list):
        logger.info(f"Thermodynamic State {i}{gk}")

    gk = gk_list[0]
    replicas = get_list_of_replicas(gk_list)

    total_steps = gk.integrator.nsnapshots * gk.integrator.nstep_per_snapshot
    exchange_interval = gk.openmm.exchange_interval
    # TODO: we shall add logger_record_interval
    logger_record_interval = exchange_interval
    traj_record_interval = gk.integrator.nstep_per_snapshot
    task_dir = str(Path(gk.dir.trj) / f"{gk.filename.stem}_hremd")
    with_pv = bool(gk.integrator.npt)
    checkpoint_interval = gk.openmm.checkpoint_interval
    tmp_directory = gk.openmm.tmp_directory
    resume = gk.openmm.resume

    # CUDA_VISIBLE_DEVICES may contain physical IDs or UUIDs, but OpenMM
    # DeviceIndex must use the dense logical ordinals visible to this process.
    gpu_ids = get_cuda_device_indices()
    logger.info(f"Detected logical GPU ids: {gpu_ids}")

    run_hremd(
        replicas=replicas,
        total_steps=total_steps,
        exchange_interval=exchange_interval,
        logger_record_interval=logger_record_interval,
        traj_record_interval=traj_record_interval,
        task_dir=task_dir,
        temperature=gk.integrator.targetT_K,
        gpu_devices=gpu_ids,
        exchange_strategy="neighbor",
        with_pv=with_pv,
        checkpoint_interval=checkpoint_interval,
        resume=resume,
        enable_profiling=False,
        tmp_directory=tmp_directory,
    )
