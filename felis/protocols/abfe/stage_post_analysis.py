# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import glob
import logging
import os
from pathlib import Path
import re

from felis.configs import load_config
from felis.protocols.abfe.config_types import ABFEInputConfig
from felis.protocols.abfe.config_types import ABStage
from felis.protocols.abfe.config_types import ABStageExecutionBase
from felis.protocols.abfe.config_types import ABStageExecutionRegister
from felis.protocols.abfe.main_fe_mbar import calc_mbar
from felis.protocols.abfe.main_fe_restraints import calc_restraints
from felis.protocols.abfe.main_fe_summarize import summarize_fe
from felis.protocols.boresch.main_boresch_restraints import BORESCH_CFG_JSON

logger = logging.getLogger(__name__)


@ABStageExecutionRegister(ABStage.mbar)
class ABStageExecution_mbar(ABStageExecutionBase):

    def __init__(self, acfg: ABFEInputConfig, **kwargs):
        super().__init__(**kwargs)
        self._acfg = acfg

    def exec(self):
        stage = self.stage
        planned_stages = self._planned_stages
        job_context = self._job_context
        acfg = self._acfg

        if stage not in planned_stages:
            return

        os.makedirs(job_context.analysis_dir, exist_ok=True)

        sol_nc_list = glob.glob(os.path.join(job_context.trj_dir, "a*.nc"))
        sol_nc_list = [nc for nc in sol_nc_list if re.search(r"^a\d+\.nc$", os.path.basename(nc))]
        try:
            done_file = os.path.join(job_context.progress_dir, f"{ABStage.sysA.value}.done")
            n_cudadev_a = load_config(open(done_file))["n_cuda_devices"]
            logger.info(f"Stage {ABStage.mbar.value}: Using n_cuda_devices = {n_cudadev_a} for {ABStage.sysA.value}")
        except Exception:
            n_cudadev_a = 8
            logger.warning(
                f"Stage {ABStage.mbar.value}: Assuming n_cuda_devices = {n_cudadev_a} for {ABStage.sysA.value}")
        expected_a_nc = [os.path.join(job_context.trj_dir, f"a{idx}.nc") for idx in range(n_cudadev_a)]
        if set(expected_a_nc) != set(sol_nc_list):
            raise ValueError(f"expected_a_nc {expected_a_nc} != pro_nc_list {sol_nc_list}")
        sol_nc_list.sort(key=lambda nc: int(os.path.basename(nc)[1:-3]))
        calc_mbar(
            stem="A",
            nc_list=sol_nc_list,
            checkpoint_interval=acfg.md_checkpoint_interval,
            outdir=job_context.analysis_dir,
        )

        pro_nc_list = glob.glob(os.path.join(job_context.trj_dir, "b*.nc"))
        pro_nc_list = [nc for nc in pro_nc_list if re.search(r"^b\d+\.nc$", os.path.basename(nc))]
        try:
            done_file = os.path.join(job_context.progress_dir, f"{ABStage.sysB.value}.done")
            n_cudadev_b = load_config(open(done_file))["n_cuda_devices"]
            logger.info(f"Stage {ABStage.mbar.value}: Using n_cuda_devices = {n_cudadev_b} for {ABStage.sysB.value}")
        except Exception:
            n_cudadev_b = 8
            logger.warning(
                f"Stage {ABStage.mbar.value}: Assuming n_cuda_devices = {n_cudadev_b} for {ABStage.sysB.value}")
        expected_b_nc = [os.path.join(job_context.trj_dir, f"b{idx}.nc") for idx in range(n_cudadev_b)]
        if set(expected_b_nc) != set(pro_nc_list):
            raise ValueError(f"expected_b_nc {expected_b_nc} != pro_nc_list {pro_nc_list}")
        pro_nc_list.sort(key=lambda nc: int(os.path.basename(nc)[1:-3]))
        calc_mbar(
            stem="B",
            nc_list=pro_nc_list,
            checkpoint_interval=acfg.md_checkpoint_interval,
            outdir=job_context.analysis_dir,
        )

        calc_restraints(
            boresch_cfg_json=f"{job_context.prepare_dir}/{BORESCH_CFG_JSON}",
            outdir=job_context.analysis_dir,
        )

        job_context.write_stage_done(stage)


@ABStageExecutionRegister(ABStage.summarize)
class ABStageExecution_summarize(ABStageExecutionBase):

    def __init__(self, acfg: ABFEInputConfig, **kwargs):
        super().__init__(**kwargs)
        self._acfg = acfg

    def exec(self):
        stage = self.stage
        planned_stages = self._planned_stages
        job_context = self._job_context
        acfg = self._acfg

        if stage not in planned_stages:
            return
        ligand = Path(acfg.sdffile).stem
        summarize_fe(
            workdir=job_context.analysis_dir,
            ligand=ligand,
            lam_sol_cfg=os.path.join(job_context.prepare_dir, "sysA_lam.json"),
            lam_pro_cfg=os.path.join(job_context.prepare_dir, "sysB_lam.json"),
        )

        job_context.write_stage_done(stage)
