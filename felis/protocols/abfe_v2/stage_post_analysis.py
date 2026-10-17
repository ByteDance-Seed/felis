# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import os

from felis.protocols.abfe.config_types import ABFEInputConfig, ABStage, ABStageExecutionBase, ABStageExecutionRegister
from felis.protocols.abfe.main_fe_restraints import calc_restraints
from felis.protocols.abfe_v2.lambda_config import read_lambda_schedule
from felis.protocols.abfe_v2.protein_lambda_rules import validate_protein_lambda_schedule
from felis.protocols.abfe_v2.utilities import generate_post_analysis_figures, read_records
from felis.protocols.boresch.main_boresch_restraints import BORESCH_CFG_JSON

logger = logging.getLogger(__name__)


@ABStageExecutionRegister(ABStage.post_analysis_v2)
class ABStageExecution_post_analysis_v2(ABStageExecutionBase):

    def __init__(self, acfg: ABFEInputConfig, **kwargs):
        super().__init__(**kwargs)
        self._acfg = acfg

    def exec(self):
        stage = self.stage
        planned_stages = self._planned_stages
        job_context = self._job_context

        if stage not in planned_stages:
            return

        sysB_lam_path = os.path.join(job_context.prepare_dir, "sysB_lam.json")
        sysB_lam = read_lambda_schedule(sysB_lam_path)
        validate_protein_lambda_schedule(sysB_lam, source=f"production protein-leg lambda schedule {sysB_lam_path}")

        os.makedirs(job_context.analysis_dir, exist_ok=True)

        sysA_records_file = os.path.join(job_context.trj_dir, "a0_hremd", "records.sqlite")
        sysB_records_file = os.path.join(job_context.trj_dir, "b0_hremd", "records.sqlite")

        sysA_records, sysA_metadata = read_records(
            sysA_records_file,
            include_metadata=True,
        )
        if not sysA_records:
            raise FileNotFoundError(f"No HREMD energy records found in {sysA_records_file}")
        generate_post_analysis_figures(
            energy_records=sysA_records,
            metadata=sysA_metadata,
            analysis_dir=job_context.analysis_dir,
            prefix="sysA",
        )

        sysB_records, sysB_metadata = read_records(
            sysB_records_file,
            include_metadata=True,
        )
        if not sysB_records:
            raise FileNotFoundError(f"No HREMD energy records found in {sysB_records_file}")
        generate_post_analysis_figures(
            energy_records=sysB_records,
            metadata=sysB_metadata,
            analysis_dir=job_context.analysis_dir,
            prefix="sysB",
        )

        calc_restraints(
            boresch_cfg_json=f"{job_context.prepare_dir}/{BORESCH_CFG_JSON}",
            outdir=job_context.analysis_dir,
        )

        job_context.write_stage_done(stage)
