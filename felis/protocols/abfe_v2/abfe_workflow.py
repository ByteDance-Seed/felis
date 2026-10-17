# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
from pathlib import Path

from felis.protocols.abfe.config_types import ABFEInputConfig
from felis.protocols.abfe.config_types import ABStage
from felis.protocols.abfe.config_types import ABStageExecutionBase
from felis.protocols.abfe.job_context import JobContext
from felis.utils import get_visible_cuda_devices

logger = logging.getLogger(__name__)


def mainfunc(acfg: ABFEInputConfig):
    logger.info("##################")
    logger.info("#  NEW ABFE JOB  #")
    logger.info("##################")
    logger.info(f"abfe config:\n{acfg}")

    cuda_device_ids = get_visible_cuda_devices()
    n_cuda_devices = len(cuda_device_ids)
    logger.info(f"n_cuda_devices {n_cuda_devices}")
    sdf_stem = Path(acfg.sdffile).stem
    sdf_abs = str(Path(acfg.sdffile).absolute())
    _tmpdir = acfg.tmpdir
    subprocess_wd = str(Path(_tmpdir) / sdf_stem)
    extra_envs = dict()
    for envstr in acfg.env_list:
        env1, value1 = envstr.split(":")
        extra_envs[env1] = value1
    job_context = JobContext(subprocess_wd, extra_envs=extra_envs, use_mps=False)

    if "all" in acfg.stages:
        # Lambda optimization is an explicit preparatory workflow. Do not add
        # multiple short HREMD optimization runs to every normal ABFE v2 job.
        planned_stages = [
            ABStage.makebox,
            ABStage.boresch_em_v2,
            ABStage.boresch_npt_v2,
            ABStage.boresch_post_process_v2,
            ABStage.sysA_em_v2,
            ABStage.sysA_v2,
            ABStage.sysB_em_v2,
            ABStage.sysB_v2,
            ABStage.post_analysis_v2,
        ]
    else:
        planned_stages = [ABStage(x) for x in acfg.stages]
    planned_stages = [stage for stage in planned_stages if not job_context.check_stage_done(stage)]
    
    ################################
    # makebox
    ################################
    
    if ABStage.makebox in planned_stages:
        exec_makebox = ABStageExecutionBase.create(ABStage.makebox,
                                                acfg,
                                                sdf_abs,
                                                planned_stages=planned_stages,
                                                job_context=job_context)
        exec_makebox.exec()

    ################################
    # boresch - v2
    ################################

    if ABStage.boresch_em_v2 in planned_stages or ABStage.boresch_npt_v2 in planned_stages or ABStage.boresch_post_process_v2 in planned_stages:
        exec_boresch_em = ABStageExecutionBase.create(ABStage.boresch_em_v2,
                                                    acfg,
                                                    planned_stages=planned_stages,
                                                    job_context=job_context)
        exec_boresch_em.exec()

        exec_boresch_npt = ABStageExecutionBase.create(ABStage.boresch_npt_v2,
                                                    acfg,
                                                    sdf_abs = sdf_abs,
                                                    planned_stages=planned_stages,
                                                    job_context=job_context)
        exec_boresch_npt.exec()

        exec_boresch_post = ABStageExecutionBase.create(ABStage.boresch_post_process_v2,
                                                        acfg,
                                                        planned_stages=planned_stages,
                                                        job_context=job_context)
        exec_boresch_post.exec()

    ################################
    # sysA
    ################################

    exec_sysA_em = ABStageExecutionBase.create(ABStage.sysA_em_v2,
                                               acfg,
                                               planned_stages=planned_stages,
                                               job_context=job_context)
    exec_sysA_em.exec()

    n_cudadev_a = n_cuda_devices
    exec_sysA_opt_lambdas = ABStageExecutionBase.create(ABStage.sysA_opt_lambdas_v2,
                                                        acfg,
                                                        sdf_abs=sdf_abs,
                                                        n_cuda_devices=n_cudadev_a,
                                                        planned_stages=planned_stages,
                                                        job_context=job_context)
    exec_sysA_opt_lambdas.exec()

    exec_sysA = ABStageExecutionBase.create(ABStage.sysA_v2,
                                            acfg,
                                            n_cuda_devices=n_cudadev_a,
                                            sdf_abs=sdf_abs,
                                            planned_stages=planned_stages,
                                            job_context=job_context)
    exec_sysA.exec()

    ################################
    # sysB
    ################################

    exec_sysB_em = ABStageExecutionBase.create(ABStage.sysB_em_v2,
                                               acfg,
                                               planned_stages=planned_stages,
                                               job_context=job_context)
    exec_sysB_em.exec()

    n_cudadev_b = n_cuda_devices
    exec_sysB_opt_lambdas = ABStageExecutionBase.create(ABStage.sysB_opt_lambdas_v2,
                                                        acfg,
                                                        sdf_abs=sdf_abs,
                                                        n_cuda_devices=n_cudadev_b,
                                                        planned_stages=planned_stages,
                                                        job_context=job_context)
    exec_sysB_opt_lambdas.exec()

    exec_sysB = ABStageExecutionBase.create(ABStage.sysB_v2,
                                            acfg,
                                            sdf_abs=sdf_abs,
                                            n_cuda_devices=n_cudadev_b,
                                            planned_stages=planned_stages,
                                            job_context=job_context)
    exec_sysB.exec()

    ################################
    # post-analysis
    ################################

    exec_post_analysis = ABStageExecutionBase.create(ABStage.post_analysis_v2, acfg, planned_stages=planned_stages, job_context=job_context)
    exec_post_analysis.exec()
