# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil

from felis.configs import dump_config
from felis.protocols.abfe.config_types import ABFEInputConfig
from felis.protocols.abfe.config_types import ABStage
from felis.protocols.abfe.config_types import ABStageExecutionBase
from felis.protocols.abfe.config_types import ABStageExecutionRegister
from felis.protocols.abfe.recipes import get_r_lambdas_dim3
from felis.protocols.abfe.recipes import get_ve_lambdas_with_res
from felis.protocols.abfe.recipes import split_replica_exchange_jobs
from felis.protocols.boresch.main_boresch_restraints import BORESCH_CFG_JSON

logger = logging.getLogger(__name__)


@ABStageExecutionRegister(ABStage.sysB_em)
class ABStageExecution_sysB_em(ABStageExecutionBase):

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

        em_version = acfg.md_pro_em_version
        cmd = [
            "python3",
            "-m",
            "felis.app.dyn.vanilla",
            "--cfg",
            "prepare/sysB_posres.json",
            f"prepare/{BORESCH_CFG_JSON}",
            "--tkv",
            "s:filename.stem:sysB_boresch_filtered",
            "s:filename.sys:prepare/sysB.top",
            "s:filename.crd:prepare/sysB.gro",
            "s:dir.trj:trj",
            f"i:integrator.minimize:{em_version}",
            "i:integrator.nsnapshots:20",
        ]
        assert job_context.run_one_call(cmd) == 0
        shutil.copy(os.path.join(job_context.trj_dir, "sysB_boresch_filtered.pdb"),
                    os.path.join(job_context.prepare_dir, "sysB_boresch_refined.pdb"))

        em_version = acfg.md_pro_em_version
        cmd_sysB_em = [
            "python3",
            "-m",
            "felis.app.dyn.vanilla",
            "--cfg",
            f"prepare/{BORESCH_CFG_JSON}",
            "--tkv",
            "s:filename.stem:sysB_em",
            "s:filename.sys:prepare/sysB.top",
            "s:filename.crd:trj/sysB_boresch_filtered.pdb",
            "s:dir.trj:trj",
            f"i:integrator.minimize:{em_version}",
            "i:integrator.nsnapshots:20",
        ]

        assert 0 == job_context.run_one_call(cmd_sysB_em)
        shutil.copy(os.path.join(job_context.trj_dir, "sysB_em.pdb"),
                    os.path.join(job_context.prepare_dir, "sysB_em.pdb"))
        shutil.copy(os.path.join(job_context.trj_dir, "sysB_em.pdb"),
                    os.path.join(job_context.prepare_dir, f"sysB_rstrn_em{em_version}.pdb"))

        job_context.write_stage_done(stage)


@ABStageExecutionRegister(ABStage.sysB)
class ABStageExecution_sysB(ABStageExecutionBase):

    def __init__(self, acfg: ABFEInputConfig, sdf_abs: str, n_cuda_devices: int, np: int = 4, **kwargs):
        super().__init__(**kwargs)
        self._acfg = acfg
        self._sdf_abs = sdf_abs
        self._n_cuda_devices = n_cuda_devices
        self._np = np

    def exec(self):
        stage = self.stage
        planned_stages = self._planned_stages
        job_context = self._job_context
        acfg = self._acfg
        sdf_abs = self._sdf_abs
        n_cuda_devices = self._n_cuda_devices
        np = self._np

        if stage not in planned_stages:
            return

        # [v, e, r]: [0., 0., 1.] -> [1., 1., 1.] -> [1., 1., 0.]
        lam_v = get_ve_lambdas_with_res(acfg.vlamrecipe,
                                        "v",
                                        ascend=True,
                                        reslam=1.0,
                                        suppl_lams=acfg.supplementary_vdw_lambda_list)
        lam_e = get_ve_lambdas_with_res(acfg.elamrecipe,
                                        "e",
                                        ascend=True,
                                        reslam=1.0,
                                        suppl_lams=acfg.supplementary_elec_lambda_list)
        lam_r = get_r_lambdas_dim3(acfg.reslamrecipe,
                                   ascend=False,
                                   vlam=1.0,
                                   elam=1.0,
                                   suppl_lams=acfg.supplementary_restraint_lambda_list)
        assert lam_v[-1] == lam_e[0]
        assert lam_e[-1] == lam_r[0]
        sysB_lam = (lam_v + lam_e[1:] + lam_r[1:]).copy()
        lam_idx_groups = split_replica_exchange_jobs(n_cuda_devices, list(range(len(sysB_lam))))
        for idx, lams in enumerate(lam_idx_groups):
            logger.info(f"Stage {stage.value} lambdas: {idx} {lams}")

        with open(os.path.join(job_context.prepare_dir, "sysB_lam.json"), "w") as f:
            dump_config({"ab": {"lam_list": sysB_lam}}, f, indent=4)

        cmd_list: list[list[str]] = []
        for idx, igroup in enumerate(lam_idx_groups):
            cmd = [
                "python3",
                "-m",
                "felis.app.dyn.repex",
                "--cfg",
                "prepare/sysB_lam.json",
                "prepare/sysB_ab_ligatoms.json",
                f"prepare/{BORESCH_CFG_JSON}",
                "--tkv",
                f"s:filename.stem:b{idx}",
                f"s:filename.monomer:{sdf_abs}",
                "s:filename.sys:prepare/sysB.top",
                "s:filename.crd:prepare/sysB_em.pdb",
                "s:filename.atom_ids:prepare/sysB_atom_ids.json",
                "s:dir.trj:trj",
                f"i:openmm.checkpoint_interval:{acfg.md_checkpoint_interval}",
                "i:integrator.npt:1",
                "i:integrator.nstep_per_snapshot:2500",
                f"i:integrator.nsnapshots:{acfg.md_pro_nsnapshots}",
                "--rextkv",
            ]
            cmd.extend([f"i:ab.ilam:{vv}" for vv in igroup])
            cmd_list.append(job_context.build_cuda_mps_mpirun_command(cmd, gpu_id=idx, np=np))

        plist = []
        for cmd in cmd_list:
            plist.append(job_context.run_one_subprocess_nowait(cmd, working_dir=None))
        for idx, p in enumerate(plist):
            assert p.wait() == 0, f"sysB {idx} failed"

        job_context.write_stage_done(stage, {"n_cuda_devices": n_cuda_devices})
