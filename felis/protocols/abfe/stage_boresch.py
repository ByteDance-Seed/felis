# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
from pathlib import Path

from felis.protocols.abfe.config_types import ABFEInputConfig, ABStage, ABStageExecutionBase, ABStageExecutionRegister
from felis.protocols.boresch.main_boresch_restraints import generate_boresch_restraints
from felis.utils import get_visible_cuda_devices
from felis.utils.omm.format_tools import trj_nc2dcd
from felis.utils.setup_logger import get_formatter, get_root_logger


@ABStageExecutionRegister(ABStage.boresch_em)
class ABStageExecution_boresch_em(ABStageExecutionBase):

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
            "--tkv",
            "s:filename.stem:sysB_boresch_em",
            "s:filename.sys:prepare/sysB.top",
            "s:filename.crd:prepare/sysB.gro",
            "s:dir.trj:trj",
            f"i:integrator.minimize:{em_version}",
            "i:integrator.nsnapshots:20",
        ]
        assert 0 == job_context.run_one_call(cmd)
        shutil.copy(os.path.join(job_context.trj_dir, "sysB_boresch_em.pdb"),
                    os.path.join(job_context.prepare_dir, f"sysB_no_rstrn_em{em_version}.pdb"))

        job_context.write_stage_done(stage)


@ABStageExecutionRegister(ABStage.boresch_npt)
class ABStageExecution_boresch_npt(ABStageExecutionBase):

    def __init__(self, acfg: ABFEInputConfig, np: int = 4, n_states: int = 4, discard: int = 100, **kwargs):
        super().__init__(**kwargs)
        self._acfg = acfg
        self._np = np
        self._n_states = n_states
        self._discard = discard

    def exec(self):
        stage = self.stage
        planned_stages = self._planned_stages
        job_context = self._job_context
        acfg = self._acfg
        np = self._np
        n_states = self._n_states
        discard = self._discard

        if stage not in planned_stages:
            return

        cmd = [
            "python3",
            "-m",
            "felis.app.dyn.repex",
            "--tkv",
            "s:filename.stem:sysB_boresch_npt",
            "s:filename.sys:prepare/sysB.top",
            "s:filename.crd:trj/sysB_boresch_em.pdb",
            "s:dir.trj:trj",
            "i:openmm.checkpoint_interval:1",
            "i:integrator.npt:1",
            f"i:integrator.nsnapshots:{acfg.md_eq_nsnapshots}",
            "i:integrator.nstep_per_snapshot:2500",
            "--rextkv",
        ]
        cmd.extend(["f:integrator.targetT_K:298.15"] * n_states)

        cuda_device_ids = get_visible_cuda_devices()
        if cuda_device_ids:
            device_id = cuda_device_ids[0]
        else:
            raise ValueError("No visible CUDA devices found.")

        boresch_npt_cmd = job_context.build_cuda_mps_mpirun_command(cmd, device_id, np=np)
        assert job_context.run_one_call(boresch_npt_cmd) == 0

        sys_pdb = os.path.join(job_context.trj_dir, "sysB_boresch_em.pdb")
        sys_top = os.path.join(job_context.prepare_dir, "sysB.top")
        trj_dcd = os.path.join(job_context.trj_dir, "sysB_boresch_npt.dcd")
        trj_nc2dcd(
            nc=os.path.join(job_context.trj_dir, "sysB_boresch_npt.nc"),
            state_index=list(range(n_states)),
            discard=discard,
            sys_pdb=sys_pdb,
            sys_top=sys_top,
            trj_dcd=trj_dcd,
            nstep_per_snapshot=2500,
        )

        job_context.write_stage_done(stage)


@ABStageExecutionRegister(ABStage.boresch_post_process)
class ABStageExecution_boresch_post_process(ABStageExecutionBase):

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

        kr, ka, kdih = acfg.k_r_a_dih
        lig_sdf = str(Path(acfg.sdffile).absolute())
        sys_top = os.path.join(job_context.prepare_dir, "sysB.top")
        trj_dcd = os.path.join(job_context.trj_dir, "sysB_boresch_npt.dcd")

        gen_restrn_logger = get_root_logger()
        gen_restrn_logname = os.path.join(job_context.prepare_dir, "sys_boresch0.log")
        gen_restrn_handler = logging.FileHandler(gen_restrn_logname, mode="a")
        gen_restrn_handler.setFormatter(get_formatter(with_lineno=True))
        gen_restrn_logger.addHandler(gen_restrn_handler)
        try:
            generate_boresch_restraints(
                kr=kr,
                ka=ka,
                kdih=kdih,
                lig_sdf=lig_sdf,
                sys_top=sys_top,
                trj_dcd=trj_dcd,
                atom_ids_json=os.path.join(job_context.prepare_dir, "sysB_atom_ids.json"),
                outdir=job_context.prepare_dir,
            )
        finally:
            gen_restrn_logger.removeHandler(gen_restrn_handler)
            gen_restrn_handler.close()

        job_context.write_stage_done(stage)
