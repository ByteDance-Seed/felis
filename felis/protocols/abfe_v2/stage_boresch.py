# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import os
import shutil
from pathlib import Path

from felis.configs import GlobalKeys, dump_config
from felis.external.mda_tools import concatenate_trajectory_files
from felis.external.mda_tools import discard_initial_trajectory_frames
from felis.protocols.abfe.config_types import ABFEInputConfig
from felis.protocols.abfe.config_types import ABStage
from felis.protocols.abfe.config_types import ABStageExecutionBase
from felis.protocols.abfe.config_types import ABStageExecutionRegister
from felis.utils import get_visible_cuda_devices
from felis.utils.setup_logger import get_formatter, get_root_logger

def _constant_boresch_v2_lam_list(n_states: int) -> list[list[float]]:
    return [[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0] for _ in range(n_states)]


@ABStageExecutionRegister(ABStage.boresch_em_v2)
class ABStageExecution_boresch_em_v2(ABStageExecutionBase):

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

@ABStageExecutionRegister(ABStage.boresch_npt_v2)
class ABStageExecution_boresch_npt_v2(ABStageExecutionBase):

    def __init__(self, acfg: ABFEInputConfig, sdf_abs: str, np: int = 4, n_states: int = 4, discard: int = 100, **kwargs):
        super().__init__(**kwargs)
        self._acfg = acfg
        self._discard = discard
        self._sdf_abs = sdf_abs
        self._n_states = n_states

    def exec(self):
        stage = self.stage
        planned_stages = self._planned_stages
        job_context = self._job_context
        acfg = self._acfg

        if stage not in planned_stages:
            return

        if acfg.boresch_sampling_protocol == "hremd_same_temperature":
            self._exec_hremd()
        elif acfg.boresch_sampling_protocol == "long_npt":
            self._exec_long_npt()
        else:
            raise ValueError(f"Unknown boresch_sampling_protocol: {acfg.boresch_sampling_protocol}")

        job_context.write_stage_done(stage)

    def _base_tkv(self, nsnapshots: int) -> list[str]:
        acfg = self._acfg
        sdf_abs = self._sdf_abs
        return [
            f"s:filename.monomer:{sdf_abs}",
            "s:filename.stem:sysB_boresch_npt",
            "s:filename.sys:prepare/sysB.top",
            "s:filename.crd:trj/sysB_boresch_em.pdb",
            "s:filename.atom_ids:prepare/sysB_atom_ids.json",
            "s:dir.trj:trj",
            f"i:openmm.checkpoint_interval:{acfg.md_checkpoint_interval}",
            f"i:openmm.exchange_interval:{acfg.md_exchange_interval}",
            f"s:openmm.tmp_directory:{acfg.omm_tmp_directory}",
            "i:integrator.npt:1",
            acfg.dt_tkv(),
            acfg.hmr_tkv(),
            f"i:integrator.nsnapshots:{nsnapshots}",
            f"i:integrator.nstep_per_snapshot:{acfg.md_boresch_traj_interval}",
        ]

    def _exec_long_npt(self):
        acfg = self._acfg
        job_context = self._job_context
        discard = self._discard

        cmd = [
            "python3",
            "-m",
            "felis.app.dyn.replica_simulation",
            "--cfg",
            "prepare/sysB_ab_ligatoms.json",
            "--tkv",
        ]
        cmd.extend(self._base_tkv(nsnapshots=acfg.md_boresch_nsnapshots * 4))
        cmd.extend(["--rextkv", "f:integrator.targetT_K:298.15"])

        cuda_device_ids = get_visible_cuda_devices()
        if cuda_device_ids:
            device_id = cuda_device_ids[0]
        else:
            raise ValueError("No visible CUDA devices found.")

        boresch_npt_cmd = job_context.build_cuda_mps_command(cmd, gpu_id=[device_id])
        assert job_context.run_one_call(boresch_npt_cmd) == 0

        sys_top = os.path.join(job_context.prepare_dir, "sysB.top")
        trj_dcd = os.path.join(job_context.trj_dir, "sysB_boresch_npt.dcd")
        gk = GlobalKeys()
        gk.filename.sys = sys_top
        discard_initial_trajectory_frames(gk, trj_dcd, discard)

    def _exec_hremd(self):
        acfg = self._acfg
        job_context = self._job_context
        n_states = self._n_states
        discard = self._discard

        lam_path = os.path.join(job_context.prepare_dir, "sysB_boresch_lam.json")
        with open(lam_path, "w") as f:
            dump_config({"ab": {"lam_list": _constant_boresch_v2_lam_list(n_states)}}, f, indent=4)

        cmd = [
            "python3",
            "-m",
            "felis.app.dyn.repex_v2",
            "--cfg",
            "prepare/sysB_boresch_lam.json",
            "prepare/sysB_ab_ligatoms.json",
            "--tkv",
        ]
        cmd.extend(self._base_tkv(nsnapshots=acfg.md_boresch_nsnapshots))
        cmd.extend(["--rextkv"])
        cmd.extend(["f:integrator.targetT_K:298.15"] * n_states)

        cuda_device_ids = get_visible_cuda_devices()
        if cuda_device_ids:
            gpu_ids = cuda_device_ids
        else:
            raise ValueError("No visible CUDA devices found.")

        boresch_npt_cmd = job_context.build_cuda_mps_command(cmd, gpu_id=gpu_ids)
        assert job_context.run_one_call(boresch_npt_cmd) == 0

        sys_top = os.path.join(job_context.prepare_dir, "sysB.top")
        gk = GlobalKeys()
        gk.filename.sys = sys_top
        hremd_dir = Path(job_context.trj_dir) / "sysB_boresch_npt_hremd"
        trj_list = [str(hremd_dir / f"replica_{idx}" / "traj.dcd") for idx in range(n_states)]
        trj_dcd = os.path.join(job_context.trj_dir, "sysB_boresch_npt.dcd")
        concatenate_trajectory_files(gk, trj_list, trj_dcd, discard=discard)

@ABStageExecutionRegister(ABStage.boresch_post_process_v2)
class ABStageExecution_boresch_post_process_v2(ABStageExecutionBase):

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
            from felis.protocols.boresch.main_boresch_restraints import generate_boresch_restraints

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
