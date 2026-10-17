# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import rdkit.Chem as Chem

from felis.protocols.abfe.config_types import ABFEInputConfig, ABStage, ABStageExecutionBase, ABStageExecutionRegister
from felis.protocols.abfe_v2.recipes import (
    get_neutral_water_leg_lambdas,
    get_charged_water_leg_lambdas,
)
from felis.protocols.abfe_v2.lambda_config import select_lambda_schedule
from felis.protocols.abfe_v2.utilities import (
    analyze_results,
    get_energy_matrix_from_records,
    read_records,
)
from felis.protocols.abfe_v2.draw_figures import plot_lambda_schedule
from felis.protocols.abfe_v2.draw_figures import plot_overlap_matrix
from felis.protocols.abfe_v2.lambda_schedule_optimizer import (
    SigmoidShape,
    infer_t_split_from_schedule,
    optimize_lambda_schedule,
)
from felis.protocols.abfe_v2.protein_lambda_rules import freeze_stage_lambda_counts
from felis.utils import get_visible_cuda_devices
from felis.configs import dump_config

logger = logging.getLogger(__name__)

# Optimized water-leg lambda schedule produced by sysA_opt_lambdas_v2 and
# consumed by sysA_v2 when present in the prepare directory.
SYSA_OPT_LAM_JSON = "sysA_opt_lam.json"


def _get_recipe_sysA_lambdas(ligand_charge: int) -> list[list[float]]:
    if abs(ligand_charge) > 0:
        return get_charged_water_leg_lambdas()
    if abs(abs(ligand_charge) - 0) < 1e-3:
        return get_neutral_water_leg_lambdas()
    raise ValueError(f"Unsupported ligand charge: {ligand_charge}")


def _select_sysA_lambdas(acfg: ABFEInputConfig, ligand_charge: int, prepare_dir: str) -> list[list[float]]:
    opt_lam_path = os.path.join(prepare_dir, SYSA_OPT_LAM_JSON)
    return select_lambda_schedule(
        configured_path=acfg.wat_lam_json,
        optimized_path=opt_lam_path,
        recipe_lambdas_fn=lambda: _get_recipe_sysA_lambdas(ligand_charge),
        schedule_name="water-leg",
    )


@ABStageExecutionRegister(ABStage.sysA_em_v2)
class ABStageExecution_sysA_em(ABStageExecutionBase):

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

        em_version = acfg.md_sol_em_version
        cmd_sysA_em = [
            "python3",
            "-m",
            "felis.app.dyn.vanilla",
            "--cfg",
            "prepare/sysA_posres.json",
            "--tkv",
            "s:filename.stem:sysA_em",
            "s:filename.sys:prepare/sysA.top",
            "s:filename.crd:prepare/sysA.gro",
            "s:dir.trj:trj",
            f"i:integrator.minimize:{em_version}",
            "i:integrator.nsnapshots:20",
        ]
        if job_context.run_one_call(cmd_sysA_em) != 0:
            raise RuntimeError("sysA energy minimization failed")
        shutil.copy(os.path.join(job_context.trj_dir, "sysA_em.pdb"),
                    os.path.join(job_context.prepare_dir, "sysA_em.pdb"))

        job_context.write_stage_done(stage)


@ABStageExecutionRegister(ABStage.sysA_v2)
class ABStageExecution_sysA(ABStageExecutionBase):

    def __init__(self, acfg: ABFEInputConfig, sdf_abs: str, n_cuda_devices: int, **kwargs):
        super().__init__(**kwargs)
        self._acfg = acfg
        self._sdf_abs = sdf_abs
        self._n_cuda_devices = n_cuda_devices

        if Path(self._sdf_abs).exists():
            mol = Chem.SDMolSupplier(self._sdf_abs, removeHs=False)
            mol = mol[0]
            self.ligand_charge = Chem.GetFormalCharge(mol)
        else:
            raise ValueError(f"SDF file {self._sdf_abs} does not exist.")


    def exec(self):
        stage = self.stage
        planned_stages = self._planned_stages
        job_context = self._job_context
        acfg = self._acfg
        sdf_abs = self._sdf_abs
        n_cuda_devices = self._n_cuda_devices

        if stage not in planned_stages:
            return

        gpu_ids = get_visible_cuda_devices()
        assert len(gpu_ids) == n_cuda_devices

        sysA_lam = _select_sysA_lambdas(acfg, self.ligand_charge, job_context.prepare_dir)

        sysA_lam_path = os.path.join(job_context.prepare_dir, "sysA_lam.json")
        with open(sysA_lam_path, "w") as f:
            dump_config({"ab": {"lam_list": sysA_lam}}, f, indent=4)
        plot_lambda_schedule(
            figure_path=os.path.join(job_context.prepare_dir, "sysA_lam.png"),
            lam_json_path=sysA_lam_path,
            title="sysA lambda schedule",
        )

        num_lambdas = len(sysA_lam)
        cmd = [
            "python3",
            "-m",
            "felis.app.dyn.repex_v2",
            "--cfg",
            "prepare/sysA_lam.json",
            "prepare/sysA_ab_ligatoms.json",
            "--tkv",
            f"s:filename.stem:a0",
            f"s:filename.monomer:{sdf_abs}",
            "s:filename.sys:prepare/sysA.top",
            "s:filename.crd:prepare/sysA_em.pdb",
            "s:filename.atom_ids:prepare/sysA_atom_ids.json",
            "s:dir.trj:trj",
            f"i:openmm.checkpoint_interval:{acfg.md_checkpoint_interval}",
            f"i:openmm.exchange_interval:{acfg.md_exchange_interval}",
            f"s:openmm.tmp_directory:{acfg.omm_tmp_directory}",
            "i:integrator.npt:1",
            acfg.dt_tkv(),
            acfg.hmr_tkv(),
            f"i:integrator.nstep_per_snapshot:{acfg.md_traj_interval}",
            f"i:integrator.nsnapshots:{acfg.md_sol_nsnapshots}",
            "--rextkv",
        ]
        cmd.extend([f"i:ab.ilam:{vv}" for vv in range(num_lambdas)])
        mps_cmd = job_context.build_cuda_mps_command(cmd, gpu_id=gpu_ids)

        if job_context.run_one_call(mps_cmd) != 0:
            raise RuntimeError("sysA failed")

        job_context.write_stage_done(stage, {"n_cuda_devices": n_cuda_devices})

@ABStageExecutionRegister(ABStage.sysA_opt_lambdas_v2)
class ABStageExecution_sysA_opt_lambdas(ABStageExecutionBase):

    def __init__(
        self,
        acfg: ABFEInputConfig,
        sdf_abs: str,
        n_cuda_devices: int,
        opt_dir_name: str = "opt_lambdas_sysA",
        nsnapshots: int = 200,
        traj_interval: int = 5000,
        exchange_interval: int = 1000,
        checkpoint_interval: int = 20000,
        max_iter: int = 5,
        eta: float = 0.5,
        charge_k: float = 8.0,
        charge_x0: float = 0.5,
        vdw_k: float = 8.0,
        vdw_x0: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._acfg = acfg
        self._sdf_abs = sdf_abs
        self._n_cuda_devices = n_cuda_devices
        self._opt_dir_name = opt_dir_name
        self._nsnapshots = nsnapshots
        self._traj_interval = traj_interval
        self._exchange_interval = exchange_interval
        self._checkpoint_interval = checkpoint_interval
        self._max_iter = max_iter
        self._eta = eta
        self._charge_k = charge_k
        self._charge_x0 = charge_x0
        self._vdw_k = vdw_k
        self._vdw_x0 = vdw_x0

        if not Path(self._sdf_abs).exists():
            raise ValueError(f"SDF file {self._sdf_abs} does not exist.")
        mol = Chem.SDMolSupplier(self._sdf_abs, removeHs=False)[0]
        self.ligand_charge = Chem.GetFormalCharge(mol)

    def _make_run_hremd_fn(self, opt_root: Path, gpu_ids: list):
        job_context = self._job_context
        acfg = self._acfg
        sdf_abs = self._sdf_abs
        prepare_dir = Path(job_context.prepare_dir)
        counter = {"k": 0}

        def run_hremd_fn(schedule):
            k = counter["k"]
            iter_dir = opt_root / f"iter_{k:02d}"
            iter_prepare = iter_dir / "prepare"
            iter_trj = iter_dir / "trj"
            iter_prepare.mkdir(parents=True, exist_ok=True)
            iter_trj.mkdir(parents=True, exist_ok=True)

            for fname in [
                "sysA.top", "sysA_em.pdb",
                "sysA_ab_ligatoms.json", "sysA_atom_ids.json",
            ]:
                src = prepare_dir / fname
                dst = iter_prepare / fname
                if not dst.exists():
                    shutil.copy(src, dst)

            lam_list = [list(map(float, row)) for row in schedule]
            with open(iter_prepare / "sysA_lam.json", "w") as f:
                dump_config({"ab": {"lam_list": lam_list}}, f, indent=4)
            np.save(iter_dir / "schedule.npy", schedule)

            num_lambdas = len(lam_list)
            cmd = [
                "python3", "-m", "felis.app.dyn.repex_v2",
                "--cfg",
                "prepare/sysA_lam.json",
                "prepare/sysA_ab_ligatoms.json",
                "--tkv",
                "s:filename.stem:a0",
                f"s:filename.monomer:{sdf_abs}",
                "s:filename.sys:prepare/sysA.top",
                "s:filename.crd:prepare/sysA_em.pdb",
                "s:filename.atom_ids:prepare/sysA_atom_ids.json",
                "s:dir.trj:trj",
                f"i:openmm.checkpoint_interval:{self._checkpoint_interval}",
                f"i:openmm.exchange_interval:{self._exchange_interval}",
                f"s:openmm.tmp_directory:{acfg.omm_tmp_directory}",
                "i:integrator.npt:1",
                acfg.dt_tkv(),
                acfg.hmr_tkv(),
                f"i:integrator.nstep_per_snapshot:{self._traj_interval}",
                f"i:integrator.nsnapshots:{self._nsnapshots}",
                "--rextkv",
            ]
            cmd.extend([f"i:ab.ilam:{vv}" for vv in range(num_lambdas)])
            mps_cmd = job_context.build_cuda_mps_command(cmd, gpu_id=gpu_ids)

            logger.info(f"iter={k} launching short HREMD: {iter_dir}")
            if job_context.run_one_call(mps_cmd, working_dir=str(iter_dir)) != 0:
                raise RuntimeError(f"sysA opt lambdas HREMD failed at iter={k}")

            energy_matrix_path = iter_trj / "a0_hremd" / "records.sqlite"
            records = read_records(energy_matrix_path)
            if not records:
                raise RuntimeError(f"No records produced at {energy_matrix_path}")

            try:
                energy_matrix, beta = get_energy_matrix_from_records(records)
                _, _, overlap_matrix = analyze_results(
                    energy_matrix, beta, with_overlap_matrix=True
                )
                if overlap_matrix is not None:
                    plot_overlap_matrix(
                        figure_path=str(iter_dir / "overlap_matrix.png"),
                        overlap_matrix=overlap_matrix,
                    )
                    np.save(iter_dir / "overlap_matrix.npy", overlap_matrix)
            except Exception as exc:
                logger.warning(f"iter={k} failed to draw overlap matrix: {exc}")

            counter["k"] += 1
            return records

        return run_hremd_fn

    def exec(self):
        stage = self.stage
        planned_stages = self._planned_stages
        job_context = self._job_context
        n_cuda_devices = self._n_cuda_devices

        if stage not in planned_stages:
            return
        if self._acfg.wat_lam_json is not None:
            logger.info("Configured water-leg lambda schedule is set; skip automatic optimization")
            job_context.write_stage_done(stage)
            return

        opt_lam_path = os.path.join(job_context.prepare_dir, SYSA_OPT_LAM_JSON)
        if os.path.isfile(opt_lam_path):
            logger.info(f"Optimized lambda schedule already exists: {opt_lam_path}; skip optimization")
            job_context.write_stage_done(stage)
            return

        gpu_ids = get_visible_cuda_devices()
        assert len(gpu_ids) == n_cuda_devices

        opt_root = Path(job_context.working_dir) / self._opt_dir_name
        opt_root.mkdir(parents=True, exist_ok=True)

        initial_schedule = np.array(
            _get_recipe_sysA_lambdas(self.ligand_charge), dtype=np.float64
        )
        t_split = infer_t_split_from_schedule(initial_schedule)
        logger.info(f"Optimizing sysA lambda schedule: N={initial_schedule.shape[0]}, t_split={t_split:.4f}")

        run_hremd_fn = self._make_run_hremd_fn(opt_root, gpu_ids)

        result = optimize_lambda_schedule(
            initial_schedule=initial_schedule,
            run_hremd_fn=run_hremd_fn,
            t_split=t_split,
            charge_shape=SigmoidShape(k=self._charge_k, x0=self._charge_x0),
            vdw_shape=SigmoidShape(k=self._vdw_k, x0=self._vdw_x0),
            max_iter=self._max_iter,
            eta=self._eta,
            schedule_transform=lambda schedule: freeze_stage_lambda_counts(
                schedule, initial_schedule
            ),
        )

        logger.info(f"Lambda optimization converged={result.converged} n_iterations={result.n_iterations}")

        np.save(opt_root / "final_schedule.npy", result.schedule)
        final_lam_list = [list(map(float, row)) for row in result.schedule]
        with open(opt_lam_path, "w") as f:
            dump_config({"ab": {"lam_list": final_lam_list}}, f, indent=4)
        plot_lambda_schedule(
            figure_path=os.path.join(job_context.prepare_dir, "sysA_opt_lam.png"),
            lam_json_path=opt_lam_path,
            title="sysA optimized lambda schedule",
        )
        logger.info(f"Saved optimized lambda schedule to {opt_lam_path}")

        job_context.write_stage_done(stage, {"converged": bool(result.converged),
                                             "n_iterations": int(result.n_iterations)})
