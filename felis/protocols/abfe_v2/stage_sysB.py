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
    get_charged_protein_leg_lambdas,
    get_neutral_protein_leg_lambdas,
)
from felis.protocols.abfe_v2.lambda_config import read_lambda_schedule
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
from felis.protocols.abfe_v2.protein_lambda_rules import (
    freeze_stage_lambda_counts,
    require_full_restraint_endpoint,
    set_restraint_lambda_count,
    validate_protein_lambda_schedule,
)
from felis.protocols.boresch.main_boresch_restraints import BORESCH_CFG_JSON
from felis.utils import get_visible_cuda_devices
from felis.configs import dump_config

logger = logging.getLogger(__name__)

# Optimized protein-leg lambda schedule produced by sysB_opt_lambdas_v2 and
# consumed by sysB_v2 when present in the prepare directory.
SYSB_OPT_LAM_JSON = "sysB_opt_lam.json"
DEFAULT_N_RESTRAINT_LAMBDAS = 6


def _get_recipe_sysB_lambdas(ligand_charge: int) -> list[list[float]]:
    if abs(ligand_charge) > 0:
        return get_charged_protein_leg_lambdas()
    if abs(abs(ligand_charge) - 0) < 1e-3:
        return get_neutral_protein_leg_lambdas()
    raise ValueError(f"Unsupported ligand charge: {ligand_charge}")


def _select_sysB_lambdas(acfg: ABFEInputConfig, ligand_charge: int, prepare_dir: str) -> list[list[float]]:
    opt_lam_path = os.path.join(prepare_dir, SYSB_OPT_LAM_JSON)
    return select_lambda_schedule(
        configured_path=acfg.pro_lam_json,
        optimized_path=opt_lam_path,
        recipe_lambdas_fn=lambda: _get_recipe_sysB_lambdas(ligand_charge),
        schedule_name="protein-leg",
    )


@ABStageExecutionRegister(ABStage.sysB_em_v2)
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


@ABStageExecutionRegister(ABStage.sysB_v2)
class ABStageExecution_sysB(ABStageExecutionBase):

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

        sysB_lam = _select_sysB_lambdas(acfg, self.ligand_charge, job_context.prepare_dir)
        validate_protein_lambda_schedule(sysB_lam, source="selected protein-leg lambda schedule")

        sysB_lam_path = os.path.join(job_context.prepare_dir, "sysB_lam.json")
        with open(sysB_lam_path, "w") as f:
            dump_config({"ab": {"lam_list": sysB_lam}}, f, indent=4)
        plot_lambda_schedule(
            figure_path=os.path.join(job_context.prepare_dir, "sysB_lam.png"),
            lam_json_path=sysB_lam_path,
            title="sysB lambda schedule",
        )

        num_lambdas = len(sysB_lam)
        cmd = [
            "python3",
            "-m",
            "felis.app.dyn.repex_v2",
            "--cfg",
            "prepare/sysB_lam.json",
            "prepare/sysB_ab_ligatoms.json",
            f"prepare/{BORESCH_CFG_JSON}",
            "--tkv",
            f"s:filename.stem:b0",
            f"s:filename.monomer:{sdf_abs}",
            "s:filename.sys:prepare/sysB.top",
            "s:filename.crd:prepare/sysB_em.pdb",
            "s:filename.atom_ids:prepare/sysB_atom_ids.json",
            "s:dir.trj:trj",
            f"i:openmm.checkpoint_interval:{acfg.md_checkpoint_interval}",
            f"i:openmm.exchange_interval:{acfg.md_exchange_interval}",
            f"s:openmm.tmp_directory:{acfg.omm_tmp_directory}",
            "i:integrator.npt:1",
            acfg.dt_tkv(),
            acfg.hmr_tkv(),
            f"i:integrator.nstep_per_snapshot:{acfg.md_traj_interval}",
            f"i:integrator.nsnapshots:{acfg.md_pro_nsnapshots}",
        ]
        # RelocationMC water enhanced sampling is a sysB-only feature.
        cmd.extend(acfg.relocation_mc_tkv())
        cmd.extend([
            "--rextkv",
        ])
        cmd.extend([f"i:ab.ilam:{vv}" for vv in range(num_lambdas)])
        mps_cmd = job_context.build_cuda_mps_command(cmd, gpu_id=gpu_ids)

        assert 0 == job_context.run_one_call(mps_cmd), "sysB failed"

        job_context.write_stage_done(stage, {"n_cuda_devices": n_cuda_devices})


@ABStageExecutionRegister(ABStage.sysB_opt_lambdas_v2)
class ABStageExecution_sysB_opt_lambdas(ABStageExecutionBase):

    def __init__(
        self,
        acfg: ABFEInputConfig,
        sdf_abs: str,
        n_cuda_devices: int,
        opt_dir_name: str = "opt_lambdas_sysB",
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
        restraint_lambda_end: float = 1.0,
        use_path_from_initial_schedule: bool = True,
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
        require_full_restraint_endpoint(restraint_lambda_end)
        self._use_path_from_initial_schedule = use_path_from_initial_schedule

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
            validate_protein_lambda_schedule(schedule, source="protein-leg optimizer candidate schedule")
            k = counter["k"]
            iter_dir = opt_root / f"iter_{k:02d}"
            iter_prepare = iter_dir / "prepare"
            iter_trj = iter_dir / "trj"
            iter_prepare.mkdir(parents=True, exist_ok=True)
            iter_trj.mkdir(parents=True, exist_ok=True)

            for fname in [
                "sysB.top", "sysB_em.pdb",
                "sysB_ab_ligatoms.json", "sysB_atom_ids.json",
                BORESCH_CFG_JSON,
            ]:
                src = prepare_dir / fname
                dst = iter_prepare / fname
                if not dst.exists():
                    shutil.copy(src, dst)

            lam_list = [list(map(float, row)) for row in schedule]
            with open(iter_prepare / "sysB_lam.json", "w") as f:
                dump_config({"ab": {"lam_list": lam_list}}, f, indent=4)

            np.save(iter_dir / "schedule.npy", schedule)

            num_lambdas = len(lam_list)
            cmd = [
                "python3", "-m", "felis.app.dyn.repex_v2",
                "--cfg",
                "prepare/sysB_lam.json",
                "prepare/sysB_ab_ligatoms.json",
                f"prepare/{BORESCH_CFG_JSON}",
                "--tkv",
                "s:filename.stem:b0",
                f"s:filename.monomer:{sdf_abs}",
                "s:filename.sys:prepare/sysB.top",
                "s:filename.crd:prepare/sysB_em.pdb",
                "s:filename.atom_ids:prepare/sysB_atom_ids.json",
                "s:dir.trj:trj",
                f"i:openmm.checkpoint_interval:{self._checkpoint_interval}",
                f"i:openmm.exchange_interval:{self._exchange_interval}",
                f"s:openmm.tmp_directory:{acfg.omm_tmp_directory}",
                "i:integrator.npt:1",
                acfg.dt_tkv(),
                acfg.hmr_tkv(),
                f"i:integrator.nstep_per_snapshot:{self._traj_interval}",
                f"i:integrator.nsnapshots:{self._nsnapshots}",
            ]
            # RelocationMC water enhanced sampling is a sysB-only feature.
            cmd.extend(acfg.relocation_mc_tkv())
            cmd.extend([
                "--rextkv",
            ])
            cmd.extend([f"i:ab.ilam:{vv}" for vv in range(num_lambdas)])
            mps_cmd = job_context.build_cuda_mps_command(cmd, gpu_id=gpu_ids)

            logger.info(f"iter={k} launching short HREMD: {iter_dir}")
            assert 0 == job_context.run_one_call(mps_cmd, working_dir=str(iter_dir)), \
                f"sysB opt lambdas HREMD failed at iter={k}"

            energy_matrix_path = iter_trj / "b0_hremd" / "records.sqlite"
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
        if self._acfg.pro_lam_json is not None:
            configured_lam = read_lambda_schedule(self._acfg.pro_lam_json)
            validate_protein_lambda_schedule(
                configured_lam,
                source=f"configured protein-leg lambda schedule {self._acfg.pro_lam_json}",
            )
            logger.info("Configured protein-leg lambda schedule is set; skip automatic optimization")
            job_context.write_stage_done(stage)
            return

        opt_lam_path = os.path.join(job_context.prepare_dir, SYSB_OPT_LAM_JSON)
        if os.path.isfile(opt_lam_path):
            optimized_lam = read_lambda_schedule(opt_lam_path)
            validate_protein_lambda_schedule(
                optimized_lam,
                source=f"optimized protein-leg lambda schedule {opt_lam_path}",
            )
            logger.info(f"Optimized lambda schedule already exists: {opt_lam_path}; skip optimization")
            job_context.write_stage_done(stage)
            return

        gpu_ids = get_visible_cuda_devices()
        assert len(gpu_ids) == n_cuda_devices

        opt_root = Path(job_context.working_dir) / self._opt_dir_name
        opt_root.mkdir(parents=True, exist_ok=True)

        initial_schedule = np.array(
            _get_recipe_sysB_lambdas(self.ligand_charge), dtype=np.float64
        )
        initial_schedule = set_restraint_lambda_count(
            initial_schedule, DEFAULT_N_RESTRAINT_LAMBDAS
        )

        t_split = infer_t_split_from_schedule(initial_schedule)
        logger.info(
            f"Optimizing sysB lambda schedule: N={initial_schedule.shape[0]}, "
            f"t_split={t_split:.4f}, path_preserving={self._use_path_from_initial_schedule}"
        )

        run_hremd_fn = self._make_run_hremd_fn(opt_root, gpu_ids)

        result = optimize_lambda_schedule(
            initial_schedule=initial_schedule,
            run_hremd_fn=run_hremd_fn,
            t_split=t_split,
            charge_shape=SigmoidShape(k=self._charge_k, x0=self._charge_x0),
            vdw_shape=SigmoidShape(k=self._vdw_k, x0=self._vdw_x0),
            use_path_from_initial_schedule=self._use_path_from_initial_schedule,
            max_iter=self._max_iter,
            eta=self._eta,
            schedule_transform=lambda schedule: freeze_stage_lambda_counts(
                schedule, initial_schedule
            ),
        )

        logger.info(f"Lambda optimization converged={result.converged} n_iterations={result.n_iterations}")
        validate_protein_lambda_schedule(result.schedule, source="optimized protein-leg lambda schedule")

        np.save(opt_root / "final_schedule.npy", result.schedule)
        final_lam_list = [list(map(float, row)) for row in result.schedule]
        with open(opt_lam_path, "w") as f:
            dump_config({"ab": {"lam_list": final_lam_list}}, f, indent=4)
        plot_lambda_schedule(
            figure_path=os.path.join(job_context.prepare_dir, "sysB_opt_lam.png"),
            lam_json_path=opt_lam_path,
            title="sysB optimized lambda schedule",
        )
        logger.info(f"Saved optimized lambda schedule to {opt_lam_path}")

        job_context.write_stage_done(stage, {"converged": bool(result.converged),
                                             "n_iterations": int(result.n_iterations)})
