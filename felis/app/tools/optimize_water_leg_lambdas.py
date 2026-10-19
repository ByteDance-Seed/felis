# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
CLI script to optimize the water-leg (sysA) lambda schedule of an ABFE v2
calculation using the Trebst-Katzgraber replica-flow profile. Exchange
acceptance ratios impose a hard adjacent-pair floor, and BAR overlap matrices
are saved as diagnostics.

This script uses **path-preserving mode**: the full thermodynamic path from
the initial schedule (the recipe by default, or ``--initial-lam-json`` if
given) is preserved and only replica positions along the path are optimized.
The path is resampled to exactly 24 windows for neutral ligands and 32 windows
for charged ligands before optimization.
The charge B coordinate is re-derived as ``lambdaB_charge = 1 - lambdaA_charge``
while ``lambdaB_vdw`` is kept free (not constrained to ``1 - lambdaA_vdw``).

Usage::

    python optimize_water_leg_lambdas.py \\
        --abfecfg abfe_v2_ejm_43.yaml \\
        --job-name ejm_43 \\
        [--opt-dir-name opt_lambdas_sysA] \\
        [--nsnapshots 200] [--traj-interval 5000] \\
        [--exchange-interval 1000] [--checkpoint-interval 20000] \\
        [--max-iter 6] [--eta 0.5] \\
        [--min-acceptance-ratio 0.2] \\
        [--charge-k 8.0] [--charge-x0 0.5] \\
        [--vdw-k 8.0] [--vdw-x0 0.5] \\
        [--t-split 0.5] [--no-plots]

Prerequisites: ``makebox`` and ``sysA_em_v2`` must already have been run for
the target ligand, so that ``prepare/sysA.top``, ``prepare/sysA_em.pdb``,
``prepare/sysA_atom_ids.json``, ``prepare/sysA_ab_ligatoms.json`` exist.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import rdkit.Chem as Chem

from felis.configs import dump_config, load_config
from felis.protocols.abfe.config_types import ABFEInputConfig
from felis.protocols.abfe.job_context import JobContext
from felis.protocols.abfe_v2.recipes import (
    get_neutral_water_leg_lambdas,
    get_charged_water_leg_lambdas,
)
from felis.protocols.abfe_v2.utilities import (
    analyze_results,
    get_acceptance_ratios_from_records,
    get_energy_matrix_from_records,
    get_flow_drops_from_records,
    get_round_trip_rate_from_records,
    read_records,
)
from felis.protocols.abfe_v2.lambda_schedule_optimizer import (
    DEFAULT_MIN_ACCEPTANCE_RATIO,
    SigmoidShape,
    infer_t_split_from_schedule,
    optimize_lambda_schedule,
)
from felis.protocols.abfe_v2.lambda_config import (
    get_water_leg_window_count,
    resample_water_leg_schedule,
)
from felis.protocols.abfe_v2.protein_lambda_rules import freeze_stage_lambda_counts
from felis.utils import get_visible_cuda_devices
from felis.utils import resolve_cuda_device_ids

logger = logging.getLogger("opt_water_leg")


def _load_overlap_plotter():
    os.environ.setdefault("MPLBACKEND", "Agg")
    from felis.protocols.abfe_v2.draw_figures import plot_overlap_matrix

    return plot_overlap_matrix


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Optimize the water-leg lambda schedule of an ABFE v2 job by "
            "equalizing adjacent-pair exchange acceptance ratios."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--abfecfg", required=True, type=Path,
                   help="Path to the ABFE input config YAML.")
    p.add_argument("--job-name", required=True, type=str,
                   help="Job / ligand name. Output is at <outdir>/<job_name>/.")
    p.add_argument("--opt-dir-name", default="opt_lambdas_sysA", type=str,
                   help="Subdirectory under the job dir for optimization "
                        "artifacts.")
    p.add_argument("--initial-lam-json", default=None, type=Path,
                   help="Path to a lam.json file to use as the initial "
                        "schedule (with structure {\"ab\": {\"lam_list\": "
                        "[[...7 floats...], ...]}}). The full thermodynamic "
                        "path is preserved and resampled to 24 windows for a "
                        "neutral ligand or 32 for a charged ligand (qB is "
                        "re-derived as 1 - qA; vdwB stays free). If not "
                        "provided, the default "
                        "get_neutral_water_leg_lambdas() or get_charged_water_leg_lambdas() "
                        "is used.")

    # Short-HREMD knobs.
    p.add_argument("--nsnapshots", default=100, type=int,
                   help="Number of snapshots per iteration (short HREMD).")
    p.add_argument("--traj-interval", default=25000, type=int,
                   help="Steps per snapshot.")
    p.add_argument("--exchange-interval", default=250, type=int,
                   help="Replica-exchange attempt interval (steps).")
    p.add_argument("--checkpoint-interval", default=25000, type=int,
                   help="Checkpoint interval (steps).")

    # Optimizer knobs.
    p.add_argument("--flow-linearity-norm", default="l2",
                   choices=["l2", "linf"],
                   help="Norm used to measure how far the flow profile is "
                        "from the ideal linear "
                        "profile, which is the flow convergence metric. 'l2' "
                        "(default) is the RMS deviation (robust); 'linf' is "
                        "the max deviation (stricter).")
    p.add_argument("--flow-smooth", dest="flow_smooth",
                   action="store_true", default=True,
                   help="Drive replica placement from a monotone-smoothed "
                        "(isotonic) flow profile to denoise "
                        "df on short runs (default; mirrors the df/dT "
                        "regression in cond-mat/0602085).")
    p.add_argument("--no-flow-smooth", dest="flow_smooth",
                   action="store_false",
                   help="Use the raw (unsmoothed) flow profile for placement.")
    p.add_argument("--flow-drop-floor-frac", default=1.0, type=float,
                   help="Floor each per-pair flow drop at this fraction of "
                        "the ideal uniform drop 1/(N-1) "
                        "before placement. Prevents near-linear regions from "
                        "being starved of replicas (which collapses their "
                        "overlap); bounds the spacing ratio to about "
                        "sqrt(drop_max / (frac/(N-1))). Smaller = stronger "
                        "bottleneck crowding.")
    p.add_argument(
        "--min-acceptance-ratio",
        default=DEFAULT_MIN_ACCEPTANCE_RATIO,
        type=float,
        help=(
            "Hard lower bound for every adjacent-pair exchange acceptance "
            "ratio. Every measured pair must be strictly greater than this "
            "value. Values below the protocol floor of 0.2 are rejected."
        ),
    )
    p.add_argument(
        "--optimization-target",
        default="acceptance",
        choices=["acceptance", "flow"],
        help=(
            "Optimization objective. 'acceptance' (default) equalizes "
            "adjacent-pair exchange acceptance ratios; 'flow' retains the "
            "legacy Trebst-Katzgraber flow objective."
        ),
    )
    p.add_argument(
        "--acceptance-plateau-tol",
        default=0.01,
        type=float,
        help=(
            "For the acceptance target, stop after --patience consecutive "
            "feasible iterations whose acceptance-standard-deviation "
            "improvement is below this absolute tolerance."
        ),
    )
    p.add_argument("--rtr-plateau-tol", default=0.05, type=float,
                   help="Stop when the relative round-trip-rate improvement "
                        "stays below this value for --patience consecutive "
                        "iterations.")
    p.add_argument("--patience", default=2, type=int,
                   help="Consecutive low-improvement iterations required to "
                        "declare a round-trip-rate plateau.")
    p.add_argument("--optimize-path-shape", dest="optimize_path_shape",
                   action="store_true", default=True,
                   help="With --optimization-target flow, jointly optimize the path "
                        "shape (t_split and lambdaB_vdw "
                        "curvature gamma) alongside s (default).")
    p.add_argument("--no-optimize-path-shape", dest="optimize_path_shape",
                   action="store_false",
                   help="Disable joint path-shape optimization; only replica "
                        "positions (s) are optimized.")
    p.add_argument("--t-split-step", default=0.05, type=float,
                   help="Greedy nudge size (s-space) applied to t_split per "
                        "iteration during joint path-shape optimization.")
    p.add_argument("--vdwb-gamma-step", default=0.15, type=float,
                   help="Multiplicative (log-space) nudge size applied to the "
                        "lambdaB_vdw curvature gamma per iteration during "
                        "joint path-shape optimization.")
    p.add_argument("--max-iter", default=5, type=int,
                   help="Maximum number of outer-loop iterations.")
    p.add_argument("--eta", default=0.5, type=float,
                   help="Damping factor in (0, 1].")
    p.add_argument("--t-split", default=None, type=float,
                   help="Phase-split point in s-space; inferred from initial "
                        "schedule if not given.")
    p.add_argument("--charge-k", default=8.0, type=float,
                   help="Sigmoid steepness for the charge phase.")
    p.add_argument("--charge-x0", default=0.5, type=float,
                   help="Sigmoid inflection for the charge phase.")
    p.add_argument("--vdw-k", default=8.0, type=float,
                   help="Sigmoid steepness for the vdW phase.")
    p.add_argument("--vdw-x0", default=0.5, type=float,
                   help="Sigmoid inflection for the vdW phase.")

    # Output controls.
    p.add_argument("--no-plots", action="store_true",
                   help="Skip generating diagnostic plots.")
    p.add_argument("--no-mps", action="store_true",
                   help="Disable wrapping the repex_v2 command via "
                        "JobContext.build_cuda_mps_command (CUDA MPS).")
    p.add_argument("--gpu-ids", default=None, type=str,
                   help="Comma-separated GPU ids to use, e.g. '0,1'. "
                        "Defaults to all visible CUDA devices.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging level.")

    return p.parse_args()


def load_initial_schedule(lam_json_path: Path) -> np.ndarray:
    """Load an ``(N, 7)`` schedule from a lam.json file.

    The file must follow the same structure that ``stage_sysA_v2`` writes
    to ``prepare/sysA_lam.json``::

        {"ab": {"lam_list": [[lambdaA_charge, lambdaB_charge,
                              lambdaC_charge, lambdaA_vdw, lambdaB_vdw,
                              torsionLambda, restraintLambda], ...]}}
    """
    with open(lam_json_path) as f:
        cfg = load_config(f)
    try:
        lam_list = cfg["ab"]["lam_list"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{lam_json_path} is missing 'ab.lam_list'; expected structure "
            "{\"ab\": {\"lam_list\": [[...7 floats...], ...]}}"
        ) from exc

    arr = np.asarray(lam_list, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(
            f"{lam_json_path}: lam_list must be (N, 7); got shape {arr.shape}"
        )
    return arr


def make_run_hremd_fn(
    opt_root: Path,
    prepare_dir: Path,
    sdf_abs: Path,
    acfg: ABFEInputConfig,
    nsnapshots: int,
    traj_interval: int,
    exchange_interval: int,
    checkpoint_interval: int,
    gpu_ids: list,
    use_mps: bool = True,
    no_plots: bool = False,
):
    """Build a ``run_hremd_fn`` closure for the optimizer.

    Each invocation creates ``opt_root/iter_NN/`` and shells out to
    ``python3 -m felis.app.dyn.repex_v2`` mirroring the production
    ``stage_sysA_v2`` command. When ``use_mps`` is True, the command is
    wrapped via :meth:`JobContext.build_cuda_mps_command` so multiple
    replicas can share a single GPU through CUDA MPS.

    Unless ``no_plots`` is set, each iteration's overlap matrix is computed
    from ``records.sqlite`` and saved as ``iter_NN/overlap_matrix.png``.
    """
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
        lam_json_path = iter_prepare / "sysA_lam.json"
        with open(lam_json_path, "w") as f:
            dump_config({"ab": {"lam_list": lam_list}}, f, indent=4)
        np.save(iter_dir / "schedule.npy", schedule)

        num_lambdas = len(lam_list)
        common_cmd = [
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
            f"i:openmm.checkpoint_interval:{checkpoint_interval}",
            f"i:openmm.exchange_interval:{exchange_interval}",
            f"s:openmm.tmp_directory:{acfg.omm_tmp_directory}",
            "i:integrator.npt:1",
            acfg.dt_tkv(),
            acfg.hmr_tkv(),
            f"i:integrator.nstep_per_snapshot:{traj_interval}",
            f"i:integrator.nsnapshots:{nsnapshots}",
            "--rextkv",
        ]
        common_cmd.extend([f"i:ab.ilam:{vv}" for vv in range(num_lambdas)])

        subprocess_env = None
        if use_mps:
            cmd = JobContext.build_cuda_mps_command(common_cmd, gpu_id=gpu_ids)
        else:
            cmd = common_cmd
            subprocess_env = os.environ.copy()
            subprocess_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        log_path = iter_dir / "repex_v2.log"
        logger.info(
            "iter=%d  launching short HREMD (mps=%s, gpus=%s): %s",
            k, use_mps, gpu_ids, iter_dir,
        )
        with open(log_path, "w") as logf:
            ret = subprocess.run(
                cmd,
                cwd=str(iter_dir),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=subprocess_env,
            )
        if ret.returncode != 0:
            raise RuntimeError(
                f"repex_v2 failed at iter={k}; see {log_path}"
            )

        energy_matrix_path = iter_trj / "a0_hremd" / "records.sqlite"
        records = read_records(energy_matrix_path)
        if not records:
            raise RuntimeError(
                f"No records produced at {energy_matrix_path}"
            )

        # Per-iteration hard-constraint artifact.
        try:
            acceptance_ratios = get_acceptance_ratios_from_records(records)
            np.save(iter_dir / "acceptance_ratios.npy", acceptance_ratios)
            logger.info(
                "iter=%d  acceptance ratios: min=%.4f max=%.4f mean=%.4f",
                k,
                float(acceptance_ratios.min()),
                float(acceptance_ratios.max()),
                float(acceptance_ratios.mean()),
            )
        except Exception as exc:
            logger.warning(
                "iter=%d  failed to compute acceptance ratios: %s", k, exc
            )

        # Per-iteration flow-objective artifact.
        try:
            flow_drops = get_flow_drops_from_records(records)
            np.save(iter_dir / "flow_drops.npy", flow_drops)
            round_trip_rate = get_round_trip_rate_from_records(records)
            logger.info(
                "iter=%d  flow drops: max=%.4f mean=%.4f  round_trip_rate=%.4e",
                k,
                float(flow_drops.max()),
                float(flow_drops.mean()),
                round_trip_rate,
            )
        except Exception as exc:
            logger.warning(
                "iter=%d  failed to compute flow drops: %s", k, exc
            )

        # Per-iteration overlap matrix figure (diagnostic).
        if not no_plots:
            try:
                plot_overlap_matrix = _load_overlap_plotter()
                energy_matrix, beta = get_energy_matrix_from_records(records)
                _, _, overlap_matrix = analyze_results(
                    energy_matrix, beta, with_overlap_matrix=True
                )
                if overlap_matrix is not None:
                    overlap_png = iter_dir / "overlap_matrix.png"
                    plot_overlap_matrix(
                        figure_path=str(overlap_png),
                        overlap_matrix=overlap_matrix,
                    )
                    np.save(iter_dir / "overlap_matrix.npy", overlap_matrix)
                    logger.info("iter=%d  saved overlap_matrix.png: %s",
                                k, overlap_png)
            except Exception as exc:
                logger.warning(
                    "iter=%d  failed to draw overlap matrix: %s", k, exc
                )

        counter["k"] += 1
        return records

    return run_hremd_fn


def make_diagnostic_plots(
    initial_schedule: np.ndarray,
    result,
    opt_root: Path,
):
    """Save flow and schedule diagnostic plots under ``opt_root``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = [d.iteration for d in result.history]

    # --- Acceptance-target convergence with flow diagnostics ---
    # The acceptance profile drives placement and selection. Flow,
    # diffusivity, and round-trip rate remain diagnostic quantities.
    if result.history[-1].flow_linearity_error is not None:
        fle = [d.flow_linearity_error for d in result.history]
        rtr = [d.round_trip_rate for d in result.history]
        min_acceptance = [d.min_acceptance_ratio for d in result.history]
        acceptance_std = [d.acceptance_std for d in result.history]
        diff_mean = [float(d.diffusivity.mean()) for d in result.history]
        diff_min = [float(d.diffusivity.min()) for d in result.history]
        diff_max = [float(d.diffusivity.max()) for d in result.history]

        fig, axes = plt.subplots(4, 1, figsize=(6, 11), sharex=True)
        axes[0].plot(iters, min_acceptance, "o-", color="C1",
                     label="minimum pair acceptance")
        axes[0].axhline(result.acceptance_threshold, color="C1", ls="--",
                        label="strict acceptance floor")
        axes[0].set_ylabel("min acceptance")
        axes[0].plot(iters, acceptance_std, "x--", color="C4",
                     label="acceptance std")
        axes[0].set_title("Acceptance-profile optimization across iterations")
        axes[0].legend()
        axes[1].plot(iters, rtr, "s-", color="C0", label="round-trip rate")
        axes[1].set_ylabel("round-trip rate")
        axes[1].legend()
        axes[2].plot(iters, fle, "o-", color="C3",
                     label="flow linearity error")
        axes[2].set_ylabel("flow linearity error")
        axes[2].legend()
        axes[3].plot(iters, diff_mean, "^-", color="C2",
                     label="diffusivity mean")
        axes[3].fill_between(iters, diff_min, diff_max, color="C2", alpha=0.2,
                             label="diffusivity min-max")
        axes[3].set_xlabel("iteration")
        axes[3].set_ylabel("diffusivity (normalized)")
        axes[3].legend()
        fig.tight_layout()
        fig.savefig(opt_root / "flow_convergence_stats.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        for d in result.history:
            ax.plot(np.arange(len(d.flow_profile)), d.flow_profile,
                    marker="o", alpha=0.7, label=f"iter {d.iteration}")
        n_h = len(result.history[-1].flow_profile)
        ideal = 1.0 - np.arange(n_h, dtype=np.float64) / (n_h - 1)
        ax.plot(np.arange(n_h), ideal, "k--", label="ideal linear")
        ax.set_xlabel("Hamiltonian (replica) index")
        ax.set_ylabel("up-walker fraction f")
        ax.set_title("Trebst-Katzgraber flow profile across iterations")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(opt_root / "flow_profiles.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=True)
        for d in result.history:
            axes[0].plot(np.arange(len(d.acceptance_ratios)),
                         d.acceptance_ratios,
                         marker="o", label=f"iter {d.iteration}")
        axes[0].axhline(result.acceptance_threshold, color="C1", ls="--",
                        label="strict acceptance floor")
        axes[0].set_ylabel("acceptance ratio")
        axes[0].set_title("Per-pair acceptance and flow diagnostics")
        axes[0].legend(fontsize=8, loc="best")
        for d in result.history:
            axes[1].plot(np.arange(len(d.flow_drops)), d.flow_drops,
                         marker="o", label=f"iter {d.iteration}")
        n_pairs = len(result.history[-1].flow_drops)
        axes[1].axhline(1.0 / n_pairs, color="k", ls="--",
                        label="ideal uniform drop 1/(N-1)")
        axes[1].set_ylabel("per-pair flow drop")
        axes[1].legend(fontsize=8, loc="best")
        for d in result.history:
            axes[2].plot(np.arange(len(d.diffusivity)), d.diffusivity,
                         marker="o", label=f"iter {d.iteration}")
        axes[2].set_xlabel("neighbor index i (between replica i and i+1)")
        axes[2].set_ylabel("diffusivity (normalized)")
        axes[2].legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(
            opt_root / "acceptance_flow_drops_diffusivity.png", dpi=150
        )
        plt.close(fig)

        # Path-shape evolution: t_split and vdwb_gamma across iterations.
        if result.history[-1].t_split is not None:
            t_splits = [d.t_split for d in result.history]
            gammas = [d.vdwb_gamma for d in result.history]
            fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
            axes[0].plot(iters, t_splits, "o-", color="C4", label="t_split")
            axes[0].set_ylabel("t_split")
            axes[0].set_title("Path-shape evolution across iterations")
            axes[0].legend()
            axes[1].plot(iters, gammas, "s-", color="C5", label="vdwb_gamma")
            axes[1].axhline(1.0, color="k", ls="--", alpha=0.5,
                            label="gamma=1 (identity)")
            axes[1].set_xlabel("iteration")
            axes[1].set_ylabel("vdwb_gamma")
            axes[1].legend()
            for d in result.history:
                if d.accepted_nudge in ("t_split", "vdwb_gamma"):
                    axes[0].axvline(d.iteration, color="green", ls=":",
                                    alpha=0.3)
                elif d.accepted_nudge == "reverted":
                    axes[0].axvline(d.iteration, color="red", ls=":",
                                    alpha=0.3)
            fig.tight_layout()
            fig.savefig(opt_root / "path_shape_evolution.png", dpi=150)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(initial_schedule[:, 0], "o--", color="C0", alpha=0.4,
            label="A_charge (init)")
    ax.plot(initial_schedule[:, 3], "s--", color="C1", alpha=0.4,
            label="A_vdw (init)")
    ax.plot(result.schedule[:, 0], "o-", color="C0",
            label="A_charge (final)")
    ax.plot(result.schedule[:, 3], "s-", color="C1",
            label="A_vdw (final)")
    ax.set_xlabel("replica index")
    ax.set_ylabel("lambda")
    ax.set_title("Water-leg schedule: initial vs. optimized")
    ax.legend()
    fig.tight_layout()
    fig.savefig(opt_root / "schedule_init_vs_final.png", dpi=150)
    plt.close(fig)

    logger.info("diagnostic plots saved under %s", opt_root)


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    abfecfg_path = args.abfecfg.resolve()
    with open(abfecfg_path) as f:
        raw_cfg = load_config(f)
    acfg = ABFEInputConfig(**raw_cfg)

    abfecfg_dir = abfecfg_path.parent
    job_dir = (abfecfg_dir / acfg.outdir / args.job_name).resolve()
    prepare_dir = job_dir / "prepare"
    opt_root = (job_dir / args.opt_dir_name).resolve()
    opt_root.mkdir(parents=True, exist_ok=True)

    sdf_abs = (abfecfg_dir / acfg.sdffile).resolve()

    for p in [
        prepare_dir / "sysA.top",
        prepare_dir / "sysA_em.pdb",
        prepare_dir / "sysA_ab_ligatoms.json",
        prepare_dir / "sysA_atom_ids.json",
        sdf_abs,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    mol = Chem.SDMolSupplier(sdf_abs, removeHs=False)[0]
    ligand_charge = Chem.GetFormalCharge(mol)

    logger.info("abfecfg     : %s", abfecfg_path)
    logger.info("job_dir     : %s", job_dir)
    logger.info("opt_root    : %s", opt_root)
    logger.info("sdf         : %s", sdf_abs)
    logger.info("lig_charge  : %.4f", ligand_charge)
    logger.info("CUDA visible: %s", get_visible_cuda_devices())

    requested_gpu_ids = None
    if args.gpu_ids is not None:
        requested_gpu_ids = [
            value.strip() for value in args.gpu_ids.split(",") if value.strip()
        ]
    gpu_ids = resolve_cuda_device_ids(requested_gpu_ids)
    if not gpu_ids:
        raise RuntimeError(
            "No GPU ids resolved. Set CUDA_VISIBLE_DEVICES or pass --gpu-ids."
        )
    use_mps = not args.no_mps
    logger.info("gpu_ids     : %s", gpu_ids)
    logger.info("use_mps     : %s", use_mps)

    if ligand_charge != 0:
        ligand_class = "charged"
        logger.info("using get_charged_water_leg_lambdas()")
        initial_schedule = np.array(get_charged_water_leg_lambdas(), dtype=np.float64)
    else:
        ligand_class = "neutral"
        logger.info("using get_neutral_water_leg_lambdas()")
        initial_schedule = np.array(get_neutral_water_leg_lambdas(), dtype=np.float64)

    if args.initial_lam_json is not None:
        lam_json_path = args.initial_lam_json.resolve()
        if not lam_json_path.exists():
            raise FileNotFoundError(
                f"--initial-lam-json file not found: {lam_json_path}"
            )
        loaded = load_initial_schedule(lam_json_path)
        initial_schedule = loaded
        logger.info("initial schedule loaded from: %s", lam_json_path)
    else:
        logger.info("initial schedule loaded from the %s recipe", ligand_class)

    source_window_count = initial_schedule.shape[0]
    if args.initial_lam_json is not None:
        # Honor the manually specified window count: a user-provided
        # --initial-lam-json defines both the thermodynamic path and the
        # number of windows, so do not resample it to the fixed 24/32 count.
        target_window_count = source_window_count
        logger.info(
            "water-leg windows: %d (honoring --initial-lam-json count, "
            "%s ligand; no resample)",
            target_window_count,
            ligand_class,
        )
    else:
        target_window_count = get_water_leg_window_count(ligand_charge)
        initial_schedule = resample_water_leg_schedule(
            initial_schedule, ligand_charge
        )
        logger.info(
            "water-leg windows: %d -> %d (%s ligand)",
            source_window_count,
            target_window_count,
            ligand_class,
        )
    n_replicas = initial_schedule.shape[0]

    if args.t_split is not None:
        t_split = args.t_split
    else:
        t_split = infer_t_split_from_schedule(initial_schedule)
    logger.info("N replicas  : %d", n_replicas)
    logger.info("t_split     : %.4f", t_split)

    run_hremd_fn = make_run_hremd_fn(
        opt_root=opt_root,
        prepare_dir=prepare_dir,
        sdf_abs=sdf_abs,
        acfg=acfg,
        nsnapshots=args.nsnapshots,
        traj_interval=args.traj_interval,
        exchange_interval=args.exchange_interval,
        checkpoint_interval=args.checkpoint_interval,
        gpu_ids=gpu_ids,
        use_mps=use_mps,
        no_plots=args.no_plots,
    )

    # Use path-preserving mode so the full thermodynamic path from the
    # recipe (or --initial-lam-json) is preserved: qB is re-derived as
    # 1 - qA, while lambdaB_vdw is kept free (not forced to 1 - lambdaA_vdw).
    use_path_mode = True
    if args.initial_lam_json is not None:
        logger.info(
            "mode        : path-preserving (using --initial-lam-json as "
            "thermodynamic path; qB = 1 - qA, vdwB kept free)"
        )
    else:
        logger.info(
            "mode        : path-preserving (using recipe schedule as "
            "thermodynamic path; qB = 1 - qA, vdwB kept free)"
        )
    logger.info("target      : %s", args.optimization_target)
    logger.info(
        "acceptance constraint: every adjacent pair > %.4f",
        args.min_acceptance_ratio,
    )

    result = optimize_lambda_schedule(
        initial_schedule=initial_schedule,
        run_hremd_fn=run_hremd_fn,
        t_split=t_split,
        charge_shape=SigmoidShape(k=args.charge_k, x0=args.charge_x0),
        vdw_shape=SigmoidShape(k=args.vdw_k, x0=args.vdw_x0),
        use_path_from_initial_schedule=use_path_mode,
        flow_linearity_norm=args.flow_linearity_norm,
        flow_smooth=args.flow_smooth,
        flow_drop_floor_frac=args.flow_drop_floor_frac,
        min_acceptance_ratio=args.min_acceptance_ratio,
        optimization_target=args.optimization_target,
        acceptance_plateau_tol=args.acceptance_plateau_tol,
        rtr_plateau_tol=args.rtr_plateau_tol,
        patience=args.patience,
        optimize_path_shape=args.optimize_path_shape,
        t_split_step=args.t_split_step,
        vdwb_gamma_step=args.vdwb_gamma_step,
        max_iter=args.max_iter,
        eta=args.eta,
        schedule_transform=lambda schedule: freeze_stage_lambda_counts(
            schedule, initial_schedule
        ),
    )

    logger.info("converged    : %s", result.converged)
    logger.info("n_iterations : %d", result.n_iterations)
    logger.info("selected iter: %d", result.selected_iteration)
    logger.info("final t_split: %.4f", result.params.t_split)
    logger.info("final vdwb_gamma: %.4f", result.params.vdwb_gamma)
    logger.info("final s      : %s", np.array2string(result.s, precision=4))

    np.save(opt_root / "final_schedule.npy", result.schedule)
    np.save(opt_root / "final_s.npy", result.s)

    with open(opt_root / "final_lam_list.json", "w") as f:
        dump_config(
            {"ab": {"lam_list": [list(map(float, r)) for r in result.schedule]}},
            f,
            indent=4,
        )

    selected = result.history[result.selected_iteration]
    params_dict = {
        "t_split": result.params.t_split,
        "canonical_t_split": (
            None if result.params.canonical_t_split is None
            else float(result.params.canonical_t_split)
        ),
        "vdwb_gamma": float(result.params.vdwb_gamma),
        "ligand_class": ligand_class,
        "ligand_charge": int(ligand_charge),
        "n_lambda_windows": int(target_window_count),
        "target": args.optimization_target,
        "flow_linearity_norm": args.flow_linearity_norm,
        "flow_smooth": bool(args.flow_smooth),
        "flow_drop_floor_frac": float(args.flow_drop_floor_frac),
        "min_acceptance_ratio": float(args.min_acceptance_ratio),
        "acceptance_plateau_tol": float(args.acceptance_plateau_tol),
        "rtr_plateau_tol": float(args.rtr_plateau_tol),
        "patience": int(args.patience),
        "optimize_path_shape": bool(args.optimize_path_shape),
        "t_split_step": float(args.t_split_step),
        "vdwb_gamma_step": float(args.vdwb_gamma_step),
        "charge_shape": {
            "k": result.params.charge_shape.k,
            "x0": result.params.charge_shape.x0,
        },
        "vdw_shape": {
            "k": result.params.vdw_shape.k,
            "x0": result.params.vdw_shape.x0,
        },
        "converged": bool(result.converged),
        "n_iterations": int(result.n_iterations),
        "selected_iteration": int(result.selected_iteration),
        "selected_min_acceptance_ratio": float(
            selected.min_acceptance_ratio
        ),
        "selected_mean_acceptance_ratio": float(
            selected.acceptance_mean
        ),
        "selected_acceptance_std": float(selected.acceptance_std),
        "selected_acceptance_range": float(selected.acceptance_range),
        "final_flow_drop_range": float(selected.flow_drop_range),
        "final_flow_drop_mean": float(selected.flow_drop_mean),
        "final_flow_drop_std": float(selected.flow_drop_std),
        "final_flow_linearity_error": float(
            selected.flow_linearity_error
        ),
        "final_round_trip_rate": float(selected.round_trip_rate),
        "final_rtr_rel_improvement": (
            None if selected.rtr_rel_improvement is None
            else float(selected.rtr_rel_improvement)
        ),
        "final_phase_flow_imbalance": (
            None if selected.phase_flow_imbalance is None
            else float(selected.phase_flow_imbalance)
        ),
        "final_diffusivity_mean": float(
            selected.diffusivity.mean()
        ),
        "final_diffusivity_min": float(
            selected.diffusivity.min()
        ),
        "final_diffusivity_max": float(
            selected.diffusivity.max()
        ),
        "path_preserving_mode": use_path_mode,
    }
    with open(opt_root / "final_params.json", "w") as f:
        json.dump(params_dict, f, indent=4)

    logger.info("saved final_schedule.npy   : %s", opt_root / "final_schedule.npy")
    logger.info("saved final_s.npy          : %s", opt_root / "final_s.npy")
    logger.info("saved final_lam_list.json  : %s", opt_root / "final_lam_list.json")
    logger.info("saved final_params.json    : %s", opt_root / "final_params.json")

    if not args.no_plots:
        make_diagnostic_plots(
            initial_schedule=initial_schedule,
            result=result,
            opt_root=opt_root,
        )


if __name__ == "__main__":
    main()
