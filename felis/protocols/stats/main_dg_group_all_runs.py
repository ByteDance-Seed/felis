# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import itertools
import json
import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from numpy.typing import NDArray
from pandas import DataFrame, merge, read_csv

from felis.protocols.correction.post_corrections import PostCorrections
from felis.utils.metrics import Metrics, get_metrics

logger = logging.getLogger(__name__)


def _openfe_like_plot_rcparams() -> dict:
    """Return plotting defaults that mimic the OpenFE/Cinnabar-style figures.

    Returns:
        dict: Matplotlib rcParams overrides used while rendering stats plots.
    """
    # NOTE: Use explicit rcParams (rather than seaborn styles) to avoid relying on
    # optional external dependencies and to keep test output stable.
    return {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.labelpad": 6,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": False,
        "legend.edgecolor": "#bfbfbf",
        "lines.linewidth": 1.2,
        "lines.markersize": 6,
    }


@contextmanager
def _mpl_style_context() -> None:
    """Apply the plotting style within a temporary context.

    Yields:
        None: Control to the caller while the rcParams overrides are active.
    """
    with plt.rc_context(rc=_openfe_like_plot_rcparams()):
        yield


def _metrics_box_text(proname: str, mean_metrics: Metrics) -> str:
    """Format the summary text shown below the scatter plots.

    Args:
        proname (str): Project or protein name printed in the metrics box.
        mean_metrics (Metrics): Aggregated model-quality metrics for the averaged prediction.

    Returns:
        str: Multiline label summarizing R2, shift, RMSE, and MAE in `kcal/mol`.
    """
    avg_R2 = mean_metrics.R2
    avg_shift = mean_metrics.Shift
    avg_rmse = mean_metrics.RMSE
    avg_mae = mean_metrics.MAE
    return (f"{proname} (average, unit: kcal/mol)\n"
            f"R$^2$ {avg_R2:.2f} / dG Shift {avg_shift:.2f}\n"
            f"RMSE {avg_rmse:.2f} / MAE {avg_mae:.2f}")


def _compute_plot_ranges(np_dG_exp: NDArray, np_dG_runs: NDArray, avg_shift: float) -> tuple[list[float], list[float]]:
    """Compute shared raw and shifted plotting ranges for dG scatter plots.

    Args:
        np_dG_exp (NDArray): Experimental free energies in `kcal/mol`. Shape `(n_ligands,)`.
        np_dG_runs (NDArray): Predicted free energies in `kcal/mol`. Shape `(n_runs, n_ligands)`.
        avg_shift (float): Global shift in `kcal/mol` applied to averaged predictions.

    Returns:
        tuple[list[float], list[float]]: Inclusive `[min, max]` ranges for raw and shifted plots.
    """
    global_dG_min, global_dG_max = float(np.min(np_dG_exp)), float(np.max(np_dG_exp))
    global_dG_min_shifted, global_dG_max_shifted = global_dG_min, global_dG_max

    for np_arr in np_dG_runs:
        global_dG_min = min(global_dG_min, float(np.min(np_arr)))
        global_dG_max = max(global_dG_max, float(np.max(np_arr)))
        global_dG_min_shifted = min(global_dG_min_shifted, float(np.min(np_arr) + avg_shift))
        global_dG_max_shifted = max(global_dG_max_shifted, float(np.max(np_arr) + avg_shift))

    range_dG = [float(np.floor(global_dG_min) - 1.0), float(np.ceil(global_dG_max) + 1.0)]
    range_dG_shifted = [float(np.floor(global_dG_min_shifted) - 1.0), float(np.ceil(global_dG_max_shifted) + 1.0)]

    return range_dG, range_dG_shifted


def _common_limits_and_ticks(range_a: list[float], range_b: list[float]) -> tuple[list[float], NDArray]:
    """Choose a shared axis range and integer tick spacing for two plot ranges.

    Args:
        range_a (list[float]): First `[min, max]` plotting range.
        range_b (list[float]): Second `[min, max]` plotting range.

    Returns:
        tuple[list[float], NDArray]: Common axis limits and evenly spaced tick locations.
    """
    l_min = min(range_a[0], range_b[0])
    l_max = max(range_a[1], range_b[1])
    # Ensure strict integer boundaries
    l_min = float(np.floor(l_min))
    l_max = float(np.ceil(l_max))

    min_span = 5.0
    while (l_max - l_min) < min_span:
        l_min -= 1.0
        if (l_max - l_min) < min_span:
            l_max += 1.0

    # Find a suitable integer stride
    # We want roughly 5 to 9 ticks.
    # Start with stride 1.
    final_stride = 1.0
    final_min, final_max = l_min, l_max

    # Common strides
    strides = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000]

    for stride in strides:
        test_min = np.floor(l_min / stride) * stride
        test_max = np.ceil(l_max / stride) * stride
        n_intervals = (test_max - test_min) / stride
        n_ticks = n_intervals + 1

        # Check if number of ticks is reasonable (e.g. <= 9)
        # Also ensure minimum 4 ticks as per original logic preference
        if 4 <= n_ticks <= 9:
            final_stride = stride
            final_min, final_max = test_min, test_max
            break

    ticks = np.arange(final_min, final_max + final_stride / 2, final_stride)
    limits = [final_min, final_max]
    return limits, ticks


def _identity_band_data(plot_range: list[float]) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Build the identity line and +/-1, +/-2 `kcal/mol` guide bands.

    Args:
        plot_range (list[float]): Axis range used to generate the guide lines.

    Returns:
        tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]: X coordinates,
        identity line, +/-1 `kcal/mol` lines, and +/-2 `kcal/mol` lines.
    """
    x = np.linspace(*plot_range)
    y0 = x
    y1, y2 = x - 1.0, x + 1.0
    y3, y4 = x - 2.0, x + 2.0
    return x, y0, y1, y2, y3, y4


def _add_identity_and_bands(ax, x: NDArray, y0: NDArray, y1: NDArray, y2: NDArray, y3: NDArray, y4: NDArray) -> None:
    """Draw the identity line and tolerance bands on a scatter axis.

    Args:
        ax: Matplotlib axis to modify in place.
        x (NDArray): X coordinates for the guide lines.
        y0 (NDArray): Identity-line Y coordinates.
        y1 (NDArray): Lower `1 kcal/mol` band.
        y2 (NDArray): Upper `1 kcal/mol` band.
        y3 (NDArray): Lower `2 kcal/mol` band.
        y4 (NDArray): Upper `2 kcal/mol` band.

    Returns:
        None: The function mutates `ax` in place.
    """
    ax.plot(x, y0, color="black", linewidth=1.2, zorder=1)
    ax.fill_between(x, y1, y2, color="#808080", alpha=0.12, zorder=0)
    ax.plot(x, y3, "--", color="black", linewidth=1.0, alpha=0.9, zorder=1)
    ax.plot(x, y4, "--", color="black", linewidth=1.0, alpha=0.9, zorder=1)


def _finalize_dg_scatter_axis(ax, xlim: list[float], ylim: list[float], ticks: NDArray, ylabel: str) -> None:
    """Apply shared axis formatting for dG comparison plots.

    Args:
        ax: Matplotlib axis to modify in place.
        xlim (list[float]): X-axis limits.
        ylim (list[float]): Y-axis limits.
        ticks (NDArray): Tick locations shared by both axes.
        ylabel (str): Y-axis label describing the prediction series.

    Returns:
        None: The function mutates `ax` in place.
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Expt. dG (kcal/mol)")
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")


def _add_metrics_box(ax, box_text: str) -> None:
    """Attach the metrics summary textbox below a plot.

    Args:
        ax: Matplotlib axis to annotate.
        box_text (str): Multiline summary text to render.

    Returns:
        None: The function mutates `ax` in place.
    """
    ax.text(
        0.5,
        -0.18,
        box_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#bfbfbf",
            "alpha": 0.95
        },
    )


def _save_close(fig, pngname: str) -> None:
    """Save a figure to disk and close it to release resources.

    Args:
        fig: Matplotlib figure to save.
        pngname (str): Output path for the PNG image.

    Returns:
        None: The figure is written to disk and then closed.
    """
    fig.savefig(pngname, bbox_inches="tight", dpi=300)
    plt.close(fig)


def read_benchmark_file(benchmark_file: str) -> Tuple[DataFrame, DataFrame, dict]:
    """Read the benchmark table and derive the per-group experimental subset.

    Args:
        benchmark_file (str): Path to the benchmark CSV file.

    Returns:
        Tuple[DataFrame, DataFrame, dict]: Full ligand-state table, one-row-per-group
        experimental table, and mapping from ligand name to benchmark group id.
    """
    bm_orig = read_csv(benchmark_file, dtype={"group_id": str, "Ligand_name": str})
    keep = ["group_id", "Ligand_name", "exp_dG_kcal_mol"]
    bm_full = bm_orig.drop(columns=[bm_orig.columns[0], *[c for c in bm_orig.columns if c not in keep]])

    group_id_set = set()
    group_id_dict = dict()
    drop_list = []
    for idx, row in bm_full.iterrows():
        group_id = row["group_id"]
        ligand_name = row["Ligand_name"]
        group_id_dict[ligand_name] = group_id
        if group_id in group_id_set:
            drop_list.append(idx)
        else:
            group_id_set.add(group_id)
    bm_exp = bm_full.drop(drop_list)
    return bm_full, bm_exp, group_id_dict


def read_result_one_run(run_result: str,
                        group_id_dict: dict,
                        select_method: str = "min") -> Tuple[DataFrame, DataFrame]:
    """Read one run result file and build its filtered representation.

    Args:
        run_result (str): Path to a per-run result file.
        group_id_dict (dict): Mapping from ligand name to benchmark group id.
        select_method (str): Filtering policy. `min` keeps the minimum-energy ligand
            state per group. `precise` expands post-correction merged nodes and keeps
            the unique precise node per group.

    Returns:
        Tuple[DataFrame, DataFrame]: Full ligand-level run table and a filtered table
        containing one representative row per benchmark group.

    Raises:
        ValueError: If `select_method` is unsupported, if a ligand is absent from the
            benchmark mapping, or if `precise` input is ambiguous.
    """
    df_full = read_csv(run_result, sep=r"[:,\t]", engine="python", dtype={"ligand": str})

    name_filter = defaultdict(list)
    for idx, row in df_full.iterrows():
        ligname = row["ligand"]
        group_id = group_id_dict[ligname]
        name_filter[group_id].append(idx)

    df_full_copy = df_full.copy()
    for group_id, idx_list in name_filter.items():
        df_full_copy.loc[idx_list, "ligand"] = group_id

    filtered_rows = []
    if select_method == "min" or select_method == "precise":
        for group_id, idx_list in name_filter.items():
            calc_dGs = [df_full.loc[idx]["dG(kcal/mol)"] for idx in idx_list]
            index_min = np.argmin(calc_dGs)
            filtered_rows.append(idx_list[index_min])
    else:
        raise ValueError(f"Unsupported select_method: {select_method}")
    df_filtered = df_full_copy.loc[filtered_rows]

    return df_full, df_filtered


def compare_full_dGs(bm_full: DataFrame,
                     df_runs_full: list[DataFrame]) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Aggregate full-state predictions across all runs.

    Args:
        bm_full (DataFrame): Benchmark table with ligand names and experimental dG values.
        df_runs_full (list[DataFrame]): Per-run ligand-state tables containing
            `dG(kcal/mol)` and, optionally, component-energy columns.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame, DataFrame]: Full dG comparison table
        plus solvent, protein, and restraint component summaries. The latter three
        values are `None` when required component columns are unavailable.
    """
    df_full = bm_full[["Ligand_name", "exp_dG_kcal_mol"]].copy()
    df_full.rename(columns={"Ligand_name": "ligand", "exp_dG_kcal_mol": "exp"}, inplace=True)

    df_comparedG = df_full.copy()
    df_comparesol = DataFrame({"ligand": df_full["ligand"].values})
    df_comparepro = DataFrame({"ligand": df_full["ligand"].values})
    df_compareres = DataFrame({"ligand": df_full["ligand"].values})

    col_missing = False
    n_ligand_states = len(bm_full)
    np_dG_runs = np.empty((0, n_ligand_states))
    for idx, idf in enumerate(df_runs_full):
        np_dG = df_full["ligand"].map(idf.set_index("ligand")["dG(kcal/mol)"]).values
        np_dG_runs = np.vstack((np_dG_runs, np_dG))

        df_comparedG = merge(df_comparedG, idf[["ligand", "dG(kcal/mol)"]], on="ligand")
        df_comparedG.rename(columns={"dG(kcal/mol)": f"r{idx+1}"}, inplace=True)

        for col in ("sol.e.1.0", "sol.v.1.0", "pro.e.0.1", "pro.v.0.1", "res.r.0.1", "res.r.1.0"):
            if col not in idf.columns:
                logger.info(f"Column {col} not found in run {idx+1}")
                col_missing = True
        if col_missing:
            continue

        df_comparesol = merge(df_comparesol, idf[["ligand", "sol.e.1.0", "sol.v.1.0"]], on="ligand")
        df_comparesol.rename(columns={
            "sol.e.1.0": f"sol.e.1.0-r{idx+1}",
            "sol.v.1.0": f"sol.v.1.0-r{idx+1}"
        },
                             inplace=True)

        df_comparepro = merge(df_comparepro, idf[["ligand", "pro.e.0.1", "pro.v.0.1"]], on="ligand")
        df_comparepro.rename(columns={
            "pro.e.0.1": f"pro.e.0.1-r{idx+1}",
            "pro.v.0.1": f"pro.v.0.1-r{idx+1}"
        },
                             inplace=True)

        df_compareres = merge(df_compareres, idf[["ligand", "res.r.0.1", "res.r.1.0"]], on="ligand")
        df_compareres.rename(columns={
            "res.r.0.1": f"res.r.0.1-r{idx+1}",
            "res.r.1.0": f"res.r.1.0-r{idx+1}"
        },
                             inplace=True)

    np_dG_mean = np.mean(np_dG_runs, axis=0)
    np_dG_std = np.std(np_dG_runs, ddof=1, axis=0)
    np_dG_range = np.max(np_dG_runs, axis=0) - np.min(np_dG_runs, axis=0)
    df_comparedG["avg/dG"] = np_dG_mean
    df_comparedG["stdev/dG"] = np_dG_std
    df_comparedG["range/dG"] = np_dG_range

    if col_missing:
        return df_comparedG, None, None, None

    df_sol_e = df_comparesol.filter(like="sol.e")
    df_sol_v = df_comparesol.filter(like="sol.v")
    np_sol_sum = df_sol_e.to_numpy() + df_sol_v.to_numpy()
    df_comparesol["stdev/sol.elec"] = df_sol_e.std(axis=1)
    df_comparesol["stdev/sol.vdw"] = df_sol_v.std(axis=1)
    df_comparesol["stdev/sol"] = np.std(np_sol_sum, ddof=1, axis=1)

    df_pro_e = df_comparepro.filter(like="pro.e")
    df_pro_v = df_comparepro.filter(like="pro.v")
    np_pro_sum = df_pro_e.to_numpy() + df_pro_v.to_numpy()
    df_comparepro["stdev/pro.elec"] = df_pro_e.std(axis=1)
    df_comparepro["stdev/pro.vdw"] = df_pro_v.std(axis=1)
    df_comparepro["stdev/pro"] = np.std(np_pro_sum, ddof=1, axis=1)

    df_res01 = df_compareres.filter(like="res.r.0.1")
    df_res10 = df_compareres.filter(like="res.r.1.0")
    np_res_sum = df_res01.to_numpy() + df_res10.to_numpy()
    df_compareres["stdev/res.r.0.1"] = df_res01.std(axis=1)
    df_compareres["stdev/res.r.1.0"] = df_res10.std(axis=1)
    df_compareres["stdev/res"] = np.std(np_res_sum, ddof=1, axis=1)

    return df_comparedG, df_comparesol, df_comparepro, df_compareres


def _compare_filtered_dGsv1(
        bm_exp: DataFrame,
        df_runs_filtered: list[DataFrame]) -> Tuple[DataFrame, NDArray, NDArray, NDArray, Metrics, list[Metrics]]:
    """Aggregate already-filtered per-group run tables.

    Args:
        bm_exp (DataFrame): One-row-per-group benchmark table with experimental values.
        df_runs_filtered (list[DataFrame]): Filtered run tables containing one row per group.

    Returns:
        Tuple[DataFrame, NDArray, NDArray, NDArray, Metrics, list[Metrics]]: Filtered
        comparison table, experimental values, mean predictions, all run predictions,
        metrics for the mean prediction, and metrics for each run.
    """
    df_exp = bm_exp[["group_id", "exp_dG_kcal_mol"]].copy()
    df_exp.rename(columns={"group_id": "ligand", "exp_dG_kcal_mol": "exp"}, inplace=True)

    df_compare_filtered = df_exp.copy()
    assert df_compare_filtered["ligand"].is_unique

    n_ligands = len(df_exp)
    np_dG_runs = np.empty((0, n_ligands))

    for idx, idf0 in enumerate(df_runs_filtered):
        assert idf0["ligand"].is_unique and set(df_compare_filtered["ligand"]) == set(idf0["ligand"])
        idf = idf0.set_index("ligand").reindex(df_compare_filtered["ligand"]).reset_index()
        np_dG = idf["dG(kcal/mol)"].values
        np_dG_runs = np.vstack((np_dG_runs, np_dG))
        df_compare_filtered[f"r{idx+1}"] = np_dG

    np_dG_mean = np.mean(np_dG_runs, axis=0)
    np_dG_std = np.std(np_dG_runs, ddof=1, axis=0)
    np_dG_range = np.max(np_dG_runs, axis=0) - np.min(np_dG_runs, axis=0)
    df_compare_filtered["avg/dG"] = np_dG_mean
    df_compare_filtered["stdev/dG"] = np_dG_std
    df_compare_filtered["range/dG"] = np_dG_range

    np_dG_exp = df_exp["exp"].values
    mean_metrics = get_metrics(np_dG_mean, np_dG_exp)
    list_of_metrics = [get_metrics(ipred, np_dG_exp) for ipred in np_dG_runs]
    return df_compare_filtered, np_dG_exp, np_dG_mean, np_dG_runs, mean_metrics, list_of_metrics


def _get_group_id(bm_full, Ligand_name):
    """Look up the benchmark group id for one ligand name.

    Args:
        bm_full: Benchmark table containing `Ligand_name` and `group_id`.
        Ligand_name: Ligand identifier to resolve.

    Returns:
        str: Benchmark group id associated with the ligand.
    """
    group_id = bm_full.loc[bm_full["Ligand_name"] == Ligand_name, "group_id"].values[0]
    return group_id


def _flatten_ligand_parts(parts_raw, fallback_ligand: str) -> list[str]:
    """Flatten nested `ligand_parts` payloads into a unique ligand list."""
    if parts_raw is None or (isinstance(parts_raw, float) and np.isnan(parts_raw)):
        return [str(fallback_ligand)]

    if isinstance(parts_raw, str):
        # NOTE: pandas TSV/CSV round-trips serialize list-valued cells using Python
        # repr (e.g. "['a', 'b']"). This module intentionally does NOT support
        # re-hydrating those artifacts; consumers should use in-memory
        # `PostCorrections.get_results()` outputs instead.
        if parts_raw.lstrip().startswith(("[", "(", "{")):
            raise ValueError(
                "ligand_parts looks like a serialized container string (e.g. from TSV/CSV autosave). "
                "Do not read ligand_parts from autosaved tables; use in-memory PostCorrections.get_results() output.")
        parts_val = parts_raw
    else:
        parts_val = parts_raw

    def _flatten(value) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            flattened = []
            for item in value:
                flattened.extend(_flatten(item))
            return flattened
        return [str(value)]

    flattened_parts = []
    seen = set()
    for ligand in _flatten(parts_val):
        if ligand in {"", "None", "nan"} or ligand in seen:
            continue
        flattened_parts.append(ligand)
        seen.add(ligand)
    return flattened_parts or [str(fallback_ligand)]


def _run_post_corrections_on_input_df(input_csv: str, correction_config_json: str) -> DataFrame:
    """Run `PostCorrections` on a nodewise ligand table.

    This helper is used by `compare_filtered_dGs_v3(..., select_method="precise")`.
    It expects `input_csv` to be an on-disk nodewise table (typically written from a
    DataFrame) with per-ligand predicted free energies and experimental references.

    Args:
        input_csv (str): Path to a CSV file containing columns `ligand`, `avg/dG`,
            and `exp`.
        correction_config_json (str): Path to a JSON correction config consumed by
            `PostCorrections`.

    Returns:
        DataFrame: Post-corrected nodewise results from `PostCorrections.get_results()`.
        The returned table includes merged `ligand` labels (for rotamer/pKa merges),
        `avg/dG`, and provenance columns such as `ligand_parts` / `avg_dG_parts`.
        The `exp` value is preserved on merged nodes when present.

    Raises:
        ValueError: If `correction_config_json` is missing, or if `input_csv` lacks
            required columns.
    """
    if not correction_config_json or not Path(correction_config_json).exists():
        raise ValueError("select_method='precise' requires correction_config_json")

    with open(correction_config_json, encoding="utf-8") as ifs:
        correction_config = json.load(ifs)

    required_columns = ["ligand", "avg/dG", "exp"]
    df_input = pd.read_csv(input_csv)
    missing_columns = [col for col in required_columns if col not in df_input.columns]
    if missing_columns:
        raise ValueError(f"Input dataframe is missing required columns for precise mode: {missing_columns}")

    postcorr = PostCorrections(input_csv, correction_config, output_dir=Path(input_csv).parent, rt=0.596)
    postcorr.run(debug=True)
    return postcorr.get_results()


def _build_postcorrections_input_df_for_run(bm_full: DataFrame, df_run: DataFrame) -> DataFrame:
    """Build one run-level nodewise input table for `PostCorrections`.
        
        Args:
            bm_full: Benchmark table containing `Ligand_name` and `exp_dG_kcal_mol`.
            df_run: One-row-per-ligand_state run table containing `ligand` and `dG(kcal/mol)` columns.

        Returns:
            DataFrame: Nodewise input table for `PostCorrections` with `ligand`, `avg/dG`, and `exp` columns.
    """
    if "ligand" not in df_run.columns or "dG(kcal/mol)" not in df_run.columns:
        raise ValueError("Each run dataframe must contain 'ligand' and 'dG(kcal/mol)' columns for precise mode.")
    if not df_run["ligand"].is_unique:
        raise ValueError("Each run dataframe must contain unique ligand rows for precise mode.")

    df_input = bm_full[["Ligand_name", "exp_dG_kcal_mol"]].copy()
    df_input.rename(columns={"Ligand_name": "ligand", "exp_dG_kcal_mol": "exp"}, inplace=True)
    run_dg_map = df_run.set_index("ligand")["dG(kcal/mol)"]
    df_input["avg/dG"] = df_input["ligand"].map(run_dg_map)

    missing_mask = df_input["avg/dG"].isna()
    if missing_mask.any():
        missing_ligands = df_input.loc[missing_mask, "ligand"].tolist()
        raise ValueError(f"Precise mode requires run results for all benchmark ligand states: {missing_ligands}")
    return df_input[["ligand", "avg/dG", "exp"]]


def _map_postcorrected_results_to_groups(bm_full: DataFrame, corrected_df: DataFrame) -> dict[str, float]:
    """Map post-corrected nodewise results back to benchmark group ids."""
    group_to_dg = dict()
    for _, row in corrected_df.iterrows():
        ligand = str(row["ligand"])
        ligand_parts = _flatten_ligand_parts(row.get("ligand_parts", ligand), ligand)
        group_ids = {_get_group_id(bm_full, ligand_part) for ligand_part in ligand_parts}
        if len(group_ids) != 1:
            raise ValueError(f"Corrected result '{ligand}' spans multiple benchmark groups: {sorted(group_ids)}")

        group_id = next(iter(group_ids))
        if group_id in group_to_dg:
            raise ValueError(
                f"Multiple corrected rows share group_id '{group_id}'; expected exactly one corrected result per benchmark group."
            )
        group_to_dg[group_id] = float(row["avg/dG"])
    return group_to_dg


def compare_filtered_dGs_v2(
        bm_full: DataFrame, bm_exp: DataFrame, df_comparedG: DataFrame, select_method: str,
        df_runs: list[DataFrame]) -> Tuple[DataFrame, NDArray, NDArray, NDArray, Metrics, list[Metrics]]:
    """Aggregate one filtered free-energy value per benchmark group across runs.

    Args:
        bm_full (DataFrame): Full benchmark ligand-state table.
        bm_exp (DataFrame): One-row-per-group benchmark table with experimental values.
        df_comparedG (DataFrame): Full-state comparison table from `compare_full_dGs`.
        select_method (str): Filtering policy. `min` picks the lowest averaged ligand
            state per group. 
        df_runs (list[DataFrame]): Per-run ligand-state tables.

    Returns:
        Tuple[DataFrame, NDArray, NDArray, NDArray, Metrics, list[Metrics]]: Filtered
        comparison table, experimental values, mean predictions, prediction standard
        deviations, all run predictions, metrics for the mean prediction, and metrics
        for each run.

    Raises:
        ValueError: If `select_method` is unsupported or if `precise` data is incomplete
            or ambiguous.
    """
    df_exp = bm_exp[["group_id", "exp_dG_kcal_mol"]].copy()
    df_exp.rename(columns={"group_id": "ligand", "exp_dG_kcal_mol": "exp"}, inplace=True)

    df_compare_filtered = df_exp.copy()
    assert df_compare_filtered["ligand"].is_unique

    n_ligands = len(df_exp)
    np_dG_runs = np.empty((0, n_ligands))

    selected_ligand_dict = dict()
    for _idx, row in df_comparedG.iterrows():
        ligand_str = row["ligand"]
        group_str = _get_group_id(bm_full, ligand_str)
        exp_dg = row["exp"]
        avg_dg = row["avg/dG"]
        if group_str not in selected_ligand_dict.keys():
            selected_ligand_dict[group_str] = (group_str, ligand_str, avg_dg, exp_dg)
        elif select_method == "min" and avg_dg < selected_ligand_dict[group_str][2]:
            selected_ligand_dict[group_str] = (group_str, ligand_str, avg_dg, exp_dg)

    for idx, idf0 in enumerate(df_runs):
        idf1 = idf0.copy()
        drop_list = []
        for idx2, row in idf1.iterrows():
            ligand_str = row["ligand"]
            group_str = _get_group_id(bm_full, ligand_str)
            _group_str, _ligand_str, _avg_dg, _exp_dg = selected_ligand_dict[group_str]
            if ligand_str != _ligand_str:
                drop_list.append(idx2)
            else:
                idf1.at[idx2, "ligand"] = group_str
        idf1 = idf1.drop(index=drop_list, inplace=False)
        idf = idf1.set_index("ligand").reindex(df_compare_filtered["ligand"]).reset_index()
        np_dG = idf["dG(kcal/mol)"].values
        np_dG_runs = np.vstack((np_dG_runs, np_dG))
        df_compare_filtered[f"r{idx+1}"] = np_dG

    np_dG_mean = np.mean(np_dG_runs, axis=0)
    np_dG_std = np.std(np_dG_runs, ddof=1, axis=0)
    np_dG_range = np.max(np_dG_runs, axis=0) - np.min(np_dG_runs, axis=0)
    df_compare_filtered["avg/dG"] = np_dG_mean
    df_compare_filtered["stdev/dG"] = np_dG_std
    df_compare_filtered["range/dG"] = np_dG_range

    np_dG_exp = df_exp["exp"].values
    mean_metrics = get_metrics(np_dG_mean, np_dG_exp)
    list_of_metrics = [get_metrics(ipred, np_dG_exp) for ipred in np_dG_runs]
    return df_compare_filtered, np_dG_exp, np_dG_mean, np_dG_std, np_dG_runs, mean_metrics, list_of_metrics


def compare_filtered_dGs_v3(
        bm_full: DataFrame,
        bm_exp: DataFrame,
        df_comparedG: DataFrame,
        select_method: str,
        df_runs: list[DataFrame],
        debug_dir: str,
        correction_config_json: str = None
) -> Tuple[DataFrame, NDArray, NDArray, NDArray, NDArray, Metrics, list[Metrics]]:
    """Aggregate one filtered free-energy value per benchmark group across runs.

    `min` mode is identical to `compare_filtered_dGs_v2`. `precise` mode runs
    `PostCorrections` independently on each run's full ligand-state table, writes
    the corrected group-level values into `run{n}_corrected_dG` columns, and then
    computes the returned mean/std/range from those corrected per-run values.
    """
    if select_method == "min":
        return compare_filtered_dGs_v2(bm_full, bm_exp, df_comparedG, select_method, df_runs)
    if select_method != "precise":
        raise ValueError(f"Unsupported select_method: {select_method}")

    assert Path(debug_dir).exists()

    df_exp = bm_exp[["group_id", "exp_dG_kcal_mol"]].copy()
    df_exp.rename(columns={"group_id": "ligand", "exp_dG_kcal_mol": "exp"}, inplace=True)

    df_compare_filtered = df_exp.copy()
    assert df_compare_filtered["ligand"].is_unique

    n_ligands = len(df_exp)
    np_dG_runs = np.empty((0, n_ligands))
    group_order = df_compare_filtered["ligand"]

    for idx, df_run in enumerate(df_runs):
        df_input = _build_postcorrections_input_df_for_run(bm_full, df_run)
        input_csv = Path(debug_dir) / f"postcorr_r{idx+1}.csv"
        df_input.to_csv(input_csv, index=False)
        corrected_df = _run_post_corrections_on_input_df(input_csv, correction_config_json)
        group_to_dg = _map_postcorrected_results_to_groups(bm_full, corrected_df)

        missing_groups = [group_id for group_id in group_order if group_id not in group_to_dg]
        if missing_groups:
            raise ValueError(f"Run {idx+1} is missing corrected result for benchmark groups: {missing_groups}")

        corrected_run = group_order.map(group_to_dg).to_numpy(dtype=float)
        np_dG_runs = np.vstack((np_dG_runs, corrected_run))
        df_compare_filtered[f"r{idx+1}"] = corrected_run

    np_dG_mean = np.mean(np_dG_runs, axis=0)
    np_dG_std = np.std(np_dG_runs, ddof=1, axis=0)
    np_dG_range = np.max(np_dG_runs, axis=0) - np.min(np_dG_runs, axis=0)
    df_compare_filtered["avg/dG"] = np_dG_mean
    df_compare_filtered["stdev/dG"] = np_dG_std
    df_compare_filtered["range/dG"] = np_dG_range

    np_dG_exp = df_exp["exp"].values
    mean_metrics = get_metrics(np_dG_mean, np_dG_exp)
    list_of_metrics = [get_metrics(ipred, np_dG_exp) for ipred in np_dG_runs]
    return df_compare_filtered, np_dG_exp, np_dG_mean, np_dG_std, np_dG_runs, mean_metrics, list_of_metrics


def dump_metrics(m: Metrics, filename: str) -> None:
    """Write one metrics dataclass to a single-row TSV file.

    Args:
        m (Metrics): Metrics object to serialize.
        filename (str): Output TSV path.

    Returns:
        None: The metrics table is written to disk.
    """
    m_dict = asdict(m)
    dump_dict = dict()
    for k, v in m_dict.items():
        dump_dict[k] = [v]
    dump_df = DataFrame(dump_dict)
    dump_df.to_csv(filename, sep="\t", float_format="%.4f", index=False)


def plot_all_metrics(np_dG_exp: NDArray, np_dG_mean: NDArray, np_dG_runs: NDArray, mean_metrics: Metrics, proname: str,
                     pngname: str, pngname_shifted: str, np_dG_stdev: NDArray, pngname_stdev: str,
                     pngname_shifted_stdev: str) -> None:
    """Render scatter and error-bar summary plots for all runs.

    Args:
        np_dG_exp (NDArray): Experimental free energies in `kcal/mol`. Shape `(n_ligands,)`.
        np_dG_mean (NDArray): Mean predicted free energies in `kcal/mol`. Shape `(n_ligands,)`.
        np_dG_runs (NDArray): Per-run predicted free energies in `kcal/mol`.
            Shape `(n_runs, n_ligands)`.
        mean_metrics (Metrics): Metrics computed from `np_dG_mean` against `np_dG_exp`.
        proname (str): Project or protein name shown in plot annotations.
        pngname (str): Output path for the raw scatter plot.
        pngname_shifted (str): Output path for the shifted scatter plot.
        np_dG_stdev (NDArray): Per-ligand standard deviation in `kcal/mol`.
            Shape `(n_ligands,)`.
        pngname_stdev (str): Output path for the raw error-bar plot.
        pngname_shifted_stdev (str): Output path for the shifted error-bar plot.

    Returns:
        None: Plot images are written to disk.
    """

    avg_shift = mean_metrics.Shift
    box_text = _metrics_box_text(proname=proname, mean_metrics=mean_metrics)

    # Keep color/marker choices stable to avoid confusing diffs when comparing figures.
    preselected_colors = itertools.cycle(["C0", "C1", "C2", "C4", "C5", "C6"])
    preselected_markers = itertools.cycle(["o", "D", "s", "p", "^"])

    # Make the average marker more prominent (but still professional).
    run_specs = [(120, "C3", "*", "average")]
    for idx in range(len(np_dG_runs)):
        run_specs.append((20, next(preselected_colors), next(preselected_markers), f"run{idx+1}"))

    range_dG, range_dG_shifted = _compute_plot_ranges(np_dG_exp=np_dG_exp, np_dG_runs=np_dG_runs, avg_shift=avg_shift)
    common_limits, common_ticks = _common_limits_and_ticks(range_dG, range_dG_shifted)
    x, y0, y1, y2, y3, y4 = _identity_band_data(common_limits)

    np_dG_mean_runs = np.vstack((np_dG_mean, np_dG_runs))

    with _mpl_style_context():
        # Scatter plots (raw and shifted)
        for shifted, ipngname in [(False, pngname), (True, pngname_shifted)]:
            fig, ax = plt.subplots(figsize=(4, 4))
            for (s, c, m, l), np_arr in zip(run_specs, np_dG_mean_runs):
                y_modified = np_arr + (avg_shift if shifted else 0.0)
                is_average = l == "average"
                facecolor = to_rgba(c, alpha=0.28 if is_average else 0.18)
                edgecolor = to_rgba(c, alpha=1.0)
                ax.scatter(
                    np_dG_exp,
                    y_modified,
                    s=s,
                    facecolors=facecolor,
                    edgecolors=edgecolor,
                    linewidths=1.2 if is_average else 0.8,
                    marker=m,
                    label=l,
                    zorder=3 if is_average else 2,
                )

            if shifted:
                _add_identity_and_bands(ax, x, y0, y1, y2, y3, y4)
                _finalize_dg_scatter_axis(ax,
                                          xlim=common_limits,
                                          ylim=common_limits,
                                          ticks=common_ticks,
                                          ylabel="Shifted ABFEP dG (kcal/mol)")
            else:
                _add_identity_and_bands(ax, x, y0, y1, y2, y3, y4)
                _finalize_dg_scatter_axis(ax,
                                          xlim=common_limits,
                                          ylim=common_limits,
                                          ticks=common_ticks,
                                          ylabel="ABFEP dG (kcal/mol)")

            ax.legend(loc="best")
            _add_metrics_box(ax, box_text)
            _save_close(fig, ipngname)

        # Error-bar plots (raw and shifted)
        for shifted, ipngname in [(False, pngname_stdev), (True, pngname_shifted_stdev)]:
            fig, ax = plt.subplots(figsize=(4, 4))

            markercolor = to_rgba("C3", alpha=1.0)
            errorcolor = to_rgba("black", alpha=0.95)
            elinewidth = 0.8
            y_modified = np_dG_mean + (avg_shift if shifted else 0.0)
            ax.errorbar(
                np_dG_exp,
                y_modified,
                yerr=np_dG_stdev,
                fmt="*",
                markersize=9,
                mfc=markercolor,
                mec=markercolor,
                ecolor=errorcolor,
                elinewidth=elinewidth,
                capsize=3,
                capthick=elinewidth,
                label=f"average+/-stdev {len(np_dG_runs)} runs",
                zorder=2,
            )

            if shifted:
                _add_identity_and_bands(ax, x, y0, y1, y2, y3, y4)
                _finalize_dg_scatter_axis(ax,
                                          xlim=common_limits,
                                          ylim=common_limits,
                                          ticks=common_ticks,
                                          ylabel="Shifted ABFEP dG (kcal/mol)")
            else:
                _add_identity_and_bands(ax, x, y0, y1, y2, y3, y4)
                _finalize_dg_scatter_axis(ax,
                                          xlim=common_limits,
                                          ylim=common_limits,
                                          ticks=common_ticks,
                                          ylabel="ABFEP dG (kcal/mol)")

            ax.legend(loc="best")
            _add_metrics_box(ax, box_text)
            _save_close(fig, ipngname)
