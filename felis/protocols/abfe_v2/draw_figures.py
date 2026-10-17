# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np

from felis.protocols.abfe_v2.lambda_config import read_lambda_schedule

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("matplotlib is not found. Please install matplotlib.") from e

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1a55FF",
    "#d6fc28",
    "#1367ad",
    "#3c064b",
    "#b317e2",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1a55FF",
    "#d6fc28",
    "#1367ad",
    "#3c064b",
    "#b317e2",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1a55FF",
    "#d6fc28",
    "#1367ad",
    "#3c064b",
    "#b317e2",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1a55FF",
    "#d6fc28",
    "#1367ad",
    "#3c064b",
    "#b317e2",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1a55FF",
    "#d6fc28",
    "#1367ad",
    "#3c064b",
    "#b317e2",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1a55FF",
    "#d6fc28",
    "#1367ad",
    "#3c064b",
    "#b317e2",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1a55FF",
    "#d6fc28",
    "#1367ad",
    "#3c064b",
    "#b317e2",
]

def _first_index(mask: np.ndarray) -> int | None:
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    return int(indices[0])


def _infer_lambda_phase_indices(schedule: np.ndarray) -> dict:
    restraint = schedule[:, 6]
    q_a = schedule[:, 0]
    vdw_a = schedule[:, 3]

    restraint_max = float(restraint.max())
    vdw_start = float(vdw_a[0])
    return {
        "restraint end": _first_index(restraint >= restraint_max - 1e-9),
        "charge end": _first_index(q_a <= 1e-9),
        "vdW start": _first_index(vdw_a < vdw_start - 1e-9),
    }


def plot_lambda_schedule(
        figure_path: str,
        lam_json_path: str = None,
        lam_list: list = None,
        title: str = None,
    ):
    """Plot an ABFE v2 lambda schedule and annotate key phase indices.

    Args:
        figure_path (str): Path to the output figure.
        lam_json_path (str, optional): Path to a JSON file containing
            ``{"ab": {"lam_list": ...}}`` or a raw lambda-list JSON array.
        lam_list (list, optional): In-memory lambda schedule with shape
            ``(N, 7)``. Used when ``lam_json_path`` is not provided.
        title (str, optional): Figure title.
    """
    if lam_json_path is None and lam_list is None:
        raise ValueError("Either lam_json_path or lam_list must be provided")
    if lam_json_path is not None:
        schedule = read_lambda_schedule(lam_json_path)
        schedule = np.asarray(schedule, dtype=np.float64)
    else:
        schedule = np.asarray(lam_list, dtype=np.float64)
        if schedule.ndim != 2 or schedule.shape[1] != 7:
            raise ValueError(
                f"lambda schedule must have shape (N, 7); got {schedule.shape}"
            )

    x = np.arange(schedule.shape[0])
    phase_indices = _infer_lambda_phase_indices(schedule)
    marker_styles = {
        "restraint end": ("#d62728", "--"),
        "charge end": ("#ff7f0e", "-."),
        "vdW start": ("#2ca02c", ":"),
    }

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), dpi=300, sharex=True)
    if title is None:
        title = "Lambda schedule"
    fig.suptitle(title)

    panels = [
        (
            axes[0],
            [
                (0, "lambdaA_charge"),
                (1, "lambdaB_charge"),
                (2, "lambdaC_charge"),
            ],
            "charge lambda",
        ),
        (
            axes[1],
            [
                (3, "lambdaA_vdw"),
                (4, "lambdaB_vdw"),
            ],
            "vdW lambda",
        ),
        (
            axes[2],
            [
                (5, "torsionLambda"),
                (6, "restraintLambda"),
            ],
            "torsion/restraint lambda",
        ),
    ]

    for ax, columns, ylabel in panels:
        for col_idx, label in columns:
            ax.plot(x, schedule[:, col_idx], marker="o", ms=2.5,
                    linewidth=1.2, label=label)
        for label, idx in phase_indices.items():
            if idx is None:
                continue
            color, linestyle = marker_styles[label]
            ax.axvline(idx, color=color, linestyle=linestyle, linewidth=1.0,
                       alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.legend(fontsize=8, loc="best")

    top_ax = axes[0]
    for offset, (label, idx) in enumerate(phase_indices.items()):
        if idx is None:
            continue
        color, _ = marker_styles[label]
        top_ax.annotate(
            f"{label} idx={idx}",
            xy=(idx, 1.0),
            xytext=(4, -12 - 14 * offset),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color=color,
        )

    axes[-1].set_xlabel("lambda index")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    return


def plot_free_energy(
        figure_path: str,
        beta: float,
        forward_time: np.ndarray,
        backward_time: np.ndarray,
        moving_time: np.ndarray,
        x_label: str,
        forward_results: list = None,
        backward_results: list = None,
        moving_results: list = None,
    ):
    """Plot BAR convergence estimates at their recorded sampling positions."""
    forward_time = np.asarray(forward_time, dtype=float)
    backward_time = np.asarray(backward_time, dtype=float)
    moving_time = np.asarray(moving_time, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    final_estimate = None
    if forward_results is not None:
        forward_dG = np.array([res["Delta_f"] / beta for res in forward_results])
        forward_dG_std = np.array([res["dDelta_f"] / beta for res in forward_results])
        if len(forward_time) != len(forward_dG):
            raise ValueError("forward_time and forward_results must have equal length")
        ax.plot(forward_time, forward_dG, label="Cumulative forward", color=COLORS[0])
        ax.fill_between(
            forward_time,
            forward_dG - forward_dG_std,
            forward_dG + forward_dG_std,
            alpha=0.2,
            color=COLORS[0],
        )
        final_estimate = forward_dG[-1]

    if backward_results is not None:
        backward_dG = np.array([res["Delta_f"] / beta for res in backward_results])
        backward_dG_std = np.array([res["dDelta_f"] / beta for res in backward_results])
        if len(backward_time) != len(backward_dG):
            raise ValueError("backward_time and backward_results must have equal length")
        ax.plot(backward_time, backward_dG, label="Cumulative backward", color=COLORS[1])
        ax.fill_between(
            backward_time,
            backward_dG - backward_dG_std,
            backward_dG + backward_dG_std,
            alpha=0.2,
            color=COLORS[1],
        )

    if moving_results is not None:
        moving_dG = np.array([res["Delta_f"] / beta for res in moving_results])
        moving_dG_std = np.array([res["dDelta_f"] / beta for res in moving_results])
        if len(moving_time) != len(moving_dG):
            raise ValueError("moving_time and moving_results must have equal length")
        ax.plot(moving_time, moving_dG, label="Independent block", color=COLORS[2])
        ax.fill_between(
            moving_time,
            moving_dG - moving_dG_std,
            moving_dG + moving_dG_std,
            alpha=0.2,
            color=COLORS[2],
        )

    if final_estimate is not None:
        ax.axhline(
            final_estimate,
            color=COLORS[0],
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
            label="Final all-sample estimate",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\Delta G$ (kJ/mol)")
    ax.set_title("Free-energy convergence by sampling block")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return

def plot_overlap_matrix(
        figure_path: str,
        overlap_matrix: np.ndarray,
        lambda_label_list: list = None
    ):
    """Plot overlaps between neighboring lambda states.

    Args:
        figure_path (str): The path to the output figure.
        overlap_matrix (np.ndarray): Overlap matrix of shape (num_replicas, num_replicas).
        lambda_label_list (List, optional): List of the label names for each lambda in output figure. If not provided, ``range(num_lambdas)`` will be used as label names.
    """
    overlap_matrix = np.asarray(overlap_matrix, dtype=float)
    if overlap_matrix.ndim != 2 or overlap_matrix.shape[0] != overlap_matrix.shape[1]:
        raise ValueError(
            "overlap_matrix must be a square two-dimensional array; "
            f"got shape {overlap_matrix.shape}"
        )

    size = overlap_matrix.shape[0]
    if size < 2:
        raise ValueError("overlap_matrix must contain at least two states")
    if not np.isfinite(overlap_matrix).all():
        raise ValueError("overlap_matrix contains NaN or infinite values")

    if lambda_label_list is None:
        lambda_label_list = [str(index) for index in range(size)]
    if len(lambda_label_list) != size:
        raise ValueError(
            "lambda_label_list length must match overlap_matrix size; "
            f"got {len(lambda_label_list)} labels for {size} states"
        )

    overlaps = np.diag(overlap_matrix, k=1)
    pair_indices = np.arange(overlaps.size)
    minimum_index = int(np.argmin(overlaps))

    figure_width = max(10.0, overlaps.size * 0.18)
    fig, ax = plt.subplots(figsize=(figure_width, 5.5))
    ax.bar(pair_indices, overlaps, width=0.75, color="#4C78A8", alpha=0.75)
    ax.plot(
        pair_indices,
        overlaps,
        color="#1F4E79",
        linewidth=1.2,
        marker="o",
        markersize=3,
    )
    ax.scatter(
        minimum_index,
        overlaps[minimum_index],
        color="#D62728",
        s=55,
        zorder=3,
        label=(
            f"Minimum: pair {lambda_label_list[minimum_index]}-"
            f"{lambda_label_list[minimum_index + 1]} "
            f"({overlaps[minimum_index]:.3f})"
        ),
    )

    ax.set_xlabel("Neighbor lambda-state pair (i, i+1)")
    ax.set_ylabel("Overlap")
    ax.set_title("Neighbor Lambda-State Overlaps")
    ax.set_xlim(-0.75, overlaps.size - 0.25)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    tick_step = max(1, int(np.ceil(overlaps.size / 16)))
    tick_indices = pair_indices[::tick_step]
    tick_labels = [
        f"{lambda_label_list[index]}-{lambda_label_list[index + 1]}"
        for index in tick_indices
    ]
    ax.set_xticks(tick_indices, tick_labels, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    return


def plot_stacked_bars(
        figure_path: str,
        exchange_matrix: dict,
        colormap: str = "turbo",
        bar_width: float = 0.5,
        ):
    """
    Plot replica sampling frequencies at each Hamiltonian.

    Replica identifiers are ordered, so colors are sampled from a continuous
    colormap instead of repeating a small categorical palette.
    """
    try:
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap, Normalize
    except ImportError as e:
        raise ImportError("matplotlib is required for this function") from e

    n_rep = exchange_matrix.shape[0]

    data_matrix = np.zeros((n_rep, n_rep))
    for ii in range(n_rep):
        # ii for rank
        for jj in range(n_rep):
            # jj for replica
            num_data = (np.abs(exchange_matrix[ii, :] - jj) < 1e-2).sum()
            data_matrix[jj, ii] = num_data

    data_matrix = data_matrix / data_matrix.sum(axis=1)

    fig = plt.figure(figsize=[4, 4], dpi=300)
    ax = fig.subplots()

    base_colormap = plt.get_cmap(colormap)
    replica_colors = base_colormap(np.linspace(0.02, 0.98, n_rep))
    replica_colormap = ListedColormap(replica_colors)
    ind = np.arange(n_rep)

    for replica_id in range(n_rep):
        ax.bar(
            ind,
            data_matrix[replica_id],
            bar_width,
            bottom=data_matrix[:replica_id].sum(axis=0),
            color=replica_colors[replica_id],
        )

    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Hamiltonian")
    ax.set_title("Replica sampling per hamiltonian")

    norm = Normalize(vmin=0, vmax=max(n_rep - 1, 1))
    tick_step = max(1, int(np.ceil(n_rep / 12)))
    colorbar_ticks = np.arange(0, n_rep, tick_step)
    if colorbar_ticks[-1] != n_rep - 1:
        colorbar_ticks = np.append(colorbar_ticks, n_rep - 1)
    colorbar_handler = plt.colorbar(
        mappable=cm.ScalarMappable(norm=norm, cmap=replica_colormap),
        ticks=colorbar_ticks,
        ax=plt.gca(),
    )
    colorbar_handler.set_label("Replica", rotation=270, labelpad=10)

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close(fig)

    return


def plot_replica_jumping(
        figure_path: str,
        replica_states: np.ndarray,
        n_rows: int = None,
        n_cols: int = None,
        ):
    """Plot replica mixing as a trajectory heatmap and ladder-span summary.

    Args:
        figure_path (str): The path to the output figure.
        replica_states: Array of shape ``(num_replicas, num_frames)`` where
            entry ``[hamiltonian, frame]`` is the continuous-replica label.
        n_rows: Retained for backward compatibility and ignored.
        n_cols: Retained for backward compatibility and ignored.
    """
    del n_rows, n_cols
    replica_states = np.asarray(replica_states)
    if replica_states.ndim != 2:
        raise ValueError(
            "replica_states must be a two-dimensional array; "
            f"got shape {replica_states.shape}"
        )

    num_replicas, num_frames = replica_states.shape
    if num_replicas < 2 or num_frames < 1:
        raise ValueError("replica_states must contain at least two replicas and one frame")
    if replica_states.min() < 0 or replica_states.max() >= num_replicas:
        raise ValueError("replica_states contains an invalid continuous-replica label")

    hamiltonian_of_replica = np.empty_like(replica_states, dtype=int)
    hamiltonian_indices = np.arange(num_replicas)
    for frame in range(num_frames):
        hamiltonian_of_replica[replica_states[:, frame], frame] = hamiltonian_indices

    ladder_min = hamiltonian_of_replica.min(axis=1)
    ladder_max = hamiltonian_of_replica.max(axis=1)
    replica_labels = np.arange(num_replicas)
    tick_step = max(1, int(np.ceil(num_replicas / 12)))
    replica_ticks = np.arange(0, num_replicas, tick_step)
    if replica_ticks[-1] != num_replicas - 1:
        replica_ticks = np.append(replica_ticks, num_replicas - 1)

    figure, (trajectory_axis, span_axis) = plt.subplots(
        nrows=2,
        figsize=(10, 6.5),
        dpi=250,
        layout="constrained",
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.18},
    )
    image = trajectory_axis.imshow(
        hamiltonian_of_replica,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="turbo",
        vmin=0,
        vmax=num_replicas - 1,
    )
    trajectory_axis.set_ylabel("Continuous replica label")
    trajectory_axis.set_yticks(replica_ticks)
    trajectory_axis.set_title("Replica trajectories across the Hamiltonian ladder")
    colorbar = figure.colorbar(image, ax=trajectory_axis, pad=0.015)
    colorbar.set_label("Hamiltonian index")

    span_axis.vlines(
        replica_labels,
        ladder_min,
        ladder_max,
        color="#4C78A8",
        linewidth=1.2,
    )
    span_axis.scatter(replica_labels, ladder_min, color="#4C78A8", s=8)
    span_axis.scatter(replica_labels, ladder_max, color="#F58518", s=8)
    span_axis.set_xlim(-0.5, num_replicas - 0.5)
    span_axis.set_ylim(-1, num_replicas)
    span_axis.set_xticks(replica_ticks)
    span_axis.set_yticks(replica_ticks)
    span_axis.set_xlabel("Continuous replica label")
    span_axis.set_ylabel("Visited\nHamiltonian")
    span_axis.set_title("Hamiltonian span visited by each continuous replica")

    figure.savefig(figure_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    return

def plot_dG_vs_replica_pairs(
        figure_path: str,
        dG_vs_replica_pairs: dict,
        num_bar_per_row: int = 8,
        orientation: str = "portrait",
        beta: float = None,
        neighbor_overlaps: np.ndarray = None,
        ):
    """Plot pairwise BAR free-energy increments and their uncertainties.

    Args:
        figure_path: Output figure path.
        dG_vs_replica_pairs: BAR results containing ``Delta_f`` and
            ``dDelta_f`` for every neighbor pair.
        num_bar_per_row: Retained for backward compatibility and ignored.
        orientation: Retained for backward compatibility and ignored.
        beta: Inverse thermal energy in ``1 / (kJ/mol)``. When supplied,
            values are converted from reduced units to kJ/mol.
        neighbor_overlaps: Optional overlap values used to annotate the
            weakest-overlap pair.
    """
    del num_bar_per_row, orientation
    d_g = np.asarray(dG_vs_replica_pairs["Delta_f"], dtype=float)
    d_g_error = np.asarray(dG_vs_replica_pairs["dDelta_f"], dtype=float)
    if d_g.ndim != 1 or d_g_error.shape != d_g.shape or d_g.size == 0:
        raise ValueError(
            "dG_vs_replica_pairs must contain equally sized, nonempty "
            "one-dimensional Delta_f and dDelta_f arrays"
        )
    if beta is not None:
        if beta <= 0:
            raise ValueError("beta must be positive")
        d_g = d_g / beta
        d_g_error = d_g_error / beta

    pair_indices = np.arange(d_g.size)
    maximum_d_g_index = int(np.argmax(np.abs(d_g)))
    colors = np.where(d_g >= 0, "#4C78A8", "#E45756")

    figure_width = max(10.0, d_g.size * 0.18)
    figure, axis = plt.subplots(figsize=(figure_width, 5.5), dpi=200)
    axis.errorbar(
        pair_indices,
        d_g,
        yerr=d_g_error,
        fmt="none",
        ecolor="#333333",
        elinewidth=0.8,
        capsize=2,
        zorder=1,
    )
    axis.plot(pair_indices, d_g, color="#555555", linewidth=0.8, zorder=2)
    axis.scatter(pair_indices, d_g, color=colors, s=20, zorder=3)
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.scatter(
        maximum_d_g_index,
        d_g[maximum_d_g_index],
        color="#D62728",
        edgecolor="white",
        linewidth=0.8,
        s=65,
        zorder=4,
        label=(
            f"Largest |dG|: {maximum_d_g_index}-{maximum_d_g_index + 1} "
            f"({d_g[maximum_d_g_index]:.2f} kJ/mol)"
        ),
    )

    if neighbor_overlaps is not None:
        neighbor_overlaps = np.asarray(neighbor_overlaps, dtype=float)
        if neighbor_overlaps.shape != d_g.shape:
            raise ValueError(
                "neighbor_overlaps must have one value per neighbor pair; "
                f"got shape {neighbor_overlaps.shape}, expected {d_g.shape}"
            )
        weakest_overlap_index = int(np.argmin(neighbor_overlaps))
        axis.axvline(
            weakest_overlap_index,
            color="#F58518",
            linestyle="--",
            linewidth=1.0,
            zorder=0,
            label=(
                f"Weakest overlap: {weakest_overlap_index}-"
                f"{weakest_overlap_index + 1} "
                f"({neighbor_overlaps[weakest_overlap_index]:.3f})"
            ),
        )

    tick_step = max(1, int(np.ceil(d_g.size / 16)))
    tick_indices = pair_indices[::tick_step]
    axis.set_xticks(tick_indices)
    axis.set_xticklabels(
        [f"{index}-{index + 1}" for index in tick_indices],
        rotation=45,
        ha="right",
    )
    axis.set_xlim(-0.75, d_g.size - 0.25)
    axis.set_xlabel("Neighbor lambda-state pair (i, i+1)")
    unit_label = "kJ/mol" if beta is not None else "reduced units"
    axis.set_ylabel(f"BAR dG ({unit_label})")
    axis.set_title("Pairwise free-energy increments")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figure_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    return



def plot_flow_profile(
        figure_path: str,
        flow_profile: np.ndarray,
        flow_drops: np.ndarray = None,
        ):
    """Plot the Trebst-Katzgraber replica-flow profile ``f(H)``.

    ``f(H)`` is the fraction of visits to Hamiltonian ``H`` made by replicas
    that most recently touched the bottom endpoint (``up`` walkers). For an
    ideal random walk it decreases linearly from 1 at ``H=0`` to 0 at the top
    Hamiltonian; a flat region marks a transport bottleneck where replicas
    pile up. When ``flow_drops`` (the per-pair drop ``f(i) - f(i+1)``) is
    given, the bottleneck boundary with the largest drop is highlighted.

    Args:
        figure_path (str): The path to the output figure.
        flow_profile (np.ndarray): Length-``N`` array of ``f(H)``.
        flow_drops (np.ndarray): Optional length ``N-1`` per-pair flow drops.
    """
    flow_profile = np.asarray(flow_profile, dtype=np.float64)
    n = len(flow_profile)
    hamiltonian = np.arange(n)

    fig, ax = plt.subplots(figsize=[6, 4], dpi=300)
    ax.plot(hamiltonian, flow_profile, "o-", ms=3, color="C0",
            label="measured flow f(H)")
    ax.plot([0, n - 1], [1.0, 0.0], "k--", alpha=0.6, label="ideal linear")

    if flow_drops is not None:
        flow_drops = np.asarray(flow_drops, dtype=np.float64)
        if len(flow_drops) == n - 1 and len(flow_drops) > 0:
            worst = int(np.argmax(flow_drops))
            ax.axvspan(worst, worst + 1, color="red", alpha=0.2,
                       label=f"max drop {worst}|{worst + 1}")

    ax.set_xlabel("Hamiltonian")
    ax.set_ylabel("up-walker fraction f(H)")
    ax.set_title("Replica-flow profile (steep slope = bottleneck)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)

    return


def plot_ladder_span(
        figure_path: str,
        replica_states: np.ndarray,
        ):
    """Plot the Hamiltonian range each continuous replica explored.

    For every continuous replica (initial conformation) the minimum and
    maximum Hamiltonian index visited over the run are drawn as a vertical
    span. Replicas spanning the full ladder (bottom to top) indicate good
    transport; short spans flag replicas trapped in a sub-region.

    Args:
        figure_path (str): The path to the output figure.
        replica_states (np.ndarray): Array of shape ``(num_replicas,
            num_frames)`` as returned by ``get_exchange_matrix_from_records``;
            entry ``[k, t]`` is the continuous-replica label occupying
            Hamiltonian ``k`` at frame ``t``.
    """
    replica_states = np.asarray(replica_states)
    n = replica_states.shape[0]
    num_frames = replica_states.shape[1]

    # Invert position -> label into label -> Hamiltonian trajectory.
    ham_of_label = np.empty((n, num_frames), dtype=int)
    for t in range(num_frames):
        ham_of_label[replica_states[:, t], t] = np.arange(n)

    lo = ham_of_label.min(axis=1)
    hi = ham_of_label.max(axis=1)
    labels = np.arange(n)

    fig, ax = plt.subplots(figsize=[6, 4], dpi=300)
    ax.fill_between(labels, lo, hi, step="mid", alpha=0.4, color="C0")
    ax.plot(labels, lo, ".", ms=3, color="C0")
    ax.plot(labels, hi, ".", ms=3, color="C1")
    ax.set_xlabel("continuous replica label")
    ax.set_ylabel("Hamiltonian range visited")
    ax.set_title("Ladder span per replica (full span = good transport)")
    ax.set_ylim(-1, n)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)

    return
