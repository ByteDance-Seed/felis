# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import logging
import os
import sqlite3
from felis.utils.bar import bar
from felis.protocols.dyn.hremd import _deserialize_energy_matrix_entries
from felis.protocols.dyn.hremd_exchange import generate_exchange_groups
from felis.protocols.abfe_v2.draw_figures import plot_free_energy
from felis.protocols.abfe_v2.draw_figures import plot_overlap_matrix
from felis.protocols.abfe_v2.draw_figures import plot_stacked_bars
from felis.protocols.abfe_v2.draw_figures import plot_replica_jumping
from felis.protocols.abfe_v2.draw_figures import plot_dG_vs_replica_pairs
from felis.protocols.abfe_v2.draw_figures import plot_flow_profile
from felis.protocols.abfe_v2.draw_figures import plot_ladder_span
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def read_records(
    path: str | Path,
    include_metadata: bool = False,
) -> list[dict] | tuple[list[dict], dict[str, str]]:
    """Read complete chunk records and optional metadata from ``records.sqlite``."""
    records = []
    metadata = {}
    path = Path(path)
    if not path.exists():
        return (records, metadata) if include_metadata else records

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        try:
            cursor = conn.execute(
                """
                SELECT record_id, chunk_id, step, beta, group_size,
                       worker_ids, old_replica_ids, new_replica_ids, accepted,
                       entry_indices, entry_values
                FROM energy_records
                ORDER BY chunk_id
                """
            )
        except sqlite3.OperationalError:
            # No energy_records table (e.g. an empty / unrelated db file).
            return (records, metadata) if include_metadata else records

        for row in cursor:
            (
                record_id,
                chunk_id,
                step,
                beta,
                group_size,
                worker_ids,
                old_replica_ids,
                new_replica_ids,
                accepted,
                entry_indices,
                entry_values,
            ) = row
            entry_indices = np.frombuffer(entry_indices, dtype=np.int64).reshape(-1, 2)
            entry_values = np.frombuffer(entry_values, dtype=np.float64)
            records.append({
                "record_id": int(record_id),
                "chunk_id": int(chunk_id),
                "step": int(step),
                "group_size": int(group_size),
                "beta": float(beta),
                "worker_ids": np.frombuffer(worker_ids, dtype=np.int64),
                "old_replica_ids": np.frombuffer(old_replica_ids, dtype=np.int64),
                "new_replica_ids": np.frombuffer(new_replica_ids, dtype=np.int64),
                "accepted": np.frombuffer(accepted, dtype=np.bool_),
                "energy_entries": _deserialize_energy_matrix_entries(
                    entry_indices, entry_values
                ),
            })
        if include_metadata:
            try:
                metadata = dict(
                    conn.execute("SELECT key, value FROM run_metadata")
                )
            except sqlite3.OperationalError:
                # Databases created before timing metadata was introduced.
                metadata = {}
    finally:
        conn.close()
    return (records, metadata) if include_metadata else records

def get_energy_matrix_from_records(records: list[dict]) -> dict:
    """Get energy matrix from records."""
    if len(records) == 0:
        raise ValueError("records must not be empty")

    num_replicas = len(records[0]["old_replica_ids"])

    u_kn_list = {}
    for ii in range(num_replicas - 1):
        u_kn_list[(ii, ii + 1)] = {"w_F": [], "w_R": []}
    beta = records[0]["beta"]
    for states in u_kn_list.keys():
        first_replica = states[0]
        second_replica = states[1]

        for record in records:
            energy_entries = record["energy_entries"]
            u_kn_list[states]["w_F"].append(
                beta * (
                    energy_entries[(second_replica, first_replica)] - energy_entries[(first_replica, first_replica)]
                )
            )

            u_kn_list[states]["w_R"].append(
                beta * (
                    energy_entries[(first_replica, second_replica)] - energy_entries[(second_replica, second_replica)]
                )
            )

        u_kn_list[states]["w_F"] = np.array(u_kn_list[states]["w_F"])
        u_kn_list[states]["w_R"] = np.array(u_kn_list[states]["w_R"])

    return u_kn_list, beta

def analyze_results(u_kn_list, beta, with_overlap_matrix: bool = True) -> dict:
    num_replicas = len(u_kn_list.keys()) + 1
    if num_replicas <= 1:
        raise ValueError("num_replicas must be at least 2")

    summated_result = {"Delta_f": 0.0, "dDelta_f": 0.0}

    overlap_matrix = np.zeros([num_replicas, num_replicas])
    dG_vs_replica_pairs = {"Delta_f": [], "dDelta_f": []}
    for first_replica in range(num_replicas - 1):
        second_replica = first_replica + 1
        w_F = u_kn_list[(first_replica, second_replica)]["w_F"]
        w_R = u_kn_list[(first_replica, second_replica)]["w_R"]
        if with_overlap_matrix:
            result = bar(w_F, w_R, compute_overlap=True)
        else:
            result = bar(w_F, w_R, compute_overlap=False)

        dG_vs_replica_pairs["Delta_f"].append(result["Delta_f"])
        dG_vs_replica_pairs["dDelta_f"].append(result["dDelta_f"])

        summated_result["Delta_f"] += result["Delta_f"]
        summated_result["dDelta_f"] += result["dDelta_f"] ** 2

        if with_overlap_matrix:
            if "overlap" not in result:
                raise ValueError("result must contain overlap")
            overlap_matrix[first_replica, first_replica] = result["overlap"]["matrix"][0, 0]
            overlap_matrix[first_replica, second_replica] = result["overlap"]["matrix"][0, 1]
            overlap_matrix[second_replica, first_replica] = result["overlap"]["matrix"][0, 1]

    summated_result["Delta_f"] /= beta
    summated_result["dDelta_f"] = np.sqrt(summated_result["dDelta_f"]) / beta

    if with_overlap_matrix:
        for ii in range(num_replicas):
            if ii == 0:
                overlap_matrix[ii, ii] = 1.0 - overlap_matrix[ii, ii + 1]
            elif ii == num_replicas - 1:
                overlap_matrix[ii, ii] = 1.0 - overlap_matrix[ii - 1, ii]
            else:
                overlap_matrix[ii, ii] = 1.0 - overlap_matrix[ii - 1, ii] - overlap_matrix[ii, ii + 1]

    if with_overlap_matrix:
        return summated_result, dG_vs_replica_pairs, overlap_matrix
    else:
        return summated_result, dG_vs_replica_pairs, None

def get_acceptance_ratios_from_records(records: list[dict]) -> np.ndarray:
    """Compute per-neighbor-pair exchange acceptance ratios from records.

    Returns a length ``(num_replicas - 1)`` array where entry ``i`` is the
    fraction of accepted swaps for the neighbor pair ``(i, i + 1)``, computed
    over all chunks in which that pair was actually attempted.

    The attempted pairs for each chunk follow the alternating even/odd
    neighbor pairing produced by
    :func:`felis.protocols.dyn.hremd_exchange.generate_exchange_groups`. Each
    record stores ``accepted`` ordered by replica id (position ``k`` holds the
    accept decision of the replica at Hamiltonian ``k`` for that chunk), so for
    an attempted pair ``{i, i + 1}`` either ``accepted[i]`` or ``accepted[i + 1]``
    gives that pair's accept flag.
    """
    if len(records) == 0:
        raise ValueError("records must not be empty")

    num_replicas = len(records[0]["old_replica_ids"])
    if num_replicas < 2:
        raise ValueError("num_replicas must be at least 2")

    num_pairs = num_replicas - 1
    attempts = np.zeros(num_pairs, dtype=np.int64)
    accepts = np.zeros(num_pairs, dtype=np.int64)

    for record in records:
        accepted = np.asarray(record["accepted"], dtype=bool)
        groups = generate_exchange_groups(record["chunk_id"], num_replicas)
        for group in groups:
            if len(group) != 2:
                continue
            low = min(group)
            attempts[low] += 1
            if accepted[low]:
                accepts[low] += 1

    ratios = np.zeros(num_pairs, dtype=np.float64)
    nonzero = attempts > 0
    ratios[nonzero] = accepts[nonzero] / attempts[nonzero]
    return ratios

def get_exchange_matrix_from_records(records: list[dict]) -> np.ndarray:
    """Get exchange matrix from records."""
    num_records = len(records)
    if num_records == 0:
        raise ValueError("records must not be empty")
    num_replicas = len(records[0]["old_replica_ids"])

    replica_states = [[i for i in range(num_replicas)]]
    for record in records:
        old_replica_states = replica_states[-1]
        worker_ids = np.asarray(record["worker_ids"], dtype=int)
        new_ids_by_worker = np.asarray(record["new_replica_ids"], dtype=int)
        if sorted(worker_ids.tolist()) != list(range(num_replicas)):
            raise ValueError("worker_ids must be a permutation of all worker ranks")
        if sorted(new_ids_by_worker.tolist()) != list(range(num_replicas)):
            raise ValueError("new_replica_ids must be a permutation of all Hamiltonian ids")
        new_replica_states = [None] * num_replicas
        # The arrays are ordered by old Hamiltonian (worker_ids records which
        # physical worker held each one). Move each continuous replica label
        # from its old-Hamiltonian slot to the newly assigned Hamiltonian.
        for old_hamiltonian, new_hamiltonian in enumerate(new_ids_by_worker):
            new_replica_states[new_hamiltonian] = old_replica_states[old_hamiltonian]
        replica_states.append(new_replica_states)

    replica_states = np.asarray(replica_states, dtype=int).T

    return replica_states


def get_flow_profile_from_records(
    records: list[dict], smooth: bool = False
) -> np.ndarray:
    """Compute the Trebst-Katzgraber replica-flow profile ``f(H)``.

    Each continuous replica is labelled by the endpoint Hamiltonian it most
    recently visited: ``up`` after touching Hamiltonian 0 and ``down`` after
    touching Hamiltonian ``N-1``. ``f(H)`` is the fraction of visits to
    Hamiltonian ``H`` made by ``up``-labelled replicas::

        f(H) = n_up(H) / (n_up(H) + n_down(H))

    For an ideal random walk over the ladder ``f`` decreases linearly from 1
    at ``H=0`` to 0 at ``H=N-1``; a flat region marks a transport bottleneck
    where replicas pile up. Returns a length-``N`` array. Hamiltonians that
    were never visited by a labelled replica (no walker had yet touched an
    endpoint) are filled by linear interpolation over the visited points so
    the profile stays usable on short runs.

    When ``smooth`` is True the raw profile is passed through monotone
    (decreasing) isotonic regression. This enforces the physical constraint
    that ``f`` is non-increasing from 1 to 0 and denoises ``df`` for use in
    the feedback-optimized replica placement (Katzgraber-Trebst-Huse-Troyer,
    cond-mat/0602085), analogous to their linear-regression fit of ``df/dT``.
    The unsmoothed profile should still be used for the honest linearity-error
    diagnostic.
    """
    if len(records) == 0:
        raise ValueError("records must not be empty")

    replica_states = get_exchange_matrix_from_records(records)  # (N, T)
    num_replicas, num_chunks = replica_states.shape
    if num_replicas < 2:
        raise ValueError("num_replicas must be at least 2")

    # ham_of_label[label, t] = Hamiltonian index of continuous replica `label`.
    ham_of_label = np.empty((num_replicas, num_chunks), dtype=int)
    for t in range(num_chunks):
        ham_of_label[replica_states[:, t], t] = np.arange(num_replicas)

    n_up = np.zeros(num_replicas, dtype=np.float64)
    n_dn = np.zeros(num_replicas, dtype=np.float64)
    for label in range(num_replicas):
        traj = ham_of_label[label]
        direction = 0  # +1 = last touched bottom (0), -1 = last touched top
        for h in traj:
            if h == 0:
                direction = 1
            elif h == num_replicas - 1:
                direction = -1
            if direction == 1:
                n_up[h] += 1.0
            elif direction == -1:
                n_dn[h] += 1.0

    total = n_up + n_dn
    visited = total > 0
    flow = np.full(num_replicas, np.nan, dtype=np.float64)
    flow[visited] = n_up[visited] / total[visited]

    # Fill unvisited Hamiltonians by linear interpolation over visited points.
    if not visited.all():
        if visited.sum() < 2:
            raise ValueError(
                "flow profile undersampled: fewer than 2 Hamiltonians were "
                "visited by an endpoint-labelled replica"
            )
        idx = np.arange(num_replicas)
        flow = np.interp(idx, idx[visited], flow[visited])

    if smooth:
        from scipy.optimize import isotonic_regression

        flow = isotonic_regression(flow, increasing=False).x

    return flow


def get_flow_drops_from_records(
    records: list[dict], drop_floor: float = 1e-6, smooth: bool = False
) -> np.ndarray:
    """Per-neighbor-pair flow drop ``f(i) - f(i+1)`` from the flow profile.

    Returns a length ``(N-1)`` array where entry ``i`` is the (non-negative)
    drop in the Trebst-Katzgraber flow across boundary ``(i, i+1)``. Larger
    values mark stronger transport bottlenecks. Drops are clipped at
    ``drop_floor`` so the value can be used as a strictly positive
    redistribution metric (analogous to the acceptance ratio).

    When ``smooth`` is True the underlying flow profile is monotone-regularized
    first (see :func:`get_flow_profile_from_records`), which is the recommended
    setting for driving the feedback-optimized replica placement.
    """
    flow = get_flow_profile_from_records(records, smooth=smooth)
    drops = flow[:-1] - flow[1:]
    return np.clip(drops, drop_floor, None)


def get_round_trip_rate_from_records(records: list[dict]) -> float:
    """Round-trip rate: completed ``0 <-> (N-1)`` traversals per chunk.

    This is the quantity the feedback-optimized scheme of
    Katzgraber-Trebst-Huse-Troyer (cond-mat/0602085) ultimately maximizes. A
    round trip is one ``bottom -> top -> bottom`` cycle of a continuous
    replica; the count is summed over all replicas and divided by the number
    of exchange chunks so the rate is comparable across runs of different
    length. Larger is better (faster mixing across the full ladder).
    """
    if len(records) == 0:
        raise ValueError("records must not be empty")

    replica_states = get_exchange_matrix_from_records(records)  # (N, T)
    num_replicas, num_chunks = replica_states.shape
    if num_replicas < 2:
        raise ValueError("num_replicas must be at least 2")

    ham_of_label = np.empty((num_replicas, num_chunks), dtype=int)
    for t in range(num_chunks):
        ham_of_label[replica_states[:, t], t] = np.arange(num_replicas)

    top = num_replicas - 1
    total_trips = 0
    for label in range(num_replicas):
        touched_bottom = False
        reached_top_after_bottom = False
        for h in ham_of_label[label]:
            if h == 0:
                if reached_top_after_bottom:
                    total_trips += 1
                    reached_top_after_bottom = False
                touched_bottom = True
            elif h == top and touched_bottom:
                reached_top_after_bottom = True

    return float(total_trips) / float(num_chunks)


def get_flow_linearity_error(records: list[dict], norm: str = "l2") -> float:
    """Deviation of the measured flow profile from the ideal linear profile.

    The optimal Trebst-Katzgraber flow profile is the straight line
    ``f_ideal(H) = 1 - H / (N - 1)`` (equivalently, all per-pair flow drops
    equal ``1 / (N - 1)``). This returns how far the measured profile is from
    that line, computed on the *raw* (unclipped) flow so that S-curvature and
    any local non-monotonic dips both count against the score:

    * ``norm="l2"`` (default): root-mean-square deviation, robust to a single
      noisy pair on short runs.
    * ``norm="linf"``: maximum absolute deviation, a stricter "every window
      close to the line" criterion.

    The result is bounded in ``[0, 1]``; ``0`` means a perfectly linear flow
    profile (ideal transport).
    """
    flow = get_flow_profile_from_records(records)
    n = len(flow)
    if n < 2:
        raise ValueError("flow profile must have at least 2 Hamiltonians")
    f_ideal = 1.0 - np.arange(n, dtype=np.float64) / (n - 1)
    deviation = flow - f_ideal
    if norm == "l2":
        return float(np.sqrt(np.mean(deviation ** 2)))
    if norm == "linf":
        return float(np.max(np.abs(deviation)))
    raise ValueError(f"norm must be 'l2' or 'linf'; got {norm!r}")

def forward_block_BAR(energy_matrix: dict,  num_blocks: int = 10):
    """
    Forward block BAR
    """
    # The key of energy matrix is (first_states_id, second_states_id), and
    # where first_states_id < second_states_id.
    num_replicas = len(energy_matrix.keys()) + 1
    if num_replicas <= 1:
        raise ValueError("num_replicas must be at least 2")
    num_samples = len(energy_matrix[(0, 1)]["w_F"])

    num_samples_per_block = int(num_samples / num_blocks)
    if num_samples % num_blocks != 0:
        raise ValueError(f"num_samples must be divisible by num_blocks: {num_samples} % {num_blocks} != 0")

    forward_results = []
    for ii in range(num_blocks):
        adjacent_replicas_results = []
        for irep in range(num_replicas - 1):
            states = (irep, irep + 1)
            w_F = energy_matrix[states]["w_F"][: (ii + 1) * num_samples_per_block]
            w_R = energy_matrix[states]["w_R"][: (ii + 1) * num_samples_per_block]
            result = bar(w_F, w_R, compute_overlap=False)
            adjacent_replicas_results.append(result)

        summated_result = {"Delta_f": 0.0, "dDelta_f": 0.0}
        for res in adjacent_replicas_results:
            summated_result["Delta_f"] += res["Delta_f"]
            summated_result["dDelta_f"] += res["dDelta_f"] ** 2
        summated_result["dDelta_f"] = np.sqrt(summated_result["dDelta_f"])

        forward_results.append(summated_result)

    return forward_results

def backward_block_BAR(energy_matrix: dict,  num_blocks: int = 10):
    """
    Backward block BAR
    """
    # The key of energy matrix is (first_states_id, second_states_id), and
    # where first_states_id < second_states_id.
    num_replicas = len(energy_matrix.keys()) + 1
    if num_replicas <= 1:
        raise ValueError("num_replicas must be at least 2")
    num_samples = len(energy_matrix[(0, 1)]["w_F"])

    num_samples_per_block = int(num_samples / num_blocks)
    if num_samples % num_blocks != 0:
        raise ValueError("num_samples must be divisible by num_blocks")

    backward_results = []
    for ii in range(num_blocks):
        adjacent_replicas_results = []
        for irep in range(num_replicas - 1):
            states = (irep, irep + 1)
            w_F = energy_matrix[states]["w_F"][ - (ii + 1) * num_samples_per_block:]
            w_R = energy_matrix[states]["w_R"][ - (ii + 1) * num_samples_per_block:]
            result = bar(w_F, w_R, compute_overlap=False)
            adjacent_replicas_results.append(result)

        summated_result = {"Delta_f": 0.0, "dDelta_f": 0.0}
        for res in adjacent_replicas_results:
            summated_result["Delta_f"] += res["Delta_f"]
            summated_result["dDelta_f"] += res["dDelta_f"] ** 2
        summated_result["dDelta_f"] = np.sqrt(summated_result["dDelta_f"])

        backward_results.append(summated_result)

    return backward_results


def moving_block_BAR(energy_matrix: dict,  num_blocks: int = 10):
    """
    Moving block BAR
    """
    # The key of energy matrix is (first_states_id, second_states_id), and
    # where first_states_id < second_states_id.
    num_replicas = len(energy_matrix.keys()) + 1
    if num_replicas <= 1:
        raise ValueError("num_replicas must be at least 2")
    num_samples = len(energy_matrix[(0, 1)]["w_F"])

    num_samples_per_block = int(num_samples / num_blocks)
    if num_samples % num_blocks != 0:
        raise ValueError("num_samples must be divisible by num_blocks")

    moving_results = []
    for ii in range(num_blocks):
        adjacent_replicas_results = []
        for irep in range(num_replicas - 1):
            states = (irep, irep + 1)
            w_F = energy_matrix[states]["w_F"][ii * num_samples_per_block: (ii + 1) * num_samples_per_block]
            w_R = energy_matrix[states]["w_R"][ii * num_samples_per_block: (ii + 1) * num_samples_per_block]
            result = bar(w_F, w_R, compute_overlap=False)
            adjacent_replicas_results.append(result)

        summated_result = {"Delta_f": 0.0, "dDelta_f": 0.0}
        for res in adjacent_replicas_results:
            summated_result["Delta_f"] += res["Delta_f"]
            summated_result["dDelta_f"] += res["dDelta_f"] ** 2
        summated_result["dDelta_f"] = np.sqrt(summated_result["dDelta_f"])

        moving_results.append(summated_result)

    return moving_results


def get_free_energy_timelines(
    energy_records: list[dict],
    num_blocks: int,
    dt_ps: float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return forward, backward, and moving-block sample coordinates.

    Coordinates are derived from the stored HREMD steps. A persisted
    integrator timestep converts them to ns; older databases safely fall back
    to MD steps rather than guessing a physical time.
    """
    if len(energy_records) == 0:
        raise ValueError("energy_records must not be empty")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")

    sample_steps = np.asarray(
        [record["step"] for record in energy_records],
        dtype=float,
    )
    if len(sample_steps) % num_blocks != 0:
        raise ValueError(
            "number of energy records must be divisible by num_blocks: "
            f"{len(sample_steps)} % {num_blocks} != 0"
        )

    samples_per_block = len(sample_steps) // num_blocks
    block_steps = sample_steps.reshape(num_blocks, samples_per_block)
    forward_steps = block_steps[:, -1]
    backward_steps = block_steps[::-1, 0]
    moving_steps = block_steps.mean(axis=1)

    if dt_ps is None:
        return forward_steps, backward_steps, moving_steps, "MD steps"
    if dt_ps <= 0:
        raise ValueError("dt_ps must be positive")

    conversion = float(dt_ps) / 1000.0
    return (
        forward_steps * conversion,
        backward_steps * conversion,
        moving_steps * conversion,
        "Time (ns)",
    )


def generate_post_analysis_figures(
        energy_records: list,
        analysis_dir: str,
        prefix: str = None,
        metadata: dict[str, str] = None,
        enable_free_energy: bool = True,
        enable_overlap_matrix: bool = True,
        enable_stacked_bars: bool = True,
        enable_replica_jumping: bool = True,
        enable_dG_vs_replica_pairs: bool = True,
        enable_flow_profile: bool = True,
        enable_ladder_span: bool = True,
    ):
    energy_matrix, beta = get_energy_matrix_from_records(energy_records)
    metadata = metadata or {}

    def _path(name: str) -> str:
        if prefix is not None:
            return os.path.join(analysis_dir, f"{prefix}_{name}")
        return os.path.join(analysis_dir, name)

    if enable_free_energy:
        num_blocks = 10
        forward_results = forward_block_BAR(energy_matrix, num_blocks=num_blocks)
        backward_results = backward_block_BAR(energy_matrix, num_blocks=num_blocks)
        moving_results = moving_block_BAR(energy_matrix, num_blocks=num_blocks)
        dt_ps = metadata.get("dt_ps")
        dt_ps = float(dt_ps) if dt_ps is not None else None
        forward_time, backward_time, moving_time, x_label = get_free_energy_timelines(
            energy_records,
            num_blocks=num_blocks,
            dt_ps=dt_ps,
        )
        free_energy_figure = _path("free_energy.png")
        logger.info(f"Saving free_energy figure to {free_energy_figure}")
        plot_free_energy(
            figure_path=free_energy_figure,
            beta=beta,
            forward_time=forward_time,
            backward_time=backward_time,
            moving_time=moving_time,
            x_label=x_label,
            forward_results=forward_results,
            backward_results=backward_results,
            moving_results=moving_results,
        )
    else:
        logger.info("Skip free energy analysis")

    overlap_matrix = None
    dG_vs_replica_pairs = None
    result = None
    if enable_overlap_matrix or enable_dG_vs_replica_pairs:
        result, dG_vs_replica_pairs, overlap_matrix = analyze_results(energy_matrix, beta, with_overlap_matrix=True)

    if enable_overlap_matrix and overlap_matrix is not None:
        overlap_matrix_figure = _path("overlap_matrix.png")
        logger.info(f"Saving overlap_matrix figure to {overlap_matrix_figure}")
        plot_overlap_matrix(
            figure_path=overlap_matrix_figure,
            overlap_matrix=overlap_matrix,
        )
    else:
        logger.info("Skip overlap matrix analysis")

    if enable_dG_vs_replica_pairs and dG_vs_replica_pairs is not None:
        dG_vs_replica_pairs_figure = _path("dG_vs_replica_pairs.png")
        logger.info(f"Saving dG_vs_replica_pairs figure to {dG_vs_replica_pairs_figure}")
        plot_dG_vs_replica_pairs(
            figure_path=dG_vs_replica_pairs_figure,
            dG_vs_replica_pairs=dG_vs_replica_pairs,
            beta=beta,
            neighbor_overlaps=np.diag(overlap_matrix, k=1),
        )
    else:
        logger.info("Skip dG vs replica pairs analysis")

    exchange_matrix = None
    if enable_stacked_bars or enable_replica_jumping or enable_ladder_span:
        exchange_matrix = get_exchange_matrix_from_records(energy_records)

    if enable_stacked_bars and exchange_matrix is not None:
        stacked_bars_figure = _path("stacked_bars.png")
        logger.info(f"Saving stacked_bars figure to {stacked_bars_figure}")
        plot_stacked_bars(
            figure_path=stacked_bars_figure,
            exchange_matrix=exchange_matrix,
        )
    else:
        logger.info("Skip stacked bars analysis")

    if enable_replica_jumping and exchange_matrix is not None:
        replica_jumping_figure = _path("replica_jumping.png")
        logger.info(f"Saving replica_jumping figure to {replica_jumping_figure}")
        plot_replica_jumping(
            figure_path=replica_jumping_figure,
            replica_states=exchange_matrix,
        )
    else:
        logger.info("Skip replica jumping analysis")

    if enable_ladder_span and exchange_matrix is not None:
        ladder_span_figure = _path("ladder_span.png")
        logger.info(f"Saving ladder_span figure to {ladder_span_figure}")
        plot_ladder_span(
            figure_path=ladder_span_figure,
            replica_states=exchange_matrix,
        )
    else:
        logger.info("Skip ladder span analysis")

    if enable_flow_profile:
        flow_profile = get_flow_profile_from_records(energy_records)
        flow_drops = get_flow_drops_from_records(energy_records)
        flow_profile_figure = _path("flow_profile.png")
        logger.info(f"Saving flow_profile figure to {flow_profile_figure}")
        plot_flow_profile(
            figure_path=flow_profile_figure,
            flow_profile=flow_profile,
            flow_drops=flow_drops,
        )
    else:
        logger.info("Skip flow profile analysis")

    if result is not None:
        result_file = _path("free_energy.json")
        logger.info(f"Saving result to {result_file}")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=4)
    else:
        logger.info("No result to save.")

    return
