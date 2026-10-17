# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Neighbor-exchange math for HREMD.

Pure functions that decide which replicas attempt to swap each chunk and
whether a proposed neighbor swap is accepted. No OpenMM, IPC, or IO state.
"""

import numpy as np


def generate_exchange_groups(chunk_id: int, n_replicas: int, strategy: str = "neighbor") -> list[set[int]]:
    """Generate exchange groups for a given chunk (only neighbor strategy supported)."""
    if n_replicas <= 0:
        return []

    if strategy != "neighbor":
        raise ValueError(f"Only 'neighbor' exchange strategy is supported, got: {strategy}")

    grps: list[set[int]] = []
    if chunk_id % 2 == 0:
        start_pair = 0
    else:
        grps.append(set([0]))
        start_pair = 1

    for i in range(start_pair, n_replicas, 2):
        if i + 1 < n_replicas:
            grps.append(set([i, i + 1]))
        else:
            grps.append(set([i]))

    return grps


def metropolis_accept(e_self_1: float, e_self_2: float, e_cross_12: float, e_cross_21: float, beta: float) -> bool:
    """Metropolis acceptance criterion for neighbor exchange."""
    e_exch = e_cross_12 + e_cross_21
    e_orig = e_self_1 + e_self_2

    if e_exch < e_orig:
        return True

    accept_prob = np.exp(-beta * (e_exch - e_orig))
    return np.random.rand() < accept_prob


def _require_neighbor_strategy(exchange_strategy: str) -> None:
    if exchange_strategy != "neighbor":
        raise ValueError(
            "Only 'neighbor' exchange strategy is supported, "
            f"got: {exchange_strategy}"
        )


def _map_exchange_groups_to_ordered_workers(
    chunk_id: int,
    ordered_workers: list[int],
    strategy: str = "neighbor",
) -> list[set[int]]:
    """Map exchange-group indices onto an explicit worker ordering."""
    index_groups = generate_exchange_groups(chunk_id, len(ordered_workers), strategy)
    return [{ordered_workers[i] for i in group} for group in index_groups]


def _build_energy_matrix(
    snapshot_by_worker: dict[int, dict],
    ordered_workers: list[int],
) -> dict[tuple[int, int], float]:
    energy_entries: dict[tuple[int, int], float] = {}
    for worker in ordered_workers:
        replica_id = int(snapshot_by_worker[worker]["replica_id"])
        energy_entries[(replica_id, replica_id)] = snapshot_by_worker[worker][
            "self_energy"
        ]
        if replica_id > 0:
            energy_entries[(replica_id - 1, replica_id)] = snapshot_by_worker[
                worker
            ]["left_energy"]
        if replica_id + 1 < len(ordered_workers):
            energy_entries[(replica_id + 1, replica_id)] = snapshot_by_worker[
                worker
            ]["right_energy"]
    return energy_entries
