# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Hamiltonian Replica Exchange Molecular Dynamics (HREMD), neighbor exchange.

Architecture
------------
``run_hremd`` spawns ``N`` worker processes (one OpenMM ``Context`` each, pinned
to GPUs) plus one coordinator process. All ``N + 1`` processes are synchronized
by a single :class:`multiprocessing.Barrier`; there is no polling loop and no
multi-state handshake.

Each chunk (``exchange_interval`` MD steps) proceeds as a strict barrier:

1. Workers run MD and evaluate their self/left/right neighbor energies into a
   shared array; trajectory frames are buffered worker-locally.
2. Barrier B1: energies ready.
3. Coordinator orders workers by Hamiltonian, forms this chunk's neighbor
   exchange groups, runs the Metropolis test using only the already-computed
   self/left/right energies (no partner round-trip), and writes each worker's
   post-exchange ``new_replica_id`` / ``accepted`` into the shared array. It
   also buffers one energy-matrix record and the exchange rows for the
   ``records.sqlite`` store.
4. Barrier B2: decisions ready.
5. Workers rebind to their post-exchange Hamiltonian.
6. On checkpoint chunks only (identical predicate on every process), a
   three-barrier handshake flushes buffered trajectory frames, the energy
   matrix, and the exchange log to disk, then workers commit OpenMM
   generation-specific checkpoints. The coordinator then publishes the common
   generation and its DCD extents in one SQLite transaction. SQLite is the
   commit authority; any external tail beyond it is ignored or truncated.

Safety: every ``barrier.wait`` carries a timeout, and any exception (in a
worker, the coordinator, or the parent) calls ``barrier.abort()`` so no process
hangs if a peer dies (e.g. a segfault/OOM inside OpenMM).
"""

import ctypes as C
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from multiprocessing.connection import wait
from pathlib import Path
from threading import BrokenBarrierError
from typing import Optional

import numpy as np
from openmm import unit
import openmm as mm

from .replica import Replica
from .hremd_exchange import (
    _build_energy_matrix,
    _map_exchange_groups_to_ordered_workers,
    _require_neighbor_strategy,
    metropolis_accept,
)
from .hremd_io import (
    _activate_replica_output_targets,
    _build_run_manifest,
    _buffer_chunk_record,
    _checkpoint_path,
    _close_logger_pool,
    _close_master_traj_pool,
    _commit_checkpoint_generation,
    _discover_resume_state,
    _drain_traj_queue_exact,
    _flush_energy_buffer,
    _flush_exchange_buffer,
    _flush_traj_pool,
    _format_progress_message,
    _format_timing_summary,
    _get_integrator_timestep_ps,
    _master_profile_path,
    _open_logger_pool,
    _open_master_traj_pool,
    _open_records_db,
    _prepare_run_manifest,
    _records_db_path,
    _reconcile_trajectories,
    _remove_records_db_files,
    _reset_records_after,
    _trajectory_frame_counts,
    _write_records_metadata,
    _trajectory_path,
    _validate_intervals,
)

# Re-exported for backward compatibility: felis.protocols.abfe_v2.utilities
# imports this energy-matrix helper from this module.
from .hremd_io import _deserialize_energy_matrix_entries  # noqa: F401

logger = logging.getLogger(__name__)

SYNC_WAIT_TIMEOUT_SECONDS = 3600.0
_PROCESS_TERMINATE_TIMEOUT_SECONDS = 5.0
_PROCESS_KILL_TIMEOUT_SECONDS = 5.0
_LOCAL_TMP_ROOT = Path("/tmp").resolve()


def _get_fork_context():
    """Return the POSIX fork context required by the HREMD driver."""
    if os.name != "posix" or "fork" not in mp.get_all_start_methods():
        raise RuntimeError(
            "HREMD requires a POSIX platform with the multiprocessing 'fork' "
            "start method; OpenMM Platform objects cannot be pickled for "
            "'spawn' or 'forkserver'."
        )
    try:
        return mp.get_context("fork")
    except ValueError as exc:
        raise RuntimeError(
            "HREMD could not initialize the required multiprocessing 'fork' "
            "context."
        ) from exc


def _create_cuda_temp_directories(
    tmp_directory: str,
    n_replicas: int,
) -> tuple[Path, list[Path]]:
    """Create one local run root and an isolated CUDA directory per rank."""
    configured_base = Path(tmp_directory).expanduser()
    if not configured_base.is_absolute():
        configured_base = _LOCAL_TMP_ROOT / configured_base
    configured_base = configured_base.resolve()
    try:
        configured_base.relative_to(_LOCAL_TMP_ROOT)
    except ValueError as exc:
        raise ValueError(
            "HREMD OpenMM temporary files must be stored under local /tmp; "
            f"got {tmp_directory!r}"
        ) from exc

    configured_base.mkdir(parents=True, exist_ok=True)
    run_root = Path(
        tempfile.mkdtemp(prefix="felis-hremd-", dir=str(configured_base))
    )
    try:
        rank_directories = []
        for rank in range(n_replicas):
            rank_directory = run_root / f"rank-{rank:03d}"
            rank_directory.mkdir()
            rank_directories.append(rank_directory)
    except BaseException:
        shutil.rmtree(run_root, ignore_errors=True)
        raise
    return run_root, rank_directories


def _build_status_struct():
    """Build the ctypes Structure for one worker's shared HREMD slot.

    Each field is written by exactly one party per chunk phase, so there is no
    write contention: workers write ``replica_id`` and the three energies before
    barrier B1; the coordinator writes ``new_replica_id`` and ``accepted``
    between B1 and B2.
    """

    class HREMDStatus(C.Structure):
        _fields_ = [
            ("replica_id", C.c_int),
            ("self_energy", C.c_double),
            ("left_energy", C.c_double),
            ("right_energy", C.c_double),
            ("new_replica_id", C.c_int),
            ("accepted", C.c_int),
        ]

    return HREMDStatus


def _barrier_sync(barrier: mp.Barrier, description: str) -> None:
    """Wait on a barrier with a timeout, aborting all parties on failure.

    On timeout or a broken barrier (a peer aborted or died), this aborts the
    barrier so every other party is released promptly, then raises so the
    process exits non-zero and ``run_hremd`` reports the failure.
    """
    try:
        barrier.wait(timeout=SYNC_WAIT_TIMEOUT_SECONDS)
    except BrokenBarrierError as exc:
        barrier.abort()
        raise RuntimeError(
            f"HREMD barrier broken or timed out while waiting for {description}"
        ) from exc


def _should_write_checkpoint(
    chunk_id: int,
    total_chunks: int,
    chunks_per_checkpoint: int,
) -> bool:
    return (
        (chunk_id + 1) % chunks_per_checkpoint == 0
        or (chunk_id + 1) == total_chunks
    )


def _should_log_progress(
    chunk_id: int,
    total_chunks: int,
    chunks_per_logger: int,
) -> bool:
    return (
        chunk_id == 0
        or (chunk_id + 1) % chunks_per_logger == 0
        or (chunk_id + 1) == total_chunks
    )


def _evaluate_neighbor_energies(
    replica: Replica,
    n_replicas: int,
    with_pv: bool,
) -> tuple[int, float, float, float]:
    current_replica_id = replica.replica_id
    self_energy = float(
        replica.evaluate_energy(with_pv=with_pv).value_in_unit(
            unit.kilojoule_per_mole
        )
    )
    left_energy = float("nan")
    right_energy = float("nan")
    if current_replica_id > 0:
        left_energy = float(
            replica.evaluate_neighbor_energy(
                current_replica_id - 1, with_pv=with_pv
            ).value_in_unit(unit.kilojoule_per_mole)
        )
    if current_replica_id + 1 < n_replicas:
        right_energy = float(
            replica.evaluate_neighbor_energy(
                current_replica_id + 1, with_pv=with_pv
            ).value_in_unit(unit.kilojoule_per_mole)
        )
    return current_replica_id, self_energy, left_energy, right_energy


def _apply_exchange_result(
    replica: Replica,
    new_replica_id: int,
) -> None:
    """Rebind the worker to its post-exchange Hamiltonian.

    Trajectory output is owned by the coordinator process (handle/buffer pool),
    so a worker only needs to update its replica_id and Hamiltonian parameters.
    """
    if new_replica_id == replica.replica_id:
        return
    replica.update_replica_id(new_replica_id)


def write_trajectory_buffered(
    replica: Replica,
    local_buffer: list,
    chunk_id: int,
) -> None:
    """Capture one frame into the worker-local buffer (no disk write).

    The frame is tagged with the Hamiltonian id the worker currently holds so
    the coordinator can file it under the correct ``replica_{id}/traj.dcd`` even
    after exchanges move the Hamiltonian to another worker. Must be called
    before the chunk's exchange handoff.
    """
    state = replica.context.getState(getPositions=True, enforcePeriodicBox=True)
    positions = np.asarray(
        state.getPositions().value_in_unit(unit.nanometer), dtype=np.float32
    )
    box = np.asarray(
        [
            list(vec.value_in_unit(unit.nanometer))
            for vec in state.getPeriodicBoxVectors()
        ],
        dtype=np.float32,
    )
    local_buffer.append((replica.replica_id, chunk_id, positions, box))


def _drain_local_traj_buffer(traj_queue, local_buffer: list) -> None:
    """Send all locally-buffered frames to the coordinator, then clear them."""
    for replica_id, chunk_id, positions, box in local_buffer:
        traj_queue.put((replica_id, chunk_id, positions, box))
    local_buffer.clear()


def _decide_exchanges(
    chunk_id: int,
    n_replicas: int,
    shared_status,
    beta: float,
    exchange_strategy: str,
    steps_per_chunk: int,
) -> tuple[dict, list[tuple]]:
    """Run this chunk's neighbor Metropolis tests from the shared energies.

    Returns ``(chunk_state, exchange_rows)`` where ``chunk_state`` is the dict
    consumed by :func:`_buffer_chunk_record` and ``exchange_rows`` are the
    per-swap-side rows ``(chunk_id, step, worker, replica_id,
    replica_try2exchange, accept)`` for the ``exchanges`` table. Also writes each
    worker's ``new_replica_id`` / ``accepted`` into the shared array so workers
    can rebind after barrier B2.
    """
    snapshot_by_worker = {
        worker: {
            "replica_id": int(shared_status[worker].replica_id),
            "self_energy": float(shared_status[worker].self_energy),
            "left_energy": float(shared_status[worker].left_energy),
            "right_energy": float(shared_status[worker].right_energy),
        }
        for worker in range(n_replicas)
    }
    ordered_workers = sorted(
        range(n_replicas),
        key=lambda worker: snapshot_by_worker[worker]["replica_id"],
    )
    energy_entries = _build_energy_matrix(snapshot_by_worker, ordered_workers)
    exchange_groups = _map_exchange_groups_to_ordered_workers(
        chunk_id, ordered_workers, exchange_strategy
    )

    new_replica_ids = {
        worker: snapshot_by_worker[worker]["replica_id"]
        for worker in range(n_replicas)
    }
    accepted = {worker: False for worker in range(n_replicas)}
    exchange_rows: list[tuple] = []
    step = (chunk_id + 1) * steps_per_chunk

    for group in exchange_groups:
        if len(group) != 2:
            continue
        # Order the pair by Hamiltonian so w_lo holds replica i and w_hi holds
        # i+1; then w_lo's "right" and w_hi's "left" neighbor energies are
        # exactly the cross terms needed by the Metropolis test.
        w_lo, w_hi = sorted(
            group, key=lambda worker: snapshot_by_worker[worker]["replica_id"]
        )
        rid_lo = snapshot_by_worker[w_lo]["replica_id"]
        rid_hi = snapshot_by_worker[w_hi]["replica_id"]
        accept = metropolis_accept(
            snapshot_by_worker[w_lo]["self_energy"],
            snapshot_by_worker[w_hi]["self_energy"],
            snapshot_by_worker[w_lo]["right_energy"],
            snapshot_by_worker[w_hi]["left_energy"],
            beta,
        )
        if accept:
            new_replica_ids[w_lo] = rid_hi
            new_replica_ids[w_hi] = rid_lo
        accepted[w_lo] = accept
        accepted[w_hi] = accept

        exchange_rows.append(
            (chunk_id, step, w_lo, rid_lo, rid_hi, int(accept))
        )
        exchange_rows.append(
            (chunk_id, step, w_hi, rid_hi, rid_lo, int(accept))
        )

    for worker in range(n_replicas):
        shared_status[worker].new_replica_id = int(new_replica_ids[worker])
        shared_status[worker].accepted = 1 if accepted[worker] else 0

    chunk_state = {
        "worker_ids": ordered_workers,
        "old_replica_ids": list(range(n_replicas)),
        "new_replica_ids": new_replica_ids,
        "accepted": accepted,
        "energy_entries": energy_entries,
    }
    return chunk_state, exchange_rows


def _coordinator_loop(
    n_replicas: int,
    shared_status,
    barrier: mp.Barrier,
    exchange_strategy: str,
    beta: float,
    total_chunks: int,
    task_dir: Path,
    steps_per_chunk: int,
    chunks_per_checkpoint: int,
    chunks_per_traj: int,
    traj_queue,
    topology,
    dt_ps: float,
    resume: bool,
    resumed_chunk_id: int = 0,
    enable_profiling: bool = False,
) -> None:
    """Coordinator process: decide neighbor exchanges and own buffered output.

    Owns the per-Hamiltonian DCD writer pool plus the energy-matrix and
    exchange-log buffers (flushed to ``records.sqlite``). Trajectory frames
    (received from workers over ``traj_queue``) and energy records are buffered
    in memory and flushed to disk only at checkpoint-chunk boundaries.
    """
    _require_neighbor_strategy(exchange_strategy)
    records_db = _open_records_db(_records_db_path(task_dir))
    _write_records_metadata(
        records_db,
        {
            "dt_ps": dt_ps,
            "steps_per_chunk": steps_per_chunk,
            "exchange_strategy": exchange_strategy,
        },
    )
    # Align the records store with the selected starting point. A genuine
    # resume preserves the committed prefix, while a fresh run (chunk zero)
    # removes every record left by an earlier use of the task directory.
    _reset_records_after(records_db, resumed_chunk_id)

    traj_handle_pool = _open_master_traj_pool(
        task_dir,
        n_replicas,
        topology,
        dt_ps,
        traj_interval=chunks_per_traj * steps_per_chunk,
        append=(resume and resumed_chunk_id > 0),
    )
    traj_buffer_pool: dict[int, list] = defaultdict(list)
    energy_record_buffer: list[dict] = []
    exchange_row_buffer: list[tuple] = []
    frames_since_checkpoint = 0

    master_profile_log = None
    master_timings: dict[str, float] = defaultdict(float)
    master_start_time = None
    if enable_profiling:
        master_profile_log = _master_profile_path(task_dir).open("w")
        master_start_time = time.perf_counter()

    try:
        for chunk_id in range(resumed_chunk_id, total_chunks):
            _barrier_sync(barrier, f"energies of chunk {chunk_id}")  # B1

            t0 = time.perf_counter() if enable_profiling else None
            chunk_state, exchange_rows = _decide_exchanges(
                chunk_id,
                n_replicas,
                shared_status,
                beta,
                exchange_strategy,
                steps_per_chunk,
            )
            _buffer_chunk_record(
                energy_record_buffer,
                record_id=chunk_id,
                chunk_id=chunk_id,
                steps_per_chunk=steps_per_chunk,
                beta=beta,
                chunk_state=chunk_state,
            )
            exchange_row_buffer.extend(exchange_rows)
            if (chunk_id + 1) % chunks_per_traj == 0:
                frames_since_checkpoint += n_replicas
            if enable_profiling:
                master_timings["exchange_decision"] += time.perf_counter() - t0

            _barrier_sync(barrier, f"decisions of chunk {chunk_id}")  # B2

            if _should_write_checkpoint(
                chunk_id, total_chunks, chunks_per_checkpoint
            ):
                _barrier_sync(barrier, f"checkpoint frames of chunk {chunk_id}")  # B3
                t0 = time.perf_counter() if enable_profiling else None
                _drain_traj_queue_exact(
                    traj_queue,
                    traj_buffer_pool,
                    expected_frames=frames_since_checkpoint,
                    timeout_seconds=SYNC_WAIT_TIMEOUT_SECONDS,
                )
                _flush_traj_pool(traj_handle_pool, traj_buffer_pool)
                frames_since_checkpoint = 0
                if enable_profiling:
                    master_timings["checkpoint_flush"] += time.perf_counter() - t0
                _barrier_sync(barrier, f"persistence flush of chunk {chunk_id}")  # B4
                # Workers atomically publish generation-specific checkpoint
                # binaries before crossing B5. Only after every worker has
                # crossed do we publish records, checkpoint metadata, and DCD
                # extents together in one SQLite transaction.
                _barrier_sync(barrier, f"checkpoint files of chunk {chunk_id}")  # B5
                with records_db:
                    _flush_energy_buffer(
                        records_db,
                        energy_record_buffer,
                        commit=False,
                        clear=False,
                    )
                    _flush_exchange_buffer(
                        records_db,
                        exchange_row_buffer,
                        commit=False,
                        clear=False,
                    )
                    _commit_checkpoint_generation(
                        records_db,
                        task_dir,
                        generation_id=chunk_id,
                        step_count=(chunk_id + 1) * steps_per_chunk,
                        n_replicas=n_replicas,
                        dcd_frames=_trajectory_frame_counts(traj_handle_pool),
                        commit=False,
                    )
                energy_record_buffer.clear()
                exchange_row_buffer.clear()
    except BaseException:
        barrier.abort()
        logger.exception("Fatal error in HREMD coordinator")
        raise
    finally:
        if enable_profiling and master_profile_log is not None:
            master_total_wall = time.perf_counter() - master_start_time
            master_profile_log.write(
                _format_timing_summary(
                    "master", dict(master_timings), master_total_wall
                )
                + "\n"
            )
            master_profile_log.close()
        _close_master_traj_pool(traj_handle_pool)
        records_db.close()


def _worker_loop(
    rank: int,
    n_replicas: int,
    replica: Replica,
    shared_status,
    barrier: mp.Barrier,
    total_chunks: int,
    steps_per_chunk: int,
    chunks_per_logger: int,
    chunks_per_traj: int,
    chunks_per_checkpoint: int,
    task_dir: Path,
    with_pv: bool,
    exchange_strategy: str,
    traj_queue,
    resume_checkpoint_paths: Optional[dict[int, str]] = None,
    resume: bool = False,
    resumed_chunk_id: int = 0,
    enable_profiling: bool = False,
) -> None:
    """Worker process: run MD and coordinate with the coordinator via barriers."""
    _require_neighbor_strategy(exchange_strategy)
    logger_pool = _open_logger_pool(task_dir, rank, n_replicas)
    _activate_replica_output_targets(replica, replica.replica_id, logger_pool)
    local_traj_buffer: list = []
    if resume and resumed_chunk_id > 0:
        if resume_checkpoint_paths is None:
            raise RuntimeError("Resume checkpoint paths were not provided")
        loaded_step = replica.load_checkpoint(
            resume_checkpoint_paths[replica.replica_id]
        )
        if loaded_step != resumed_chunk_id * steps_per_chunk:
            raise RuntimeError(
                f"Rank {rank}: checkpoint step {loaded_step} does not match "
                f"resumed_chunk_id {resumed_chunk_id} "
                f"(steps_per_chunk={steps_per_chunk})"
            )
    shared_status[rank].replica_id = replica.replica_id
    shared_status[rank].new_replica_id = replica.replica_id
    shared_status[rank].accepted = 0

    try:
        replica.replica_logger.info(
            f"Rank {rank}: starting HREMD. total_chunks={total_chunks}, "
            f"steps_per_chunk={steps_per_chunk}, initial_replica_id={replica.replica_id}"
        )
        visited_replica_ids: list[int] = []
        worker_start_time = time.perf_counter()
        worker_timings: dict[str, float] = defaultdict(float)
        timestep_ps = _get_integrator_timestep_ps(replica)
        simulated_ns_per_chunk = None

        if timestep_ps is not None:
            simulated_ns_per_chunk = steps_per_chunk * timestep_ps / 1000.0
            replica.replica_logger.info(
                f"Rank {rank}: timestep={timestep_ps:.6f} ps, "
                f"simulated_time_per_chunk={simulated_ns_per_chunk:.6f} ns"
            )

        for chunk_id in range(resumed_chunk_id, total_chunks):
            t0 = time.perf_counter() if enable_profiling else None
            replica.step(steps_per_chunk)
            if enable_profiling:
                worker_timings["md_step"] += time.perf_counter() - t0

            t0 = time.perf_counter() if enable_profiling else None
            current_replica_id, self_energy, left_energy, right_energy = (
                _evaluate_neighbor_energies(replica, n_replicas, with_pv)
            )
            shared_status[rank].replica_id = current_replica_id
            shared_status[rank].self_energy = self_energy
            shared_status[rank].left_energy = left_energy
            shared_status[rank].right_energy = right_energy
            if enable_profiling:
                worker_timings["energy_eval"] += time.perf_counter() - t0

            if (chunk_id + 1) % chunks_per_traj == 0:
                t0 = time.perf_counter() if enable_profiling else None
                write_trajectory_buffered(replica, local_traj_buffer, chunk_id)
                if enable_profiling:
                    worker_timings["trajectory_write"] += time.perf_counter() - t0

            t0 = time.perf_counter() if enable_profiling else None
            _barrier_sync(barrier, f"energies of chunk {chunk_id}")  # B1
            _barrier_sync(barrier, f"decisions of chunk {chunk_id}")  # B2
            if enable_profiling:
                worker_timings["wait_exchange"] += time.perf_counter() - t0

            next_replica_id = int(shared_status[rank].new_replica_id)
            visited_replica_ids.append(current_replica_id)
            if _should_log_progress(chunk_id, total_chunks, chunks_per_logger):
                elapsed_seconds = time.perf_counter() - worker_start_time
                logger_pool[current_replica_id]["logger"].info(
                    _format_progress_message(
                        rank=rank,
                        chunk_id=chunk_id,
                        total_chunks=total_chunks,
                        steps_per_chunk=steps_per_chunk,
                        replica_id=current_replica_id,
                        simulated_ns_per_chunk=simulated_ns_per_chunk,
                        elapsed_seconds=elapsed_seconds,
                        exchange_counts=Counter(visited_replica_ids),
                        n_replicas=n_replicas,
                        resumed_chunk_id=resumed_chunk_id,
                    )
                )

            t0 = time.perf_counter() if enable_profiling else None
            _apply_exchange_result(replica, next_replica_id)
            _activate_replica_output_targets(
                replica, replica.replica_id, logger_pool
            )
            shared_status[rank].replica_id = replica.replica_id
            if enable_profiling:
                worker_timings["exchange_rebind"] += time.perf_counter() - t0

            if _should_write_checkpoint(
                chunk_id, total_chunks, chunks_per_checkpoint
            ):
                t0 = time.perf_counter() if enable_profiling else None
                # Checkpoint barrier: push buffered frames to the coordinator,
                # wait for it to persist trajectory + energy records, then
                # atomically write this generation's checkpoint binary. The
                # coordinator publishes the common generation in SQLite.
                _drain_local_traj_buffer(traj_queue, local_traj_buffer)
                _barrier_sync(barrier, f"checkpoint frames of chunk {chunk_id}")  # B3
                _barrier_sync(barrier, f"persistence flush of chunk {chunk_id}")  # B4
                step_count = replica.get_step_count()
                expected_step = (chunk_id + 1) * steps_per_chunk
                if step_count != expected_step:
                    raise RuntimeError(
                        f"Rank {rank}: checkpoint step {step_count} does not match "
                        f"expected step {expected_step} for chunk {chunk_id}"
                    )
                replica._checkpoint_path = str(
                    _checkpoint_path(task_dir, replica.replica_id, chunk_id)
                )
                replica.write_checkpoint(
                    chunk_id=chunk_id, step_count=step_count
                )
                _barrier_sync(barrier, f"checkpoint files of chunk {chunk_id}")  # B5
                if enable_profiling:
                    worker_timings["checkpoint_write"] += time.perf_counter() - t0

        if enable_profiling:
            worker_total_wall = time.perf_counter() - worker_start_time
            replica.replica_logger.info(
                _format_timing_summary(
                    f"worker rank {rank}",
                    dict(worker_timings),
                    worker_total_wall,
                )
            )

        replica.replica_logger.info(f"Rank {rank}: HREMD done.")
    finally:
        _close_logger_pool(logger_pool)
        replica.close_checkpoint()


def _run_replica(
    rank: int,
    n_replicas: int,
    replica: Optional[Replica],
    shared_status,
    barrier: mp.Barrier,
    total_chunks: int,
    steps_per_chunk: int,
    chunks_per_logger: int,
    chunks_per_traj: int,
    chunks_per_checkpoint: int,
    task_dir: str,
    exchange_strategy: str,
    beta: float,
    with_pv: bool,
    traj_queue,
    topology,
    dt_ps: float,
    resume_checkpoint_paths: Optional[dict[int, str]] = None,
    resume: bool = False,
    resumed_chunk_id: int = 0,
    enable_profiling: bool = False,
) -> None:
    """Entry point for worker (rank < n_replicas) and coordinator (rank == n_replicas)."""
    try:
        if rank == n_replicas:
            _coordinator_loop(
                n_replicas=n_replicas,
                shared_status=shared_status,
                barrier=barrier,
                exchange_strategy=exchange_strategy,
                beta=beta,
                total_chunks=total_chunks,
                task_dir=Path(task_dir),
                steps_per_chunk=steps_per_chunk,
                chunks_per_checkpoint=chunks_per_checkpoint,
                chunks_per_traj=chunks_per_traj,
                traj_queue=traj_queue,
                topology=topology,
                dt_ps=dt_ps,
                resume=resume,
                resumed_chunk_id=resumed_chunk_id,
                enable_profiling=enable_profiling,
            )
            return

        assert replica is not None, "Worker requires a Replica instance"
        replica.set_num_replicas(n_replicas)
        replica.init_context()
        _worker_loop(
            rank=rank,
            n_replicas=n_replicas,
            replica=replica,
            shared_status=shared_status,
            barrier=barrier,
            total_chunks=total_chunks,
            steps_per_chunk=steps_per_chunk,
            chunks_per_logger=chunks_per_logger,
            chunks_per_traj=chunks_per_traj,
            chunks_per_checkpoint=chunks_per_checkpoint,
            task_dir=Path(task_dir),
            with_pv=with_pv,
            exchange_strategy=exchange_strategy,
            traj_queue=traj_queue,
            resume_checkpoint_paths=resume_checkpoint_paths,
            resume=resume,
            resumed_chunk_id=resumed_chunk_id,
            enable_profiling=enable_profiling,
        )
    except BaseException:
        # Break the barrier so every other party is released promptly even if
        # this process is mid-MD and not currently waiting on the barrier.
        barrier.abort()
        if replica is not None and getattr(replica, "replica_logger", None) is not None:
            replica.replica_logger.exception(
                "Fatal error in worker rank %s", rank
            )
        else:
            logger.exception("Fatal error in process rank %s", rank)
        raise


def _start_and_wait_for_processes(processes: list, barrier) -> None:
    """Start and monitor all children, with bounded cleanup on any failure."""
    started_processes = []

    def _abort_and_stop_started_processes() -> list[str]:
        cleanup_issues = []
        try:
            barrier.abort()
        except BaseException as exc:
            cleanup_issues.append(
                f"barrier abort failed: {type(exc).__name__}: {exc}"
            )

        for rank, process in enumerate(started_processes):
            try:
                if process.is_alive():
                    process.terminate()
            except BaseException as exc:
                cleanup_issues.append(
                    f"rank={rank}, name={process.name}: terminate failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        terminate_deadline = time.monotonic() + _PROCESS_TERMINATE_TIMEOUT_SECONDS
        for rank, process in enumerate(started_processes):
            try:
                process.join(timeout=max(0.0, terminate_deadline - time.monotonic()))
            except BaseException as exc:
                cleanup_issues.append(
                    f"rank={rank}, name={process.name}: join after terminate failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        survivors = []
        for rank, process in enumerate(started_processes):
            try:
                if process.is_alive():
                    survivors.append((rank, process))
            except BaseException as exc:
                cleanup_issues.append(
                    f"rank={rank}, name={process.name}: status check failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        for rank, process in survivors:
            try:
                process.kill()
            except BaseException as exc:
                cleanup_issues.append(
                    f"rank={rank}, name={process.name}: kill failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        kill_deadline = time.monotonic() + _PROCESS_KILL_TIMEOUT_SECONDS
        for rank, process in survivors:
            try:
                process.join(timeout=max(0.0, kill_deadline - time.monotonic()))
            except BaseException as exc:
                cleanup_issues.append(
                    f"rank={rank}, name={process.name}: join after kill failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        for rank, process in survivors:
            try:
                if process.is_alive():
                    cleanup_issues.append(
                        f"rank={rank}, name={process.name}, pid={process.pid}: "
                        "still alive after kill"
                    )
            except BaseException as exc:
                cleanup_issues.append(
                    f"rank={rank}, name={process.name}: final status check failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        return cleanup_issues

    try:
        for process in processes:
            process.start()
            started_processes.append(process)
    except BaseException:
        cleanup_issues = _abort_and_stop_started_processes()
        for issue in cleanup_issues:
            logger.error("HREMD process cleanup issue: %s", issue)
        raise

    process_by_sentinel = {
        process.sentinel: (rank, process)
        for rank, process in enumerate(started_processes)
    }
    failed_processes = []
    try:
        while process_by_sentinel:
            for sentinel in wait(tuple(process_by_sentinel)):
                _rank, process = process_by_sentinel.pop(sentinel)
                process.join(timeout=0)

            failed_processes = [
                f"rank={rank}, name={process.name}, pid={process.pid}, "
                f"exitcode={process.exitcode}"
                for rank, process in enumerate(started_processes)
                if process.exitcode is not None and process.exitcode != 0
            ]
            if failed_processes:
                break
    except BaseException:
        cleanup_issues = _abort_and_stop_started_processes()
        for issue in cleanup_issues:
            logger.error("HREMD process cleanup issue: %s", issue)
        raise

    if failed_processes:
        cleanup_issues = _abort_and_stop_started_processes()
        message = (
            "Fatal error in HREMD run: subprocess exit failure(s): "
            + "; ".join(failed_processes)
        )
        if cleanup_issues:
            message += "; cleanup issue(s): " + "; ".join(cleanup_issues)
        raise RuntimeError(message)


def run_hremd(
    replicas: list[Replica],
    total_steps: int,
    exchange_interval: int, # Energy matrix record interval is the same as exchange interval
    logger_record_interval: int,
    traj_record_interval: int,
    task_dir: str,
    temperature: float,
    gpu_devices: list[int],
    exchange_strategy: str = "neighbor",
    with_pv: bool = True,
    checkpoint_interval: Optional[int] = None,
    resume: bool = True,
    enable_profiling: bool = False,
    tmp_directory: str = "/tmp", # Use /tmp as temp directory for OpenMM Cuda kernel compilation.
) -> None:
    """Run Hamiltonian Replica Exchange Molecular Dynamics (neighbor exchange only)."""
    n_replicas = len(replicas)
    if n_replicas == 0:
        raise ValueError("No replicas provided")
    if checkpoint_interval is None:
        checkpoint_interval = traj_record_interval
    _validate_intervals(
        total_steps,
        exchange_interval,
        traj_record_interval,
        logger_record_interval,
        checkpoint_interval,
    )

    total_chunks = total_steps // exchange_interval
    chunks_per_logger = logger_record_interval // exchange_interval
    chunks_per_traj = traj_record_interval // exchange_interval
    chunks_per_checkpoint = checkpoint_interval // exchange_interval

    kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin
    )
    beta = 1.0 / (kB * temperature)

    fork_context = _get_fork_context()

    # Shared status array for coordinator-worker communication.
    status_struct = _build_status_struct()
    shared_status = fork_context.Array(status_struct, n_replicas)

    task_path = Path(task_dir)
    task_path.mkdir(parents=True, exist_ok=True)
    topology = replicas[0].topology
    dt_ps = _get_integrator_timestep_ps(replicas[0])
    if dt_ps is None or dt_ps <= 0:
        raise ValueError("Cannot determine a positive integrator timestep in ps")

    # records.sqlite is the sole authority for both run identity and committed
    # checkpoint generations. Validate identity before discovering a resume
    # point or returning early for an already-complete run.
    manifest = _build_run_manifest(
        replicas,
        temperature=temperature,
        exchange_strategy=exchange_strategy,
        with_pv=with_pv,
        exchange_interval=exchange_interval,
        traj_record_interval=traj_record_interval,
        checkpoint_interval=checkpoint_interval,
    )
    records_path = _records_db_path(task_path)
    if not resume:
        _remove_records_db_files(records_path)
    records_db = _open_records_db(records_path)
    resume_state = None
    try:
        _prepare_run_manifest(
            records_db,
            task_path,
            manifest,
            resume=resume,
        )
        if resume:
            resume_state = _discover_resume_state(
                records_db,
                task_path,
                n_replicas,
                exchange_interval,
                expected_atoms=topology.getNumAtoms(),
            )
        resumed_chunk_id = (
            0 if resume_state is None else resume_state.resumed_chunk_id
        )
        resume_checkpoint_paths = (
            None
            if resume_state is None
            else {
                replica_id: str(path)
                for replica_id, path in resume_state.checkpoint_paths.items()
            }
        )
        _reset_records_after(records_db, resumed_chunk_id)
        if resume_state is not None:
            _reconcile_trajectories(
                task_path,
                resume_state,
                expected_atoms=topology.getNumAtoms(),
            )
    finally:
        records_db.close()

    if resumed_chunk_id > total_chunks:
        raise RuntimeError(
            "Cannot resume HREMD to a shorter target: committed chunks "
            f"{resumed_chunk_id}, requested chunks {total_chunks}."
        )
    if resumed_chunk_id == total_chunks and resume_state is not None:
        logger.info(
            "HREMD already complete (%s/%s chunks); nothing to do.",
            resumed_chunk_id,
            total_chunks,
        )
        return
    if resume_state is not None:
        logger.info(
            "Resuming HREMD from committed generation %s (chunk %s/%s)",
            resume_state.generation_id,
            resumed_chunk_id,
            total_chunks,
        )

    # Ensure per-replica directories exist (the coordinator owns the trajectory
    # files and opens them when it starts).
    for replica_id in range(n_replicas):
        _trajectory_path(task_path, replica_id).parent.mkdir(
            parents=True, exist_ok=True
        )

    if not gpu_devices:
        raise ValueError("No GPU devices provided")

    run_temp_root, rank_temp_directories = _create_cuda_temp_directories(
        tmp_directory,
        n_replicas,
    )
    traj_queue = None
    try:
        n_gpus = len(gpu_devices)
        n_replicas_per_gpu = (n_replicas + n_gpus - 1) // n_gpus
        platform = mm.Platform.getPlatformByName("CUDA")
        for rank, replica in enumerate(replicas):
            device_id = gpu_devices[rank // n_replicas_per_gpu]
            platform_properties = {
                "DeviceIndex": str(device_id),
                "Precision": "mixed",
                "TempDirectory": str(rank_temp_directories[rank]),
            }
            replica.set_platform(platform, platform_properties)
            logger.info(
                "Assigned GPU device %s to replica %s (replica_id %s), "
                "CUDA TempDirectory=%s",
                device_id,
                rank,
                replica.replica_id,
                rank_temp_directories[rank],
            )

        if n_replicas % n_gpus != 0:
            logger.warning(
                "Number of replicas (%s) is not a multiple of number of GPUs (%s). "
                "Some GPUs will be assigned more replicas than others.",
                n_replicas,
                n_gpus,
            )

        # OpenMM Platform objects are inherited rather than serialized. Every
        # synchronization object and child process must therefore come from the
        # same explicit POSIX fork context.
        barrier = fork_context.Barrier(n_replicas + 1)
        traj_queue = fork_context.Queue()
        processes = []
        for rank in range(n_replicas):
            processes.append(
                fork_context.Process(
                    target=_run_replica,
                    args=(
                        rank,
                        n_replicas,
                        replicas[rank],
                        shared_status,
                        barrier,
                        total_chunks,
                        exchange_interval,
                        chunks_per_logger,
                        chunks_per_traj,
                        chunks_per_checkpoint,
                        task_dir,
                        exchange_strategy,
                        beta,
                        with_pv,
                        traj_queue,
                        topology,
                        dt_ps,
                        resume_checkpoint_paths,
                        resume,
                        resumed_chunk_id,
                        enable_profiling,
                    ),
                )
            )

        processes.append(
            fork_context.Process(
                target=_run_replica,
                args=(
                    n_replicas,
                    n_replicas,
                    None,
                    shared_status,
                    barrier,
                    total_chunks,
                    exchange_interval,
                    chunks_per_logger,
                    chunks_per_traj,
                    chunks_per_checkpoint,
                    task_dir,
                    exchange_strategy,
                    beta,
                    with_pv,
                    traj_queue,
                    topology,
                    dt_ps,
                    resume_checkpoint_paths,
                    resume,
                    resumed_chunk_id,
                    enable_profiling,
                ),
            )
        )
        _start_and_wait_for_processes(processes, barrier)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt: aborting HREMD run")
        raise
    except Exception:
        logger.exception("Fatal error in HREMD run")
        raise
    finally:
        try:
            if traj_queue is not None:
                traj_queue.close()
                traj_queue.join_thread()
        finally:
            shutil.rmtree(run_temp_root, ignore_errors=True)
