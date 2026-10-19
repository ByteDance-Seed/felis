# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Pure IO, serialization, path, validation, and formatting helpers for HREMD.

This module holds everything that does not touch the OpenMM ``Replica`` object
or the inter-process synchronization machinery: the SQLite records store
(energy matrix + exchange log), the per-Hamiltonian DCD writer pool, the
per-replica logger/checkpoint pools and path helpers, resume discovery, interval
validation, and human-readable formatting. The HREMD engine (``hremd.py``) and
the exchange math (``hremd_exchange.py``) build on top of these helpers.
"""

import hashlib
import json
import logging
import os
import queue as queue_module
import re
import sqlite3
import struct
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import openmm as mm
from openmm import unit
from openmm.app import DCDFile

from .replica import Replica


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HREMDResumeState:
    """One complete checkpoint generation selected from the SQLite manifest."""

    generation_id: int
    resumed_chunk_id: int
    step_count: int
    checkpoint_paths: dict[int, Path]
    dcd_frames: dict[int, int]


def _get_integrator_timestep_ps(replica: Replica) -> Optional[float]:
    """Return the replica integrator timestep in ps when available."""
    if replica.integrator is None:
        return None

    step_size = replica.integrator.getStepSize()
    try:
        return float(step_size.value_in_unit(unit.picoseconds))
    except AttributeError:
        return float(step_size)


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact ``HH:MM:SS`` string."""
    total_seconds = int(round(max(0.0, seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_progress_message(
    rank: int,
    chunk_id: int,
    total_chunks: int,
    steps_per_chunk: int,
    replica_id: int,
    simulated_ns_per_chunk: Optional[float],
    elapsed_seconds: float,
    exchange_counts: Optional[Counter] = None,
    n_replicas: Optional[int] = None,
    resumed_chunk_id: int = 0,
) -> str:
    """Build a human-readable progress message for the per-replica log."""
    completed_chunks = max(0, chunk_id + 1)
    total_steps_done = completed_chunks * steps_per_chunk
    progress_pct = 0.0
    if total_chunks > 0:
        progress_pct = 100.0 * completed_chunks / total_chunks
    parts = [
        f"Rank {rank}: progress {completed_chunks}/{total_chunks} "
        f"({progress_pct:.1f}%), steps={total_steps_done}, current_replica_id={replica_id}"
    ]

    if simulated_ns_per_chunk is not None:
        simulated_ns = completed_chunks * simulated_ns_per_chunk
        total_simulated_ns = total_chunks * simulated_ns_per_chunk
        parts.append(
            f"simulated={simulated_ns:.6f}/{total_simulated_ns:.6f} ns"
        )
        if elapsed_seconds > 0:
            ns_per_day = simulated_ns * 86400.0 / elapsed_seconds
            parts.append(f"speed={ns_per_day:.3f} ns/day")

    parts.append(f"elapsed={elapsed_seconds:.1f}s")

    # Estimate the remaining wall time from the simulated-time throughput of
    # this session only, so the ETA stays correct after a resume (elapsed time
    # covers only the simulated time produced since resumed_chunk_id, not the
    # simulated time restored from a checkpoint).
    if simulated_ns_per_chunk is not None and elapsed_seconds > 0:
        simulated_ns_this_session = (
            completed_chunks - resumed_chunk_id
        ) * simulated_ns_per_chunk
        remaining_simulated_ns = (total_chunks - completed_chunks) * simulated_ns_per_chunk
        if simulated_ns_this_session > 0 and remaining_simulated_ns > 0:
            seconds_per_ns = elapsed_seconds / simulated_ns_this_session
            eta_seconds = remaining_simulated_ns * seconds_per_ns
            parts.append(f"eta={_format_duration(eta_seconds)} ({eta_seconds:.1f}s)")

    if exchange_counts is not None and n_replicas is not None and exchange_counts:
        total = sum(exchange_counts.values())
        ratios = " ".join(
            f"{i}:{exchange_counts[i] / total:.3f}" for i in range(n_replicas)
        )
        parts.append(f"exchange_ratio={ratios}")

    return " | ".join(parts)


def _format_timing_summary(
    title: str,
    timings: dict[str, float],
    total_wall_seconds: float,
) -> str:
    """Format a compact timing summary sorted by descending cost."""
    lines = [f"{title} timing summary (total_wall={total_wall_seconds:.3f}s)"]
    if total_wall_seconds <= 0:
        return lines[0]

    for name, seconds in sorted(
        timings.items(), key=lambda item: item[1], reverse=True
    ):
        pct = 100.0 * seconds / total_wall_seconds
        lines.append(f"  - {name}: {seconds:.3f}s ({pct:.1f}%)")
    other_seconds = max(0.0, total_wall_seconds - sum(timings.values()))
    other_pct = 100.0 * other_seconds / total_wall_seconds
    lines.append(f"  - other: {other_seconds:.3f}s ({other_pct:.1f}%)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _trajectory_path(task_dir: Path, replica_id: int) -> Path:
    return task_dir / f"replica_{replica_id}" / "traj.dcd"


def _replica_log_path(task_dir: Path, replica_id: int) -> Path:
    return task_dir / f"replica_{replica_id}" / "log.txt"


def _checkpoint_path(
    task_dir: Path, replica_id: int, generation_id: int
) -> Path:
    """Return the generation-specific OpenMM checkpoint path."""
    return (
        task_dir
        / f"replica_{replica_id}"
        / f"checkpoint.chunk_{generation_id:012d}.chk"
    )


def _records_db_path(task_dir: Path) -> Path:
    return task_dir / "records.sqlite"


def _remove_records_db_files(path: Path) -> None:
    """Remove a SQLite database and WAL side files for an explicit fresh run."""
    for candidate in (
        path,
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
        Path(f"{path}-journal"),
    ):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def _master_profile_path(task_dir: Path) -> Path:
    return task_dir / "master_profile.log"


# ---------------------------------------------------------------------------
# Run identity and committed checkpoint generations
# ---------------------------------------------------------------------------
def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _run_signature(manifest: dict) -> str:
    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()


def _serialize_openmm_object(value) -> str:
    """Serialize an OpenMM object for a stable run-identity digest."""
    try:
        return mm.XmlSerializer.serialize(value)
    except Exception:
        if isinstance(value, (mm.System, mm.Integrator)):
            raise
        # Lightweight test doubles are supported without weakening production
        # identity, where System and Integrator must serialize through OpenMM.
        state = getattr(value, "__dict__", {})
        return _canonical_json(
            {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": state,
            }
        )


def _integrator_identity(value) -> str:
    """Hash restart-relevant integrator configuration, excluding initial RNG seed."""
    serialized = _serialize_openmm_object(value)
    # OpenMM checkpoints restore integrator/RNG state. The config default seed
    # is regenerated when a CLI process restarts, so it is intentionally not
    # part of run identity.
    serialized = re.sub(r' randomSeed="[^"]*"', "", serialized)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _array_identity(value, target_unit=None):
    if value is None:
        return None
    if target_unit is not None and hasattr(value, "value_in_unit"):
        value = value.value_in_unit(target_unit)
    array = np.ascontiguousarray(value, dtype=np.float64)
    return {
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _topology_identity(topology) -> dict:
    """Return a compact canonical topology digest and structural counts."""
    if topology is None or not hasattr(topology, "atoms"):
        payload = {
            "type": f"{type(topology).__module__}.{type(topology).__qualname__}"
        }
        return {
            "atom_count": 0,
            "bond_count": 0,
            "sha256": hashlib.sha256(
                _canonical_json(payload).encode("utf-8")
            ).hexdigest(),
        }
    atoms = []
    for atom in topology.atoms():
        residue = atom.residue
        chain = residue.chain
        atoms.append(
            (
                int(atom.index),
                atom.name,
                None if atom.element is None else int(atom.element.atomic_number),
                residue.name,
                str(residue.id),
                str(chain.id),
            )
        )
    bonds = sorted(
        (min(int(a.index), int(b.index)), max(int(a.index), int(b.index)))
        for a, b in topology.bonds()
    )
    payload = {
        "atoms": atoms,
        "bonds": bonds,
        "periodic_box_vectors_nm": _array_identity(
            topology.getPeriodicBoxVectors(), unit.nanometer
        ),
    }
    return {
        "atom_count": len(atoms),
        "bond_count": len(bonds),
        "sha256": hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    }


def _shared_replica_identity(replica) -> dict:
    return {
        "system_xml_sha256": hashlib.sha256(
            _serialize_openmm_object(getattr(replica, "system", None)).encode(
                "utf-8"
            )
        ).hexdigest(),
        "lambda_schedules": getattr(replica, "lambda_schedules", None),
        "is_boresch_restraint": bool(
            getattr(replica, "is_boresch_restraint", False)
        ),
        "initial_positions_nm": _array_identity(
            getattr(replica, "initial_positions", None), unit.nanometer
        ),
        "initial_box_vectors_nm": _array_identity(
            getattr(replica, "initial_box_vectors", None), unit.nanometer
        ),
        "topology": _topology_identity(getattr(replica, "topology", None)),
    }


def _build_run_manifest(
    replicas: list[Replica],
    *,
    temperature: float,
    exchange_strategy: str,
    with_pv: bool,
    exchange_interval: int,
    traj_record_interval: int,
    checkpoint_interval: int,
) -> dict:
    """Build compact immutable identity; total_steps is intentionally absent."""
    first = replicas[0]
    shared = _shared_replica_identity(first)
    for index, replica in enumerate(replicas[1:], start=1):
        candidate = _shared_replica_identity(replica)
        differing = sorted(
            key for key in shared if candidate.get(key) != shared.get(key)
        )
        if differing:
            raise ValueError(
                "HREMD replicas disagree on shared run identity: "
                f"replica_index={index}, replica_id={replica.replica_id}, "
                f"fields={differing}"
            )

    return {
        "format": 2,
        "n_replicas": len(replicas),
        "shared": shared,
        "replicas": [
            {
                "replica_id": int(replica.replica_id),
                "integrator_xml_sha256": _integrator_identity(
                    replica.integrator
                ),
                "temperature": getattr(replica, "temperature", None),
                "pressure": getattr(replica, "pressure", None),
                "velocity_temperature_K": getattr(
                    replica, "velocity_temperature_K", None
                ),
            }
            for replica in replicas
        ],
        "temperature": float(temperature),
        "exchange_strategy": str(exchange_strategy),
        "with_pv": bool(with_pv),
        "exchange_interval": int(exchange_interval),
        "traj_record_interval": int(traj_record_interval),
        "checkpoint_interval": int(checkpoint_interval),
        "integrator_timestep_ps": _get_integrator_timestep_ps(first),
    }


def _manifest_artifacts_exist(task_dir: Path, conn: sqlite3.Connection) -> bool:
    for table in ("energy_records", "exchanges", "checkpoint_generations"):
        row = conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
        if row is not None:
            return True
    for replica_dir in task_dir.glob("replica_*"):
        if any(replica_dir.glob("checkpoint*.chk")):
            return True
        traj_path = replica_dir / "traj.dcd"
        if traj_path.exists() and traj_path.stat().st_size:
            return True
    return False


def _manifest_diff_summary(stored, requested) -> dict:
    if not isinstance(stored, dict) or not isinstance(requested, dict):
        return {"manifest": "corrupt_or_non_object"}
    top_level = sorted(
        key
        for key in set(stored) | set(requested)
        if key not in {"shared", "replicas"}
        and stored.get(key) != requested.get(key)
    )
    stored_shared = stored.get("shared", {})
    requested_shared = requested.get("shared", {})
    shared_fields = sorted(
        key
        for key in set(stored_shared) | set(requested_shared)
        if stored_shared.get(key) != requested_shared.get(key)
    )
    stored_replicas = stored.get("replicas", [])
    requested_replicas = requested.get("replicas", [])
    replica_indices = [
        index
        for index in range(max(len(stored_replicas), len(requested_replicas)))
        if index >= len(stored_replicas)
        or index >= len(requested_replicas)
        or stored_replicas[index] != requested_replicas[index]
    ]
    return {
        "top_level_fields": top_level,
        "shared_fields": shared_fields,
        "replica_indices": replica_indices[:16],
        "replica_difference_count": len(replica_indices),
    }


def _prepare_run_manifest(
    conn: sqlite3.Connection,
    task_dir: Path,
    manifest: dict,
    *,
    resume: bool,
) -> str:
    """Create or validate the singleton run identity in ``records.sqlite``."""
    signature = _run_signature(manifest)
    row = conn.execute(
        "SELECT signature, manifest_json FROM hremd_run_manifest WHERE id = 1"
    ).fetchone()

    if not resume:
        with conn:
            conn.execute("DELETE FROM energy_records")
            conn.execute("DELETE FROM exchanges")
            conn.execute("DELETE FROM run_metadata")
            conn.execute("DELETE FROM checkpoint_artifacts")
            conn.execute("DELETE FROM checkpoint_generations")
            conn.execute("DELETE FROM hremd_run_manifest")
            conn.execute(
                "INSERT INTO hremd_run_manifest (id, format, signature, manifest_json) "
                "VALUES (1, 1, ?, ?)",
                (signature, _canonical_json(manifest)),
            )
        return signature

    if row is None:
        if _manifest_artifacts_exist(task_dir, conn):
            raise RuntimeError(
                "Cannot resume HREMD: existing output has no SQLite run manifest. "
                "Use resume=False or a fresh task directory."
            )
        with conn:
            conn.execute(
                "INSERT INTO hremd_run_manifest (id, format, signature, manifest_json) "
                "VALUES (1, 1, ?, ?)",
                (signature, _canonical_json(manifest)),
            )
        return signature

    stored_signature, stored_json = row
    if stored_signature != signature:
        try:
            stored_manifest = json.loads(stored_json)
        except (TypeError, json.JSONDecodeError):
            stored_manifest = None
        differences = _manifest_diff_summary(stored_manifest, manifest)
        raise RuntimeError(
            "Cannot resume HREMD: run identity does not match records.sqlite. "
            f"stored_signature={stored_signature}, requested_signature={signature}, "
            f"differences={differences}"
        )
    return signature


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _commit_checkpoint_generation(
    conn: sqlite3.Connection,
    task_dir: Path,
    *,
    generation_id: int,
    step_count: int,
    n_replicas: int,
    dcd_frames: dict[int, int],
    commit: bool = True,
) -> None:
    """Transactionally publish a complete common checkpoint generation."""
    artifacts = []
    for replica_id in range(n_replicas):
        checkpoint_path = _checkpoint_path(task_dir, replica_id, generation_id)
        try:
            file_size = checkpoint_path.stat().st_size
        except OSError as exc:
            raise RuntimeError(
                f"Cannot commit checkpoint generation {generation_id}: missing "
                f"checkpoint for replica {replica_id}: {checkpoint_path}"
            ) from exc
        if file_size <= 0:
            raise RuntimeError(
                f"Cannot commit checkpoint generation {generation_id}: empty "
                f"checkpoint for replica {replica_id}: {checkpoint_path}"
            )
        try:
            relative_path = checkpoint_path.relative_to(task_dir)
        except ValueError as exc:  # pragma: no cover - internal path invariant
            raise RuntimeError("Checkpoint path escaped the HREMD task directory") from exc
        artifacts.append(
            (
                generation_id,
                replica_id,
                str(relative_path),
                file_size,
                _file_sha256(checkpoint_path),
                int(dcd_frames[replica_id]),
            )
        )

    completed_chunks = generation_id + 1

    def _write_rows():
        conn.execute(
            "INSERT OR REPLACE INTO checkpoint_generations "
            "(generation_id, completed_chunks, step_count) VALUES (?, ?, ?)",
            (generation_id, completed_chunks, int(step_count)),
        )
        conn.execute(
            "DELETE FROM checkpoint_artifacts WHERE generation_id = ?",
            (generation_id,),
        )
        conn.executemany(
            "INSERT INTO checkpoint_artifacts "
            "(generation_id, replica_id, checkpoint_path, checkpoint_size, "
            "checkpoint_sha256, dcd_frames) VALUES (?, ?, ?, ?, ?, ?)",
            artifacts,
        )

    if commit:
        with conn:
            _write_rows()
    else:
        _write_rows()


def _discover_resume_state(
    conn: sqlite3.Connection,
    task_dir: Path,
    n_replicas: int,
    steps_per_chunk: int,
    expected_atoms: Optional[int] = None,
) -> Optional[HREMDResumeState]:
    """Select the newest complete committed generation, falling back as needed."""
    generations = conn.execute(
        "SELECT generation_id, completed_chunks, step_count "
        "FROM checkpoint_generations ORDER BY generation_id DESC"
    ).fetchall()
    if not generations:
        return None

    invalid = []
    for generation_id, completed_chunks, step_count in generations:
        if (
            completed_chunks != generation_id + 1
            or step_count != completed_chunks * steps_per_chunk
        ):
            invalid.append(f"generation {generation_id}: inconsistent chunk/step metadata")
            continue
        rows = conn.execute(
            "SELECT replica_id, checkpoint_path, checkpoint_size, "
            "checkpoint_sha256, dcd_frames FROM checkpoint_artifacts "
            "WHERE generation_id = ? ORDER BY replica_id",
            (generation_id,),
        ).fetchall()
        if [row[0] for row in rows] != list(range(n_replicas)):
            invalid.append(f"generation {generation_id}: incomplete replica set")
            continue

        checkpoint_paths = {}
        dcd_frames = {}
        complete = True
        for (
            replica_id,
            relative_path,
            expected_size,
            expected_sha256,
            frame_count,
        ) in rows:
            try:
                checkpoint_path = (task_dir / relative_path).resolve()
                checkpoint_path.relative_to(task_dir.resolve())
            except (OSError, ValueError):
                invalid.append(
                    f"generation {generation_id}: checkpoint path escapes task "
                    f"directory for replica {replica_id}: {relative_path}"
                )
                complete = False
                break
            try:
                actual_size = checkpoint_path.stat().st_size
            except OSError:
                actual_size = -1
            actual_sha256 = None
            if actual_size == expected_size and actual_size > 0:
                actual_sha256 = _file_sha256(checkpoint_path)
            if (
                actual_size != expected_size
                or actual_size <= 0
                or actual_sha256 != expected_sha256
            ):
                invalid.append(
                    f"generation {generation_id}: invalid checkpoint for replica "
                    f"{replica_id} (expected size/hash {expected_size}/"
                    f"{expected_sha256}, got {actual_size}/{actual_sha256})"
                )
                complete = False
                break
            checkpoint_paths[replica_id] = checkpoint_path
            dcd_frames[replica_id] = int(frame_count)
        if complete and expected_atoms is not None:
            for replica_id, expected_frames in dcd_frames.items():
                dcd_path = _trajectory_path(task_dir, replica_id)
                if expected_frames == 0 and not dcd_path.exists():
                    continue
                try:
                    layout = _read_dcd_layout(dcd_path, expected_atoms)
                except (OSError, RuntimeError) as exc:
                    invalid.append(
                        f"generation {generation_id}: invalid DCD for replica "
                        f"{replica_id}: {exc}"
                    )
                    complete = False
                    break
                if layout["complete_frames"] < expected_frames:
                    invalid.append(
                        f"generation {generation_id}: DCD for replica {replica_id} "
                        f"has {layout['complete_frames']} complete frame(s), needs "
                        f"{expected_frames}"
                    )
                    complete = False
                    break
        if complete:
            if invalid:
                logger.warning(
                    "Falling back to checkpoint generation %s; newer generation(s) "
                    "were unusable: %s",
                    generation_id,
                    "; ".join(invalid),
                )
            return HREMDResumeState(
                generation_id=int(generation_id),
                resumed_chunk_id=int(completed_chunks),
                step_count=int(step_count),
                checkpoint_paths=checkpoint_paths,
                dcd_frames=dcd_frames,
            )

    raise RuntimeError(
        "Cannot resume HREMD: no complete committed checkpoint generation is "
        "usable: " + "; ".join(invalid)
    )


# ---------------------------------------------------------------------------
# Logger pool (per replica_id / Hamiltonian)
# ---------------------------------------------------------------------------
def _open_logger_pool(task_dir: Path, rank: int, n_replicas: int) -> dict[int, dict]:
    logger_pool: dict[int, dict] = {}
    for replica_id in range(n_replicas):
        log_path = _replica_log_path(task_dir, replica_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger_name = f"felis.hremd.worker.{rank}.replica.{replica_id}"
        replica_logger = logging.getLogger(logger_name)
        replica_logger.setLevel(logging.INFO)
        replica_logger.propagate = False
        for handler in list(replica_logger.handlers):
            replica_logger.removeHandler(handler)
            handler.close()
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        replica_logger.addHandler(handler)
        logger_pool[replica_id] = {
            "logger": replica_logger,
            "handler": handler,
            "path": str(log_path),
        }
    return logger_pool


def _close_logger_pool(logger_pool: dict[int, dict]) -> None:
    for entry in logger_pool.values():
        replica_logger = entry["logger"]
        handler = entry["handler"]
        replica_logger.removeHandler(handler)
        handler.flush()
        handler.close()


def _activate_replica_output_targets(
    replica: Replica,
    replica_id: int,
    logger_pool: dict[int, dict],
) -> None:
    replica.replica_logger = logger_pool[replica_id]["logger"]
    replica._logger_path = logger_pool[replica_id]["path"]
    replica._logger_handler = logger_pool[replica_id]["handler"]


# ---------------------------------------------------------------------------
# SQLite records store (energy matrix + exchange log)
# ---------------------------------------------------------------------------
def _open_records_db(path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the per-task records database.

    Holds the energy-matrix records (one row per chunk, keyed by ``chunk_id``)
    and the exchange log (one row per swap-side, keyed by ``(step, worker)``).
    WAL journalling with ``synchronous=FULL`` keeps each committed checkpoint
    flush durable; ``INSERT OR REPLACE`` makes re-running a not-yet-committed
    chunk idempotent instead of appending duplicates.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS energy_records (
            chunk_id INTEGER PRIMARY KEY,
            record_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            beta REAL NOT NULL,
            group_size INTEGER NOT NULL,
            worker_ids BLOB NOT NULL,
            old_replica_ids BLOB NOT NULL,
            new_replica_ids BLOB NOT NULL,
            accepted BLOB NOT NULL,
            entry_indices BLOB NOT NULL,
            entry_values BLOB NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS exchanges (
            step INTEGER NOT NULL,
            worker INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL,
            replica_id INTEGER NOT NULL,
            replica_try2exchange INTEGER NOT NULL,
            accept INTEGER NOT NULL,
            PRIMARY KEY (step, worker)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hremd_run_manifest (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            format INTEGER NOT NULL,
            signature TEXT NOT NULL,
            manifest_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS checkpoint_generations (
            generation_id INTEGER PRIMARY KEY,
            completed_chunks INTEGER NOT NULL,
            step_count INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS checkpoint_artifacts (
            generation_id INTEGER NOT NULL,
            replica_id INTEGER NOT NULL,
            checkpoint_path TEXT NOT NULL,
            checkpoint_size INTEGER NOT NULL,
            checkpoint_sha256 TEXT NOT NULL,
            dcd_frames INTEGER NOT NULL,
            PRIMARY KEY (generation_id, replica_id),
            FOREIGN KEY (generation_id) REFERENCES checkpoint_generations(generation_id)
                ON DELETE CASCADE
        )
        """
    )
    conn.commit()
    return conn


def _write_records_metadata(
    conn: sqlite3.Connection,
    metadata: dict[str, int | float | str],
) -> None:
    """Persist immutable HREMD run metadata alongside chunk records."""
    if not metadata:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO run_metadata (key, value) VALUES (?, ?)",
        [(str(key), str(value)) for key, value in metadata.items()],
    )
    conn.commit()


def _reset_records_after(conn: sqlite3.Connection, resumed_chunk_id: int) -> None:
    """Drop state at or beyond the selected starting chunk before writing.

    On resume the engine restarts from ``resumed_chunk_id``. A crash between the
    energy flush (barrier B4) and the checkpoint commit (B5) of a previous run
    can leave rows for chunks the checkpoints do not yet cover; deleting them
    here keeps the store consistent with the selected checkpoint generation.
    For a fresh run, ``resumed_chunk_id`` is zero, so all stale records and
    checkpoint metadata are removed.
    """
    with conn:
        conn.execute(
            "DELETE FROM energy_records WHERE chunk_id >= ?",
            (resumed_chunk_id,),
        )
        conn.execute(
            "DELETE FROM exchanges WHERE chunk_id >= ?",
            (resumed_chunk_id,),
        )
        conn.execute(
            "DELETE FROM checkpoint_artifacts WHERE generation_id >= ?",
            (resumed_chunk_id,),
        )
        conn.execute(
            "DELETE FROM checkpoint_generations WHERE generation_id >= ?",
            (resumed_chunk_id,),
        )


def _serialize_energy_matrix_entries(
    energy_entries: dict[tuple[int, int], float],
    group_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    items = sorted(energy_entries.items())
    entry_indices = np.empty((len(items), 2), dtype=np.int64)
    entry_values = np.empty((len(items),), dtype=np.float64)
    for idx, ((row_id, col_id), value) in enumerate(items):
        if not (0 <= row_id < group_size and 0 <= col_id < group_size):
            raise ValueError(
                f"energy_matrix key {(row_id, col_id)} is outside group size {group_size}"
            )
        entry_indices[idx] = (row_id, col_id)
        entry_values[idx] = float(value)
    return entry_indices, entry_values


def _deserialize_energy_matrix_entries(
    entry_indices: np.ndarray,
    entry_values: np.ndarray,
) -> dict[tuple[int, int], float]:
    if entry_indices.ndim != 2 or entry_indices.shape[1] != 2:
        raise ValueError(
            "energy_matrix entry indices must have shape (n_entries, 2)"
        )
    if entry_values.ndim != 1 or len(entry_indices) != len(entry_values):
        raise ValueError(
            "energy_matrix entry values must have shape (n_entries,)"
        )
    return {
        (int(row_id), int(col_id)): float(value)
        for (row_id, col_id), value in zip(entry_indices, entry_values)
    }


def _buffer_chunk_record(
    energy_record_buffer: list[dict],
    record_id: int,
    chunk_id: int,
    steps_per_chunk: int,
    beta: float,
    chunk_state: dict,
) -> None:
    """Append one finalized chunk energy record to the in-memory buffer.

    The record is not written to disk here; it is flushed to the records
    database only at the next checkpoint barrier so that on-disk state always
    ends exactly at a committed checkpoint.
    """
    energy_record_buffer.append(
        {
            "record_id": record_id,
            "chunk_id": chunk_id,
            "step": (chunk_id + 1) * steps_per_chunk,
            "beta": beta,
            "worker_ids": list(chunk_state["worker_ids"]),
            "old_replica_ids": list(chunk_state["old_replica_ids"]),
            "new_replica_ids": [
                int(chunk_state["new_replica_ids"][worker])
                for worker in chunk_state["worker_ids"]
            ],
            "accepted": [
                bool(chunk_state["accepted"][worker])
                for worker in chunk_state["worker_ids"]
            ],
            "energy_entries": dict(chunk_state["energy_entries"]),
        }
    )


def _flush_energy_buffer(
    conn: sqlite3.Connection,
    energy_record_buffer: list[dict],
    *,
    commit: bool = True,
    clear: bool = True,
) -> None:
    """Commit all buffered energy-matrix records to the database in one txn.

    Uses ``INSERT OR REPLACE`` keyed by ``chunk_id`` so re-running a chunk that
    was flushed but never committed by a previous run overwrites the stale row
    rather than creating a duplicate.
    """
    if not energy_record_buffer:
        return
    rows = []
    for record in energy_record_buffer:
        group_size = len(record["worker_ids"])
        entry_indices, entry_values = _serialize_energy_matrix_entries(
            record["energy_entries"], group_size
        )
        rows.append(
            (
                int(record["chunk_id"]),
                int(record["record_id"]),
                int(record["step"]),
                float(record["beta"]),
                group_size,
                np.asarray(record["worker_ids"], dtype=np.int64).tobytes(),
                np.asarray(record["old_replica_ids"], dtype=np.int64).tobytes(),
                np.asarray(record["new_replica_ids"], dtype=np.int64).tobytes(),
                np.asarray(record["accepted"], dtype=np.bool_).tobytes(),
                entry_indices.tobytes(),
                entry_values.tobytes(),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO energy_records (
            chunk_id, record_id, step, beta, group_size,
            worker_ids, old_replica_ids, new_replica_ids, accepted,
            entry_indices, entry_values
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    if commit:
        conn.commit()
    if clear:
        energy_record_buffer.clear()


def _flush_exchange_buffer(
    conn: sqlite3.Connection,
    exchange_row_buffer: list[tuple],
    *,
    commit: bool = True,
    clear: bool = True,
) -> None:
    """Commit all buffered exchange rows to the database in one txn.

    Each row is ``(chunk_id, step, worker, replica_id, replica_try2exchange,
    accept)`` and is upserted keyed by ``(step, worker)``.
    """
    if not exchange_row_buffer:
        return
    conn.executemany(
        """
        INSERT OR REPLACE INTO exchanges (
            chunk_id, step, worker, replica_id, replica_try2exchange, accept
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        exchange_row_buffer,
    )
    if commit:
        conn.commit()
    if clear:
        exchange_row_buffer.clear()


# ---------------------------------------------------------------------------
# DCD committed-extent reconciliation
# ---------------------------------------------------------------------------
def _read_dcd_layout(path: Path, expected_atoms: int) -> dict[str, int]:
    """Read the fixed-size OpenMM DCD layout and actual complete frame count."""
    with path.open("rb") as handle:
        header = handle.read(100)
        if len(header) < 100 or header[4:8] != b"CORD":
            raise RuntimeError(f"Invalid DCD header: {path}")
        if struct.unpack("<i", header[:4])[0] != 84:
            raise RuntimeError(f"Unsupported DCD byte order/header: {path}")
        header_frames = struct.unpack("<i", header[8:12])[0]
        first_step = struct.unpack("<i", header[12:16])[0]
        interval = struct.unpack("<i", header[16:20])[0]
        has_box = struct.unpack("<i", header[48:52])[0] != 0
        comments_bytes = struct.unpack("<i", header[92:96])[0]
        atom_offset = 104 + comments_bytes
        handle.seek(atom_offset)
        atom_data = handle.read(4)
        if len(atom_data) != 4:
            raise RuntimeError(f"Truncated DCD atom-count record: {path}")
        atom_count = struct.unpack("<i", atom_data)[0]

    if atom_count != expected_atoms:
        raise RuntimeError(
            f"DCD atom count mismatch at {path}: expected {expected_atoms}, "
            f"found {atom_count}"
        )
    header_size = 112 + comments_bytes
    frame_size = 3 * (8 + 4 * atom_count) + (56 if has_box else 0)
    payload_size = path.stat().st_size - header_size
    if payload_size < 0:
        raise RuntimeError(f"Truncated DCD header: {path}")
    complete_frames, trailing_bytes = divmod(payload_size, frame_size)
    return {
        "header_frames": int(header_frames),
        "first_step": int(first_step),
        "interval": int(interval),
        "header_size": int(header_size),
        "frame_size": int(frame_size),
        "complete_frames": int(complete_frames),
        "trailing_bytes": int(trailing_bytes),
    }


def _reconcile_dcd_file(
    path: Path, expected_frames: int, expected_atoms: int
) -> None:
    """Truncate uncommitted DCD tail while rejecting missing committed frames."""
    if expected_frames == 0 and not path.exists():
        return
    if not path.exists():
        raise RuntimeError(
            f"Cannot resume HREMD: missing DCD with {expected_frames} committed "
            f"frame(s): {path}"
        )

    layout = _read_dcd_layout(path, expected_atoms)
    if layout["complete_frames"] < expected_frames:
        raise RuntimeError(
            f"Cannot resume HREMD: DCD {path} has only "
            f"{layout['complete_frames']} complete frame(s), but records.sqlite "
            f"commits {expected_frames}."
        )

    committed_size = layout["header_size"] + expected_frames * layout["frame_size"]
    needs_repair = (
        path.stat().st_size != committed_size
        or layout["header_frames"] != expected_frames
    )
    if not needs_repair:
        return

    with path.open("r+b") as handle:
        handle.truncate(committed_size)
        handle.seek(8)
        handle.write(struct.pack("<i", expected_frames))
        handle.seek(20)
        last_step = 0
        if expected_frames > 0:
            last_step = layout["first_step"] + (expected_frames - 1) * layout["interval"]
        handle.write(struct.pack("<i", last_step))
        handle.flush()
        os.fsync(handle.fileno())


def _reconcile_trajectories(
    task_dir: Path,
    resume_state: HREMDResumeState,
    *,
    expected_atoms: int,
) -> None:
    for replica_id, expected_frames in resume_state.dcd_frames.items():
        _reconcile_dcd_file(
            _trajectory_path(task_dir, replica_id),
            expected_frames,
            expected_atoms,
        )


def _trajectory_frame_counts(pool: dict[int, tuple]) -> dict[int, int]:
    """Return the DCD writer counts that were durably flushed at B4."""
    return {
        int(replica_id): int(writer._modelCount)
        for replica_id, (_file_handle, writer) in pool.items()
    }


# ---------------------------------------------------------------------------
# Trajectory writer pool (one persistent DCD per Hamiltonian / replica_id)
# ---------------------------------------------------------------------------
def _open_master_traj_pool(
    task_dir: Path,
    n_replicas: int,
    topology,
    dt_ps: float,
    traj_interval: int,
    append: bool,
) -> dict[int, tuple]:
    """Open one persistent DCD writer per Hamiltonian (replica_id).

    Returns a handle pool mapping ``replica_id -> (file_handle, DCDFile)``.
    On resume (``append=True``) existing files are reopened in append mode; on
    a fresh run they are truncated (``wb``) to wipe any stale partial output.
    """
    if topology is None:
        raise ValueError("topology is required to open the master trajectory pool")
    pool: dict[int, tuple] = {}
    for replica_id in range(n_replicas):
        traj_path = _trajectory_path(task_dir, replica_id)
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        do_append = append and traj_path.exists() and traj_path.stat().st_size > 0
        mode = "r+b" if do_append else "wb"
        file_handle = open(traj_path, mode)
        writer = DCDFile(
            file_handle,
            topology,
            dt_ps * unit.picoseconds,
            firstStep=traj_interval,
            interval=traj_interval,
            append=do_append,
        )
        pool[replica_id] = (file_handle, writer)
    return pool


def _close_master_traj_pool(pool: dict[int, tuple]) -> None:
    for file_handle, _writer in pool.values():
        try:
            file_handle.flush()
            file_handle.close()
        except (OSError, ValueError):
            pass


def _drain_traj_queue_exact(
    traj_queue,
    traj_buffer_pool: dict[int, list],
    expected_frames: int,
    timeout_seconds: float,
) -> None:
    """Drain exactly ``expected_frames`` frames from the queue into the pool.

    ``mp.Queue`` uses an asynchronous feeder thread, so a worker returning from
    ``put`` does not guarantee the item is immediately readable. Draining by a
    known frame count (rather than until ``Empty``) prevents silently dropping
    in-flight trajectory frames at the checkpoint barrier. Raises ``TimeoutError``
    if the expected frames do not arrive within ``timeout_seconds`` (caller
    treats this as an abort condition).
    """
    collected = 0
    deadline = time.perf_counter() + timeout_seconds
    while collected < expected_frames:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:.1f}s while draining "
                f"trajectory queue: collected {collected}/{expected_frames} frames"
            )
        try:
            replica_id, chunk_id, positions, box = traj_queue.get(
                timeout=min(1.0, remaining)
            )
        except queue_module.Empty:
            continue
        traj_buffer_pool[replica_id].append((chunk_id, positions, box))
        collected += 1


def _flush_traj_pool(
    pool: dict[int, tuple],
    traj_buffer_pool: dict[int, list],
) -> None:
    """Write all buffered frames to their per-Hamiltonian DCD files in order.

    Frames are sorted by ``chunk_id`` so each ``replica_{id}/traj.dcd`` receives
    its frames in chronological order regardless of which worker produced them.
    """
    for replica_id, (file_handle, writer) in pool.items():
        frames = traj_buffer_pool.get(replica_id)
        if not frames:
            continue
        frames.sort(key=lambda item: item[0])
        for _chunk_id, positions, box in frames:
            writer.writeModel(
                positions=positions * unit.nanometer,
                periodicBoxVectors=box * unit.nanometer,
            )
        file_handle.flush()
        os.fsync(file_handle.fileno())
        frames.clear()


# ---------------------------------------------------------------------------
# Interval validation
# ---------------------------------------------------------------------------
def _validate_intervals(
    total_steps: int,
    exchange_interval: int,
    traj_record_interval: int,
    logger_record_interval: int,
    checkpoint_interval: Optional[int],
) -> None:
    """Validate HREMD step intervals for the buffer-until-checkpoint strategy.

    Checkpointing is mandatory: trajectory frames and energy records are
    buffered in memory and flushed only at checkpoint-chunk boundaries, so
    ``checkpoint_interval`` must be a positive, chunk-aligned multiple of
    ``exchange_interval`` that is at least as large as ``traj_record_interval``.
    """
    if exchange_interval <= 0:
        raise ValueError("exchange_interval must be positive")
    if total_steps % exchange_interval != 0:
        raise ValueError(
            f"total_steps ({total_steps}) must be divisible by "
            f"exchange_interval ({exchange_interval})"
        )
    if traj_record_interval <= 0:
        raise ValueError("traj_record_interval must be positive")
    if traj_record_interval % exchange_interval != 0:
        raise ValueError(
            "traj_record_interval must be a multiple of exchange_interval"
        )
    if logger_record_interval <= 0:
        raise ValueError("logger_record_interval must be positive")
    if logger_record_interval % exchange_interval != 0:
        raise ValueError(
            "logger_record_interval must be a multiple of exchange_interval"
        )
    if checkpoint_interval is None or checkpoint_interval <= 0:
        raise ValueError(
            "checkpoint_interval must be a positive integer; trajectory and "
            "energy-matrix output are buffered until each checkpoint flush"
        )
    if checkpoint_interval % exchange_interval != 0:
        raise ValueError(
            "checkpoint_interval must be a multiple of exchange_interval"
        )
    if checkpoint_interval < traj_record_interval:
        raise ValueError(
            "checkpoint_interval must be >= traj_record_interval so that every "
            "buffered trajectory frame is flushed at a checkpoint"
        )
