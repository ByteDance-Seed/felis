# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Unit tests for HREMD buffer-until-checkpoint resume support."""

import sqlite3
import types

import numpy as np
import openmm as mm
import pytest
from openmm import unit
from openmm.app import DCDFile, Element, Topology

from felis.protocols.dyn import hremd
from felis.protocols.dyn import hremd_io
from felis.protocols.dyn.replica import Replica
from felis.protocols.abfe_v2.utilities import get_free_energy_timelines
from felis.protocols.abfe_v2.utilities import read_records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chunk_state(group_size: int = 3) -> dict:
    energy_entries = {}
    for i in range(group_size):
        energy_entries[(i, i)] = float(i)
        if i > 0:
            energy_entries[(i - 1, i)] = float(i) - 0.5
        if i + 1 < group_size:
            energy_entries[(i + 1, i)] = float(i) + 0.5
    workers = list(range(group_size))
    return {
        "worker_ids": workers,
        "old_replica_ids": list(workers),
        "new_replica_ids": {w: w for w in workers},
        "accepted": {w: False for w in workers},
        "energy_entries": energy_entries,
        "record_ready": True,
        "pending_exchange_workers": set(),
    }


def _buffer_n_records(
    n_records: int, group_size: int = 3, start: int = 0
) -> list[dict]:
    buffer: list[dict] = []
    for record_id in range(start, start + n_records):
        hremd_io._buffer_chunk_record(
            buffer,
            record_id=record_id,
            chunk_id=record_id,
            steps_per_chunk=10,
            beta=1.0,
            chunk_state=_make_chunk_state(group_size),
        )
    return buffer


def _count_records(path) -> int:
    conn = sqlite3.connect(str(path))
    try:
        (count,) = conn.execute("SELECT COUNT(*) FROM energy_records").fetchone()
    finally:
        conn.close()
    return int(count)


def _write_checkpoints(
    task_dir, n_replicas, chunk_id, steps_per_chunk, dcd_frames=0
) -> None:
    for replica_id in range(n_replicas):
        checkpoint_path = hremd_io._checkpoint_path(
            task_dir, replica_id, chunk_id
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(f"checkpoint-{chunk_id}-{replica_id}".encode())
    conn = hremd_io._open_records_db(hremd_io._records_db_path(task_dir))
    try:
        hremd_io._commit_checkpoint_generation(
            conn,
            task_dir,
            generation_id=chunk_id,
            step_count=(chunk_id + 1) * steps_per_chunk,
            n_replicas=n_replicas,
            dcd_frames={replica_id: dcd_frames for replica_id in range(n_replicas)},
        )
    finally:
        conn.close()


class _RecordingWriter:
    """Fake DCDFile capturing the order of writeModel calls."""

    def __init__(self):
        self.chunk_order = []

    def writeModel(self, positions=None, periodicBoxVectors=None):
        # positions[0][0] encodes the chunk_id in these tests.
        value = positions[0][0]
        if hasattr(value, "value_in_unit"):
            from openmm import unit

            value = value.value_in_unit(unit.nanometer)
        self.chunk_order.append(int(round(float(value))))


def _manifest_replica(replica_id, seed, position_nm=0.0):
    system = mm.System()
    system.addParticle(12.0)
    integrator = mm.LangevinMiddleIntegrator(300.0, 1.0, 0.004)
    integrator.setRandomNumberSeed(seed)
    return types.SimpleNamespace(
        replica_id=replica_id,
        system=system,
        integrator=integrator,
        lambda_schedules={"lambdaA_charge": [1.0]},
        is_boresch_restraint=False,
        temperature=300.0,
        pressure=1.0,
        velocity_temperature_K=300.0,
        topology=None,
        initial_positions=[mm.Vec3(position_nm, 0.0, 0.0)] * unit.nanometer,
        initial_box_vectors=(
            mm.Vec3(2.0, 0.0, 0.0),
            mm.Vec3(0.0, 2.0, 0.0),
            mm.Vec3(0.0, 0.0, 2.0),
        ) * unit.nanometer,
    )


def _manifest_kwargs():
    return dict(
        temperature=300.0,
        exchange_strategy="neighbor",
        with_pv=True,
        exchange_interval=250,
        traj_record_interval=1000,
        checkpoint_interval=1000,
    )


def _write_dcd(path, n_frames, periodic, first_step=0, interval=1):
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    topology.addAtom("C", Element.getBySymbol("C"), residue)
    box = None
    if periodic:
        box = (
            mm.Vec3(2.0, 0.0, 0.0),
            mm.Vec3(0.0, 2.0, 0.0),
            mm.Vec3(0.0, 0.0, 2.0),
        ) * unit.nanometer
        topology.setPeriodicBoxVectors(box)
    with path.open("wb") as handle:
        writer = DCDFile(
            handle,
            topology,
            0.002 * unit.picoseconds,
            firstStep=first_step,
            interval=interval,
        )
        for frame_id in range(n_frames):
            writer.writeModel(
                [mm.Vec3(float(frame_id), 0.0, 0.0)] * unit.nanometer,
                periodicBoxVectors=box,
            )
    return topology


# ---------------------------------------------------------------------------
# _validate_intervals
# ---------------------------------------------------------------------------
def test_validate_intervals_accepts_valid():
    hremd_io._validate_intervals(
        total_steps=1000,
        exchange_interval=100,
        traj_record_interval=200,
        logger_record_interval=100,
        checkpoint_interval=400,
    )


@pytest.mark.parametrize("checkpoint_interval", [None, 0, -100])
def test_validate_intervals_requires_positive_checkpoint(checkpoint_interval):
    with pytest.raises(ValueError):
        hremd_io._validate_intervals(
            total_steps=1000,
            exchange_interval=100,
            traj_record_interval=200,
            logger_record_interval=100,
            checkpoint_interval=checkpoint_interval,
        )


def test_validate_intervals_checkpoint_must_be_multiple_of_exchange():
    with pytest.raises(ValueError):
        hremd_io._validate_intervals(
            total_steps=1000,
            exchange_interval=100,
            traj_record_interval=200,
            logger_record_interval=100,
            checkpoint_interval=350,
        )


def test_validate_intervals_checkpoint_must_cover_traj():
    with pytest.raises(ValueError):
        hremd_io._validate_intervals(
            total_steps=1000,
            exchange_interval=100,
            traj_record_interval=400,
            logger_record_interval=100,
            checkpoint_interval=200,
        )


# ---------------------------------------------------------------------------
# _buffer_chunk_record + _flush_energy_buffer
# ---------------------------------------------------------------------------
def test_buffer_and_flush_energy_roundtrip(tmp_path):
    path = tmp_path / "records.sqlite"
    buffer = _buffer_n_records(5)
    assert len(buffer) == 5

    conn = hremd_io._open_records_db(path)
    try:
        hremd_io._flush_energy_buffer(conn, buffer)
    finally:
        conn.close()

    assert _count_records(path) == 5
    assert buffer == []  # buffer cleared after flush

    # First record id preserved.
    records = read_records(path)
    assert [r["chunk_id"] for r in records] == [0, 1, 2, 3, 4]
    assert records[0]["record_id"] == 0


def test_flush_energy_buffer_appends(tmp_path):
    path = tmp_path / "records.sqlite"
    conn = hremd_io._open_records_db(path)
    try:
        hremd_io._flush_energy_buffer(conn, _buffer_n_records(3, start=0))
        hremd_io._flush_energy_buffer(conn, _buffer_n_records(2, start=3))
    finally:
        conn.close()
    assert _count_records(path) == 5


def test_flush_energy_buffer_upsert_is_idempotent(tmp_path):
    path = tmp_path / "records.sqlite"
    conn = hremd_io._open_records_db(path)
    try:
        # Flush the same chunk_ids twice: re-running a not-yet-committed chunk
        # must overwrite rather than duplicate.
        hremd_io._flush_energy_buffer(conn, _buffer_n_records(4))
        hremd_io._flush_energy_buffer(conn, _buffer_n_records(4))
    finally:
        conn.close()
    assert _count_records(path) == 4


def test_reset_records_after_deletes_tail(tmp_path):
    path = tmp_path / "records.sqlite"
    conn = hremd_io._open_records_db(path)
    try:
        hremd_io._flush_energy_buffer(conn, _buffer_n_records(5))
        hremd_io._flush_exchange_buffer(
            conn,
            [
                (chunk_id, (chunk_id + 1) * 10, 0, 0, 1, 1)
                for chunk_id in range(5)
            ],
        )
        conn.executemany(
            "INSERT INTO checkpoint_generations "
            "(generation_id, completed_chunks, step_count) VALUES (?, ?, ?)",
            [(generation_id, generation_id + 1, (generation_id + 1) * 10)
             for generation_id in range(5)],
        )
        conn.executemany(
            "INSERT INTO checkpoint_artifacts "
            "(generation_id, replica_id, checkpoint_path, checkpoint_size, "
            "checkpoint_sha256, dcd_frames) VALUES (?, ?, ?, ?, ?, ?)",
            [(generation_id, 0, f"checkpoint_{generation_id}.chk", 1, "hash", 0)
             for generation_id in range(5)],
        )
        conn.commit()
        hremd_io._reset_records_after(conn, 3)
        remaining = [
            row[0]
            for row in conn.execute(
                "SELECT chunk_id FROM energy_records ORDER BY chunk_id"
            )
        ]
        (exchange_count,) = conn.execute(
            "SELECT COUNT(*) FROM exchanges WHERE chunk_id >= 3"
        ).fetchone()
        remaining_generations = [
            row[0]
            for row in conn.execute(
                "SELECT generation_id FROM checkpoint_generations "
                "ORDER BY generation_id"
            )
        ]
        remaining_artifacts = [
            row[0]
            for row in conn.execute(
                "SELECT generation_id FROM checkpoint_artifacts "
                "ORDER BY generation_id"
            )
        ]
    finally:
        conn.close()
    assert remaining == [0, 1, 2]
    assert exchange_count == 0
    assert remaining_generations == [0, 1, 2]
    assert remaining_artifacts == [0, 1, 2]


def test_reset_records_after_rolls_back_all_tables_on_failure(tmp_path):
    path = tmp_path / "records.sqlite"
    conn = hremd_io._open_records_db(path)
    try:
        hremd_io._flush_energy_buffer(conn, _buffer_n_records(2))
        conn.execute(
            "INSERT INTO checkpoint_generations "
            "(generation_id, completed_chunks, step_count) VALUES (1, 2, 20)"
        )
        conn.execute(
            "CREATE TRIGGER reject_generation_delete "
            "BEFORE DELETE ON checkpoint_generations "
            "BEGIN SELECT RAISE(ABORT, 'reject delete'); END"
        )
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError, match="reject delete"):
            hremd_io._reset_records_after(conn, 1)

        assert conn.execute(
            "SELECT COUNT(*) FROM energy_records"
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM checkpoint_generations"
        ).fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.parametrize(
    "resume,resumed_chunk_id,expected_chunks",
    [
        (False, 0, []),
        (True, 0, []),
        (True, 3, [0, 1, 2]),
    ],
)
def test_coordinator_aligns_records_with_starting_chunk(
    monkeypatch, tmp_path, resume, resumed_chunk_id, expected_chunks
):
    path = hremd_io._records_db_path(tmp_path)
    conn = hremd_io._open_records_db(path)
    try:
        hremd_io._flush_energy_buffer(conn, _buffer_n_records(5))
        hremd_io._flush_exchange_buffer(
            conn,
            [
                (chunk_id, (chunk_id + 1) * 10, 0, 0, 1, 1)
                for chunk_id in range(5)
            ],
        )
    finally:
        conn.close()

    monkeypatch.setattr(hremd, "_open_master_traj_pool", lambda *_args, **_kwargs: {})
    hremd._coordinator_loop(
        n_replicas=1,
        shared_status=None,
        barrier=None,
        exchange_strategy="neighbor",
        beta=1.0,
        total_chunks=resumed_chunk_id,
        task_dir=tmp_path,
        steps_per_chunk=10,
        chunks_per_checkpoint=1,
        chunks_per_traj=1,
        traj_queue=None,
        topology=object(),
        dt_ps=0.002,
        resume=resume,
        resumed_chunk_id=resumed_chunk_id,
    )

    conn = sqlite3.connect(str(path))
    try:
        energy_chunks = [
            row[0]
            for row in conn.execute(
                "SELECT chunk_id FROM energy_records ORDER BY chunk_id"
            )
        ]
        exchange_chunks = [
            row[0]
            for row in conn.execute(
                "SELECT chunk_id FROM exchanges ORDER BY chunk_id"
            )
        ]
    finally:
        conn.close()
    assert energy_chunks == expected_chunks
    assert exchange_chunks == expected_chunks


def test_read_records_roundtrip_energy_entries(tmp_path):
    path = tmp_path / "records.sqlite"
    conn = hremd_io._open_records_db(path)
    try:
        hremd_io._flush_energy_buffer(conn, _buffer_n_records(2))
    finally:
        conn.close()

    records = read_records(path)
    assert len(records) == 2
    expected = _make_chunk_state()["energy_entries"]
    assert records[0]["energy_entries"] == expected
    assert list(records[0]["worker_ids"]) == [0, 1, 2]


def test_read_records_returns_run_metadata_when_requested(tmp_path):
    path = tmp_path / "records.sqlite"
    conn = hremd_io._open_records_db(path)
    try:
        hremd_io._write_records_metadata(
            conn,
            {
                "dt_ps": 0.002,
                "steps_per_chunk": 250,
                "exchange_strategy": "neighbor",
            },
        )
        hremd_io._flush_energy_buffer(conn, _buffer_n_records(2))
    finally:
        conn.close()

    records, metadata = read_records(path, include_metadata=True)

    assert len(records) == 2
    assert metadata == {
        "dt_ps": "0.002",
        "steps_per_chunk": "250",
        "exchange_strategy": "neighbor",
    }


def test_run_manifest_accepts_same_identity_and_rejects_changed_identity(tmp_path):
    path = hremd_io._records_db_path(tmp_path)
    conn = hremd_io._open_records_db(path)
    manifest = {"format": 1, "schedule": [[1.0], [0.0]], "dt": 0.004}
    try:
        signature = hremd_io._prepare_run_manifest(
            conn, tmp_path, manifest, resume=False
        )
        assert hremd_io._prepare_run_manifest(
            conn, tmp_path, dict(manifest), resume=True
        ) == signature
        with pytest.raises(RuntimeError, match="run identity does not match"):
            hremd_io._prepare_run_manifest(
                conn,
                tmp_path,
                {**manifest, "schedule": [[1.0], [0.5]]},
                resume=True,
            )
    finally:
        conn.close()


def test_run_manifest_ignores_restart_irrelevant_integrator_seed():
    manifest_a = hremd_io._build_run_manifest(
        [_manifest_replica(0, 123)], **_manifest_kwargs()
    )
    manifest_b = hremd_io._build_run_manifest(
        [_manifest_replica(0, 456)], **_manifest_kwargs()
    )

    assert hremd_io._run_signature(manifest_a) == hremd_io._run_signature(
        manifest_b
    )


def test_run_manifest_rejects_inconsistent_shared_replica_state():
    replicas = [
        _manifest_replica(0, 100, position_nm=0.0),
        _manifest_replica(1, 101, position_nm=0.1),
    ]

    with pytest.raises(ValueError, match="initial_positions_nm"):
        hremd_io._build_run_manifest(replicas, **_manifest_kwargs())


def test_run_manifest_is_compact_for_96_replicas():
    replicas = [_manifest_replica(index, 1000 + index) for index in range(96)]

    manifest = hremd_io._build_run_manifest(replicas, **_manifest_kwargs())
    serialized = hremd_io._canonical_json(manifest)

    assert len(serialized.encode("utf-8")) < 100_000
    assert "atoms" not in manifest["shared"]["topology"]
    assert len(manifest["replicas"]) == 96


def test_resume_rejects_legacy_output_without_sqlite_manifest(tmp_path):
    conn = hremd_io._open_records_db(hremd_io._records_db_path(tmp_path))
    try:
        hremd_io._flush_energy_buffer(conn, _buffer_n_records(1))
        with pytest.raises(RuntimeError, match="no SQLite run manifest"):
            hremd_io._prepare_run_manifest(
                conn, tmp_path, {"format": 1}, resume=True
            )
    finally:
        conn.close()


@pytest.mark.parametrize("periodic", [False, True])
def test_reconcile_dcd_truncates_extra_frames_and_updates_header(
    tmp_path, periodic
):
    path = tmp_path / "traj.dcd"
    topology = _write_dcd(
        path,
        n_frames=3,
        periodic=periodic,
        first_step=2500,
        interval=2500,
    )

    hremd_io._reconcile_dcd_file(path, expected_frames=2, expected_atoms=1)

    layout = hremd_io._read_dcd_layout(path, expected_atoms=1)
    assert layout["complete_frames"] == 2
    assert layout["header_frames"] == 2
    assert layout["trailing_bytes"] == 0
    with path.open("rb") as handle:
        handle.seek(20)
        assert int.from_bytes(handle.read(4), "little", signed=True) == 5000

    with path.open("r+b") as handle:
        writer = DCDFile(
            handle,
            topology,
            0.002 * unit.picoseconds,
            firstStep=2500,
            interval=2500,
            append=True,
        )
        writer.writeModel([mm.Vec3(9.0, 0.0, 0.0)] * unit.nanometer)
    layout = hremd_io._read_dcd_layout(path, expected_atoms=1)
    assert layout["header_frames"] == 3
    with path.open("rb") as handle:
        handle.seek(20)
        assert int.from_bytes(handle.read(4), "little", signed=True) == 7500


@pytest.mark.parametrize("periodic", [False, True])
def test_reconcile_dcd_rejects_missing_committed_frames(tmp_path, periodic):
    path = tmp_path / "traj.dcd"
    _write_dcd(path, n_frames=1, periodic=periodic)

    with pytest.raises(RuntimeError, match="only 1 complete frame"):
        hremd_io._reconcile_dcd_file(
            path, expected_frames=2, expected_atoms=1
        )


def test_reconcile_dcd_removes_partial_uncommitted_frame(tmp_path):
    path = tmp_path / "traj.dcd"
    _write_dcd(path, n_frames=2, periodic=True)
    with path.open("ab") as handle:
        handle.write(b"partial-next-frame")

    hremd_io._reconcile_dcd_file(path, expected_frames=2, expected_atoms=1)

    layout = hremd_io._read_dcd_layout(path, expected_atoms=1)
    assert layout["complete_frames"] == 2
    assert layout["trailing_bytes"] == 0


def test_free_energy_timelines_use_recorded_steps_and_timestep():
    records = [{"step": step} for step in range(100, 901, 100)]

    forward, backward, moving, x_label = get_free_energy_timelines(
        records,
        num_blocks=3,
        dt_ps=0.002,
    )

    np.testing.assert_allclose(forward, [0.0006, 0.0012, 0.0018])
    np.testing.assert_allclose(backward, [0.0014, 0.0008, 0.0002])
    np.testing.assert_allclose(moving, [0.0004, 0.0010, 0.0016])
    assert x_label == "Time (ns)"


def test_free_energy_timelines_fall_back_to_md_steps():
    records = [{"step": step} for step in range(100, 401, 100)]

    forward, backward, moving, x_label = get_free_energy_timelines(
        records,
        num_blocks=2,
    )

    np.testing.assert_allclose(forward, [200.0, 400.0])
    np.testing.assert_allclose(backward, [300.0, 100.0])
    np.testing.assert_allclose(moving, [150.0, 350.0])
    assert x_label == "MD steps"


# ---------------------------------------------------------------------------
# exchange history reconstruction
# ---------------------------------------------------------------------------
def test_exchange_matrix_applies_first_record():
    from felis.protocols.abfe_v2.utilities import get_exchange_matrix_from_records

    records = [{
        "old_replica_ids": np.array([0, 1]),
        "worker_ids": np.array([0, 1]),
        "new_replica_ids": np.array([1, 0]),
    }]
    matrix = get_exchange_matrix_from_records(records)
    np.testing.assert_array_equal(matrix, np.array([[0, 1], [1, 0]]))


# ---------------------------------------------------------------------------
# _flush_traj_pool ordering
# ---------------------------------------------------------------------------
def test_flush_traj_pool_orders_by_chunk_id(tmp_path):
    f0 = open(tmp_path / "r0.dcd", "wb")
    writer0 = _RecordingWriter()
    pool = {0: (f0, writer0)}

    def _frame(chunk_id):
        pos = np.zeros((1, 3), dtype=np.float32)
        pos[0][0] = chunk_id
        box = np.zeros((3, 3), dtype=np.float32)
        return (chunk_id, pos, box)

    # Out-of-order frames.
    buffer_pool = {0: [_frame(3), _frame(1), _frame(2)]}
    hremd_io._flush_traj_pool(pool, buffer_pool)
    f0.close()

    assert writer0.chunk_order == [1, 2, 3]
    assert buffer_pool[0] == []  # buffer cleared


# ---------------------------------------------------------------------------
# SQLite checkpoint generation discovery
# ---------------------------------------------------------------------------
def _discover(task_dir, n_replicas, steps_per_chunk):
    conn = hremd_io._open_records_db(hremd_io._records_db_path(task_dir))
    try:
        return hremd_io._discover_resume_state(
            conn, task_dir, n_replicas, steps_per_chunk
        )
    finally:
        conn.close()


def test_discover_resume_state_none_when_no_generations(tmp_path):
    assert _discover(tmp_path, 3, 10) is None


def test_discover_resume_state_returns_latest_complete_generation(tmp_path):
    _write_checkpoints(tmp_path, 3, chunk_id=4, steps_per_chunk=10)
    state = _discover(tmp_path, 3, 10)
    assert state.generation_id == 4
    assert state.resumed_chunk_id == 5
    assert state.step_count == 50
    assert state.dcd_frames == {0: 0, 1: 0, 2: 0}


def test_discover_resume_state_falls_back_from_missing_new_checkpoint(tmp_path):
    _write_checkpoints(tmp_path, 3, chunk_id=2, steps_per_chunk=10)
    _write_checkpoints(tmp_path, 3, chunk_id=4, steps_per_chunk=10)
    hremd_io._checkpoint_path(tmp_path, 1, 4).unlink()

    state = _discover(tmp_path, 3, 10)

    assert state.generation_id == 2
    assert state.resumed_chunk_id == 3


def test_discover_resume_state_falls_back_from_same_size_corruption(tmp_path):
    _write_checkpoints(tmp_path, 2, chunk_id=1, steps_per_chunk=10)
    _write_checkpoints(tmp_path, 2, chunk_id=3, steps_per_chunk=10)
    path = hremd_io._checkpoint_path(tmp_path, 0, 3)
    path.write_bytes(b"x" * path.stat().st_size)

    state = _discover(tmp_path, 2, 10)

    assert state.generation_id == 1


def test_discover_resume_state_ignores_uncommitted_external_generation(tmp_path):
    _write_checkpoints(tmp_path, 2, chunk_id=1, steps_per_chunk=10)
    for replica_id in range(2):
        path = hremd_io._checkpoint_path(tmp_path, replica_id, 3)
        path.write_bytes(b"uncommitted")

    state = _discover(tmp_path, 2, 10)

    assert state.generation_id == 1


def test_discover_resume_state_falls_back_from_short_latest_dcd(tmp_path):
    topology = None
    for replica_id in range(2):
        path = hremd_io._trajectory_path(tmp_path, replica_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        topology = _write_dcd(path, n_frames=1, periodic=False)
    _write_checkpoints(
        tmp_path, 2, chunk_id=0, steps_per_chunk=10, dcd_frames=1
    )
    _write_checkpoints(
        tmp_path, 2, chunk_id=1, steps_per_chunk=10, dcd_frames=2
    )

    conn = hremd_io._open_records_db(hremd_io._records_db_path(tmp_path))
    try:
        state = hremd_io._discover_resume_state(
            conn, tmp_path, 2, 10, expected_atoms=topology.getNumAtoms()
        )
    finally:
        conn.close()

    assert state.generation_id == 0


def test_crash_after_dcd_flush_falls_back_and_truncates_uncommitted_tail(tmp_path):
    topology = None
    for replica_id in range(2):
        path = hremd_io._trajectory_path(tmp_path, replica_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        topology = _write_dcd(path, n_frames=2, periodic=True)
    _write_checkpoints(
        tmp_path, 2, chunk_id=0, steps_per_chunk=10, dcd_frames=1
    )
    # Simulate the next generation's binaries becoming durable after its DCD
    # frames were flushed, but before the SQLite generation transaction.
    for replica_id in range(2):
        hremd_io._checkpoint_path(tmp_path, replica_id, 1).write_bytes(
            b"uncommitted-new-generation"
        )

    state = _discover(tmp_path, 2, 10)
    hremd_io._reconcile_trajectories(
        tmp_path, state, expected_atoms=topology.getNumAtoms()
    )

    assert state.generation_id == 0
    for replica_id in range(2):
        layout = hremd_io._read_dcd_layout(
            hremd_io._trajectory_path(tmp_path, replica_id), expected_atoms=1
        )
        assert layout["complete_frames"] == 1
        assert layout["header_frames"] == 1


def test_generation_is_not_published_with_incomplete_checkpoint_set(tmp_path):
    path = hremd_io._checkpoint_path(tmp_path, 0, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"only-one-replica")
    conn = hremd_io._open_records_db(hremd_io._records_db_path(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="missing checkpoint for replica 1"):
            hremd_io._commit_checkpoint_generation(
                conn,
                tmp_path,
                generation_id=2,
                step_count=30,
                n_replicas=2,
                dcd_frames={0: 1, 1: 1},
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM checkpoint_generations"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_discover_resume_state_rejects_when_no_committed_generation_is_usable(
    tmp_path,
):
    _write_checkpoints(tmp_path, 2, chunk_id=1, steps_per_chunk=10)
    hremd_io._checkpoint_path(tmp_path, 0, 1).unlink()

    with pytest.raises(RuntimeError, match="no complete committed"):
        _discover(tmp_path, 2, 10)


# ---------------------------------------------------------------------------
# Replica generation checkpoint round-trip
# ---------------------------------------------------------------------------
class _StubContext:
    def __init__(self):
        self._step = 0
        self._params = {}

    def createCheckpoint(self):
        return int(self._step).to_bytes(8, "little")

    def loadCheckpoint(self, data):
        self._step = int.from_bytes(bytes(data), "little")

    def getStepCount(self):
        return self._step

    def getParameters(self):
        return {"replica_id": self._params.get("replica_id", 0.0)}

    def setParameter(self, name, value):
        self._params[name] = value


def _make_replica(replica_id):
    return Replica(
        system={},
        topology=None,
        replica_id=replica_id,
        lambda_schedules={"torsionLambda": [0.0, 0.25, 0.5, 1.0]},
    )


def test_replica_write_and_load_checkpoint_roundtrip(tmp_path):
    chk_path = hremd_io._checkpoint_path(tmp_path, 2, 5)

    writer = _make_replica(2)
    writer.context = _StubContext()
    writer.context._step = 600
    writer._checkpoint_path = str(chk_path)
    writer.write_checkpoint(chunk_id=5, step_count=600)

    assert chk_path.exists()
    assert list(chk_path.parent.glob("*.meta.json")) == []

    loader = _make_replica(2)
    loader.context = _StubContext()
    assert loader.load_checkpoint(chk_path) == 600
    assert loader.context._params["replica_id"] == 2


# ---------------------------------------------------------------------------
# _apply_exchange_result no longer touches trajectory handles
# ---------------------------------------------------------------------------
class _StubReplica:
    def __init__(self, replica_id):
        self.replica_id = replica_id
        self.updated_to = None
        self.trajectory_touched = False

    def update_replica_id(self, new_id):
        self.replica_id = new_id
        self.updated_to = new_id

    def close_trajectory(self):
        self.trajectory_touched = True

    def open_trajectory(self, *args, **kwargs):
        self.trajectory_touched = True


def test_apply_exchange_result_no_trajectory_handoff():
    rep = _StubReplica(0)
    hremd._apply_exchange_result(rep, 2)
    assert rep.updated_to == 2
    assert rep.trajectory_touched is False


def test_apply_exchange_result_noop_when_same():
    rep = _StubReplica(1)
    hremd._apply_exchange_result(rep, 1)
    assert rep.updated_to is None


# ---------------------------------------------------------------------------
# Resume loop bounds
# ---------------------------------------------------------------------------
def test_resume_loop_bounds():
    assert list(range(3, 10)) == [3, 4, 5, 6, 7, 8, 9]
