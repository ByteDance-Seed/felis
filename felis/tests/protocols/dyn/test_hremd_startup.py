# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Tests for HREMD multiprocessing and CUDA temporary-directory startup."""

import os
import signal
import time
import types
from pathlib import Path

import pytest

from felis.protocols.dyn import hremd
from felis.protocols.dyn import replica as replica_module


class _FakeIntegrator:
    def getStepSize(self):
        return 0.001


class _FakeReplica:
    def __init__(self, replica_id):
        self.replica_id = replica_id
        self.topology = object()
        self.integrator = _FakeIntegrator()
        self.platform_properties = None

    def set_platform(self, _platform, properties):
        self.platform_properties = properties


class _RecordingForkContext:
    def __init__(self, context):
        self.context = context
        self.calls = []

    def Array(self, *args, **kwargs):
        self.calls.append("Array")
        return self.context.Array(*args, **kwargs)

    def Barrier(self, *args, **kwargs):
        self.calls.append("Barrier")
        return self.context.Barrier(*args, **kwargs)

    def Queue(self, *args, **kwargs):
        self.calls.append("Queue")
        return self.context.Queue(*args, **kwargs)

    def Process(self, *args, **kwargs):
        self.calls.append("Process")
        return self.context.Process(*args, **kwargs)


def test_cuda_temp_directories_are_unique_for_96_ranks_and_each_run(tmp_path):
    run_root, rank_directories = hremd._create_cuda_temp_directories(
        str(tmp_path / "cuda"),
        96,
    )
    other_run_root, _ = hremd._create_cuda_temp_directories(
        str(tmp_path / "cuda"),
        1,
    )
    try:
        assert run_root != other_run_root
        assert len(rank_directories) == 96
        assert len(set(rank_directories)) == 96
        assert all(path.is_dir() for path in rank_directories)
        assert all(path.parent == run_root for path in rank_directories)
    finally:
        hremd.shutil.rmtree(run_root)
        hremd.shutil.rmtree(other_run_root)

    assert not run_root.exists()
    assert not other_run_root.exists()


def test_cuda_temp_directories_map_relative_base_under_tmp(tmp_path):
    relative_base = (tmp_path / "relative-cuda").relative_to(Path("/tmp"))
    run_root, rank_directories = hremd._create_cuda_temp_directories(
        str(relative_base),
        2,
    )
    try:
        assert run_root.parent == Path("/tmp") / relative_base
        assert [path.name for path in rank_directories] == ["rank-000", "rank-001"]
    finally:
        hremd.shutil.rmtree(run_root)


def test_cuda_temp_directories_reject_nonlocal_absolute_base():
    with pytest.raises(ValueError, match="under local /tmp"):
        hremd._create_cuda_temp_directories("/root/felis-hremd-test", 2)


def test_cuda_temp_directories_reject_symlink_escape(tmp_path):
    escape = tmp_path / "escape"
    escape.symlink_to("/root", target_is_directory=True)

    with pytest.raises(ValueError, match="under local /tmp"):
        hremd._create_cuda_temp_directories(str(escape / "felis-hremd-test"), 2)


class _FakeProcess:
    def __init__(self, fail_start=False):
        self.fail_start = fail_start
        self.started = False
        self.terminated = False
        self.killed = False
        self.joined = 0
        self.exitcode = 0
        self.name = "fake"
        self.pid = 1234

    def start(self):
        if self.fail_start:
            raise RuntimeError("injected process start failure")
        self.started = True

    def join(self, timeout=None):
        del timeout
        self.joined += 1

    def is_alive(self):
        return self.started and not self.terminated and not self.killed

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class _FakeBarrier:
    def __init__(self):
        self.aborted = False

    def abort(self):
        self.aborted = True


def _put_large_payload(queue, payload_ready):
    queue.put(b"x" * (8 * 1024 * 1024))
    payload_ready.set()


def _fail_after_payload_is_queued(payload_ready):
    if not payload_ready.wait(timeout=5.0):
        os._exit(18)
    os._exit(17)


def test_partial_process_start_failure_terminates_started_children():
    barrier = _FakeBarrier()
    first = _FakeProcess()
    second = _FakeProcess(fail_start=True)
    never_started = _FakeProcess()

    with pytest.raises(RuntimeError, match="injected process start failure"):
        hremd._start_and_wait_for_processes(
            [first, second, never_started],
            barrier,
        )

    assert barrier.aborted is True
    assert first.terminated is True
    assert first.joined == 1
    assert second.started is False
    assert never_started.started is False


@pytest.mark.serial
def test_child_failure_does_not_wait_for_blocked_queue_feeder():
    context = hremd._get_fork_context()
    traj_queue = context.Queue()
    payload_ready = context.Event()
    barrier = context.Barrier(2)
    worker = context.Process(
        target=_put_large_payload,
        args=(traj_queue, payload_ready),
        name="blocked-worker",
    )
    coordinator = context.Process(
        target=_fail_after_payload_is_queued,
        args=(payload_ready,),
        name="failing-coordinator",
    )
    processes = [worker, coordinator]

    def _raise_timeout(_signum, _frame):
        raise TimeoutError("HREMD process monitor did not return")

    previous_alarm_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    previous_alarm_seconds = signal.alarm(0)
    try:
        started_at = time.monotonic()
        signal.alarm(15)
        with pytest.raises(RuntimeError) as exc_info:
            hremd._start_and_wait_for_processes(processes, barrier)
        signal.alarm(0)
        elapsed = time.monotonic() - started_at

        error_message = str(exc_info.value)
        assert "rank=1" in error_message
        assert "name=failing-coordinator" in error_message
        assert "exitcode=17" in error_message
        assert elapsed < 10.0
        assert barrier.broken
        assert worker.exitcode in (-signal.SIGTERM, -signal.SIGKILL)
        assert all(not process.is_alive() for process in processes)
        child_pids = {process.pid for process in processes}
        assert child_pids.isdisjoint(
            {process.pid for process in hremd.mp.active_children()}
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
        if previous_alarm_seconds:
            signal.alarm(previous_alarm_seconds)
        for process in processes:
            if process.pid is None:
                continue
            process.join(timeout=0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            if not process.is_alive():
                process.close()
        traj_queue.close()
        traj_queue.join_thread()


@pytest.mark.parametrize("outcome", ["success", "error", "interrupt"])
def test_run_hremd_cleans_run_temp_root(monkeypatch, tmp_path, outcome):
    replicas = [_FakeReplica(0), _FakeReplica(1)]
    captured = {}
    original_create = hremd._create_cuda_temp_directories
    recording_context = _RecordingForkContext(hremd._get_fork_context())

    def capture_create(base, count):
        run_root, rank_directories = original_create(base, count)
        captured["run_root"] = run_root
        captured["rank_directories"] = rank_directories
        return run_root, rank_directories

    monkeypatch.setattr(hremd, "_get_fork_context", lambda: recording_context)
    monkeypatch.setattr(hremd, "_create_cuda_temp_directories", capture_create)
    monkeypatch.setattr(
        hremd.mm.Platform,
        "getPlatformByName",
        lambda _name: object(),
    )

    def start_and_wait(_processes, _barrier):
        assert captured["run_root"].is_dir()
        assert all(path.is_dir() for path in captured["rank_directories"])
        if outcome == "error":
            raise RuntimeError("injected HREMD startup failure")
        if outcome == "interrupt":
            raise KeyboardInterrupt()

    monkeypatch.setattr(hremd, "_start_and_wait_for_processes", start_and_wait)

    kwargs = dict(
        replicas=replicas,
        total_steps=1,
        exchange_interval=1,
        logger_record_interval=1,
        traj_record_interval=1,
        task_dir=str(tmp_path / "task"),
        temperature=300.0,
        gpu_devices=[0],
        checkpoint_interval=1,
        resume=False,
        tmp_directory=str(tmp_path / "cuda"),
    )
    if outcome == "error":
        with pytest.raises(RuntimeError, match="injected HREMD startup failure"):
            hremd.run_hremd(**kwargs)
    elif outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            hremd.run_hremd(**kwargs)
    else:
        hremd.run_hremd(**kwargs)

    assert not captured["run_root"].exists()
    assert (
        replicas[0].platform_properties["TempDirectory"]
        != replicas[1].platform_properties["TempDirectory"]
    )
    assert recording_context.calls.count("Array") == 1
    assert recording_context.calls.count("Barrier") == 1
    assert recording_context.calls.count("Queue") == 1
    assert recording_context.calls.count("Process") == 3


def test_get_fork_context_has_clear_unsupported_platform_error(monkeypatch):
    monkeypatch.setattr(hremd.mp, "get_all_start_methods", lambda: ["spawn"])

    with pytest.raises(RuntimeError, match="requires a POSIX platform"):
        hremd._get_fork_context()


def test_get_list_of_replicas_uses_each_state_stochastic_seed(monkeypatch):
    schedules = {
        "torsionLambda": [1.0, 1.0, 1.0],
        "reslam": [0.0, 0.0, 0.0],
    }
    gks = []
    for replica_id in range(3):
        gks.append(
            types.SimpleNamespace(
                ab=types.SimpleNamespace(lambda_schedules=schedules),
                boresch=types.SimpleNamespace(proatoms=None),
                integrator=types.SimpleNamespace(
                    targetP_bar=1.0 + replica_id,
                    targetT_K=300.0 + replica_id,
                    randomseed=1000 + replica_id,
                ),
            )
        )

    system_calls = []
    integrator_calls = []
    integrators = [object(), object(), object()]

    def fake_get_system_topology(gk, enable_perturb_nonbonded):
        system_calls.append((gk, enable_perturb_nonbonded))
        return replica_module.mm.System(), object()

    def fake_get_integrator(gk):
        integrator_calls.append(gk)
        return integrators[gks.index(gk)]

    monkeypatch.setattr(
        replica_module, "get_system_topology", fake_get_system_topology
    )
    monkeypatch.setattr(
        replica_module, "get_initial_pos_pbc", lambda _gk: (object(), object())
    )
    monkeypatch.setattr(replica_module, "get_integrator", fake_get_integrator)

    replicas = replica_module.get_list_of_replicas(gks)

    assert system_calls == [(gks[0], True)]
    assert integrator_calls == gks
    assert [replica.velocity_seed for replica in replicas] == [1000, 1001, 1002]
    assert [replica.integrator for replica in replicas] == integrators
