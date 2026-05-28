# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from felis.protocols.abfe.config_types import ABStage
from felis.protocols.abfe.job_context import JobContext
import felis.protocols.abfe.job_context as job_context_mod
from felis.utils import resolve_path


class FakeProc:
    """Minimal Popen stand-in whose ``wait()`` returns a configurable code."""

    def __init__(self, returncode: int = 0):
        self._returncode = returncode

    def wait(self):
        return self._returncode


def patch_popen_capture(monkeypatch, *, returncode: int = 0, with_shell: bool = True):
    """Replace ``subprocess.Popen`` with a recorder.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        returncode: Value returned by ``FakeProc.wait()``.
        with_shell: If True, the fake Popen accepts a ``shell`` keyword
            (matching ``_run_one_subprocess``); otherwise it matches the
            signature used by ``run_one_subprocess_nowait``.

    Returns:
        A dict that will be populated with the most recent call's
        ``args``, ``cwd``, ``env``, and (when ``with_shell``) ``shell``.
    """

    captured: dict = {}
    proc = FakeProc(returncode=returncode)

    if with_shell:

        def fake_popen(popen_args, cwd, env, shell=False):
            captured["args"] = popen_args
            captured["cwd"] = cwd
            captured["env"] = env
            captured["shell"] = shell
            return proc
    else:

        def fake_popen(popen_args, cwd, env):
            captured["args"] = popen_args
            captured["cwd"] = cwd
            captured["env"] = env
            return proc

    monkeypatch.setattr(job_context_mod.subprocess, "Popen", fake_popen)
    return captured


@pytest.mark.parametrize(
    "use_mps,expected_args_type",
    [
        (False, str),
        (True, list),
    ],
)
def test_job_context_init_creates_dirs_and_handles_missing_mps(monkeypatch, tmp_path, use_mps, expected_args_type):
    calls = []

    def fake_run_one_subprocess(self, args, working_dir, extra_envs=None):
        calls.append((args, working_dir, extra_envs))
        raise OSError("missing mps control")

    monkeypatch.setattr(JobContext, "_run_one_subprocess", fake_run_one_subprocess)

    working_dir = tmp_path / "wd"
    jc = JobContext(str(working_dir), extra_envs={"X": "Y"}, use_mps=use_mps)

    expected_working_dir = resolve_path(str(working_dir))
    assert jc.working_dir == expected_working_dir
    assert jc.envs["X"] == "Y"

    assert Path(jc.prepare_dir).is_dir()
    assert Path(jc.trj_dir).is_dir()
    assert Path(jc.progress_dir).is_dir()

    assert len(calls) == 1
    args, wdir, _extra = calls[0]
    assert isinstance(args, expected_args_type)
    assert wdir == expected_working_dir
    if use_mps:
        assert args == ["nvidia-cuda-mps-control", "-d"]
    else:
        assert args == "echo quit | nvidia-cuda-mps-control"


@pytest.mark.parametrize(
    "args,expected_shell",
    [
        (["echo", "hi"], False),
        ("echo hi", True),
    ],
)
def test_run_one_subprocess_merges_env_and_sets_shell(monkeypatch, tmp_path, args, expected_shell):
    captured = patch_popen_capture(monkeypatch, returncode=7, with_shell=True)

    jc = JobContext.__new__(JobContext)
    jc.envs = {"A": "1", "B": "2"}

    wdir = str(tmp_path / "run")
    err = JobContext._run_one_subprocess(jc, args=args, working_dir=wdir, extra_envs={"B": "override", "C": "3"})

    assert err == 7
    assert captured["cwd"] == wdir
    assert captured["shell"] is expected_shell
    assert captured["env"] == {"A": "1", "B": "override", "C": "3"}


def test_run_one_call_subprocess_uses_default_working_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_run_one_subprocess(self, args, working_dir, extra_envs=None):
        captured["args"] = args
        captured["working_dir"] = working_dir
        captured["extra_envs"] = extra_envs
        return 0

    jc = JobContext.__new__(JobContext)
    jc.working_dir = str(tmp_path / "default")
    monkeypatch.setattr(JobContext, "_run_one_subprocess", fake_run_one_subprocess)

    err = jc.run_one_call(["echo", "hi"], extra_envs={"K": "V"})

    assert err == 0
    assert captured["args"] == ["echo", "hi"]
    assert captured["working_dir"] == str(tmp_path / "default")
    assert captured["extra_envs"] == {"K": "V"}


def test_run_one_call_subprocess_uses_custom_working_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_run_one_subprocess(self, args, working_dir, extra_envs=None):
        captured["working_dir"] = working_dir
        return 0

    jc = JobContext.__new__(JobContext)
    jc.working_dir = str(tmp_path / "default")
    monkeypatch.setattr(JobContext, "_run_one_subprocess", fake_run_one_subprocess)

    custom = str(tmp_path / "custom")
    jc.run_one_call(["echo"], working_dir=custom)

    assert captured["working_dir"] == custom


def test_run_one_call_python_function_calls_callable():
    called = []

    def f(a, b):
        called.append((a, b))

    jc = JobContext.__new__(JobContext)
    err = jc.run_one_call([f, 1, 2])

    assert err == 0
    assert called == [(1, 2)]


def test_run_one_subprocess_nowait_merges_env_and_uses_default_wdir(monkeypatch, tmp_path):
    captured = patch_popen_capture(monkeypatch, with_shell=False)

    jc = JobContext.__new__(JobContext)
    jc.envs = {"A": "1"}
    jc.working_dir = str(tmp_path / "wd")

    p = jc.run_one_subprocess_nowait(["cmd"], working_dir=None, extra_envs={"B": "2"})

    assert isinstance(p, FakeProc)
    assert captured["cwd"] == str(tmp_path / "wd")
    assert captured["env"] == {"A": "1", "B": "2"}


def test_stage_done_and_write_done(tmp_path):
    jc = JobContext.__new__(JobContext)
    jc.progress_dir = str(tmp_path / "progress")
    Path(jc.progress_dir).mkdir(parents=True, exist_ok=True)

    stage = ABStage.makebox
    assert jc.check_stage_done(stage) is False

    jc.write_stage_done(stage, extra_info_dict={"foo": "bar"})
    done_path = Path(jc.progress_dir) / f"{stage.value}.done"
    assert done_path.is_file()

    assert jc.check_stage_done(stage) is True

    payload = json.loads(done_path.read_text())
    assert payload == {"foo": "bar"}


def test_job_context_init_accepts_none_extra_envs(monkeypatch, tmp_path):
    calls = []

    def fake_run_one_subprocess(self, args, working_dir, extra_envs=None):
        calls.append((args, working_dir, extra_envs))
        return 0

    monkeypatch.setattr(JobContext, "_run_one_subprocess", fake_run_one_subprocess)

    jc = JobContext(str(tmp_path / "wd"), extra_envs=None, use_mps=False)
    assert isinstance(jc.envs, dict)
    assert len(calls) == 1
    assert calls[0][2] is None


def test_run_one_subprocess_defaults_extra_envs_to_empty(monkeypatch, tmp_path):
    captured = patch_popen_capture(monkeypatch, with_shell=True)

    jc = JobContext.__new__(JobContext)
    jc.envs = {"A": "1"}
    err = JobContext._run_one_subprocess(jc, args=["echo"], working_dir=str(tmp_path), extra_envs=None)

    assert err == 0
    assert captured["env"] == {"A": "1"}


def test_run_one_subprocess_nowait_defaults_extra_envs_to_empty(monkeypatch, tmp_path):
    captured = patch_popen_capture(monkeypatch, with_shell=False)

    jc = JobContext.__new__(JobContext)
    jc.envs = {"A": "1"}
    jc.working_dir = str(tmp_path / "wd")
    p = jc.run_one_subprocess_nowait(["cmd"], working_dir=None, extra_envs=None)

    assert isinstance(p, FakeProc)
    assert captured["env"] == {"A": "1"}
