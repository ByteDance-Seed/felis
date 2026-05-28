# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import subprocess

import felis.utils.cuda_tools as cuda_tools


def test_get_visible_cuda_devices_parses_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,, ,3")

    assert cuda_tools.get_visible_cuda_devices() == ["1", "2", "3"]


def test_get_visible_cuda_devices_env_whitespace_means_no_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "   ")

    assert cuda_tools.get_visible_cuda_devices() == []


def test_get_visible_cuda_devices_falls_back_to_nvidia_smi(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def fake_run(args, *, capture_output, text, check):
        assert args == ["nvidia-smi", "-L"]
        assert capture_output is True
        assert text is True
        assert check is True
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="GPU 0: A\nGPU 1: B\n", stderr="")

    monkeypatch.setattr(cuda_tools.subprocess, "run", fake_run)

    assert cuda_tools.get_visible_cuda_devices() == ["0", "1"]


def test_get_visible_cuda_devices_returns_empty_when_detection_fails(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def fake_run(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(cuda_tools.subprocess, "run", fake_run)

    assert cuda_tools.get_visible_cuda_devices() == []
