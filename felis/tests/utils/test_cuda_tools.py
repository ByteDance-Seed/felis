# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import subprocess

import pytest

import felis.utils.cuda_tools as cuda_tools


def test_get_visible_cuda_devices_parses_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,, ,3")

    assert cuda_tools.get_visible_cuda_devices() == ["1", "2", "3"]


@pytest.mark.parametrize("env_value", ["", "   "])
def test_get_visible_cuda_devices_empty_env_means_no_devices(
    monkeypatch, env_value
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env_value)

    def fail_if_nvidia_smi_runs(*_args, **_kwargs):
        raise AssertionError("explicit empty CUDA visibility must not fall back")

    monkeypatch.setattr(cuda_tools.subprocess, "run", fail_if_nvidia_smi_runs)

    assert cuda_tools.get_visible_cuda_devices() == []
    assert cuda_tools.get_cuda_device_indices() == []


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


@pytest.mark.parametrize(
    "visible_tokens",
    [
        "4,7",
        "GPU-aaaaaaaa,GPU-bbbbbbbb",
    ],
)
def test_cuda_device_indices_are_dense_for_physical_ids_and_uuids(
    monkeypatch, visible_tokens
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_tokens)

    assert cuda_tools.get_cuda_device_indices() == ["0", "1"]


@pytest.mark.parametrize(
    "visible_tokens,expected",
    [
        ("4,7", ["7", "4"]),
        ("GPU-aaaaaaaa,GPU-bbbbbbbb", ["GPU-bbbbbbbb", "GPU-aaaaaaaa"]),
    ],
)
def test_resolve_cuda_device_ids_maps_inherited_logical_ordinals(
    monkeypatch, visible_tokens, expected
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_tokens)

    assert cuda_tools.resolve_cuda_device_ids(["1", "0"]) == expected
    assert cuda_tools.resolve_cuda_device_ids(None) == visible_tokens.split(",")


def test_resolve_cuda_device_ids_rejects_outside_inherited_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")

    with pytest.raises(ValueError, match="outside inherited visibility"):
        cuda_tools.resolve_cuda_device_ids(["2"])
    with pytest.raises(ValueError, match="outside inherited visibility"):
        cuda_tools.resolve_cuda_device_ids(["GPU-c"])


def test_resolve_cuda_device_ids_preserves_requests_without_inherited_visibility(
    monkeypatch,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert cuda_tools.resolve_cuda_device_ids(["2", "GPU-a"]) == ["2", "GPU-a"]
