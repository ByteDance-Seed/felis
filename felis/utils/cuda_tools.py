# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Query CUDA device visibility.

This module contains lightweight helpers for detecting which CUDA devices are
available to the current process. The primary source of truth is
`CUDA_VISIBLE_DEVICES`; when it is not set, the helpers fall back to querying
`nvidia-smi`.
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def get_visible_cuda_devices() -> list[str]:
    """Return CUDA device IDs visible to the current process.

    The lookup follows this precedence:

    1. If `CUDA_VISIBLE_DEVICES` is set, return its comma-separated entries
       (trimmed, excluding empty tokens).
    2. Otherwise, run `nvidia-smi -L` and return a dense 0-based list
       (e.g., `['0', '1']`) whose length matches the number of GPUs reported.
    3. If GPU detection fails (e.g., `nvidia-smi` missing), return an empty
       list.

    Returns:
        list[str]: Visible CUDA device identifiers as strings.
    """
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_value is not None:
        return [
            device_id.strip()
            for device_id in env_value.split(",")
            if device_id.strip()
        ]

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
        )
        return [str(i) for i in range(result.stdout.count("GPU "))]
    except Exception:
        return []


def get_cuda_device_indices() -> list[str]:
    """Return dense logical ordinals for OpenMM ``DeviceIndex``.

    Visibility tokens such as physical IDs or GPU UUIDs belong in
    ``CUDA_VISIBLE_DEVICES``. Once that environment is applied, CUDA remaps the
    visible devices to dense logical ordinals, which are the values OpenMM must
    receive.
    """
    return [str(index) for index, _device in enumerate(get_visible_cuda_devices())]


def resolve_cuda_device_ids(requested_ids: list[str] = None) -> list[str]:
    """Resolve requested logical GPU IDs within inherited CUDA visibility.

    When ``CUDA_VISIBLE_DEVICES`` is set, integer requests are logical ordinals
    into that inherited list. This prevents a child process from escaping a
    scheduler-provided device allocation. Non-integer identifiers must already
    appear in the inherited list. Without inherited visibility, requests are
    treated as physical IDs or UUIDs.
    """
    visible_ids = get_visible_cuda_devices()
    if requested_ids is None:
        return visible_ids

    requested = [str(device_id).strip() for device_id in requested_ids]
    if any(not device_id for device_id in requested):
        raise ValueError("requested CUDA device IDs must not be empty")

    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        return requested
    if not visible_ids:
        raise ValueError("No CUDA devices are visible in CUDA_VISIBLE_DEVICES")

    resolved = []
    for device_id in requested:
        try:
            logical_id = int(device_id)
        except ValueError:
            if device_id not in visible_ids:
                raise ValueError(
                    f"CUDA device {device_id!r} is outside inherited visibility "
                    f"{visible_ids}"
                )
            resolved.append(device_id)
            continue
        if logical_id < 0 or logical_id >= len(visible_ids):
            raise ValueError(
                f"CUDA logical device {logical_id} is outside inherited "
                f"visibility {visible_ids}"
            )
        resolved.append(visible_ids[logical_id])
    return resolved
