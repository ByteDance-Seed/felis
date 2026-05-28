# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

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
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env_value != "":
        ids = [id_.strip() for id_ in env_value.split(",") if id_.strip() != ""]
    else:
        try:
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
            ids = [str(i) for i in range(result.stdout.count("GPU "))]
        except Exception:
            ids = []
    return ids
