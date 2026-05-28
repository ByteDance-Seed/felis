# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Load and dump small configuration objects.

This module provides thin wrappers around YAML or JSON loading and JSON dumping for
configuration dictionaries used across the FELIS workflows.

Notes:
    - Loading is performed with `yaml.safe_load` to avoid executing arbitrary
      Python objects.
    - Dumping is performed with `json.dump` for deterministic, portable output.
"""

import json
import logging
from typing import Any, TextIO

import yaml

logger = logging.getLogger(__name__)


def load_config(file_handle: TextIO) -> Any:
    """Load a YAML or JSON configuration from an open file handle.

    Args:
        file_handle (TextIO): Open file-like object positioned at the start of
            a YAML or JSON document.

    Returns:
        Any: Parsed YAML or JSON content. This is typically a mapping for FELIS
            configuration files, but may also be `None` for an empty file.
    """
    return yaml.safe_load(file_handle)


def dump_config(data: dict, file_handle: TextIO, indent: int | None = None) -> None:
    """Dump a configuration mapping as JSON to an open file handle.

    Args:
        data (dict): Configuration mapping to serialize.
        file_handle (TextIO): Open file-like object to write JSON into.
        indent (int): JSON indentation level. Use a negative value (default)
            or `None` to produce compact JSON.

    Raises:
        TypeError: If `data` contains non-JSON-serializable objects.
        OSError: If writing to `file_handle` fails.
    """
    if indent is None or indent < 0:
        json.dump(data, file_handle)
    else:
        json.dump(data, file_handle, indent=indent)