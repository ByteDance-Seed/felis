# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Lightweight filesystem/path helpers.

This module contains a few small utilities used throughout FELIS for:

- Normalizing user-provided paths to absolute paths.
- Creating directories and producing output file paths.
- Copying directory trees while tolerating pre-existing destinations.

The functions intentionally return plain strings in a few places to ease
serialization (e.g., when embedding paths into YAML/JSON configs).
"""

import logging
from pathlib import Path
import shutil
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_path(path) -> str:
    """Resolve a filesystem path to an absolute path string.

    Unlike :meth:`pathlib.Path.resolve`, this helper uses ``strict=False`` so it
    can resolve paths that do not exist yet (common for output paths).

    Args:
        path: A path-like value, either a string or a :class:`pathlib.Path`.

    Returns:
        The resolved absolute path as a string.
    """
    if isinstance(path, Path):
        p = path
    else:
        p = Path(path)
    resolved = p.resolve(strict=False)
    return str(resolved)


def mkdir_and_get_filepath(basedir: Optional[str], dirname: Optional[str], filename: str, ext: Optional[str]) -> str:
    """Create a directory (if needed) and return a file path under it.

    The directory path is built as ``basedir/dirname`` with either component
    optional. The directory is created with ``parents=True`` and
    ``exist_ok=True``.

    The returned path is ``<dir>/<filename>`` and, if ``ext`` is provided, its
    suffix is set via :meth:`pathlib.Path.with_suffix`.

    Notes:
        ``ext`` should typically include a leading dot (e.g., ``".yaml"``).
        Passing ``"yaml"`` will result in a suffix of ``".yaml"`` not being
        applied as you might expect.

    Args:
        basedir: Base directory (string) or ``None``.
        dirname: Subdirectory name (string) or ``None``.
        filename: File name without extension.
        ext: File extension (including dot) or ``None``.

    Returns:
        The constructed file path as a string.
    """
    new_path = Path()
    if basedir:
        new_path = new_path / Path(basedir)
    if dirname:
        new_path = new_path / Path(dirname)
    new_path.mkdir(parents=True, exist_ok=True)

    new_path = new_path / Path(filename)
    if ext:
        if ext.startswith("."):
            new_path = new_path.with_suffix(ext)
        else:
            dot_ext = "." + ext
            new_path = new_path.with_suffix(dot_ext)
    return str(new_path)


def copy_tree(src: str, dst: str) -> None:
    """Copy a directory tree from ``src`` to ``dst``.

    This is a thin wrapper around :func:`shutil.copytree` with
    ``dirs_exist_ok=True`` to allow copying into an existing directory.

    Args:
        src: Source directory path.
        dst: Destination directory path.
    """
    if resolve_path(src) == resolve_path(dst):
        return
    Path(dst).mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
