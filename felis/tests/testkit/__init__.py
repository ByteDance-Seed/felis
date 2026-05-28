# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from typing import Iterable


def _list_relative_files(root: Path) -> list[str]:

    files: list[str] = []
    for p in root.rglob("*"):
        if p.is_file():
            files.append(p.relative_to(root).as_posix())
    return sorted(files)


def _format_files_block(label: str, paths: Iterable[str]) -> str:
    paths = sorted(paths)
    if not paths:
        return f"{label} (0): []"
    listing = "\n  ".join(paths)
    return f"{label} ({len(paths)}):\n  {listing}"


def assert_manifest(root: Path, expected_rel_files: Iterable[str]) -> None:
    """Assert the set of files under root equals the expected manifest."""

    actual = set(_list_relative_files(root))
    expected = set(expected_rel_files)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        message = "\n".join([
            f"Manifest mismatch under {root}:",
            _format_files_block("Missing (expected but not found)", missing),
            _format_files_block("Unexpected (found but not expected)", unexpected),
        ])
        raise AssertionError(message)


def assert_manifest_contains(root: Path, required_rel_files: Iterable[str]) -> None:
    """Assert the set of files under root contains every entry in required_rel_files.

    Unlike :func:`assert_manifest`, this does not fail when extra files are present;
    it only checks that every required file exists.
    """

    actual = set(_list_relative_files(root))
    required = set(required_rel_files)
    missing = required - actual
    if missing:
        message = "\n".join([
            f"Required files missing under {root}:",
            _format_files_block("Missing", missing),
        ])
        raise AssertionError(message)


_spec = importlib.util.find_spec("felis")
_package_dir = Path(_spec.origin).parent
_project_root = str(_package_dir.parent)


def package_root() -> str:
    return _project_root
