# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from felis.utils.file_path import copy_tree
from felis.utils.file_path import mkdir_and_get_filepath
from felis.utils.file_path import resolve_path


def test_resolve_path_accepts_str_and_normalizes(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    p = resolve_path("a/b/../c.txt")
    assert p == str((tmp_path / "a" / "c.txt").resolve(strict=False))


def test_resolve_path_accepts_path_object(tmp_path: Path):
    p = tmp_path / "x" / "y.txt"
    assert resolve_path(p) == str(p.resolve(strict=False))


def test_mkdir_and_get_filepath_creates_parent_dirs(tmp_path: Path):
    out = mkdir_and_get_filepath(str(tmp_path), "d1/d2", "file", ".dat")
    out_p = Path(out)

    assert out_p.suffix == ".dat"
    assert out_p.parent.is_dir()
    assert out_p.parent == tmp_path / "d1" / "d2"


def test_mkdir_and_get_filepath_replaces_suffix_when_ext_given(tmp_path: Path):
    out = mkdir_and_get_filepath(str(tmp_path), None, "file.txt", ".log")
    assert Path(out).name == "file.log"
    assert Path(out).parent == tmp_path


def test_mkdir_and_get_filepath_replaces_suffix_when_no_dot_ext_given(tmp_path: Path):
    out = mkdir_and_get_filepath(str(tmp_path), None, "file.txt", "log")
    assert Path(out).name == "file.log"
    assert Path(out).parent == tmp_path


def test_mkdir_and_get_filepath_supports_relative_basedir_none(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    out = mkdir_and_get_filepath(None, "rel/dir", "file", None)
    # When basedir is None, the function returns a relative path.
    assert Path(out).resolve(strict=False) == (tmp_path / "rel" / "dir" / "file").resolve(strict=False)
    assert (tmp_path / "rel" / "dir").is_dir()


def test_copy_tree_copies_contents(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "a.txt").write_text("hello")

    copy_tree(str(src), str(dst))

    assert (dst / "sub" / "a.txt").read_text() == "hello"


def test_copy_tree_overwrites_existing_files(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("new")
    (dst / "a.txt").write_text("old")

    copy_tree(str(src), str(dst))

    assert (dst / "a.txt").read_text() == "new"


def test_copy_tree_noop_when_src_equals_dst(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("keep")

    copy_tree(str(src), str(src))

    assert (src / "a.txt").read_text() == "keep"
