# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import pytest

import felis.external.mda_tools as mt


class _DummyGK:
    """Minimal GlobalKeys-like object for unit tests."""

    def __init__(self, ecosystem: str, sys_top: str):

        class _OpenMM:
            params_ecosystem = ecosystem

        class _Filename:
            sys = sys_top

        self.openmm = _OpenMM()
        self.filename = _Filename()


def test_get_mda_universe_gromacs_adds_elements_and_uses_itp(monkeypatch: pytest.MonkeyPatch):
    calls = {"universe": [], "get_top_system_symbols": []}

    class DummyUniverse:

        def __init__(self, gmxtop, trj, topology_format=None):
            calls["universe"].append((gmxtop, trj, topology_format))
            self.topology_attrs = {}

        def add_TopologyAttr(self, name, values):
            self.topology_attrs[name] = values

    monkeypatch.setattr(mt, "Universe", DummyUniverse)

    def _fake_get_top_system_symbols(gmxtop):
        calls["get_top_system_symbols"].append(gmxtop)
        return ["c", "cl", "zn"]

    monkeypatch.setattr(mt, "get_top_system_symbols", _fake_get_top_system_symbols)

    gk = _DummyGK(ecosystem="gromacs", sys_top="a.top")
    u = mt.get_mda_universe(gk, trj="a.xtc")

    assert isinstance(u, DummyUniverse)
    assert calls["universe"] == [("a.top", "a.xtc", "ITP")]
    assert calls["get_top_system_symbols"] == ["a.top"]

    elems = u.topology_attrs["elements"]
    assert isinstance(elems, np.ndarray)
    assert elems.dtype == object
    assert elems.tolist() == ["C", "CL", "ZN"]


def test_get_mda_universe_raises_for_non_gromacs():
    gk = _DummyGK(ecosystem="amber", sys_top="a.top")
    with pytest.raises(NotImplementedError, match="amber"):
        mt.get_mda_universe(gk, trj="a.xtc")


def test_get_trj_nsnapshots_returns_len_of_trajectory(monkeypatch: pytest.MonkeyPatch):

    class DummyUniverse:

        def __init__(self, n):
            self.trajectory = [None] * n

    monkeypatch.setattr(mt, "get_mda_universe", lambda _gk, _trj: DummyUniverse(7))

    gk = _DummyGK(ecosystem="gromacs", sys_top="a.top")
    assert mt.get_trj_nsnapshots(gk, trj="a.xtc") == 7


def test_discard_initial_trajectory_frames_rewrites_remaining_frames(monkeypatch: pytest.MonkeyPatch, tmp_path):
    written_frames = []

    class DummyAtoms:

        n_atoms = 3

        def __init__(self):
            self.current_frame = None

    class DummyTrajectory:

        def __init__(self, atoms):
            self._atoms = atoms
            self._frames = list(range(5))

        def __len__(self):
            return len(self._frames)

        def __iter__(self):
            for frame_id in self._frames:
                self._atoms.current_frame = frame_id
                yield object()

    class DummyUniverse:

        def __init__(self):
            self.atoms = DummyAtoms()
            self.trajectory = DummyTrajectory(self.atoms)

    class DummyWriter:

        def __init__(self, path, n_atoms):
            self.path = path
            self.n_atoms = n_atoms

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, atoms):
            written_frames.append((self.path, self.n_atoms, atoms.current_frame))

    monkeypatch.setattr(mt, "get_mda_universe", lambda _gk, _trj: DummyUniverse())
    monkeypatch.setattr(mt, "Writer", DummyWriter)
    monkeypatch.setattr(mt.os, "replace", lambda src, dst: written_frames.append(("replace", src, dst)))

    trj = tmp_path / "traj.dcd"
    gk = _DummyGK(ecosystem="gromacs", sys_top="a.top")
    mt.discard_initial_trajectory_frames(gk, str(trj), 2)

    tmp_trj = str(tmp_path / "traj.discard.tmp.dcd")
    assert written_frames == [
        (tmp_trj, 3, 2),
        (tmp_trj, 3, 3),
        (tmp_trj, 3, 4),
        ("replace", tmp_path / "traj.discard.tmp.dcd", trj),
    ]


def test_discard_initial_trajectory_frames_rejects_empty_result(monkeypatch: pytest.MonkeyPatch, tmp_path):

    class DummyUniverse:

        atoms = object()
        trajectory = [object(), object()]

    monkeypatch.setattr(mt, "get_mda_universe", lambda _gk, _trj: DummyUniverse())

    gk = _DummyGK(ecosystem="gromacs", sys_top="a.top")
    with pytest.raises(ValueError, match="Cannot discard 2 frames"):
        mt.discard_initial_trajectory_frames(gk, str(tmp_path / "traj.dcd"), 2)
