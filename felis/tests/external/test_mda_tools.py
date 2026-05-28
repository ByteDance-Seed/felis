# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

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
