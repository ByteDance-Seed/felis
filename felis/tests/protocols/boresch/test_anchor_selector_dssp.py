# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np


class _Atom:

    def __init__(self, resname: str, resid: int):
        self.resname = resname
        self.resid = resid


class _AtomGroup:

    def __init__(self, atoms):
        self.atoms = atoms


class _Universe:

    def __init__(self, atoms):
        self._ag = _AtomGroup(atoms)

    def select_atoms(self, _sel: str):
        return self._ag


class _DSSPResults:

    def __init__(self, dssp):
        self.dssp = dssp


class _DummyDSSP:

    def __init__(self, _ag):
        self.results = None

    def run(self):
        return self


def test_compute_residue_to_dssp_2d(monkeypatch):
    import felis.protocols.boresch.anchor_dssp as mod

    atoms = [
        _Atom("ALA", 1),
        _Atom("ALA", 1),
        _Atom("GLY", 2),
    ]
    u = _Universe(atoms)

    dssp_arr = np.array([list("H-"), list("G-")])

    def dssp_factory(_ag):
        d = _DummyDSSP(_ag)
        d.results = _DSSPResults(dssp_arr)
        return d

    monkeypatch.setattr(mod, "DSSP", dssp_factory)
    out = mod.compute_residue_to_dssp(u)
    assert out == {"ALA1": ("H", 1), "GLY2": ("-", 0)}


def test_compute_residue_to_dssp_1d(monkeypatch):
    import felis.protocols.boresch.anchor_dssp as mod

    atoms = [
        _Atom("ALA", 1),
        _Atom("GLY", 2),
    ]
    u = _Universe(atoms)

    dssp_arr = np.array(list("H-"))

    def dssp_factory(_ag):
        d = _DummyDSSP(_ag)
        d.results = _DSSPResults(dssp_arr)
        return d

    monkeypatch.setattr(mod, "DSSP", dssp_factory)
    out = mod.compute_residue_to_dssp(u)
    assert out == {"ALA1": ("H", 1), "GLY2": ("-", 0)}
