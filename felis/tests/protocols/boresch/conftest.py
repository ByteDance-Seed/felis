# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared fake objects for Boresch anchor tests.

These fakes intentionally stay tiny: they only mimic the small subset of the
SDFMolecule / MDAnalysis Universe / atom interfaces touched by the anchor unit
tests so the tests can avoid building real molecular systems.
"""

from typing import Iterable, Mapping


class FakeAtom:
    """Minimal stand-in for an MDAnalysis atom with a name and index."""

    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index


class FakeLigAtom:
    """Minimal stand-in for an MDAnalysis ligand atom holding only an index."""

    def __init__(self, index: int):
        self.index = index


class FakeSdf:
    """Minimal SDFMolecule stand-in returning a fixed 4-membered ring."""

    def __init__(self, excluded: Iterable[int]):
        # a 4-member ring 0-1-2-3-0
        self._bonds = [[1, 3], [0, 2], [1, 3], [2, 0]]
        self._excluded = set(excluded)

    def get_bonds(self):
        return self._bonds

    def get_excluded_atoms(self, _smarts):
        return self._excluded

    def is_heavy(self, _idx: int):
        return True

    def is_terminal(self, _idx: int):
        return False


class FakeResidUniverse:
    """Universe whose ``select_atoms('resid N')`` returns mapped atoms.

    Tests that need to inspect specific residue atoms use this variant.
    """

    def __init__(self, resid_to_atoms: Mapping[int, list]):
        self._resid_to_atoms = dict(resid_to_atoms)

    def select_atoms(self, sel: str):
        assert sel.startswith("resid ")
        resid = int(sel.split()[1])
        return self._resid_to_atoms.get(resid, [])


class FakeTrajectoryUniverse:
    """Universe whose ``select_atoms`` always returns ``[]`` but exposes
    a ``trajectory`` of a configurable length.

    Tests that exercise wrappers/selectors which only count frames or
    delegate downstream calls use this variant.
    """

    def __init__(self, n_frames: int = 10):
        self.trajectory = [object()] * n_frames

    def select_atoms(self, _sel: str):
        return []


def make_lig_atoms(start_index: int = 100, count: int = 4) -> list[FakeLigAtom]:
    """Construct ``count`` consecutive fake ligand atoms starting at ``start_index``."""

    return [FakeLigAtom(start_index + i) for i in range(count)]


def make_sdf(excluded: Iterable[int] = ()) -> FakeSdf:
    """Construct a :class:`FakeSdf` with the given excluded atom indices."""

    return FakeSdf(excluded)


def make_resid_universe(resid_to_atoms: Mapping[int, list]) -> FakeResidUniverse:
    """Construct a :class:`FakeResidUniverse` from a resid -> atoms mapping."""

    return FakeResidUniverse(resid_to_atoms)


def make_trajectory_universe(n_frames: int = 10) -> FakeTrajectoryUniverse:
    """Construct a :class:`FakeTrajectoryUniverse` with ``n_frames`` frames."""

    return FakeTrajectoryUniverse(n_frames=n_frames)
