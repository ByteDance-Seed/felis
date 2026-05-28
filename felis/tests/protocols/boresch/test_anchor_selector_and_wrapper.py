# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from felis.tests.protocols.boresch.conftest import make_trajectory_universe


def test_select_boresch_anchors_prefers_prolif(monkeypatch):
    import felis.protocols.boresch.anchors as mod

    monkeypatch.setattr(mod, "SDFMolecule", lambda _p: type("X", (), {"get_excluded_atoms": lambda *_a, **_k: set()})())
    monkeypatch.setattr(mod, "get_mda_universe", lambda *_a, **_k: make_trajectory_universe())
    monkeypatch.setattr(mod, "compute_residue_to_dssp", lambda _u: {})
    monkeypatch.setattr(mod, "collect_candidate_pairs_from_prolif", lambda *_a, **_k:
                        ([((0, "ALA1", "HBDonor"), [0, 1, 2, 3])], 4))
    monkeypatch.setattr(mod, "find_restraint_from_candidates", lambda *_a, **_k: ([1, 2, 3], [4, 5, 6]))
    monkeypatch.setattr(mod, "collect_candidate_pairs_from_distance", lambda *_a, **_k:
                        (_ for _ in ()).throw(AssertionError("distance path should not be used")))

    p123, l123 = mod.find_boresch_anchors("top", "trj", "out", "protein", "resname M00", "lig.sdf", "atom_ids.yaml")
    assert (p123, l123) == ([1, 2, 3], [4, 5, 6])


def test_select_boresch_anchors_falls_back_to_distance(monkeypatch):
    import felis.protocols.boresch.anchors as mod

    monkeypatch.setattr(mod, "SDFMolecule", lambda _p: type("X", (), {"get_excluded_atoms": lambda *_a, **_k: set()})())
    monkeypatch.setattr(mod, "get_mda_universe", lambda *_a, **_k: make_trajectory_universe())
    monkeypatch.setattr(mod, "compute_residue_to_dssp", lambda _u: {})
    monkeypatch.setattr(mod, "collect_candidate_pairs_from_prolif", lambda *_a, **_k: ([], 0))
    monkeypatch.setattr(mod, "collect_candidate_pairs_from_distance", lambda *_a, **_k:
                        ([((0, "ALA1", "VdWContact"), [0])], 1))
    monkeypatch.setattr(mod, "find_restraint_from_candidates", lambda *_a, **_k: ([7, 8, 9], [10, 11, 12]))

    p123, l123 = mod.find_boresch_anchors("top", "trj", "out", "protein", "resname M00", "lig.sdf", "atom_ids.yaml")
    assert (p123, l123) == ([7, 8, 9], [10, 11, 12])


def test_select_boresch_anchors_raises_if_distance_fails(monkeypatch):
    import felis.protocols.boresch.anchors as mod

    monkeypatch.setattr(mod, "SDFMolecule", lambda _p: type("X", (), {"get_excluded_atoms": lambda *_a, **_k: set()})())
    monkeypatch.setattr(mod, "get_mda_universe", lambda *_a, **_k: make_trajectory_universe())
    monkeypatch.setattr(mod, "compute_residue_to_dssp", lambda _u: {})
    monkeypatch.setattr(mod, "collect_candidate_pairs_from_prolif", lambda *_a, **_k: ([], 0))
    monkeypatch.setattr(mod, "collect_candidate_pairs_from_distance", lambda *_a, **_k:
                        ([((0, "ALA1", "VdWContact"), [0])], 1))
    monkeypatch.setattr(mod, "find_restraint_from_candidates", lambda *_a, **_k: (None, None))

    with pytest.raises(RuntimeError):
        mod.find_boresch_anchors("top", "trj", "out", "protein", "resname M00", "lig.sdf", "atom_ids.yaml")
