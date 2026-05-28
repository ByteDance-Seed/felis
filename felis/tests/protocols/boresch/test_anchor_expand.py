# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from felis.tests.protocols.boresch.conftest import make_lig_atoms, make_sdf


def test_expand_candidate_to_anchor_results_excluded_atom(monkeypatch):
    import felis.protocols.boresch.anchor_expand as mod

    sdf = make_sdf(excluded={0})
    mda_ligand = make_lig_atoms(start_index=100, count=4)

    # avoid needing a real Universe
    monkeypatch.setattr(mod, "filter_frames_based_on_angle", lambda *_a, **_k: ([0, 1, 2], 0.0))
    out = mod.expand_candidate_to_anchor_results(sdf,
                                                 mda_ligand,
                                                 u=None,
                                                 n_frames_threshold=2,
                                                 ligand_idx_A=0,
                                                 pro1=10,
                                                 pro2=11,
                                                 pro3=12,
                                                 frames=[0, 1, 2],
                                                 res_name="ALA1",
                                                 ixn_type="HBDonor")
    assert out == []


def test_expand_candidate_to_anchor_results_generates_one_result(monkeypatch):
    import felis.protocols.boresch.anchor_expand as mod

    sdf = make_sdf(excluded=set())
    mda_ligand = make_lig_atoms(start_index=100, count=4)

    def fake_filter(_u, frames, angle_atoms, _valid_range):
        # Make all (A,B,C) and (a,A,B) checks pass threshold.
        if angle_atoms[0] == 100 or angle_atoms[:2] == (10, 100):
            return frames[:2], 5.0
        return frames, 1.0

    monkeypatch.setattr(mod, "filter_frames_based_on_angle", fake_filter)
    out = mod.expand_candidate_to_anchor_results(sdf,
                                                 mda_ligand,
                                                 u=None,
                                                 n_frames_threshold=2,
                                                 ligand_idx_A=0,
                                                 pro1=10,
                                                 pro2=11,
                                                 pro3=12,
                                                 frames=[0, 1, 2],
                                                 res_name="ALA1",
                                                 ixn_type="HBDonor")
    assert len(out) >= 1
    assert any(r["ABC"] == [100, 101, 102] for r in out)
    got = next(r for r in out if r["ABC"] == [100, 101, 102])
    assert got["abc"] == [10, 11, 12]
    assert got["n_Aab"] == 2
    assert got["n_aAB"] == 2
    assert got["n_ABC"] == 2


def test_expand_candidate_to_anchor_results_threshold_not_met(monkeypatch):
    import felis.protocols.boresch.anchor_expand as mod

    sdf = make_sdf(excluded=set())
    mda_ligand = make_lig_atoms(start_index=100, count=4)

    def fake_filter(_u, frames, angle_atoms, _valid_range):
        # Make all (A,B,C) and (a,A,B) checks fail threshold.
        if angle_atoms[0] == 100 or angle_atoms[:2] == (10, 100):
            return frames[:1], 5.0
        return frames, 1.0

    monkeypatch.setattr(mod, "filter_frames_based_on_angle", fake_filter)
    out = mod.expand_candidate_to_anchor_results(sdf,
                                                 mda_ligand,
                                                 u=None,
                                                 n_frames_threshold=2,
                                                 ligand_idx_A=0,
                                                 pro1=10,
                                                 pro2=11,
                                                 pro3=12,
                                                 frames=[0, 1, 2],
                                                 res_name="ALA1",
                                                 ixn_type="HBDonor")
    assert out == []
