# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from felis.tests.protocols.boresch.conftest import FakeAtom, make_resid_universe


def test_find_restraint_from_candidates_scores_and_merges(monkeypatch):
    import felis.protocols.boresch.anchor_scoring as mod

    # Resid 1 ok, resid 2 missing atoms -> skipped.
    u = make_resid_universe({
        1: [FakeAtom("N", 10), FakeAtom("CA", 11), FakeAtom("C", 12)],
        2: [FakeAtom("N", 20), FakeAtom("CA", 21)],
        3: [FakeAtom("N", 30), FakeAtom("CA", 31), FakeAtom("C", 32)],
        4: [FakeAtom("N", 40), FakeAtom("N", 41),
            FakeAtom("CA", 42), FakeAtom("C", 43)],
        5: [FakeAtom("N", 50),
            FakeAtom("CA", 51),
            FakeAtom("CA", 52),
            FakeAtom("C", 53),
            FakeAtom("C", 54)],
    })

    # Patch expansion to avoid MDAnalysis angle work; emit deterministic candidates.
    def fake_expand(_sdfmol, _mda_ligand, _u, _thr, ligand_idx_A, pro1, pro2, pro3, _frames, res_name, ixn_type):
        if res_name == "GLY2":
            return []
        if ligand_idx_A == 0:
            return [{
                "res_name": res_name,
                "ixn_type": ixn_type,
                "n_Aab": 20,
                "n_aAB": 20,
                "diff_Aab": 10.0,
                "diff_aAB": 10.0,
                "abc": [pro1, pro2, pro3],
                "ABC": [100, 101, 102],
                "n_abc": 20,
                "n_ABC": 20,
                "diff_abc": 0.0,
                "diff_ABC": 0.0,
            }]
        # Provide one result in the 40..60 branch and one in the >60 branch.
        return [
            {
                "res_name": res_name,
                "ixn_type": ixn_type,
                "n_Aab": 20,
                "n_aAB": 20,
                "diff_Aab": 45.0,
                "diff_aAB": 5.0,
                "abc": [pro1, pro2, pro3],
                "ABC": [200, 201, 202],
                "n_abc": 20,
                "n_ABC": 20,
                "diff_abc": 0.0,
                "diff_ABC": 0.0,
            },
            {
                "res_name": res_name,
                "ixn_type": ixn_type,
                "n_Aab": 20,
                "n_aAB": 20,
                "diff_Aab": 80.0,
                "diff_aAB": 10.0,
                "abc": [pro1, pro2, pro3],
                "ABC": [201, 202, 203],
                "n_abc": 20,
                "n_ABC": 20,
                "diff_abc": 0.0,
                "diff_ABC": 0.0,
            },
        ]

    monkeypatch.setattr(mod, "expand_candidate_to_anchor_results", fake_expand)

    # Two entries for same (ligand,res) with different ixn types should merge to max priority (Anionic).
    candidate_pair_to_frames = [
        ((0, "ALA1", "HBDonor"), [0, 1, 2, 3]),
        ((0, "ALA1", "Anionic"), [2, 3, 4, 5]),
        ((1, "SER3", "HBDonor"), [0, 1, 2, 3, 4, 5]),
        ((2, "GLY2", "HBDonor"), [0, 1, 2, 3, 4, 5]),
        ((3, "VAL4", "HBDonor"), [0, 1, 2, 3, 4, 5]),
        ((4, "LEU5", "HBDonor"), [0, 1, 2, 3, 4, 5]),
    ]
    residue_to_dssp = {"ALA1": ("H", 1), "SER3": ("-", 0)}

    p123, l123 = mod.find_restraint_from_candidates(
        sdfmol=object(),
        mda_ligand=[],
        u=u,
        candidate_pair_to_frames=candidate_pair_to_frames,
        n_frames_threshold=1,
        total_frames=100,
        residue_to_dssp=residue_to_dssp,
        backup_code_path=False,
    )
    assert p123 == [10, 11, 12]
    assert l123 == [100, 101, 102]

    # backup path executes the alternate sorting branch
    p123_b, l123_b = mod.find_restraint_from_candidates(
        sdfmol=object(),
        mda_ligand=[],
        u=u,
        candidate_pair_to_frames=candidate_pair_to_frames,
        n_frames_threshold=1,
        total_frames=100,
        residue_to_dssp=residue_to_dssp,
        backup_code_path=True,
    )
    assert (p123_b, l123_b) == (p123, l123)

    # cover IxnPriority ordering
    assert mod.IxnPriority.HBDonor < mod.IxnPriority.Anionic


def test_find_restraint_from_candidates_empty_returns_none():
    import felis.protocols.boresch.anchor_scoring as mod

    u = make_resid_universe({1: [FakeAtom("N", 1), FakeAtom("CA", 2), FakeAtom("C", 3)]})
    p123, l123 = mod.find_restraint_from_candidates(object(), [],
                                                    u, [],
                                                    1,
                                                    10,
                                                    residue_to_dssp={},
                                                    backup_code_path=False)
    assert p123 is None and l123 is None
