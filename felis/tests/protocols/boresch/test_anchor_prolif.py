# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

def test_collect_candidate_pairs_from_prolif_filters_and_dedup(monkeypatch):
    import felis.protocols.boresch.anchor_prolif as mod

    # 4 frames contain the same atom/residue/type, with duplicated interactions in-frame.
    frame_to_interactions = {}
    for fr in range(4):
        frame_to_interactions[fr] = {
            "LIG-ALA1": {
                "HBDonor": [
                    {
                        "indices": {
                            "ligand": [0]
                        }
                    },
                    {
                        "indices": {
                            "ligand": [0]
                        }
                    },
                ],
                "Ignored": [{
                    "indices": {
                        "ligand": [1]
                    }
                }],
            }
        }

    monkeypatch.setattr(mod, "analyze_frames_by_prolif", lambda *_args, **_kwargs: frame_to_interactions)

    class _Sdf:

        def find_nonterminal_heavy_atom_near(self, idx, _excluded):
            return idx

    candidates, thr = mod.collect_candidate_pairs_from_prolif(
        "out",
        "top",
        "trj",
        "lig.sdf",
        "protein",
        "resname M00",
        _Sdf(),
        set(),
        total_frames=10,
    )
    assert thr == 4
    assert candidates == [((0, "ALA1", "HBDonor"), [0, 1, 2, 3])]


def test_collect_candidate_pairs_from_prolif_none(monkeypatch):
    import felis.protocols.boresch.anchor_prolif as mod

    monkeypatch.setattr(mod, "analyze_frames_by_prolif", lambda *_args, **_kwargs: None)
    candidates, thr = mod.collect_candidate_pairs_from_prolif(
        "out",
        "top",
        "trj",
        "lig.sdf",
        "protein",
        "resname M00",
        object(),
        set(),
        total_frames=10,
    )
    assert candidates == []
    assert thr == 0
