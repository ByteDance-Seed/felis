# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np


class _Residue:

    def __init__(self, resname: str, resid: int):
        self.resname = resname
        self.resid = resid


class _Atom:

    def __init__(self, index: int, residue: _Residue):
        self.index = index
        self.residue = residue


class _Frame:

    def __init__(self, natoms: int):
        self.positions = np.zeros((natoms, 3), dtype=float)
        self.dimensions = np.array([10, 10, 10, 90, 90, 90], dtype=float)


class _Universe:

    def __init__(self, frames, atoms):
        self.trajectory = frames
        self.atoms = atoms


def test_collect_candidate_pairs_from_distance(tmp_path, monkeypatch):
    import felis.protocols.boresch.anchor_distance as mod

    atom_ids_path = tmp_path / "atom_ids.yaml"
    atom_ids_path.write_text("""
protein_backbone:
  system:
    - [100, 101]
ligands:
  M00:
    - [200, 201]
""".strip())

    # Dummy universe with 2 frames and atoms; p1 always 100 -> residue ALA1.
    residue = _Residue("ALA", 1)
    atoms = {100: _Atom(100, residue), 101: _Atom(101, residue)}
    u = _Universe([_Frame(202) for _ in range(2)], atoms)

    class _Sdf:

        def find_nonterminal_heavy_atom_near(self, idx, _excluded):
            # mark atom 0 as not suitable -> triggers symbl_lig[0] = "0" branch
            if idx == 0:
                return None
            return idx

    monkeypatch.setattr(mod, "get_top_system_symbols", lambda _top: ["C"] * 300)
    monkeypatch.setattr(mod, "distance_array", lambda *_args, **_kwargs: np.array([[1.0, 2.0]]))
    monkeypatch.setattr(mod, "argsort_2d_array", lambda _m: np.array([[0, 0]]))

    candidates, thr = mod.collect_candidate_pairs_from_distance(u,
                                                                "top",
                                                                str(atom_ids_path),
                                                                _Sdf(),
                                                                set(),
                                                                total_frames=2)
    assert thr == int(2 * mod.FREQ_THRESHOLD_DIST)
    assert candidates == [((201, "ALA1", "VdWContact"), [0, 1])]
