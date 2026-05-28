# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from felis.protocols.boresch.anchor_constants import EXCLUDED_SMARTS
from felis.protocols.boresch.anchor_ligand import SDFMolecule, find_certain_elements_default, up_to_n_bonds


def _write_sdf(tmp_path, smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    AllChem.UFFOptimizeMolecule(mol)
    path = tmp_path / "mol.sdf"
    w = Chem.SDWriter(str(path))
    w.write(mol)
    w.close()
    return path, mol


def test_get_excluded_atoms_matches_rdkit_substruct(tmp_path):
    sdf_path, mol = _write_sdf(tmp_path, "CS(=O)(=O)CC(=O)O")
    sdfmol = SDFMolecule(str(sdf_path))

    expected = set()
    for smarts, index_list in EXCLUDED_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        for match in mol.GetSubstructMatches(patt):
            expected.update(match[i] for i in index_list)

    assert sdfmol.get_excluded_atoms(EXCLUDED_SMARTS) == expected


def test_find_nonterminal_heavy_atom_near_prefers_nonterminal_and_caches(tmp_path):
    sdf_path, _mol = _write_sdf(tmp_path, "CC(C)C")  # isobutane: atom 1 is the central carbon
    sdfmol = SDFMolecule(str(sdf_path))

    excluded = set()
    assert sdfmol.find_nonterminal_heavy_atom_near(0, excluded) == 1
    # second call hits cache
    assert sdfmol.find_nonterminal_heavy_atom_near(0, excluded) == 1

    # exercise get_elements/get_bonds caches and the None path for nonterminal search
    _ = sdfmol.get_elements()
    _ = sdfmol.get_elements()
    _ = sdfmol.get_bonds()
    _ = sdfmol.get_bonds()
    assert sdfmol.get_numatoms() > 0
    assert sdfmol.is_heavy(0) is True
    # Hydrogens are explicit in the SDF we write, so this carbon is not terminal by the current definition.
    assert sdfmol.is_terminal(0) is False

    sdf_path2, _mol2 = _write_sdf(tmp_path, "C")  # methane: no nonterminal heavy atoms
    sdfmol2 = SDFMolecule(str(sdf_path2))
    assert sdfmol2.find_nonterminal_heavy_atom_near(0, set()) is None
    # cached None
    assert sdfmol2.find_nonterminal_heavy_atom_near(0, set()) is None


def test_find_certain_elements_default_fallbacks_and_errors():
    assert find_certain_elements_default(["0", "C", "0"]) == [1]
    with pytest.raises(ValueError):
        find_certain_elements_default(["0", "0"])


def test_up_to_n_bonds_collects_neighbors():
    # A small linear chain 0-1-2-3
    bonds = [[1], [0, 2], [1, 3], [2]]
    assert up_to_n_bonds(1, 1, bonds, sort=True) == [0, 2]
    assert up_to_n_bonds(2, 1, bonds, sort=True) == [0, 2, 3]
