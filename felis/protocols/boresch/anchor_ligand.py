# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Optional

import rdkit.Chem as Chem


def up_to_n_bonds(n: int, idx: int, bonds: list[list[int]], sort: bool = False) -> list[int]:
    db = defaultdict(set)
    for a1, a2list in enumerate(bonds):
        for a2 in a2list:
            db[a1].add(a2)
            db[a2].add(a1)

    inbatoms: list[int] = []
    inbatoms.append(idx)
    while n > 0:
        n = n - 1
        to_add: list[int] = []
        for a in inbatoms:
            for ai in db[a]:
                to_add.append(ai)
        for a in to_add:
            if a not in inbatoms:
                inbatoms.append(a)
    inblist: list[int] = []
    for a in inbatoms:
        if a != idx:
            inblist.append(a)
    if sort:
        return sorted(inblist)
    return inblist


def _find_certain_elements(symb: list[str], elements: list[str]) -> list[int]:
    found: list[int] = []
    for i, s in enumerate(symb):
        if s in elements:
            found.append(i)
    if len(found) == 0:
        raise ValueError(f"Molecule does not contain {elements} atoms.")
    return found


def find_certain_elements_default(symb: list[str]) -> list[int]:
    elists: list[list[str]] = [["N", "O"]]
    elists.append(elists[-1] + ["P", "S"])
    elists.append(elists[-1] + ["I", "Br", "Cl", "F"])
    elists.append(elists[-1] + ["Si", "C"])
    for el in elists:
        try:
            return _find_certain_elements(symb, el)
        except ValueError:
            continue
    raise ValueError("Cannot find anchor elements for Boresch restraints.")


class SDFMolecule:

    def __init__(self, sdf_file: str) -> None:
        self._mol = Chem.SDMolSupplier(sdf_file, removeHs=False)[0]
        self._elements: list[str] | None = None
        self._bonds: list[list[int]] | None = None
        self._nonterminal_heavy_cache: dict[int, Optional[int]] = {}

    def get_numatoms(self) -> int:
        return self._mol.GetNumAtoms()

    def get_elements(self) -> list[str]:
        if self._elements is not None:
            return self._elements
        self._elements = [atom.GetSymbol() for atom in self._mol.GetAtoms()]
        return self._elements

    def get_bonds(self) -> list[list[int]]:
        if self._bonds is not None:
            return self._bonds
        self._bonds = [[] for _ in self.get_elements()]
        for bond in self._mol.GetBonds():
            atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
            self._bonds[atom1.GetIdx()].append(atom2.GetIdx())
            self._bonds[atom2.GetIdx()].append(atom1.GetIdx())
        for bond in self._bonds:
            bond.sort()
        return self._bonds

    def is_heavy(self, idx: int) -> bool:
        return self._mol.GetAtomWithIdx(idx).GetAtomicNum() > 1

    def is_terminal(self, idx: int) -> bool:
        return len(self.get_bonds()[idx]) == 1

    def _get_excluded_atoms_impl(self, smarts: str, index_list: list[int]) -> set[int]:
        pattern = Chem.MolFromSmarts(smarts)
        matches: tuple[tuple] = self._mol.GetSubstructMatches(pattern)
        excluded: list[int] = []
        for m in matches:
            excluded.extend([m[i] for i in index_list])
        return set(excluded)

    def get_excluded_atoms(self, excluded_smarts: dict[str, list[int]]) -> set[int]:
        excluded_atoms: set[int] = set()
        for smarts, index_list in excluded_smarts.items():
            excluded = self._get_excluded_atoms_impl(smarts, index_list)
            excluded_atoms.update(excluded)
        return excluded_atoms

    def find_nonterminal_heavy_atom_near(self, idx: int, excluded_atoms: set[int]) -> Optional[int]:
        if idx in self._nonterminal_heavy_cache:
            return self._nonterminal_heavy_cache[idx]

        bonds = self.get_bonds()
        nbs_all = [idx] + up_to_n_bonds(3, idx, bonds, sort=False)
        for a in nbs_all:
            a_nbs = bonds[a]
            if self.is_heavy(a) and a not in excluded_atoms and len(a_nbs) > 1:
                a_nbs_filtered = [y for y in a_nbs if self.is_heavy(y)]
                if len(a_nbs_filtered) > 1:
                    self._nonterminal_heavy_cache[idx] = a
                    return a
        self._nonterminal_heavy_cache[idx] = None
        return None
