# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict

from rdkit import Chem

logger = logging.getLogger(__name__)


def find_symmetry_rank(rkmol: Chem.Mol) -> list[int]:
    """Return a canonical rank. 
       Symmetric atoms are assigned the same rank.
       Fails in special cases, i.g. 'C1CN(CCOCCN2CCOCC2)CCO1'.
    """
    rkmol = Chem.Mol(rkmol)

    # modify atom/bond feature
    for bond in rkmol.GetBonds():
        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetIsConjugated(False)
    for atom in rkmol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetIsotope(0)
        atom.SetHybridization(Chem.HybridizationType.S)

    canon_rank = list(Chem.CanonicalRankAtoms(rkmol, breakTies=False, includeChirality=False))
    return canon_rank


def find_equivalent_atoms(rkmol: Chem.Mol) -> dict[int, list[int]]:

    canon_rank = find_symmetry_rank(rkmol)
    book = defaultdict(list)
    for i, rank in enumerate(canon_rank):
        book[rank].append(i)
    equiv_record = defaultdict(list)
    for lst in book.values():
        if len(lst) > 1:
            lst.sort()
            equiv_record[lst[0]] = lst
    return dict(equiv_record)
