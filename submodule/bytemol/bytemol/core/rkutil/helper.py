# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Iterable, List, Tuple

from PIL import ImageDraw, ImageFont
from PIL.Image import Image
from rdkit import Chem

from bytemol.utils.environ import find_default_font

logger = logging.getLogger(__name__)


def add_text(img: Image, text: str, fontsize: int = 24, xy: Tuple[int, int] = (0, 0)) -> Image:
    fnt = ImageFont.truetype(font=find_default_font(), size=fontsize)
    drawer = ImageDraw.Draw(img)
    drawer.text(xy, text, font=fnt, fill=(0, 0, 0))
    return img


def dedup_rkmols(mols: Iterable[Chem.Mol]) -> List[Chem.Mol]:
    seen = set()
    result = []
    for mol in mols:
        mol_copy = Chem.AddHs(mol)
        smiles = Chem.MolToSmiles(mol_copy, isomericSmiles=True, canonical=True)
        if smiles not in seen:
            seen.add(smiles)
            result.append(mol)
    return result


def sorted_tuple(atom_tuple: Tuple[int]) -> Tuple[int]:
    ''' put smaller numbers first, to ensure unique representation for equivalent
        bonds, angles and dihedrals.
        e.g., (3,1,2)->(2,1,3), (4,3,2,1)->(1,2,3,4), (1,3,2,1)->(1,2,3,1)
        the last case may not occur in our program.
    '''
    for i in range(len(atom_tuple) // 2):
        if atom_tuple[i] > atom_tuple[-i - 1]:
            return tuple(atom_tuple[::-1])
        elif atom_tuple[i] < atom_tuple[-i - 1]:
            return tuple(atom_tuple)
    return tuple(atom_tuple)


def sorted_atomids(atomids: Iterable[int], is_improper: bool = False) -> Tuple:
    atomids = list(atomids)
    if is_improper:
        assert len(atomids) == 4
        return tuple([atomids[0]] + sorted(atomids[1:]))
    else:
        assert 0 < len(atomids) < 5
        if len(atomids) == 1:
            return tuple(atomids)
        return sorted_tuple(atomids)
