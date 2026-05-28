# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import typing as T

from rdkit import Chem

logger = logging.getLogger(__name__)

num_bondorder = {
    1: Chem.BondType.SINGLE,
    1.5: Chem.BondType.AROMATIC,
    "ar": Chem.BondType.AROMATIC,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}

periodic_table = Chem.GetPeriodicTable()

atomnum_elem: T.Dict[int, str] = {i: periodic_table.GetElementSymbol(i) for i in range(1, 119)}
atomnum_mass: T.Dict[int, float] = {i: periodic_table.GetAtomicWeight(i) for i in range(1, 119)}
elem_atomnum: T.Dict[str, int] = {elem: atomnum for atomnum, elem in atomnum_elem.items()}
elem_mass: T.Dict[str, float] = {elem: atomnum_mass[atomnum] for atomnum, elem in atomnum_elem.items()}


def get_atomnum_by_mass(mass: float) -> int:
    '''find the atomnum closest to mass'''
    retnum = 1
    dif = abs(mass - atomnum_mass[1])
    for anum, amass in atomnum_mass.items():
        adif = abs(mass - amass)
        if adif < dif:
            retnum = anum
            dif = adif
    return retnum
