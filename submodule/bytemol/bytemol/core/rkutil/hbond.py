# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List

from rdkit import Chem

from bytemol.core.rkutil.match_and_map import find_mapped_smarts_matches

logger = logging.getLogger(__name__)


def get_hbond_donors(rkmol: Chem.Mol, extend: bool = False) -> List[int]:
    HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0,+1])]')
    extendedHDonorSmarts = Chem.MolFromSmarts(
        '[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0,+1]),$([C;!H0]),$([c;!H0])]')
    if extend:
        matches = find_mapped_smarts_matches(rkmol, extendedHDonorSmarts)
    else:
        matches = find_mapped_smarts_matches(rkmol, HDonorSmarts)
    return sorted([_[0] for _ in matches])


def get_hbond_acceptors(rkmol: Chem.Mol, extend: bool = False) -> List[int]:
    HAcceptorSmarts = Chem.MolFromSmarts('[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                         '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                         '$([nH0,o,s;+0]),$([F,Cl,Br,I])]')
    matches = find_mapped_smarts_matches(rkmol, HAcceptorSmarts)
    return sorted([_[0] for _ in matches])


def find_intramolecular_hbonds(rkmol: Chem.Mol,
                               conf_id: int = 0,
                               dist_upper_bound: float = 2.5,
                               angle_lower_bound: float = 120,
                               extend: bool = False):

    def calc_dist_angle(_conf, i, j, k):
        # X_i-H_j...Y_k
        dist = (_conf.GetAtomPosition(j) - _conf.GetAtomPosition(k)).Length()
        angle = Chem.rdMolTransforms.GetAngleDeg(_conf, i, j, k)
        return dist, angle

    if extend:
        hb_donors = get_hbond_donors(rkmol, extend=True)
        hb_acceptors = get_hbond_acceptors(rkmol, extend=True)
    else:
        hb_donors = get_hbond_donors(rkmol, extend=False)
        hb_acceptors = get_hbond_acceptors(rkmol, extend=False)
    conf = rkmol.GetConformer(conf_id)
    res = []
    for i in hb_donors:
        for hydrogen in rkmol.GetAtomWithIdx(i).GetNeighbors():
            # check is HX1
            if hydrogen.GetAtomicNum() == 1 and hydrogen.GetDegree() == 1:
                for k in hb_acceptors:
                    dist, angle = calc_dist_angle(conf, i, hydrogen.GetIdx(), k)
                    if dist <= dist_upper_bound and angle >= angle_lower_bound:
                        res.append((i, hydrogen.GetIdx(), k, dist, angle))  # X-H...j, dist H...j, angle X-H...j
    return res
