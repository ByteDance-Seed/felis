# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

import MDAnalysis

from felis.protocols.boresch.anchor_constants import EXCLUDED_SMARTS, VALID_ANGLE_RANGE
from felis.protocols.boresch.anchor_geometry_filters import filter_frames_based_on_angle
from felis.protocols.boresch.anchor_ligand import SDFMolecule

logger = logging.getLogger(__name__)


def expand_candidate_to_anchor_results(sdfmol: SDFMolecule, mda_ligand, u: MDAnalysis.Universe, n_frames_threshold: int,
                                       ligand_idx_A: int, pro1: int, pro2: int, pro3: int, frames: list[int],
                                       res_name: str, ixn_type: str) -> list[dict]:
    """Given a candidate (ligand atom A + protein residue backbone), enumerate possible anchor triples."""
    bonds = sdfmol.get_bonds()
    excluded_atoms: set[int] = sdfmol.get_excluded_atoms(EXCLUDED_SMARTS)

    if ligand_idx_A in excluded_atoms:
        logger.info(f"Ligand atom {ligand_idx_A} is excluded by the smarts patterns.")
        return []

    # find B from A, and C from B
    ligand_idx_B_list = [idx for idx in bonds[ligand_idx_A] if sdfmol.is_heavy(idx) and not sdfmol.is_terminal(idx)]
    ligand_idx_BC_list: list[tuple[int, int]] = []
    for idxB in ligand_idx_B_list:
        ligand_idx_BC_list.extend([(idxB, idx)
                                   for idx in bonds[idxB]
                                   if sdfmol.is_heavy(idx) and not sdfmol.is_terminal(idx) and idx != ligand_idx_A])
    global_lig_BC = [(mda_ligand[idxB].index, mda_ligand[idxC].index) for (idxB, idxC) in ligand_idx_BC_list]
    global_lig_A = mda_ligand[ligand_idx_A].index
    global_pro_a = pro1
    global_pro_b = pro2
    global_pro_c = pro3

    results: list[dict] = []
    valid_frames_1, mean_diff1 = filter_frames_based_on_angle(u, frames, (global_pro_a, global_pro_b, global_pro_c),
                                                              VALID_ANGLE_RANGE)
    valid_frames_2, mean_diff2 = filter_frames_based_on_angle(u, frames, (global_lig_A, global_pro_a, global_pro_b),
                                                              VALID_ANGLE_RANGE)
    for (global_lig_B, global_lig_C) in global_lig_BC:
        valid_frames_3, mean_diff3 = filter_frames_based_on_angle(u, frames, (global_lig_A, global_lig_B, global_lig_C),
                                                                  VALID_ANGLE_RANGE)
        valid_frames_4, mean_diff4 = filter_frames_based_on_angle(u, frames, (global_pro_a, global_lig_A, global_lig_B),
                                                                  VALID_ANGLE_RANGE)
        if len(valid_frames_3) >= n_frames_threshold and len(valid_frames_4) >= n_frames_threshold:
            results.append({
                "res_name": res_name,
                "ixn_type": ixn_type,
                "n_Aab": len(valid_frames_2),
                "n_aAB": len(valid_frames_4),
                "diff_Aab": mean_diff2,
                "diff_aAB": mean_diff4,
                "abc": [int(global_pro_a), int(global_pro_b), int(global_pro_c)],
                "ABC": [int(global_lig_A), int(global_lig_B), int(global_lig_C)],
                "n_abc": len(valid_frames_1),
                "n_ABC": len(valid_frames_3),
                "diff_abc": mean_diff1,
                "diff_ABC": mean_diff3,
            })
    return results
