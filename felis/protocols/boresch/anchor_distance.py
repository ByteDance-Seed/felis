# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
import logging

import MDAnalysis
from MDAnalysis.lib.distances import distance_array
import numpy as np

from felis.configs import load_config
from felis.external.prolif_tools import ProteinLigandInteraction
from felis.protocols.boresch.anchor_constants import FREQ_THRESHOLD_DIST
from felis.protocols.boresch.anchor_ligand import find_certain_elements_default
from felis.utils.numpy_tools import argsort_2d_array
from felis.utils.topfile_tools import get_top_system_symbols

logger = logging.getLogger(__name__)


def collect_candidate_pairs_from_distance(u: MDAnalysis.Universe, gmxtop: str, atom_ids_file: str, sdfmol,
                                          excluded_atoms: set[int], total_frames: int):
    with open(atom_ids_file) as f_atom_ids_file:
        atom_ids = load_config(f_atom_ids_file)
    backbone_atoms = atom_ids["protein_backbone"]["system"][0]
    ligand_atoms = atom_ids["ligands"]["M00"][0]
    ligand_natoms = len(ligand_atoms)

    ligand_atoms_heavy_mask = np.zeros(ligand_natoms, dtype=bool)
    for i in range(ligand_natoms):
        ii = sdfmol.find_nonterminal_heavy_atom_near(i, excluded_atoms)
        if ii is not None:
            ligand_atoms_heavy_mask[ii] = True

    symbl_cpx = get_top_system_symbols(gmxtop)
    symbl_lig = [symbl_cpx[i] for i in ligand_atoms]
    for i in range(ligand_natoms):
        if ligand_atoms_heavy_mask[i] == 0:
            symbl_lig[i] = "0"
    selected_ligand_atoms = find_certain_elements_default(symbl_lig)
    selected_ligatoms_cpx = [ligand_atoms[i] for i in selected_ligand_atoms]

    frame_to_ixn0: dict[int, tuple[int, int]] = {}
    for nprocessed, t in enumerate(u.trajectory):
        pos_cpx = t.positions
        dims = t.dimensions
        selected_ligand_pos = pos_cpx[selected_ligatoms_cpx]
        backbone_pos = pos_cpx[backbone_atoms]
        lig_bb_dist_matrix = distance_array(selected_ligand_pos, backbone_pos, box=dims, backend="OpenMP")
        min_lig_bb_idx = argsort_2d_array(lig_bb_dist_matrix)
        l1idx, p1idx = min_lig_bb_idx[0]
        l1 = selected_ligatoms_cpx[l1idx]
        p1 = backbone_atoms[p1idx]
        if lig_bb_dist_matrix[l1idx, p1idx] <= max(ProteinLigandInteraction.VICINITY_CUTOFF, 10.0):
            frame_to_ixn0[nprocessed] = (l1, p1)
        if (nprocessed % 100 == 0):
            logger.info(f"processed {nprocessed:8d}")

    pair_to_ixn1: dict[str, Counter] = {}
    for _frame_no, (l1, p1) in frame_to_ixn0.items():
        p1res = u.atoms[p1].residue  # pylint: disable=unsubscriptable-object
        residue_name = f"{p1res.resname}{p1res.resid}"
        if residue_name not in pair_to_ixn1:
            pair_to_ixn1[residue_name] = Counter()
        pair_to_ixn1[residue_name][l1] += 1

    pair_to_frames: dict[tuple[int, str, str], list[int]] = {}
    for frame_no, (_l1, p1) in frame_to_ixn0.items():
        p1res = u.atoms[p1].residue  # pylint: disable=unsubscriptable-object
        residue_name = f"{p1res.resname}{p1res.resid}"
        l1_most_common = pair_to_ixn1[residue_name].most_common(1)[0][0]
        atom_res_pair = (l1_most_common, residue_name, "VdWContact")
        pair_to_frames.setdefault(atom_res_pair, []).append(frame_no)

    n_frames_threshold = int(total_frames * FREQ_THRESHOLD_DIST)
    candidate_pair_to_frames = [
        (pair, frames) for (pair, frames) in pair_to_frames.items() if len(frames) >= n_frames_threshold
    ]
    logger.info(f"Found {len(candidate_pair_to_frames)} candidate interactions.")
    return candidate_pair_to_frames, n_frames_threshold
