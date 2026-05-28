# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

from felis.external.prolif_tools import analyze_frames_by_prolif
from felis.external.prolif_tools import ProteinLigandInteraction
from felis.protocols.boresch.anchor_constants import FREQ_THRESHOLD
from felis.protocols.boresch.anchor_constants import PROLIF_JSON

logger = logging.getLogger(__name__)


def collect_candidate_pairs_from_prolif(out_dir: str, gmxtop: str, trj: str, lig_sdf: str, pro_select: str,
                                        lig_select: str, sdfmol, excluded_atoms: set[int],
                                        total_frames: int) -> tuple[list[tuple[tuple[int, str, str], list[int]]], int]:
    frame_to_interactions = analyze_frames_by_prolif(out_dir, PROLIF_JSON, gmxtop, trj, lig_sdf, pro_select, lig_select)
    if frame_to_interactions is None:
        return [], 0

    logger.info(
        f"Focusing on {ProteinLigandInteraction.default_interactions} interactions. Other types of interactions are ignored."
    )
    dummy_pli = ProteinLigandInteraction(interactions=ProteinLigandInteraction.default_interactions)

    pair_to_frames: dict[tuple[int, str, str], list[int]] = {}
    for frame_no, frame in frame_to_interactions.items():
        atom_res_pair_set: set[tuple[int, str, str]] = set()
        for pair_name, pair in frame.items():
            residue_name = pair_name.split("-")[1]
            for interaction_type, all_interactions in pair.items():
                if interaction_type in dummy_pli.interactions:
                    for inter in all_interactions:
                        ligand_atomidx = inter["indices"]["ligand"][0]
                        anchor_atom_idx = sdfmol.find_nonterminal_heavy_atom_near(ligand_atomidx, excluded_atoms)
                        if anchor_atom_idx is not None:
                            atom_res_pair_set.add((anchor_atom_idx, residue_name, interaction_type))
        for atom_res_pair in atom_res_pair_set:
            pair_to_frames.setdefault(atom_res_pair, []).append(frame_no)

    n_frames_threshold = int(total_frames * FREQ_THRESHOLD)
    candidate_pair_to_frames = [
        (pair, frames) for (pair, frames) in pair_to_frames.items() if len(frames) >= n_frames_threshold
    ]
    logger.info(f"Found {len(candidate_pair_to_frames)} candidate interactions.")
    return candidate_pair_to_frames, n_frames_threshold
