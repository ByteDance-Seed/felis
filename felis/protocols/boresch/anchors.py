# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import MDAnalysis

from felis.configs import GlobalKeys
from felis.external.mda_tools import get_mda_universe
from felis.protocols.boresch.anchor_constants import EXCLUDED_SMARTS
from felis.protocols.boresch.anchor_distance import collect_candidate_pairs_from_distance
from felis.protocols.boresch.anchor_dssp import compute_residue_to_dssp
from felis.protocols.boresch.anchor_ligand import SDFMolecule
from felis.protocols.boresch.anchor_prolif import collect_candidate_pairs_from_prolif
from felis.protocols.boresch.anchor_scoring import find_restraint_from_candidates

logger = logging.getLogger(__name__)


def find_boresch_anchors(gmxtop: str, trj: str, out_dir: str, pro_select: str, lig_select: str, lig_sdf: str,
                         atom_ids_file: str) -> tuple[Optional[list], Optional[list]]:
    """Find Boresch anchors (protein p1,p2,p3 and ligand l1,l2,l3) from a trajectory."""
    sdfmol = SDFMolecule(lig_sdf)
    gk = GlobalKeys()
    gk.filename.sys = gmxtop
    u: MDAnalysis.Universe = get_mda_universe(gk, trj)
    mda_ligand = u.select_atoms(lig_select)
    total_frames = len(u.trajectory)
    excluded_atoms = sdfmol.get_excluded_atoms(EXCLUDED_SMARTS)
    residue_to_dssp = compute_residue_to_dssp(u)

    logger.info("Trying to find boresch anchors from ProLIF")
    candidate_pair_to_frames, n_frames_threshold = collect_candidate_pairs_from_prolif(
        out_dir, gmxtop, trj, lig_sdf, pro_select, lig_select, sdfmol, excluded_atoms, total_frames)
    if candidate_pair_to_frames:
        p123, l123 = find_restraint_from_candidates(sdfmol,
                                                    mda_ligand,
                                                    u,
                                                    candidate_pair_to_frames,
                                                    n_frames_threshold,
                                                    total_frames,
                                                    residue_to_dssp,
                                                    backup_code_path=False)
        if p123 is not None and l123 is not None:
            return p123, l123

    logger.info("Trying to find boresch anchors from default method")
    candidate_pair_to_frames, n_frames_threshold = collect_candidate_pairs_from_distance(
        u, gmxtop, atom_ids_file, sdfmol, excluded_atoms, total_frames)
    p123, l123 = find_restraint_from_candidates(sdfmol,
                                                mda_ligand,
                                                u,
                                                candidate_pair_to_frames,
                                                n_frames_threshold,
                                                total_frames,
                                                residue_to_dssp,
                                                backup_code_path=True)
    if p123 is None or l123 is None:
        raise RuntimeError("Failed to find boresch anchors from distance")
    return p123, l123
