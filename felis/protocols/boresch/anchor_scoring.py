# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from enum import Enum
from typing import Optional

import MDAnalysis

from felis.protocols.boresch.anchor_constants import SORT_KEY_RESOLUTION_THRESHOLD
from felis.protocols.boresch.anchor_expand import expand_candidate_to_anchor_results

logger = logging.getLogger(__name__)


class IxnPriority(Enum):
    HBDonor = "HBDonor"
    HBAcceptor = "HBAcceptor"
    Anionic = "Anionic"
    Cationic = "Cationic"
    XBAcceptor = "XBAcceptor"
    XBDonor = "XBDonor"
    CationPi = "CationPi"
    PiCation = "PiCation"
    EdgeToFace = "EdgeToFace"
    FaceToFace = "FaceToFace"
    VdWContact = "VdWContact"

    def priority(self) -> float:
        weights = {
            "HBDonor": 0.6,
            "HBAcceptor": 0.6,
            "Anionic": 0.9,
            "Cationic": 0.9,
            "XBAcceptor": 0.4,
            "XBDonor": 0.4,
            "CationPi": 0.7,
            "PiCation": 0.7,
            "EdgeToFace": 0.1,
            "FaceToFace": 0.1,
            "VdWContact": 0.0,
        }
        return weights.get(self.name, 0.0)

    def __lt__(self, other) -> bool:
        return self.priority() < other.priority()


def find_restraint_from_candidates(sdfmol,
                                   mda_ligand,
                                   u: MDAnalysis.Universe,
                                   candidate_pair_to_frames,
                                   n_frames_threshold: int,
                                   total_frames: int,
                                   residue_to_dssp: Optional[dict[str, tuple[str, int]]] = None,
                                   backup_code_path: bool = False):
    res2dssp = residue_to_dssp if residue_to_dssp is not None else {}
    enhanced_residues_filters = set((
        IxnPriority.Anionic,
        IxnPriority.Cationic,
        IxnPriority.CationPi,
        IxnPriority.PiCation,
        IxnPriority.HBDonor,
        IxnPriority.HBAcceptor,
        IxnPriority.XBDonor,
        IxnPriority.XBAcceptor,
    ))
    enhanced_residues = defaultdict(list)
    merged_cdd_dict: dict[tuple[int, str], tuple[str, list[int]]] = {}
    for (ligand_idx_A, res_name, ixn_type), frames in candidate_pair_to_frames:
        if IxnPriority(ixn_type) in enhanced_residues_filters:
            enhanced_residues[res_name].append(frames)
        if (ligand_idx_A, res_name) not in merged_cdd_dict:
            merged_cdd_dict[(ligand_idx_A, res_name)] = (ixn_type, frames)
        else:
            old_ixn_type, old_frames = merged_cdd_dict[(ligand_idx_A, res_name)]
            old_priority = IxnPriority(old_ixn_type)
            priority = IxnPriority(ixn_type)
            new_priority = max(old_priority, priority)
            new_ixn_type = new_priority.name
            new_frames = sorted(list(set(old_frames) | set(frames)))
            merged_cdd_dict[(ligand_idx_A, res_name)] = (new_ixn_type, new_frames)
    merged_candidate_pair_to_frames = [((k[0], k[1], v[0]), v[1]) for k, v in merged_cdd_dict.items()]
    logger.info(f"After merging, there are {len(merged_candidate_pair_to_frames)} candidate pairs")

    enhanced_residues_counts: dict[str, list[int]] = {}
    for k, v in enhanced_residues.items():
        if len(v) >= 2:
            enhanced_residues_counts[k] = sorted([len(frames) for frames in v], reverse=True)

    all_results: list[dict] = []
    for (ligand_idx_A, res_name, ixn_type), frames in merged_candidate_pair_to_frames:
        logger.info(f"Assessing {ixn_type} ligand atom {ligand_idx_A} and {res_name} from {len(frames)} frames")
        resid = res_name[3:]
        residue_mda_group = u.select_atoms(f"resid {resid}")
        pro_1, pro_2, pro_3, pro_error_found = None, None, None, False
        for atom in residue_mda_group:
            if atom.name == "N":
                if pro_1 is not None:
                    pro_error_found = True
                else:
                    pro_1 = atom.index
            elif atom.name == "CA":
                if pro_2 is not None:
                    pro_error_found = True
                else:
                    pro_2 = atom.index
            elif atom.name == "C":
                if pro_3 is not None:
                    pro_error_found = True
                else:
                    pro_3 = atom.index
        if None in (pro_1, pro_2, pro_3):
            pro_error_found = True
        if pro_error_found:
            logger.warning(f"Did not find anchor atoms from residue {res_name}")
            continue

        perms = [
            (pro_1, pro_2, pro_3),
            (pro_1, pro_3, pro_2),
            (pro_2, pro_1, pro_3),
            (pro_2, pro_3, pro_1),
            (pro_3, pro_1, pro_2),
            (pro_3, pro_2, pro_1),
        ]
        for a, b, c in perms:
            all_results.extend(
                expand_candidate_to_anchor_results(sdfmol, mda_ligand, u, n_frames_threshold, ligand_idx_A, a, b, c,
                                                   frames, res_name, ixn_type))

    sort_key_resolution = int(SORT_KEY_RESOLUTION_THRESHOLD * total_frames)
    sort_key_resolution = max(1, sort_key_resolution)
    if backup_code_path:
        all_results = sorted(all_results,
                             key=lambda x: (-((x["n_aAB"] + x["n_Aab"]) // sort_key_resolution), x["diff_aAB"] + x[
                                 "diff_Aab"], x["diff_abc"] + x["diff_ABC"]))
    else:
        max_n = total_frames
        if len(all_results):
            max_n = max([xn for sublist in [[x["n_aAB"], x["n_Aab"]] for x in all_results] for xn in sublist])
        n_buckets, boost_steps = 50, 3
        step_val = max(10, total_frames // n_buckets)

        def step_func(input_val: int, maxn: int, step: int, maxout: int) -> int:
            offset = maxn - input_val
            if offset >= 0:
                steps_down = offset // step
                return maxout - steps_down
            steps_up = (-offset) // step
            return maxout + steps_up

        all_scores = [x for x in all_results]
        for x in all_scores:
            x_res = x["res_name"]
            x_dssp = res2dssp.get(x_res, ("-", 0))
            x_dssp_score = int(x_dssp[1])

            diff_angles = x["diff_aAB"] + x["diff_Aab"]
            taper_factor = 1.0
            if diff_angles > 60.0:
                taper_factor = 0.7
            elif diff_angles > 40.0:
                taper_factor = 0.7 + (60.0 - diff_angles) / (60.0 - 40.0) * 0.3
            x_aAB = int(taper_factor * x["n_aAB"])
            x_Aab = int(taper_factor * x["n_Aab"])
            x_boost = boost_steps * step_val * x_dssp_score
            x_boost_aAB = x_aAB + x_boost
            x_boost_Aab = x_Aab + x_boost
            x_n_score_aAB = step_func(x_boost_aAB, max_n, step_val, n_buckets)
            x_n_score_Aab = step_func(x_boost_Aab, max_n, step_val, n_buckets)
            x_n_score = x_n_score_aAB + x_n_score_Aab

            x_ixn = x["ixn_type"]
            x_ixn_priority = IxnPriority(x_ixn)
            x_ixn_score = x_ixn_priority.priority()
            x_enhanced_score = 0
            if x_res in enhanced_residues_counts:
                enhanced_nmax0, enhanced_nmax1 = enhanced_residues_counts[x_res][:2]
                if float(enhanced_nmax1) / enhanced_nmax0 > 0.6:
                    x_enhanced_score += int(n_buckets * 0.15)
            x_score = x_n_score * 1 + x_ixn_score * 1 + x_enhanced_score * 1
            x_score = -int(x_score * 100)
            logger.info("\n".join([
                "",
                f"Score: {x_score} {x['abc']} {x['ABC']} {x_res} {x_dssp_score} {x_ixn} {x_ixn_score}",
                f"       enhanced {x_enhanced_score} angle taper {taper_factor:.3f} {x['diff_aAB']:.2f} {x['diff_Aab']:.2f}",
                f"       {x_n_score} boost {x_boost} aAB {x_aAB} {x_n_score_aAB} Aab {x_Aab} {x_n_score_Aab}",
            ]))
            x["ranking_score"] = x_score
        all_results = sorted(all_scores, key=lambda x: (x["ranking_score"], x["abc"], x["ABC"]))

    if len(all_results) == 0:
        logger.warning("No restraint anchors was found from ProLIF interactions.")
        return None, None
    logger.info("Candidates are as follows.")
    for a in all_results:
        logger.info(f"{a}")
    logger.info("Return the first result.")
    res = all_results[0]
    return res["abc"], res["ABC"]
