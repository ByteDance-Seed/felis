# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import MDAnalysis

from felis.configs import dump_config
from felis.configs import GlobalKeys
from felis.configs import load_config
from felis.external.mda_tools import get_mda_universe
from prolif import Fingerprint
from prolif import Molecule
from prolif.plotting.network import LigNetwork

logger = logging.getLogger(__name__)


class ProteinLigandInteraction:

    allowed_interactions = [
        "HBDonor",
        "HBAcceptor",
        "Anionic",
        "Cationic",
        "XBAcceptor",
        "XBDonor",
        "CationPi",
        "PiCation",
        "VdWContact",
        # "Hydrophobic",
        "EdgeToFace",
        "FaceToFace",
        "MetalAcceptor",
        "MetalDonor",
        # "PiStacking",
    ]
    default_interactions = [
        "HBDonor",
        "HBAcceptor",
        "Anionic",
        "Cationic",
        "XBAcceptor",
        "XBDonor",
        "CationPi",
        "PiCation",
        "EdgeToFace",
        "FaceToFace",
    ]
    VICINITY_CUTOFF = 6.0  # Angstrom
    FREQ_THRESHOLD = 0.05
    MAX_N_JOBS = 8

    def __init__(self, interactions: Optional[Union[str, list[str]]] = None, parameters: Optional[dict] = None) -> None:
        if interactions is None:
            self.interactions = [i for i in self.default_interactions]
        elif isinstance(interactions, str):
            self.interactions = [interactions]
        elif isinstance(interactions, list):
            self.interactions = [i for i in interactions]
        else:
            raise ValueError(f"Invalid interactions: {interactions}")
        self.parameters = parameters

        # sanity check
        new_interactions = set(self.interactions)
        allowed_interactions = set(self.allowed_interactions)
        invalid_interactions = new_interactions - allowed_interactions
        assert new_interactions.issubset(allowed_interactions), f"Invalid interactions: {invalid_interactions}"

    @classmethod
    def _convert_dict_keys(cls, fp: Fingerprint) -> dict:
        ifp_tmp = fp.ifp
        data = {}
        for frame_id in ifp_tmp:
            data_frame = {}
            for residue_pair in ifp_tmp[frame_id]:
                residue_pair_key = str(residue_pair[0]) + "-" + str(residue_pair[1])
                data_frame[residue_pair_key] = ifp_tmp[frame_id][residue_pair]
            data[frame_id] = data_frame
        return data

    @classmethod
    def to_json(cls, fp: Fingerprint, out_path: Optional[str] = None) -> dict:
        data = cls._convert_dict_keys(fp)
        if out_path is not None and out_path.endswith(".json"):
            with open(out_path, "w") as f:
                dump_config(data, f, indent=2)
        return data

    def analyze_gmx(self,
                    gmxtop: str,
                    trj: str,
                    lig_sdf: str,
                    pro_select: str,
                    lig_select: str,
                    n_workers: Optional[int] = None,
                    frame_ids: Optional[Union[list[int], slice]] = None,
                    freq_threshold: Optional[float] = None) -> tuple[LigNetwork, Fingerprint]:
        gk = GlobalKeys()
        gk.filename.sys = gmxtop
        u: MDAnalysis.Universe = get_mda_universe(gk, trj)
        mda_protein = u.select_atoms(pro_select)
        mda_ligand = u.select_atoms(lig_select)

        n_jobs = n_workers
        n_traj = len(u.trajectory)
        s = None
        if frame_ids is None:
            s = None
        elif isinstance(frame_ids, slice):
            s = frame_ids
        elif isinstance(frame_ids, list):
            n_jobs = 1
            assert n_traj > max(frame_ids)
            s = frame_ids
        if n_jobs is None:
            n_jobs = min(self.MAX_N_JOBS, os.cpu_count() or 1)

        fp = Fingerprint(interactions=self.interactions,
                         parameters=self.parameters,
                         count=True,
                         vicinity_cutoff=self.VICINITY_CUTOFF)
        logger.info(f"Running ProLIF on {n_traj} frames with {n_jobs} workers.")
        if s is None:
            fp.run(traj=u.trajectory, lig=(lig_sdf, mda_ligand), prot=mda_protein, n_jobs=n_jobs)
        else:
            fp.run(traj=u.trajectory[s], lig=(lig_sdf, mda_ligand), prot=mda_protein, n_jobs=n_jobs)

        if freq_threshold is None:
            freq_threshold = self.FREQ_THRESHOLD
        try:
            # ProLIF may not find any interaction so the fingerprint may be empty
            lignet = LigNetwork.from_fingerprint(fp,
                                                 ligand_mol=Molecule.from_lig_sdf_mda(lig_sdf, mda_ligand),
                                                 threshold=freq_threshold)
            return lignet, fp
        except Exception as e:
            logger.exception("ProLIF failed to build LigNetwork: %s", e)
            return None, None


def analyze_frames_by_prolif(out_dir: str, json_file: str, gmxtop: str, trj: str, lig_sdf: str, pro_select: str,
                             lig_select: str) -> Optional[dict[int, dict[str, dict[str, tuple[dict]]]]]:
    """
    If a non-empty JSON file is found, return the frame-to-interactions dictionary stored in this file.
    If an empty JSON file is found, return None.

    Otherwise, run ProLIF on the trajectory, save the results to the JSON file,
    and return the frame-to-interactions dictionary.
    Depending on what ProLIF finds, the JSON file may be empty.
    """

    json_str = f"{out_dir}/{json_file}"
    if Path(json_str).is_file():
        # Preserve historical behavior: if the file exists and is empty, return None
        # and do NOT try to recompute.
        return _read_existing_prolif_results(json_str)

    frame_to_interactions = _find_non_vdw_interactions(out_dir, json_str, gmxtop, trj, lig_sdf, pro_select, lig_select)
    if frame_to_interactions is not None:
        return frame_to_interactions

    return _find_vdw_interactions_with_tols(json_str, gmxtop, trj, lig_sdf, pro_select, lig_select, (0.5, 1.0, 1.5))


def _read_existing_prolif_results(json_str: str) -> Optional[dict]:
    """Read existing ProLIF results.

    - If the file exists and is non-empty (as dict), return it.
    - If the file exists but contains an empty dict, return None.
    - If the file does not exist, return None.
    """
    logger.info(f"Found prolif results in ``{json_str}''.")
    with open(json_str) as f_json_str:
        frame_to_interactions = load_config(f_json_str)
        if not frame_to_interactions or len(frame_to_interactions.keys()) == 0:
            logger.info(f"Found no interactions in ``{json_str}''.")
            return None
        return frame_to_interactions


def _find_non_vdw_interactions(out_dir: str, json_str: str, gmxtop: str, trj: str, lig_sdf: str, pro_select: str,
                               lig_select: str) -> Optional[dict]:
    """Run ProLIF with all interactions except VdWContact."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    interaction_list = list(set(ProteinLigandInteraction.allowed_interactions) - set(["VdWContact"]))
    pli = ProteinLigandInteraction(interaction_list)
    lignet, fp = pli.analyze_gmx(gmxtop, trj, lig_sdf, pro_select, lig_select)
    if not fp:
        return None

    plt.clf()
    fp.plot_barcode(dpi=300)
    plt.savefig(json_str.replace(".json", "_barcode.png"), dpi=300)
    lignet.save(json_str.replace(".json", "_lignet.html"))
    frame_to_interactions = pli.to_json(fp, json_str)
    return frame_to_interactions


def _find_vdw_interactions_with_tols(json_str: str, gmxtop: str, trj: str, lig_sdf: str, pro_select: str,
                                     lig_select: str, tols: tuple[float, ...]) -> Optional[dict]:
    logger.warning("Trying to find VdWContact interactions")
    for tol in tols:
        logger.info(f"VdWContact tolerance set to {tol} angstrom")
        pli = ProteinLigandInteraction(interactions=["VdWContact"], parameters={"VdWContact": {"tolerance": tol}})
        _, fp = pli.analyze_gmx(gmxtop, trj, lig_sdf, pro_select, lig_select)
        if fp:
            frame_to_interactions = pli.to_json(fp, json_str)
            return frame_to_interactions
        logger.warning(f"VdWContact tolerance set to {tol} angstrom did not find any interactions.")

    logger.warning("No restraint anchors was found due to empty ProLIF fingerprint.")
    with open(json_str, "w") as fw:
        dump_config(dict(), fw)
    return None
