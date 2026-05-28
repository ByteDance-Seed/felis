# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

import MDAnalysis
import numpy as np
from MDAnalysis.analysis.dssp import DSSP

logger = logging.getLogger(__name__)


def compute_residue_to_dssp(u: MDAnalysis.Universe) -> dict[str, tuple[str, int]]:
    ag_protein = u.select_atoms("protein and not resname ACE NME and (name N or name CA or name C or name O)")
    u_resname_set: set[str] = set()
    u_resname_list: list[str] = []
    for u_pa in ag_protein.atoms:
        u_pa_resname = f"{u_pa.resname}{u_pa.resid}"
        if u_pa_resname not in u_resname_set:
            u_resname_set.add(u_pa_resname)
            u_resname_list.append(u_pa_resname)
    dssp = DSSP(ag_protein).run()

    def get_dssp_score(idssp):
        one_score = "HGIEB"
        return np.isin(idssp, list(one_score)).astype(int)

    if dssp.results.dssp.ndim == 2:
        dssp_results0 = dssp.results.dssp[0]
    else:
        dssp_results0 = dssp.results.dssp
    dssp_scores_raw = get_dssp_score(dssp.results.dssp)
    # MDAnalysis may return a 1D array for a single frame; keep the API robust.
    if getattr(dssp_scores_raw, "ndim", 1) == 1:
        dssp_scores_mean = dssp_scores_raw
    else:
        dssp_scores_mean = np.mean(dssp_scores_raw, axis=0)
    dssp_scores = np.where(dssp_scores_mean > 0.5, 1, 0)
    assert len(u_resname_list) == len(dssp_results0), f"{len(u_resname_list)} {len(dssp_results0)}"
    residue_to_dssp = dict(zip(u_resname_list, zip(dssp_results0, dssp_scores)))
    logger.info(f"DSSP Frame 0: {residue_to_dssp}")
    return residue_to_dssp
