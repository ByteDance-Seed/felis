# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
import logging
from pathlib import Path

from felis.configs import dump_config
from felis.configs import GKBoresch
from felis.configs import GKIntegrator
from felis.protocols.boresch.anchors import find_boresch_anchors
from felis.protocols.boresch.calculate_and_write_dg import calculate_and_write_dg_to_file
from felis.protocols.boresch.filter_mask_histogram_plot import filter_and_mask_and_get_and_plot_histogram
from felis.protocols.boresch.find_ideal_geometry import find_ideal_geometry
from felis.protocols.boresch.process_trajectory_get_raw_counts import process_trajectory_and_get_raw_counts

logger = logging.getLogger(__name__)

BORESCH_CFG_JSON = "sys_boresch_cfg.json"


def generate_boresch_restraints(kr: float, ka: float, kdih: float, lig_sdf: str, sys_top: str, trj_dcd: str,
                                atom_ids_json: str, outdir: str) -> None:
    Path(outdir).mkdir(parents=True, exist_ok=True)

    ############################################
    # find anchor atoms and write them to file #
    ############################################

    p123, l123 = find_boresch_anchors(sys_top,
                                      trj_dcd,
                                      out_dir=outdir,
                                      pro_select="protein",
                                      lig_select="resname M00",
                                      lig_sdf=lig_sdf,
                                      atom_ids_file=atom_ids_json)

    gk_boresch = GKBoresch()
    gk_boresch.k_r_a_dih_kcal = [kr, ka, kdih]
    gk_boresch.ligatoms = l123
    gk_boresch.proatoms = p123

    #############################
    # find the ideal geometries #
    #############################

    T = GKIntegrator.targetT_K
    raw = process_trajectory_and_get_raw_counts(sys_top, trj_dcd, p123, l123)
    histogram_result = filter_and_mask_and_get_and_plot_histogram(outdir, raw, nbins=20)
    ideal = find_ideal_geometry(histogram_result)
    r0, theta0, phi0 = ideal["r0"], ideal["theta0"], ideal["phi0"]
    alpha0, beta0, gamma0 = ideal["alpha0"], ideal["beta0"], ideal["gamma0"]
    gk_boresch.r_theta_phi = [r0, theta0, phi0]
    gk_boresch.alpha_beta_gamma = [alpha0, beta0, gamma0]

    boresch_cfg_dict = {"boresch": asdict(gk_boresch)}
    with open(f"{outdir}/{BORESCH_CFG_JSON}", "w") as f:
        dump_config(boresch_cfg_dict, f, indent=2)

    calculate_and_write_dg_to_file(outdir, T, ideal, kr, ka, kdih)
