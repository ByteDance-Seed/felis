# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pandas import DataFrame

from felis.configs import GlobalKeys
from felis.protocols.boresch.dg_correction import BR_correction_2023


def calc_restraints(
    boresch_cfg_json: str,
    outdir: str,
):
    gk = GlobalKeys()
    gk.update_by_cfg(boresch_cfg_json)
    gk.check()

    harmonic = True
    T = gk.integrator.targetT_K
    r0, theta0, phi0 = gk.boresch.r_theta_phi
    alpha0, beta0, gamma0 = gk.boresch.alpha_beta_gamma
    kr, kang, kdih = gk.boresch.k_r_a_dih_kcal
    correction_kcal = BR_correction_2023(harmonic, T, r0, theta0, phi0, alpha0, beta0, gamma0, kr, kang, kdih)

    df = DataFrame()
    df["StateA"] = ["R00"]
    df["StateB"] = ["R01"]
    df["dG(F)"] = [f"{correction_kcal:.4f}"]
    df["dG(B)"] = [f"{-correction_kcal:.4f}"]
    df["dG(F+B)"] = [0.0]
    df["MBAR"] = [f"{correction_kcal:.4f}"]

    csv_out = f"{outdir}/R_fe_table.tsv"
    df.to_csv(csv_out, sep="\t", index=False)
