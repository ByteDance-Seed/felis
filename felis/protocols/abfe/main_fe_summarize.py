# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import glob
import logging

import numpy as np
from pandas import DataFrame
from pandas import read_csv

from felis.configs import GlobalKeys
from felis.utils.image_tools import merge_images

logger = logging.getLogger(__name__)


def summarize_fe(workdir: str, ligand: str, lam_sol_cfg: str, lam_pro_cfg: str):
    # solv
    df_sol = read_csv(f"{workdir}/A_fe_table.tsv", sep="\t")
    gk_sol = GlobalKeys()
    gk_sol.update_by_cfg(lam_sol_cfg)
    gk_sol.check()
    range_elec, range_vdw = [], []
    for i, lams in enumerate(gk_sol.ab.lam_list):
        vl, el, _rl = lams
        if vl == 1.0:
            range_elec.append(i)
        if el == 0.0:
            range_vdw.append(i)
    sum_sol_e = np.sum(df_sol["MBAR"][range_elec[0]:range_elec[-1]])
    sum_sol_v = np.sum(df_sol["MBAR"][range_vdw[0]:range_vdw[-1]])

    # prot
    df_pro = read_csv(f"{workdir}/B_fe_table.tsv", sep="\t")
    gk_pro = GlobalKeys()
    gk_pro.update_by_cfg(lam_pro_cfg)
    gk_pro.check()
    range_vdw, range_elec, range_res = [], [], []
    for i, lams in enumerate(gk_pro.ab.lam_list):
        vl, el, rl = lams
        if vl == 1.0:
            if el == 1.0:
                range_res.append(i)
            if rl == 1.0:
                range_elec.append(i)
        if el == 0.0:
            range_vdw.append(i)
    sum_pro_v = np.sum(df_pro["MBAR"][range_vdw[0]:range_vdw[-1]])
    sum_pro_e = np.sum(df_pro["MBAR"][range_elec[0]:range_elec[-1]])
    sum_pro_res = np.sum(df_pro["MBAR"][range_res[0]:range_res[-1]])

    # res
    df_r = read_csv(f"{workdir}/R_fe_table.tsv", sep="\t")
    sum_r = np.sum(df_r["MBAR"])

    df = DataFrame()
    df["ligand"] = [ligand]
    df["sol.e.1.0"] = [f"{sum_sol_e:.4f}"]
    df["sol.v.1.0"] = [f"{sum_sol_v:.4f}"]
    df["sol"] = [f"{sum_sol_e+sum_sol_v:.4f}"]
    df["pro.e.0.1"] = [f"{sum_pro_e:.4f}"]
    df["pro.v.0.1"] = [f"{sum_pro_v:.4f}"]
    df["pro"] = [f"{sum_pro_e + sum_pro_v:.4f}"]
    df["res.r.0.1"] = [f"{sum_r:.4f}"]
    df["res.r.1.0"] = [f"{sum_pro_res:.4f}"]
    df["res"] = [f"{sum_r + sum_pro_res:.4f}"]
    df["dG(kcal/mol)"] = [f"{sum_sol_e+sum_sol_v+sum_pro_e+sum_pro_v+sum_pro_res+sum_r:.4f}"]

    csv_out = f"{workdir}/sys_abfe.tsv"
    df.to_csv(csv_out, sep="\t", index=False)

    try:
        png_list = glob.glob(f"{workdir}/_A*_A*.png")
        if png_list:
            png_list.sort()
            merge_images(png_list, 8, f"{workdir}/sys_A_fe_merge.png", clean=True)

        png_list = glob.glob(f"{workdir}/_B*_B*.png")
        if png_list:
            png_list.sort()
            merge_images(png_list, 8, f"{workdir}/sys_B_fe_merge.png", clean=True)

        png_list = glob.glob(f"{workdir}/_converge_A*_A*.png")
        if png_list:
            png_list.sort()
            merge_images(png_list, 8, f"{workdir}/sys_A_converge_merge.png", clean=True)

        png_list = glob.glob(f"{workdir}/_converge_B*_B*.png")
        if png_list:
            png_list.sort()
            merge_images(png_list, 8, f"{workdir}/sys_B_converge_merge.png", clean=True)
    except Exception as e:
        logger.error(f"{e}")
