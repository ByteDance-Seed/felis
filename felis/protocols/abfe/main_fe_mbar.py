# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from felis.configs import GlobalKeys
from felis.protocols.abfe.free_energy_tools import calc_bar
from felis.protocols.abfe.free_energy_tools import calc_raw_dG
from felis.utils.omm.omm_tools import get_mbar_results


def plot_free_energy(xas0: NDArray, xas1: NDArray, xbs0: NDArray, xbs1: NDArray, outdir: str, results: dict, stem0: str,
                     stem1: str):
    unit = results.get("unit", "kJ/mol")
    if "dG+" in results and "dG-" in results:
        nbins = 25
        alpha = 0.2

        png = f"{outdir}/_{stem0}_{stem1}.png"
        plt.clf()

        dG_u = xas1 - xas0
        dG_v = xbs0 - xbs1
        raw_u = results["dG+"]
        raw_v = results["dG-"]
        bar = results.get("bar", None)

        pdf1, edges1 = np.histogram(dG_u, bins=nbins, density=True)
        center1 = 0.5 * (edges1[:-1] + edges1[1:])
        line, = plt.plot(center1, pdf1, linewidth=8, label=f"Fwd: +$\\langle${stem1}-{stem0}$\\rangle_a$")
        plt.axvline(x=raw_u, color=line.get_color(), linestyle="dashed")
        plt.fill_between(center1, pdf1, alpha=alpha)

        pdf2, edges2 = np.histogram(-dG_v, bins=nbins, density=True)
        center2 = 0.5 * (edges2[:-1] + edges2[1:])
        line, = plt.plot(center2, pdf2, linewidth=8, label=f"Bwd: -$\\langle${stem0}-{stem1}$\\rangle_b$")
        plt.axvline(x=-raw_v, color=line.get_color(), linestyle="dashed")
        plt.fill_between(center2, pdf2, alpha=alpha)

        label = f"Fwd: {raw_u:.2f} {unit}\nBwd: {raw_v:.2f} {unit}"
        if bar:
            label = label + f"\nBar: {bar:.2f} {unit}"
        plt.plot(center2, 0 * center2, color="white", label=label)

        plt.xlabel(r"$\Delta$U" + f" ({unit})")
        plt.ylabel("Probability Density")
        plt.ylim(0, None)
        plt.legend()

        plt.tight_layout()
        plt.savefig(png)

    if "bar_convergence" in results and results["bar_convergence"]:
        y_range_png_alpha = 0.1
        y_range_percent = 0.01

        png = f"{outdir}/_converge_{stem0}_{stem1}.png"
        bar_convergence_results = np.array(results["bar_convergence"])
        f_bar, b_bar = bar_convergence_results[:, 0], bar_convergence_results[:, 1]
        n_data = len(f_bar)
        x_data = np.arange(n_data) + 1

        y_mid = f_bar[-1]
        y_range = abs(y_mid * y_range_percent)
        y_low, y_high = y_mid - y_range, y_mid + y_range

        plt.clf()

        plt.plot(x_data, f_bar, marker="o", linewidth=8, markersize=14, label=f"{stem1}-{stem0} Time: +")
        plt.plot(x_data, b_bar, marker="o", linewidth=8, markersize=14, label=f"{stem1}-{stem0} Time: -")
        plt.axhline(y=y_mid, xmin=0., xmax=1., color="k", linestyle="dotted")
        plt.fill_between(x_data, y_low, y_high, color="k", alpha=y_range_png_alpha)

        plt.xlabel("Number of Sample Blocks")
        plt.ylabel(f"BAR dG ({unit})")

        plt.xticks(x_data)
        plt.xlim(x_data[0], x_data[-1])
        plt.legend()

        plt.tight_layout()
        plt.savefig(png)


def calc_mbar(
    stem: str,
    nc_list: list[str],
    checkpoint_interval: int,
    outdir: str,
) -> DataFrame:
    for nc_file in nc_list:
        assert Path(nc_file).is_file(), nc_file

    cnt_i, cnt_j = 0, 0
    state_A, state_B = [], []
    dG_F, dG_B, dG_FB, mbar = [], [], [], []
    n_uv_split = 5
    overall_convergence = []
    for nc_file in nc_list:
        gk = GlobalKeys()
        gk.openmm.checkpoint_interval = checkpoint_interval
        mbar_res = get_mbar_results(nc_file, gk)
        kT_kJ_mol = mbar_res["kT_kJ_mol"]
        kT_kcal_mol = kT_kJ_mol / 4.184
        ndiscard = mbar_res["n_discard"]
        ridx = mbar_res["replica_index"]
        ematx = mbar_res["energy_matrix"]
        n_states = mbar_res["n_states"]
        _nrep, niter = ridx.shape
        cnt_j = cnt_j + n_states - 1

        val_fe, _std_fe = mbar_res["fe"], mbar_res["std_fe"]
        for state_i in range(cnt_i, cnt_j):
            state_j = state_i + 1
            state0, state1 = state_i - cnt_i, state_j - cnt_i

            maska, maskb = (ridx == state0), (ridx == state1)
            ra, rb = np.argmax(maska, axis=0), np.argmax(maskb, axis=0)
            xas0 = ematx[ra, state0, np.arange(niter)][ndiscard:]
            xas1 = ematx[ra, state1, np.arange(niter)][ndiscard:]
            xbs0 = ematx[rb, state0, np.arange(niter)][ndiscard:]
            xbs1 = ematx[rb, state1, np.arange(niter)][ndiscard:]
            ui = xas1 - xas0
            vj = xbs0 - xbs1
            g1, _s1 = calc_raw_dG(ui)
            g2, _s2 = calc_raw_dG(vj)

            g1 *= kT_kcal_mol
            g2 *= kT_kcal_mol
            g_mbar = val_fe[state0, state1] * kT_kcal_mol

            stem0 = f"{stem}{state_i:02d}"
            stem1 = f"{stem}{state_j:02d}"
            state_A.append(stem0)
            state_B.append(stem1)
            dG_F.append(f"{g1:.4f}")
            dG_B.append(f"{g2:.4f}")
            dG_FB.append(f"{(g1+g2):.4f}")
            mbar.append(f"{g_mbar:.4f}")

            len_uv = len(ui)
            assert len_uv == len(vj)
            uv_idx_array = np.array([_ + 1 for _ in range(len_uv)])
            uv_split = np.array_split(uv_idx_array, n_uv_split)
            bar_convergence_results = []
            for i_split in uv_split:
                idx = i_split[-1]
                f_ui, f_vj = ui[:idx], vj[:idx]
                b_ui, b_vj = ui[::-1][:idx], vj[::-1][:idx]
                f_bar, _s_f_bar = calc_bar(f_ui, f_vj)
                b_bar, _s_b_bar = calc_bar(b_ui, b_vj)
                f_bar *= kT_kcal_mol
                b_bar *= kT_kcal_mol
                bar_convergence_results.append([f_bar, b_bar])
            overall_convergence.append(bar_convergence_results)

            plot_free_energy(xas0 * kT_kcal_mol, xas1 * kT_kcal_mol, xbs0 * kT_kcal_mol, xbs1 * kT_kcal_mol, outdir, {
                "dG+": g1,
                "dG-": g2,
                "unit": "kcal/mol",
                "bar_convergence": bar_convergence_results
            }, stem0, stem1)

        cnt_i = cnt_j

    overall_convergence = np.array(overall_convergence)
    n_dg, _, _ = overall_convergence.shape
    fb_total = []
    df_all_conv = DataFrame()
    df_all_conv["StateA"] = [f"{stem}{i:02d}" for i in range(n_dg)] + [f"{stem}bar"]
    df_all_conv["StateB"] = [f"{stem}{i+1:02d}" for i in range(n_dg)] + [f"{stem}bar"]
    for i in range(n_uv_split):
        f_total = np.sum(overall_convergence[:, i, 0])
        b_total = np.sum(overall_convergence[:, i, 1])
        fb_total.append([f_total, b_total])
        df_all_conv[f"Block-{i+1}(+)"] = [f"{x:.4f}" for x in overall_convergence[:, i, 0]] + [f"{f_total:.4f}"]
        df_all_conv[f"Block-{i+1}(-)"] = [f"{x:.4f}" for x in overall_convergence[:, i, 1]] + [f"{b_total:.4f}"]
    csv_out = f"{outdir}/{stem}_converge_table.tsv"
    df_all_conv.to_csv(csv_out, sep="\t", index=False)
    plot_free_energy(None, None, None, None, outdir, {
        "unit": "kcal/mol",
        "bar_convergence": fb_total,
    }, f"{stem}bar", f"{stem}bar")

    df = DataFrame()
    df["StateA"] = state_A
    df["StateB"] = state_B
    df["dG(F)"] = dG_F
    df["dG(B)"] = dG_B
    df["dG(F+B)"] = dG_FB
    df["MBAR"] = mbar

    csv_out = f"{outdir}/{stem}_fe_table.tsv"
    df.to_csv(csv_out, sep="\t", index=False)
    return df
