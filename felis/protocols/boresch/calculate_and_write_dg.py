# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from felis.protocols.boresch.dg_correction import BR_correction_2023


def calculate_and_write_dg_to_file(outdir: str,
                                   T: float,
                                   ideal: dict[str, float],
                                   kr: float,
                                   ka: float,
                                   kdih: float,
                                   filename: str = "sysR_fe_table.txt",
                                   harmonic: bool = True) -> float:
    """Calculate analytical Boresch correction and write the dG table."""
    correction_kcal = BR_correction_2023(harmonic, T, ideal["r0"], ideal["theta0"], ideal["phi0"], ideal["alpha0"],
                                         ideal["beta0"], ideal["gamma0"], kr, ka, kdih)

    cal_to_J = 4.184
    delim = "\t"
    outpath = Path(outdir) / filename
    with outpath.open("w") as fw:
        fw.write(
            f"StateA{delim}StateB{delim}dG+{delim}std/dG+{delim}dG-{delim}std/dG-{delim}BAR{delim}std/BAR{delim}dG(F+B)\n"
        )
        stem0, stem1 = "Sum", "kJ/mol"
        g1 = correction_kcal * cal_to_J
        g2 = -g1
        ba = g1
        g3 = 0
        s1, s2, sb = "", "", ""
        fw.write(
            f"{stem0}{delim}{stem1}{delim}{g1:.4f}{delim}{s1}{delim}{g2:.4f}{delim}{s2}{delim}{ba:.4f}{delim}{sb}{delim}{g3:.4f}\n"
        )
        stem1 = "kcal/mol"
        g1 = correction_kcal
        g2 = -g1
        ba = g1
        fw.write(
            f"{stem0}{delim}{stem1}{delim}{g1:.4f}{delim}{s1}{delim}{g2:.4f}{delim}{s2}{delim}{ba:.4f}{delim}{sb}{delim}{g3:.4f}\n"
        )

    return float(correction_kcal)
