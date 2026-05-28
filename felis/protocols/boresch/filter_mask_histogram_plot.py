# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from felis.protocols.boresch.anchor_constants import VALID_ANGLE_RANGE


def filter_and_mask_and_get_and_plot_histogram(outdir: str,
                                               raw: dict[str, np.ndarray],
                                               nbins: int = 20,
                                               plot_filename: str = "sys_boresch0.png") -> dict[str, Any]:
    """Filter geometry series, compute histograms, and write the diagnostic plot.

    Args:
        outdir: Output directory.
        raw: Output of `process_trajectory_and_get_raw_counts`.
        nbins: Histogram bin count.
        plot_filename: Plot filename under `outdir`.

    Returns:
        A dict containing:
          - geom_mask: bool mask used for filtering
          - filtered: dict of filtered arrays
          - normalized: dict containing phi360/beta360/gamma360 and filtered variants
          - binmax: list of peak locations (same order as `names`)
    """
    r = raw["r"]
    theta = raw["theta"]
    phi = raw["phi"]
    alpha = raw["alpha"]
    beta = raw["beta"]
    gamma = raw["gamma"]

    phi360 = np.where(phi >= 0., phi, phi + 360.)
    beta360 = np.where(beta >= 0., beta, beta + 360.)
    gamma360 = np.where(gamma >= 0., gamma, gamma + 360.)

    low, high = VALID_ANGLE_RANGE
    geom_mask = (low <= theta) & (theta <= high) & (low <= alpha) & (alpha <= high)
    if not np.any(geom_mask):
        raise ValueError("No frames satisfy the angular filter; cannot derive Boresch geometry from empty selection.")

    filtered_r = r[geom_mask]
    filtered_theta = theta[geom_mask]
    filtered_phi = phi[geom_mask]
    filtered_alpha = alpha[geom_mask]
    filtered_beta = beta[geom_mask]
    filtered_gamma = gamma[geom_mask]

    filtered_phi360 = np.where(filtered_phi >= 0., filtered_phi, filtered_phi + 360.)
    filtered_beta360 = np.where(filtered_beta >= 0., filtered_beta, filtered_beta + 360.)
    filtered_gamma360 = np.where(filtered_gamma >= 0., filtered_gamma, filtered_gamma + 360.)

    names = [
        "theta\nP2P1L1", "r\nP1L1", "alpha\nP1L1L2", "phi\nP3P2P1L1", "beta\nP2P1L1L2", "gamma\nP1L1L2L3",
        "phi'\nP3P2P1L1", "beta'\nP2P1L1L2", "gamma'\nP1L1L2L3"
    ]
    properties = [
        (theta, filtered_theta),
        (r, filtered_r),
        (alpha, filtered_alpha),
        (phi, filtered_phi),
        (beta, filtered_beta),
        (gamma, filtered_gamma),
        (phi360, filtered_phi360),
        (beta360, filtered_beta360),
        (gamma360, filtered_gamma360),
    ]
    ranges = [
        (0., 180.),
        None,
        (0., 180.),
        (-180., 180.),
        (-180., 180.),
        (-180., 180.),
        (0., 360.),
        (0., 360.),
        (0., 360.),
    ]

    pdfs: list[tuple[np.ndarray, np.ndarray]] = []
    bins: list[tuple[np.ndarray, np.ndarray]] = []
    binmax: list[float] = []

    for (x, filtered_x), rnge in zip(properties, ranges):
        pdf1, bins1 = np.histogram(x, bins=nbins, range=rnge, density=True)
        bin_centers = 0.5 * (bins1[:-1] + bins1[1:])
        pdf_f, bins_f = np.histogram(filtered_x, bins=nbins, range=rnge, density=True)
        bin_centers_f = 0.5 * (bins_f[:-1] + bins_f[1:])
        pdfs.append((pdf1, pdf_f))
        bins.append((bin_centers, bin_centers_f))
        binmax.append(float(bin_centers_f[np.argmax(pdf_f)]))

    fig, axes = plt.subplots(nrows=3, ncols=3)
    for i in range(3):
        for j in range(3):
            k = 3 * i + j
            n = names[k]
            pdf1, pdf_f = pdfs[k]
            bins1, bins_f = bins[k]
            axes[i, j].plot(bins1, pdf1)
            axes[i, j].plot(bins_f, pdf_f, "--")
            axes[i, j].set_title(n)
    fig.tight_layout()
    plot_path = Path(outdir) / plot_filename
    plt.savefig(str(plot_path))

    return {
        "geom_mask": geom_mask,
        "filtered": {
            "r": filtered_r,
            "theta": filtered_theta,
            "phi": filtered_phi,
            "alpha": filtered_alpha,
            "beta": filtered_beta,
            "gamma": filtered_gamma,
        },
        "normalized": {
            "phi360": phi360,
            "beta360": beta360,
            "gamma360": gamma360,
            "filtered_phi360": filtered_phi360,
            "filtered_beta360": filtered_beta360,
            "filtered_gamma360": filtered_gamma360,
        },
        "binmax": binmax,
    }
