# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Compute free-energy estimates from reduced-potential (kT) samples.

This module provides small, NumPy-based utilities used for
computing free-energy differences and uncertainty estimates from 1D arrays of
reduced potentials (energies scaled by ``RT``/``kT``).
"""

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def calc_raw_dG(x: NDArray) -> NDArray:
    """Estimate a free-energy difference from reduced potentials via FEP.

    Computes ``dG = -log(mean(exp(-x)))`` using a numerically-stabilized form
    (subtracting ``mean(x)`` before exponentiation). Here ``x`` is the reduced
    potential (a.k.a. RT-scaled energy): ``x = U / RT``.

    The returned standard deviation is an asymptotic uncertainty estimate based
    on the variance of the exponential estimator.

    Args:
        x (NDArray): Reduced potentials ``U/RT``. Shape ``(n,)``.

    Returns:
        NDArray: ``[dG, stdev]`` in reduced free-energy units (kT). Shape
        ``(2,)``.
    """

    x_mean = np.mean(x)
    x1 = x - x_mean  # m=E[x]; x1=x-m
    exp_nx1 = np.exp(-x1)  # exp(-x+m)
    exp_nx1_mean = np.mean(exp_nx1)  # E[exp(-x+m)]
    dG = x_mean - np.log(exp_nx1_mean)  # m - log E[exp(-x+m)]

    exp_n2x1 = exp_nx1 * exp_nx1  # exp(-2x+2m)
    exp_n2x1_mean = np.mean(exp_n2x1)  # E[exp(-2x+2m)]
    variance = (exp_n2x1_mean / exp_nx1_mean**2 - 1.) / len(x)
    stdev = variance**0.5
    return np.array([dG, stdev])


def fermi_dirac(x: NDArray) -> NDArray:
    """Evaluate a Fermi-Dirac/logistic function used by BAR.

    This implementation returns ``1 / (1 + exp(x))`` and is vectorized over the
    input array.

    Args:
        x (NDArray): Input values. Any shape.

    Returns:
        NDArray: Values in ``[0, 1]`` with the same shape as ``x``.
    """
    return 1.0 / (1.0 + np.exp(x))


def calc_bar(ui: NDArray, vj: NDArray) -> NDArray:
    """Estimate a free-energy difference using Bennett acceptance ratio (BAR).

    Uses Newton iteration to solve the BAR self-consistency equation for the
    reduced free-energy difference ``c0`` between two states, given samples
    ``ui`` and ``vj`` (both in reduced potential units). The function returns an
    asymptotic uncertainty estimate based on sample variances.

    If the iteration does not converge within a fixed number of iterations, a
    warning is logged and the last iterate is returned.

    Args:
        ui (NDArray): Reduced-potential differences from state ``u``. Shape
            ``(nu,)``.
        vj (NDArray): Reduced-potential differences from state ``v``. Shape
            ``(nv,)``.

    Returns:
        NDArray: ``[dG, stdev]`` in reduced free-energy units (kT). Shape
        ``(2,)``.
    """

    eps = 1e-6
    maxiter = 100

    nu, nv = len(ui), len(vj)
    ui_mean, vj_mean = np.mean(ui), np.mean(vj)
    uv_mean = (ui_mean * nu - vj_mean * nv) / (nu + nv)

    diverge = True
    c0, niter = uv_mean, 0
    while diverge and niter <= maxiter:
        cv = fermi_dirac(c0 + vj)
        uc = fermi_dirac(ui - c0)
        cv_mean = np.mean(cv)
        uc_mean = np.mean(uc)
        cv2_mean = np.mean(cv * cv)
        uc2_mean = np.mean(uc * uc)
        # target function F(c), and let G(c) = d/dc F(c)
        Fc = cv_mean - uc_mean
        Gc = cv2_mean - cv_mean + uc2_mean - uc_mean
        dc = Fc / Gc

        if abs(dc) < eps:
            diverge = False
        else:
            c0, niter = c0 - dc, niter + 1
    if diverge:
        logger.warning(f"bar did not converge after {maxiter} iterations.")

    variance = (cv2_mean / cv_mean**2 - 1.) / nv + (uc2_mean / uc_mean**2 - 1.) / nu
    stdev = variance**0.5
    return np.array([c0, stdev])
