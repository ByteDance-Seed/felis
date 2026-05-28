# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import math


def BR_correction_2006(harmonic: bool, T_k: float, r0_nm: float, theta0_deg: float, _phi0_deg: float,
                       _alpha0_deg: float, _beta0_deg: float, _gamma0_deg: float, Kr_kcal_ang2, Kang_kcal_rad2,
                       Kdihed_kcal_rad2):
    """
    Returns the analytical correction of Boresch restraints in kcal/mol: Free System -> Restrained System.

    V=0.5 * k * x**2 if harmonic otherwise k * x**2.
    k=k_harmonic or k=2*k_anharmonic.

    Calculation of Standard Binding Free Energies: Aromatic Molecules in the T4 Lysozyme L99A Mutant

    Yuqing Deng & Benoit Roux
    J. Chem. Theory Comput. 2006, 2, 1255-1273

    https://pubs.acs.org/doi/epdf/10.1021/ct060037v
    """

    coeff = 1.0 if harmonic else 2.0
    T, r0, theta0 = T_k, r0_nm * 10, theta0_deg
    kr = coeff * Kr_kcal_ang2
    ka = coeff * Kang_kcal_rad2
    kd = coeff * Kdihed_kcal_rad2

    R = 8.314 / 1000 / 4.184  # kcal/mol/K
    C0 = 1. / 1660  # angstrom^-3
    pi, radian = math.pi, 180. / math.pi
    log, sin, sqrt = math.log, math.sin, math.sqrt

    RT = R * T
    Ft = r0**2 * sin(theta0 / radian) * sqrt((2 * pi * RT)**3 / (kr * ka * kd))
    Fr = 1 / (8 * pi**2) * sqrt((2 * pi * RT)**3 / (ka * kd * kd))
    return -RT * log(Fr * Ft * C0)


def BR_correction_2023(harmonic: bool, T_k: float, r0_nm: float, theta0_deg: float, _phi0_deg: float, alpha0_deg: float,
                       _beta0_deg: float, _gamma0_deg: float, Kr_kcal_ang2, Kang_kcal_rad2, Kdihed_kcal_rad2):
    """
    Returns the analytical correction of Boresch restraints in kcal/mol: Free System -> Restrained System.

    V=0.5 * k * x**2 if harmonic otherwise k * x**2.
    k=k_harmonic or k=2*k_anharmonic.

    Enhancing Hit Discovery in Virtual Screening through Absolute Protein-Ligand Binding Free-Energy Calculations

    Wei Chen & Lingle Wang et al.
    J Chem. Inf. Model. 2023, 63, 3171-3185

    https://pubs.acs.org/doi/10.1021/acs.jcim.3c00013
    https://chemrxiv.org/engage/chemrxiv/article-details/63a23f6116e9a872d32f81ef


    Rewritten from https://github.com/freeenergylab/FEP-SPell-ABFE

    MIT License

    Copyright (c) 2024 freeenergylab

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    coeff = 1.0 if harmonic else 2.0
    T = T_k
    r0 = r0_nm * 10
    theta0 = theta0_deg
    alpha0 = alpha0_deg
    Kr = coeff * Kr_kcal_ang2
    Kang = coeff * Kang_kcal_rad2
    Kdihed = coeff * Kdihed_kcal_rad2

    R = 8.314 / 1000 / 4.184  # kcal/mol/K
    V0 = 1660.  # angstrom^3
    pi, radian = math.pi, 180. / math.pi
    erf, exp, log, sin, sqrt = math.erf, math.exp, math.log, math.sin, math.sqrt

    RT = R * T
    Z_dist_r = RT * r0 / (2 * Kr) * exp(-Kr / RT * r0**2) + sqrt(pi) / (4 *
                                                                        (Kr / RT)**1.5) * (1 + 2 * Kr / RT * r0**2) * (
                                                                            1 + erf(sqrt(Kr / RT) * r0))  # Eq. 4
    Z_ang_alpha_r = sqrt(pi / (Kang / RT)) * exp(-RT / (4. * Kang)) * sin(alpha0 / radian)  # Eq. 5
    Z_ang_theta_r = sqrt(pi / (Kang / RT)) * exp(-RT / (4. * Kang)) * sin(theta0 / radian)  # Eq. 5
    Z_dihed_r = sqrt(pi / (Kdihed / RT)) * erf(pi * sqrt(Kdihed / RT))  # Eq. 6
    result = -RT * log(Z_dist_r * Z_ang_theta_r * Z_ang_alpha_r * Z_dihed_r**3 / (8 * pi**2 * V0))  # ChemRxiv Eq. 3
    return result
