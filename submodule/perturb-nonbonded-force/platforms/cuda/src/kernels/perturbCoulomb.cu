/*
 * This file is part of PerturbNonbondedPlugin.
 * Portions are adapted from the OpenMM 8.5.0 NonbondedForce CUDA/common
 * platform kernels. See https://openmm.org/development.
 *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.
 * Authors: Peter Eastman
 * Modifications copyright (c) 2026 Bytedance Ltd. and/or its affiliates
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

{
#if USE_EWALD
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
    const real alphaR = EWALD_ALPHA * r;
    const real expAlphaRSqr = EXP(-alphaR * alphaR);
#if HAS_COULOMB
    const real prefactor = ONE_4PI_EPS0 * CHARGE1 * CHARGE2 * invR;
#else
    const real prefactor = 0.0f;
#endif

#ifdef USE_DOUBLE_PRECISION
    const real erfcAlphaR = erfc(alphaR);
#else
    // This approximation for erfc is from Abramowitz and Stegun (1964) p. 299.  They cite the following as
    // the original source: C. Hastings, Jr., Approximations for Digital Computers (1955).  It has a maximum
    // error of 1.5e-7.

    const real t = RECIP(1.0f + 0.3275911f * alphaR);
    const real erfcAlphaR = (0.254829592f + (-0.284496736f + (1.421413741f + (-1.453152027f + 1.061405429f * t) * t) * t) * t) * t * expAlphaRSqr;
#endif
    real tempForce = 0.0f;
    
    tempForce = prefactor * (erfcAlphaR + alphaR * expAlphaRSqr * TWO_OVER_SQRT_PI);
    tempEnergy += includeInteraction ? prefactor * erfcAlphaR : 0;

    dEdR += includeInteraction ? tempForce * invR * invR : 0;
#else
#ifdef USE_CUTOFF
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
#else
    unsigned int includeInteraction = (!isExcluded);
#endif
    real tempForce = 0.0f;

#if HAS_COULOMB
#ifdef USE_CUTOFF
    const real prefactor = ONE_4PI_EPS0 * CHARGE1 * CHARGE2;
    tempForce += prefactor * (invR - 2.0f * REACTION_FIELD_K * r2);
    //real coulombEnergy = prefactor * (invR + REACTION_FIELD_K * r2 - REACTION_FIELD_C);
    tempEnergy += includeInteraction ? prefactor * (invR + REACTION_FIELD_K * r2 - REACTION_FIELD_C) : 0;
#else
    const real prefactor = ONE_4PI_EPS0 * CHARGE1 * CHARGE2 * invR;
    tempForce += prefactor;
    tempEnergy += includeInteraction ? prefactor : 0;
#endif
#endif
    dEdR += includeInteraction ? tempForce * invR * invR : 0;
#endif
}
