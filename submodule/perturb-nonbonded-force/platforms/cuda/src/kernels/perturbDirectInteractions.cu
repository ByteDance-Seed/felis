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

#if HAS_LENNARD_JONES
    /*
    real sig = A_SIGMA_EPSILON1.x + A_SIGMA_EPSILON2.x;
    real sig2 = invR * sig;
    sig2 *= sig2;
    real sig6 = sig2 * sig2 * sig2;
    real eps = A_SIGMA_EPSILON1.y * A_SIGMA_EPSILON2.y;
    real epssig6 = sig6 * eps;
    tempForce = epssig6 * (12.0f * sig6 - 6.0f);
    real ljEnergy = epssig6 * (sig6 - 1.0f);
    */
    
    
    const unsigned int NORMAL_ATOM_TYPE = 0;
    const unsigned int DUMMY_ATOM_TYPE = 1;
    const unsigned int ALCHEM_ATOM_TYPE = 2;

    real sigA = A_SIGMA_EPSILON1.x + A_SIGMA_EPSILON2.x;
    real sigB = B_SIGMA_EPSILON1.x + B_SIGMA_EPSILON2.x;
    real epsA = A_SIGMA_EPSILON1.y * A_SIGMA_EPSILON2.y;
    real epsB = B_SIGMA_EPSILON1.y * B_SIGMA_EPSILON2.y;
    real sigA2 = sigA * sigA;
    real sigB2 = sigB * sigB;
    real sigA6 = sigA2 * sigA2 * sigA2;
    real sigB6 = sigB2 * sigB2 * sigB2;
    real vdwLambdaA = LJ_LAMBDA1.x;
    real vdwLambdaB = LJ_LAMBDA1.y;

    if (r < 1e-5){
        r = 1e-5;
    }
    real r2 = r * r;
    real r6 = r2 * r2 * r2;
    real invR2 = invR * invR;
    real invR6 = invR2 * invR2 * invR2;
    real ljEnergy = 0.0f;
    
    if (LJ_ATOM_TYPE1 == DUMMY_ATOM_TYPE || LJ_ATOM_TYPE2 == DUMMY_ATOM_TYPE){
        real r6scA = SOFTCORE_ALPHA * (1 - vdwLambdaA) * (1 - vdwLambdaA) * sigA6+ r6;
        real r6scB = SOFTCORE_ALPHA * (1 - vdwLambdaB) * (1 - vdwLambdaB) * sigB6+ r6;
        real v6Asc = sigA6 / r6scA;
        real v6Bsc = sigB6 / r6scB;
        real v12Asc = v6Asc * v6Asc;
        real v12Bsc = v6Bsc * v6Bsc;
        real eAsc = vdwLambdaA * epsA * (v12Asc - v6Asc);
        real eBsc = vdwLambdaB * epsB * (v12Bsc - v6Bsc);

        tempForce = vdwLambdaA * epsA * (2 * v6Asc - 1) * 6 * v6Asc * r6 / r6scA + vdwLambdaB * epsB * (2 * v6Bsc - 1) * 6 * v6Bsc * r6 / r6scB;
        ljEnergy = eAsc + eBsc;
    } 
    else if (LJ_ATOM_TYPE1 == ALCHEM_ATOM_TYPE || LJ_ATOM_TYPE2 == ALCHEM_ATOM_TYPE){
        real v6A = sigA6 * invR6;
        real v6B = sigB6 * invR6;
        real v12A = v6A * v6A;
        real v12B = v6B * v6B;
        real eA = epsA * (v12A - v6A);
        real eB = epsB * (v12B - v6B);
        tempForce = vdwLambdaA * epsA * (2 * v6A - 1) * 6 * v6A + vdwLambdaB * epsB * (2 * v6B - 1) * 6 * v6B;
        ljEnergy = vdwLambdaA * eA + vdwLambdaB * eB;
    } 
    else {
        real v6A = sigA6  * invR6;
        real v12A = v6A * v6A;
        real eA = epsA * (v12A - v6A);
        ljEnergy = eA;
        tempForce = epsA * (2 * v6A - 1) * 6 * v6A;
    }
    

    tempForce += prefactor * (erfcAlphaR + alphaR * expAlphaRSqr * TWO_OVER_SQRT_PI);
    tempEnergy += includeInteraction ? ljEnergy + prefactor * erfcAlphaR : 0;
#else

    tempForce = prefactor * (erfcAlphaR + alphaR * expAlphaRSqr * TWO_OVER_SQRT_PI);
    tempEnergy += includeInteraction ? prefactor * erfcAlphaR : 0;
#endif
    dEdR += includeInteraction ? tempForce * invR * invR : 0;
#else
#ifdef USE_CUTOFF
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
#else
    unsigned int includeInteraction = (!isExcluded);
#endif
    real tempForce = 0.0f;

#if HAS_LENNARD_JONES
    /*
    real sig = A_SIGMA_EPSILON1.x + A_SIGMA_EPSILON2.x;
    real sig2 = invR * sig;
    sig2 *= sig2;
    real sig6 = sig2 * sig2 * sig2;
    real epssig6 = sig6 * (A_SIGMA_EPSILON1.y * A_SIGMA_EPSILON2.y);
    tempForce = epssig6 * (12.0f * sig6 - 6.0f);
    real ljEnergy = includeInteraction ? epssig6 * (sig6 - 1.0f);
    */
    
    const unsigned int NORMAL_ATOM_TYPE = 0;
    const unsigned int DUMMY_ATOM_TYPE = 1;
    const unsigned int ALCHEM_ATOM_TYPE = 2;

    real sigA = A_SIGMA_EPSILON1.x + A_SIGMA_EPSILON2.x;
    real sigB = B_SIGMA_EPSILON1.x + B_SIGMA_EPSILON2.x;
    real epsA = A_SIGMA_EPSILON1.y * A_SIGMA_EPSILON2.y;
    real epsB = B_SIGMA_EPSILON1.y * B_SIGMA_EPSILON2.y;
    real sigA2 = sigA * sigA;
    real sigB2 = sigB * sigB;
    real sigA6 = sigA2 * sigA2 * sigA2;
    real sigB6 = sigB2 * sigB2 * sigB2;
    real vdwLambdaA = LJ_LAMBDA1.x;
    real vdwLambdaB = LJ_LAMBDA1.y;

    if (r < 1e-5){
        r = 1e-5;
    }
    real r2 = r * r;
    real r6 = r2 * r2 * r2;
    real invR2 = invR * invR;
    real invR6 = invR2 * invR2 * invR2;
    real ljEnergy = 0.0f;
    
    if (LJ_ATOM_TYPE1 == DUMMY_ATOM_TYPE || LJ_ATOM_TYPE2 == DUMMY_ATOM_TYPE){
        real r6scA = SOFTCORE_ALPHA * (1 - vdwLambdaA) * (1 - vdwLambdaA) * sigA6+ r6;
        real r6scB = SOFTCORE_ALPHA * (1 - vdwLambdaB) * (1 - vdwLambdaB) * sigB6+ r6;
        real v6Asc = sigA6 / r6scA;
        real v6Bsc = sigB6 / r6scB;
        real v12Asc = v6Asc * v6Asc;
        real v12Bsc = v6Bsc * v6Bsc;
        real eAsc = vdwLambdaA * epsA * (v12Asc - v6Asc);
        real eBsc = vdwLambdaB * epsB * (v12Bsc - v6Bsc);

        tempForce = vdwLambdaA * epsA * (2 * v6Asc - 1) * 6 * v6Asc * r6 / r6scA + vdwLambdaB * epsB * (2 * v6Bsc - 1) * 6 * v6Bsc * r6 / r6scB;
        ljEnergy = eAsc + eBsc;
    } 
    else if (LJ_ATOM_TYPE1 == ALCHEM_ATOM_TYPE || LJ_ATOM_TYPE2 == ALCHEM_ATOM_TYPE){
        real v6A = sigA6 * invR6;
        real v6B = sigB6 * invR6;
        real v12A = v6A * v6A;
        real v12B = v6B * v6B;
        real eA = epsA * (v12A - v6A);
        real eB = epsB * (v12B - v6B);
        tempForce = vdwLambdaA * epsA * (2 * v6A - 1) * 6 * v6A + vdwLambdaB * epsB * (2 * v6B - 1) * 6 * v6B;
        ljEnergy = vdwLambdaA * eA + vdwLambdaB * eB;
    } 
    else {
        real v6A = sigA6  * invR6;
        real v12A = v6A * v6A;
        real eA = epsA * (v12A - v6A);
        ljEnergy = eA;
        tempForce = epsA * (2 * v6A - 1) * 6 * v6A;
    }
    
    tempEnergy += includeInteraction ? ljEnergy : 0;
#endif

#if HAS_COULOMB
#ifdef USE_CUTOFF
    const real prefactor = ONE_4PI_EPS0 * CHARGE1 * CHARGE2;
    tempForce += prefactor * (invR - 2.0f * REACTION_FIELD_K * r2);
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
