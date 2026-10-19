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

/**
 * Compute the alchem-nonbonded parameters for particles and exceptions.
 */
 extern "C" __global__ void updatePerturbParameters(
    GLOBAL mixed* RESTRICT energyBuffer, 
    int includeSelfEnergy,
    int includePerturbDispersionCorrection,
    GLOBAL real* RESTRICT globalParams, int numAtoms, 
    GLOBAL const float4* RESTRICT alchemParticleParams,
    GLOBAL const float4* RESTRICT alchemSigmaEpsilonParams,
    GLOBAL float2* RESTRICT ASigmaEpsilonParams,
    GLOBAL float2* RESTRICT BSigmaEpsilonParams,
    GLOBAL real4* RESTRICT posq, 
    GLOBAL real* RESTRICT charge,
    GLOBAL float2* RESTRICT ljLambdas, 
    GLOBAL float4* RESTRICT particleParamOffsets, 
    GLOBAL int* RESTRICT particleOffsetIndices,
    double qA, double qB, double qC, double sqrt_rest2, double vdwA, double vdwB, double Ecorr
#ifdef HAS_EXCEPTIONS
    , int numExceptions, 
    GLOBAL const float4* RESTRICT baseExceptionParams, 
    GLOBAL float4* RESTRICT exceptionParams,
    GLOBAL float4* RESTRICT exceptionParamOffsets, 
    GLOBAL int* RESTRICT exceptionOffsetIndices
#endif
    ) {
mixed energy = 0;

// Compute particle parameters.

for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE) {
    //float4 params = baseParticleParams[i];
    float4 alchemParams = alchemParticleParams[i];
    float4 alchemLJParams = alchemSigmaEpsilonParams[i];
    //float4 alchemLJParams = alchemSigmaEpsilonParams[i];

// Interpolate the charge and scale it if the particle is being rest2 scaled.
#ifdef USE_POSQ_CHARGES
    posq[i].w = qA*alchemParams.x + qB*alchemParams.y + qC*alchemParams.z;
    if (alchemParams.w != 0.0)
        posq[i].w *= sqrt_rest2;
#else
    charge[i] = qA*alchemParams.x + qB*alchemParams.y + qC*alchemParams.z;
    if (alchemParams.w != 0.0)
        charge[i] *= sqrt_rest2;
#endif
    ljLambdas[i] = make_float2(vdwA, vdwB);
    ASigmaEpsilonParams[i].x = alchemLJParams.x;
    ASigmaEpsilonParams[i].y = alchemLJParams.y;
    BSigmaEpsilonParams[i].x = alchemLJParams.z;
    BSigmaEpsilonParams[i].y = alchemLJParams.w;

    // Scale the epsilon parameters if the particle is being rest2 scaled.
    if (alchemParams.w != 0.0){
        ASigmaEpsilonParams[i].y *= sqrt_rest2;
        BSigmaEpsilonParams[i].y *= sqrt_rest2;
    }

#ifdef INCLUDE_EWALD
    #ifdef USE_POSQ_CHARGES
        energy -= EWALD_SELF_ENERGY_SCALE*posq[i].w*posq[i].w;
    #else
        energy -= EWALD_SELF_ENERGY_SCALE*charge[i]*charge[i];
    #endif
#endif

}

// Compute exception parameters.

#ifdef HAS_EXCEPTIONS
for (int i = GLOBAL_ID; i < numExceptions; i += GLOBAL_SIZE) {
    float4 params = baseExceptionParams[i];
    exceptionParams[i] = make_float4((float) (ONE_4PI_EPS0*params.x), (float) params.y, (float) (4*params.z), 0);
}
#endif

if (includeSelfEnergy)
    energyBuffer[GLOBAL_ID] += energy;
if (includePerturbDispersionCorrection)
    energyBuffer[GLOBAL_ID] += Ecorr;
}

extern "C" __global__ void reOrderPositions(
    GLOBAL const real4* RESTRICT posq,
    GLOBAL const real* charges,
    GLOBAL const int* RESTRICT atomOrder,
    GLOBAL const int numAtoms,
    GLOBAL real3* RESTRICT orderedPositions,
    GLOBAL real* RESTRICT orderedCharges
) { 
    // Reorder the positions in posq according to the atomOrder.
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= numAtoms) return;

    // Use atomOrder to reorder the positions.
    int index = atomOrder[i]; // index in natural orders.
    orderedPositions[index].x = posq[i].x;
    orderedPositions[index].y = posq[i].y;
    orderedPositions[index].z = posq[i].z;
    //orderedPositions[index].w = posq[i].w;

#if USE_POSQ_CHARGES
    orderedCharges[index] = posq[i].w;
#else
    orderedCharges[index] = charges[i];
#endif
}