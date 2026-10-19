// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifdef CONTEXT_PRECISION_POSITIONS
#define QUERY_POSITION_BASE_TYPE real4
#define QUERY_REAL mixed
#define QUERY_REAL3 mixed3
#define MAKE_QUERY_REAL3 make_mixed3
#if defined(USE_MIXED_PRECISION)
#define QUERY_RSQRT rsqrt
#define QUERY_EXP exp
#else
#define QUERY_RSQRT RSQRT
#define QUERY_EXP EXP
#endif
#else
#define QUERY_POSITION_BASE_TYPE float4
#define QUERY_REAL real
#define QUERY_REAL3 real3
#define MAKE_QUERY_REAL3 make_real3
#define QUERY_RSQRT RSQRT
#define QUERY_EXP EXP
#endif

inline __device__ QUERY_REAL3 loadQueryPosition(
        GLOBAL const QUERY_POSITION_BASE_TYPE* RESTRICT positionBase,
#ifdef CONTEXT_PRECISION_POSITIONS
        GLOBAL const float4* RESTRICT positionCorrection,
#endif
        int index) {
    const QUERY_POSITION_BASE_TYPE base = positionBase[index];
#ifdef USE_MIXED_PRECISION
#ifdef CONTEXT_PRECISION_POSITIONS
    const float4 correction = positionCorrection[index];
    return MAKE_QUERY_REAL3((QUERY_REAL) base.x+(QUERY_REAL) correction.x,
            (QUERY_REAL) base.y+(QUERY_REAL) correction.y,
            (QUERY_REAL) base.z+(QUERY_REAL) correction.z);
#endif
#endif
    return MAKE_QUERY_REAL3((QUERY_REAL) base.x, (QUERY_REAL) base.y, (QUERY_REAL) base.z);
}

inline __device__ QUERY_REAL3 loadSystemPosition(
        GLOBAL const real4* RESTRICT positionBase,
#ifdef CONTEXT_PRECISION_POSITIONS
        GLOBAL const float4* RESTRICT positionCorrection,
#endif
        int index) {
    const real4 base = positionBase[index];
#ifdef USE_MIXED_PRECISION
#ifdef CONTEXT_PRECISION_POSITIONS
    const float4 correction = positionCorrection[index];
    return MAKE_QUERY_REAL3((QUERY_REAL) base.x+(QUERY_REAL) correction.x,
            (QUERY_REAL) base.y+(QUERY_REAL) correction.y,
            (QUERY_REAL) base.z+(QUERY_REAL) correction.z);
#endif
#endif
    return MAKE_QUERY_REAL3((QUERY_REAL) base.x, (QUERY_REAL) base.y, (QUERY_REAL) base.z);
}

#ifndef GRID_SIZE_X
KERNEL void computeParticleInteractionEnergies(
        int numQueries,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const real4* RESTRICT posq,
#ifdef CONTEXT_PRECISION_POSITIONS
        GLOBAL const float4* RESTRICT posqCorrection,
#endif
        GLOBAL const real* RESTRICT charges,
        GLOBAL const float2* RESTRICT sigmaEpsilonA,
        GLOBAL const float2* RESTRICT sigmaEpsilonB,
        GLOBAL const int* RESTRICT ljAtomTypes,
        GLOBAL const QUERY_POSITION_BASE_TYPE* RESTRICT queryPositions,
#ifdef CONTEXT_PRECISION_POSITIONS
        GLOBAL const float4* RESTRICT queryPositionCorrection,
#endif
        GLOBAL const float4* RESTRICT queryParameters,
        GLOBAL const int* RESTRICT excludedAtomStarts,
        GLOBAL const int* RESTRICT excludedAtomCounts,
        GLOBAL const int* RESTRICT excludedAtoms,
        GLOBAL const int* RESTRICT atomOrder,
        int hasExclusions,
        double lambdaA_vdw, double lambdaB_vdw,
        int startBlockIndex, int numBlocks, int useBlockPruning,
        GLOBAL const real4* RESTRICT blockCenter,
        GLOBAL const real4* RESTRICT blockBoundingBox,
        GLOBAL float4* RESTRICT queryForces,
        GLOBAL float* RESTRICT queryEnergies) {
    for (int query = GLOBAL_ID; query < numQueries; query += GLOBAL_SIZE) {
        const QUERY_REAL3 position = loadQueryPosition(queryPositions,
#ifdef CONTEXT_PRECISION_POSITIONS
                queryPositionCorrection,
#endif
                query);
        const float4 queryParameter = queryParameters[query];
        const QUERY_REAL charge1 = (QUERY_REAL) queryParameter.x;
        const QUERY_REAL sigma1 = (QUERY_REAL) queryParameter.y;
        const QUERY_REAL epsilon1 = (QUERY_REAL) queryParameter.z;
        QUERY_REAL energy = 0;
        QUERY_REAL3 force = MAKE_QUERY_REAL3(0);

        for (int block = startBlockIndex; block < startBlockIndex+numBlocks; block++) {
            const real4 center = blockCenter[block];
            const real4 blockSize = blockBoundingBox[block];
            const QUERY_REAL3 size = MAKE_QUERY_REAL3((QUERY_REAL) blockSize.x,
                    (QUERY_REAL) blockSize.y, (QUERY_REAL) blockSize.z);
            QUERY_REAL3 blockDelta = MAKE_QUERY_REAL3((QUERY_REAL) center.x-position.x,
                    (QUERY_REAL) center.y-position.y, (QUERY_REAL) center.z-position.z);
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
            // The first test uses the enclosing sphere stored in center.w.  The
            // second is a tighter query-to-AABB test.  The padded cutoff makes
            // this conservative with respect to OpenMM's block coordinates.
            if (useBlockPruning) {
                if (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z >=
                        (PADDED_CUTOFF+center.w)*(PADDED_CUTOFF+center.w))
                    continue;
                blockDelta.x = max((QUERY_REAL) 0, fabs(blockDelta.x)-size.x);
                blockDelta.y = max((QUERY_REAL) 0, fabs(blockDelta.y)-size.y);
                blockDelta.z = max((QUERY_REAL) 0, fabs(blockDelta.z)-size.z);
                if (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z >= PADDED_CUTOFF_SQUARED)
                    continue;
            }

            for (int atomInBlock = 0; atomInBlock < TILE_SIZE; atomInBlock++) {
                const int atom = block*TILE_SIZE+atomInBlock;
                if (atom >= NUM_ATOMS)
                    break;
                bool excluded = false;
                if (hasExclusions) {
                    const int hostAtom = atomOrder[atom];
                    const int first = excludedAtomStarts[query];
                    const int last = first+excludedAtomCounts[query];
                    for (int index = first; index < last; index++)
                        excluded |= (excludedAtoms[index] == hostAtom);
                }
                if (excluded)
                    continue;

                const QUERY_REAL3 atomPosition = loadSystemPosition(posq,
#ifdef CONTEXT_PRECISION_POSITIONS
                        posqCorrection,
#endif
                        atom);
                QUERY_REAL3 delta = atomPosition-position;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                QUERY_REAL r2 = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
                if (r2 <= 1e-12f || r2 >= CUTOFF_SQUARED)
                    continue;
                const QUERY_REAL invR = QUERY_RSQRT(r2);
                const QUERY_REAL r = r2*invR;
#ifdef USE_POSQ_CHARGES
                const QUERY_REAL charge2 = (QUERY_REAL) posq[atom].w;
#else
                const QUERY_REAL charge2 = (QUERY_REAL) charges[atom];
#endif
                QUERY_REAL pairEnergy = 0;
                QUERY_REAL radialForce = 0;

#if HAS_COULOMB
                const QUERY_REAL chargeProduct = ONE_4PI_EPS0*charge1*charge2;
#ifdef USE_EWALD
                const QUERY_REAL alphaR = EWALD_ALPHA*r;
                const QUERY_REAL expAlphaRSqr = QUERY_EXP(-alphaR*alphaR);
#if defined(USE_DOUBLE_PRECISION) || (defined(CONTEXT_PRECISION_POSITIONS) && defined(USE_MIXED_PRECISION))
                const QUERY_REAL erfcAlphaR = erfc(alphaR);
#else
                const QUERY_REAL t = RECIP(1.0f+0.3275911f*alphaR);
                const QUERY_REAL erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
                pairEnergy += chargeProduct*invR*erfcAlphaR;
                radialForce += chargeProduct*invR*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
#else
                pairEnergy += chargeProduct*(invR+REACTION_FIELD_K*r2-REACTION_FIELD_C);
                radialForce += chargeProduct*(invR-2*REACTION_FIELD_K*r2);
#endif
#endif

#if HAS_LENNARD_JONES
                const QUERY_REAL sigA = sigma1+sigmaEpsilonA[atom].x;
                const QUERY_REAL sigB = sigma1+sigmaEpsilonB[atom].x;
                const QUERY_REAL epsA = epsilon1*sigmaEpsilonA[atom].y;
                const QUERY_REAL epsB = epsilon1*sigmaEpsilonB[atom].y;
                const QUERY_REAL r6 = r2*r2*r2;
                const QUERY_REAL sigA2 = sigA*sigA;
                const QUERY_REAL sigB2 = sigB*sigB;
                const QUERY_REAL sigA6 = sigA2*sigA2*sigA2;
                const QUERY_REAL sigB6 = sigB2*sigB2*sigB2;
                QUERY_REAL ljEnergy = 0;
                QUERY_REAL ljForce = 0;
                if (ljAtomTypes[atom] == 1) {
                    const QUERY_REAL r6scA = SOFTCORE_ALPHA*(1-lambdaA_vdw)*(1-lambdaA_vdw)*sigA6+r6;
                    const QUERY_REAL r6scB = SOFTCORE_ALPHA*(1-lambdaB_vdw)*(1-lambdaB_vdw)*sigB6+r6;
                    const QUERY_REAL v6A = sigA6/r6scA;
                    const QUERY_REAL v6B = sigB6/r6scB;
                    ljEnergy = lambdaA_vdw*epsA*(v6A*v6A-v6A)+lambdaB_vdw*epsB*(v6B*v6B-v6B);
                    ljForce = lambdaA_vdw*epsA*(2*v6A-1)*6*v6A*r6/r6scA+
                              lambdaB_vdw*epsB*(2*v6B-1)*6*v6B*r6/r6scB;
                }
                else {
                    const QUERY_REAL v6A = sigA6/r6;
                    const QUERY_REAL v6B = sigB6/r6;
                    if (ljAtomTypes[atom] == 2) {
                        ljEnergy = lambdaA_vdw*epsA*(v6A*v6A-v6A)+lambdaB_vdw*epsB*(v6B*v6B-v6B);
                        ljForce = lambdaA_vdw*epsA*(2*v6A-1)*6*v6A+
                                  lambdaB_vdw*epsB*(2*v6B-1)*6*v6B;
                    }
                    else {
                        ljEnergy = epsA*(v6A*v6A-v6A);
                        ljForce = epsA*(2*v6A-1)*6*v6A;
                    }
                }
                pairEnergy += ljEnergy;
                radialForce += ljForce;
#endif
                const QUERY_REAL scale = radialForce/r2;
                force += delta*scale;
                energy += pairEnergy;
            }
        }
        queryForces[query] = make_float4((float) force.x, (float) force.y, (float) force.z, 0.0f);
        queryEnergies[query] = (float) energy;
    }
}
#ifdef USE_EWALD
KERNEL void computeRealPmeDirectEnergies(
        int numQueries, int numAtomsPerMolecule,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const real4* RESTRICT posq,
#ifdef CONTEXT_PRECISION_POSITIONS
        GLOBAL const float4* RESTRICT posqCorrection,
#endif
        GLOBAL const real* RESTRICT charges,
        GLOBAL const float2* RESTRICT sigmaEpsilonA,
        GLOBAL const float2* RESTRICT sigmaEpsilonB,
        GLOBAL const int* RESTRICT ljAtomTypes,
        GLOBAL const int* RESTRICT atomOrder,
        GLOBAL const int* RESTRICT realAtoms,
        double lambdaA_vdw, double lambdaB_vdw,
        GLOBAL float* RESTRICT queryEnergies) {
    for (int query = GLOBAL_ID; query < numQueries; query += GLOBAL_SIZE) {
        const int hostQuery = realAtoms[query];
        int orderedQuery = -1;
        for (int ordered = 0; ordered < NUM_ATOMS; ordered++)
            if (atomOrder[ordered] == hostQuery) {
                orderedQuery = ordered;
                break;
            }
        if (orderedQuery < 0) {
            queryEnergies[query] = 0.0f;
            continue;
        }
        const int queryMolecule = query/numAtomsPerMolecule;
        const QUERY_REAL3 queryPosition = loadSystemPosition(posq,
#ifdef CONTEXT_PRECISION_POSITIONS
                posqCorrection,
#endif
                orderedQuery);
#ifdef USE_POSQ_CHARGES
        const QUERY_REAL charge1 = (QUERY_REAL) posq[orderedQuery].w;
#else
        const QUERY_REAL charge1 = (QUERY_REAL) charges[orderedQuery];
#endif
        const QUERY_REAL sigma1A = sigmaEpsilonA[orderedQuery].x;
        const QUERY_REAL sigma1B = sigmaEpsilonB[orderedQuery].x;
        const QUERY_REAL epsilon1A = sigmaEpsilonA[orderedQuery].y;
        const QUERY_REAL epsilon1B = sigmaEpsilonB[orderedQuery].y;
        const int queryType = ljAtomTypes[orderedQuery];
        QUERY_REAL energy = 0;
        for (int atom = 0; atom < NUM_ATOMS; atom++) {
            const int hostAtom = atomOrder[atom];
            bool sameMolecule = false;
            for (int site = 0; site < numAtomsPerMolecule; site++)
                sameMolecule |= (realAtoms[queryMolecule*numAtomsPerMolecule+site] == hostAtom);
            if (sameMolecule)
                continue;
            const QUERY_REAL3 atomPosition = loadSystemPosition(posq,
#ifdef CONTEXT_PRECISION_POSITIONS
                    posqCorrection,
#endif
                    atom);
            QUERY_REAL3 delta = atomPosition-queryPosition;
            APPLY_PERIODIC_TO_DELTA(delta)
            const QUERY_REAL r2 = dot(delta, delta);
            if (r2 <= 1e-12f || r2 >= CUTOFF_SQUARED)
                continue;
            const QUERY_REAL invR = QUERY_RSQRT(r2);
            const QUERY_REAL r = r2*invR;
#ifdef USE_POSQ_CHARGES
            const QUERY_REAL charge2 = (QUERY_REAL) posq[atom].w;
#else
            const QUERY_REAL charge2 = (QUERY_REAL) charges[atom];
#endif
#if HAS_COULOMB
            const QUERY_REAL alphaR = EWALD_ALPHA*r;
            const QUERY_REAL expAlphaRSqr = QUERY_EXP(-alphaR*alphaR);
#if defined(USE_DOUBLE_PRECISION) || (defined(CONTEXT_PRECISION_POSITIONS) && defined(USE_MIXED_PRECISION))
            const QUERY_REAL erfcAlphaR = erfc(alphaR);
#else
            const QUERY_REAL t = RECIP(1.0f+0.3275911f*alphaR);
            const QUERY_REAL erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
            energy += ONE_4PI_EPS0*charge1*charge2*invR*erfcAlphaR;
#endif
#if HAS_LENNARD_JONES
            const QUERY_REAL sigA = sigma1A+sigmaEpsilonA[atom].x;
            const QUERY_REAL sigB = sigma1B+sigmaEpsilonB[atom].x;
            const QUERY_REAL epsA = epsilon1A*sigmaEpsilonA[atom].y;
            const QUERY_REAL epsB = epsilon1B*sigmaEpsilonB[atom].y;
            const QUERY_REAL r6 = r2*r2*r2;
            const QUERY_REAL sigA2 = sigA*sigA;
            const QUERY_REAL sigB2 = sigB*sigB;
            const QUERY_REAL sigA6 = sigA2*sigA2*sigA2;
            const QUERY_REAL sigB6 = sigB2*sigB2*sigB2;
            const int atomType = ljAtomTypes[atom];
            if (queryType == 1 || atomType == 1) {
                const QUERY_REAL r6scA = SOFTCORE_ALPHA*(1-lambdaA_vdw)*(1-lambdaA_vdw)*sigA6+r6;
                const QUERY_REAL r6scB = SOFTCORE_ALPHA*(1-lambdaB_vdw)*(1-lambdaB_vdw)*sigB6+r6;
                const QUERY_REAL v6A = sigA6/r6scA;
                const QUERY_REAL v6B = sigB6/r6scB;
                energy += lambdaA_vdw*epsA*(v6A*v6A-v6A)+lambdaB_vdw*epsB*(v6B*v6B-v6B);
            }
            else {
                const QUERY_REAL v6A = sigA6/r6;
                const QUERY_REAL v6B = sigB6/r6;
                energy += queryType == 2 || atomType == 2 ?
                        lambdaA_vdw*epsA*(v6A*v6A-v6A)+lambdaB_vdw*epsB*(v6B*v6B-v6B) :
                        epsA*(v6A*v6A-v6A);
            }
#endif
        }
        queryEnergies[query] = (float) energy;
    }
}

KERNEL void computeCandidateSubsetEnergies(
        int numQueries, int numSourceAtoms,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const real4* RESTRICT posq,
#ifdef CONTEXT_PRECISION_POSITIONS
        GLOBAL const float4* RESTRICT posqCorrection,
#endif
        GLOBAL const real* RESTRICT charges,
        GLOBAL const float2* RESTRICT sigmaEpsilonA,
        GLOBAL const float2* RESTRICT sigmaEpsilonB,
        GLOBAL const int* RESTRICT ljAtomTypes,
        GLOBAL const int* RESTRICT atomOrder,
        GLOBAL const QUERY_POSITION_BASE_TYPE* RESTRICT queryPositions,
#ifdef CONTEXT_PRECISION_POSITIONS
        GLOBAL const float4* RESTRICT queryPositionCorrection,
#endif
        GLOBAL const float4* RESTRICT queryParameters,
        GLOBAL const int* RESTRICT sourceAtoms,
        double lambdaA_vdw, double lambdaB_vdw,
        GLOBAL float* RESTRICT queryEnergies) {
    for (int query = GLOBAL_ID; query < numQueries; query += GLOBAL_SIZE) {
        const QUERY_REAL3 position = loadQueryPosition(queryPositions,
#ifdef CONTEXT_PRECISION_POSITIONS
                queryPositionCorrection,
#endif
                query);
        const float4 queryParameter = queryParameters[query];
        const QUERY_REAL charge1 = (QUERY_REAL) queryParameter.x;
        const QUERY_REAL sigma1 = (QUERY_REAL) queryParameter.y;
        const QUERY_REAL epsilon1 = (QUERY_REAL) queryParameter.z;
        QUERY_REAL energy = 0;
        for (int source = 0; source < numSourceAtoms; source++) {
            const int hostAtom = sourceAtoms[source];
            int atom = -1;
            for (int ordered = 0; ordered < NUM_ATOMS; ordered++)
                if (atomOrder[ordered] == hostAtom) {
                    atom = ordered;
                    break;
                }
            if (atom < 0)
                continue;
            const QUERY_REAL3 atomPosition = loadSystemPosition(posq,
#ifdef CONTEXT_PRECISION_POSITIONS
                    posqCorrection,
#endif
                    atom);
            QUERY_REAL3 delta = atomPosition-position;
            APPLY_PERIODIC_TO_DELTA(delta)
            const QUERY_REAL r2 = dot(delta, delta);
            if (r2 <= 1e-12f || r2 >= CUTOFF_SQUARED)
                continue;
            const QUERY_REAL invR = QUERY_RSQRT(r2);
            const QUERY_REAL r = r2*invR;
#ifdef USE_POSQ_CHARGES
            const QUERY_REAL charge2 = (QUERY_REAL) posq[atom].w;
#else
            const QUERY_REAL charge2 = (QUERY_REAL) charges[atom];
#endif
#if HAS_COULOMB
            const QUERY_REAL alphaR = EWALD_ALPHA*r;
            const QUERY_REAL expAlphaRSqr = QUERY_EXP(-alphaR*alphaR);
#if defined(USE_DOUBLE_PRECISION) || (defined(CONTEXT_PRECISION_POSITIONS) && defined(USE_MIXED_PRECISION))
            const QUERY_REAL erfcAlphaR = erfc(alphaR);
#else
            const QUERY_REAL t = RECIP(1.0f+0.3275911f*alphaR);
            const QUERY_REAL erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
            energy += ONE_4PI_EPS0*charge1*charge2*invR*erfcAlphaR;
#endif
#if HAS_LENNARD_JONES
            const QUERY_REAL sigA = sigma1+sigmaEpsilonA[atom].x;
            const QUERY_REAL sigB = sigma1+sigmaEpsilonB[atom].x;
            const QUERY_REAL epsA = epsilon1*sigmaEpsilonA[atom].y;
            const QUERY_REAL epsB = epsilon1*sigmaEpsilonB[atom].y;
            const QUERY_REAL r6 = r2*r2*r2;
            const QUERY_REAL sigA2 = sigA*sigA;
            const QUERY_REAL sigB2 = sigB*sigB;
            const QUERY_REAL sigA6 = sigA2*sigA2*sigA2;
            const QUERY_REAL sigB6 = sigB2*sigB2*sigB2;
            if (ljAtomTypes[atom] == 1) {
                const QUERY_REAL r6scA = SOFTCORE_ALPHA*(1-lambdaA_vdw)*(1-lambdaA_vdw)*sigA6+r6;
                const QUERY_REAL r6scB = SOFTCORE_ALPHA*(1-lambdaB_vdw)*(1-lambdaB_vdw)*sigB6+r6;
                const QUERY_REAL v6A = sigA6/r6scA;
                const QUERY_REAL v6B = sigB6/r6scB;
                energy += lambdaA_vdw*epsA*(v6A*v6A-v6A)+lambdaB_vdw*epsB*(v6B*v6B-v6B);
            }
            else {
                const QUERY_REAL v6A = sigA6/r6;
                const QUERY_REAL v6B = sigB6/r6;
                energy += ljAtomTypes[atom] == 2 ?
                        lambdaA_vdw*epsA*(v6A*v6A-v6A)+lambdaB_vdw*epsB*(v6B*v6B-v6B) :
                        epsA*(v6A*v6A-v6A);
            }
#endif
        }
        queryEnergies[query] = (float) energy;
    }
}
#endif
#endif

#ifdef GRID_SIZE_X
KERNEL void sumMixedValues(
        int numValues,
        GLOBAL const mixed* RESTRICT values,
        GLOBAL mixed* RESTRICT sum) {
    if (GLOBAL_ID != 0)
        return;
    mixed result = 0;
    for (int index = 0; index < numValues; index++)
        result += values[index];
    sum[0] = result;
}

KERNEL void interpolatePmePotentialAtPositions(
        int numPositions,
        GLOBAL const real* RESTRICT pmeGrid,
        GLOBAL const QUERY_POSITION_BASE_TYPE* RESTRICT positions,
#ifdef CONTEXT_PRECISION_POSITIONS
        GLOBAL const float4* RESTRICT positionCorrection,
#endif
        GLOBAL float* RESTRICT potentials,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    for (int atom = GLOBAL_ID; atom < numPositions; atom += GLOBAL_SIZE) {
        QUERY_REAL3 data[PME_ORDER];
        const QUERY_REAL scale = RECIP((QUERY_REAL) (PME_ORDER-1));
        QUERY_REAL potential = 0;
        QUERY_REAL3 position = loadQueryPosition(positions,
#ifdef CONTEXT_PRECISION_POSITIONS
                positionCorrection,
#endif
                atom);
        APPLY_PERIODIC_TO_POS(position)
        QUERY_REAL3 t = MAKE_QUERY_REAL3(position.x*recipBoxVecX.x+position.y*recipBoxVecY.x+position.z*recipBoxVecZ.x,
                                         position.y*recipBoxVecY.y+position.z*recipBoxVecZ.y,
                                         position.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x)%GRID_SIZE_X, ((int) t.y)%GRID_SIZE_Y, ((int) t.z)%GRID_SIZE_Z);
        QUERY_REAL3 dr = MAKE_QUERY_REAL3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
        data[PME_ORDER-1] = MAKE_QUERY_REAL3(0);
        data[1] = dr;
        data[0] = MAKE_QUERY_REAL3(1)-dr;
        for (int order = 3; order < PME_ORDER; order++) {
            const QUERY_REAL div = RECIP((QUERY_REAL) (order-1));
            data[order-1] = div*dr*data[order-2];
            for (int k = 1; k < order-1; k++)
                data[order-k-1] = div*((dr+MAKE_QUERY_REAL3(k))*data[order-k-2]+(MAKE_QUERY_REAL3(order-k)-dr)*data[order-k-1]);
            data[0] = div*(MAKE_QUERY_REAL3(1)-dr)*data[0];
        }
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < PME_ORDER-1; j++)
            data[PME_ORDER-j-1] = scale*((dr+MAKE_QUERY_REAL3(j))*data[PME_ORDER-j-2]+(MAKE_QUERY_REAL3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(MAKE_QUERY_REAL3(1)-dr)*data[0];
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase *= GRID_SIZE_Y*GRID_SIZE_Z;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase+ybase*GRID_SIZE_Z;
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = gridIndex.z+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    potential += data[ix].x*data[iy].y*data[iz].z*pmeGrid[ybase+zindex];
                }
            }
        }
        potentials[atom] = (float) potential;
    }
}

KERNEL void computePmeReciprocalAtomicEnergies(
        int numAtoms,
        GLOBAL const real* RESTRICT pmeGrid,
        GLOBAL const real4* RESTRICT posq,
        GLOBAL const real* RESTRICT charges,
        GLOBAL const int* RESTRICT atomOrder,
        GLOBAL mixed* RESTRICT energies,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    for (int atom = GLOBAL_ID; atom < numAtoms; atom += GLOBAL_SIZE) {
        real3 data[PME_ORDER];
        const real scale = RECIP((real) (PME_ORDER-1));
        real potential = 0;
        const real4 position4 = posq[atom];
        real3 position = trimTo3(position4);
        APPLY_PERIODIC_TO_POS(position)
        real3 t = make_real3(position.x*recipBoxVecX.x+position.y*recipBoxVecY.x+position.z*recipBoxVecZ.x,
                             position.y*recipBoxVecY.y+position.z*recipBoxVecZ.y,
                             position.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x)%GRID_SIZE_X, ((int) t.y)%GRID_SIZE_Y, ((int) t.z)%GRID_SIZE_Z);
        real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
        data[PME_ORDER-1] = make_real3(0);
        data[1] = dr;
        data[0] = make_real3(1)-dr;
        for (int order = 3; order < PME_ORDER; order++) {
            const real div = RECIP((real) (order-1));
            data[order-1] = div*dr*data[order-2];
            for (int k = 1; k < order-1; k++)
                data[order-k-1] = div*((dr+make_real3(k))*data[order-k-2]+(make_real3(order-k)-dr)*data[order-k-1]);
            data[0] = div*(make_real3(1)-dr)*data[0];
        }
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < PME_ORDER-1; j++)
            data[PME_ORDER-j-1] = scale*((dr+make_real3(j))*data[PME_ORDER-j-2]+(make_real3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(make_real3(1)-dr)*data[0];
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase *= GRID_SIZE_Y*GRID_SIZE_Z;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase+ybase*GRID_SIZE_Z;
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = gridIndex.z+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    potential += data[ix].x*data[iy].y*data[iz].z*pmeGrid[ybase+zindex];
                }
            }
        }
#ifdef USE_POSQ_CHARGES
        const real charge = position4.w;
#else
        const real charge = charges[atom];
#endif
        energies[atomOrder[atom]] = (mixed) (0.5*SQRT_ONE_4PI_EPS0*charge*potential);
    }
}
#endif
