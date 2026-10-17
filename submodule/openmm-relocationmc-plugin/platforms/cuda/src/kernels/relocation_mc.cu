// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#ifndef MTM_BLOCK_SIZE
#define MTM_BLOCK_SIZE 256
#endif
#ifndef MTM_HAS_LJ
#define MTM_HAS_LJ 1
#endif
#ifndef ONE_4PI_EPS0
#define ONE_4PI_EPS0 138.93545764438198
#endif
#ifndef EWALD_ALPHA
#define EWALD_ALPHA 1.0
#endif
#ifndef CUTOFF_SQUARED
#define CUTOFF_SQUARED 1.0
#endif
#define MTM_NEGATIVE_INFINITY (-__int_as_float(0x7f800000))

__device__ inline mixed3 loadMtmContextPosition(
        GLOBAL const real4* RESTRICT base,
        GLOBAL const float4* RESTRICT correction, int index) {
    const real4 value = base[index];
#ifdef USE_MIXED_PRECISION
    const float4 corr = correction[index];
    return make_mixed3((mixed) value.x+(mixed) corr.x,
                       (mixed) value.y+(mixed) corr.y,
                       (mixed) value.z+(mixed) corr.z);
#else
    return make_mixed3((mixed) value.x, (mixed) value.y, (mixed) value.z);
#endif
}

__device__ inline void storeMtmContextPosition(
        mixed3 value, GLOBAL real4* RESTRICT base,
        GLOBAL float4* RESTRICT correction, int index) {
#ifdef USE_MIXED_PRECISION
    const real x = (real) value.x;
    const real y = (real) value.y;
    const real z = (real) value.z;
    base[index] = make_real4(x, y, z, 0);
    correction[index] = make_float4((float) (value.x-(mixed) x),
                                    (float) (value.y-(mixed) y),
                                    (float) (value.z-(mixed) z), 0.0f);
#else
    base[index] = make_real4((real) value.x, (real) value.y, (real) value.z, 0);
#endif
}

__device__ inline void copyMtmContextPosition(
        GLOBAL const real4* RESTRICT sourceBase,
        GLOBAL const float4* RESTRICT sourceCorrection, int sourceIndex,
        GLOBAL real4* RESTRICT destinationBase,
        GLOBAL float4* RESTRICT destinationCorrection, int destinationIndex) {
    destinationBase[destinationIndex] = sourceBase[sourceIndex];
#ifdef USE_MIXED_PRECISION
    destinationCorrection[destinationIndex] = sourceCorrection[sourceIndex];
#endif
}

__device__ inline mixed3 applyMtmPeriodicDelta(mixed3 delta,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    const mixed lx = (mixed) periodicBoxVecX.x;
    const mixed ly = (mixed) periodicBoxVecY.y;
    const mixed lz = (mixed) periodicBoxVecZ.z;
    delta.x -= floor(delta.x/lx+(mixed) 0.5)*lx;
    delta.y -= floor(delta.y/ly+(mixed) 0.5)*ly;
    delta.z -= floor(delta.z/lz+(mixed) 0.5)*lz;
    return delta;
}

__device__ inline mixed3 applyMtmPeriodicPosition(mixed3 position,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    const mixed lx = (mixed) periodicBoxVecX.x;
    const mixed ly = (mixed) periodicBoxVecY.y;
    const mixed lz = (mixed) periodicBoxVecZ.z;
    position.x -= floor(position.x/lx)*lx;
    position.y -= floor(position.y/ly)*ly;
    position.z -= floor(position.z/lz)*lz;
    return position;
}

KERNEL void reorderMtmPositionsContextPrecision(
        int numAtoms,
        GLOBAL const real4* RESTRICT posq,
        GLOBAL const float4* RESTRICT posqCorrection,
        GLOBAL const int* RESTRICT atomOrder,
        GLOBAL real4* RESTRICT orderedBase,
        GLOBAL float4* RESTRICT orderedCorrection) {
    for (int atom = GLOBAL_ID; atom < numAtoms; atom += GLOBAL_SIZE) {
        const int hostAtom = atomOrder[atom];
        copyMtmContextPosition(posq, posqCorrection, atom,
                orderedBase, orderedCorrection, hostAtom);
    }
}

KERNEL void computeMtmLigandCenterContextPrecision(
        int numCenterAtoms,
        GLOBAL const real4* RESTRICT orderedBase,
        GLOBAL const float4* RESTRICT orderedCorrection,
        GLOBAL const int* RESTRICT centerAtoms,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL real4* RESTRICT centerBase,
        GLOBAL float4* RESTRICT centerCorrection) {
    if (GLOBAL_ID != 0)
        return;
    const mixed3 reference = loadMtmContextPosition(orderedBase, orderedCorrection, centerAtoms[0]);
    mixed3 sum = reference;
    for (int index = 1; index < numCenterAtoms; index++) {
        mixed3 delta = loadMtmContextPosition(orderedBase, orderedCorrection, centerAtoms[index])-reference;
        delta = applyMtmPeriodicDelta(delta, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
        sum += reference+delta;
    }
    mixed3 center = sum/(mixed) numCenterAtoms;
    center = applyMtmPeriodicPosition(center, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
    storeMtmContextPosition(center, centerBase, centerCorrection, 0);
}

KERNEL void classifyMtmMoleculesContextPrecision(
        int numMolecules, mixed sphereRadiusSquared,
        GLOBAL const real4* RESTRICT orderedBase,
        GLOBAL const float4* RESTRICT orderedCorrection,
        GLOBAL const real4* RESTRICT centerBase,
        GLOBAL const float4* RESTRICT centerCorrection,
        GLOBAL const int* RESTRICT referenceAtoms,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL int* RESTRICT moleculeRegionFlags,
        GLOBAL int* RESTRICT regionCounts) {
    const mixed3 center = loadMtmContextPosition(centerBase, centerCorrection, 0);
    for (int molecule = GLOBAL_ID; molecule < numMolecules; molecule += GLOBAL_SIZE) {
        mixed3 delta = loadMtmContextPosition(orderedBase, orderedCorrection,
                referenceAtoms[molecule])-center;
        delta = applyMtmPeriodicDelta(delta, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
        const mixed distanceSquared = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
        const int region = distanceSquared < (mixed) sphereRadiusSquared ? 0 : 1;
        moleculeRegionFlags[molecule] = region;
        atomicAdd(&regionCounts[region], 1);
    }
}

KERNEL void gatherMtmRealPositionsContextPrecision(
        int numRealAtoms,
        GLOBAL const int* RESTRICT realAtomIndices,
        GLOBAL const real4* RESTRICT orderedBase,
        GLOBAL const float4* RESTRICT orderedCorrection,
        GLOBAL real4* RESTRICT realBase,
        GLOBAL float4* RESTRICT realCorrection) {
    for (int atom = GLOBAL_ID; atom < numRealAtoms; atom += GLOBAL_SIZE)
        copyMtmContextPosition(orderedBase, orderedCorrection, realAtomIndices[atom],
                realBase, realCorrection, atom);
}

KERNEL void reorderPositions(
        int numAtoms,
        GLOBAL const real4* RESTRICT posq,
        GLOBAL const int* RESTRICT atomOrder,
        GLOBAL float4* RESTRICT orderedPositions) {
    for (int atom = GLOBAL_ID; atom < numAtoms; atom += GLOBAL_SIZE) {
        const int hostAtom = atomOrder[atom];
        const real4 position = posq[atom];
        orderedPositions[hostAtom] = make_float4((float) position.x, (float) position.y,
                                                (float) position.z, 0.0f);
    }
}

KERNEL void computeLigandCenter(
        int numCenterAtoms,
        GLOBAL const float4* RESTRICT orderedPositions,
        GLOBAL const int* RESTRICT centerAtoms,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL float4* RESTRICT center) {
    if (GLOBAL_ID != 0)
        return;
    const float4 reference4 = orderedPositions[centerAtoms[0]];
    const real3 reference = make_real3(reference4.x, reference4.y, reference4.z);
    real3 sum = reference;
    for (int index = 1; index < numCenterAtoms; index++) {
        const float4 position4 = orderedPositions[centerAtoms[index]];
        real3 delta = make_real3(position4.x, position4.y, position4.z)-reference;
        APPLY_PERIODIC_TO_DELTA(delta)
        sum += reference+delta;
    }
    real3 value = sum/(real) numCenterAtoms;
    APPLY_PERIODIC_TO_POS(value)
    center[0] = make_float4((float) value.x, (float) value.y, (float) value.z, 0.0f);
}

KERNEL void classifyMolecules(
        int numMolecules,
        float sphereRadiusSquared,
        GLOBAL const float4* RESTRICT orderedPositions,
        GLOBAL const float4* RESTRICT ligandCenter,
        GLOBAL const int* RESTRICT referenceAtoms,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL int* RESTRICT moleculeRegionFlags,
        GLOBAL int* RESTRICT regionCounts) {
    for (int molecule = GLOBAL_ID; molecule < numMolecules; molecule += GLOBAL_SIZE) {
        const float4 reference4 = orderedPositions[referenceAtoms[molecule]];
        const float4 center4 = ligandCenter[0];
        real3 delta = make_real3(reference4.x-center4.x, reference4.y-center4.y, reference4.z-center4.z);
        APPLY_PERIODIC_TO_DELTA(delta)
        const real distanceSquared = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
        const int region = distanceSquared < sphereRadiusSquared ? 0 : 1;
        moleculeRegionFlags[molecule] = region;
        atomicAdd(&regionCounts[region], 1);
    }
}

KERNEL void gatherRealPositions(
        int numRealAtoms,
        GLOBAL const int* RESTRICT realAtomIndices,
        GLOBAL const float4* RESTRICT orderedPositions,
        GLOBAL float4* RESTRICT realPositions) {
    for (int atom = GLOBAL_ID; atom < numRealAtoms; atom += GLOBAL_SIZE)
        realPositions[atom] = orderedPositions[realAtomIndices[atom]];
}

KERNEL void sampleInnerAnchors(
        int numCandidates,
        float sphereRadius,
        GLOBAL const float4* RESTRICT ligandCenter,
        GLOBAL const float4* RESTRICT randomValues,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL float4* RESTRICT anchors) {
    for (int candidate = GLOBAL_ID; candidate < numCandidates; candidate += GLOBAL_SIZE) {
        const float4 random = randomValues[candidate];
        real3 direction = make_real3(random.x, random.y, random.z);
        real normSquared = direction.x*direction.x+direction.y*direction.y+direction.z*direction.z;
        if (normSquared < 1e-12f)
            direction = make_real3(1, 0, 0);
        else
            direction *= RSQRT(normSquared);
        const real radius = sphereRadius*cbrtf(min(max(random.w, 0.0f), 0.99999994f));
        const float4 center4 = ligandCenter[0];
        real3 position = make_real3(center4.x, center4.y, center4.z)+radius*direction;
        APPLY_PERIODIC_TO_POS(position)
        anchors[candidate] = make_float4((float) position.x, (float) position.y, (float) position.z, 0.0f);
    }
}

KERNEL void sampleExternalAnchors(
        int numCandidates,
        float sphereRadiusSquared,
        GLOBAL const float4* RESTRICT ligandCenter,
        GLOBAL const float4* RESTRICT randomValues,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL float4* RESTRICT anchors,
        GLOBAL int* RESTRICT validFlags,
        GLOBAL int* RESTRICT regionCounts) {
    for (int candidate = GLOBAL_ID; candidate < numCandidates; candidate += GLOBAL_SIZE) {
        if (validFlags[candidate] != 0)
            continue;
        const float4 random = randomValues[candidate];
        real3 position = make_real3(random.x*periodicBoxVecX.x,
                                    random.y*periodicBoxVecY.y,
                                    random.z*periodicBoxVecZ.z);
        const float4 center4 = ligandCenter[0];
        real3 delta = position-make_real3(center4.x, center4.y, center4.z);
        APPLY_PERIODIC_TO_DELTA(delta)
        const real distanceSquared = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
        if (distanceSquared >= sphereRadiusSquared) {
            anchors[candidate] = make_float4((float) position.x, (float) position.y, (float) position.z, 0.0f);
            validFlags[candidate] = 1;
            atomicAdd(&regionCounts[2], 1);
        }
    }
}

__device__ inline unsigned long long mtmMix64(unsigned long long value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value^(value >> 30))*0xBF58476D1CE4E5B9ULL;
    value = (value^(value >> 27))*0x94D049BB133111EBULL;
    return value^(value >> 31);
}

__device__ inline float mtmUniform(unsigned long long seed, unsigned long long attempt,
        unsigned int stream, unsigned int index, unsigned int lane) {
    unsigned long long counter = seed;
    counter ^= 0xD2B74407B1CE6E93ULL*(attempt+1ULL);
    counter ^= 0xCA5A826395121157ULL*((unsigned long long) stream+1ULL);
    counter ^= 0x9E3779B97F4A7C15ULL*((unsigned long long) index+1ULL);
    counter ^= 0x94D049BB133111EBULL*((unsigned long long) lane+1ULL);
    const unsigned int mantissa = (unsigned int) (mtmMix64(counter) >> 40);
    return ((float) mantissa+0.5f)*(1.0f/16777216.0f);
}

__device__ inline float mtmGumbel(unsigned long long seed, unsigned long long attempt,
        unsigned int stream, unsigned int index) {
    const float uniform = mtmUniform(seed, attempt, stream, index, 0);
    return -logf(-logf(uniform));
}

__device__ inline real mtmErfcAlpha(real r) {
    const real alphaR = (real) EWALD_ALPHA*r;
    const real expAlphaRSqr = EXP(-alphaR*alphaR);
#ifdef USE_DOUBLE_PRECISION
    return erfc(alphaR);
#else
    const real t = RECIP(1.0f+0.3275911f*alphaR);
    return (0.254829592f+(-0.284496736f+(1.421413741f+
            (-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
}

__device__ inline mixed mtmErfcAlphaContextPrecision(mixed r) {
    const mixed alphaR = (mixed) EWALD_ALPHA*r;
    const mixed expAlphaRSqr = exp(-alphaR*alphaR);
#if defined(USE_DOUBLE_PRECISION) || defined(USE_MIXED_PRECISION)
    return erfc(alphaR);
#else
    const mixed t = ((mixed) 1)/((mixed) 1+(mixed) 0.3275911*alphaR);
    return ((mixed) 0.254829592+((mixed) -0.284496736+
            ((mixed) 1.421413741+((mixed) -1.453152027+
             (mixed) 1.061405429*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
}

KERNEL void selectMtmRealMolecule(
        int numMolecules, int sourceRegion, float beta,
        unsigned long long randomSeed, unsigned long long attempt,
        GLOBAL const int* RESTRICT moleculeRegionFlags,
        GLOBAL const float* RESTRICT moleculeEnergies,
        GLOBAL float4* RESTRICT selectedData,
        GLOBAL float4* RESTRICT proposalScalars) {
    if (blockIdx.x != 0)
        return;
    const int thread = threadIdx.x;
    __shared__ float maxima[MTM_BLOCK_SIZE];
    __shared__ float sums[MTM_BLOCK_SIZE];
    __shared__ float bestScores[MTM_BLOCK_SIZE];
    __shared__ int bestIndices[MTM_BLOCK_SIZE];
    __shared__ int invalidFlags[MTM_BLOCK_SIZE];
    __shared__ int eligibleCounts[MTM_BLOCK_SIZE];
    float localMaximum = MTM_NEGATIVE_INFINITY;
    int invalid = !isfinite(beta) || beta <= 0.0f;
    int eligibleCount = 0;
    for (int molecule = thread; molecule < numMolecules; molecule += blockDim.x) {
        const float energy = moleculeEnergies[molecule];
        const float scaled = beta*energy;
        if (!isfinite(energy) || !isfinite(scaled)) {
            // Source energies are a global partition.  Any nonfinite value,
            // including one outside sourceRegion, invalidates the proposal.
            invalid = 1;
            continue;
        }
        if (moleculeRegionFlags[molecule] == sourceRegion) {
            localMaximum = max(localMaximum, scaled);
            eligibleCount++;
        }
    }
    maxima[thread] = localMaximum;
    invalidFlags[thread] = invalid;
    eligibleCounts[thread] = eligibleCount;
    __syncthreads();
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (thread < offset) {
            maxima[thread] = max(maxima[thread], maxima[thread+offset]);
            invalidFlags[thread] |= invalidFlags[thread+offset];
            eligibleCounts[thread] += eligibleCounts[thread+offset];
        }
        __syncthreads();
    }
    if (thread == 0) {
        selectedData[0] = make_float4(-1.0f, 0.0f, 0.0f, 0.0f);
        proposalScalars[0] = make_float4(MTM_NEGATIVE_INFINITY, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();
    if (invalidFlags[0] || eligibleCounts[0] == 0 || !isfinite(maxima[0]))
        return;
    float localSum = 0.0f;
    float localBest = MTM_NEGATIVE_INFINITY;
    int localIndex = -1;
    for (int molecule = thread; molecule < numMolecules; molecule += blockDim.x) {
        if (moleculeRegionFlags[molecule] != sourceRegion)
            continue;
        const float shifted = beta*moleculeEnergies[molecule]-maxima[0];
        localSum += expf(shifted);
        const float score = shifted+mtmGumbel(randomSeed, attempt, 1, molecule);
        if (score > localBest || (score == localBest && (localIndex < 0 || molecule < localIndex))) {
            localBest = score;
            localIndex = molecule;
        }
    }
    sums[thread] = localSum;
    bestScores[thread] = localBest;
    bestIndices[thread] = localIndex;
    __syncthreads();
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (thread < offset) {
            sums[thread] += sums[thread+offset];
            const float otherScore = bestScores[thread+offset];
            const int otherIndex = bestIndices[thread+offset];
            if (otherScore > bestScores[thread] ||
                    (otherScore == bestScores[thread] && otherIndex >= 0 &&
                     (bestIndices[thread] < 0 || otherIndex < bestIndices[thread]))) {
                bestScores[thread] = otherScore;
                bestIndices[thread] = otherIndex;
            }
        }
        __syncthreads();
    }
    if (thread == 0) {
        const int selected = bestIndices[0];
        if (selected < 0 || selected >= numMolecules ||
                !(sums[0] > 0.0f) || !isfinite(sums[0]))
            return;
        const float logSum = maxima[0]+logf(sums[0]);
        const float energy = moleculeEnergies[selected];
        const float probability = expf(beta*energy-logSum);
        if (isfinite(logSum) && isfinite(probability) && probability > 0.0f && probability <= 1.0f) {
            selectedData[0] = make_float4((float) selected, energy, probability, 1.0f);
            proposalScalars[0] = make_float4(logSum, 0.0f, sums[0], 0.0f);
        }
    }
}

KERNEL void generateMtmCandidatePositionsContextPrecision(
        int numCandidates, int numAtomsPerMolecule, int referenceSite,
        mixed sphereRadiusSquared, unsigned long long randomSeed, unsigned long long attempt,
        GLOBAL const real4* RESTRICT ligandCenterBase,
        GLOBAL const float4* RESTRICT ligandCenterCorrection,
        GLOBAL const real4* RESTRICT selectedDeletedBase,
        GLOBAL const float4* RESTRICT selectedDeletedCorrection,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL real4* RESTRICT anchorBase,
        GLOBAL float4* RESTRICT anchorCorrection,
        GLOBAL int* RESTRICT candidateRegions,
        GLOBAL real4* RESTRICT candidateBase,
        GLOBAL float4* RESTRICT candidateCorrection) {
    for (int candidate = GLOBAL_ID; candidate < numCandidates; candidate += GLOBAL_SIZE) {
        mixed3 anchor = make_mixed3(
                (mixed) mtmUniform(randomSeed, attempt, 2, candidate, 0)*(mixed) periodicBoxVecX.x,
                (mixed) mtmUniform(randomSeed, attempt, 2, candidate, 1)*(mixed) periodicBoxVecY.y,
                (mixed) mtmUniform(randomSeed, attempt, 2, candidate, 2)*(mixed) periodicBoxVecZ.z);
        const mixed3 center = loadMtmContextPosition(ligandCenterBase, ligandCenterCorrection, 0);
        const mixed3 centerDelta = applyMtmPeriodicDelta(anchor-center,
                periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
        const mixed distanceSquared = centerDelta.x*centerDelta.x+
                centerDelta.y*centerDelta.y+centerDelta.z*centerDelta.z;
        candidateRegions[candidate] = distanceSquared < (mixed) sphereRadiusSquared ? 0 : 1;
        storeMtmContextPosition(anchor, anchorBase, anchorCorrection, candidate);

        const mixed u1 = (mixed) mtmUniform(randomSeed, attempt, 3, candidate, 0);
        const mixed u2 = (mixed) mtmUniform(randomSeed, attempt, 3, candidate, 1);
        const mixed u3 = (mixed) mtmUniform(randomSeed, attempt, 3, candidate, 2);
        const mixed root1 = sqrt((mixed) 1-u1);
        const mixed root2 = sqrt(u1);
        const mixed angle2 = (mixed) 6.2831853071795864769*u2;
        const mixed angle3 = (mixed) 6.2831853071795864769*u3;
        const mixed qx = root1*sin(angle2);
        const mixed qy = root1*cos(angle2);
        const mixed qz = root2*sin(angle3);
        const mixed qw = root2*cos(angle3);
        const mixed3 reference = loadMtmContextPosition(
                selectedDeletedBase, selectedDeletedCorrection, referenceSite);
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            mixed3 relative = loadMtmContextPosition(selectedDeletedBase,
                    selectedDeletedCorrection, site)-reference;
            relative = applyMtmPeriodicDelta(relative,
                    periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
            const mixed3 q = make_mixed3(qx, qy, qz);
            const mixed3 twiceCross = (mixed) 2*make_mixed3(
                    q.y*relative.z-q.z*relative.y,
                    q.z*relative.x-q.x*relative.z,
                    q.x*relative.y-q.y*relative.x);
            const mixed3 secondCross = make_mixed3(
                    q.y*twiceCross.z-q.z*twiceCross.y,
                    q.z*twiceCross.x-q.x*twiceCross.z,
                    q.x*twiceCross.y-q.y*twiceCross.x);
            const mixed3 position = anchor+relative+qw*twiceCross+secondCross;
            const int index = candidate*numAtomsPerMolecule+site;
            storeMtmContextPosition(position, candidateBase, candidateCorrection, index);
        }
    }
}

KERNEL void generateMtmCandidatePositions(
        int numCandidates, int numAtomsPerMolecule, int referenceSite,
        float sphereRadiusSquared, unsigned long long randomSeed, unsigned long long attempt,
        GLOBAL const float4* RESTRICT ligandCenter,
        GLOBAL const float4* RESTRICT selectedDeletedPositions,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL float4* RESTRICT anchors,
        GLOBAL int* RESTRICT candidateRegions,
        GLOBAL float4* RESTRICT candidatePositions) {
    for (int candidate = GLOBAL_ID; candidate < numCandidates; candidate += GLOBAL_SIZE) {
        real3 anchor = make_real3(mtmUniform(randomSeed, attempt, 2, candidate, 0)*periodicBoxVecX.x,
                                 mtmUniform(randomSeed, attempt, 2, candidate, 1)*periodicBoxVecY.y,
                                 mtmUniform(randomSeed, attempt, 2, candidate, 2)*periodicBoxVecZ.z);
        const float4 center4 = ligandCenter[0];
        real3 centerDelta = anchor-make_real3(center4.x, center4.y, center4.z);
        APPLY_PERIODIC_TO_DELTA(centerDelta)
        const real distanceSquared = dot(centerDelta, centerDelta);
        candidateRegions[candidate] = distanceSquared < sphereRadiusSquared ? 0 : 1;
        anchors[candidate] = make_float4((float) anchor.x, (float) anchor.y, (float) anchor.z, 0.0f);

        const float u1 = mtmUniform(randomSeed, attempt, 3, candidate, 0);
        const float u2 = mtmUniform(randomSeed, attempt, 3, candidate, 1);
        const float u3 = mtmUniform(randomSeed, attempt, 3, candidate, 2);
        const float root1 = sqrtf(1.0f-u1);
        const float root2 = sqrtf(u1);
        const float angle2 = 6.2831853071795864769f*u2;
        const float angle3 = 6.2831853071795864769f*u3;
        const float qx = root1*sinf(angle2);
        const float qy = root1*cosf(angle2);
        const float qz = root2*sinf(angle3);
        const float qw = root2*cosf(angle3);
        const float4 reference4 = selectedDeletedPositions[referenceSite];
        const real3 reference = make_real3(reference4.x, reference4.y, reference4.z);
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            const float4 source4 = selectedDeletedPositions[site];
            real3 relative = make_real3(source4.x, source4.y, source4.z)-reference;
            APPLY_PERIODIC_TO_DELTA(relative)
            const real3 q = make_real3(qx, qy, qz);
            const real3 twiceCross = (real) 2.0f*make_real3(
                    q.y*relative.z-q.z*relative.y,
                    q.z*relative.x-q.x*relative.z,
                    q.x*relative.y-q.y*relative.x);
            const real3 secondCross = make_real3(
                    q.y*twiceCross.z-q.z*twiceCross.y,
                    q.z*twiceCross.x-q.x*twiceCross.z,
                    q.x*twiceCross.y-q.y*twiceCross.x);
            const real3 rotated = relative+(real) qw*twiceCross+secondCross;
            const real3 position = anchor+rotated;
            const int index = candidate*numAtomsPerMolecule+site;
            candidatePositions[index] = make_float4((float) position.x, (float) position.y,
                                                    (float) position.z, 0.0f);
        }
    }
}

KERNEL void generateCandidatePositions(
        int numCandidateSites,
        int numOrientations,
        int numAtomsPerMolecule,
        GLOBAL const float4* RESTRICT anchors,
        GLOBAL const float4* RESTRICT rotatedReferencePositions,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL float4* RESTRICT candidatePositions) {
    const int numCandidates = numCandidateSites*numOrientations;
    const int numCandidateAtoms = numCandidates*numAtomsPerMolecule;
    for (int index = GLOBAL_ID; index < numCandidateAtoms; index += GLOBAL_SIZE) {
        const int candidate = index/numAtomsPerMolecule;
        const int atom = index-candidate*numAtomsPerMolecule;
        // Site-major layout preserves ordered first-success semantics:
        // [site0/orientation0..N-1, site1/orientation0..N-1, ...].
        const int anchorSite = candidate/numOrientations;
        const int orientation = candidate-anchorSite*numOrientations;
        const float4 anchor = anchors[anchorSite];
        const float4 offset = rotatedReferencePositions[orientation*numAtomsPerMolecule+atom];
        // Anchors are already wrapped.  Keep every other site in the same
        // molecular image so raw constrained distances remain rigid when an
        // offset crosses a periodic boundary.
        const real3 position = make_real3(anchor.x+offset.x, anchor.y+offset.y, anchor.z+offset.z);
        candidatePositions[index] = make_float4((float) position.x, (float) position.y,
                                                (float) position.z, 0.0f);
    }
}

KERNEL void sumRealMoleculeEnergies(
        int numMolecules,
        int numAtomsPerMolecule,
        GLOBAL const float* RESTRICT atomEnergies,
        GLOBAL float* RESTRICT moleculeEnergies) {
    for (int molecule = GLOBAL_ID; molecule < numMolecules; molecule += GLOBAL_SIZE) {
        float energy = 0.0f;
        for (int atom = 0; atom < numAtomsPerMolecule; atom++)
            energy += atomEnergies[molecule*numAtomsPerMolecule+atom];
        moleculeEnergies[molecule] = energy;
    }
}

KERNEL void selectRealMolecule(
        int numMolecules,
        int sourceRegion,
        int targetRegion,
        float beta,
        float selectionUniform,
        GLOBAL const int* RESTRICT moleculeRegionFlags,
        GLOBAL const float* RESTRICT moleculeEnergies,
        GLOBAL float4* RESTRICT selectedData,
        GLOBAL float4* RESTRICT proposalScalars) {
    if (GLOBAL_ID != 0)
        return;
    // Initialize a fail-closed result before examining any proposal quantity.
    // This also prevents stale selectedData from surviving an invalid partition.
    selectedData[0] = make_float4(-1.0f, 0.0f, 0.0f, 0.0f);
    proposalScalars[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (!isfinite(beta) || !isfinite(selectionUniform))
        return;

    float sourceMaximum = -FLT_MAX;
    float targetMaximum = -FLT_MAX;
    bool sourceFound = false;
    bool targetFound = false;
    bool validPartition = true;
    for (int molecule = 0; molecule < numMolecules; molecule++) {
        const int region = moleculeRegionFlags[molecule];
        if (region != sourceRegion && region != targetRegion)
            continue;
        const float energy = moleculeEnergies[molecule];
        const float scaled = beta*energy;
        if (!isfinite(energy) || !isfinite(scaled)) {
            validPartition = false;
            continue;
        }
        if (region == sourceRegion) {
            sourceMaximum = max(sourceMaximum, scaled);
            sourceFound = true;
        }
        if (region == targetRegion) {
            targetMaximum = max(targetMaximum, scaled);
            targetFound = true;
        }
    }
    if (!validPartition || !sourceFound)
        return;

    float sourceSum = 0.0f;
    float targetSum = 0.0f;
    for (int molecule = 0; molecule < numMolecules; molecule++) {
        const int region = moleculeRegionFlags[molecule];
        const float scaled = beta*moleculeEnergies[molecule];
        if (region == sourceRegion)
            sourceSum += expf(scaled-sourceMaximum);
        if (region == targetRegion)
            targetSum += expf(scaled-targetMaximum);
    }
    if (!(sourceSum > 0.0f) || !isfinite(sourceSum) || !isfinite(targetSum))
        return;
    const float sourceLogSum = sourceMaximum+logf(sourceSum);
    const float targetLogSum = targetFound ? targetMaximum+logf(targetSum) : -FLT_MAX;
    if (!isfinite(sourceLogSum) || !isfinite(targetLogSum))
        return;

    const float threshold = min(max(selectionUniform, 0.0f), 0.99999994f)*sourceSum;
    float cumulative = 0.0f;
    int selected = -1;
    for (int molecule = 0; molecule < numMolecules; molecule++) {
        if (moleculeRegionFlags[molecule] != sourceRegion)
            continue;
        cumulative += expf(beta*moleculeEnergies[molecule]-sourceMaximum);
        if (selected < 0 && cumulative > threshold)
            selected = molecule;
    }
    if (selected < 0)
        for (int molecule = numMolecules-1; molecule >= 0; molecule--)
            if (moleculeRegionFlags[molecule] == sourceRegion) {
                selected = molecule;
                break;
            }
    const float energy = selected >= 0 ? moleculeEnergies[selected] : 0.0f;
    const float probability = selected >= 0 ? expf(beta*energy-sourceLogSum) : 0.0f;
    if (selected < 0 || !isfinite(energy) || !isfinite(probability) || probability < 0.0f || probability > 1.0f)
        return;
    selectedData[0] = make_float4((float) selected, energy, probability, 0.0f);
    proposalScalars[0] = make_float4(sourceLogSum, targetLogSum, sourceSum, targetSum);
}

KERNEL void extractCharges(
        int numAtoms,
        GLOBAL const float4* RESTRICT parameters,
        GLOBAL float* RESTRICT charges) {
    for (int atom = GLOBAL_ID; atom < numAtoms; atom += GLOBAL_SIZE)
        charges[atom] = parameters[atom].x;
}

KERNEL void assembleRealPmeRemovalEnergies(
        int numMolecules,
        int numAtomsPerMolecule,
        GLOBAL const float4* RESTRICT parameters,
        GLOBAL const float* RESTRICT directEnergies,
        GLOBAL const float* RESTRICT fullPotentials,
        GLOBAL float* RESTRICT moleculeEnergies,
        GLOBAL float4* RESTRICT components) {
    for (int molecule = GLOBAL_ID; molecule < numMolecules; molecule += GLOBAL_SIZE) {
        float direct = 0.0f;
        float fullCross = 0.0f;
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            const int index = molecule*numAtomsPerMolecule+site;
            direct += directEnergies[index];
            fullCross += SQRT_ONE_4PI_EPS0*parameters[index].x*fullPotentials[index];
        }
        // This is the historical source-selection score: direct environment
        // interaction plus q*psi_full.  The latter contains one complete
        // isolated-molecule reciprocal mesh contribution r(pose); it is not the
        // coordinate-independent analytic Ewald self correction.  Preserve raw
        // components for diagnosis, but poison the aggregate with a
        // rejection sentinel if either component or the total is nonfinite.
        const float energy = direct+fullCross;
        moleculeEnergies[molecule] = isfinite(direct) && isfinite(fullCross) && isfinite(energy) ?
                energy : nanf("");
        components[molecule] = make_float4(direct, fullCross, 0.0f, fullCross);
    }
}

KERNEL void gatherMtmSelectedMoleculeContextPrecision(
        int numAtomsPerMolecule, int selectedMolecule,
        GLOBAL const real4* RESTRICT allBase,
        GLOBAL const float4* RESTRICT allCorrection,
        GLOBAL const float4* RESTRICT allParameters,
        GLOBAL real4* RESTRICT selectedBase,
        GLOBAL float4* RESTRICT selectedCorrection,
        GLOBAL float* RESTRICT selectedCharges) {
    for (int site = GLOBAL_ID; site < numAtomsPerMolecule; site += GLOBAL_SIZE) {
        const int index = selectedMolecule*numAtomsPerMolecule+site;
        copyMtmContextPosition(allBase, allCorrection, index,
                selectedBase, selectedCorrection, site);
        selectedCharges[site] = allParameters[index].x;
    }
}

KERNEL void gatherSelectedMoleculeData(
        int numAtomsPerMolecule,
        int selectedMolecule,
        GLOBAL const float4* RESTRICT allPositions,
        GLOBAL const float4* RESTRICT allParameters,
        GLOBAL float4* RESTRICT selectedPositions,
        GLOBAL float* RESTRICT selectedCharges) {
    for (int site = GLOBAL_ID; site < numAtomsPerMolecule; site += GLOBAL_SIZE) {
        const int index = selectedMolecule*numAtomsPerMolecule+site;
        selectedPositions[site] = allPositions[index];
        selectedCharges[site] = allParameters[index].x;
    }
}

KERNEL void assembleExactPmeEnergies(
        int numCandidates,
        int numAtomsPerMolecule,
        GLOBAL const float4* RESTRICT candidateParameters,
        GLOBAL const float* RESTRICT candidateFullDirect,
        GLOBAL const float* RESTRICT candidateDeletedDirect,
        GLOBAL const float* RESTRICT candidateFullPotentials,
        GLOBAL const float* RESTRICT candidateDeletedPotentials,
        GLOBAL const float4* RESTRICT deletedParameters,
        GLOBAL const float* RESTRICT deletedFullDirect,
        GLOBAL const float* RESTRICT deletedFullPotentials,
        GLOBAL const float* RESTRICT deletedSelfPotentials,
        GLOBAL const float4* RESTRICT selectedRemovalComponents,
        GLOBAL float* RESTRICT candidateEnergies,
        GLOBAL float4* RESTRICT selectedDeletedData,
        GLOBAL float4* RESTRICT candidateComponents,
        GLOBAL float4* RESTRICT deletedComponents) {
    if (GLOBAL_ID != 0)
        return;
    float deletedDirect = 0.0f;
    float deletedEnvironmentMesh = 0.0f;
    for (int site = 0; site < numAtomsPerMolecule; site++) {
        deletedDirect += deletedFullDirect[site];
        deletedEnvironmentMesh += SQRT_ONE_4PI_EPS0*deletedParameters[site].x*
                (deletedFullPotentials[site]-deletedSelfPotentials[site]);
    }
    const float deletedEnergy = deletedDirect+deletedEnvironmentMesh;
    const float removalMesh = selectedRemovalComponents[0].y;
    const bool validDeletedEnergy = isfinite(deletedDirect) && isfinite(deletedEnvironmentMesh) &&
            isfinite(removalMesh) && isfinite(deletedEnergy);
    selectedDeletedData[0].y = validDeletedEnergy ? deletedEnergy : nanf("");
    deletedComponents[0] = make_float4(deletedDirect, deletedEnvironmentMesh, removalMesh, 0.0f);
    for (int candidate = 0; candidate < numCandidates; candidate++) {
        float fullDirect = 0.0f;
        float deletedDirect = 0.0f;
        float fullMeshCross = 0.0f;
        float deletedMeshCross = 0.0f;
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            const int index = candidate*numAtomsPerMolecule+site;
            fullDirect += candidateFullDirect[index];
            deletedDirect += candidateDeletedDirect[index];
            const float charge = candidateParameters[index].x;
            fullMeshCross += SQRT_ONE_4PI_EPS0*charge*candidateFullPotentials[index];
            deletedMeshCross += SQRT_ONE_4PI_EPS0*charge*candidateDeletedPotentials[index];
        }
        const float environmentDirect = fullDirect-deletedDirect;
        const float insertionMesh = fullMeshCross-deletedMeshCross;
        const float energy = environmentDirect+insertionMesh;
        // Keep every raw diagnostic component, but never expose a nonfinite
        // aggregate as a candidate that the acceptance transform can select.
        const bool validCandidate = isfinite(fullDirect) && isfinite(deletedDirect) &&
                isfinite(fullMeshCross) && isfinite(deletedMeshCross) &&
                isfinite(environmentDirect) && isfinite(insertionMesh) && isfinite(energy);
        candidateEnergies[candidate] = validCandidate ? energy : nanf("");
        candidateComponents[candidate] = make_float4(environmentDirect, fullMeshCross,
                deletedMeshCross, insertionMesh);
    }
}

KERNEL void packMtmDeletedQueriesContextPrecision(
        int numAtomsPerMolecule, int numCandidateAtoms, int numRealAtoms,
        GLOBAL const real4* RESTRICT selectedDeletedBase,
        GLOBAL const float4* RESTRICT selectedDeletedCorrection,
        GLOBAL const real4* RESTRICT candidateBase,
        GLOBAL const float4* RESTRICT candidateCorrection,
        GLOBAL const real4* RESTRICT realBase,
        GLOBAL const float4* RESTRICT realCorrection,
        GLOBAL real4* RESTRICT queryBase,
        GLOBAL float4* RESTRICT queryCorrection) {
    const int total = numAtomsPerMolecule+numCandidateAtoms+numRealAtoms;
    for (int index = GLOBAL_ID; index < total; index += GLOBAL_SIZE) {
        if (index < numAtomsPerMolecule)
            copyMtmContextPosition(selectedDeletedBase, selectedDeletedCorrection, index,
                    queryBase, queryCorrection, index);
        else if (index < numAtomsPerMolecule+numCandidateAtoms)
            copyMtmContextPosition(candidateBase, candidateCorrection,
                    index-numAtomsPerMolecule, queryBase, queryCorrection, index);
        else
            copyMtmContextPosition(realBase, realCorrection,
                    index-numAtomsPerMolecule-numCandidateAtoms,
                    queryBase, queryCorrection, index);
    }
}

KERNEL void packMtmSelectedQueriesContextPrecision(
        int numAtomsPerMolecule, int numRealAtoms,
        GLOBAL const real4* RESTRICT selectedBase,
        GLOBAL const float4* RESTRICT selectedCorrection,
        GLOBAL const real4* RESTRICT realBase,
        GLOBAL const float4* RESTRICT realCorrection,
        GLOBAL real4* RESTRICT queryBase,
        GLOBAL float4* RESTRICT queryCorrection) {
    const int total = numAtomsPerMolecule+numRealAtoms;
    for (int index = GLOBAL_ID; index < total; index += GLOBAL_SIZE) {
        if (index < numAtomsPerMolecule)
            copyMtmContextPosition(selectedBase, selectedCorrection, index,
                    queryBase, queryCorrection, index);
        else
            copyMtmContextPosition(realBase, realCorrection,
                    index-numAtomsPerMolecule, queryBase, queryCorrection, index);
    }
}

KERNEL void gatherMtmSelectedCandidateContextPrecision(
        int numAtomsPerMolecule, int selectedCandidate,
        GLOBAL const real4* RESTRICT candidateBase,
        GLOBAL const float4* RESTRICT candidateCorrection,
        GLOBAL real4* RESTRICT selectedBase,
        GLOBAL float4* RESTRICT selectedCorrection) {
    for (int site = GLOBAL_ID; site < numAtomsPerMolecule; site += GLOBAL_SIZE)
        copyMtmContextPosition(candidateBase, candidateCorrection,
                selectedCandidate*numAtomsPerMolecule+site,
                selectedBase, selectedCorrection, site);
}

KERNEL void commitMtmSelectedPositionsContextPrecision(
        int numAtoms, int numAtomsPerMolecule,
        GLOBAL const int* RESTRICT atomOrder,
        GLOBAL const int* RESTRICT movedHostAtoms,
        GLOBAL const real4* RESTRICT selectedBase,
        GLOBAL const float4* RESTRICT selectedCorrection,
        GLOBAL real4* RESTRICT posq,
        GLOBAL float4* RESTRICT posqCorrection) {
    for (int atom = GLOBAL_ID; atom < numAtoms; atom += GLOBAL_SIZE) {
        const int hostAtom = atomOrder[atom];
        for (int site = 0; site < numAtomsPerMolecule; site++)
            if (movedHostAtoms[site] == hostAtom) {
                real4 value = posq[atom];
                const real4 selected = selectedBase[site];
                value.x = selected.x;
                value.y = selected.y;
                value.z = selected.z;
                posq[atom] = value;
#ifdef USE_MIXED_PRECISION
                posqCorrection[atom] = selectedCorrection[site];
#endif
                break;
            }
    }
}

KERNEL void packDeletedPmeQueries(
        int numAtomsPerMolecule, int numCandidateAtoms, int numRealAtoms,
        GLOBAL const float4* RESTRICT selectedDeletedPositions,
        GLOBAL const float4* RESTRICT candidatePositions,
        GLOBAL const float4* RESTRICT realPositions,
        GLOBAL float4* RESTRICT queryPositions) {
    const int total = numAtomsPerMolecule+numCandidateAtoms+numRealAtoms;
    for (int index = GLOBAL_ID; index < total; index += GLOBAL_SIZE) {
        if (index < numAtomsPerMolecule)
            queryPositions[index] = selectedDeletedPositions[index];
        else if (index < numAtomsPerMolecule+numCandidateAtoms)
            queryPositions[index] = candidatePositions[index-numAtomsPerMolecule];
        else
            queryPositions[index] = realPositions[index-numAtomsPerMolecule-numCandidateAtoms];
    }
}

KERNEL void packSelectedPmeQueries(
        int numAtomsPerMolecule, int numRealAtoms,
        GLOBAL const float4* RESTRICT selectedCandidatePositions,
        GLOBAL const float4* RESTRICT realPositions,
        GLOBAL float4* RESTRICT queryPositions) {
    const int total = numAtomsPerMolecule+numRealAtoms;
    for (int index = GLOBAL_ID; index < total; index += GLOBAL_SIZE)
        queryPositions[index] = index < numAtomsPerMolecule ? selectedCandidatePositions[index] :
                realPositions[index-numAtomsPerMolecule];
}

KERNEL void assembleMtmCandidateEnergies(
        int numCandidates, int numAtomsPerMolecule,
        GLOBAL const float4* RESTRICT candidateParameters,
        GLOBAL const float* RESTRICT candidateFullDirect,
        GLOBAL const float* RESTRICT candidateDeletedDirect,
        GLOBAL const float* RESTRICT candidateFullPotentials,
        GLOBAL const float* RESTRICT deletedQueryPotentials,
        GLOBAL const float4* RESTRICT deletedParameters,
        GLOBAL const float* RESTRICT deletedFullDirect,
        GLOBAL const float* RESTRICT deletedFullPotentials,
        GLOBAL float* RESTRICT candidateEnergies,
        GLOBAL float4* RESTRICT candidateComponents,
        GLOBAL float4* RESTRICT energyScalars,
        GLOBAL float4* RESTRICT deletedComponents) {
    if (GLOBAL_ID == 0) {
        float deletedDirect = 0.0f;
        float deletedEnvironmentMesh = 0.0f;
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            deletedDirect += deletedFullDirect[site];
            deletedEnvironmentMesh += SQRT_ONE_4PI_EPS0*deletedParameters[site].x*
                    (deletedFullPotentials[site]-deletedQueryPotentials[site]);
        }
        const float deletedEnergy = deletedDirect+deletedEnvironmentMesh;
        energyScalars[0] = make_float4(isfinite(deletedDirect) && isfinite(deletedEnvironmentMesh) &&
                isfinite(deletedEnergy) ? deletedEnergy : nanf(""), 0.0f, 0.0f, 0.0f);
        deletedComponents[0] = make_float4(deletedDirect, deletedEnvironmentMesh, 0.0f, deletedEnvironmentMesh);
    }
    for (int candidate = GLOBAL_ID; candidate < numCandidates; candidate += GLOBAL_SIZE) {
        float fullDirect = 0.0f;
        float deletedDirect = 0.0f;
        float fullMeshCross = 0.0f;
        float deletedMeshCross = 0.0f;
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            const int index = candidate*numAtomsPerMolecule+site;
            fullDirect += candidateFullDirect[index];
            deletedDirect += candidateDeletedDirect[index];
            const float charge = candidateParameters[index].x;
            fullMeshCross += SQRT_ONE_4PI_EPS0*charge*candidateFullPotentials[index];
            deletedMeshCross += SQRT_ONE_4PI_EPS0*charge*
                    deletedQueryPotentials[numAtomsPerMolecule+index];
        }
        const float environmentDirect = fullDirect-deletedDirect;
        const float insertionMesh = fullMeshCross-deletedMeshCross;
        const float energy = environmentDirect+insertionMesh;
        // Preserve +Inf for diagnostics.  NaN or negative infinity is stored as
        // NaN; selection assigns every nonfinite candidate exact zero weight.
        const bool invalid = isnan(fullDirect) || isnan(deletedDirect) ||
                isnan(fullMeshCross) || isnan(deletedMeshCross) ||
                isnan(environmentDirect) || isnan(insertionMesh) || isnan(energy) ||
                (isinf(fullDirect) && fullDirect < 0.0f) ||
                (isinf(deletedDirect) && deletedDirect < 0.0f) ||
                (isinf(fullMeshCross) && fullMeshCross < 0.0f) ||
                (isinf(deletedMeshCross) && deletedMeshCross < 0.0f) ||
                (isinf(environmentDirect) && environmentDirect < 0.0f) ||
                (isinf(insertionMesh) && insertionMesh < 0.0f) ||
                (isinf(energy) && energy < 0.0f);
        candidateEnergies[candidate] = invalid ? nanf("") : energy;
        candidateComponents[candidate] = make_float4(environmentDirect, fullMeshCross,
                deletedMeshCross, insertionMesh);
    }
}

KERNEL void selectMtmCandidate(
        int numCandidates, int targetRegion, float beta,
        unsigned long long randomSeed, unsigned long long attempt,
        GLOBAL const int* RESTRICT candidateRegions,
        GLOBAL const float* RESTRICT candidateEnergies,
        GLOBAL const float4* RESTRICT sourceScalars,
        GLOBAL float4* RESTRICT selectedData,
        GLOBAL float4* RESTRICT logSums) {
    if (blockIdx.x != 0)
        return;
    const int thread = threadIdx.x;
    __shared__ float maxima[MTM_BLOCK_SIZE];
    __shared__ float sums[MTM_BLOCK_SIZE];
    __shared__ float bestScores[MTM_BLOCK_SIZE];
    __shared__ int bestIndices[MTM_BLOCK_SIZE];
    __shared__ int invalidFlags[MTM_BLOCK_SIZE];
    __shared__ int positiveCounts[MTM_BLOCK_SIZE];
    float localMaximum = MTM_NEGATIVE_INFINITY;
    int invalid = !isfinite(beta) || beta <= 0.0f;
    int positiveCount = 0;
    for (int candidate = thread; candidate < numCandidates; candidate += blockDim.x) {
        if (candidateRegions[candidate] != targetRegion)
            continue;
        const float energy = candidateEnergies[candidate];
        if (!isfinite(energy))
            continue;
        const float logWeight = -beta*energy;
        if (!isfinite(logWeight))
            continue;
        localMaximum = max(localMaximum, logWeight);
        positiveCount++;
    }
    maxima[thread] = localMaximum;
    invalidFlags[thread] = invalid;
    positiveCounts[thread] = positiveCount;
    __syncthreads();
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (thread < offset) {
            maxima[thread] = max(maxima[thread], maxima[thread+offset]);
            invalidFlags[thread] |= invalidFlags[thread+offset];
            positiveCounts[thread] += positiveCounts[thread+offset];
        }
        __syncthreads();
    }
    if (thread == 0) {
        selectedData[0] = make_float4(-1.0f, 0.0f, 0.0f, 0.0f);
        logSums[0] = make_float4(sourceScalars[0].x, MTM_NEGATIVE_INFINITY, MTM_NEGATIVE_INFINITY, MTM_NEGATIVE_INFINITY);
    }
    __syncthreads();
    if (invalidFlags[0] || positiveCounts[0] == 0 || !isfinite(maxima[0]))
        return;
    float localSum = 0.0f;
    float localBest = MTM_NEGATIVE_INFINITY;
    int localIndex = -1;
    for (int candidate = thread; candidate < numCandidates; candidate += blockDim.x) {
        if (candidateRegions[candidate] != targetRegion)
            continue;
        const float energy = candidateEnergies[candidate];
        if (!isfinite(energy))
            continue;
        const float logWeight = -beta*energy;
        if (!isfinite(logWeight))
            continue;
        const float shifted = logWeight-maxima[0];
        if (isfinite(shifted)) {
            localSum += expf(shifted);
            const float score = shifted+mtmGumbel(randomSeed, attempt, 4, candidate);
            if (score > localBest || (score == localBest &&
                    (localIndex < 0 || candidate < localIndex))) {
                localBest = score;
                localIndex = candidate;
            }
        }
    }
    sums[thread] = localSum;
    bestScores[thread] = localBest;
    bestIndices[thread] = localIndex;
    __syncthreads();
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (thread < offset) {
            sums[thread] += sums[thread+offset];
            const float otherScore = bestScores[thread+offset];
            const int otherIndex = bestIndices[thread+offset];
            if (otherScore > bestScores[thread] ||
                    (otherScore == bestScores[thread] && otherIndex >= 0 &&
                     (bestIndices[thread] < 0 || otherIndex < bestIndices[thread]))) {
                bestScores[thread] = otherScore;
                bestIndices[thread] = otherIndex;
            }
        }
        __syncthreads();
    }
    if (thread == 0) {
        const int selected = bestIndices[0];
        if (selected < 0 || selected >= numCandidates ||
                !(sums[0] > 0.0f) || !isfinite(sums[0]))
            return;
        const float logWeightSum = maxima[0]+logf(sums[0]);
        const float energy = candidateEnergies[selected];
        const float probability = expf(-beta*energy-logWeightSum);
        const float4 previous = logSums[0];
        if (isfinite(logWeightSum) && isfinite(probability) && probability > 0.0f && probability <= 1.0f) {
            selectedData[0] = make_float4((float) selected, energy, probability, 1.0f);
            logSums[0] = make_float4(previous.x, previous.y, logWeightSum, previous.w);
        }
    }
}

KERNEL void gatherMtmSelectedCandidate(
        int numAtomsPerMolecule,
        GLOBAL const float4* RESTRICT selectedData,
        GLOBAL const float4* RESTRICT candidatePositions,
        GLOBAL float4* RESTRICT selectedPositions) {
    const int selected = (int) selectedData[0].x;
    if (selected < 0)
        return;
    for (int site = GLOBAL_ID; site < numAtomsPerMolecule; site += GLOBAL_SIZE)
        selectedPositions[site] = candidatePositions[selected*numAtomsPerMolecule+site];
}

KERNEL void gatherMtmSelectedComponent(
        GLOBAL const float4* RESTRICT selectedData,
        GLOBAL const float4* RESTRICT candidateComponents,
        GLOBAL float4* RESTRICT selectedComponent) {
    if (GLOBAL_ID == 0) {
        const int selected = (int) selectedData[0].x;
        selectedComponent[0] = selected >= 0 ? candidateComponents[selected] :
                make_float4(nanf(""), nanf(""), nanf(""), nanf(""));
    }
}

KERNEL void computeMtmDirectPairCorrectionsContextPrecision(
        int numMolecules, int numAtomsPerMolecule, int selectedMolecule,
        GLOBAL const real4* RESTRICT realBase,
        GLOBAL const float4* RESTRICT realCorrection,
        GLOBAL const float4* RESTRICT realParameters,
        GLOBAL const real4* RESTRICT selectedDeletedBase,
        GLOBAL const float4* RESTRICT selectedDeletedCorrection,
        GLOBAL const real4* RESTRICT selectedCandidateBase,
        GLOBAL const float4* RESTRICT selectedCandidateCorrection,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL float* RESTRICT directCorrections) {
    for (int molecule = GLOBAL_ID; molecule < numMolecules; molecule += GLOBAL_SIZE) {
        if (molecule == selectedMolecule) {
            directCorrections[molecule] = 0.0f;
            continue;
        }
        mixed oldInteraction = 0;
        mixed newInteraction = 0;
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            const int index = molecule*numAtomsPerMolecule+site;
            const mixed3 position = loadMtmContextPosition(realBase, realCorrection, index);
            const float4 parameter = realParameters[index];
            for (int movedSite = 0; movedSite < numAtomsPerMolecule; movedSite++) {
                const float4 movedParameter = realParameters[selectedMolecule*numAtomsPerMolecule+movedSite];
                const mixed3 oldPosition = loadMtmContextPosition(
                        selectedDeletedBase, selectedDeletedCorrection, movedSite);
                const mixed3 newPosition = loadMtmContextPosition(
                        selectedCandidateBase, selectedCandidateCorrection, movedSite);
                const mixed3 oldDelta = applyMtmPeriodicDelta(oldPosition-position,
                        periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                const mixed3 newDelta = applyMtmPeriodicDelta(newPosition-position,
                        periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                const mixed oldR2 = oldDelta.x*oldDelta.x+oldDelta.y*oldDelta.y+oldDelta.z*oldDelta.z;
                const mixed newR2 = newDelta.x*newDelta.x+newDelta.y*newDelta.y+newDelta.z*newDelta.z;
                if (oldR2 > (mixed) 1e-12 && oldR2 < (mixed) CUTOFF_SQUARED) {
                    const mixed invR = ((mixed) 1)/sqrt(oldR2);
                    const mixed r = oldR2*invR;
                    oldInteraction += (mixed) ONE_4PI_EPS0*(mixed) parameter.x*(mixed) movedParameter.x*
                            invR*mtmErfcAlphaContextPrecision(r);
#if MTM_HAS_LJ
                    const mixed sigma = (mixed) parameter.y+(mixed) movedParameter.y;
                    const mixed epsilon = (mixed) parameter.z*(mixed) movedParameter.z;
                    const mixed sigma2 = sigma*sigma;
                    const mixed sigma6 = sigma2*sigma2*sigma2;
                    const mixed v6 = sigma6/(oldR2*oldR2*oldR2);
                    oldInteraction += epsilon*(v6*v6-v6);
#endif
                }
                if (newR2 > (mixed) 1e-12 && newR2 < (mixed) CUTOFF_SQUARED) {
                    const mixed invR = ((mixed) 1)/sqrt(newR2);
                    const mixed r = newR2*invR;
                    newInteraction += (mixed) ONE_4PI_EPS0*(mixed) parameter.x*(mixed) movedParameter.x*
                            invR*mtmErfcAlphaContextPrecision(r);
#if MTM_HAS_LJ
                    const mixed sigma = (mixed) parameter.y+(mixed) movedParameter.y;
                    const mixed epsilon = (mixed) parameter.z*(mixed) movedParameter.z;
                    const mixed sigma2 = sigma*sigma;
                    const mixed sigma6 = sigma2*sigma2*sigma2;
                    const mixed v6 = sigma6/(newR2*newR2*newR2);
                    newInteraction += epsilon*(v6*v6-v6);
#endif
                }
            }
        }
        const mixed correction = newInteraction-oldInteraction;
        directCorrections[molecule] = isfinite(correction) ? (float) correction : nanf("");
    }
}

KERNEL void computeMtmDirectPairCorrections(
        int numMolecules, int numAtomsPerMolecule, int selectedMolecule,
        GLOBAL const float4* RESTRICT realPositions,
        GLOBAL const float4* RESTRICT realParameters,
        GLOBAL const float4* RESTRICT selectedDeletedPositions,
        GLOBAL const float4* RESTRICT selectedCandidatePositions,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL float* RESTRICT directCorrections) {
    for (int molecule = GLOBAL_ID; molecule < numMolecules; molecule += GLOBAL_SIZE) {
        if (molecule == selectedMolecule) {
            directCorrections[molecule] = 0.0f;
            continue;
        }
        real oldInteraction = 0;
        real newInteraction = 0;
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            const int index = molecule*numAtomsPerMolecule+site;
            const float4 position4 = realPositions[index];
            const float4 parameter = realParameters[index];
            const real3 position = make_real3(position4.x, position4.y, position4.z);
            for (int movedSite = 0; movedSite < numAtomsPerMolecule; movedSite++) {
                const float4 movedParameter = realParameters[selectedMolecule*numAtomsPerMolecule+movedSite];
                const float4 old4 = selectedDeletedPositions[movedSite];
                const float4 new4 = selectedCandidatePositions[movedSite];
                real3 oldDelta = make_real3(old4.x, old4.y, old4.z)-position;
                real3 newDelta = make_real3(new4.x, new4.y, new4.z)-position;
                APPLY_PERIODIC_TO_DELTA(oldDelta)
                APPLY_PERIODIC_TO_DELTA(newDelta)
                const real oldR2 = dot(oldDelta, oldDelta);
                const real newR2 = dot(newDelta, newDelta);
                if (oldR2 > (real) 1e-12f && oldR2 < (real) CUTOFF_SQUARED) {
                    const real invR = RSQRT(oldR2);
                    const real r = oldR2*invR;
                    oldInteraction += (real) ONE_4PI_EPS0*parameter.x*movedParameter.x*invR*mtmErfcAlpha(r);
#if MTM_HAS_LJ
                    const real sigma = parameter.y+movedParameter.y;
                    const real epsilon = parameter.z*movedParameter.z;
                    const real sigma2 = sigma*sigma;
                    const real sigma6 = sigma2*sigma2*sigma2;
                    const real v6 = sigma6/(oldR2*oldR2*oldR2);
                    oldInteraction += epsilon*(v6*v6-v6);
#endif
                }
                if (newR2 > (real) 1e-12f && newR2 < (real) CUTOFF_SQUARED) {
                    const real invR = RSQRT(newR2);
                    const real r = newR2*invR;
                    newInteraction += (real) ONE_4PI_EPS0*parameter.x*movedParameter.x*invR*mtmErfcAlpha(r);
#if MTM_HAS_LJ
                    const real sigma = parameter.y+movedParameter.y;
                    const real epsilon = parameter.z*movedParameter.z;
                    const real sigma2 = sigma*sigma;
                    const real sigma6 = sigma2*sigma2*sigma2;
                    const real v6 = sigma6/(newR2*newR2*newR2);
                    newInteraction += epsilon*(v6*v6-v6);
#endif
                }
            }
        }
        const real correction = newInteraction-oldInteraction;
        directCorrections[molecule] = isfinite(correction) ? (float) correction : nanf("");
    }
}

KERNEL void computeMtmFinalSums(
        int numMolecules, int numCandidates, int numAtomsPerMolecule,
        int selectedMolecule, int sourceRegion, int targetRegion, float beta,
        GLOBAL const int* RESTRICT moleculeRegions,
        GLOBAL const int* RESTRICT candidateRegions,
        GLOBAL const float* RESTRICT realMoleculeEnergies,
        GLOBAL const float4* RESTRICT realParameters,
        GLOBAL const float* RESTRICT candidateEnergies,
        GLOBAL const float* RESTRICT directCorrections,
        GLOBAL const float* RESTRICT deletedQueryPotentials,
        int deletedRealPotentialOffset,
        GLOBAL const float* RESTRICT selectedQueryPotentials,
        GLOBAL const float4* RESTRICT selectedData,
        GLOBAL float* RESTRICT proposedMoleculeScores,
        GLOBAL float4* RESTRICT logSums,
        GLOBAL float4* RESTRICT energyScalars) {
    if (blockIdx.x != 0)
        return;
    const int thread = threadIdx.x;
    const int selectedCandidate = (int) selectedData[0].x;
    __shared__ float maxima[MTM_BLOCK_SIZE];
    __shared__ float sums[MTM_BLOCK_SIZE];
    __shared__ int invalidFlags[MTM_BLOCK_SIZE];
    __shared__ float selectedSourceScore;
    __shared__ float isolatedReciprocalDelta;
    if (thread == 0) {
        float selectedIsolatedReciprocal = 0.0f;
        float deletedIsolatedReciprocal = 0.0f;
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            const float charge = realParameters[selectedMolecule*numAtomsPerMolecule+site].x;
            selectedIsolatedReciprocal += SQRT_ONE_4PI_EPS0*charge*selectedQueryPotentials[site];
            deletedIsolatedReciprocal += SQRT_ONE_4PI_EPS0*charge*deletedQueryPotentials[site];
        }
        const float selectedEnergy = selectedCandidate >= 0 ? candidateEnergies[selectedCandidate] : nanf("");
        selectedSourceScore = selectedEnergy+selectedIsolatedReciprocal;
        // q*psi from an isolated-molecule scratch mesh is a full cross quantity.
        // Its half difference is the candidate quadratic contribution to the
        // physical PME delta.  It is not the analytic Ewald self correction.
        isolatedReciprocalDelta = 0.5f*(selectedIsolatedReciprocal-deletedIsolatedReciprocal);
        invalidFlags[0] = selectedCandidate < 0 || !isfinite(selectedEnergy) ||
                !isfinite(selectedSourceScore) || !isfinite(isolatedReciprocalDelta);
    }
    __syncthreads();

    float localMaximum = thread == 0 && !invalidFlags[0] ? beta*selectedSourceScore : -FLT_MAX;
    int invalid = invalidFlags[0];
    for (int molecule = thread; molecule < numMolecules; molecule += blockDim.x) {
        if (moleculeRegions[molecule] != targetRegion) {
            proposedMoleculeScores[molecule] = nanf("");
            continue;
        }
        float reciprocalCorrection = 0.0f;
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            const int index = molecule*numAtomsPerMolecule+site;
            const float charge = realParameters[index].x;
            reciprocalCorrection += SQRT_ONE_4PI_EPS0*charge*
                    (selectedQueryPotentials[numAtomsPerMolecule+index]-
                     deletedQueryPotentials[deletedRealPotentialOffset+index]);
        }
        const float score = realMoleculeEnergies[molecule]+directCorrections[molecule]+reciprocalCorrection;
        proposedMoleculeScores[molecule] = score;
        const float scaled = beta*score;
        if (!isfinite(score) || !isfinite(scaled))
            invalid = 1;
        else
            localMaximum = max(localMaximum, scaled);
    }
    maxima[thread] = localMaximum;
    invalidFlags[thread] = invalid;
    __syncthreads();
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (thread < offset) {
            maxima[thread] = max(maxima[thread], maxima[thread+offset]);
            invalidFlags[thread] |= invalidFlags[thread+offset];
        }
        __syncthreads();
    }
    float localSum = 0.0f;
    if (!invalidFlags[0]) {
        if (thread == 0)
            localSum += expf(beta*selectedSourceScore-maxima[0]);
        for (int molecule = thread; molecule < numMolecules; molecule += blockDim.x)
            if (moleculeRegions[molecule] == targetRegion)
                localSum += expf(beta*proposedMoleculeScores[molecule]-maxima[0]);
    }
    sums[thread] = localSum;
    __syncthreads();
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (thread < offset)
            sums[thread] += sums[thread+offset];
        __syncthreads();
    }
    const float proposedLogSum = !invalidFlags[0] && sums[0] > 0.0f && isfinite(sums[0]) ?
            maxima[0]+logf(sums[0]) : nanf("");
    __syncthreads();

    const float deletedEnergy = energyScalars[0].x;
    localMaximum = thread == 0 && isfinite(deletedEnergy) ? -beta*deletedEnergy : MTM_NEGATIVE_INFINITY;
    invalid = !isfinite(deletedEnergy);
    for (int candidate = thread; candidate < numCandidates; candidate += blockDim.x) {
        if (candidate == selectedCandidate || candidateRegions[candidate] != sourceRegion)
            continue;
        const float energy = candidateEnergies[candidate];
        if (!isfinite(energy))
            continue;
        const float logWeight = -beta*energy;
        if (isfinite(logWeight))
            localMaximum = max(localMaximum, logWeight);
    }
    maxima[thread] = localMaximum;
    invalidFlags[thread] = invalid;
    __syncthreads();
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (thread < offset) {
            maxima[thread] = max(maxima[thread], maxima[thread+offset]);
            invalidFlags[thread] |= invalidFlags[thread+offset];
        }
        __syncthreads();
    }
    localSum = 0.0f;
    if (!invalidFlags[0]) {
        if (thread == 0)
            localSum += expf(-beta*deletedEnergy-maxima[0]);
        for (int candidate = thread; candidate < numCandidates; candidate += blockDim.x) {
            if (candidate == selectedCandidate || candidateRegions[candidate] != sourceRegion)
                continue;
            const float energy = candidateEnergies[candidate];
            const float logWeight = -beta*energy;
            if (isfinite(energy) && isfinite(logWeight))
                localSum += expf(logWeight-maxima[0]);
        }
    }
    sums[thread] = localSum;
    __syncthreads();
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (thread < offset)
            sums[thread] += sums[thread+offset];
        __syncthreads();
    }
    if (thread == 0) {
        const float reverseLogSum = !invalidFlags[0] && sums[0] > 0.0f && isfinite(sums[0]) ?
                maxima[0]+logf(sums[0]) : nanf("");
        const float4 previous = logSums[0];
        logSums[0] = make_float4(previous.x, proposedLogSum, previous.z, reverseLogSum);
        const float selectedEnergy = selectedCandidate >= 0 ? candidateEnergies[selectedCandidate] : nanf("");
        energyScalars[0] = make_float4(deletedEnergy, selectedEnergy, selectedSourceScore, isolatedReciprocalDelta);
    }
}

KERNEL void sumCandidateMoleculeEnergies(
        int numCandidates,
        int numAtomsPerMolecule,
        GLOBAL const float4* RESTRICT parameters,
        GLOBAL const float* RESTRICT directEnergies,
        GLOBAL const float* RESTRICT pmePotentials,
        GLOBAL float* RESTRICT moleculeEnergies) {
    for (int candidate = GLOBAL_ID; candidate < numCandidates; candidate += GLOBAL_SIZE) {
        float energy = 0.0f;
        for (int atom = 0; atom < numAtomsPerMolecule; atom++) {
            const int index = candidate*numAtomsPerMolecule+atom;
            energy += directEnergies[index]+SQRT_ONE_4PI_EPS0*parameters[index].x*pmePotentials[index];
        }
        moleculeEnergies[candidate] = energy;
    }
}

// Preserve the legacy ordered first-success proposal exactly for finite inputs.
// This is not a categorical CBMC selection: each candidate gets an independent
// Bernoulli draw and the lowest-index success is returned.  Invalid candidates
// have probability zero; invalid global proposal terms reject every candidate.
KERNEL void selectCandidateMolecule(
        int numCandidates,
        float beta,
        float sourceLogSum,
        float targetLogSum,
        float cavityBias,
        float movingBias,
        GLOBAL const float* RESTRICT acceptanceUniforms,
        GLOBAL const float* RESTRICT candidateEnergies,
        GLOBAL float4* RESTRICT selectedData) {
    if (GLOBAL_ID != 0)
        return;
    selectedData[0] = make_float4(-1.0f, 0.0f, 0.0f, 0.0f);
    if (!isfinite(beta) || !isfinite(sourceLogSum) || !isfinite(targetLogSum) ||
            !isfinite(cavityBias) || !isfinite(movingBias))
        return;

    int selected = -1;
    float selectedProbability = 0.0f;
    for (int candidate = 0; candidate < numCandidates; candidate++) {
        const float energy = candidateEnergies[candidate];
        const float acceptanceUniform = acceptanceUniforms[candidate];
        if (!isfinite(energy) || !isfinite(acceptanceUniform) ||
                acceptanceUniform < 0.0f || acceptanceUniform > 1.0f)
            continue;
        const float scaledEnergy = beta*energy;
        if (!isfinite(scaledEnergy))
            continue;
        const float combinedMaximum = max(targetLogSum, scaledEnergy);
        const float insertionLogSum = targetLogSum <= -FLT_MAX/2 ? scaledEnergy :
                combinedMaximum+logf(expf(targetLogSum-combinedMaximum)+expf(scaledEnergy-combinedMaximum));
        if (!isfinite(combinedMaximum) || !isfinite(insertionLogSum))
            continue;
        const float logProbability = sourceLogSum-insertionLogSum+cavityBias+movingBias;
        if (!isfinite(logProbability))
            continue;
        // min(1, exp(logProbability)) is evaluated without overflowing exp for
        // positive finite logs, preserving its finite-input mathematical value.
        const float probability = logProbability >= 0.0f ? 1.0f : expf(logProbability);
        if (!isfinite(probability) || probability < 0.0f || probability > 1.0f)
            continue;
        selectedProbability = max(selectedProbability, probability);
        if (selected < 0 && acceptanceUniform < probability) {
            selected = candidate;
            selectedProbability = probability;
        }
    }
    const float energy = selected >= 0 ? candidateEnergies[selected] : 0.0f;
    selectedData[0] = make_float4((float) selected, energy, selectedProbability, selected >= 0 ? 1.0f : 0.0f);
}

KERNEL void gatherSelectedCandidatePositions(
        int numAtomsPerMolecule,
        int selectedCandidate,
        GLOBAL const float4* RESTRICT candidatePositions,
        GLOBAL float4* RESTRICT selectedPositions) {
    for (int site = GLOBAL_ID; site < numAtomsPerMolecule; site += GLOBAL_SIZE)
        selectedPositions[site] = candidatePositions[selectedCandidate*numAtomsPerMolecule+site];
}
