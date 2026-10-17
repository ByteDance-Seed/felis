// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#ifndef OPENMM_CUDA_RELOCATIONMC_KERNELS_H_
#define OPENMM_CUDA_RELOCATIONMC_KERNELS_H_

#include "RelocationMC.h"
#include "RelocationMCKernels.h"
#include "RelocationMCOrientations.h"
#include "internal/PerturbNonbondedForceImpl.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaContext.h"
#include <cuda.h>
#include <cstdint>
#include <random>
#include <sstream>
#include <vector>

namespace OpenMM {
namespace RelocationMCPlugin {

/** CUDA implementation of the cavity-biased delete/reinsert protocol.
 *
 * Region classification, candidate generation, interaction-energy assembly, and
 * ordered first-success decisions remain GPU-resident.  PerturbNonbondedForce
 * evaluates exact PME environment interactions by default and the historical
 * asymmetric estimator on request.  Proposal summaries are downloaded for each
 * call; an accepted move additionally transfers coordinates and velocities so
 * ContextImpl can commit the rigid transform and invalidate position caches.
 */
class CudaCalcRelocationMCKernel : public CalcRelocationMCKernel {
public:
    enum class CandidateEnergyFault {
        None,
        SingleNaN,
        SinglePositiveInfinity,
        SingleNegativeInfinity,
        AllNaN,
        AllPositiveInfinity,
        AllNegativeInfinity
    };

    CudaCalcRelocationMCKernel(std::string name, const Platform& platform,
            CudaContext& cu, const System& system);
    ~CudaCalcRelocationMCKernel() override;

    void initialize(const System& system, const RelocationMC& mc, ContextImpl& context) override;
    ExchangeResult attemptExchange(ContextImpl& context, std::stringstream& logStream) override;
    void copyParametersToContext(ContextImpl& context, const RelocationMC& mc) override;
    static double computeProposalProbabilityForTesting(int sourceCount, int otherCount) {
        return computeProposalProbability(sourceCount, otherCount);
    }
    static bool isCandidateDecisionValidForTesting(float selectedCandidate, float candidateEnergy,
            float probability, float acceptedFlag, int numCandidates) {
        return isCandidateDecisionValid(selectedCandidate, candidateEnergy, probability,
                acceptedFlag, numCandidates);
    }
    /** Test-only fault injection consumed by the CUDA integration regression. */
    static void setCandidateEnergyFaultForTesting(CandidateEnergyFault fault) {
        candidateEnergyFaultForTesting = fault;
    }
    static void setForceNonfiniteCandidateEnergiesForTesting(bool enabled) {
        candidateEnergyFaultForTesting = enabled ? CandidateEnergyFault::AllNaN : CandidateEnergyFault::None;
    }
    static void setCaptureMtmDiagnosticsForTesting(bool enabled) {
        captureMtmDiagnosticsForTesting = enabled;
    }
    static const std::vector<float>& getLastMtmProposedScoresForTesting() {
        return lastMtmProposedScoresForTesting;
    }

private:
    static double computeProposalProbability(int sourceCount, int otherCount);
    static bool isCandidateDecisionValid(float selectedCandidate, float candidateEnergy,
            float probability, float acceptedFlag, int numCandidates);
    static thread_local CandidateEnergyFault candidateEnergyFaultForTesting;
    static thread_local bool captureMtmDiagnosticsForTesting;
    static thread_local std::vector<float> lastMtmProposedScoresForTesting;

    CudaContext& cu;
    PerturbNonbondedPlugin::PerturbNonbondedForceImpl* nonbonded;

    CudaArray ligandAtomIndices;
    CudaArray referenceAtomIndices;
    CudaArray realAtomIndicesDevice;
    CudaArray orderedPositions;
    CudaArray ligandCenter;
    CudaArray moleculeRegionFlags;
    CudaArray regionCounts;
    CudaArray proposalScalars;
    CudaArray selectedRealData;
    CudaArray selectedCandidateData;
    CudaArray selectedDeletedPositions;
    CudaArray selectedCandidatePositions;
    CudaArray mtmLogSums;
    CudaArray mtmEnergyScalars;

    CudaArray realPositions;
    CudaArray realParameters;
    CudaArray realExcludedStarts;
    CudaArray realExcludedCounts;
    CudaArray realExcludedAtoms;
    CudaArray realForces;
    CudaArray realAtomEnergies;
    CudaArray realPmePotentials;
    CudaArray selectedDeletedPmePotentials;
    CudaArray realCharges;
    CudaArray realPmeRemovalComponents;
    CudaArray realMoleculeEnergies;
    CudaArray realEnergyComponents;

    CudaArray candidateAnchors;
    CudaArray candidateRandomNumbers;
    CudaArray candidateAcceptanceRandom;
    CudaArray candidateValidFlags;
    CudaArray candidateRegionFlags;
    CudaArray rotatedReferencePositions;
    CudaArray candidatePositions;
    CudaArray candidateParameters;
    CudaArray candidateForces;
    CudaArray candidateDirectEnergies;
    CudaArray candidateDeletedDirectEnergies;
    CudaArray candidatePmePotentials;
    CudaArray candidateDeletedPmePotentials;
    CudaArray candidateCharges;
    CudaArray candidateMoleculeEnergies;
    CudaArray candidateEnergyComponents;
    CudaArray selectedCandidateEnergyComponents;
    CudaArray deletedPmeQueryPositions;
    CudaArray deletedPmeQueryPotentials;
    CudaArray selectedPmeQueryPositions;
    CudaArray selectedPmeQueryPotentials;
    CudaArray mtmDirectPairCorrections;
    CudaArray mtmProposedMoleculeScores;

    // Context-precision position chain used only by protocol v2.  Base arrays
    // are float4 in single/mixed precision and double4 in double precision.
    // Mixed precision also carries an exact float4 correction array.
    CudaArray mtmOrderedPositionBase;
    CudaArray mtmOrderedPositionCorrection;
    CudaArray mtmLigandCenterBase;
    CudaArray mtmLigandCenterCorrection;
    CudaArray mtmRealPositionBase;
    CudaArray mtmRealPositionCorrection;
    CudaArray mtmSelectedDeletedPositionBase;
    CudaArray mtmSelectedDeletedPositionCorrection;
    CudaArray mtmCandidateAnchorBase;
    CudaArray mtmCandidateAnchorCorrection;
    CudaArray mtmCandidatePositionBase;
    CudaArray mtmCandidatePositionCorrection;
    CudaArray mtmSelectedCandidatePositionBase;
    CudaArray mtmSelectedCandidatePositionCorrection;
    CudaArray mtmDeletedQueryPositionBase;
    CudaArray mtmDeletedQueryPositionCorrection;
    CudaArray mtmSelectedQueryPositionBase;
    CudaArray mtmSelectedQueryPositionCorrection;

    CUfunction reorderPositionsKernel;
    CUfunction computeLigandCenterKernel;
    CUfunction classifyMoleculesKernel;
    CUfunction gatherRealPositionsKernel;
    CUfunction sampleInnerAnchorsKernel;
    CUfunction sampleExternalAnchorsKernel;
    CUfunction generateCandidatePositionsKernel;
    CUfunction sumRealMoleculeEnergiesKernel;
    CUfunction selectRealMoleculeKernel;
    CUfunction sumCandidateMoleculeEnergiesKernel;
    CUfunction selectCandidateMoleculeKernel;
    CUfunction gatherSelectedCandidatePositionsKernel;
    CUfunction gatherSelectedMoleculeDataKernel;
    CUfunction extractChargesKernel;
    CUfunction assembleRealPmeRemovalEnergiesKernel;
    CUfunction assembleExactPmeEnergiesKernel;
    CUfunction selectMtmRealMoleculeKernel;
    CUfunction generateMtmCandidatePositionsKernel;
    CUfunction assembleMtmCandidateEnergiesKernel;
    CUfunction selectMtmCandidateKernel;
    CUfunction gatherMtmSelectedCandidateKernel;
    CUfunction gatherMtmSelectedComponentKernel;
    CUfunction packDeletedPmeQueriesKernel;
    CUfunction packSelectedPmeQueriesKernel;
    CUfunction computeMtmDirectPairCorrectionsKernel;
    CUfunction computeMtmFinalSumsKernel;
    CUfunction reorderMtmPositionsContextPrecisionKernel;
    CUfunction computeMtmLigandCenterContextPrecisionKernel;
    CUfunction classifyMtmMoleculesContextPrecisionKernel;
    CUfunction gatherMtmRealPositionsContextPrecisionKernel;
    CUfunction gatherMtmSelectedMoleculeContextPrecisionKernel;
    CUfunction generateMtmCandidatePositionsContextPrecisionKernel;
    CUfunction gatherMtmSelectedCandidateContextPrecisionKernel;
    CUfunction packMtmDeletedQueriesContextPrecisionKernel;
    CUfunction packMtmSelectedQueriesContextPrecisionKernel;
    CUfunction computeMtmDirectPairCorrectionsContextPrecisionKernel;
    CUfunction commitMtmSelectedPositionsContextPrecisionKernel;

    double temperature;
    double sphereRadius;
    int numCandidateSites;
    int numCandidateMolecules;
    int numAtomsPerMolecule;
    int protocolVersion;
    bool orientationsEnabled;
    bool useCavityBias;
    bool useLegacyAsymmetricEnergies;
    bool verbose;
    uint64_t randomSeed;
    uint64_t effectiveRandomSeed;
    uint64_t attemptCounter;
    double pmeAlpha;
    double directCutoff;

    std::vector<int> ligandAtoms;
    std::vector<LabileMol> labileMolecules;
    std::vector<LabileMolSiteParameters> siteParameters;
    std::vector<Vec3> referencePositions;
    PreparedOrientations preparedOrientations;
    int numPreparedOrientations;
    std::vector<int> realAtomIndices;
    std::vector<int> excludedStarts;
    std::vector<int> excludedCounts;
    std::vector<int> excludedAtomIndices;

    std::mt19937_64 random;
};

} // namespace RelocationMCPlugin
} // namespace OpenMM

#endif
