// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#include "CudaRelocationMCKernels.h"
#include "CudaRelocationMCKernelSources.h"
#include "openmm/cuda/CudaKernelSources.h"
#include "openmm/OpenMMException.h"
#include "openmm/AndersenThermostat.h"
#include "openmm/CMAPTorsionForce.h"
#include "openmm/CMMotionRemover.h"
#include "openmm/CustomAngleForce.h"
#include "openmm/CustomBondForce.h"
#include "openmm/CustomCompoundBondForce.h"
#include "openmm/CustomTorsionForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/MonteCarloAnisotropicBarostat.h"
#include "openmm/MonteCarloBarostat.h"
#include "openmm/MonteCarloMembraneBarostat.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/RBTorsionForce.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include "internal/RelocationMCVelocity.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using namespace OpenMM;
using namespace OpenMM::RelocationMCPlugin;
using namespace PerturbNonbondedPlugin;
using std::string;
using std::stringstream;
using std::vector;

thread_local CudaCalcRelocationMCKernel::CandidateEnergyFault
        CudaCalcRelocationMCKernel::candidateEnergyFaultForTesting =
                CudaCalcRelocationMCKernel::CandidateEnergyFault::None;
thread_local bool CudaCalcRelocationMCKernel::captureMtmDiagnosticsForTesting = false;
thread_local vector<float> CudaCalcRelocationMCKernel::lastMtmProposedScoresForTesting;

namespace {
constexpr double BOLTZMANN = 0.00831446261815324;
constexpr double ONE_4PI_EPS0_VALUE = 138.93545764438198;
constexpr double PI = 3.14159265358979323846;

mm_float4 toFloat4(const Vec3& value, float w = 0.0f) {
    return mm_float4((float) value[0], (float) value[1], (float) value[2], w);
}

bool isFiniteSourceDecision(const mm_float4& selected, const mm_float4& scalars, int numMolecules) {
    if (!std::isfinite(selected.x) || !std::isfinite(selected.y) || !std::isfinite(selected.z) ||
            selected.x != std::trunc(selected.x) || selected.x < 0.0f || selected.x >= numMolecules ||
            selected.z < 0.0f || selected.z > 1.0f)
        return false;
    return std::isfinite(scalars.x) && std::isfinite(scalars.y) &&
            std::isfinite(scalars.z) && std::isfinite(scalars.w) &&
            scalars.z > 0.0f && scalars.w >= 0.0f;
}

bool isFiniteVec3(const Vec3& value) {
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

bool isFiniteRotation(const RelocationMCVelocity::RotationMatrix& rotation) {
    return isFiniteVec3(rotation[0]) && isFiniteVec3(rotation[1]) && isFiniteVec3(rotation[2]);
}

uint64_t mixCounter(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value^(value >> 30))*0xBF58476D1CE4E5B9ULL;
    value = (value^(value >> 27))*0x94D049BB133111EBULL;
    return value^(value >> 31);
}

double counterUniform(uint64_t seed, uint64_t attempt, uint32_t stream, uint32_t index = 0, uint32_t lane = 0) {
    uint64_t counter = seed;
    counter ^= 0xD2B74407B1CE6E93ULL*(attempt+1ULL);
    counter ^= 0xCA5A826395121157ULL*(static_cast<uint64_t>(stream)+1ULL);
    counter ^= 0x9E3779B97F4A7C15ULL*(static_cast<uint64_t>(index)+1ULL);
    counter ^= 0x94D049BB133111EBULL*(static_cast<uint64_t>(lane)+1ULL);
    const uint32_t mantissa = static_cast<uint32_t>(mixCounter(counter) >> 40);
    return (static_cast<double>(mantissa)+0.5)/16777216.0;
}

void validateMtmRigidInteraction(const vector<int>& atoms, const vector<int>& moleculeByAtom,
        const string& forceName) {
    int labileMolecule = -1;
    bool containsLabile = false;
    for (int atom : atoms) {
        const int molecule = atom >= 0 && atom < (int) moleculeByAtom.size() ? moleculeByAtom[atom] : -1;
        if (molecule >= 0) {
            containsLabile = true;
            if (labileMolecule < 0)
                labileMolecule = molecule;
            else if (labileMolecule != molecule)
                throw OpenMMException("RelocationMC MTM does not support "+forceName+" terms spanning labile molecules.");
        }
        else if (containsLabile)
            throw OpenMMException("RelocationMC MTM does not support "+forceName+" terms between labile and nonlabile atoms.");
    }
    if (containsLabile)
        for (int atom : atoms)
            if (atom < 0 || atom >= (int) moleculeByAtom.size() || moleculeByAtom[atom] != labileMolecule)
                throw OpenMMException("RelocationMC MTM does not support "+forceName+" terms between labile and nonlabile atoms.");
}

void validateMtmHamiltonian(const System& system, const vector<LabileMol>& molecules) {
    vector<int> moleculeByAtom(system.getNumParticles(), -1);
    for (int molecule = 0; molecule < (int) molecules.size(); molecule++)
        for (int atom : molecules[molecule].atomIndices)
            moleculeByAtom[atom] = molecule;
    for (int constraint = 0; constraint < system.getNumConstraints(); constraint++) {
        int first, second;
        double distance;
        system.getConstraintParameters(constraint, first, second, distance);
        validateMtmRigidInteraction({first, second}, moleculeByAtom, "constraint");
    }
    for (int index = 0; index < system.getNumForces(); index++) {
        const Force& force = system.getForce(index);
        if (dynamic_cast<const PerturbNonbondedForce*>(&force) != nullptr ||
                dynamic_cast<const RelocationMC*>(&force) != nullptr ||
                dynamic_cast<const CMMotionRemover*>(&force) != nullptr ||
                dynamic_cast<const AndersenThermostat*>(&force) != nullptr ||
                dynamic_cast<const MonteCarloBarostat*>(&force) != nullptr ||
                dynamic_cast<const MonteCarloAnisotropicBarostat*>(&force) != nullptr ||
                dynamic_cast<const MonteCarloMembraneBarostat*>(&force) != nullptr)
            continue;
        if (const HarmonicBondForce* bonded = dynamic_cast<const HarmonicBondForce*>(&force)) {
            for (int term = 0; term < bonded->getNumBonds(); term++) {
                int first, second;
                double length, k;
                bonded->getBondParameters(term, first, second, length, k);
                validateMtmRigidInteraction({first, second}, moleculeByAtom, "HarmonicBondForce");
            }
            continue;
        }
        if (const CustomBondForce* bonded = dynamic_cast<const CustomBondForce*>(&force)) {
            for (int term = 0; term < bonded->getNumBonds(); term++) {
                int first, second;
                vector<double> parameters;
                bonded->getBondParameters(term, first, second, parameters);
                validateMtmRigidInteraction({first, second}, moleculeByAtom, "CustomBondForce");
            }
            continue;
        }
        if (const HarmonicAngleForce* bonded = dynamic_cast<const HarmonicAngleForce*>(&force)) {
            for (int term = 0; term < bonded->getNumAngles(); term++) {
                int first, second, third;
                double angle, k;
                bonded->getAngleParameters(term, first, second, third, angle, k);
                validateMtmRigidInteraction({first, second, third}, moleculeByAtom, "HarmonicAngleForce");
            }
            continue;
        }
        if (const CustomAngleForce* bonded = dynamic_cast<const CustomAngleForce*>(&force)) {
            for (int term = 0; term < bonded->getNumAngles(); term++) {
                int first, second, third;
                vector<double> parameters;
                bonded->getAngleParameters(term, first, second, third, parameters);
                validateMtmRigidInteraction({first, second, third}, moleculeByAtom, "CustomAngleForce");
            }
            continue;
        }
        if (const PeriodicTorsionForce* bonded = dynamic_cast<const PeriodicTorsionForce*>(&force)) {
            for (int term = 0; term < bonded->getNumTorsions(); term++) {
                int first, second, third, fourth, periodicity;
                double phase, k;
                bonded->getTorsionParameters(term, first, second, third, fourth, periodicity, phase, k);
                validateMtmRigidInteraction({first, second, third, fourth}, moleculeByAtom, "PeriodicTorsionForce");
            }
            continue;
        }
        if (const RBTorsionForce* bonded = dynamic_cast<const RBTorsionForce*>(&force)) {
            for (int term = 0; term < bonded->getNumTorsions(); term++) {
                int first, second, third, fourth;
                double c0, c1, c2, c3, c4, c5;
                bonded->getTorsionParameters(term, first, second, third, fourth, c0, c1, c2, c3, c4, c5);
                validateMtmRigidInteraction({first, second, third, fourth}, moleculeByAtom, "RBTorsionForce");
            }
            continue;
        }
        if (const CustomTorsionForce* bonded = dynamic_cast<const CustomTorsionForce*>(&force)) {
            for (int term = 0; term < bonded->getNumTorsions(); term++) {
                int first, second, third, fourth;
                vector<double> parameters;
                bonded->getTorsionParameters(term, first, second, third, fourth, parameters);
                validateMtmRigidInteraction({first, second, third, fourth}, moleculeByAtom, "CustomTorsionForce");
            }
            continue;
        }
        if (const CMAPTorsionForce* bonded = dynamic_cast<const CMAPTorsionForce*>(&force)) {
            for (int term = 0; term < bonded->getNumTorsions(); term++) {
                int map, a1, a2, a3, a4, b1, b2, b3, b4;
                bonded->getTorsionParameters(term, map, a1, a2, a3, a4, b1, b2, b3, b4);
                validateMtmRigidInteraction({a1, a2, a3, a4, b1, b2, b3, b4}, moleculeByAtom, "CMAPTorsionForce");
            }
            continue;
        }
        if (const CustomCompoundBondForce* bonded = dynamic_cast<const CustomCompoundBondForce*>(&force)) {
            for (int term = 0; term < bonded->getNumBonds(); term++) {
                vector<int> atoms;
                vector<double> parameters;
                bonded->getBondParameters(term, atoms, parameters);
                validateMtmRigidInteraction(atoms, moleculeByAtom, "CustomCompoundBondForce");
            }
            continue;
        }
        throw OpenMMException("RelocationMC MTM encountered an unsupported additional Force: "+force.getName());
    }
}

void validateMtmLabileParameters(const PerturbNonbondedForce& force,
        const vector<LabileMol>& molecules, const vector<LabileMolSiteParameters>& siteParameters) {
    vector<bool> labileAtom(force.getNumParticles(), false);
    for (const LabileMol& molecule : molecules)
        for (int atom : molecule.atomIndices) {
            if (atom < 0 || atom >= force.getNumParticles())
                throw OpenMMException("RelocationMC MTM encountered an invalid labile atom index.");
            labileAtom[atom] = true;
        }
    for (int offset = 0; offset < force.getNumParticleParameterOffsets(); offset++) {
        string parameter;
        int atom;
        double chargeScale, sigmaScale, epsilonScale;
        force.getParticleParameterOffset(offset, parameter, atom, chargeScale, sigmaScale, epsilonScale);
        if (atom >= 0 && atom < (int) labileAtom.size() && labileAtom[atom])
            throw OpenMMException("RelocationMC MTM does not support parameter offsets on labile atoms.");
    }
    for (const LabileMol& molecule : molecules)
        for (int site = 0; site < (int) siteParameters.size(); site++) {
            double chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB;
            bool isDummy, isRest2;
            force.getParticleParameters(molecule.atomIndices[site], chargeA, sigmaA, epsilonA,
                    chargeB, sigmaB, epsilonB, isDummy, isRest2);
            const LabileMolSiteParameters& expected = siteParameters[site];
            const bool matches = std::isfinite(chargeA) && std::isfinite(sigmaA) && std::isfinite(epsilonA) &&
                    std::isfinite(chargeB) && std::isfinite(sigmaB) && std::isfinite(epsilonB) &&
                    chargeA == expected.charge && chargeB == expected.charge &&
                    sigmaA == expected.sigma && sigmaB == expected.sigma &&
                    epsilonA == expected.epsilon && epsilonB == expected.epsilon;
            if (isDummy || isRest2 || !matches)
                throw OpenMMException("RelocationMC MTM requires lambda-invariant ordinary labile particles matching the registered site parameters.");
        }
}
}

CudaCalcRelocationMCKernel::CudaCalcRelocationMCKernel(string name, const Platform& platform,
        CudaContext& cu, const System&) : CalcRelocationMCKernel(name, platform), cu(cu),
        nonbonded(nullptr), reorderPositionsKernel(nullptr), computeLigandCenterKernel(nullptr), classifyMoleculesKernel(nullptr),
        gatherRealPositionsKernel(nullptr), sampleInnerAnchorsKernel(nullptr), sampleExternalAnchorsKernel(nullptr),
        generateCandidatePositionsKernel(nullptr), sumRealMoleculeEnergiesKernel(nullptr),
        selectRealMoleculeKernel(nullptr), sumCandidateMoleculeEnergiesKernel(nullptr),
        selectCandidateMoleculeKernel(nullptr), gatherSelectedCandidatePositionsKernel(nullptr),
        selectMtmRealMoleculeKernel(nullptr), generateMtmCandidatePositionsKernel(nullptr),
        assembleMtmCandidateEnergiesKernel(nullptr), selectMtmCandidateKernel(nullptr),
        gatherMtmSelectedCandidateKernel(nullptr), gatherMtmSelectedComponentKernel(nullptr),
        packDeletedPmeQueriesKernel(nullptr),
        packSelectedPmeQueriesKernel(nullptr), computeMtmDirectPairCorrectionsKernel(nullptr),
        computeMtmFinalSumsKernel(nullptr), reorderMtmPositionsContextPrecisionKernel(nullptr),
        computeMtmLigandCenterContextPrecisionKernel(nullptr), classifyMtmMoleculesContextPrecisionKernel(nullptr),
        gatherMtmRealPositionsContextPrecisionKernel(nullptr), gatherMtmSelectedMoleculeContextPrecisionKernel(nullptr),
        generateMtmCandidatePositionsContextPrecisionKernel(nullptr), gatherMtmSelectedCandidateContextPrecisionKernel(nullptr),
        packMtmDeletedQueriesContextPrecisionKernel(nullptr), packMtmSelectedQueriesContextPrecisionKernel(nullptr),
        computeMtmDirectPairCorrectionsContextPrecisionKernel(nullptr),
        commitMtmSelectedPositionsContextPrecisionKernel(nullptr), temperature(0.0), sphereRadius(0.0),
        numCandidateSites(0), numCandidateMolecules(0), numAtomsPerMolecule(0), protocolVersion(1),
        orientationsEnabled(false), useCavityBias(false), useLegacyAsymmetricEnergies(false), verbose(false),
        randomSeed(0), effectiveRandomSeed(0), attemptCounter(0), pmeAlpha(0.0), directCutoff(0.0), random(0) {
}

CudaCalcRelocationMCKernel::~CudaCalcRelocationMCKernel() {
}

void CudaCalcRelocationMCKernel::initialize(const System& system, const RelocationMC& mc, ContextImpl& context) {
    ContextSelector selector(cu);
    temperature = mc.getTemperature();
    sphereRadius = mc.getSphereRadius();
    numCandidateSites = mc.getNumCandidateSitesPerExchange();
    protocolVersion = mc.getProtocolVersion();
    orientationsEnabled = mc.getOrientationsEnabled();
    useCavityBias = mc.getUseCavityBias();
    useLegacyAsymmetricEnergies = mc.getUseLegacyAsymmetricEnergies();
    verbose = mc.getVerbose();
    randomSeed = mc.getRandomSeed();
    effectiveRandomSeed = randomSeed == 0 ? std::random_device{}() : randomSeed;
    attemptCounter = 0;
    random.seed(effectiveRandomSeed);

    ligandAtoms = mc.getLigandAtoms();
    labileMolecules = mc.getLabileMols();
    siteParameters = mc.getLabileMolSiteParameters();
    referencePositions = mc.getLabileMolReferencePositions();
    numAtomsPerMolecule = mc.getNumAtomsPerLabileMol();
    if (protocolVersion == RelocationMC::MTM) {
        if (!orientationsEnabled)
            throw OpenMMException("RelocationMC MTM requires Haar orientations; orientationsEnabled must be true.");
        if (useCavityBias)
            throw OpenMMException("RelocationMC MTM uses a whole-box proposal and does not permit an explicit cavity-bias volume factor.");
        if (useLegacyAsymmetricEnergies)
            throw OpenMMException("RelocationMC MTM requires environment-only exact-PME energies.");
        numPreparedOrientations = 1;
        numCandidateMolecules = numCandidateSites;
        Vec3 initialBox[3];
        context.getPeriodicBoxVectors(initialBox[0], initialBox[1], initialBox[2]);
        const double initialBoxVolume = initialBox[0][0]*initialBox[1][1]*initialBox[2][2];
        const MtmInitialBoxPoolEstimate estimate = computeMtmInitialBoxPoolEstimate(
                numCandidateMolecules, sphereRadius, initialBoxVolume);
        const string warning = formatMtmInitialBoxPoolWarning(estimate);
        if (!warning.empty())
            std::cerr << warning;
    }
    else {
        numPreparedOrientations = orientationsEnabled ? preparedOrientations.size() : 1;
        const size_t legacyCandidates = static_cast<size_t>(numCandidateSites)*numPreparedOrientations;
        if (legacyCandidates > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw OpenMMException("RelocationMC candidate count exceeds the supported integer range.");
        numCandidateMolecules = static_cast<int>(legacyCandidates);
    }
    const size_t candidateAtomCount = static_cast<size_t>(numCandidateMolecules)*numAtomsPerMolecule;
    const size_t realAtomCount = labileMolecules.size()*static_cast<size_t>(numAtomsPerMolecule);
    const size_t maximumExactFloatIndex = 1ULL << 24;
    if (labileMolecules.size() > maximumExactFloatIndex ||
            static_cast<size_t>(numCandidateMolecules) > maximumExactFloatIndex ||
            candidateAtomCount > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            realAtomCount > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            candidateAtomCount+realAtomCount+numAtomsPerMolecule >
                    static_cast<size_t>(std::numeric_limits<int>::max()))
        throw OpenMMException("RelocationMC molecule, candidate, or packed query count exceeds its supported range.");

    for (ForceImpl* force : context.getForceImpls()) {
        PerturbNonbondedForceImpl* candidate = dynamic_cast<PerturbNonbondedForceImpl*>(force);
        if (candidate != nullptr) {
            if (nonbonded != nullptr)
                throw OpenMMException("RelocationMC requires exactly one PerturbNonbondedForce.");
            nonbonded = candidate;
        }
    }
    if (nonbonded == nullptr)
        throw OpenMMException("RelocationMC requires a PerturbNonbondedForce in the System.");
    const PerturbNonbondedForce& nonbondedOwner = nonbonded->getOwner();
    if (nonbondedOwner.getNonbondedMethod() != PerturbNonbondedForce::PME)
        throw OpenMMException("RelocationMC candidate queries require PerturbNonbondedForce::PME.");
    int pmeGridX, pmeGridY, pmeGridZ;
    nonbonded->getPMEParameters(pmeAlpha, pmeGridX, pmeGridY, pmeGridZ);
    directCutoff = nonbondedOwner.getCutoffDistance();
    if (protocolVersion == RelocationMC::MTM) {
        validateMtmHamiltonian(system, labileMolecules);
        if (!nonbondedOwner.getIncludeDirectSpace())
            throw OpenMMException("RelocationMC MTM requires PerturbNonbondedForce direct-space interactions.");
        validateMtmLabileParameters(nonbondedOwner, labileMolecules, siteParameters);
        vector<bool> labileAtom(nonbondedOwner.getNumParticles(), false);
        for (const LabileMol& molecule : labileMolecules)
            for (int atom : molecule.atomIndices)
                labileAtom[atom] = true;
        for (int atom : ligandAtoms)
            if (atom >= 0 && atom < (int) labileAtom.size() && labileAtom[atom])
                throw OpenMMException("RelocationMC MTM requires ligand-center atoms to be disjoint from labile molecules.");
    }
    // Exact rigid relocation does not support intramolecular nonbonded terms.
    // Every pair inside a labile molecule must be represented by a zeroed
    // exception, and nonzero intramolecular 1-4 replacements are rejected.
    if (!useLegacyAsymmetricEnergies) {
        std::map<std::pair<int, int>, std::pair<double, double> > exceptionParameters;
        for (int exception = 0; exception < nonbondedOwner.getNumExceptions(); exception++) {
            int atom1, atom2;
            double chargeProduct, sigma, epsilon;
            nonbondedOwner.getExceptionParameters(exception, atom1, atom2, chargeProduct, sigma, epsilon);
            if (atom2 < atom1)
                std::swap(atom1, atom2);
            exceptionParameters[std::make_pair(atom1, atom2)] = std::make_pair(chargeProduct, epsilon);
        }
        std::map<int, int> labileMoleculeByAtom;
        for (int molecule = 0; molecule < (int)labileMolecules.size(); molecule++)
            for (int atom : labileMolecules[molecule].atomIndices)
                labileMoleculeByAtom[atom] = molecule;
        for (const LabileMol& molecule : labileMolecules)
            for (int first = 0; first < (int)molecule.atomIndices.size(); first++)
                for (int second = first+1; second < (int)molecule.atomIndices.size(); second++) {
                    int atom1 = molecule.atomIndices[first];
                    int atom2 = molecule.atomIndices[second];
                    if (atom2 < atom1)
                        std::swap(atom1, atom2);
                    const auto parameters = exceptionParameters.find(std::make_pair(atom1, atom2));
                    if (parameters == exceptionParameters.end())
                        throw OpenMMException("RelocationMC requires every intramolecular labile pair to have a zeroed PerturbNonbondedForce exception.");
                    if (parameters->second.first != 0.0 || parameters->second.second != 0.0)
                        throw OpenMMException("RelocationMC does not support nonzero intramolecular exceptions in labile molecules.");
                }
        for (const auto& parameters : exceptionParameters) {
            const auto first = labileMoleculeByAtom.find(parameters.first.first);
            const auto second = labileMoleculeByAtom.find(parameters.first.second);
            const bool firstLabile = first != labileMoleculeByAtom.end();
            const bool secondLabile = second != labileMoleculeByAtom.end();
            if ((firstLabile || secondLabile) && !(firstLabile && secondLabile && first->second == second->second))
                throw OpenMMException("RelocationMC exact PME mode does not support cross-molecule or labile-environment nonbonded exceptions.");
        }
    }

    realAtomIndices.reserve(labileMolecules.size()*numAtomsPerMolecule);
    excludedStarts.reserve(realAtomIndices.capacity());
    excludedCounts.reserve(realAtomIndices.capacity());
    for (const LabileMol& molecule : labileMolecules) {
        for (int atom : molecule.atomIndices) {
            realAtomIndices.push_back(atom);
            excludedStarts.push_back((int) excludedAtomIndices.size());
            excludedCounts.push_back((int) molecule.atomIndices.size());
            excludedAtomIndices.insert(excludedAtomIndices.end(), molecule.atomIndices.begin(), molecule.atomIndices.end());
        }
    }

    const int numRealAtoms = (int) realAtomIndices.size();
    const int numMolecules = (int) labileMolecules.size();
    const int numCandidateSitesLocal = numCandidateSites;
    const int numCandidates = numCandidateMolecules;
    const int numCandidateAtoms = numCandidates*numAtomsPerMolecule;
    vector<int> referenceAtoms(numMolecules);
    for (int molecule = 0; molecule < numMolecules; molecule++)
        referenceAtoms[molecule] = labileMolecules[molecule].referenceAtom;
    ligandAtomIndices.initialize<int>(cu, (int) ligandAtoms.size(), "relocationLigandAtoms");
    referenceAtomIndices.initialize<int>(cu, numMolecules, "relocationReferenceAtoms");
    realAtomIndicesDevice.initialize<int>(cu, numRealAtoms, "relocationRealAtomIndices");
    orderedPositions.initialize<mm_float4>(cu, cu.getPaddedNumAtoms(), "relocationOrderedPositions");
    ligandCenter.initialize<mm_float4>(cu, 1, "relocationLigandCenter");
    moleculeRegionFlags.initialize<int>(cu, numMolecules, "relocationMoleculeRegionFlags");
    regionCounts.initialize<int>(cu, 3, "relocationRegionCounts");
    proposalScalars.initialize<mm_float4>(cu, 1, "relocationProposalScalars");
    selectedRealData.initialize<mm_float4>(cu, 1, "relocationSelectedRealData");
    selectedCandidateData.initialize<mm_float4>(cu, 1, "relocationSelectedCandidateData");
    selectedDeletedPositions.initialize<mm_float4>(cu, numAtomsPerMolecule, "relocationSelectedDeletedPositions");
    selectedCandidatePositions.initialize<mm_float4>(cu, numAtomsPerMolecule, "relocationSelectedCandidatePositions");
    mtmLogSums.initialize<mm_float4>(cu, 1, "relocationMtmLogSums");
    mtmEnergyScalars.initialize<mm_float4>(cu, 1, "relocationMtmEnergyScalars");
    ligandAtomIndices.upload(ligandAtoms);
    referenceAtomIndices.upload(referenceAtoms);
    realAtomIndicesDevice.upload(realAtomIndices);

    realPositions.initialize<mm_float4>(cu, numRealAtoms, "relocationRealPositions");
    realParameters.initialize<mm_float4>(cu, numRealAtoms, "relocationRealParameters");
    realExcludedStarts.initialize<int>(cu, numRealAtoms, "relocationRealExcludedStarts");
    realExcludedCounts.initialize<int>(cu, numRealAtoms, "relocationRealExcludedCounts");
    realExcludedAtoms.initialize<int>(cu, (int) excludedAtomIndices.size(), "relocationRealExcludedAtoms");
    realForces.initialize<mm_float4>(cu, numRealAtoms, "relocationRealForces");
    realAtomEnergies.initialize<float>(cu, numRealAtoms, "relocationRealAtomEnergies");
    realPmePotentials.initialize<float>(cu, numRealAtoms, "relocationRealPmePotentials");
    selectedDeletedPmePotentials.initialize<float>(cu, numAtomsPerMolecule, "relocationSelectedDeletedPmePotentials");
    realCharges.initialize<float>(cu, numRealAtoms, "relocationRealCharges");
    realPmeRemovalComponents.initialize<mm_float4>(cu, numMolecules, "relocationRealPmeRemovalComponents");
    realMoleculeEnergies.initialize<float>(cu, (int) labileMolecules.size(), "relocationRealMoleculeEnergies");
    realEnergyComponents.initialize<mm_float4>(cu, 1, "relocationRealEnergyComponents");

    candidateAnchors.initialize<mm_float4>(cu, numCandidateSitesLocal, "relocationCandidateAnchors");
    candidateRandomNumbers.initialize<mm_float4>(cu, numCandidateSitesLocal, "relocationCandidateRandomNumbers");
    candidateAcceptanceRandom.initialize<float>(cu, numCandidates, "relocationCandidateAcceptanceRandom");
    candidateValidFlags.initialize<int>(cu, numCandidateSitesLocal, "relocationCandidateValidFlags");
    candidateRegionFlags.initialize<int>(cu, numCandidates, "relocationCandidateRegionFlags");
    rotatedReferencePositions.initialize<mm_float4>(cu, numPreparedOrientations*numAtomsPerMolecule,
            "relocationRotatedReferencePositions");
    candidatePositions.initialize<mm_float4>(cu, numCandidateAtoms, "relocationCandidatePositions");
    candidateParameters.initialize<mm_float4>(cu, numCandidateAtoms, "relocationCandidateParameters");
    candidateForces.initialize<mm_float4>(cu, numCandidateAtoms, "relocationCandidateForces");
    candidateDirectEnergies.initialize<float>(cu, numCandidateAtoms, "relocationCandidateDirectEnergies");
    candidateDeletedDirectEnergies.initialize<float>(cu, numCandidateAtoms, "relocationCandidateDeletedDirectEnergies");
    candidatePmePotentials.initialize<float>(cu, numCandidateAtoms, "relocationCandidatePmePotentials");
    candidateDeletedPmePotentials.initialize<float>(cu, numCandidateAtoms, "relocationCandidateDeletedPmePotentials");
    candidateCharges.initialize<float>(cu, numCandidateAtoms, "relocationCandidateCharges");
    candidateMoleculeEnergies.initialize<float>(cu, numCandidates, "relocationCandidateMoleculeEnergies");
    candidateEnergyComponents.initialize<mm_float4>(cu, numCandidates, "relocationCandidateEnergyComponents");
    selectedCandidateEnergyComponents.initialize<mm_float4>(cu, 1, "relocationSelectedCandidateEnergyComponents");
    const int deletedQueryCount = numAtomsPerMolecule+numCandidateAtoms+numRealAtoms;
    const int selectedQueryCount = numAtomsPerMolecule+numRealAtoms;
    deletedPmeQueryPositions.initialize<mm_float4>(cu, deletedQueryCount, "relocationDeletedPmeQueryPositions");
    deletedPmeQueryPotentials.initialize<float>(cu, deletedQueryCount, "relocationDeletedPmeQueryPotentials");
    selectedPmeQueryPositions.initialize<mm_float4>(cu, selectedQueryCount, "relocationSelectedPmeQueryPositions");
    selectedPmeQueryPotentials.initialize<float>(cu, selectedQueryCount, "relocationSelectedPmeQueryPotentials");
    mtmDirectPairCorrections.initialize<float>(cu, numMolecules, "relocationMtmDirectPairCorrections");
    mtmProposedMoleculeScores.initialize<float>(cu, numMolecules, "relocationMtmProposedMoleculeScores");
    if (protocolVersion == RelocationMC::MTM) {
        if (cu.getUseDoublePrecision()) {
            mtmOrderedPositionBase.initialize<mm_double4>(cu, cu.getPaddedNumAtoms(), "relocationMtmOrderedPositionBase");
            mtmLigandCenterBase.initialize<mm_double4>(cu, 1, "relocationMtmLigandCenterBase");
            mtmRealPositionBase.initialize<mm_double4>(cu, numRealAtoms, "relocationMtmRealPositionBase");
            mtmSelectedDeletedPositionBase.initialize<mm_double4>(cu, numAtomsPerMolecule, "relocationMtmSelectedDeletedBase");
            mtmCandidateAnchorBase.initialize<mm_double4>(cu, numCandidates, "relocationMtmCandidateAnchorBase");
            mtmCandidatePositionBase.initialize<mm_double4>(cu, numCandidateAtoms, "relocationMtmCandidatePositionBase");
            mtmSelectedCandidatePositionBase.initialize<mm_double4>(cu, numAtomsPerMolecule, "relocationMtmSelectedCandidateBase");
            mtmDeletedQueryPositionBase.initialize<mm_double4>(cu, deletedQueryCount, "relocationMtmDeletedQueryBase");
            mtmSelectedQueryPositionBase.initialize<mm_double4>(cu, selectedQueryCount, "relocationMtmSelectedQueryBase");
        }
        else {
            mtmOrderedPositionBase.initialize<mm_float4>(cu, cu.getPaddedNumAtoms(), "relocationMtmOrderedPositionBase");
            mtmLigandCenterBase.initialize<mm_float4>(cu, 1, "relocationMtmLigandCenterBase");
            mtmRealPositionBase.initialize<mm_float4>(cu, numRealAtoms, "relocationMtmRealPositionBase");
            mtmSelectedDeletedPositionBase.initialize<mm_float4>(cu, numAtomsPerMolecule, "relocationMtmSelectedDeletedBase");
            mtmCandidateAnchorBase.initialize<mm_float4>(cu, numCandidates, "relocationMtmCandidateAnchorBase");
            mtmCandidatePositionBase.initialize<mm_float4>(cu, numCandidateAtoms, "relocationMtmCandidatePositionBase");
            mtmSelectedCandidatePositionBase.initialize<mm_float4>(cu, numAtomsPerMolecule, "relocationMtmSelectedCandidateBase");
            mtmDeletedQueryPositionBase.initialize<mm_float4>(cu, deletedQueryCount, "relocationMtmDeletedQueryBase");
            mtmSelectedQueryPositionBase.initialize<mm_float4>(cu, selectedQueryCount, "relocationMtmSelectedQueryBase");
        }
        if (cu.getUseMixedPrecision()) {
            mtmOrderedPositionCorrection.initialize<mm_float4>(cu, cu.getPaddedNumAtoms(), "relocationMtmOrderedPositionCorrection");
            mtmLigandCenterCorrection.initialize<mm_float4>(cu, 1, "relocationMtmLigandCenterCorrection");
            mtmRealPositionCorrection.initialize<mm_float4>(cu, numRealAtoms, "relocationMtmRealPositionCorrection");
            mtmSelectedDeletedPositionCorrection.initialize<mm_float4>(cu, numAtomsPerMolecule, "relocationMtmSelectedDeletedCorrection");
            mtmCandidateAnchorCorrection.initialize<mm_float4>(cu, numCandidates, "relocationMtmCandidateAnchorCorrection");
            mtmCandidatePositionCorrection.initialize<mm_float4>(cu, numCandidateAtoms, "relocationMtmCandidatePositionCorrection");
            mtmSelectedCandidatePositionCorrection.initialize<mm_float4>(cu, numAtomsPerMolecule, "relocationMtmSelectedCandidateCorrection");
            mtmDeletedQueryPositionCorrection.initialize<mm_float4>(cu, deletedQueryCount, "relocationMtmDeletedQueryCorrection");
            mtmSelectedQueryPositionCorrection.initialize<mm_float4>(cu, selectedQueryCount, "relocationMtmSelectedQueryCorrection");
        }
    }

    realExcludedStarts.upload(excludedStarts);
    realExcludedCounts.upload(excludedCounts);
    realExcludedAtoms.upload(excludedAtomIndices);

    vector<mm_float4> realParameterValues(numRealAtoms);
    for (int molecule = 0; molecule < (int) labileMolecules.size(); molecule++)
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            const LabileMolSiteParameters& parameter = siteParameters[site];
            realParameterValues[molecule*numAtomsPerMolecule+site] = mm_float4((float) parameter.charge,
                    (float) (0.5*parameter.sigma), (float) (2.0*std::sqrt(parameter.epsilon)), 0.0f);
        }
    realParameters.upload(realParameterValues);

    vector<mm_float4> candidateParameterValues(numCandidateAtoms);
    for (int candidate = 0; candidate < numCandidates; candidate++)
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            const LabileMolSiteParameters& parameter = siteParameters[site];
            candidateParameterValues[candidate*numAtomsPerMolecule+site] = mm_float4((float) parameter.charge,
                    (float) (0.5*parameter.sigma), (float) (2.0*std::sqrt(parameter.epsilon)), 0.0f);
        }
    candidateParameters.upload(candidateParameterValues);
    vector<float> candidateChargeValues(numCandidateAtoms);
    for (int index = 0; index < numCandidateAtoms; index++)
        candidateChargeValues[index] = candidateParameterValues[index].x;
    candidateCharges.upload(candidateChargeValues);

    vector<mm_float4> rotatedValues(numPreparedOrientations*numAtomsPerMolecule);
    for (int orientation = 0; orientation < numPreparedOrientations; orientation++)
        for (int site = 0; site < numAtomsPerMolecule; site++) {
            const Vec3 rotated = orientationsEnabled ? preparedOrientations.rotate(orientation, referencePositions[site]) :
                    referencePositions[site];
            rotatedValues[orientation*numAtomsPerMolecule+site] = toFloat4(rotated);
        }
    rotatedReferencePositions.upload(rotatedValues);

    std::map<string, string> defines;
    defines["SQRT_ONE_4PI_EPS0"] = cu.doubleToString(std::sqrt(ONE_4PI_EPS0_VALUE));
    defines["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0_VALUE);
    defines["EWALD_ALPHA"] = cu.doubleToString(pmeAlpha);
    defines["CUTOFF_SQUARED"] = cu.doubleToString(directCutoff*directCutoff);
    defines["MTM_HAS_LJ"] = nonbondedOwner.getIncludeLJ() ? "1" : "0";
    defines["MTM_BLOCK_SIZE"] = "256";
    defines["USE_PERIODIC"] = "1";
    CUmodule module = cu.createModule(CudaKernelSources::vectorOps+
            CudaRelocationMCKernelSources::relocation_mc, defines);
    reorderPositionsKernel = cu.getKernel(module, "reorderPositions");
    computeLigandCenterKernel = cu.getKernel(module, "computeLigandCenter");
    classifyMoleculesKernel = cu.getKernel(module, "classifyMolecules");
    gatherRealPositionsKernel = cu.getKernel(module, "gatherRealPositions");
    sampleInnerAnchorsKernel = cu.getKernel(module, "sampleInnerAnchors");
    sampleExternalAnchorsKernel = cu.getKernel(module, "sampleExternalAnchors");
    generateCandidatePositionsKernel = cu.getKernel(module, "generateCandidatePositions");
    sumRealMoleculeEnergiesKernel = cu.getKernel(module, "sumRealMoleculeEnergies");
    selectRealMoleculeKernel = cu.getKernel(module, "selectRealMolecule");
    sumCandidateMoleculeEnergiesKernel = cu.getKernel(module, "sumCandidateMoleculeEnergies");
    selectCandidateMoleculeKernel = cu.getKernel(module, "selectCandidateMolecule");
    gatherSelectedCandidatePositionsKernel = cu.getKernel(module, "gatherSelectedCandidatePositions");
    gatherSelectedMoleculeDataKernel = cu.getKernel(module, "gatherSelectedMoleculeData");
    extractChargesKernel = cu.getKernel(module, "extractCharges");
    assembleRealPmeRemovalEnergiesKernel = cu.getKernel(module, "assembleRealPmeRemovalEnergies");
    assembleExactPmeEnergiesKernel = cu.getKernel(module, "assembleExactPmeEnergies");
    selectMtmRealMoleculeKernel = cu.getKernel(module, "selectMtmRealMolecule");
    generateMtmCandidatePositionsKernel = cu.getKernel(module, "generateMtmCandidatePositions");
    assembleMtmCandidateEnergiesKernel = cu.getKernel(module, "assembleMtmCandidateEnergies");
    selectMtmCandidateKernel = cu.getKernel(module, "selectMtmCandidate");
    gatherMtmSelectedCandidateKernel = cu.getKernel(module, "gatherMtmSelectedCandidate");
    gatherMtmSelectedComponentKernel = cu.getKernel(module, "gatherMtmSelectedComponent");
    packDeletedPmeQueriesKernel = cu.getKernel(module, "packDeletedPmeQueries");
    packSelectedPmeQueriesKernel = cu.getKernel(module, "packSelectedPmeQueries");
    computeMtmDirectPairCorrectionsKernel = cu.getKernel(module, "computeMtmDirectPairCorrections");
    computeMtmFinalSumsKernel = cu.getKernel(module, "computeMtmFinalSums");
    reorderMtmPositionsContextPrecisionKernel = cu.getKernel(module, "reorderMtmPositionsContextPrecision");
    computeMtmLigandCenterContextPrecisionKernel = cu.getKernel(module, "computeMtmLigandCenterContextPrecision");
    classifyMtmMoleculesContextPrecisionKernel = cu.getKernel(module, "classifyMtmMoleculesContextPrecision");
    gatherMtmRealPositionsContextPrecisionKernel = cu.getKernel(module, "gatherMtmRealPositionsContextPrecision");
    gatherMtmSelectedMoleculeContextPrecisionKernel = cu.getKernel(module, "gatherMtmSelectedMoleculeContextPrecision");
    generateMtmCandidatePositionsContextPrecisionKernel = cu.getKernel(module, "generateMtmCandidatePositionsContextPrecision");
    gatherMtmSelectedCandidateContextPrecisionKernel = cu.getKernel(module, "gatherMtmSelectedCandidateContextPrecision");
    packMtmDeletedQueriesContextPrecisionKernel = cu.getKernel(module, "packMtmDeletedQueriesContextPrecision");
    packMtmSelectedQueriesContextPrecisionKernel = cu.getKernel(module, "packMtmSelectedQueriesContextPrecision");
    computeMtmDirectPairCorrectionsContextPrecisionKernel = cu.getKernel(module, "computeMtmDirectPairCorrectionsContextPrecision");
    commitMtmSelectedPositionsContextPrecisionKernel = cu.getKernel(module, "commitMtmSelectedPositionsContextPrecision");
}

double CudaCalcRelocationMCKernel::computeProposalProbability(int sourceCount, int otherCount) {
    if (sourceCount <= 0)
        return 0.0;
    return otherCount > 0 ? 0.5 : 1.0;
}

bool CudaCalcRelocationMCKernel::isCandidateDecisionValid(float selectedCandidate,
        float candidateEnergy, float probability, float acceptedFlag, int numCandidates) {
    if (!std::isfinite(selectedCandidate) || !std::isfinite(candidateEnergy) ||
            !std::isfinite(probability) || !std::isfinite(acceptedFlag) ||
            selectedCandidate != std::trunc(selectedCandidate) ||
            probability < 0.0f || probability > 1.0f ||
            (acceptedFlag != 0.0f && acceptedFlag != 1.0f))
        return false;
    if (acceptedFlag == 1.0f)
        return selectedCandidate >= 0.0f && selectedCandidate < numCandidates;
    return selectedCandidate == -1.0f;
}

ExchangeResult CudaCalcRelocationMCKernel::attemptExchange(ContextImpl& context,
        stringstream& logStream) {
    ContextSelector selector(cu);
    ExchangeResult result;
    result.protocolVersion = protocolVersion;
    if (captureMtmDiagnosticsForTesting)
        lastMtmProposedScoresForTesting.clear();
    const bool useMtm = protocolVersion == RelocationMC::MTM;
    const uint64_t attemptIndex = useMtm ? attemptCounter++ : 0;
    Vec3 box[3];
    context.getPeriodicBoxVectors(box[0], box[1], box[2]);

    const int numAtoms = cu.getNumAtoms();
    const int numRealAtoms = (int) realAtomIndices.size();
    const int numMolecules = (int) labileMolecules.size();
    const int numLigands = (int) ligandAtoms.size();
    const int numCandidateAtoms = numCandidateMolecules*numAtomsPerMolecule;
    if (useMtm) {
        const double shortestBox = std::min(box[0][0], std::min(box[1][1], box[2][2]));
        const double boxVolumeCheck = box[0][0]*box[1][1]*box[2][2];
        const double innerVolumeCheck = 4.0*PI*sphereRadius*sphereRadius*sphereRadius/3.0;
        if (box[0][1] != 0.0 || box[0][2] != 0.0 || box[1][0] != 0.0 ||
                box[1][2] != 0.0 || box[2][0] != 0.0 || box[2][1] != 0.0)
            throw OpenMMException("RelocationMC MTM supports orthorhombic periodic boxes only.");
        if (!std::isfinite(shortestBox) || !std::isfinite(boxVolumeCheck) || !std::isfinite(innerVolumeCheck) ||
                shortestBox <= 0.0 || sphereRadius > 0.5*shortestBox ||
                !(innerVolumeCheck > 0.0) || !(innerVolumeCheck < boxVolumeCheck))
            throw OpenMMException("RelocationMC MTM requires a non-self-overlapping sphere with positive inner and outer box volumes.");
        if (!nonbonded->getOwner().getIncludeDirectSpace())
            throw OpenMMException("RelocationMC MTM requires PerturbNonbondedForce direct-space interactions.");
        validateMtmLabileParameters(nonbonded->getOwner(), labileMolecules, siteParameters);
    }
    // CUDA driver launch arguments must exactly match the scalar ABI declared
    // in relocation_mc.cu.  Legacy uses float; MTM uses mixed (double for
    // mixed/double precision and float for single precision).
    float sphereRadiusFloat = static_cast<float>(sphereRadius);
    float radiusSquared = sphereRadiusFloat*sphereRadiusFloat;
    double radiusSquaredDouble = sphereRadius*sphereRadius;
    void* mtmRadiusSquaredPointer = (cu.getUseMixedPrecision() || cu.getUseDoublePrecision()) ?
            static_cast<void*>(&radiusSquaredDouble) : static_cast<void*>(&radiusSquared);
    const float beta = (float) (1.0/(BOLTZMANN*temperature));
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    CUdeviceptr nullCorrection = 0;
    const bool useMtmCorrection = useMtm && cu.getUseMixedPrecision();
    CUdeviceptr contextCorrection = useMtmCorrection ? cu.getPosqCorrection().getDevicePointer() : nullCorrection;
    CUdeviceptr orderedCorrection = useMtmCorrection ? mtmOrderedPositionCorrection.getDevicePointer() : nullCorrection;
    CUdeviceptr centerCorrection = useMtmCorrection ? mtmLigandCenterCorrection.getDevicePointer() : nullCorrection;
    CUdeviceptr realCorrection = useMtmCorrection ? mtmRealPositionCorrection.getDevicePointer() : nullCorrection;
    CUdeviceptr deletedCorrection = useMtmCorrection ? mtmSelectedDeletedPositionCorrection.getDevicePointer() : nullCorrection;
    CUdeviceptr candidateCorrection = useMtmCorrection ? mtmCandidatePositionCorrection.getDevicePointer() : nullCorrection;
    CUdeviceptr candidateAnchorCorrection = useMtmCorrection ? mtmCandidateAnchorCorrection.getDevicePointer() : nullCorrection;
    CUdeviceptr selectedCorrection = useMtmCorrection ? mtmSelectedCandidatePositionCorrection.getDevicePointer() : nullCorrection;
    CUdeviceptr deletedQueryCorrection = useMtmCorrection ? mtmDeletedQueryPositionCorrection.getDevicePointer() : nullCorrection;
    CUdeviceptr selectedQueryCorrection = useMtmCorrection ? mtmSelectedQueryPositionCorrection.getDevicePointer() : nullCorrection;

    cu.clearBuffer(regionCounts);
    if (useMtm) {
        void* reorderArgs[] = {const_cast<int*>(&numAtoms), &cu.getPosq().getDevicePointer(),
                &contextCorrection, &cu.getAtomIndexArray().getDevicePointer(),
                &mtmOrderedPositionBase.getDevicePointer(), &orderedCorrection};
        cu.executeKernel(reorderMtmPositionsContextPrecisionKernel, reorderArgs, numAtoms);
        void* centerArgs[] = {const_cast<int*>(&numLigands), &mtmOrderedPositionBase.getDevicePointer(),
                &orderedCorrection, &ligandAtomIndices.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
                cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                &mtmLigandCenterBase.getDevicePointer(), &centerCorrection};
        cu.executeKernel(computeMtmLigandCenterContextPrecisionKernel, centerArgs, 1, 1);
        void* classifyArgs[] = {const_cast<int*>(&numMolecules), mtmRadiusSquaredPointer,
                &mtmOrderedPositionBase.getDevicePointer(), &orderedCorrection,
                &mtmLigandCenterBase.getDevicePointer(), &centerCorrection,
                &referenceAtomIndices.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
                cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                &moleculeRegionFlags.getDevicePointer(), &regionCounts.getDevicePointer()};
        cu.executeKernel(classifyMtmMoleculesContextPrecisionKernel, classifyArgs, numMolecules);
    }
    else {
        void* reorderArgs[] = {const_cast<int*>(&numAtoms), &cu.getPosq().getDevicePointer(),
                &cu.getAtomIndexArray().getDevicePointer(), &orderedPositions.getDevicePointer()};
        cu.executeKernel(reorderPositionsKernel, reorderArgs, numAtoms);
        void* centerArgs[] = {const_cast<int*>(&numLigands), &orderedPositions.getDevicePointer(),
                &ligandAtomIndices.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
                cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(), &ligandCenter.getDevicePointer()};
        cu.executeKernel(computeLigandCenterKernel, centerArgs, 1, 1);
        void* classifyArgs[] = {const_cast<int*>(&numMolecules), const_cast<float*>(&radiusSquared),
                &orderedPositions.getDevicePointer(), &ligandCenter.getDevicePointer(),
                &referenceAtomIndices.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
                cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                &moleculeRegionFlags.getDevicePointer(), &regionCounts.getDevicePointer()};
        cu.executeKernel(classifyMoleculesKernel, classifyArgs, numMolecules);
    }
    vector<int> counts;
    regionCounts.download(counts);
    const int numInner = counts[0];
    const int numExternal = counts[1];
    if (numInner+numExternal == 0)
        return result;

    bool exchangeToInner;
    if (numExternal == 0)
        exchangeToInner = false;
    else if (numInner == 0)
        exchangeToInner = true;
    else
        exchangeToInner = useMtm ? counterUniform(effectiveRandomSeed, attemptIndex, 0) < 0.5 :
                uniform(random) < 0.5f;
    const int sourceRegion = exchangeToInner ? 1 : 0;
    const int targetRegion = exchangeToInner ? 0 : 1;
    const int sourceCount = exchangeToInner ? numExternal : numInner;
    const int targetCount = exchangeToInner ? numInner : numExternal;
    if (sourceCount == 0)
        return result;
    result.attempted = true;
    result.exchangeToInner = exchangeToInner;

    if (useMtm) {
        void* gatherRealArgs[] = {const_cast<int*>(&numRealAtoms), &realAtomIndicesDevice.getDevicePointer(),
                &mtmOrderedPositionBase.getDevicePointer(), &orderedCorrection,
                &mtmRealPositionBase.getDevicePointer(), &realCorrection};
        cu.executeKernel(gatherMtmRealPositionsContextPrecisionKernel, gatherRealArgs, numRealAtoms);
    }
    else {
        void* gatherRealArgs[] = {const_cast<int*>(&numRealAtoms), &realAtomIndicesDevice.getDevicePointer(),
                &orderedPositions.getDevicePointer(), &realPositions.getDevicePointer()};
        cu.executeKernel(gatherRealPositionsKernel, gatherRealArgs, numRealAtoms);
    }
    cu.clearBuffer(realForces);
    cu.clearBuffer(realAtomEnergies);
    if (useLegacyAsymmetricEnergies) {
        nonbonded->getInteractionEnergyForRealParticlesCuda(context, numRealAtoms,
                (CudaArrayDevicePointer) realPositions.getDevicePointer(),
                (CudaArrayDevicePointer) realParameters.getDevicePointer(),
                (CudaArrayDevicePointer) realExcludedStarts.getDevicePointer(),
                (CudaArrayDevicePointer) realExcludedCounts.getDevicePointer(),
                (CudaArrayDevicePointer) realExcludedAtoms.getDevicePointer(),
                (CudaArrayDevicePointer) realForces.getDevicePointer(),
                (CudaArrayDevicePointer) realAtomEnergies.getDevicePointer());
        void* realSumArgs[] = {const_cast<int*>(&numMolecules), &numAtomsPerMolecule,
                &realAtomEnergies.getDevicePointer(), &realMoleculeEnergies.getDevicePointer()};
        cu.executeKernel(sumRealMoleculeEnergiesKernel, realSumArgs, numMolecules);
    }
    else {
        nonbonded->getDirectEnergyForRealParticlesPmeCuda(context, numRealAtoms,
                (CudaArrayDevicePointer) realAtomIndicesDevice.getDevicePointer(), numAtomsPerMolecule,
                (CudaArrayDevicePointer) realAtomEnergies.getDevicePointer());
        if (useMtm && cu.getUseDoublePrecision())
            nonbonded->getPMEPotentialAtPositionsCudaContextPrecision(context, numRealAtoms,
                    (CudaArrayDevicePointer) mtmRealPositionBase.getDevicePointer(),
                    (CudaArrayDevicePointer) nullCorrection,
                    (CudaArrayDevicePointer) realPmePotentials.getDevicePointer());
        else
            nonbonded->getPMEPotentialAtPositionsCuda(context, numRealAtoms,
                    (CudaArrayDevicePointer) (useMtm ? mtmRealPositionBase.getDevicePointer() :
                            realPositions.getDevicePointer()),
                    (CudaArrayDevicePointer) realPmePotentials.getDevicePointer());
        void* chargeArgs[] = {const_cast<int*>(&numRealAtoms), &realParameters.getDevicePointer(),
                &realCharges.getDevicePointer()};
        cu.executeKernel(extractChargesKernel, chargeArgs, numRealAtoms);
        void* assemblyArgs[] = {const_cast<int*>(&numMolecules), &numAtomsPerMolecule,
                &realParameters.getDevicePointer(), &realAtomEnergies.getDevicePointer(),
                &realPmePotentials.getDevicePointer(), &realMoleculeEnergies.getDevicePointer(),
                &realPmeRemovalComponents.getDevicePointer()};
        cu.executeKernel(assembleRealPmeRemovalEnergiesKernel, assemblyArgs, numMolecules);
    }

    if (useMtm) {
        const unsigned long long mtmSeed = effectiveRandomSeed;
        const unsigned long long mtmAttempt = attemptIndex;
        void* selectRealArgs[] = {const_cast<int*>(&numMolecules), const_cast<int*>(&sourceRegion),
                const_cast<float*>(&beta), const_cast<unsigned long long*>(&mtmSeed),
                const_cast<unsigned long long*>(&mtmAttempt), &moleculeRegionFlags.getDevicePointer(),
                &realMoleculeEnergies.getDevicePointer(), &selectedRealData.getDevicePointer(),
                &proposalScalars.getDevicePointer()};
        cu.executeKernel(selectMtmRealMoleculeKernel, selectRealArgs, 256, 256);
    }
    else {
        const float realSelectionUniform = uniform(random);
        void* selectRealArgs[] = {const_cast<int*>(&numMolecules), const_cast<int*>(&sourceRegion),
                const_cast<int*>(&targetRegion), const_cast<float*>(&beta),
                const_cast<float*>(&realSelectionUniform), &moleculeRegionFlags.getDevicePointer(),
                &realMoleculeEnergies.getDevicePointer(), &selectedRealData.getDevicePointer(),
                &proposalScalars.getDevicePointer()};
        cu.executeKernel(selectRealMoleculeKernel, selectRealArgs, 1, 1);
    }
    vector<mm_float4> selectedReal, scalarValues;
    selectedRealData.download(selectedReal);
    proposalScalars.download(scalarValues);
    if (!isFiniteSourceDecision(selectedReal[0], scalarValues[0], numMolecules)) {
        if (verbose)
            logStream << "rejectedInvalidSourceDecision=1"
                      << " selectedMolecule=" << selectedReal[0].x
                      << " deletedEnergy=" << selectedReal[0].y
                      << " deletionProbability=" << selectedReal[0].z
                      << " sourceLogSum=" << scalarValues[0].x
                      << " targetLogSum=" << scalarValues[0].y << '\n';
        return result;
    }
    const int selectedMolecule = (int) selectedReal[0].x;
    result.movedLabileMol = selectedMolecule;
    result.deletedMoleculeEnergy = selectedReal[0].y;
    result.deletedSourceScore = selectedReal[0].y;
    result.deletionProbability = selectedReal[0].z;
    result.sourceLogSum = scalarValues[0].x;
    result.movedAtomIndices = labileMolecules[selectedMolecule].atomIndices;

    // Compact the selected deleted molecule entirely on device.  The same
    // positions/charges drive its one-source scratch PME map below.
    CUdeviceptr selectedRealChargePointer = realCharges.getDevicePointer()+
            (CUdeviceptr)selectedMolecule*numAtomsPerMolecule*sizeof(float);
    if (useMtm) {
        void* selectedMoleculeArgs[] = {&numAtomsPerMolecule, const_cast<int*>(&selectedMolecule),
                &mtmRealPositionBase.getDevicePointer(), &realCorrection,
                &realParameters.getDevicePointer(), &mtmSelectedDeletedPositionBase.getDevicePointer(),
                &deletedCorrection, &selectedRealChargePointer};
        cu.executeKernel(gatherMtmSelectedMoleculeContextPrecisionKernel,
                selectedMoleculeArgs, numAtomsPerMolecule);
    }
    else {
        void* selectedMoleculeArgs[] = {&numAtomsPerMolecule, const_cast<int*>(&selectedMolecule),
                &realPositions.getDevicePointer(), &realParameters.getDevicePointer(),
                &selectedDeletedPositions.getDevicePointer(), &selectedRealChargePointer};
        cu.executeKernel(gatherSelectedMoleculeDataKernel, selectedMoleculeArgs, numAtomsPerMolecule);
    }

    if (useMtm) {
        const LabileMol& movedMolecule = labileMolecules[selectedMolecule];
        const auto reference = std::find(movedMolecule.atomIndices.begin(), movedMolecule.atomIndices.end(),
                movedMolecule.referenceAtom);
        if (reference == movedMolecule.atomIndices.end())
            throw OpenMMException("RelocationMC selected a molecule whose reference atom is not one of its sites.");
        const int referenceSite = static_cast<int>(reference-movedMolecule.atomIndices.begin());
        const unsigned long long mtmSeed = effectiveRandomSeed;
        const unsigned long long mtmAttempt = attemptIndex;
        void* generateArgs[] = {&numCandidateMolecules, &numAtomsPerMolecule,
                const_cast<int*>(&referenceSite), mtmRadiusSquaredPointer,
                const_cast<unsigned long long*>(&mtmSeed), const_cast<unsigned long long*>(&mtmAttempt),
                &mtmLigandCenterBase.getDevicePointer(), &centerCorrection,
                &mtmSelectedDeletedPositionBase.getDevicePointer(), &deletedCorrection,
                cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
                cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                &mtmCandidateAnchorBase.getDevicePointer(), &candidateAnchorCorrection,
                &candidateRegionFlags.getDevicePointer(), &mtmCandidatePositionBase.getDevicePointer(),
                &candidateCorrection};
        cu.executeKernel(generateMtmCandidatePositionsContextPrecisionKernel,
                generateArgs, numCandidateMolecules);
    }
    else {
        vector<mm_float4> randomValues(numCandidateSites);
        auto fillAnchorRandoms = [&]() {
            for (int candidate = 0; candidate < numCandidateSites; candidate++) {
                if (exchangeToInner)
                    randomValues[candidate] = mm_float4(normal(random), normal(random), normal(random), uniform(random));
                else
                    randomValues[candidate] = mm_float4(uniform(random), uniform(random), uniform(random), 0.0f);
            }
            candidateRandomNumbers.upload(randomValues);
        };
        fillAnchorRandoms();
        if (exchangeToInner) {
            void* sampleArgs[] = {&numCandidateSites, &sphereRadiusFloat,
                    &ligandCenter.getDevicePointer(), &candidateRandomNumbers.getDevicePointer(),
                    cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
                    cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    &candidateAnchors.getDevicePointer()};
            cu.executeKernel(sampleInnerAnchorsKernel, sampleArgs, numCandidateSites);
        }
        else {
            cu.clearBuffer(candidateValidFlags);
            counts[2] = 0;
            regionCounts.upload(counts);
            const int maxSamplingRounds = 10000;
            for (int round = 0; round < maxSamplingRounds && counts[2] < numCandidateSites; round++) {
                if (round > 0)
                    fillAnchorRandoms();
                void* sampleArgs[] = {&numCandidateSites, const_cast<float*>(&radiusSquared),
                        &ligandCenter.getDevicePointer(), &candidateRandomNumbers.getDevicePointer(),
                        cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
                        cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                        &candidateAnchors.getDevicePointer(), &candidateValidFlags.getDevicePointer(),
                        &regionCounts.getDevicePointer()};
                cu.executeKernel(sampleExternalAnchorsKernel, sampleArgs, numCandidateSites);
                regionCounts.download(counts);
            }
            if (counts[2] != numCandidateSites)
                throw OpenMMException("RelocationMC could not fill all external candidate anchors within 10000 rounds.");
        }

        void* generateArgs[] = {&numCandidateSites, &numPreparedOrientations, &numAtomsPerMolecule,
                &candidateAnchors.getDevicePointer(), &rotatedReferencePositions.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
                cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                &candidatePositions.getDevicePointer()};
        cu.executeKernel(generateCandidatePositionsKernel, generateArgs, numCandidateAtoms);
    }
    cu.clearBuffer(candidateForces);
    cu.clearBuffer(candidateDirectEnergies);
    cu.clearBuffer(candidatePmePotentials);
    if (useMtm) {
        if (cu.getUseDoublePrecision()) {
            nonbonded->getDirectEnergyForCandidateParticlesCudaContextPrecision(context, numCandidateAtoms,
                    (CudaArrayDevicePointer) mtmCandidatePositionBase.getDevicePointer(),
                    (CudaArrayDevicePointer) nullCorrection,
                    (CudaArrayDevicePointer) candidateParameters.getDevicePointer(),
                    (CudaArrayDevicePointer) candidateForces.getDevicePointer(),
                    (CudaArrayDevicePointer) candidateDirectEnergies.getDevicePointer());
            nonbonded->getPMEPotentialAtPositionsCudaContextPrecision(context, numCandidateAtoms,
                    (CudaArrayDevicePointer) mtmCandidatePositionBase.getDevicePointer(),
                    (CudaArrayDevicePointer) nullCorrection,
                    (CudaArrayDevicePointer) candidatePmePotentials.getDevicePointer());
        }
        else {
            nonbonded->getDirectEnergyForCandidateParticlesCuda(context, numCandidateAtoms,
                    (CudaArrayDevicePointer) mtmCandidatePositionBase.getDevicePointer(),
                    (CudaArrayDevicePointer) candidateParameters.getDevicePointer(),
                    (CudaArrayDevicePointer) candidateForces.getDevicePointer(),
                    (CudaArrayDevicePointer) candidateDirectEnergies.getDevicePointer());
            nonbonded->getPMEPotentialAtPositionsCuda(context, numCandidateAtoms,
                    (CudaArrayDevicePointer) mtmCandidatePositionBase.getDevicePointer(),
                    (CudaArrayDevicePointer) candidatePmePotentials.getDevicePointer());
        }
    }
    else {
        nonbonded->getDirectEnergyForCandidateParticlesCuda(context, numCandidateAtoms,
                (CudaArrayDevicePointer) candidatePositions.getDevicePointer(),
                (CudaArrayDevicePointer) candidateParameters.getDevicePointer(),
                (CudaArrayDevicePointer) candidateForces.getDevicePointer(),
                (CudaArrayDevicePointer) candidateDirectEnergies.getDevicePointer());
        nonbonded->getPMEPotentialAtPositionsCuda(context, numCandidateAtoms,
                (CudaArrayDevicePointer) candidatePositions.getDevicePointer(),
                (CudaArrayDevicePointer) candidatePmePotentials.getDevicePointer());
    }

    if (useLegacyAsymmetricEnergies) {
        void* candidateSumArgs[] = {&numCandidateMolecules, &numAtomsPerMolecule,
                &candidateParameters.getDevicePointer(), &candidateDirectEnergies.getDevicePointer(),
                &candidatePmePotentials.getDevicePointer(), &candidateMoleculeEnergies.getDevicePointer()};
        cu.executeKernel(sumCandidateMoleculeEnergiesKernel, candidateSumArgs, numCandidateMolecules);
    }
    else {
        // Candidate full direct energies already scan the complete Context posq.
        // Subtract only D so every u value is the interaction with B=x\\D.
        if (useMtm && cu.getUseDoublePrecision())
            nonbonded->getDirectEnergyForCandidateSubsetCudaContextPrecision(context, numCandidateAtoms,
                    (CudaArrayDevicePointer) mtmCandidatePositionBase.getDevicePointer(),
                    (CudaArrayDevicePointer) nullCorrection,
                    (CudaArrayDevicePointer) candidateParameters.getDevicePointer(), numAtomsPerMolecule,
                    (CudaArrayDevicePointer) (realAtomIndicesDevice.getDevicePointer()+
                            (CUdeviceptr)selectedMolecule*numAtomsPerMolecule*sizeof(int)),
                    (CudaArrayDevicePointer) candidateDeletedDirectEnergies.getDevicePointer());
        else
            nonbonded->getDirectEnergyForCandidateSubsetCuda(context, numCandidateAtoms,
                    (CudaArrayDevicePointer) (useMtm ? mtmCandidatePositionBase.getDevicePointer() :
                            candidatePositions.getDevicePointer()),
                    (CudaArrayDevicePointer) candidateParameters.getDevicePointer(), numAtomsPerMolecule,
                    (CudaArrayDevicePointer) (realAtomIndicesDevice.getDevicePointer()+
                            (CUdeviceptr)selectedMolecule*numAtomsPerMolecule*sizeof(int)),
                    (CudaArrayDevicePointer) candidateDeletedDirectEnergies.getDevicePointer());
        CUdeviceptr selectedRemovalComponentPointer = realPmeRemovalComponents.getDevicePointer()+
                (CUdeviceptr)selectedMolecule*sizeof(mm_float4);
        CUdeviceptr selectedParameterPointer = realParameters.getDevicePointer()+
                (CUdeviceptr)selectedMolecule*numAtomsPerMolecule*sizeof(mm_float4);
        CUdeviceptr selectedDirectPointer = realAtomEnergies.getDevicePointer()+
                (CUdeviceptr)selectedMolecule*numAtomsPerMolecule*sizeof(float);
        CUdeviceptr selectedFullPotentialPointer = realPmePotentials.getDevicePointer()+
                (CUdeviceptr)selectedMolecule*numAtomsPerMolecule*sizeof(float);
        if (useMtm && cu.getUseDoublePrecision())
            nonbonded->getPMEPotentialAtPositionsCudaContextPrecision(context, numAtomsPerMolecule,
                    (CudaArrayDevicePointer) mtmSelectedDeletedPositionBase.getDevicePointer(),
                    (CudaArrayDevicePointer) nullCorrection,
                    (CudaArrayDevicePointer) selectedFullPotentialPointer);
        else
            nonbonded->getPMEPotentialAtPositionsCuda(context, numAtomsPerMolecule,
                    (CudaArrayDevicePointer) (useMtm ? mtmSelectedDeletedPositionBase.getDevicePointer() :
                            selectedDeletedPositions.getDevicePointer()),
                    (CudaArrayDevicePointer) selectedFullPotentialPointer);
        if (useMtm) {
            const int deletedQueryCount = numAtomsPerMolecule+numCandidateAtoms+numRealAtoms;
            void* packArgs[] = {&numAtomsPerMolecule, const_cast<int*>(&numCandidateAtoms),
                    const_cast<int*>(&numRealAtoms), &mtmSelectedDeletedPositionBase.getDevicePointer(),
                    &deletedCorrection, &mtmCandidatePositionBase.getDevicePointer(), &candidateCorrection,
                    &mtmRealPositionBase.getDevicePointer(), &realCorrection,
                    &mtmDeletedQueryPositionBase.getDevicePointer(), &deletedQueryCorrection};
            cu.executeKernel(packMtmDeletedQueriesContextPrecisionKernel, packArgs, deletedQueryCount);
            if (cu.getUseDoublePrecision())
                nonbonded->getPMEPotentialFromParticlesCudaContextPrecision(context, numAtomsPerMolecule,
                        (CudaArrayDevicePointer) mtmSelectedDeletedPositionBase.getDevicePointer(),
                        (CudaArrayDevicePointer) nullCorrection,
                        (CudaArrayDevicePointer) selectedRealChargePointer, deletedQueryCount,
                        (CudaArrayDevicePointer) mtmDeletedQueryPositionBase.getDevicePointer(),
                        (CudaArrayDevicePointer) nullCorrection,
                        (CudaArrayDevicePointer) deletedPmeQueryPotentials.getDevicePointer());
            else
                nonbonded->getPMEPotentialFromParticlesCuda(context, numAtomsPerMolecule,
                        (CudaArrayDevicePointer) mtmSelectedDeletedPositionBase.getDevicePointer(),
                        (CudaArrayDevicePointer) selectedRealChargePointer, deletedQueryCount,
                        (CudaArrayDevicePointer) mtmDeletedQueryPositionBase.getDevicePointer(),
                        (CudaArrayDevicePointer) deletedPmeQueryPotentials.getDevicePointer());
            void* exactEnergyArgs[] = {&numCandidateMolecules, &numAtomsPerMolecule,
                    &candidateParameters.getDevicePointer(), &candidateDirectEnergies.getDevicePointer(),
                    &candidateDeletedDirectEnergies.getDevicePointer(), &candidatePmePotentials.getDevicePointer(),
                    &deletedPmeQueryPotentials.getDevicePointer(), &selectedParameterPointer,
                    &selectedDirectPointer, &selectedFullPotentialPointer,
                    &candidateMoleculeEnergies.getDevicePointer(), &candidateEnergyComponents.getDevicePointer(),
                    &mtmEnergyScalars.getDevicePointer(), &realEnergyComponents.getDevicePointer()};
            cu.executeKernel(assembleMtmCandidateEnergiesKernel, exactEnergyArgs,
                    std::max(numCandidateMolecules, 1));
        }
        else {
            nonbonded->getPMEPotentialFromParticlesCuda(context, numAtomsPerMolecule,
                    (CudaArrayDevicePointer) selectedDeletedPositions.getDevicePointer(),
                    (CudaArrayDevicePointer) selectedRealChargePointer, numAtomsPerMolecule,
                    (CudaArrayDevicePointer) selectedDeletedPositions.getDevicePointer(),
                    (CudaArrayDevicePointer) selectedDeletedPmePotentials.getDevicePointer());
            nonbonded->getPMEPotentialFromParticlesCuda(context, numAtomsPerMolecule,
                    (CudaArrayDevicePointer) selectedDeletedPositions.getDevicePointer(),
                    (CudaArrayDevicePointer) selectedRealChargePointer, numCandidateAtoms,
                    (CudaArrayDevicePointer) candidatePositions.getDevicePointer(),
                    (CudaArrayDevicePointer) candidateDeletedPmePotentials.getDevicePointer());
            void* exactEnergyArgs[] = {&numCandidateMolecules, &numAtomsPerMolecule,
                    &candidateParameters.getDevicePointer(), &candidateDirectEnergies.getDevicePointer(),
                    &candidateDeletedDirectEnergies.getDevicePointer(), &candidatePmePotentials.getDevicePointer(),
                    &candidateDeletedPmePotentials.getDevicePointer(), &selectedParameterPointer,
                    &selectedDirectPointer, &selectedFullPotentialPointer,
                    &selectedDeletedPmePotentials.getDevicePointer(), &selectedRemovalComponentPointer,
                    &candidateMoleculeEnergies.getDevicePointer(), &selectedRealData.getDevicePointer(),
                    &candidateEnergyComponents.getDevicePointer(), &realEnergyComponents.getDevicePointer()};
            cu.executeKernel(assembleExactPmeEnergiesKernel, exactEnergyArgs, 1, 1);
        }
        if (!useMtm) {
            selectedRealData.download(selectedReal);
            result.deletedMoleculeEnergy = selectedReal[0].y;
        }
        if (!useMtm && !std::isfinite(result.deletedMoleculeEnergy)) {
            vector<mm_float4> deletedComponents;
            realEnergyComponents.download(deletedComponents);
            result.deletedDirectEnergy = deletedComponents[0].x;
            result.deletedFullMeshCrossEnergy = deletedComponents[0].y;
            result.deletedMeshSelfEnergy = deletedComponents[0].z;
            result.deletedMeshRemovalEnergy = deletedComponents[0].w;
            if (verbose)
                logStream << "rejectedInvalidDeletedEnergy=1"
                          << " deleteDirect=" << result.deletedDirectEnergy
                          << " deleteFullMeshCross=" << result.deletedFullMeshCrossEnergy
                          << " deleteMeshSelf=" << result.deletedMeshSelfEnergy
                          << " deleteMeshRemoval=" << result.deletedMeshRemovalEnergy << '\n';
            return result;
        }
    }

    if (candidateEnergyFaultForTesting != CandidateEnergyFault::None) {
        const bool allCandidates = candidateEnergyFaultForTesting == CandidateEnergyFault::AllNaN ||
                candidateEnergyFaultForTesting == CandidateEnergyFault::AllPositiveInfinity ||
                candidateEnergyFaultForTesting == CandidateEnergyFault::AllNegativeInfinity;
        float injectedValue;
        if (candidateEnergyFaultForTesting == CandidateEnergyFault::SingleNaN ||
                candidateEnergyFaultForTesting == CandidateEnergyFault::AllNaN)
            injectedValue = std::numeric_limits<float>::quiet_NaN();
        else if (candidateEnergyFaultForTesting == CandidateEnergyFault::SinglePositiveInfinity ||
                candidateEnergyFaultForTesting == CandidateEnergyFault::AllPositiveInfinity)
            injectedValue = std::numeric_limits<float>::infinity();
        else
            injectedValue = -std::numeric_limits<float>::infinity();
        vector<float> injected;
        if (allCandidates)
            injected.assign(numCandidateMolecules, injectedValue);
        else {
            candidateMoleculeEnergies.download(injected);
            int injectedCandidate = 0;
            if (useMtm) {
                vector<int> candidateRegions;
                candidateRegionFlags.download(candidateRegions);
                const auto target = std::find(candidateRegions.begin(), candidateRegions.end(), targetRegion);
                if (target != candidateRegions.end())
                    injectedCandidate = static_cast<int>(target-candidateRegions.begin());
            }
            injected[injectedCandidate] = injectedValue;
        }
        candidateMoleculeEnergies.upload(injected);
    }

    const double boxVolume = box[0][0]*box[1][1]*box[2][2];
    const double innerVolume = 4.0*PI*sphereRadius*sphereRadius*sphereRadius/3.0;
    const double sourceVolume = exchangeToInner ? boxVolume-innerVolume : innerVolume;
    const double targetVolume = exchangeToInner ? innerVolume : boxVolume-innerVolume;
    const double forwardProposal = computeProposalProbability(sourceCount, targetCount);
    const double reverseProposal = computeProposalProbability(targetCount+1, sourceCount-1);
    if (forwardProposal <= 0.0 || reverseProposal <= 0.0)
        throw OpenMMException("RelocationMC encountered a zero forward or reverse direction probability.");
    const float cavityBias = useCavityBias ? (float) std::log(targetVolume/sourceVolume) : 0.0f;
    const float movingBias = (float) std::log(reverseProposal/forwardProposal);
    const float sourceLogSum = scalarValues[0].x;
    const float targetLogSum = scalarValues[0].y;
    vector<mm_float4> selectedCandidate;
    int acceptedCandidate = -1;
    if (useMtm) {
        const unsigned long long mtmSeed = effectiveRandomSeed;
        const unsigned long long mtmAttempt = attemptIndex;
        void* selectCandidateArgs[] = {&numCandidateMolecules, const_cast<int*>(&targetRegion),
                const_cast<float*>(&beta), const_cast<unsigned long long*>(&mtmSeed),
                const_cast<unsigned long long*>(&mtmAttempt), &candidateRegionFlags.getDevicePointer(),
                &candidateMoleculeEnergies.getDevicePointer(), &proposalScalars.getDevicePointer(),
                &selectedCandidateData.getDevicePointer(), &mtmLogSums.getDevicePointer()};
        cu.executeKernel(selectMtmCandidateKernel, selectCandidateArgs, 256, 256);
        selectedCandidateData.download(selectedCandidate);
        const mm_float4 selection = selectedCandidate[0];
        if (!isCandidateDecisionValid(selection.x, selection.y, selection.z, selection.w,
                numCandidateMolecules) || selection.w != 1.0f) {
            if (verbose)
                logStream << "rejectedEmptyOrInvalidMtmForwardPool=1"
                          << " selectedCandidate=" << selection.x << '\n';
            return result;
        }
        acceptedCandidate = (int) selection.x;
        result.candidateSelectionProbability = selection.z;
        void* gatherArgs[] = {&numAtomsPerMolecule, &acceptedCandidate,
                &mtmCandidatePositionBase.getDevicePointer(), &candidateCorrection,
                &mtmSelectedCandidatePositionBase.getDevicePointer(), &selectedCorrection};
        cu.executeKernel(gatherMtmSelectedCandidateContextPrecisionKernel,
                gatherArgs, numAtomsPerMolecule);

        const int selectedQueryCount = numAtomsPerMolecule+numRealAtoms;
        void* packSelectedArgs[] = {&numAtomsPerMolecule, const_cast<int*>(&numRealAtoms),
                &mtmSelectedCandidatePositionBase.getDevicePointer(), &selectedCorrection,
                &mtmRealPositionBase.getDevicePointer(), &realCorrection,
                &mtmSelectedQueryPositionBase.getDevicePointer(), &selectedQueryCorrection};
        cu.executeKernel(packMtmSelectedQueriesContextPrecisionKernel,
                packSelectedArgs, selectedQueryCount);
        if (cu.getUseDoublePrecision())
            nonbonded->getPMEPotentialFromParticlesCudaContextPrecision(context, numAtomsPerMolecule,
                    (CudaArrayDevicePointer) mtmSelectedCandidatePositionBase.getDevicePointer(),
                    (CudaArrayDevicePointer) nullCorrection,
                    (CudaArrayDevicePointer) selectedRealChargePointer, selectedQueryCount,
                    (CudaArrayDevicePointer) mtmSelectedQueryPositionBase.getDevicePointer(),
                    (CudaArrayDevicePointer) nullCorrection,
                    (CudaArrayDevicePointer) selectedPmeQueryPotentials.getDevicePointer());
        else
            nonbonded->getPMEPotentialFromParticlesCuda(context, numAtomsPerMolecule,
                    (CudaArrayDevicePointer) mtmSelectedCandidatePositionBase.getDevicePointer(),
                    (CudaArrayDevicePointer) selectedRealChargePointer, selectedQueryCount,
                    (CudaArrayDevicePointer) mtmSelectedQueryPositionBase.getDevicePointer(),
                    (CudaArrayDevicePointer) selectedPmeQueryPotentials.getDevicePointer());
        if (cu.getUseDoublePrecision()) {
            void* directCorrectionArgs[] = {const_cast<int*>(&numMolecules), &numAtomsPerMolecule,
                    const_cast<int*>(&selectedMolecule), &mtmRealPositionBase.getDevicePointer(),
                    &nullCorrection, &realParameters.getDevicePointer(),
                    &mtmSelectedDeletedPositionBase.getDevicePointer(), &nullCorrection,
                    &mtmSelectedCandidatePositionBase.getDevicePointer(), &nullCorrection,
                    cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
                    cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    &mtmDirectPairCorrections.getDevicePointer()};
            cu.executeKernel(computeMtmDirectPairCorrectionsContextPrecisionKernel,
                    directCorrectionArgs, numMolecules);
        }
        else {
            void* directCorrectionArgs[] = {const_cast<int*>(&numMolecules), &numAtomsPerMolecule,
                    const_cast<int*>(&selectedMolecule), &mtmRealPositionBase.getDevicePointer(),
                    &realParameters.getDevicePointer(), &mtmSelectedDeletedPositionBase.getDevicePointer(),
                    &mtmSelectedCandidatePositionBase.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
                    cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    &mtmDirectPairCorrections.getDevicePointer()};
            cu.executeKernel(computeMtmDirectPairCorrectionsKernel, directCorrectionArgs, numMolecules);
        }
        int deletedRealPotentialOffset = numAtomsPerMolecule+numCandidateAtoms;
        void* finalSumArgs[] = {const_cast<int*>(&numMolecules), &numCandidateMolecules,
                &numAtomsPerMolecule, const_cast<int*>(&selectedMolecule),
                const_cast<int*>(&sourceRegion), const_cast<int*>(&targetRegion), const_cast<float*>(&beta),
                &moleculeRegionFlags.getDevicePointer(), &candidateRegionFlags.getDevicePointer(),
                &realMoleculeEnergies.getDevicePointer(), &realParameters.getDevicePointer(),
                &candidateMoleculeEnergies.getDevicePointer(), &mtmDirectPairCorrections.getDevicePointer(),
                &deletedPmeQueryPotentials.getDevicePointer(), &deletedRealPotentialOffset,
                &selectedPmeQueryPotentials.getDevicePointer(), &selectedCandidateData.getDevicePointer(),
                &mtmProposedMoleculeScores.getDevicePointer(), &mtmLogSums.getDevicePointer(),
                &mtmEnergyScalars.getDevicePointer()};
        cu.executeKernel(computeMtmFinalSumsKernel, finalSumArgs, 256, 256);
        if (captureMtmDiagnosticsForTesting)
            mtmProposedMoleculeScores.download(lastMtmProposedScoresForTesting);
        else
            lastMtmProposedScoresForTesting.clear();
        vector<mm_float4> logSumValues, energyValues;
        mtmLogSums.download(logSumValues);
        mtmEnergyScalars.download(energyValues);
        const mm_float4 sums = logSumValues[0];
        const mm_float4 energies = energyValues[0];
        // The Context component oracle proves that its finite-grid PME energy
        // contains this isolated-molecule reciprocal mesh half-difference in
        // addition to u_C-u_D.  This is not the analytic Ewald self correction,
        // whose charge-only contribution cancels exactly.
        const double energyResidual = energies.w;
        const double logRatio = movingBias+sums.x-sums.y+
                beta*(energies.z-selectedReal[0].y)+sums.z-sums.w-beta*energyResidual;
        const bool valid = std::isfinite(sums.x) && std::isfinite(sums.y) &&
                std::isfinite(sums.z) && std::isfinite(sums.w) &&
                std::isfinite(energies.x) && std::isfinite(energies.y) &&
                std::isfinite(energies.z) && std::isfinite(energies.w) && std::isfinite(logRatio);
        if (!valid) {
            if (verbose)
                logStream << "rejectedInvalidMtmGlobalTerms=1" << '\n';
            return result;
        }
        const double acceptanceProbability = logRatio >= 0.0 ? 1.0 : std::exp(logRatio);
        if (!std::isfinite(acceptanceProbability) || acceptanceProbability < 0.0 || acceptanceProbability > 1.0)
            return result;
        result.deletedMoleculeEnergy = energies.x;
        result.candidateMoleculeEnergy = energies.y;
        result.candidateSourceScore = energies.z;
        result.proposedSourceLogSum = sums.y;
        result.forwardLogWeightSum = sums.z;
        result.reverseLogWeightSum = sums.w;
        result.logAcceptanceRatio = logRatio;
        result.energyDelta = energies.y-energies.x+energyResidual;
        result.energyResidual = energyResidual;
        result.isolatedReciprocalDelta = energies.w;
        result.acceptanceProbability = acceptanceProbability;
        result.insertionProbability = acceptanceProbability;
        result.accepted = counterUniform(effectiveRandomSeed, attemptIndex, 5) < acceptanceProbability;
    }
    else {
        // Legacy reverse-partition approximation and ordered first-success rule.
        vector<float> acceptanceUniforms(numCandidateMolecules);
        for (float& value : acceptanceUniforms)
            value = uniform(random);
        candidateAcceptanceRandom.upload(acceptanceUniforms);
        void* selectCandidateArgs[] = {&numCandidateMolecules, const_cast<float*>(&beta),
                const_cast<float*>(&sourceLogSum), const_cast<float*>(&targetLogSum),
                const_cast<float*>(&cavityBias), const_cast<float*>(&movingBias),
                &candidateAcceptanceRandom.getDevicePointer(),
                &candidateMoleculeEnergies.getDevicePointer(), &selectedCandidateData.getDevicePointer()};
        cu.executeKernel(selectCandidateMoleculeKernel, selectCandidateArgs, 1, 1);
        selectedCandidateData.download(selectedCandidate);
        const mm_float4 rawCandidateDecision = selectedCandidate[0];
        const bool validCandidateDecision = isCandidateDecisionValid(rawCandidateDecision.x,
                rawCandidateDecision.y, rawCandidateDecision.z, rawCandidateDecision.w,
                numCandidateMolecules);
        if (validCandidateDecision) {
            acceptedCandidate = (int) rawCandidateDecision.x;
            result.candidateMoleculeEnergy = rawCandidateDecision.y;
            result.insertionProbability = rawCandidateDecision.z;
            result.accepted = rawCandidateDecision.w == 1.0f;
        }
        else {
            result.candidateMoleculeEnergy = 0.0;
            result.insertionProbability = 0.0;
            result.accepted = false;
            if (verbose)
                logStream << "rejectedInvalidCandidateDecision=1"
                          << " selectedCandidate=" << rawCandidateDecision.x
                          << " candidateEnergy=" << rawCandidateDecision.y
                          << " insertionProbability=" << rawCandidateDecision.z
                          << " acceptedFlag=" << rawCandidateDecision.w << '\n';
        }
    }
    if (!useLegacyAsymmetricEnergies && result.accepted) {
        vector<mm_float4> deletedComponents, candidateComponents;
        realEnergyComponents.download(deletedComponents);
        if (useMtm) {
            void* gatherComponentArgs[] = {&selectedCandidateData.getDevicePointer(),
                    &candidateEnergyComponents.getDevicePointer(),
                    &selectedCandidateEnergyComponents.getDevicePointer()};
            cu.executeKernel(gatherMtmSelectedComponentKernel, gatherComponentArgs, 1, 1);
            selectedCandidateEnergyComponents.download(candidateComponents);
        }
        else
            candidateEnergyComponents.download(candidateComponents);
        result.deletedDirectEnergy = deletedComponents[0].x;
        result.deletedFullMeshCrossEnergy = deletedComponents[0].y;
        result.deletedMeshSelfEnergy = deletedComponents[0].z;
        result.deletedMeshRemovalEnergy = deletedComponents[0].w;
        const int componentIndex = useMtm ? 0 : acceptedCandidate;
        result.candidateEnvironmentDirectEnergy = candidateComponents[componentIndex].x;
        result.candidateFullMeshCrossEnergy = candidateComponents[componentIndex].y;
        result.candidateDeletedMeshCrossEnergy = candidateComponents[componentIndex].z;
        // In environment-interaction mode w stores the reciprocal interaction
        // correction, not an isolated candidate self energy.
        result.candidateMeshSelfEnergy = candidateComponents[componentIndex].w;
        const bool validComponents = std::isfinite(result.deletedDirectEnergy) &&
                std::isfinite(result.deletedFullMeshCrossEnergy) &&
                std::isfinite(result.deletedMeshSelfEnergy) &&
                std::isfinite(result.deletedMeshRemovalEnergy) &&
                std::isfinite(result.candidateEnvironmentDirectEnergy) &&
                std::isfinite(result.candidateFullMeshCrossEnergy) &&
                std::isfinite(result.candidateDeletedMeshCrossEnergy) &&
                std::isfinite(result.candidateMeshSelfEnergy);
        if (!validComponents) {
            result.accepted = false;
            result.insertionProbability = 0.0;
            if (verbose)
                logStream << "rejectedInvalidEnergyComponents=1"
                          << " selectedCandidate=" << acceptedCandidate << '\n';
        }
    }

    if (result.accepted) {
        vector<Vec3> generatedPositions;
        if (useMtm) {
            generatedPositions.resize(numAtomsPerMolecule);
            if (cu.getUseDoublePrecision()) {
                vector<mm_double4> generated;
                mtmSelectedCandidatePositionBase.download(generated);
                for (int site = 0; site < numAtomsPerMolecule; site++)
                    generatedPositions[site] = Vec3(generated[site].x, generated[site].y, generated[site].z);
            }
            else {
                vector<mm_float4> generatedBase;
                mtmSelectedCandidatePositionBase.download(generatedBase);
                if (cu.getUseMixedPrecision()) {
                    vector<mm_float4> generatedCorrection;
                    mtmSelectedCandidatePositionCorrection.download(generatedCorrection);
                    for (int site = 0; site < numAtomsPerMolecule; site++)
                        generatedPositions[site] = Vec3((double) generatedBase[site].x+generatedCorrection[site].x,
                                (double) generatedBase[site].y+generatedCorrection[site].y,
                                (double) generatedBase[site].z+generatedCorrection[site].z);
                }
                else
                    for (int site = 0; site < numAtomsPerMolecule; site++)
                        generatedPositions[site] = Vec3(generatedBase[site].x, generatedBase[site].y, generatedBase[site].z);
            }
        }
        else {
            void* gatherArgs[] = {&numAtomsPerMolecule, const_cast<int*>(&acceptedCandidate),
                    &candidatePositions.getDevicePointer(), &selectedCandidatePositions.getDevicePointer()};
            cu.executeKernel(gatherSelectedCandidatePositionsKernel, gatherArgs, numAtomsPerMolecule);
            vector<mm_float4> generated;
            selectedCandidatePositions.download(generated);
            generatedPositions.resize(numAtomsPerMolecule);
            for (int site = 0; site < numAtomsPerMolecule; site++)
                generatedPositions[site] = Vec3(generated[site].x, generated[site].y, generated[site].z);
        }
        bool validGeneratedPositions = generatedPositions.size() == static_cast<size_t>(numAtomsPerMolecule);
        for (const Vec3& position : generatedPositions)
            validGeneratedPositions = validGeneratedPositions && isFiniteVec3(position);
        if (!validGeneratedPositions) {
            result.accepted = false;
            result.insertionProbability = 0.0;
            if (verbose)
                logStream << "rejectedInvalidCandidatePositions=1"
                          << " selectedCandidate=" << acceptedCandidate << '\n';
        }
        if (result.accepted) {
            vector<Vec3> positions, velocities;
            context.getPositions(positions);
            context.getVelocities(velocities);

            vector<Vec3> sourcePositions(numAtomsPerMolecule), sourceVelocities(numAtomsPerMolecule);
            vector<double> masses(numAtomsPerMolecule);
            const LabileMol& movedMolecule = labileMolecules[selectedMolecule];
            const auto reference = std::find(movedMolecule.atomIndices.begin(), movedMolecule.atomIndices.end(),
                    movedMolecule.referenceAtom);
            if (reference == movedMolecule.atomIndices.end())
                throw OpenMMException("RelocationMC selected a molecule whose reference atom is not one of its sites.");
            bool validSourceState = true;
            double totalMass = 0.0;
            for (int site = 0; site < numAtomsPerMolecule; site++) {
                const int atom = result.movedAtomIndices[site];
                sourcePositions[site] = positions[atom];
                sourceVelocities[site] = velocities[atom];
                masses[site] = context.getSystem().getParticleMass(atom);
                validSourceState = validSourceState && isFiniteVec3(sourcePositions[site]) &&
                        isFiniteVec3(sourceVelocities[site]) && std::isfinite(masses[site]) && masses[site] >= 0.0;
                totalMass += masses[site];
            }
            validSourceState = validSourceState && std::isfinite(totalMass) && totalMass > 0.0;
            if (!validSourceState) {
                result.accepted = false;
                result.insertionProbability = 0.0;
                if (verbose)
                    logStream << "rejectedInvalidSourceState=1"
                              << " selectedMolecule=" << selectedMolecule << '\n';
            }

            if (result.accepted) {
                const int referenceSite = static_cast<int>(reference-movedMolecule.atomIndices.begin());
                result.newPositions = generatedPositions;
                // The downloaded accepted coordinates are the definitive target frame.
                // Deriving the proper best-fit Q from them avoids any dependency on
                // candidate index/origin conventions and uses the coordinates that will
                // actually be committed.
                const RelocationMCVelocity::RotationMatrix velocityRotation =
                        RelocationMCVelocity::computeRotationBetweenConformations(sourcePositions,
                                result.newPositions, box, referenceSite);
                if (!isFiniteRotation(velocityRotation)) {
                    result.accepted = false;
                    result.insertionProbability = 0.0;
                    result.newPositions.clear();
                    if (verbose)
                        logStream << "rejectedInvalidVelocityRotation=1"
                                  << " selectedCandidate=" << acceptedCandidate << '\n';
                }
                if (result.accepted) {
                    const vector<Vec3> rotatedVelocities = RelocationMCVelocity::rotateVelocitiesPreservingCOM(
                            masses, sourceVelocities, velocityRotation);
                    bool validRotatedVelocities = rotatedVelocities.size() == sourceVelocities.size();
                    for (const Vec3& velocity : rotatedVelocities)
                        validRotatedVelocities = validRotatedVelocities && isFiniteVec3(velocity);
                    if (!validRotatedVelocities) {
                        result.accepted = false;
                        result.insertionProbability = 0.0;
                        result.newPositions.clear();
                        if (verbose)
                            logStream << "rejectedInvalidRotatedVelocities=1"
                                      << " selectedCandidate=" << acceptedCandidate << '\n';
                    }
                    if (result.accepted) {
                        for (int site = 0; site < numAtomsPerMolecule; site++) {
                            // Candidate generation wrapped the anchor once and retained every
                            // other site in the same molecular image.  Do not independently
                            // wrap sites here: that can split a constrained molecule across PBC.
                            positions[result.movedAtomIndices[site]] = result.newPositions[site];
                            velocities[result.movedAtomIndices[site]] = rotatedVelocities[site];
                        }
                        // All geometry, mass, and velocity validation above precedes mutation.
                        // setPositions() invalidates position-dependent force caches; the public
                        // OpenMM API then requires a separate setVelocities() call.  A platform
                        // failure in that second call is therefore the sole residual non-atomic
                        // commit case and is surfaced to the caller rather than hidden by an
                        // untestable rollback attempt.
                        context.setPositions(positions);
                        if (useMtm) {
                            // Context::setPositions() preserves the energy-base coordinates but may
                            // canonicalize or drop mixed low-order corrections.  Restore the exact
                            // accepted base+correction representation after it has invalidated all
                            // position-dependent caches, so the paired reverse pose is bitwise the
                            // state that was actually committed.
                            CUdeviceptr movedHostAtoms = realAtomIndicesDevice.getDevicePointer()+
                                    (CUdeviceptr) selectedMolecule*numAtomsPerMolecule*sizeof(int);
                            void* commitArgs[] = {const_cast<int*>(&numAtoms), &numAtomsPerMolecule,
                                    &cu.getAtomIndexArray().getDevicePointer(), &movedHostAtoms,
                                    &mtmSelectedCandidatePositionBase.getDevicePointer(), &selectedCorrection,
                                    &cu.getPosq().getDevicePointer(), &contextCorrection};
                            cu.executeKernel(commitMtmSelectedPositionsContextPrecisionKernel,
                                    commitArgs, numAtoms);
                        }
                        context.setVelocities(velocities);
                    }
                }
            }
        }
    }
    if (verbose)
        logStream << "deleteDirect=" << result.deletedDirectEnergy
                  << " deleteFullMeshCross=" << result.deletedFullMeshCrossEnergy
                  << " deleteMeshSelf=" << result.deletedMeshSelfEnergy
                  << " deleteMeshRemoval=" << result.deletedMeshRemovalEnergy
                  << " insertEnvironmentDirect=" << result.candidateEnvironmentDirectEnergy
                  << " insertFullMeshCross=" << result.candidateFullMeshCrossEnergy
                  << " insertDeletedMeshCross=" << result.candidateDeletedMeshCrossEnergy
                  << " insertMeshSelf=" << result.candidateMeshSelfEnergy << '\n'
                  << "innerMolecules=" << numInner << " externalMolecules=" << numExternal
                  << " sourceLogSum=" << sourceLogSum << " targetLogSum=" << targetLogSum
                  << " selectedMolecule=" << selectedMolecule
                  << " selectedCandidate=" << acceptedCandidate << '\n';
    return result;
}
void CudaCalcRelocationMCKernel::copyParametersToContext(ContextImpl&, const RelocationMC& mc) {
    bool moleculeDataChanged = mc.getLabileMols().size() != labileMolecules.size();
    if (!moleculeDataChanged)
        for (int index = 0; index < (int) labileMolecules.size(); index++)
            moleculeDataChanged = moleculeDataChanged ||
                    mc.getLabileMols()[index].atomIndices != labileMolecules[index].atomIndices ||
                    mc.getLabileMols()[index].referenceAtom != labileMolecules[index].referenceAtom;
    bool siteParametersChanged = mc.getLabileMolSiteParameters().size() != siteParameters.size();
    if (!siteParametersChanged)
        for (int index = 0; index < (int) siteParameters.size(); index++) {
            const LabileMolSiteParameters& first = mc.getLabileMolSiteParameters()[index];
            const LabileMolSiteParameters& second = siteParameters[index];
            siteParametersChanged = siteParametersChanged || first.charge != second.charge ||
                    first.sigma != second.sigma || first.epsilon != second.epsilon;
        }
    bool referencePositionsChanged = mc.getLabileMolReferencePositions().size() != referencePositions.size();
    if (!referencePositionsChanged)
        for (int index = 0; index < (int) referencePositions.size(); index++)
            referencePositionsChanged = referencePositionsChanged ||
                    mc.getLabileMolReferencePositions()[index] != referencePositions[index];
    if (mc.getNumCandidateSitesPerExchange() != numCandidateSites ||
            mc.getNumAtomsPerLabileMol() != numAtomsPerMolecule ||
            mc.getLigandAtoms() != ligandAtoms || moleculeDataChanged || siteParametersChanged ||
            referencePositionsChanged || mc.getOrientationsEnabled() != orientationsEnabled ||
            mc.getProtocolVersion() != protocolVersion)
        throw OpenMMException("RelocationMC topology, site data, protocol, candidate count, and orientation setting require Context reinitialization.");
    if (protocolVersion == RelocationMC::MTM && (mc.getUseCavityBias() || mc.getUseLegacyAsymmetricEnergies()))
        throw OpenMMException("RelocationMC MTM does not support cavityBias or legacy asymmetric energies.");
    temperature = mc.getTemperature();
    sphereRadius = mc.getSphereRadius();
    useCavityBias = mc.getUseCavityBias();
    useLegacyAsymmetricEnergies = mc.getUseLegacyAsymmetricEnergies();
    verbose = mc.getVerbose();
    if (mc.getRandomSeed() != randomSeed) {
        randomSeed = mc.getRandomSeed();
        effectiveRandomSeed = randomSeed == 0 ? std::random_device{}() : randomSeed;
        attemptCounter = 0;
        random.seed(effectiveRandomSeed);
    }
}
