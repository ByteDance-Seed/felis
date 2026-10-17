// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#include "internal/RelocationMCImpl.h"
#include "RelocationMCKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <iomanip>
#include <set>

using namespace OpenMM;
using namespace OpenMM::RelocationMCPlugin;
using std::map;
using std::string;
using std::vector;

MtmInitialBoxPoolEstimate OpenMM::RelocationMCPlugin::computeMtmInitialBoxPoolEstimate(
        int candidateCount, double sphereRadius, double boxVolume) {
    MtmInitialBoxPoolEstimate estimate = {candidateCount,
            std::isfinite(sphereRadius) ? sphereRadius : 0.0,
            std::isfinite(boxVolume) ? boxVolume : 0.0,
            0.0, 0.0, 0.0, false};
    if (candidateCount <= 0 || !std::isfinite(sphereRadius) || sphereRadius <= 0.0 ||
            !std::isfinite(boxVolume) || boxVolume <= 0.0)
        return estimate;
    const double innerVolume = 4.0*3.14159265358979323846*sphereRadius*sphereRadius*sphereRadius/3.0;
    const double probability = innerVolume/boxVolume;
    if (!std::isfinite(innerVolume) || !std::isfinite(probability) ||
            !(probability > 0.0 && probability < 1.0))
        return estimate;
    const double expected = candidateCount*probability;
    const double emptyProbability = std::exp(candidateCount*std::log1p(-probability));
    if (!std::isfinite(expected) || !std::isfinite(emptyProbability) ||
            emptyProbability < 0.0 || emptyProbability > 1.0)
        return estimate;
    estimate.innerProbability = probability;
    estimate.expectedInnerCandidates = expected;
    estimate.emptyInnerProbability = emptyProbability;
    estimate.valid = true;
    return estimate;
}

bool OpenMM::RelocationMCPlugin::shouldWarnMtmInitialBoxPool(
        const MtmInitialBoxPoolEstimate& estimate) {
    return estimate.valid && estimate.emptyInnerProbability > 0.01;
}

string OpenMM::RelocationMCPlugin::formatMtmInitialBoxPoolWarning(
        const MtmInitialBoxPoolEstimate& estimate) {
    std::stringstream message;
    message << std::setprecision(10);
    if (!estimate.valid) {
        message << "RelocationMC MTM warning: initial-box whole-box candidate-pool estimate is unavailable; "
                << "K=" << estimate.candidateCount
                << " R=" << estimate.sphereRadius
                << " Vbox=" << estimate.boxVolume
                << " p=unavailable expected=unavailable P0=unavailable. "
                << "This initial-box estimate does not change the protocol or reject proposals.\n";
        return message.str();
    }
    if (!shouldWarnMtmInitialBoxPool(estimate))
        return "";
    message << "RelocationMC MTM warning: initial-box whole-box candidate-pool estimate has P0>0.01; "
            << "K=" << estimate.candidateCount
            << " R=" << estimate.sphereRadius
            << " Vbox=" << estimate.boxVolume
            << " p=" << estimate.innerProbability
            << " expected=" << estimate.expectedInnerCandidates
            << " P0=" << estimate.emptyInnerProbability << ". "
            << "This initial-box estimate does not change the protocol or reject proposals.\n";
    return message.str();
}

RelocationMCImpl::RelocationMCImpl(const RelocationMC& owner) : owner(owner),
        protocolVersion(owner.getProtocolVersion()),
        lastAutomaticAttemptStep(std::numeric_limits<long long>::min()), numAttempted(0), numInvoked(0) {
}

RelocationMCImpl::~RelocationMCImpl() {
}

void RelocationMCImpl::validateConfiguration(ContextImpl& context) const {
    const System& system = context.getSystem();
    if (!system.usesPeriodicBoundaryConditions())
        throw OpenMMException("RelocationMC requires a periodic System.");
    Vec3 box[3];
    context.getPeriodicBoxVectors(box[0], box[1], box[2]);
    if (box[0][1] != 0.0 || box[0][2] != 0.0 || box[1][0] != 0.0 ||
            box[1][2] != 0.0 || box[2][0] != 0.0 || box[2][1] != 0.0)
        throw OpenMMException("RelocationMC supports orthorhombic periodic boxes only.");
    if (owner.getLigandAtoms().empty())
        throw OpenMMException("RelocationMC requires at least one ligand atom for defining its unique ligand center.");
    if (owner.getNumLabileMols() == 0)
        throw OpenMMException("RelocationMC requires at least one labile molecule.");
    if (owner.getLabileMolSiteParameters().empty())
        throw OpenMMException("RelocationMC requires labile-molecule site parameters.");
    if (owner.getLabileMolReferencePositions().size() != owner.getLabileMolSiteParameters().size())
        throw OpenMMException("RelocationMC requires one reference position per labile-molecule site.");

    const int numParticles = system.getNumParticles();
    for (int atom : owner.getLigandAtoms())
        if (atom < 0 || atom >= numParticles)
            throw OpenMMException("RelocationMC ligand atom index is out of range.");

    std::set<int> assignedAtoms;
    for (const auto& molecule : owner.getLabileMols()) {
        if (molecule.atomIndices.size() != owner.getLabileMolSiteParameters().size())
            throw OpenMMException("RelocationMC labile molecules and site parameters have different sizes.");
        double totalMass = 0.0;
        for (int atom : molecule.atomIndices) {
            if (atom < 0 || atom >= numParticles)
                throw OpenMMException("RelocationMC labile-molecule atom index is out of range.");
            if (!assignedAtoms.insert(atom).second)
                throw OpenMMException("A particle belongs to more than one RelocationMC labile molecule.");
            totalMass += system.getParticleMass(atom);
        }
        if (!(totalMass > 0.0))
            throw OpenMMException("RelocationMC requires positive total mass for every labile molecule.");
    }
}

void RelocationMCImpl::initialize(ContextImpl& context) {
    validateConfiguration(context);
    kernel = context.getPlatform().createKernel(CalcRelocationMCKernel::Name(), context);
    kernel.getAs<CalcRelocationMCKernel>().initialize(context.getSystem(), owner, context);
}

bool RelocationMCImpl::isExchangeDue(const Context& context) const {
    const int frequency = owner.getExchangeFrequency();
    const long long step = context.getStepCount();
    return frequency > 0 && step > 0 && step % frequency == 0 && step != lastAutomaticAttemptStep;
}

ExchangeResult RelocationMCImpl::performExchange(ContextImpl& context, bool* forcesInvalid) {
    // Preserve the existing pre-attempt full-Hamiltonian evaluation that keeps
    // RelocationMC's reciprocal-space queries synchronized with the current
    // PerturbNonbondedForce PME grid.  ContextImpl avoids re-entering
    // updateContextState().  Verbose logging reuses this value at no added cost.
    const double initialPotentialEnergy = context.calcForcesAndEnergy(false, true, 0xFFFFFFFF);
    ExchangeResult result = kernel.getAs<CalcRelocationMCKernel>().attemptExchange(context, logStream);
    // Record the protocol cached by this Context even when a platform kernel
    // returns before it can populate proposal diagnostics.
    result.protocolVersion = static_cast<int>(protocolVersion);
    if (result.accepted && forcesInvalid != nullptr)
        *forcesInvalid = true;
    // The post-commit full Context evaluation is logging-only and therefore is
    // paid only for accepted verbose attempts.
    const double updatedPotentialEnergy = owner.getVerbose() && result.accepted ?
            context.calcForcesAndEnergy(false, true, 0xFFFFFFFF) : 0.0;
    ++numInvoked;
    if (result.attempted)
        ++numAttempted;
    writeLog(result, initialPotentialEnergy, updatedPotentialEnergy);
    return result;
}

ExchangeResult RelocationMCImpl::attemptExchange(Context& context) {
    return performExchange(getContextImpl(context));
}

void RelocationMCImpl::updateContextState(ContextImpl& context, bool& forcesInvalid) {
    if (!isExchangeDue(context.getOwner()))
        return;
    lastAutomaticAttemptStep = context.getOwner().getStepCount();
    performExchange(context, &forcesInvalid);
}

void RelocationMCImpl::updateParametersInContext(ContextImpl& context) {
    validateConfiguration(context);
    if (owner.getProtocolVersion() != protocolVersion)
        throw OpenMMException("RelocationMC protocol changes require Context reinitialization.");
    kernel.getAs<CalcRelocationMCKernel>().copyParametersToContext(context, owner);
}

void RelocationMCImpl::writeLog(const ExchangeResult& result, double initialPotentialEnergy,
        double updatedPotentialEnergy) {
    if (!owner.getVerbose())
        return;
    if (owner.getLogFileName().empty())
        throw OpenMMException("RelocationMC verbose logging requires a log file name.");
    if (!logFile.is_open()) {
        logFile.open(owner.getLogFileName(), std::ios::out | std::ios::app);
        if (!logFile)
            throw OpenMMException("RelocationMC could not open its log file.");
        logFile << std::setprecision(17);
    }
    logFile << "-----" << numInvoked << "------\n";
    logFile << "attempted=" << result.attempted
            << " accepted=" << result.accepted
            << " proposalCount=" << numAttempted
            << " direction=" << (result.exchangeToInner ? "outer-to-inner" : "inner-to-outer")
            << " molecule=" << result.movedLabileMol
            << " protocolVersion=" << result.protocolVersion
            << " deletionProbability=" << result.deletionProbability
            << " insertionProbability=" << result.insertionProbability
            << " candidateSelectionProbability=" << result.candidateSelectionProbability
            << " acceptanceProbability=" << result.acceptanceProbability << '\n';
    if (result.protocolVersion == static_cast<int>(RelocationMC::MTM))
        logFile << "deletedSourceScore=" << result.deletedSourceScore
                << " candidateSourceScore=" << result.candidateSourceScore
                << " sourceLogSum=" << result.sourceLogSum
                << " proposedSourceLogSum=" << result.proposedSourceLogSum
                << " forwardLogWeightSum=" << result.forwardLogWeightSum
                << " reverseLogWeightSum=" << result.reverseLogWeightSum
                << " logAcceptanceRatio=" << result.logAcceptanceRatio
                << " energyDelta_kJ_per_mol=" << result.energyDelta
                << " energyResidual_kJ_per_mol=" << result.energyResidual
                << " isolatedReciprocalDelta_kJ_per_mol=" << result.isolatedReciprocalDelta << '\n';
    if (result.accepted) {
        logFile << "initPotentialEnergy_kJ_per_mol=" << initialPotentialEnergy << '\n';
        logFile << "updatedPotentialEnergy_kJ_per_mol=" << updatedPotentialEnergy << '\n';
        logFile << "deleteMoleculeEnergy_kJ_per_mol=" << result.deletedMoleculeEnergy << '\n';
        logFile << "insertMoleculeEnergy_kJ_per_mol=" << result.candidateMoleculeEnergy << '\n';
    }
    if (!logStream.str().empty()) {
        logFile << logStream.str();
        logStream.str("");
        logStream.clear();
    }
    logFile.flush();
}

map<string, double> RelocationMCImpl::getDefaultParameters() {
    return map<string, double>();
}

vector<string> RelocationMCImpl::getKernelNames() {
    return {CalcRelocationMCKernel::Name()};
}
