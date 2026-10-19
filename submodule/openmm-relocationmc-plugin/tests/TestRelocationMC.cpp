// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#include "RelocationMC.h"
#include "RelocationMCKernels.h"
#include "RelocationMCOrientations.h"
#include "internal/RelocationMCVelocity.h"
#include "openmm/KernelFactory.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/reference/ReferencePlatform.h"
#include <openmm/OpenMMException.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

using namespace OpenMM;
using namespace OpenMM::RelocationMCPlugin;

static void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Function>
static void requireThrows(Function function, const char* message) {
    try {
        function();
    }
    catch (const OpenMMException&) {
        return;
    }
    throw std::runtime_error(message);
}

class NoEligibleRelocationKernel : public CalcRelocationMCKernel {
public:
    NoEligibleRelocationKernel(std::string name, const Platform& platform) :
            CalcRelocationMCKernel(name, platform) {
    }
    void initialize(const System&, const RelocationMC&, ContextImpl&) override {
    }
    ExchangeResult attemptExchange(ContextImpl&, std::stringstream&) override {
        return ExchangeResult();
    }
    void copyParametersToContext(ContextImpl&, const RelocationMC&) override {
    }
};

class NoEligibleRelocationKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl&) const override {
        if (name != CalcRelocationMCKernel::Name())
            throw OpenMMException("Unexpected test kernel name.");
        return new NoEligibleRelocationKernel(name, platform);
    }
};

static std::string createTemporaryLogPath() {
    std::stringstream path;
    path << "/tmp/relocationmc-no-eligible-" << getpid() << ".log";
    std::remove(path.str().c_str());
    return path.str();
}

static std::string readTextFile(const std::string& path) {
    std::ifstream input(path);
    std::stringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

static int countOccurrences(const std::string& text, const std::string& needle) {
    int count = 0;
    for (size_t position = 0; (position = text.find(needle, position)) != std::string::npos;
            position += needle.size())
        count++;
    return count;
}

static void requireVecClose(const Vec3& actual, const Vec3& expected, double tolerance,
        const char* message) {
    const Vec3 difference = actual-expected;
    if (difference.dot(difference) > tolerance*tolerance)
        throw std::runtime_error(message);
}

static Vec3 wrap(const Vec3& position, const Vec3 box[3]) {
    Vec3 wrapped = position;
    for (int axis = 0; axis < 3; axis++)
        wrapped[axis] -= std::floor(wrapped[axis]/box[axis][axis])*box[axis][axis];
    return wrapped;
}

static Vec3 massWeightedCOM(const std::vector<double>& masses, const std::vector<Vec3>& velocities) {
    Vec3 com(0.0, 0.0, 0.0);
    double totalMass = 0.0;
    for (int site = 0; site < static_cast<int>(masses.size()); site++) {
        com += velocities[site]*masses[site];
        totalMass += masses[site];
    }
    return com*(1.0/totalMass);
}

static double relativeKineticEnergy(const std::vector<double>& masses,
        const std::vector<Vec3>& velocities) {
    const Vec3 com = massWeightedCOM(masses, velocities);
    double energy = 0.0;
    for (int site = 0; site < static_cast<int>(masses.size()); site++) {
        const Vec3 relative = velocities[site]-com;
        energy += 0.5*masses[site]*relative.dot(relative);
    }
    return energy;
}

static void testMtmInitialBoxPoolEstimate() {
    const int candidates = 100;
    const double radius = 0.5;
    const double boxVolume = 100.0;
    const double probability = (4.0*3.14159265358979323846*radius*radius*radius/3.0)/boxVolume;
    const MtmInitialBoxPoolEstimate estimate = computeMtmInitialBoxPoolEstimate(
            candidates, radius, boxVolume);
    require(estimate.valid, "A valid MTM initial-box pool estimate was rejected.");
    require(std::fabs(estimate.innerProbability-probability) < 1e-15,
            "MTM initial-box inner probability is incorrect.");
    require(std::fabs(estimate.expectedInnerCandidates-candidates*probability) < 1e-14,
            "MTM initial-box expected inner count is incorrect.");
    require(std::fabs(estimate.emptyInnerProbability-
            std::exp(candidates*std::log1p(-probability))) < 1e-15,
            "MTM initial-box empty-pool probability is incorrect.");
    require(shouldWarnMtmInitialBoxPool(estimate),
            "An MTM initial-box empty-pool probability above one percent did not warn.");
    const std::string warning = formatMtmInitialBoxPoolWarning(estimate);
    for (const std::string& field : {"K=", "R=", "Vbox=", "p=", "expected=", "P0="})
        require(warning.find(field) != std::string::npos,
                "MTM initial-box warning omitted a required estimate field.");
    require(warning.find("initial-box") != std::string::npos &&
            warning.find("does not change the protocol or reject proposals") != std::string::npos,
            "MTM initial-box warning omitted its scope or nonrejecting semantics.");

    MtmInitialBoxPoolEstimate boundary = estimate;
    boundary.emptyInnerProbability = 0.01;
    require(!shouldWarnMtmInitialBoxPool(boundary),
            "The exact one-percent MTM empty-pool boundary warned unexpectedly.");
    boundary.emptyInnerProbability = std::nextafter(0.01, 1.0);
    require(shouldWarnMtmInitialBoxPool(boundary),
            "An MTM empty-pool probability just above one percent did not warn.");
    boundary.emptyInnerProbability = std::nextafter(0.01, 0.0);
    require(!shouldWarnMtmInitialBoxPool(boundary),
            "An MTM empty-pool probability below one percent warned unexpectedly.");

    const MtmInitialBoxPoolEstimate k8192 = computeMtmInitialBoxPoolEstimate(8192, 0.75, 64.0);
    require(k8192.valid && k8192.expectedInnerCandidates > 200.0 &&
            k8192.emptyInnerProbability >= 0.0 && !shouldWarnMtmInitialBoxPool(k8192),
            "The representative K=8192 MTM pool estimate is invalid or unexpectedly warns.");
    for (const MtmInitialBoxPoolEstimate& invalid : {
            computeMtmInitialBoxPoolEstimate(16, radius, 0.0),
            computeMtmInitialBoxPoolEstimate(16, 2.0, 1.0),
            computeMtmInitialBoxPoolEstimate(16, radius, std::numeric_limits<double>::quiet_NaN())}) {
        require(!invalid.valid && !shouldWarnMtmInitialBoxPool(invalid),
                "An invalid MTM initial-box estimate was treated as a valid probability warning.");
        const std::string invalidWarning = formatMtmInitialBoxPoolWarning(invalid);
        require(!invalidWarning.empty() && invalidWarning.find("unavailable") != std::string::npos &&
                invalidWarning.find("nan") == std::string::npos,
                "An invalid MTM initial-box estimate did not produce a finite explanatory warning.");
    }
}

static void testNoEligibleMoleculeBookkeeping() {
    ReferencePlatform platform;
    platform.registerKernelFactory(CalcRelocationMCKernel::Name(),
            new NoEligibleRelocationKernelFactory());
    System system;
    system.addParticle(12.0);
    system.addParticle(18.0);
    system.setDefaultPeriodicBoxVectors(Vec3(3.0, 0.0, 0.0), Vec3(0.0, 3.0, 0.0),
            Vec3(0.0, 0.0, 3.0));
    const std::string logPath = createTemporaryLogPath();
    RelocationMC* relocation = new RelocationMC(300.0, 0.5, 0, 4);
    relocation->setProtocolVersion(RelocationMC::Legacy);
    relocation->setLigandAtoms({0});
    relocation->setLabileMolSiteParameters({{0.0, 0.3, 0.0}});
    relocation->setLabileMolReferencePositions({Vec3(0.0, 0.0, 0.0)});
    relocation->addLabileMol({1}, 1);
    relocation->setVerbose(true);
    relocation->setLogFileName(logPath);
    system.addForce(relocation);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    const std::vector<Vec3> positions = {Vec3(1.5, 1.5, 1.5), Vec3(0.5, 0.5, 0.5)};
    const std::vector<Vec3> velocities = {Vec3(0.1, 0.2, 0.3), Vec3(-0.4, 0.5, -0.6)};
    context.setPositions(positions);
    context.setVelocities(velocities);
    const ExchangeResult first = relocation->attemptExchange(context);
    const ExchangeResult second = relocation->attemptExchange(context);
    require(!first.attempted && !first.accepted && !second.attempted && !second.accepted,
            "A no-eligible kernel invocation was incorrectly reported as an attempted proposal.");
    require(first.protocolVersion == static_cast<int>(RelocationMC::Legacy) &&
                    second.protocolVersion == static_cast<int>(RelocationMC::Legacy),
            "The Context protocol was not recorded on an early-return result.");
    const State state = context.getState(State::Positions | State::Velocities);
    for (int atom = 0; atom < 2; atom++) {
        requireVecClose(state.getPositions()[atom], positions[atom], 0.0,
                "A no-eligible kernel invocation changed positions.");
        requireVecClose(state.getVelocities()[atom], velocities[atom], 0.0,
                "A no-eligible kernel invocation changed velocities.");
    }
    const std::string log = readTextFile(logPath);
    require(countOccurrences(log, "attempted=0 accepted=0 proposalCount=0") == 2,
            "No-eligible invocations advanced actual-proposal accounting.");
    require(countOccurrences(log, "-----1------\n") == 1 &&
            countOccurrences(log, "-----2------\n") == 1,
            "Log invocation numbering did not advance independently of actual proposals.");
    relocation->setUseCavityBias(false);
    relocation->setProtocolVersion(RelocationMC::MTM);
    requireThrows([&]() { relocation->updateParametersInContext(context); },
            "Changing protocol on an initialized Context should require reinitialization.");
    std::remove(logPath.c_str());
}

static void testProtocolConfiguration() {
    RelocationMC relocation;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    requireThrows([&]() { relocation.setTemperature(nan); },
            "A nonfinite temperature should be rejected.");
    requireThrows([&]() { relocation.setTemperature(inf); },
            "An infinite temperature should be rejected.");
    requireThrows([&]() { relocation.setSphereRadius(nan); },
            "A nonfinite sphere radius should be rejected.");
    requireThrows([&]() { relocation.setLabileMolSiteParameters({{nan, 0.3, 0.1}}); },
            "Nonfinite labile site parameters should be rejected.");
    requireThrows([&]() { relocation.setLabileMolReferencePositions({Vec3(0.0, inf, 0.0)}); },
            "Nonfinite labile reference positions should be rejected.");
    require(relocation.getProtocolVersion() == RelocationMC::MTM,
            "RelocationMC did not default to the MTM protocol.");
    require(relocation.getOrientationsEnabled() && !relocation.getUseCavityBias() &&
                    !relocation.getUseLegacyAsymmetricEnergies(),
            "RelocationMC MTM defaults are inconsistent.");
    requireThrows([&]() { relocation.setProtocolVersion(
            static_cast<RelocationMC::ProtocolVersion>(0)); },
            "Protocol version 0 should be rejected.");
    requireThrows([&]() { relocation.setProtocolVersion(
            static_cast<RelocationMC::ProtocolVersion>(3)); },
            "An unknown protocol version should be rejected.");
    require(relocation.getProtocolVersion() == RelocationMC::MTM,
            "A rejected protocol update changed the configured protocol.");
    requireThrows([&]() { relocation.setOrientationsEnabled(false); },
            "MTM should reject disabling Haar orientations.");
    requireThrows([&]() { relocation.setUseCavityBias(true); },
            "MTM should reject a separate cavity-volume bias.");
    requireThrows([&]() { relocation.setUseLegacyAsymmetricEnergies(true); },
            "MTM should reject legacy asymmetric energies.");
    require(relocation.getOrientationsEnabled() && !relocation.getUseCavityBias() &&
                    !relocation.getUseLegacyAsymmetricEnergies(),
            "A rejected MTM setting partially changed the Force.");

    relocation.setProtocolVersion(RelocationMC::Legacy);
    relocation.setOrientationsEnabled(false);
    relocation.setUseCavityBias(true);
    relocation.setUseLegacyAsymmetricEnergies(true);
    require(relocation.getProtocolVersion() == RelocationMC::Legacy &&
                    !relocation.getOrientationsEnabled() && relocation.getUseCavityBias() &&
                    relocation.getUseLegacyAsymmetricEnergies(),
            "Legacy protocol compatibility settings were not retained.");

    ExchangeResult result;
    require(result.protocolVersion == 0 && result.candidateSelectionProbability == 0.0 &&
                    result.acceptanceProbability == 0.0 && result.deletedSourceScore == 0.0 &&
                    result.candidateSourceScore == 0.0 && result.sourceLogSum == 0.0 &&
                    result.proposedSourceLogSum == 0.0 && result.forwardLogWeightSum == 0.0 &&
                    result.reverseLogWeightSum == 0.0 && result.logAcceptanceRatio == 0.0 &&
                    result.energyDelta == 0.0 && result.energyResidual == 0.0 &&
                    result.isolatedReciprocalDelta == 0.0,
            "New ExchangeResult protocol diagnostics did not default to zero.");
}

static void testVelocityRotationUtilities() {
    using namespace RelocationMCVelocity;
    const Vec3 box[3] = {Vec3(4.0, 0.0, 0.0), Vec3(0.0, 4.0, 0.0), Vec3(0.0, 0.0, 4.0)};
    const PreparedOrientations prepared;
    const std::vector<double> orientationMasses = {16.0, 1.0, 2.0};
    const std::vector<Vec3> orientationVelocities = {
        Vec3(1.2, -0.4, 0.3), Vec3(-0.7, 1.1, 0.8), Vec3(0.2, -1.3, 0.5)};
    const Vec3 orientationCOM = massWeightedCOM(orientationMasses, orientationVelocities);
    const double orientationKE = relativeKineticEnergy(orientationMasses, orientationVelocities);
    for (int orientation = 0; orientation < prepared.size(); orientation++) {
        const RotationMatrix& matrix = prepared.getRotationMatrix(orientation);
        for (int first = 0; first < 3; first++) {
            require(std::fabs(matrix[first].dot(matrix[first])-1.0) < 2e-12,
                    "A normalized prepared orientation basis image was not unit length.");
            for (int second = first+1; second < 3; second++)
                require(std::fabs(matrix[first].dot(matrix[second])) < 2e-12,
                        "A normalized prepared orientation matrix was not orthogonal.");
        }
        const double determinant = matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])-
                matrix[1][0]*(matrix[0][1]*matrix[2][2]-matrix[0][2]*matrix[2][1])+
                matrix[2][0]*(matrix[0][1]*matrix[1][2]-matrix[0][2]*matrix[1][1]);
        require(std::fabs(determinant-1.0) < 3e-12,
                "A normalized prepared orientation was not a proper rotation.");
        const std::vector<Vec3> transformed = rotateVelocitiesPreservingCOM(
                orientationMasses, orientationVelocities, matrix);
        requireVecClose(massWeightedCOM(orientationMasses, transformed), orientationCOM, 2e-13,
                "A normalized prepared orientation did not preserve COM velocity.");
        require(std::fabs(relativeKineticEnergy(orientationMasses, transformed)-orientationKE) < 2e-12,
                "A normalized prepared orientation did not preserve relative kinetic energy.");
    }
    const RotationMatrix sourceOrientation = {
        Vec3(0.0, 1.0, 0.0), Vec3(-1.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0)};
    const RotationMatrix targetOrientation = {
        Vec3(1.0, 0.0, 0.0), Vec3(0.0, -1.0, 0.0), Vec3(0.0, 0.0, -1.0)};
    const RotationMatrix expectedRotation = {
        Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, -1.0)};
    const std::vector<Vec3> reference = {
        Vec3(0.0, 0.0, 0.0), Vec3(0.1, 0.0, 0.0), Vec3(0.02, 0.08, 0.03)};
    const Vec3 sourceAnchor(3.98, 3.97, 0.02);
    std::vector<Vec3> source(reference.size());
    for (int site = 0; site < static_cast<int>(reference.size()); site++)
        source[site] = wrap(sourceAnchor+applyRotation(sourceOrientation, reference[site]), box);
    const Vec3 targetAnchor(1.2, 2.3, 3.4);
    std::vector<Vec3> target(reference.size());
    for (int site = 0; site < static_cast<int>(reference.size()); site++)
        target[site] = wrap(targetAnchor+applyRotation(targetOrientation, reference[site]), box);
    const RotationMatrix coordinateDerivedRotation = computeRotationBetweenConformations(source, target, box, 0);
    for (int axis = 0; axis < 3; axis++)
        requireVecClose(coordinateDerivedRotation[axis], expectedRotation[axis], 1.0e-12,
                "Coordinate-derived source-to-accepted rotation was not exact.");
    std::vector<Vec3> shiftedReference = {Vec3(0.2, -0.3, 0.4), Vec3(0.3, -0.3, 0.4),
                                          Vec3(0.22, -0.22, 0.43)};
    std::vector<Vec3> shiftedTarget(shiftedReference.size());
    for (int site = 0; site < static_cast<int>(shiftedReference.size()); site++)
        shiftedTarget[site] = wrap(targetAnchor+applyRotation(targetOrientation, shiftedReference[site]), box);
    const RotationMatrix nonOriginReferenceRotation = computeRotationBetweenConformations(
            source, shiftedTarget, box, 0);
    for (int axis = 0; axis < 3; axis++)
        requireVecClose(nonOriginReferenceRotation[axis], expectedRotation[axis], 1.0e-12,
                "Actual-coordinate rotation depended on the candidate template origin.");
    std::vector<Vec3> ordinaryNoisySource = source;
    ordinaryNoisySource[1][0] += 8.0e-5;
    ordinaryNoisySource[2][1] -= 1.2e-4;
    const RotationMatrix noisyFit = computeRotationBetweenConformations(
            ordinaryNoisySource, target, box, 0);
    for (int axis = 0; axis < 3; axis++)
        require(std::fabs(noisyFit[axis].dot(noisyFit[axis])-1.0) < 1.0e-12,
                "Production best-fit rotation for ordinary water noise was not orthogonal.");
    std::vector<Vec3> distortedSource = source;
    distortedSource[2][0] += 0.002;
    requireThrows([&]() { computeRotationBetweenConformations(
            distortedSource, target, box, 0); },
            "Production rotation should reject a grossly distorted source molecule.");
    const std::vector<double> masses = {16.0, 1.0, 2.0};
    const std::vector<Vec3> velocities = {
        Vec3(1.2, -0.4, 0.3), Vec3(-0.7, 1.1, 0.8), Vec3(0.2, -1.3, 0.5)};
    const std::vector<Vec3> rotated = rotateVelocitiesPreservingCOM(
            masses, velocities, coordinateDerivedRotation);
    const Vec3 originalCOM = massWeightedCOM(masses, velocities);
    const Vec3 rotatedCOM = massWeightedCOM(masses, rotated);
    requireVecClose(rotatedCOM, originalCOM, 1.0e-13,
            "Unequal-mass velocity rotation did not preserve COM velocity and linear momentum.");
    require(std::fabs(relativeKineticEnergy(masses, rotated)-relativeKineticEnergy(masses, velocities)) < 1.0e-12,
            "Velocity rotation did not preserve mass-weighted relative kinetic energy.");
    for (int site = 0; site < static_cast<int>(velocities.size()); site++)
        requireVecClose(rotated[site], originalCOM+applyRotation(expectedRotation, velocities[site]-originalCOM),
                1.0e-13, "Velocity rotation did not apply the exact known rigid rotation.");

    const RotationMatrix singleSiteRotation = computeRotationBetweenConformations(
            {Vec3(3.9, 0.1, 0.2)}, {Vec3(1.3, 2.2, 3.1)}, box, 0);
    for (int axis = 0; axis < 3; axis++)
        requireVecClose(singleSiteRotation[axis], identityRotation()[axis], 0.0,
                "Single-site degeneracy did not select the identity rotation.");
    const std::vector<Vec3> singleVelocity = {Vec3(0.4, -0.2, 0.7)};
    requireVecClose(rotateVelocitiesPreservingCOM({3.0}, singleVelocity, singleSiteRotation)[0],
            singleVelocity[0], 0.0, "Single-site degeneracy changed its velocity.");

    const RotationMatrix halfTurn = bestFitProperRotation(
            {Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0)},
            {Vec3(1.0, 0.0, 0.0), Vec3(0.0, -1.0, 0.0), Vec3(0.0, 0.0, -1.0)});
    requireVecClose(applyRotation(halfTurn, Vec3(0.0, 1.0, 0.0)), Vec3(0.0, -1.0, 0.0),
            1.0e-12, "Horn Jacobi solver did not select the largest-algebraic 180-degree rotation.");
    const RotationMatrix linearRotation = computeRotationBetweenConformations(
            {Vec3(0.2, 0.2, 0.2), Vec3(0.1, 0.2, 0.2)},
            {Vec3(1.0, 1.0, 1.0), Vec3(1.1, 1.0, 1.0)}, box, 0);
    requireVecClose(applyRotation(linearRotation, Vec3(-1.0, 0.0, 0.0)),
            Vec3(1.0, 0.0, 0.0), 1.0e-12,
            "Linear degeneracy did not use the deterministic minimum-angle axis rotation.");
    for (int axis = 0; axis < 3; axis++)
        require(std::fabs(linearRotation[axis].dot(linearRotation[axis])-1.0) < 1.0e-12,
                "Linear-degeneracy rotation was not orthogonal.");

    requireThrows([&]() { rotateVelocitiesPreservingCOM({0.0}, singleVelocity, identityRotation()); },
            "A zero-total-mass molecule should be rejected by the velocity transform.");
}

int main() {
    try {
        testMtmInitialBoxPoolEstimate();
        testNoEligibleMoleculeBookkeeping();
        testProtocolConfiguration();
        testVelocityRotationUtilities();
        RelocationMC relocation(300.0, 0.5, 25, 32);
        relocation.setRandomSeed(12);
        relocation.setLigandAtoms({0, 1});
        relocation.setLabileMolSiteParameters({{-0.8, 0.3, 0.5}, {0.4, 0.0, 0.0}});
        relocation.setLabileMolReferencePositions({Vec3(0, 0, 0), Vec3(0.1, 0, 0)});
        relocation.addLabileMol({2, 3}, 2);
        relocation.addLabileMol({4, 5}, 4);

        require(relocation.getTemperature() == 300.0, "Unexpected temperature.");
        require(relocation.getSphereRadius() == 0.5, "Unexpected sphere radius.");
        require(relocation.getExchangeFrequency() == 25, "Unexpected exchange frequency.");
        require(relocation.getNumCandidateSitesPerExchange() == 32, "Unexpected candidate count.");
        require(relocation.getNumLabileMols() == 2, "Unexpected labile molecule count.");
        require(relocation.getNumAtomsPerLabileMol() == 2, "Unexpected labile molecule size.");

        std::vector<int> atoms;
        int referenceAtom = -1;
        relocation.getLabileMolParameters(1, atoms, referenceAtom);
        require(atoms == std::vector<int>({4, 5}) && referenceAtom == 4,
                "Labile molecule parameters were not retained.");

        requireThrows([&]() { relocation.addLabileMol({5, 6}, 5); },
                "Overlapping labile molecules should be rejected.");
        requireThrows([&]() { relocation.setExchangeFrequency(-1); },
                "Negative exchange frequency should be rejected.");
        requireThrows([&]() { relocation.setNumCandidateSitesPerExchange(0); },
                "Zero candidate sites should be rejected.");

        System zeroMassSystem;
        for (int i = 0; i < 3; i++)
            zeroMassSystem.addParticle(i == 0 ? 1.0 : 0.0);
        zeroMassSystem.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
        RelocationMC* zeroMassRelocation = new RelocationMC(300.0, 0.5, 0, 4);
        zeroMassRelocation->setLigandAtoms({0});
        zeroMassRelocation->setLabileMolSiteParameters({{0.0, 0.3, 0.0}});
        zeroMassRelocation->setLabileMolReferencePositions({Vec3(0, 0, 0)});
        zeroMassRelocation->addLabileMol({1}, 1);
        zeroMassRelocation->addLabileMol({2}, 2);
        zeroMassSystem.addForce(zeroMassRelocation);
        VerletIntegrator zeroMassIntegrator(0.001);
        requireThrows([&]() { Context context(zeroMassSystem, zeroMassIntegrator,
                Platform::getPlatformByName("Reference")); },
                "Zero-total-mass labile molecules should be rejected.");

        System triclinicSystem;
        for (int i = 0; i < 3; i++)
            triclinicSystem.addParticle(1.0);
        triclinicSystem.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0.2, 3, 0), Vec3(0, 0, 3));
        RelocationMC* triclinicRelocation = new RelocationMC(300.0, 0.5, 0, 4);
        triclinicRelocation->setLigandAtoms({0});
        triclinicRelocation->setLabileMolSiteParameters({{0.0, 0.3, 0.0}});
        triclinicRelocation->setLabileMolReferencePositions({Vec3(0, 0, 0)});
        triclinicRelocation->addLabileMol({1}, 1);
        triclinicRelocation->addLabileMol({2}, 2);
        triclinicSystem.addForce(triclinicRelocation);
        VerletIntegrator integrator(0.001);
        requireThrows([&]() { Context context(triclinicSystem, integrator, Platform::getPlatformByName("Reference")); },
                "Triclinic boxes should be rejected.");

        std::cout << "TestRelocationMC: OK" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
