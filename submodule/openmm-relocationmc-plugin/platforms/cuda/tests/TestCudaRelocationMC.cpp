// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#include "CudaTests.h"
#include "openmm/cuda/CudaKernelSources.h"
#include "CudaRelocationMCKernelSources.h"
#include "CudaRelocationMCKernels.h"
#include "RelocationMC.h"
#include "RelocationMCOrientations.h"
#include "PerturbNonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/CMMotionRemover.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/CustomIntegrator.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "internal/RelocationMCVelocity.h"
#include "openmm/VerletIntegrator.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

using namespace OpenMM;
using namespace OpenMM::RelocationMCPlugin;
using namespace PerturbNonbondedPlugin;
using std::vector;

static void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

static std::string createTemporaryLogPath(const std::string& label) {
    static int sequence = 0;
    std::stringstream path;
    path << "/tmp/relocationmc-" << label << '-' << getpid() << '-' << sequence++ << ".log";
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

static void requireClose(double actual, double expected, double tolerance, const std::string& message) {
    const double scale = std::max(1.0, std::max(std::fabs(actual), std::fabs(expected)));
    require(std::fabs(actual-expected) <= tolerance*scale, message);
}

static void requireVecClose(const Vec3& actual, const Vec3& expected, double tolerance,
        const std::string& message) {
    const Vec3 difference = actual-expected;
    require(difference.dot(difference) <= tolerance*tolerance, message);
}

template <class Function>
static void requireThrowsContaining(Function function, const std::string& text,
        const std::string& message) {
    try {
        function();
    }
    catch (const OpenMMException& error) {
        require(std::string(error.what()).find(text) != std::string::npos, message+": "+error.what());
        return;
    }
    throw std::runtime_error(message);
}

static Vec3 massWeightedCOM(const vector<double>& masses, const vector<Vec3>& velocities) {
    Vec3 com(0.0, 0.0, 0.0);
    double totalMass = 0.0;
    for (int site = 0; site < static_cast<int>(masses.size()); site++) {
        com += velocities[site]*masses[site];
        totalMass += masses[site];
    }
    return com*(1.0/totalMass);
}

static double relativeKineticEnergy(const vector<double>& masses, const vector<Vec3>& velocities) {
    const Vec3 com = massWeightedCOM(masses, velocities);
    double energy = 0.0;
    for (int site = 0; site < static_cast<int>(masses.size()); site++) {
        const Vec3 relative = velocities[site]-com;
        energy += 0.5*masses[site]*relative.dot(relative);
    }
    return energy;
}

void testPreparedOrientations() {
    PreparedOrientations prepared;
    require(prepared.size() == 60, "Prepared orientation table should contain 60 rotations.");
    const Vec3 reference(0.1, -0.03, 0.07);
    const double referenceNorm2 = reference.dot(reference);
    const vector<double> testMasses = {16.0, 1.0, 2.0};
    const vector<Vec3> testVelocities = {
        Vec3(1.2, -0.4, 0.3), Vec3(-0.7, 1.1, 0.8), Vec3(0.2, -1.3, 0.5)};
    const Vec3 testCOM = massWeightedCOM(testMasses, testVelocities);
    const double testRelativeKE = relativeKineticEnergy(testMasses, testVelocities);
    std::set<std::array<long long, 3> > distinct;
    for (int orientation = 0; orientation < prepared.size(); orientation++) {
        const std::array<Vec3, 3>& matrix = prepared.getRotationMatrix(orientation);
        for (int i = 0; i < 3; i++) {
            require(std::fabs(matrix[i].dot(matrix[i])-1.0) < 2e-12,
                    "Prepared orientation matrix basis image is not unit length.");
            for (int j = i+1; j < 3; j++)
                require(std::fabs(matrix[i].dot(matrix[j])) < 2e-12,
                        "Prepared orientation matrix basis images are not orthogonal.");
        }
        const double determinant = matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])-
                matrix[1][0]*(matrix[0][1]*matrix[2][2]-matrix[0][2]*matrix[2][1])+
                matrix[2][0]*(matrix[0][1]*matrix[1][2]-matrix[0][2]*matrix[1][1]);
        require(std::fabs(determinant-1.0) < 3e-12, "Prepared orientation matrix determinant differs from one.");
        const Vec3 rotated = prepared.rotate(orientation, reference);
        require(std::fabs(rotated.dot(rotated)-referenceNorm2) < 3e-14,
                "Prepared orientation did not preserve rigid distance.");
        vector<Vec3> rotatedVelocities(testVelocities.size());
        for (int site = 0; site < static_cast<int>(testVelocities.size()); site++)
            rotatedVelocities[site] = testCOM+prepared.rotate(orientation, testVelocities[site]-testCOM);
        requireVecClose(massWeightedCOM(testMasses, rotatedVelocities), testCOM, 2e-13,
                "A prepared orientation did not preserve unequal-mass COM velocity.");
        requireClose(relativeKineticEnergy(testMasses, rotatedVelocities), testRelativeKE, 2e-13,
                "A prepared orientation did not preserve relative kinetic energy.");
        distinct.insert({(long long) std::llround(rotated[0]*1e6),
                         (long long) std::llround(rotated[1]*1e6),
                         (long long) std::llround(rotated[2]*1e6)});
    }
    require(distinct.size() > 30, "Prepared proposal table did not cover multiple orientations.");

    const int candidates = 10000;
    const int repeats = 101;
    std::mt19937_64 generator(20260829);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::uniform_int_distribution<int> orientation(0, prepared.size()-1);
    auto median = [&](bool usePrepared) {
        vector<double> timings;
        timings.reserve(repeats);
        volatile double checksum = 0.0;
        for (int repeat = -10; repeat < repeats; repeat++) {
            const auto start = std::chrono::steady_clock::now();
            for (int candidate = 0; candidate < candidates; candidate++) {
                Vec3 rotated;
                if (usePrepared)
                    rotated = prepared.rotate(orientation(generator), reference);
                else {
                    const double u1 = uniform(generator), u2 = uniform(generator), u3 = uniform(generator);
                    const double x = std::sqrt(1-u1)*std::sin(2*M_PI*u2);
                    const double y = std::sqrt(1-u1)*std::cos(2*M_PI*u2);
                    const double z = std::sqrt(u1)*std::sin(2*M_PI*u3);
                    const double w = std::sqrt(u1)*std::cos(2*M_PI*u3);
                    const Vec3 q(x, y, z);
                    rotated = reference+2.0*q.cross(q.cross(reference)+w*reference);
                }
                checksum += rotated[0];
            }
            const auto stop = std::chrono::steady_clock::now();
            if (repeat >= 0)
                timings.push_back(std::chrono::duration<double, std::micro>(stop-start).count());
        }
        std::sort(timings.begin(), timings.end());
        return timings[timings.size()/2]+checksum*0.0;
    };
    const double oldMedian = median(false);
    const double preparedMedian = median(true);
    std::cout << "Orientation generation benchmark: candidates=" << candidates
              << " old_median_us=" << oldMedian
              << " prepared_median_us=" << preparedMedian
              << " speedup=" << oldMedian/preparedMedian << std::endl;
}

static PerturbNonbondedForce* createNonbonded(const vector<double>& charges);

void testGpuProposalOracles() {
    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    PerturbNonbondedForce* nonbonded = createNonbonded({1e-6, -1e-6, 1e-6, -1e-6});
    system.addForce(nonbonded);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(3.9, 2, 2), Vec3(0.1, 2, 2), Vec3(0.2, 2, 2), Vec3(1.0, 2, 2)});
    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(
            nonbonded->getCudaPlatformData(context));
    CudaContext& cu = *data.contexts[0];
    ContextSelector selector(cu);
    std::map<std::string, std::string> defines;
    defines["USE_PERIODIC"] = "1";
    defines["SQRT_ONE_4PI_EPS0"] = "1";
    CUmodule module = cu.createModule(CudaKernelSources::vectorOps+
            CudaRelocationMCKernelSources::relocation_mc, defines);
    CUfunction reorder = cu.getKernel(module, "reorderPositions");
    CUfunction centerKernel = cu.getKernel(module, "computeLigandCenter");
    CUfunction classify = cu.getKernel(module, "classifyMolecules");
    CUfunction sampleInner = cu.getKernel(module, "sampleInnerAnchors");
    CUfunction sampleExternal = cu.getKernel(module, "sampleExternalAnchors");
    CUfunction selectReal = cu.getKernel(module, "selectRealMolecule");
    CUfunction selectCandidate = cu.getKernel(module, "selectCandidateMolecule");

    CudaArray ordered, center, ligandIndices, referenceIndices, classifiedRegions, counts, randoms, anchors, valids;
    ordered.initialize<mm_float4>(cu, cu.getPaddedNumAtoms(), "oracleOrdered");
    center.initialize<mm_float4>(cu, 1, "oracleCenter");
    ligandIndices.initialize<int>(cu, 2, "oracleLigands");
    referenceIndices.initialize<int>(cu, 2, "oracleReferences");
    classifiedRegions.initialize<int>(cu, 2, "oracleClassifiedRegions");
    counts.initialize<int>(cu, 3, "oracleCounts");
    randoms.initialize<mm_float4>(cu, 4, "oracleRandoms");
    anchors.initialize<mm_float4>(cu, 4, "oracleAnchors");
    valids.initialize<int>(cu, 4, "oracleValids");
    ligandIndices.upload(vector<int>{0, 1});
    referenceIndices.upload(vector<int>{2, 3});
    int numAtoms = 4, numCenters = 2, numMoleculesForRegion = 2;
    void* reorderArgs[] = {&numAtoms, &cu.getPosq().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(),
            &ordered.getDevicePointer()};
    cu.executeKernel(reorder, reorderArgs, numAtoms);
    void* centerArgs[] = {&numCenters, &ordered.getDevicePointer(), &ligandIndices.getDevicePointer(),
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
            cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(), &center.getDevicePointer()};
    cu.executeKernel(centerKernel, centerArgs, 1, 1);
    vector<mm_float4> centerHost;
    center.download(centerHost);
    const double centerDistance = std::min(std::fabs((double) centerHost[0].x), std::fabs((double) centerHost[0].x-4.0));
    require(centerDistance < 1e-5 && std::fabs(centerHost[0].y-2.0) < 1e-5,
            "PBC-unwrapped ligand center disagrees with host oracle.");
    float radiusSquared = 0.25f;
    cu.clearBuffer(counts);
    void* classifyArgs[] = {&numMoleculesForRegion, &radiusSquared, &ordered.getDevicePointer(),
            &center.getDevicePointer(), &referenceIndices.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
            cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(),
            cu.getPeriodicBoxVecZPointer(), &classifiedRegions.getDevicePointer(), &counts.getDevicePointer()};
    cu.executeKernel(classify, classifyArgs, numMoleculesForRegion);
    vector<int> classifiedHost, countHost;
    classifiedRegions.download(classifiedHost);
    counts.download(countHost);
    require(classifiedHost == vector<int>({0, 1}) && countHost[0] == 1 && countHost[1] == 1,
            "GPU region classification disagrees with the unique-center host oracle.");

    const vector<mm_float4> innerRandoms = {mm_float4(1, 0, 0, 0), mm_float4(0, 1, 0, 0.125f),
                                            mm_float4(0, 0, 1, 0.5f), mm_float4(1, 1, 1, 0.999f)};
    randoms.upload(innerRandoms);
    int numSamples = 4;
    float radius = 0.5f;
    void* innerArgs[] = {&numSamples, &radius, &center.getDevicePointer(), &randoms.getDevicePointer(),
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
            cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(), &anchors.getDevicePointer()};
    cu.executeKernel(sampleInner, innerArgs, numSamples);
    vector<mm_float4> anchorHost;
    anchors.download(anchorHost);
    for (const mm_float4& anchor : anchorHost) {
        double dx = anchor.x-centerHost[0].x;
        dx -= std::floor(dx/4.0+0.5)*4.0;
        require(dx*dx+(anchor.y-2.0)*(anchor.y-2.0)+(anchor.z-2.0)*(anchor.z-2.0) <= 0.250001,
                "GPU inner anchor escaped the spherical proposal region.");
    }
    const vector<mm_float4> externalRandoms = {mm_float4(0.5f, 0.5f, 0.5f, 0), mm_float4(0.75f, 0.75f, 0.75f, 0),
                                               mm_float4(0.25f, 0.75f, 0.5f, 0), mm_float4(0.5f, 0.25f, 0.75f, 0)};
    randoms.upload(externalRandoms);
    cu.clearBuffer(valids);
    cu.clearBuffer(counts);
    void* externalArgs[] = {&numSamples, &radiusSquared, &center.getDevicePointer(), &randoms.getDevicePointer(),
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
            cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(), &anchors.getDevicePointer(),
            &valids.getDevicePointer(), &counts.getDevicePointer()};
    cu.executeKernel(sampleExternal, externalArgs, numSamples);
    anchors.download(anchorHost);
    vector<int> validHost;
    valids.download(validHost);
    for (int i = 0; i < numSamples; i++) {
        require(validHost[i] == 1, "Deterministic external oracle sample was unexpectedly rejected.");
        double dx = anchorHost[i].x-centerHost[0].x;
        dx -= std::floor(dx/4.0+0.5)*4.0;
        require(dx*dx+(anchorHost[i].y-2.0)*(anchorHost[i].y-2.0)+(anchorHost[i].z-2.0)*(anchorHost[i].z-2.0) >= 0.25,
                "GPU external anchor entered the spherical proposal region.");
    }

    const int n = 4;
    CudaArray regions, energies, selected, scalars, uniforms;
    regions.initialize<int>(cu, n, "oracleRegions");
    energies.initialize<float>(cu, n, "oracleEnergies");
    selected.initialize<mm_float4>(cu, 1, "oracleSelected");
    scalars.initialize<mm_float4>(cu, 1, "oracleScalars");
    uniforms.initialize<float>(cu, n, "oracleUniforms");
    const vector<int> regionValues = {0, 1, 0, 1};
    const vector<float> energyValues = {-1.0f, 0.5f, 2.0f, -0.25f};
    regions.upload(regionValues);
    energies.upload(energyValues);
    int source = 0, target = 1;
    float beta = 0.75f, selectionUniform = 0.37f;
    void* realArgs[] = {const_cast<int*>(&n), &source, &target, &beta, &selectionUniform,
            &regions.getDevicePointer(), &energies.getDevicePointer(),
            &selected.getDevicePointer(), &scalars.getDevicePointer()};
    cu.executeKernel(selectReal, realArgs, 1, 1);
    vector<mm_float4> selectedHost, scalarsHost;
    selected.download(selectedHost);
    scalars.download(scalarsHost);
    const double sourceMax = std::max(beta*energyValues[0], beta*energyValues[2]);
    const double sourceSum = std::exp(beta*energyValues[0]-sourceMax)+std::exp(beta*energyValues[2]-sourceMax);
    const double targetMax = std::max(beta*energyValues[1], beta*energyValues[3]);
    const double targetSum = std::exp(beta*energyValues[1]-targetMax)+std::exp(beta*energyValues[3]-targetMax);
    const int expectedReal = selectionUniform*sourceSum < std::exp(beta*energyValues[0]-sourceMax) ? 0 : 2;
    require((int) selectedHost[0].x == expectedReal, "GPU real inverse-CDF selection disagrees with host oracle.");
    require(std::fabs(scalarsHost[0].x-(sourceMax+std::log(sourceSum))) < 2e-6,
            "GPU source logsumexp disagrees with host oracle.");
    require(std::fabs(scalarsHost[0].y-(targetMax+std::log(targetSum))) < 2e-6,
            "GPU target logsumexp disagrees with host oracle.");

    const vector<float> acceptance = {0.9f, 0.1f, 0.2f, 0.3f};
    uniforms.upload(acceptance);
    float sourceLogSum = 0.7f, targetLogSum = -0.2f, cavityBias = 0.0f, movingBias = 0.0f;
    void* candidateArgs[] = {const_cast<int*>(&n), &beta, &sourceLogSum, &targetLogSum,
            &cavityBias, &movingBias, &uniforms.getDevicePointer(), &energies.getDevicePointer(),
            &selected.getDevicePointer()};
    cu.executeKernel(selectCandidate, candidateArgs, 1, 1);
    selected.download(selectedHost);
    int expectedCandidate = -1;
    float expectedProbability = 0.0f;
    for (int i = 0; i < n; i++) {
        const float scaled = beta*energyValues[i];
        const float maximum = std::max(targetLogSum, scaled);
        const float insertion = maximum+std::log(std::exp(targetLogSum-maximum)+std::exp(scaled-maximum));
        const float probability = std::min(1.0f, std::exp(sourceLogSum-insertion));
        expectedProbability = std::max(expectedProbability, probability);
        if (expectedCandidate < 0 && acceptance[i] < probability) {
            expectedCandidate = i;
            expectedProbability = probability;
        }
    }
    require((int) selectedHost[0].x == expectedCandidate,
            "GPU ordered first-success candidate selection disagrees with host oracle.");
    require(std::fabs(selectedHost[0].z-expectedProbability) < 2e-6,
            "GPU candidate acceptance probability disagrees with host oracle.");
    double firstSuccessMass = 0.0;
    double priorFailure = 1.0;
    for (int i = 0; i < n; i++) {
        const double scaled = beta*energyValues[i];
        const double maximum = std::max((double) targetLogSum, scaled);
        const double insertion = maximum+std::log(std::exp(targetLogSum-maximum)+std::exp(scaled-maximum));
        const double probability = std::min(1.0, std::exp(sourceLogSum-insertion));
        firstSuccessMass += priorFailure*probability;
        priorFailure *= 1.0-probability;
    }
    require(std::fabs(firstSuccessMass+priorFailure-1.0) < 1e-12,
            "Ordered first-success probabilities and no-success probability do not normalize.");
    struct DirectionCase { int source, other; double expected; };
    const DirectionCase directionCases[] = {
        {0, 0, 0.0}, {0, 1, 0.0}, {1, 0, 1.0}, {1, 1, 0.5}, {2, 0, 1.0}, {2, 3, 0.5}
    };
    for (const DirectionCase& test : directionCases)
        require(CudaCalcRelocationMCKernel::computeProposalProbabilityForTesting(test.source, test.other) == test.expected,
                "Direction probability boundary table disagrees with the proposal helper.");
    const double forwardDirection = CudaCalcRelocationMCKernel::computeProposalProbabilityForTesting(2, 0);
    const double reverseDirection = CudaCalcRelocationMCKernel::computeProposalProbabilityForTesting(1, 1);
    require(std::fabs(std::log(reverseDirection/forwardDirection)-std::log(0.5)) < 1e-12,
            "The corrected reverse/forward direction-bias sign failed at an empty-region boundary.");
}

void testMtmGpuOracles() {
    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    PerturbNonbondedForce* nonbonded = createNonbonded({1e-6, -1e-6, 1e-6, -1e-6});
    system.addForce(nonbonded);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(1, 1, 1), Vec3(2, 1, 1), Vec3(1, 2, 1), Vec3(1, 1, 2)});
    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(
            nonbonded->getCudaPlatformData(context));
    CudaContext& cu = *data.contexts[0];
    ContextSelector selector(cu);
    std::map<std::string, std::string> defines;
    defines["USE_PERIODIC"] = "1";
    defines["SQRT_ONE_4PI_EPS0"] = "1";
    defines["MTM_BLOCK_SIZE"] = "256";
    CUmodule module = cu.createModule(CudaKernelSources::vectorOps+
            CudaRelocationMCKernelSources::relocation_mc, defines);
    CUfunction generateMtm = cu.getKernel(module, "generateMtmCandidatePositions");
    CUfunction selectMtmReal = cu.getKernel(module, "selectMtmRealMolecule");
    CUfunction selectMtm = cu.getKernel(module, "selectMtmCandidate");
    CUfunction finalSums = cu.getKernel(module, "computeMtmFinalSums");

    // A single kernel invocation generates independent whole-box anchors and
    // Haar rotations of the actual selected D conformation.  Verify rigid
    // geometry, SO(3) moments, deterministic counter RNG, and random region counts.
    const int numHaarCandidates = 8192;
    const int numSites = 3;
    int referenceSite = 0;
    float sphereRadiusSquared = 1.0f;
    unsigned long long seed = 0x123456789ABCDEF0ULL;
    unsigned long long attempt = 17;
    CudaArray ligandCenter, selectedDeleted, haarAnchors, haarRegions, haarPositions;
    ligandCenter.initialize<mm_float4>(cu, 1, "mtmOracleLigandCenter");
    selectedDeleted.initialize<mm_float4>(cu, numSites, "mtmOracleSelectedDeleted");
    haarAnchors.initialize<mm_float4>(cu, numHaarCandidates, "mtmOracleHaarAnchors");
    haarRegions.initialize<int>(cu, numHaarCandidates, "mtmOracleHaarRegions");
    haarPositions.initialize<mm_float4>(cu, numHaarCandidates*numSites, "mtmOracleHaarPositions");
    ligandCenter.upload(vector<mm_float4>{mm_float4(2, 2, 2, 0)});
    // The first bond crosses x PBC.  MTM must minimum-image unwrap it before rotation.
    selectedDeleted.upload(vector<mm_float4>{mm_float4(3.9f, 2.0f, 2.0f, 0),
                                              mm_float4(0.1f, 2.0f, 2.0f, 0),
                                              mm_float4(3.9f, 2.3f, 2.0f, 0)});
    void* generateArgs[] = {const_cast<int*>(&numHaarCandidates), const_cast<int*>(&numSites),
            &referenceSite, &sphereRadiusSquared, &seed, &attempt,
            &ligandCenter.getDevicePointer(), &selectedDeleted.getDevicePointer(),
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &haarAnchors.getDevicePointer(), &haarRegions.getDevicePointer(),
            &haarPositions.getDevicePointer()};
    cu.executeKernel(generateMtm, generateArgs, numHaarCandidates);
    vector<mm_float4> anchorValues, positionValues;
    vector<int> regionValues;
    haarAnchors.download(anchorValues);
    haarPositions.download(positionValues);
    haarRegions.download(regionValues);
    vector<mm_float4> firstAnchors = anchorValues;
    vector<mm_float4> firstPositions = positionValues;
    vector<int> firstRegions = regionValues;
    cu.executeKernel(generateMtm, generateArgs, numHaarCandidates);
    haarAnchors.download(anchorValues);
    haarPositions.download(positionValues);
    haarRegions.download(regionValues);
    require(anchorValues.size() == firstAnchors.size() && positionValues.size() == firstPositions.size() &&
            regionValues == firstRegions, "MTM counter-based candidate generation changed for the same key.");
    for (int i = 0; i < numHaarCandidates; i++) {
        require(anchorValues[i].x == firstAnchors[i].x && anchorValues[i].y == firstAnchors[i].y &&
                anchorValues[i].z == firstAnchors[i].z, "MTM anchor RNG is not bitwise deterministic.");
        for (int site = 0; site < numSites; site++) {
            const mm_float4& first = firstPositions[i*numSites+site];
            const mm_float4& second = positionValues[i*numSites+site];
            require(first.x == second.x && first.y == second.y && first.z == second.z,
                    "MTM Haar rotation is not bitwise deterministic.");
        }
    }
    double firstMoment[3] = {0.0, 0.0, 0.0};
    double secondMoment[3] = {0.0, 0.0, 0.0};
    double crossMoment[3] = {0.0, 0.0, 0.0};
    int innerCount = 0;
    for (int candidate = 0; candidate < numHaarCandidates; candidate++) {
        const mm_float4& anchor = anchorValues[candidate];
        const mm_float4& site0 = positionValues[candidate*numSites];
        const mm_float4& site1 = positionValues[candidate*numSites+1];
        const mm_float4& site2 = positionValues[candidate*numSites+2];
        require(std::fabs(site0.x-anchor.x) < 2e-6 && std::fabs(site0.y-anchor.y) < 2e-6 &&
                std::fabs(site0.z-anchor.z) < 2e-6, "MTM reference site does not equal its whole-box anchor.");
        const Vec3 bond1(site1.x-site0.x, site1.y-site0.y, site1.z-site0.z);
        const Vec3 bond2(site2.x-site0.x, site2.y-site0.y, site2.z-site0.z);
        requireClose(bond1.dot(bond1), 0.04, 3e-5, "Haar rotation changed the PBC-unwrapped first bond length.");
        requireClose(bond2.dot(bond2), 0.09, 3e-5, "Haar rotation changed the second bond length.");
        requireClose(bond1.dot(bond2), 0.0, 3e-5, "Haar rotation changed the selected D internal angle.");
        const Vec3 unit = bond1*(1.0/0.2);
        for (int axis = 0; axis < 3; axis++) {
            firstMoment[axis] += unit[axis];
            secondMoment[axis] += unit[axis]*unit[axis];
        }
        crossMoment[0] += unit[0]*unit[1];
        crossMoment[1] += unit[0]*unit[2];
        crossMoment[2] += unit[1]*unit[2];
        innerCount += regionValues[candidate] == 0;
    }
    for (int axis = 0; axis < 3; axis++) {
        require(std::fabs(firstMoment[axis]/numHaarCandidates) < 0.025,
                "Haar orientation first moment is inconsistent with zero.");
        require(std::fabs(secondMoment[axis]/numHaarCandidates-1.0/3.0) < 0.025,
                "Haar orientation second moment is inconsistent with isotropy.");
        require(std::fabs(crossMoment[axis]/numHaarCandidates) < 0.025,
                "Haar orientation cross moment is inconsistent with isotropy.");
    }
    const double expectedInnerFraction = (4.0*M_PI/3.0)/(4.0*4.0*4.0);
    require(std::fabs((double) innerCount/numHaarCandidates-expectedInnerFraction) < 0.012,
            "Whole-box candidate region count is inconsistent with the spherical volume fraction.");

    // Energy-biased source selection uses the same cooperative stable reduction
    // as production and a separate counter stream from anchors and candidates.
    const int numSourceMolecules = 4;
    int sourceRegionForSelection = 0;
    float sourceBeta = 0.75f;
    unsigned long long sourceAttempt = 31;
    CudaArray sourceRegions, sourceEnergies, sourceSelected, sourceReduction;
    sourceRegions.initialize<int>(cu, numSourceMolecules, "mtmOracleSourceRegions");
    sourceEnergies.initialize<float>(cu, numSourceMolecules, "mtmOracleSourceEnergies");
    sourceSelected.initialize<mm_float4>(cu, 1, "mtmOracleSourceSelected");
    sourceReduction.initialize<mm_float4>(cu, 1, "mtmOracleSourceReduction");
    sourceRegions.upload(vector<int>{0, 1, 0, 1});
    const vector<float> sourceEnergyValues = {-1.0f, 0.5f, 2.0f, -0.25f};
    sourceEnergies.upload(sourceEnergyValues);
    auto runSourceSelection = [&]() {
        void* args[] = {const_cast<int*>(&numSourceMolecules), &sourceRegionForSelection, &sourceBeta,
                &seed, &sourceAttempt, &sourceRegions.getDevicePointer(), &sourceEnergies.getDevicePointer(),
                &sourceSelected.getDevicePointer(), &sourceReduction.getDevicePointer()};
        cu.executeKernel(selectMtmReal, args, 256, 256);
        vector<mm_float4> selectedHost, reductionHost;
        sourceSelected.download(selectedHost);
        sourceReduction.download(reductionHost);
        return std::make_pair(selectedHost[0], reductionHost[0]);
    };
    const auto sourceFirst = runSourceSelection();
    const auto sourceSecond = runSourceSelection();
    const double sourceMaximum = std::max(sourceBeta*sourceEnergyValues[0], sourceBeta*sourceEnergyValues[2]);
    const double sourceLogSum = sourceMaximum+std::log(
            std::exp(sourceBeta*sourceEnergyValues[0]-sourceMaximum)+
            std::exp(sourceBeta*sourceEnergyValues[2]-sourceMaximum));
    require(sourceFirst.first.x == sourceSecond.first.x && sourceFirst.first.y == sourceSecond.first.y &&
            sourceFirst.first.z == sourceSecond.first.z, "MTM source Gumbel-Max is not deterministic.");
    const int selectedSource = (int) sourceFirst.first.x;
    require(selectedSource == 0 || selectedSource == 2,
            "MTM source selection chose a molecule outside the source region.");
    requireClose(sourceFirst.second.x, sourceLogSum, 3e-6,
            "MTM source logsumexp disagrees with the CPU oracle.");
    requireClose(sourceFirst.first.z,
            std::exp(sourceBeta*sourceEnergyValues[selectedSource]-sourceLogSum), 3e-6,
            "MTM source selection probability disagrees with its normalized weight.");
    for (float invalidEnergy : {std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        sourceEnergies.upload(vector<float>{sourceEnergyValues[0], invalidEnergy,
                                            sourceEnergyValues[2], sourceEnergyValues[3]});
        const auto invalidSource = runSourceSelection();
        require(invalidSource.first.x == -1.0f && invalidSource.first.w == 0.0f &&
                std::isinf(invalidSource.second.x) && invalidSource.second.x < 0.0f,
                "A nonfinite all-real source partition did not fail closed.");
    }
    sourceEnergies.upload(sourceEnergyValues);
    sourceRegions.upload(vector<int>{1, 1, 1, 1});
    const auto emptySource = runSourceSelection();
    require(emptySource.first.x == -1.0f && emptySource.first.w == 0.0f &&
            std::isinf(emptySource.second.x) && emptySource.second.x < 0.0f,
            "An empty MTM source region did not return before indexing a molecule.");

    // A large common log-weight must not absorb the O(1) Gumbel perturbation.
    // Subtract the analytically known offset when comparing logsumexp outputs.
    const float commonEnergyOffset = 524288.0f;
    sourceRegions.upload(vector<int>{0, 0, 1, 1});
    const vector<float> equalSourceEnergies = {0.0f, 0.0f, 3.0f, -2.0f};
    const vector<float> shiftedSourceEnergies = {commonEnergyOffset, commonEnergyOffset,
                                                 commonEnergyOffset+3.0f, commonEnergyOffset-2.0f};
    bool selectedBothSourceIndices[2] = {false, false};
    for (int draw = 0; draw < 128; draw++) {
        sourceAttempt = 5000+draw;
        sourceEnergies.upload(equalSourceEnergies);
        const auto baseline = runSourceSelection();
        sourceEnergies.upload(shiftedSourceEnergies);
        const auto shifted = runSourceSelection();
        require(baseline.first.x == shifted.first.x,
                "A large common source energy offset changed Gumbel-Max selection.");
        requireClose(shifted.first.z, baseline.first.z, 0.01,
                "A large common source energy offset changed the normalized probability.");
        requireClose(shifted.second.x-sourceBeta*commonEnergyOffset, baseline.second.x, 0.02,
                "Source logsumexp did not shift by the expected common beta*energy offset.");
        const int chosen = (int) baseline.first.x;
        require(chosen == 0 || chosen == 1, "Equal-weight source test selected an ineligible molecule.");
        selectedBothSourceIndices[chosen] = true;
    }
    require(selectedBothSourceIndices[0] && selectedBothSourceIndices[1],
            "Large-offset source regression did not exercise both equal-weight outcomes.");

    // The target categorical uses untruncated exp(-beta*u) weights and one
    // counter-based Gumbel-Max draw.  Check stable logsumexp, determinism, and frequencies.
    const int numCandidates = 4;
    int targetRegion = 1;
    float beta = 1.0f;
    CudaArray candidateRegions, candidateEnergies, sourceScalars, selectedData, logSums;
    candidateRegions.initialize<int>(cu, numCandidates, "mtmOracleCandidateRegions");
    candidateEnergies.initialize<float>(cu, numCandidates, "mtmOracleCandidateEnergies");
    sourceScalars.initialize<mm_float4>(cu, 1, "mtmOracleSourceScalars");
    selectedData.initialize<mm_float4>(cu, 1, "mtmOracleSelectedData");
    logSums.initialize<mm_float4>(cu, 1, "mtmOracleLogSums");
    candidateRegions.upload(vector<int>{1, 1, 1, 0});
    const vector<float> categoricalEnergies = {(float)-std::log(0.1), (float)-std::log(0.2),
                                                (float)-std::log(0.7), -100.0f};
    candidateEnergies.upload(categoricalEnergies);
    sourceScalars.upload(vector<mm_float4>{mm_float4(1.2345f, 0, 0, 0)});
    auto runSelection = [&](unsigned long long drawAttempt) {
        void* args[] = {const_cast<int*>(&numCandidates), &targetRegion, &beta, &seed, &drawAttempt,
                &candidateRegions.getDevicePointer(), &candidateEnergies.getDevicePointer(),
                &sourceScalars.getDevicePointer(), &selectedData.getDevicePointer(), &logSums.getDevicePointer()};
        cu.executeKernel(selectMtm, args, 256, 256);
        vector<mm_float4> selectedHost, sumHost;
        selectedData.download(selectedHost);
        logSums.download(sumHost);
        return std::make_pair(selectedHost[0], sumHost[0]);
    };
    const auto deterministicFirst = runSelection(41);
    const auto deterministicSecond = runSelection(41);
    require(deterministicFirst.first.x == deterministicSecond.first.x &&
            deterministicFirst.first.y == deterministicSecond.first.y &&
            deterministicFirst.first.z == deterministicSecond.first.z &&
            deterministicFirst.second.z == deterministicSecond.second.z,
            "MTM Gumbel-Max selection changed for the same counter key.");
    requireClose(deterministicFirst.second.x, 1.2345, 2e-6, "MTM selection did not preserve log Z_S.");
    requireClose(deterministicFirst.second.z, 0.0, 3e-6, "MTM forward logsumexp disagrees with exact weights.");
    const int categoricalDraws = 1024;
    int selectedCounts[3] = {0, 0, 0};
    for (int draw = 0; draw < categoricalDraws; draw++) {
        const auto value = runSelection(1000+draw);
        const int selected = (int) value.first.x;
        require(selected >= 0 && selected < 3 && value.first.w == 1.0f,
                "MTM Gumbel-Max selected an ineligible source-region candidate.");
        selectedCounts[selected]++;
        const double expectedProbability = selected == 0 ? 0.1 : (selected == 1 ? 0.2 : 0.7);
        requireClose(value.first.z, expectedProbability, 4e-5,
                "MTM reported categorical probability disagrees with normalized weight.");
    }
    const double expectedFrequencies[3] = {0.1, 0.2, 0.7};
    for (int candidate = 0; candidate < 3; candidate++)
        require(std::fabs((double) selectedCounts[candidate]/categoricalDraws-expectedFrequencies[candidate]) < 0.05,
                "MTM Gumbel-Max frequencies disagree with the categorical distribution.");

    candidateRegions.upload(vector<int>{1, 1, 0, 0});
    const vector<float> equalCandidateEnergies = {0.0f, 0.0f, 5.0f, -3.0f};
    const vector<float> shiftedCandidateEnergies = {commonEnergyOffset, commonEnergyOffset,
                                                    commonEnergyOffset+5.0f, commonEnergyOffset-3.0f};
    bool selectedBothCandidateIndices[2] = {false, false};
    for (int draw = 0; draw < 128; draw++) {
        candidateEnergies.upload(equalCandidateEnergies);
        const auto baseline = runSelection(6000+draw);
        candidateEnergies.upload(shiftedCandidateEnergies);
        const auto shifted = runSelection(6000+draw);
        require(baseline.first.x == shifted.first.x,
                "A large common candidate energy offset changed Gumbel-Max selection.");
        requireClose(shifted.first.z, baseline.first.z, 0.01,
                "A large common candidate energy offset changed the normalized probability.");
        requireClose(shifted.second.z+beta*commonEnergyOffset, baseline.second.z, 0.02,
                "Candidate logsumexp did not shift by the expected common -beta*energy offset.");
        const int chosen = (int) baseline.first.x;
        require(chosen == 0 || chosen == 1, "Equal-weight candidate test selected an ineligible pose.");
        selectedBothCandidateIndices[chosen] = true;
    }
    require(selectedBothCandidateIndices[0] && selectedBothCandidateIndices[1],
            "Large-offset candidate regression did not exercise both equal-weight outcomes.");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const int safetyCapacity = 8192;
    CudaArray safetyRegions, safetyEnergies;
    safetyRegions.initialize<int>(cu, safetyCapacity, "mtmSafetyCandidateRegions");
    safetyEnergies.initialize<float>(cu, safetyCapacity, "mtmSafetyCandidateEnergies");
    auto runSafetySelection = [&](int count, const vector<int>& regions, const vector<float>& energies,
            float selectionBeta, unsigned long long drawAttempt) {
        require(count >= 0 && count <= safetyCapacity && (int) regions.size() == count &&
                (int) energies.size() == count, "Invalid MTM selection-safety test input.");
        vector<int> regionStorage(safetyCapacity, 0);
        vector<float> energyStorage(safetyCapacity, 0.0f);
        std::copy(regions.begin(), regions.end(), regionStorage.begin());
        std::copy(energies.begin(), energies.end(), energyStorage.begin());
        safetyRegions.upload(regionStorage);
        safetyEnergies.upload(energyStorage);
        void* args[] = {&count, &targetRegion, &selectionBeta, &seed, &drawAttempt,
                &safetyRegions.getDevicePointer(), &safetyEnergies.getDevicePointer(),
                &sourceScalars.getDevicePointer(), &selectedData.getDevicePointer(), &logSums.getDevicePointer()};
        cu.executeKernel(selectMtm, args, 256, 256);
        vector<mm_float4> selectedHost, sumHost;
        selectedData.download(selectedHost);
        logSums.download(sumHost);
        return std::make_pair(selectedHost[0], sumHost[0]);
    };
    auto requireEmptyForwardPool = [&](const std::pair<mm_float4, mm_float4>& decision,
            const char* message) {
        require(decision.first.x == -1.0f && decision.first.y == 0.0f &&
                decision.first.z == 0.0f && decision.first.w == 0.0f &&
                std::isinf(decision.second.z) && decision.second.z < 0.0f, message);
    };

    const int safetyK = 16;
    requireEmptyForwardPool(runSafetySelection(safetyK, vector<int>(safetyK, 0),
            vector<float>(safetyK, 0.0f), 1.0f, 77),
            "K=16 empty target pool did not return safely before candidate indexing.");
    int safetyAttempt = 78;
    for (float zeroWeightEnergy : {nan, inf, -inf})
        requireEmptyForwardPool(runSafetySelection(safetyK, vector<int>(safetyK, 1),
                vector<float>(safetyK, zeroWeightEnergy), 1.0f, safetyAttempt++),
                "K=16 all-nonfinite zero-weight pool did not return safely.");

    vector<int> targetRegions(safetyK, 1);
    for (float zeroWeightEnergy : {nan, inf, -inf, -std::numeric_limits<float>::max()}) {
        vector<float> oneZeroWeight(safetyK, 0.0f);
        oneZeroWeight[3] = zeroWeightEnergy;
        const float selectionBeta = std::isfinite(zeroWeightEnergy) ? 2.0f : 1.0f;
        const auto oneSkipped = runSafetySelection(
                safetyK, targetRegions, oneZeroWeight, selectionBeta, safetyAttempt++);
        require(oneSkipped.first.x >= 0.0f && oneSkipped.first.x < safetyK &&
                oneSkipped.first.x != 3.0f && oneSkipped.first.w == 1.0f,
                "A single nonfinite-weight MTM candidate was not skipped.");
        requireClose(oneSkipped.second.z, std::log(15.0), 4e-6,
                "A single nonfinite-weight MTM candidate contributed to the forward weight sum.");
        requireClose(oneSkipped.first.z, 1.0/15.0, 4e-6,
                "A single nonfinite-weight MTM candidate changed the normalized finite weight.");
    }

    vector<float> finiteUnderflow(safetyK, 0.0f);
    finiteUnderflow[5] = std::numeric_limits<float>::max();
    const auto finiteNegativeInfinityLogWeight = runSafetySelection(
            safetyK, targetRegions, finiteUnderflow, 2.0f, 80);
    require(finiteNegativeInfinityLogWeight.first.x >= 0.0f &&
            finiteNegativeInfinityLogWeight.first.x < safetyK &&
            finiteNegativeInfinityLogWeight.first.x != 5.0f,
            "A finite energy with log weight -Inf was not assigned exact zero weight.");
    requireClose(finiteNegativeInfinityLogWeight.second.z, std::log(15.0), 4e-6,
            "A finite energy with log weight -Inf contributed to the forward weight sum.");

    requireEmptyForwardPool(runSafetySelection(0, {}, {}, 1.0f, safetyAttempt++),
            "K=0 MTM selection did not return safely before candidate indexing.");

    vector<int> k8192Regions(safetyCapacity, 1);
    vector<float> k8192Energies(safetyCapacity, inf);
    requireEmptyForwardPool(runSafetySelection(safetyCapacity, k8192Regions,
            k8192Energies, 1.0f, 85),
            "K=8192 all-+Inf exact-zero-weight pool did not return safely.");
    k8192Energies[safetyCapacity/2] = 0.0f;
    const auto k8192SingleFinite = runSafetySelection(
            safetyCapacity, k8192Regions, k8192Energies, 1.0f, 86);
    require(k8192SingleFinite.first.x == safetyCapacity/2 &&
            k8192SingleFinite.first.z == 1.0f && k8192SingleFinite.first.w == 1.0f,
            "K=8192 pool with one positive finite weight did not select its sole supported candidate.");
    requireClose(k8192SingleFinite.second.z, 0.0, 2e-6,
            "K=8192 pool with one finite weight reported the wrong log weight sum.");
    int oneCandidateCount = 1;
    candidateRegions.upload(vector<int>{1, 0, 0, 0});
    candidateEnergies.upload(vector<float>{2.5f, 0.0f, 0.0f, 0.0f});
    unsigned long long oneCandidateAttempt = 80;
    void* oneCandidateArgs[] = {&oneCandidateCount, &targetRegion, &beta, &seed, &oneCandidateAttempt,
            &candidateRegions.getDevicePointer(), &candidateEnergies.getDevicePointer(),
            &sourceScalars.getDevicePointer(), &selectedData.getDevicePointer(), &logSums.getDevicePointer()};
    cu.executeKernel(selectMtm, oneCandidateArgs, 256, 256);
    vector<mm_float4> oneCandidateSelected, oneCandidateSums;
    selectedData.download(oneCandidateSelected);
    logSums.download(oneCandidateSums);
    require(oneCandidateSelected[0].x == 0.0f && oneCandidateSelected[0].z == 1.0f &&
            oneCandidateSelected[0].w == 1.0f, "K=1 MTM categorical did not select its sole valid trial.");
    requireClose(oneCandidateSums[0].z, -2.5, 3e-6,
            "K=1 MTM forward log weight did not reduce to -beta*u.");

    // Synthetic exact proposed-state source scores exercise Z_T(y), W_r, the
    // old-pose term, reciprocal pair corrections, isolated reciprocal diagnostic, and the final log ratio.
    const int numMolecules = 3;
    const int finalCandidates = 4;
    const int atomsPerMolecule = 1;
    int selectedMolecule = 0;
    int sourceRegion = 0;
    int proposedTargetRegion = 1;
    int deletedRealPotentialOffset = 1;
    float finalBeta = 0.5f;
    CudaArray moleculeRegions, finalCandidateRegions, realEnergies, realParameters,
            finalCandidateEnergies, directCorrections, deletedPotentials, selectedPotentials,
            finalSelectedData, proposedScores, finalLogSums, energyScalars;
    moleculeRegions.initialize<int>(cu, numMolecules, "mtmOracleMoleculeRegions");
    finalCandidateRegions.initialize<int>(cu, finalCandidates, "mtmOracleFinalCandidateRegions");
    realEnergies.initialize<float>(cu, numMolecules, "mtmOracleRealEnergies");
    realParameters.initialize<mm_float4>(cu, numMolecules, "mtmOracleRealParameters");
    finalCandidateEnergies.initialize<float>(cu, finalCandidates, "mtmOracleFinalCandidateEnergies");
    directCorrections.initialize<float>(cu, numMolecules, "mtmOracleDirectCorrections");
    deletedPotentials.initialize<float>(cu, 1+numMolecules, "mtmOracleDeletedPotentials");
    selectedPotentials.initialize<float>(cu, 1+numMolecules, "mtmOracleSelectedPotentials");
    finalSelectedData.initialize<mm_float4>(cu, 1, "mtmOracleFinalSelectedData");
    proposedScores.initialize<float>(cu, numMolecules, "mtmOracleProposedScores");
    finalLogSums.initialize<mm_float4>(cu, 1, "mtmOracleFinalLogSums");
    energyScalars.initialize<mm_float4>(cu, 1, "mtmOracleEnergyScalars");
    moleculeRegions.upload(vector<int>{0, 1, 1});
    finalCandidateRegions.upload(vector<int>{1, 0, 0, 1});
    realEnergies.upload(vector<float>{1.0f, 2.0f, -1.0f});
    realParameters.upload(vector<mm_float4>{mm_float4(1.0f, 0, 0, 0),
                                            mm_float4(-0.5f, 0, 0, 0),
                                            mm_float4(0.25f, 0, 0, 0)});
    const vector<float> finalEnergyValues = {-0.5f, 1.5f, inf, 2.0f};
    finalCandidateEnergies.upload(finalEnergyValues);
    directCorrections.upload(vector<float>{0.0f, 0.3f, -0.2f});
    deletedPotentials.upload(vector<float>{0.2f, 0.0f, 0.4f, -0.6f});
    selectedPotentials.upload(vector<float>{0.6f, 0.0f, -0.2f, 0.8f});
    finalSelectedData.upload(vector<mm_float4>{mm_float4(0, -0.5f, 0, 1)});
    const auto logAdd = [](double first, double second) {
        const double maximum = std::max(first, second);
        return maximum+std::log(std::exp(first-maximum)+std::exp(second-maximum));
    };
    const double expectedLogWf = logAdd(-finalBeta*finalEnergyValues[0], -finalBeta*finalEnergyValues[3]);
    finalLogSums.upload(vector<mm_float4>{mm_float4(1.25f, 0.0f, (float) expectedLogWf, 0.0f)});
    const float deletedEnergy = 0.7f;
    energyScalars.upload(vector<mm_float4>{mm_float4(deletedEnergy, 0, 0, 0)});
    void* finalArgs[] = {const_cast<int*>(&numMolecules), const_cast<int*>(&finalCandidates),
            const_cast<int*>(&atomsPerMolecule), &selectedMolecule, &sourceRegion,
            &proposedTargetRegion, &finalBeta, &moleculeRegions.getDevicePointer(),
            &finalCandidateRegions.getDevicePointer(), &realEnergies.getDevicePointer(),
            &realParameters.getDevicePointer(), &finalCandidateEnergies.getDevicePointer(),
            &directCorrections.getDevicePointer(), &deletedPotentials.getDevicePointer(),
            &deletedRealPotentialOffset, &selectedPotentials.getDevicePointer(),
            &finalSelectedData.getDevicePointer(), &proposedScores.getDevicePointer(),
            &finalLogSums.getDevicePointer(), &energyScalars.getDevicePointer()};
    cu.executeKernel(finalSums, finalArgs, 256, 256);
    vector<mm_float4> finalSumHost, energyHost;
    vector<float> proposedScoreHost;
    finalLogSums.download(finalSumHost);
    energyScalars.download(energyHost);
    proposedScores.download(proposedScoreHost);
    const double selectedSourceScore = -0.5+0.6;
    const double targetScore1 = 2.0+0.3+(-0.5)*(-0.2-0.4);
    const double targetScore2 = -1.0-0.2+0.25*(0.8-(-0.6));
    const double expectedLogZT = logAdd(logAdd(finalBeta*selectedSourceScore,
            finalBeta*targetScore1), finalBeta*targetScore2);
    const double expectedLogWr = logAdd(-finalBeta*deletedEnergy, -finalBeta*1.5);
    const double expectedIsolatedReciprocalDelta = 0.5*(0.6-0.2);
    requireClose(finalSumHost[0].x, 1.25, 3e-6, "MTM final reduction changed log Z_S.");
    requireClose(finalSumHost[0].y, expectedLogZT, 4e-6, "MTM exact proposed-state log Z_T disagrees with CPU oracle.");
    requireClose(finalSumHost[0].z, expectedLogWf, 4e-6, "MTM forward log weight sum disagrees with CPU oracle.");
    requireClose(finalSumHost[0].w, expectedLogWr, 4e-6, "MTM reverse log weight sum omitted old D or source trials.");
    requireClose(energyHost[0].x, deletedEnergy, 2e-6, "MTM final reduction changed u_D.");
    requireClose(energyHost[0].y, -0.5, 2e-6, "MTM final reduction changed u_C.");
    requireClose(energyHost[0].z, selectedSourceScore, 3e-6,
            "MTM proposed source score s_C has the wrong isolated reciprocal convention.");
    require(std::isnan(proposedScoreHost[0]), "MTM proposed score output did not mask the moved source molecule.");
    requireClose(proposedScoreHost[1], targetScore1, 3e-6,
            "MTM proposed score output disagrees for the first old target molecule.");
    requireClose(proposedScoreHost[2], targetScore2, 3e-6,
            "MTM proposed score output disagrees for the second old target molecule.");
    requireClose(energyHost[0].w, expectedIsolatedReciprocalDelta, 3e-6,
            "MTM isolated reciprocal delta disagrees with the CPU component oracle.");
    const double movingBias = std::log(0.5);
    const double sourceScoreD = 1.0;
    // The validated physical delta adds the isolated reciprocal contribution
    // to uC-uD and includes it in the final acceptance ratio.
    const double gpuLogRatio = movingBias+finalSumHost[0].x-finalSumHost[0].y+
            finalBeta*(energyHost[0].z-sourceScoreD)+finalSumHost[0].z-finalSumHost[0].w-
            finalBeta*energyHost[0].w;
    const double expectedLogRatio = movingBias+1.25-expectedLogZT+
            finalBeta*(selectedSourceScore-sourceScoreD)+expectedLogWf-expectedLogWr-
            finalBeta*expectedIsolatedReciprocalDelta;
    requireClose(gpuLogRatio, expectedLogRatio, 5e-6, "MTM final detailed-balance log ratio disagrees with CPU oracle.");

    for (float invalidReverseEnergy : {nan, -inf}) {
        vector<float> invalidReverseEnergies = finalEnergyValues;
        invalidReverseEnergies[2] = invalidReverseEnergy;
        finalCandidateEnergies.upload(invalidReverseEnergies);
        finalLogSums.upload(vector<mm_float4>{mm_float4(1.25f, 0.0f, (float) expectedLogWf, 0.0f)});
        energyScalars.upload(vector<mm_float4>{mm_float4(deletedEnergy, 0, 0, 0)});
        cu.executeKernel(finalSums, finalArgs, 256, 256);
        finalLogSums.download(finalSumHost);
        requireClose(finalSumHost[0].w, expectedLogWr, 4e-6,
                "A nonfinite reverse candidate was not assigned zero weight.");
    }
    finalCandidateEnergies.upload(finalEnergyValues);
    directCorrections.upload(vector<float>{0.0f, nan, -0.2f});
    finalLogSums.upload(vector<mm_float4>{mm_float4(1.25f, 0.0f, (float) expectedLogWf, 0.0f)});
    energyScalars.upload(vector<mm_float4>{mm_float4(deletedEnergy, 0, 0, 0)});
    cu.executeKernel(finalSums, finalArgs, 256, 256);
    finalLogSums.download(finalSumHost);
    require(std::isnan(finalSumHost[0].y),
            "A nonfinite proposed-state source partition did not fail closed.");
    directCorrections.upload(vector<float>{0.0f, 0.3f, -0.2f});
    energyScalars.upload(vector<mm_float4>{mm_float4(nan, 0, 0, 0)});
    finalLogSums.upload(vector<mm_float4>{mm_float4(1.25f, 0.0f, (float) expectedLogWf, 0.0f)});
    cu.executeKernel(finalSums, finalArgs, 256, 256);
    finalLogSums.download(finalSumHost);
    require(std::isnan(finalSumHost[0].w), "A nonfinite old-pose reverse weight did not fail closed.");
}

void testGpuNonfiniteProposalsFailClosed() {
    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    PerturbNonbondedForce* nonbonded = createNonbonded({1e-6, -1e-6, 1e-6, -1e-6});
    system.addForce(nonbonded);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(3.9, 2, 2), Vec3(0.1, 2, 2), Vec3(0.2, 2, 2), Vec3(1.0, 2, 2)});
    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(
            nonbonded->getCudaPlatformData(context));
    CudaContext& cu = *data.contexts[0];
    ContextSelector selector(cu);
    std::map<std::string, std::string> defines;
    defines["USE_PERIODIC"] = "1";
    defines["SQRT_ONE_4PI_EPS0"] = "1";
    CUmodule module = cu.createModule(CudaKernelSources::vectorOps+
            CudaRelocationMCKernelSources::relocation_mc, defines);
    CUfunction selectReal = cu.getKernel(module, "selectRealMolecule");
    CUfunction selectCandidate = cu.getKernel(module, "selectCandidateMolecule");
    CUfunction assembleExact = cu.getKernel(module, "assembleExactPmeEnergies");
    CUfunction assembleMtm = cu.getKernel(module, "assembleMtmCandidateEnergies");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const int n = 4;
    CudaArray energies, uniforms, selected;
    energies.initialize<float>(cu, n, "nonfiniteCandidateEnergies");
    uniforms.initialize<float>(cu, n, "nonfiniteCandidateUniforms");
    selected.initialize<mm_float4>(cu, 1, "nonfiniteCandidateSelected");
    uniforms.upload(vector<float>(n, 0.0f));
    auto runCandidateSelection = [&](const vector<float>& values, float beta, float sourceLogSum,
            float targetLogSum, float cavityBias, float movingBias) {
        energies.upload(values);
        void* args[] = {const_cast<int*>(&n), &beta, &sourceLogSum, &targetLogSum,
                &cavityBias, &movingBias, &uniforms.getDevicePointer(), &energies.getDevicePointer(),
                &selected.getDevicePointer()};
        cu.executeKernel(selectCandidate, args, 1, 1);
        vector<mm_float4> result;
        selected.download(result);
        return result[0];
    };

    const vector<float> mixedEnergies = {nan, 0.25f, inf, -inf};
    mm_float4 decision = runCandidateSelection(mixedEnergies, 0.75f, 0.7f, -0.2f, 0.0f, 0.0f);
    require((int) decision.x == 1 && decision.w == 1.0f && std::isfinite(decision.y) &&
            std::isfinite(decision.z) && decision.z > 0.0f && decision.z <= 1.0f,
            "CUDA candidate selection did not skip NaN/+Inf/-Inf before selecting the finite candidate.");
    decision = runCandidateSelection({-1.0f, 0.5f, 2.0f, -0.25f}, 0.75f,
            1.0e30f, -0.2f, 0.0f, 0.0f);
    require(decision.x == 0.0f && decision.z == 1.0f && decision.w == 1.0f,
            "A finite favorable log probability did not saturate at one without exp overflow.");
    decision = runCandidateSelection({nan, inf, -inf, nan}, 0.75f, 0.7f, -0.2f, 0.0f, 0.0f);
    require(decision.x == -1.0f && decision.y == 0.0f && decision.z == 0.0f && decision.w == 0.0f,
            "An all-nonfinite CUDA candidate set did not fail closed with zero probability.");

    const vector<float> finiteEnergies = {-1.0f, 0.5f, 2.0f, -0.25f};
    uniforms.upload(vector<float>(n, 1.0f));
    decision = runCandidateSelection(finiteEnergies, 0.75f, 0.7f, -0.2f, 0.0f, 0.0f);
    float expectedRejectedProbability = 0.0f;
    for (float energy : finiteEnergies) {
        const float scaled = 0.75f*energy;
        const float maximum = std::max(-0.2f, scaled);
        const float insertion = maximum+std::log(std::exp(-0.2f-maximum)+std::exp(scaled-maximum));
        expectedRejectedProbability = std::max(expectedRejectedProbability,
                std::min(1.0f, std::exp(0.7f-insertion)));
    }
    require(decision.x == -1.0f && decision.y == 0.0f && decision.w == 0.0f &&
            std::fabs(decision.z-expectedRejectedProbability) < 2e-6f,
            "A finite no-success decision did not preserve the maximum diagnostic probability.");
    uniforms.upload(vector<float>(n, 0.0f));

    for (float invalidValue : {nan, inf, -inf})
        for (int invalidGlobal = 0; invalidGlobal < 5; invalidGlobal++) {
            float beta = 0.75f, sourceLogSum = 0.7f, targetLogSum = -0.2f;
            float cavityBias = 0.1f, movingBias = -0.1f;
            float* globals[] = {&beta, &sourceLogSum, &targetLogSum, &cavityBias, &movingBias};
            *globals[invalidGlobal] = invalidValue;
            decision = runCandidateSelection({-1.0f, 0.5f, 2.0f, -0.25f}, beta,
                    sourceLogSum, targetLogSum, cavityBias, movingBias);
            require(decision.x == -1.0f && decision.y == 0.0f && decision.z == 0.0f && decision.w == 0.0f,
                    "A nonfinite global candidate-selection term did not reject the complete proposal.");
        }

    CudaArray regions, realScalars;
    regions.initialize<int>(cu, n, "nonfiniteRealRegions");
    realScalars.initialize<mm_float4>(cu, 1, "nonfiniteRealScalars");
    regions.upload(vector<int>{0, 1, 0, 1});
    int sourceRegion = 0, targetRegion = 1;
    float beta = 0.75f, selectionUniform = 0.37f;
    for (const vector<float>& poisonedPartition :
            {vector<float>{-1.0f, nan, 2.0f, -0.25f},
             vector<float>{inf, 0.5f, 2.0f, -0.25f},
             vector<float>{-1.0f, 0.5f, 2.0f, -inf}}) {
        energies.upload(poisonedPartition);
        void* args[] = {const_cast<int*>(&n), &sourceRegion, &targetRegion, &beta, &selectionUniform,
                &regions.getDevicePointer(), &energies.getDevicePointer(), &selected.getDevicePointer(),
                &realScalars.getDevicePointer()};
        cu.executeKernel(selectReal, args, 1, 1);
        vector<mm_float4> selectedHost, scalarHost;
        selected.download(selectedHost);
        realScalars.download(scalarHost);
        require(selectedHost[0].x == -1.0f && selectedHost[0].z == 0.0f &&
                scalarHost[0].x == 0.0f && scalarHost[0].y == 0.0f,
                "A nonfinite source/target deletion partition did not fail closed.");
    }

    CudaArray candidateParameters, candidateFullDirect, candidateDeletedDirect;
    CudaArray candidateFullPotentials, candidateDeletedPotentials, deletedParameters;
    CudaArray deletedFullDirect, deletedFullPotentials, deletedSelfPotentials;
    CudaArray removalComponents, assembledEnergies, candidateComponents, deletedComponents;
    CudaArray mtmDeletedQueryPotentials, mtmEnergyScalars;
    candidateParameters.initialize<mm_float4>(cu, n, "nonfiniteAssemblyCandidateParameters");
    candidateFullDirect.initialize<float>(cu, n, "nonfiniteAssemblyFullDirect");
    candidateDeletedDirect.initialize<float>(cu, n, "nonfiniteAssemblyDeletedDirect");
    candidateFullPotentials.initialize<float>(cu, n, "nonfiniteAssemblyFullPotentials");
    candidateDeletedPotentials.initialize<float>(cu, n, "nonfiniteAssemblyDeletedPotentials");
    deletedParameters.initialize<mm_float4>(cu, 1, "nonfiniteAssemblyDeletedParameters");
    deletedFullDirect.initialize<float>(cu, 1, "nonfiniteAssemblyDeletedFullDirect");
    deletedFullPotentials.initialize<float>(cu, 1, "nonfiniteAssemblyDeletedFullPotentials");
    deletedSelfPotentials.initialize<float>(cu, 1, "nonfiniteAssemblyDeletedSelfPotentials");
    removalComponents.initialize<mm_float4>(cu, 1, "nonfiniteAssemblyRemovalComponents");
    assembledEnergies.initialize<float>(cu, n, "nonfiniteAssemblyEnergies");
    candidateComponents.initialize<mm_float4>(cu, n, "nonfiniteAssemblyCandidateComponents");
    deletedComponents.initialize<mm_float4>(cu, 1, "nonfiniteAssemblyDeletedComponents");
    mtmDeletedQueryPotentials.initialize<float>(cu, n+1, "nonfiniteMtmDeletedQueryPotentials");
    mtmEnergyScalars.initialize<mm_float4>(cu, 1, "nonfiniteMtmEnergyScalars");
    candidateParameters.upload(vector<mm_float4>(n, mm_float4(1.0f, 0.0f, 0.0f, 0.0f)));
    candidateFullDirect.upload(vector<float>{2.0f, nan, inf, -inf});
    candidateDeletedDirect.upload(vector<float>(n, 1.0f));
    candidateFullPotentials.upload(vector<float>(n, 0.0f));
    candidateDeletedPotentials.upload(vector<float>(n, 0.0f));
    mtmDeletedQueryPotentials.upload(vector<float>(n+1, 0.0f));
    deletedParameters.upload(vector<mm_float4>{mm_float4(1.0f, 0.0f, 0.0f, 0.0f)});
    deletedFullDirect.upload(vector<float>{3.0f});
    deletedFullPotentials.upload(vector<float>{4.0f});
    deletedSelfPotentials.upload(vector<float>{1.0f});
    removalComponents.upload(vector<mm_float4>{mm_float4(0.0f, 2.0f, 0.0f, 0.0f)});
    selected.upload(vector<mm_float4>{mm_float4(0.0f, 0.0f, 0.0f, 0.0f)});
    int atomsPerMolecule = 1;
    auto runExactAssembly = [&]() {
        void* args[] = {const_cast<int*>(&n), &atomsPerMolecule,
                &candidateParameters.getDevicePointer(), &candidateFullDirect.getDevicePointer(),
                &candidateDeletedDirect.getDevicePointer(), &candidateFullPotentials.getDevicePointer(),
                &candidateDeletedPotentials.getDevicePointer(), &deletedParameters.getDevicePointer(),
                &deletedFullDirect.getDevicePointer(), &deletedFullPotentials.getDevicePointer(),
                &deletedSelfPotentials.getDevicePointer(), &removalComponents.getDevicePointer(),
                &assembledEnergies.getDevicePointer(), &selected.getDevicePointer(),
                &candidateComponents.getDevicePointer(), &deletedComponents.getDevicePointer()};
        cu.executeKernel(assembleExact, args, 1, 1);
    };
    runExactAssembly();
    vector<float> assembledHost;
    vector<mm_float4> candidateComponentsHost, deletedComponentsHost, selectedHost;
    assembledEnergies.download(assembledHost);
    candidateComponents.download(candidateComponentsHost);
    deletedComponents.download(deletedComponentsHost);
    selected.download(selectedHost);
    require(std::fabs(assembledHost[0]-1.0f) < 1e-6f && std::isnan(assembledHost[1]) &&
            std::isnan(assembledHost[2]) && std::isnan(assembledHost[3]),
            "Exact PME assembly did not replace every nonfinite candidate total with a rejection sentinel.");
    require(std::isnan(candidateComponentsHost[1].x) && std::isinf(candidateComponentsHost[2].x) &&
            candidateComponentsHost[2].x > 0.0f && std::isinf(candidateComponentsHost[3].x) &&
            candidateComponentsHost[3].x < 0.0f,
            "Exact PME assembly did not preserve nonfinite raw components for diagnosis.");
    require(std::fabs(selectedHost[0].y-6.0f) < 1e-6f,
            "Finite exact deleted energy changed while adding nonfinite candidate defenses.");

    void* mtmAssemblyArgs[] = {const_cast<int*>(&n), &atomsPerMolecule,
            &candidateParameters.getDevicePointer(), &candidateFullDirect.getDevicePointer(),
            &candidateDeletedDirect.getDevicePointer(), &candidateFullPotentials.getDevicePointer(),
            &mtmDeletedQueryPotentials.getDevicePointer(), &deletedParameters.getDevicePointer(),
            &deletedFullDirect.getDevicePointer(), &deletedFullPotentials.getDevicePointer(),
            &assembledEnergies.getDevicePointer(), &candidateComponents.getDevicePointer(),
            &mtmEnergyScalars.getDevicePointer(), &deletedComponents.getDevicePointer()};
    cu.executeKernel(assembleMtm, mtmAssemblyArgs, n);
    assembledEnergies.download(assembledHost);
    require(std::fabs(assembledHost[0]-1.0f) < 1e-6f && std::isnan(assembledHost[1]) &&
            std::isinf(assembledHost[2]) && assembledHost[2] > 0.0f && std::isnan(assembledHost[3]),
            "MTM energy assembly did not preserve the expected nonfinite zero-weight sentinels.");

    deletedFullDirect.upload(vector<float>{nan});
    runExactAssembly();
    selected.download(selectedHost);
    deletedComponents.download(deletedComponentsHost);
    require(std::isnan(selectedHost[0].y) && std::isnan(deletedComponentsHost[0].x),
            "A nonfinite exact deleted-energy component did not produce a fail-closed sentinel.");

    require(CudaCalcRelocationMCKernel::isCandidateDecisionValidForTesting(1.0f, -2.0f, 0.5f, 1.0f, n),
            "Host candidate-decision guard rejected a valid accepted decision.");
    require(CudaCalcRelocationMCKernel::isCandidateDecisionValidForTesting(-1.0f, 0.0f, 0.4f, 0.0f, n),
            "Host candidate-decision guard rejected a valid rejected decision.");
    for (const mm_float4& invalid :
            {mm_float4(nan, 0.0f, 0.0f, 0.0f), mm_float4(1.0f, nan, 0.5f, 1.0f),
             mm_float4(1.0f, inf, 0.5f, 1.0f), mm_float4(1.0f, -inf, 0.5f, 1.0f),
             mm_float4(1.0f, 0.0f, nan, 1.0f), mm_float4(1.0f, 0.0f, 1.1f, 1.0f),
             mm_float4(1.0f, 0.0f, 0.5f, 0.5f), mm_float4(4.0f, 0.0f, 0.5f, 1.0f),
             mm_float4(0.0f, 0.0f, 0.5f, 0.0f)})
        require(!CudaCalcRelocationMCKernel::isCandidateDecisionValidForTesting(
                        invalid.x, invalid.y, invalid.z, invalid.w, n),
                "Host candidate-decision guard accepted a malformed device result.");
}

static PerturbNonbondedForce* createNonbonded(const vector<double>& charges) {
    PerturbNonbondedForce* force = new PerturbNonbondedForce();
    force->setNonbondedMethod(PerturbNonbondedForce::PME);
    force->setCutoffDistance(0.9);
    force->setUseDispersionCorrection(false);
    force->addGlobalParameter("lambdaA_charge", 1.0);
    force->addGlobalParameter("lambdaB_charge", 0.0);
    force->addGlobalParameter("lambdaC_charge", 0.0);
    force->addGlobalParameter("rest2_scale", 1.0);
    force->addGlobalParameter("lambdaA_vdw", 1.0);
    force->addGlobalParameter("lambdaB_vdw", 0.0);
    for (double charge : charges)
        force->addParticle(charge, 0.3, 0.0, charge, 0.3, 0.0, false, false);
    return force;
}

static std::unique_ptr<System> createRefreshTestSystem(RelocationMC*& relocation,
        bool verbose, const std::string& logFileName, int exchangeFrequency = 0) {
    std::unique_ptr<System> system(new System());
    for (int atom = 0; atom < 4; atom++)
        system->addParticle(1.0);
    system->setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    PerturbNonbondedForce* nonbonded = createNonbonded({});
    nonbonded->addParticle(0.8, 0.3, 0.0, -0.6, 0.3, 0.0, false, false);
    nonbonded->addParticle(0.2, 0.3, 0.0, 0.2, 0.3, 0.0, false, false);
    nonbonded->addParticle(0.2, 0.3, 0.0, 0.2, 0.3, 0.0, false, false);
    nonbonded->addParticle(-0.5, 0.3, 0.0, 0.7, 0.3, 0.0, false, false);
    system->addForce(nonbonded);

    relocation = new RelocationMC(1.0e9, 0.75, exchangeFrequency, 32);
    relocation->setProtocolVersion(RelocationMC::Legacy);
    relocation->setRandomSeed(20260831);
    relocation->setUseCavityBias(false);
    relocation->setLigandAtoms({0});
    relocation->setLabileMolSiteParameters({{0.2, 0.3, 0.0}});
    relocation->setLabileMolReferencePositions({Vec3(0, 0, 0)});
    relocation->setOrientationsEnabled(false);
    relocation->setVerbose(verbose);
    relocation->setLogFileName(logFileName);
    relocation->addLabileMol({1}, 1);
    relocation->addLabileMol({2}, 2);
    system->addForce(relocation);
    return system;
}

void testNonfiniteDecisionPreservesContextState() {
    platform.setPropertyDefaultValue("UseCpuPme", "false");
    const std::string logPath = createTemporaryLogPath("nonfinite-no-mutation");
    RelocationMC* relocation;
    std::unique_ptr<System> system = createRefreshTestSystem(relocation, true, logPath);
    VerletIntegrator integrator(0.001);
    Context context(*system, integrator, platform);
    const vector<Vec3> positions = {Vec3(2.0, 2.0, 2.0), Vec3(2.25, 2.0, 2.0),
                                    Vec3(0.25, 0.25, 0.25), Vec3(3.2, 3.0, 2.8)};
    const vector<Vec3> velocities = {Vec3(0.1, -0.2, 0.3), Vec3(-0.4, 0.5, -0.6),
                                     Vec3(0.7, -0.8, 0.9), Vec3(-1.0, 1.1, -1.2)};
    context.setPositions(positions);
    context.setVelocities(velocities);
    const State before = context.getState(State::Positions | State::Velocities);

    struct CandidateEnergyFaultGuard {
        CandidateEnergyFaultGuard() {
            CudaCalcRelocationMCKernel::setForceNonfiniteCandidateEnergiesForTesting(true);
        }
        ~CandidateEnergyFaultGuard() {
            CudaCalcRelocationMCKernel::setForceNonfiniteCandidateEnergiesForTesting(false);
        }
    };
    ExchangeResult result;
    {
        CandidateEnergyFaultGuard faultGuard;
        result = relocation->attemptExchange(context);
    }

    const State state = context.getState(State::Positions | State::Velocities);
    require(result.attempted && !result.accepted,
            "A full CUDA attempt with injected nonfinite candidate energies did not reject safely.");
    require(result.insertionProbability == 0.0 && result.newPositions.empty(),
            "A rejected nonfinite CUDA attempt retained an acceptance probability or target positions.");
    for (int atom = 0; atom < static_cast<int>(positions.size()); atom++) {
        requireVecClose(state.getPositions()[atom], before.getPositions()[atom], 0.0,
                "A rejected nonfinite CUDA attempt changed Context positions.");
        requireVecClose(state.getVelocities()[atom], before.getVelocities()[atom], 0.0,
                "A rejected nonfinite CUDA attempt changed Context velocities.");
    }
    const std::string log = readTextFile(logPath);
    require(countOccurrences(log, "attempted=1 accepted=0") == 1,
            "A rejected nonfinite CUDA attempt was not recorded exactly once as unaccepted.");
    require(log.find("updatedPotentialEnergy_kJ_per_mol=") == std::string::npos,
            "A rejected nonfinite CUDA attempt entered accepted-move energy logging.");
    std::remove(logPath.c_str());
}

static double getLoggedValue(const std::string& log, const std::string& key) {
    const size_t start = log.find(key);
    require(start != std::string::npos, "Expected log key was not found: "+key);
    const size_t valueStart = start+key.size();
    const size_t valueEnd = log.find('\n', valueStart);
    require(valueEnd != std::string::npos, "Logged value was not newline terminated: "+key);
    return std::stod(log.substr(valueStart, valueEnd-valueStart));
}

static void requireAbsent(const std::string& text, const std::string& needle, const std::string& message) {
    require(text.find(needle) == std::string::npos, message);
}

static double getPotentialEnergy(Context& context) {
    return context.getState(State::Energy).getPotentialEnergy();
}

void testMinimalCudaProtocol() {
    System system;
    for (int i = 0; i < 7; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    PerturbNonbondedForce* nonbonded = createNonbonded({0.0, 0.2, -0.2, 0.2, -0.2, 0.35, -0.15});
    nonbonded->setParticleParameters(5, 0.35, 0.34, 0.45, 0.35, 0.34, 0.45, false, false);
    nonbonded->setParticleParameters(6, -0.15, 0.28, 0.30, -0.15, 0.28, 0.30, false, false);
    nonbonded->addException(1, 2, 0.0, 0.3, 0.0);
    nonbonded->addException(3, 4, 0.0, 0.3, 0.0);
    system.addForce(nonbonded);

    RelocationMC* relocation = new RelocationMC(300.0, 0.75, 0, 32);
    relocation->setProtocolVersion(RelocationMC::Legacy);
    relocation->setRandomSeed(20260829);
    relocation->setLigandAtoms({0});
    relocation->setLabileMolSiteParameters({{0.2, 0.3, 0.0}, {-0.2, 0.3, 0.0}});
    relocation->setLabileMolReferencePositions({Vec3(0, 0, 0), Vec3(0.1, 0, 0)});
    relocation->setOrientationsEnabled(true);
    relocation->addLabileMol({1, 2}, 1);
    relocation->addLabileMol({3, 4}, 3);
    system.addForce(relocation);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    vector<Vec3> initial = {Vec3(2, 2, 2), Vec3(2.25, 2, 2), Vec3(2.35, 2, 2),
                            Vec3(0.25, 0.25, 0.25), Vec3(0.35, 0.25, 0.25),
                            Vec3(2.65, 2.15, 2.05), Vec3(0.65, 0.35, 0.30)};
    context.setPositions(initial);
    const int forceGroups = 0xFFFFFFFF;
    const double initialEnergy = context.getState(State::Energy, false, forceGroups).getPotentialEnergy();
    ExchangeResult result = relocation->attemptExchange(context);
    require(result.attempted, "CUDA relocation did not attempt a proposal.");
    require(result.movedLabileMol >= 0 && result.movedLabileMol < 2,
            "CUDA relocation selected an invalid molecule.");
    require(std::isfinite(result.deletedMoleculeEnergy), "Deletion energy is not finite.");
    require(std::isfinite(result.insertionProbability), "Insertion probability is not finite.");
    require(result.accepted, "The fixed-seed energy-consistency proposal was not accepted.");
    if (result.accepted) {
        require(result.newPositions.size() == 2, "Accepted CUDA relocation returned the wrong site count.");
        const State finalState = context.getState(State::Positions | State::Energy, false, forceGroups);
        const vector<Vec3> finalPositions = finalState.getPositions();
        const double finalEnergy = finalState.getPotentialEnergy();
        const double deltaQuery = result.candidateMoleculeEnergy-result.deletedMoleculeEnergy;
        const double deltaTotal = finalEnergy-initialEnergy;
        const double error = deltaQuery-deltaTotal;
        std::cout << "Energy consistency: E_delete=" << result.deletedMoleculeEnergy
                  << " E_insert=" << result.candidateMoleculeEnergy
                  << " deltaQuery=" << deltaQuery
                  << " deltaTotal=" << deltaTotal
                  << " error=" << error
                  << " deleteDirect=" << result.deletedDirectEnergy
                  << " deleteFullMeshCross=" << result.deletedFullMeshCrossEnergy
                  << " deleteMeshSelf=" << result.deletedMeshSelfEnergy
                  << " deleteMeshRemoval=" << result.deletedMeshRemovalEnergy
                  << " insertEnvironmentDirect=" << result.candidateEnvironmentDirectEnergy
                  << " insertFullMeshCross=" << result.candidateFullMeshCrossEnergy
                  << " insertDeletedMeshCross=" << result.candidateDeletedMeshCrossEnergy
                  << " insertMeshSelf=" << result.candidateMeshSelfEnergy << std::endl;
        require(std::fabs(error) <= 0.01, "CUDA exact PME query and full Context energy differences disagree by more than 0.01 kJ/mol.");
        require(std::isfinite(deltaQuery), "CUDA query energy difference is not finite.");
        require(std::isfinite(deltaTotal), "Full Context energy difference is not finite.");
        for (int site = 0; site < 2; site++) {
            Vec3 delta = finalPositions[result.movedAtomIndices[site]]-result.newPositions[site];
            require(delta.dot(delta) < 1e-10, "Accepted CUDA relocation did not update Context positions.");
        }
        Vec3 bond = finalPositions[result.movedAtomIndices[1]]-finalPositions[result.movedAtomIndices[0]];
        for (int axis = 0; axis < 3; axis++)
            bond[axis] -= std::floor(bond[axis]/4.0+0.5)*4.0;
        require(std::fabs(std::sqrt(bond.dot(bond))-0.1) < 1e-5,
                "A candidate orientation did not preserve the rigid molecular geometry.");
    }
    std::cout << "Relocation result: attempted=" << result.attempted
              << " accepted=" << result.accepted
              << " moved=" << result.movedLabileMol
              << " deleteEnergy=" << result.deletedMoleculeEnergy
              << " candidateEnergy=" << result.candidateMoleculeEnergy
              << " insertionProbability=" << result.insertionProbability << std::endl;
}

void testMtmFullAttemptAndStateSafety() {
    platform.setPropertyDefaultValue("UseCpuPme", "false");
    System system;
    system.addParticle(1.0);
    const vector<double> waterMasses = {16.0, 1.0, 1.0};
    for (int molecule = 0; molecule < 2; molecule++)
        for (double mass : waterMasses)
            system.addParticle(mass);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    const double oh = 0.09572;
    const double hx = -0.02399872;
    const double hy = 0.09266272;
    const double hh = std::sqrt((oh-hx)*(oh-hx)+hy*hy);
    HarmonicBondForce* intramolecularBonds = new HarmonicBondForce();
    for (int first : {1, 4}) {
        system.addConstraint(first, first+1, oh);
        system.addConstraint(first, first+2, oh);
        system.addConstraint(first+1, first+2, hh);
        intramolecularBonds->addBond(first, first+1, oh, 1000.0);
        intramolecularBonds->addBond(first, first+2, oh, 1000.0);
        intramolecularBonds->addBond(first+1, first+2, hh, 1000.0);
    }
    system.addForce(intramolecularBonds);
    system.addForce(new CMMotionRemover(1));
    PerturbNonbondedForce* nonbonded = createNonbonded({0.0, -0.834, 0.417, 0.417,
                                                         -0.834, 0.417, 0.417});
    for (int oxygen : {1, 4}) {
        nonbonded->setParticleParameters(oxygen, -0.834, 0.315, 0.65,
                -0.834, 0.315, 0.65, false, false);
        for (int hydrogen : {oxygen+1, oxygen+2})
            nonbonded->setParticleParameters(hydrogen, 0.417, 0.1, 0.0,
                    0.417, 0.1, 0.0, false, false);
        nonbonded->addException(oxygen, oxygen+1, 0.0, 0.3, 0.0);
        nonbonded->addException(oxygen, oxygen+2, 0.0, 0.3, 0.0);
        nonbonded->addException(oxygen+1, oxygen+2, 0.0, 0.3, 0.0);
    }
    system.addForce(nonbonded);
    RelocationMC* relocation = new RelocationMC(300.0, 0.75, 0, 256);
    relocation->setUseCavityBias(false);
    relocation->setProtocolVersion(RelocationMC::MTM);
    relocation->setLigandAtoms({0});
    relocation->setLabileMolSiteParameters({{-0.834, 0.315, 0.65},
                                             {0.417, 0.1, 0.0},
                                             {0.417, 0.1, 0.0}});
    relocation->setLabileMolReferencePositions({Vec3(0, 0, 0), Vec3(oh, 0, 0), Vec3(hx, hy, 0)});
    relocation->addLabileMol({1, 2, 3}, 1);
    relocation->addLabileMol({4, 5, 6}, 4);
    relocation->setRandomSeed(9000);
    system.addForce(relocation);

    VerletIntegrator integrator(0.001);
    integrator.setConstraintTolerance(1e-7);
    std::map<std::string, std::string> properties;
    properties["UseCpuPme"] = "false";
    Context context(system, integrator, platform, properties);
    CudaPlatform::PlatformData& platformData = *static_cast<CudaPlatform::PlatformData*>(
            nonbonded->getCudaPlatformData(context));
    CudaContext& cu = *platformData.contexts[0];
    ContextSelector selector(cu);
    const int numRealAtoms = 6;
    const int atomsPerWater = 3;
    CudaArray realIndices, scorePositions, directAtomEnergies, fullPotentials;
    CudaArray isolatedPositions, isolatedCharges, isolatedPotentials;
    realIndices.initialize<int>(cu, numRealAtoms, "mtmComponentRealIndices");
    if (cu.getUseDoublePrecision())
        scorePositions.initialize<mm_double4>(cu, numRealAtoms, "mtmComponentPositions");
    else
        scorePositions.initialize<mm_float4>(cu, numRealAtoms, "mtmComponentPositions");
    directAtomEnergies.initialize<float>(cu, numRealAtoms, "mtmComponentDirectEnergies");
    fullPotentials.initialize<float>(cu, numRealAtoms, "mtmComponentFullPotentials");
    if (cu.getUseDoublePrecision())
        isolatedPositions.initialize<mm_double4>(cu, atomsPerWater, "mtmComponentIsolatedPositions");
    else
        isolatedPositions.initialize<mm_float4>(cu, atomsPerWater, "mtmComponentIsolatedPositions");
    isolatedCharges.initialize<float>(cu, atomsPerWater, "mtmComponentIsolatedCharges");
    isolatedPotentials.initialize<float>(cu, atomsPerWater, "mtmComponentIsolatedPotentials");
    realIndices.upload(vector<int>{1, 2, 3, 4, 5, 6});
    const vector<double> siteCharges = {-0.834, 0.417, 0.417};
    isolatedCharges.upload(vector<float>{-0.834f, 0.417f, 0.417f});
    auto computeSourceScores = [&]() {
        const State state = context.getState(State::Positions | State::Energy);
        if (cu.getUseDoublePrecision()) {
            vector<mm_double4> queryPositions(numRealAtoms);
            for (int atom = 0; atom < numRealAtoms; atom++) {
                const Vec3& position = state.getPositions()[atom+1];
                queryPositions[atom] = mm_double4(position[0], position[1], position[2], 0.0);
            }
            scorePositions.upload(queryPositions);
        }
        else {
            vector<mm_float4> queryPositions(numRealAtoms);
            for (int atom = 0; atom < numRealAtoms; atom++) {
                const Vec3& position = state.getPositions()[atom+1];
                queryPositions[atom] = mm_float4((float) position[0], (float) position[1],
                        (float) position[2], 0.0f);
            }
            scorePositions.upload(queryPositions);
        }
        // Acceptance uses base-only float positions in both single and mixed;
        // only double uses the ContextPrecision query path.
        nonbonded->getDirectEnergyForRealParticlesPmeCuda(context, numRealAtoms,
                realIndices.getDevicePointer(), atomsPerWater, directAtomEnergies.getDevicePointer());
        if (cu.getUseDoublePrecision())
            nonbonded->getPMEPotentialAtPositionsCudaContextPrecision(context, numRealAtoms,
                    scorePositions.getDevicePointer(), 0, fullPotentials.getDevicePointer());
        else
            nonbonded->getPMEPotentialAtPositionsCuda(context, numRealAtoms,
                    scorePositions.getDevicePointer(), fullPotentials.getDevicePointer());
        vector<float> directValues, potentialValues;
        directAtomEnergies.download(directValues);
        fullPotentials.download(potentialValues);
        vector<double> scores(2, 0.0);
        const double sqrtOne4PiEps0 = std::sqrt(138.93545764438198);
        for (int molecule = 0; molecule < 2; molecule++)
            for (int site = 0; site < atomsPerWater; site++) {
                const int index = molecule*atomsPerWater+site;
                scores[molecule] += directValues[index]+sqrtOne4PiEps0*siteCharges[site]*potentialValues[index];
            }
        return scores;
    };
    auto computeIsolatedReciprocalContribution = [&](const vector<Vec3>& positions) {
        require((int) positions.size() == atomsPerWater,
                "Isolated reciprocal oracle received the wrong molecule size.");
        if (cu.getUseDoublePrecision()) {
            vector<mm_double4> base(atomsPerWater);
            for (int site = 0; site < atomsPerWater; site++)
                base[site] = mm_double4(positions[site][0], positions[site][1], positions[site][2], 0.0);
            isolatedPositions.upload(base);
            nonbonded->getPMEPotentialFromParticlesCudaContextPrecision(context,
                    atomsPerWater, isolatedPositions.getDevicePointer(), 0,
                    isolatedCharges.getDevicePointer(), atomsPerWater,
                    isolatedPositions.getDevicePointer(), 0, isolatedPotentials.getDevicePointer());
        }
        else {
            vector<mm_float4> base(atomsPerWater);
            for (int site = 0; site < atomsPerWater; site++)
                base[site] = mm_float4((float) positions[site][0], (float) positions[site][1],
                        (float) positions[site][2], 0.0f);
            isolatedPositions.upload(base);
            // Single and mixed acceptance both use the base-only legacy scratch API.
            nonbonded->getPMEPotentialFromParticlesCuda(context, atomsPerWater,
                    isolatedPositions.getDevicePointer(), isolatedCharges.getDevicePointer(),
                    atomsPerWater, isolatedPositions.getDevicePointer(), isolatedPotentials.getDevicePointer());
        }
        vector<float> potentials;
        isolatedPotentials.download(potentials);
        double contribution = 0.0;
        const double sqrtOne4PiEps0 = std::sqrt(138.93545764438198);
        for (int site = 0; site < atomsPerWater; site++)
            contribution += sqrtOne4PiEps0*siteCharges[site]*potentials[site];
        return contribution;
    };
    struct MtmDiagnosticCaptureGuard {
        MtmDiagnosticCaptureGuard() {
            CudaCalcRelocationMCKernel::setCaptureMtmDiagnosticsForTesting(true);
        }
        ~MtmDiagnosticCaptureGuard() {
            CudaCalcRelocationMCKernel::setCaptureMtmDiagnosticsForTesting(false);
        }
    } diagnosticCaptureGuard;
    const vector<Vec3> initialPositions = {
            Vec3(2.0, 2.0, 2.0),
            Vec3(2.30, 2.0, 2.0), Vec3(2.30+oh, 2.0, 2.0), Vec3(2.30+hx, 2.0+hy, 2.0),
            Vec3(0.50, 0.50, 0.50), Vec3(0.50+oh, 0.50, 0.50), Vec3(0.50+hx, 0.50+hy, 0.50)};
    const vector<Vec3> initialVelocities = {
            Vec3(0.1, -0.2, 0.3),
            Vec3(1.2, -0.4, 0.3), Vec3(-0.7, 1.1, 0.8), Vec3(0.2, -1.3, 0.5),
            Vec3(-0.6, 0.9, -0.2), Vec3(1.4, 0.3, -0.7), Vec3(-0.1, -1.2, 0.6)};
    auto isInner = [](const Vec3& position, const Vec3& center) {
        Vec3 delta = position-center;
        for (int axis = 0; axis < 3; axis++)
            delta[axis] -= std::floor(delta[axis]/4.0+0.5)*4.0;
        return delta.dot(delta) < 0.75*0.75;
    };
    auto distance = [](const Vec3& first, const Vec3& second) {
        Vec3 delta = second-first;
        for (int axis = 0; axis < 3; axis++)
            delta[axis] -= std::floor(delta[axis]/4.0+0.5)*4.0;
        return std::sqrt(delta.dot(delta));
    };
    auto logSumExp = [](const vector<double>& values) {
        const double maximum = *std::max_element(values.begin(), values.end());
        double sum = 0.0;
        for (double value : values)
            sum += std::exp(value-maximum);
        return maximum+std::log(sum);
    };
    ExchangeResult accepted;
    State acceptedBefore;
    double acceptedInitialEnergy = 0.0;
    int acceptedCount = 0;
    const double mtmBeta = 1.0/(0.00831446261815324*300.0);
    const double movingBias = std::log(2.0); // one molecule in each region, reverse is forced.
    for (uint64_t seedValue = 9001; seedValue <= 9064 && acceptedCount < 3; seedValue++) {
        context.setPositions(initialPositions);
        context.setVelocities(initialVelocities);
        relocation->setRandomSeed(seedValue);
        relocation->updateParametersInContext(context);
        acceptedBefore = context.getState(State::Positions | State::Velocities | State::Energy);
        acceptedInitialEnergy = acceptedBefore.getPotentialEnergy();
        const vector<double> beforeScores = computeSourceScores();
        const ExchangeResult result = relocation->attemptExchange(context);
        const State after = context.getState(State::Positions | State::Velocities | State::Energy);
        require(result.attempted && result.protocolVersion == RelocationMC::MTM,
                "A fixed-seed MTM integration proposal was not attempted under protocol v2.");
        if (!result.accepted) {
            for (int atom = 0; atom < system.getNumParticles(); atom++) {
                requireVecClose(after.getPositions()[atom], acceptedBefore.getPositions()[atom], 0.0,
                        "A rejected finite MTM proposal changed Context positions.");
                requireVecClose(after.getVelocities()[atom], acceptedBefore.getVelocities()[atom], 0.0,
                        "A rejected finite MTM proposal changed Context velocities.");
            }
            continue;
        }

        const vector<double> afterScores = computeSourceScores();
        const vector<float> proposedScores =
                CudaCalcRelocationMCKernel::getLastMtmProposedScoresForTesting();
        require(proposedScores.size() == 2,
                "MTM proposed-state diagnostic did not capture one score per labile molecule.");
        if (cu.getUseMixedPrecision()) {
            vector<mm_float4> committedBase, committedCorrection;
            vector<int> atomOrder;
            cu.getPosq().download(committedBase);
            cu.getPosqCorrection().download(committedCorrection);
            cu.getAtomIndexArray().download(atomOrder);
            for (int site = 0; site < atomsPerWater; site++) {
                const int hostAtom = result.movedAtomIndices[site];
                int deviceAtom = -1;
                for (int atom = 0; atom < cu.getNumAtoms(); atom++)
                    if (atomOrder[atom] == hostAtom) {
                        deviceAtom = atom;
                        break;
                    }
                require(deviceAtom >= 0, "Mixed commit oracle could not map a moved host atom.");
                const float baseValues[3] = {committedBase[deviceAtom].x,
                        committedBase[deviceAtom].y, committedBase[deviceAtom].z};
                const float correctionValues[3] = {committedCorrection[deviceAtom].x,
                        committedCorrection[deviceAtom].y, committedCorrection[deviceAtom].z};
                for (int axis = 0; axis < 3; axis++) {
                    const double generated = result.newPositions[site][axis];
                    const float expectedEnergyBase = (float) generated;
                    const float expectedCorrection = (float) (generated-(double) expectedEnergyBase);
                    require(baseValues[axis] == expectedEnergyBase,
                            "Mixed accepted candidate energy-base differs from committed Context posq base.");
                    require(correctionValues[axis] == expectedCorrection,
                            "Mixed accepted candidate correction did not round-trip through Context commit.");
                    require((double) baseValues[axis]+correctionValues[axis] == generated,
                            "Mixed committed base+correction does not exactly reconstruct the selected candidate.");
                }
            }
        }
        const int moved = result.movedLabileMol;
        const int oldTarget = 1-moved;
        require(std::isnan(proposedScores[moved]),
                "MTM proposed-state diagnostic did not mask the moved source molecule.");
        require(std::fabs(proposedScores[oldTarget]-afterScores[oldTarget]) <= 0.02,
                "GPU pair-corrected old-target source score disagrees with direct proposed-state reconstruction.");
        require(std::fabs(result.deletedSourceScore-beforeScores[moved]) <= 0.02,
                "MTM selected deleted source score disagrees with direct current-state reconstruction.");
        require(std::fabs(result.candidateSourceScore-afterScores[moved]) <= 0.02,
                "MTM selected candidate source score disagrees with direct proposed-state reconstruction.");
        require(std::fabs(result.sourceLogSum-mtmBeta*beforeScores[moved]) <= 0.01,
                "MTM source partition disagrees with the directly reconstructed one-source partition.");
        const double reconstructedTargetLogSum = logSumExp({mtmBeta*afterScores[0], mtmBeta*afterScores[1]});
        require(std::fabs(result.proposedSourceLogSum-reconstructedTargetLogSum) <= 0.01,
                "MTM proposed-state log Z_T disagrees with exhaustive direct reconstruction.");

        const double oldTargetLogWeight = mtmBeta*afterScores[oldTarget];
        const double partitionMaximum = std::max(result.proposedSourceLogSum, oldTargetLogWeight);
        const double candidateMass = std::exp(result.proposedSourceLogSum-partitionMaximum)-
                std::exp(oldTargetLogWeight-partitionMaximum);
        require(candidateMass > 0.0 && std::isfinite(candidateMass),
                "MTM proposed partition could not isolate the moved candidate source score.");
        const double partitionInferredCandidateScore =
                (partitionMaximum+std::log(candidateMass))/mtmBeta;
        require(std::fabs(partitionInferredCandidateScore-afterScores[moved]) <= 0.05,
                "MTM partition-inferred candidate source score disagrees with direct reconstruction.");
        const double ratioInferredCandidateScore = result.deletedSourceScore+
                (result.logAcceptanceRatio-movingBias-result.sourceLogSum+
                 result.proposedSourceLogSum-result.forwardLogWeightSum+
                 result.reverseLogWeightSum+mtmBeta*result.energyResidual)/mtmBeta;
        require(std::fabs(ratioInferredCandidateScore-afterScores[moved]) <= 0.05,
                "MTM log-ratio-inferred candidate source score disagrees with direct reconstruction.");
        require(std::fabs(result.deletedMoleculeEnergy-
                (result.deletedDirectEnergy+result.deletedFullMeshCrossEnergy)) <= 0.02,
                "MTM u_D does not equal its direct plus environment reciprocal components.");
        require(std::fabs(result.candidateMoleculeEnergy-
                (result.candidateEnvironmentDirectEnergy+result.candidateMeshSelfEnergy)) <= 0.02,
                "MTM u_C does not equal its direct plus environment reciprocal components.");
        const double deletedIsolatedCross = result.deletedSourceScore-result.deletedMoleculeEnergy;
        const double candidateIsolatedCross = result.candidateSourceScore-result.candidateMoleculeEnergy;
        const int movedOffsetForComponents = result.movedAtomIndices[0];
        const vector<Vec3> deletedPose = {acceptedBefore.getPositions()[movedOffsetForComponents],
                acceptedBefore.getPositions()[movedOffsetForComponents+1],
                acceptedBefore.getPositions()[movedOffsetForComponents+2]};
        const vector<Vec3> candidatePose = {after.getPositions()[movedOffsetForComponents],
                after.getPositions()[movedOffsetForComponents+1],
                after.getPositions()[movedOffsetForComponents+2]};
        const double deletedScratchContribution = computeIsolatedReciprocalContribution(deletedPose);
        const double candidateScratchContribution = computeIsolatedReciprocalContribution(candidatePose);
        require(std::fabs(deletedIsolatedCross-deletedScratchContribution) <= 0.02,
                "MTM deleted isolated reciprocal contribution disagrees with the base-position scratch oracle.");
        require(std::fabs(candidateIsolatedCross-candidateScratchContribution) <= 0.02,
                "MTM candidate isolated reciprocal contribution disagrees with the base-position scratch oracle.");
        require(std::fabs(result.isolatedReciprocalDelta-
                0.5*(candidateIsolatedCross-deletedIsolatedCross)) <= 0.02,
                "MTM isolated reciprocal diagnostic disagrees with reconstructed C/D isolated reciprocal contributions.");
        const double expectedLogRatio = movingBias+result.sourceLogSum-result.proposedSourceLogSum+
                mtmBeta*(result.candidateSourceScore-result.deletedSourceScore)+
                result.forwardLogWeightSum-result.reverseLogWeightSum-mtmBeta*result.energyResidual;
        require(std::fabs(result.logAcceptanceRatio-expectedLogRatio) <= 0.01,
                "MTM final log ratio disagrees with reconstructed protocol components.");
        const double fullDelta = after.getPotentialEnergy()-acceptedInitialEnergy;
        const double environmentDelta = result.candidateMoleculeEnergy-result.deletedMoleculeEnergy;
        const double correctedDelta = environmentDelta+result.energyResidual;
        const double correctedError = fullDelta-correctedDelta;
        const double environmentOnlyError = fullDelta-environmentDelta;
        const double deltaTolerance = cu.getUseDoublePrecision() ? 2.0e-6 : 6.0e-5;
        requireClose(result.energyResidual, result.isolatedReciprocalDelta, 1e-14,
                "MTM energy residual must equal the isolated reciprocal delta.");
        requireClose(result.energyDelta, correctedDelta, 3e-6,
                "MTM energyDelta does not include the isolated reciprocal residual.");
        require(std::fabs(correctedError) <= deltaTolerance,
                "MTM residual-corrected energy delta exceeds its precision-specific Context tolerance.");
        require(std::fabs(environmentOnlyError) > deltaTolerance &&
                std::fabs(environmentOnlyError) > 5.0*std::fabs(correctedError),
                "Environment-only delta is not significantly worse than the isolated-reciprocal-corrected delta.");
        std::cout << "MTM component oracle seed=" << seedValue
                  << " uD=" << result.deletedMoleculeEnergy
                  << " uC=" << result.candidateMoleculeEnergy
                  << " sD=" << result.deletedSourceScore
                  << " sC=" << result.candidateSourceScore
                  << " deletedIsolatedCross=" << deletedIsolatedCross
                  << " candidateIsolatedCross=" << candidateIsolatedCross
                  << " isolatedReciprocalDelta=" << result.isolatedReciprocalDelta
                  << " contextDelta=" << fullDelta
                  << " environmentOnlyError=" << fullDelta-environmentDelta
                  << " isolatedCorrectedError=" << fullDelta-(environmentDelta+result.isolatedReciprocalDelta)
                  << std::endl;
        accepted = result;
        acceptedCount++;
    }
    require(acceptedCount >= 3, "Fewer than three fixed MTM seeds produced accepted full attempts.");
    const State acceptedAfter = context.getState(State::Positions | State::Velocities | State::Energy);
    require(std::isfinite(accepted.sourceLogSum) && std::isfinite(accepted.proposedSourceLogSum) &&
            std::isfinite(accepted.forwardLogWeightSum) && std::isfinite(accepted.reverseLogWeightSum) &&
            std::isfinite(accepted.logAcceptanceRatio) && std::isfinite(accepted.acceptanceProbability) &&
            std::isfinite(accepted.candidateSelectionProbability) && std::isfinite(accepted.deletedSourceScore) &&
            std::isfinite(accepted.candidateSourceScore) && std::isfinite(accepted.energyResidual) &&
            std::isfinite(accepted.isolatedReciprocalDelta),
            "An accepted MTM attempt returned nonfinite compact protocol diagnostics.");
    const double expectedAcceptance = accepted.logAcceptanceRatio >= 0.0 ? 1.0 :
            std::exp(accepted.logAcceptanceRatio);
    requireClose(accepted.acceptanceProbability, expectedAcceptance, 3e-6,
            "MTM final probability disagrees with its log acceptance ratio.");
    requireClose(accepted.insertionProbability, accepted.acceptanceProbability, 1e-14,
            "MTM compatibility insertionProbability does not map to the final acceptance probability.");
    require(accepted.candidateSelectionProbability > 0.0 && accepted.candidateSelectionProbability <= 1.0 &&
            accepted.deletionProbability > 0.0 && accepted.deletionProbability <= 1.0,
            "MTM source or candidate categorical probability is outside (0,1].");
    const double contextDelta = acceptedAfter.getPotentialEnergy()-acceptedInitialEnergy;
    const double environmentDelta = accepted.candidateMoleculeEnergy-accepted.deletedMoleculeEnergy;
    requireClose(accepted.energyResidual, accepted.isolatedReciprocalDelta, 1e-14,
            "MTM energy residual must equal the isolated reciprocal delta.");
    requireClose(accepted.energyDelta, environmentDelta+accepted.energyResidual, 3e-6,
            "MTM energyDelta must equal uC-uD plus the isolated reciprocal residual.");
    const double environmentOnlyError = contextDelta-environmentDelta;
    const double isolatedCorrectedError = contextDelta-accepted.energyDelta;
    const double deltaTolerance = cu.getUseDoublePrecision() ? 2.0e-6 : 6.0e-5;
    require(std::fabs(isolatedCorrectedError) <= deltaTolerance,
            "MTM residual-corrected energy delta exceeds its precision-specific Context tolerance.");
    require(std::fabs(environmentOnlyError) > deltaTolerance &&
            std::fabs(environmentOnlyError) > 5.0*std::fabs(isolatedCorrectedError),
            "Environment-only delta is not significantly worse than the residual-corrected delta.");
    std::cout << "MTM energy hypotheses: contextDelta=" << contextDelta
              << " environmentDelta=" << environmentDelta
              << " isolatedReciprocalDelta=" << accepted.isolatedReciprocalDelta
              << " environmentOnlyError=" << environmentOnlyError
              << " isolatedCorrectedError=" << isolatedCorrectedError << std::endl;

    int innerBefore = 0;
    int innerAfter = 0;
    for (int oxygen : {1, 4}) {
        innerBefore += isInner(acceptedBefore.getPositions()[oxygen], acceptedBefore.getPositions()[0]);
        innerAfter += isInner(acceptedAfter.getPositions()[oxygen], acceptedAfter.getPositions()[0]);
    }
    require(innerAfter == innerBefore+(accepted.exchangeToInner ? 1 : -1),
            "Accepted MTM relocation did not update source/target molecule counts by one.");
    const int movedOffset = accepted.movedAtomIndices[0];
    const int unmovedOffset = movedOffset == 1 ? 4 : 1;
    for (int site = 0; site < 3; site++) {
        requireVecClose(acceptedAfter.getPositions()[unmovedOffset+site],
                acceptedBefore.getPositions()[unmovedOffset+site], 0.0,
                "Accepted MTM relocation changed an unmoved molecule position.");
        requireVecClose(acceptedAfter.getVelocities()[unmovedOffset+site],
                acceptedBefore.getVelocities()[unmovedOffset+site], 0.0,
                "Accepted MTM relocation changed an unmoved molecule velocity.");
    }
    requireVecClose(acceptedAfter.getPositions()[0], acceptedBefore.getPositions()[0], 0.0,
            "Accepted MTM relocation changed the ligand position.");
    requireVecClose(acceptedAfter.getVelocities()[0], acceptedBefore.getVelocities()[0], 0.0,
            "Accepted MTM relocation changed the ligand velocity.");
    for (const std::pair<int, int>& pair : {std::make_pair(0, 1), std::make_pair(0, 2),
            std::make_pair(1, 2)})
        requireClose(distance(acceptedBefore.getPositions()[movedOffset+pair.first],
                                     acceptedBefore.getPositions()[movedOffset+pair.second]),
                     distance(acceptedAfter.getPositions()[movedOffset+pair.first],
                                    acceptedAfter.getPositions()[movedOffset+pair.second]), 3e-6,
                     "Accepted MTM Haar pose did not preserve the actual source rigid geometry.");
    const vector<Vec3> sourceVelocities = {acceptedBefore.getVelocities()[movedOffset],
            acceptedBefore.getVelocities()[movedOffset+1], acceptedBefore.getVelocities()[movedOffset+2]};
    const vector<Vec3> finalVelocities = {acceptedAfter.getVelocities()[movedOffset],
            acceptedAfter.getVelocities()[movedOffset+1], acceptedAfter.getVelocities()[movedOffset+2]};
    requireVecClose(massWeightedCOM(waterMasses, finalVelocities),
            massWeightedCOM(waterMasses, sourceVelocities), 3e-6,
            "Accepted MTM relocation did not preserve the moved molecule COM velocity.");
    requireClose(relativeKineticEnergy(waterMasses, finalVelocities),
            relativeKineticEnergy(waterMasses, sourceVelocities), 3e-6,
            "Accepted MTM relocation did not preserve COM-relative kinetic energy.");

    // Whole-pool nonfinite faults have no positive weight and must reject without
    // changing Context state.
    using CandidateEnergyFault = CudaCalcRelocationMCKernel::CandidateEnergyFault;
    const vector<CandidateEnergyFault> rejectingFaults = {
            CandidateEnergyFault::AllNaN,
            CandidateEnergyFault::AllPositiveInfinity,
            CandidateEnergyFault::AllNegativeInfinity};
    for (CandidateEnergyFault fault : rejectingFaults) {
        const State rejectedBefore = context.getState(State::Positions | State::Velocities);
        CudaCalcRelocationMCKernel::setCandidateEnergyFaultForTesting(fault);
        ExchangeResult rejected;
        try {
            rejected = relocation->attemptExchange(context);
        }
        catch (...) {
            CudaCalcRelocationMCKernel::setCandidateEnergyFaultForTesting(CandidateEnergyFault::None);
            throw;
        }
        CudaCalcRelocationMCKernel::setCandidateEnergyFaultForTesting(CandidateEnergyFault::None);
        const State rejectedAfter = context.getState(State::Positions | State::Velocities);
        require(rejected.attempted && !rejected.accepted && rejected.protocolVersion == RelocationMC::MTM,
                "Injected invalid or empty MTM pool did not reject under protocol v2.");
        require(rejected.insertionProbability == 0.0 && rejected.newPositions.empty(),
                "A rejected MTM fault retained an acceptance probability or target positions.");
        for (int atom = 0; atom < system.getNumParticles(); atom++) {
            requireVecClose(rejectedAfter.getPositions()[atom], rejectedBefore.getPositions()[atom], 0.0,
                    "Rejected MTM fault changed Context positions.");
            requireVecClose(rejectedAfter.getVelocities()[atom], rejectedBefore.getVelocities()[atom], 0.0,
                    "Rejected MTM fault changed Context velocities.");
        }
    }

    // A single nonfinite candidate is a supported zero-weight trial, so each
    // fault must complete normally even though the final Metropolis decision can vary.
    for (CandidateEnergyFault fault : {CandidateEnergyFault::SingleNaN,
            CandidateEnergyFault::SinglePositiveInfinity,
            CandidateEnergyFault::SingleNegativeInfinity}) {
        CudaCalcRelocationMCKernel::setCandidateEnergyFaultForTesting(fault);
        ExchangeResult singleFaultResult;
        try {
            singleFaultResult = relocation->attemptExchange(context);
        }
        catch (...) {
            CudaCalcRelocationMCKernel::setCandidateEnergyFaultForTesting(CandidateEnergyFault::None);
            throw;
        }
        CudaCalcRelocationMCKernel::setCandidateEnergyFaultForTesting(CandidateEnergyFault::None);
        require(singleFaultResult.attempted && singleFaultResult.protocolVersion == RelocationMC::MTM,
                "A single nonfinite zero-weight fault did not complete an MTM proposal safely.");
    }
}

void testMtmUnsupportedConfigurationsRejected() {
    platform.setPropertyDefaultValue("UseCpuPme", "false");
    enum InvalidCase { Dummy, Rest2, ParameterMismatch, ParticleOffset, NoDirectSpace,
                       CrossConstraint, CrossBond, UnknownExternalForce };
    const vector<std::pair<InvalidCase, std::string> > cases = {
            {Dummy, "lambda-invariant ordinary"},
            {Rest2, "lambda-invariant ordinary"},
            {ParameterMismatch, "matching the registered site parameters"},
            {ParticleOffset, "parameter offsets"},
            {NoDirectSpace, "direct-space"},
            {CrossConstraint, "constraint terms between labile and nonlabile"},
            {CrossBond, "HarmonicBondForce terms between labile and nonlabile"},
            {UnknownExternalForce, "unsupported additional Force"}};
    for (const auto& test : cases) {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
        PerturbNonbondedForce* nonbonded = createNonbonded({0.0, 0.2});
        if (test.first == Dummy)
            nonbonded->setParticleParameters(1, 0.2, 0.3, 0.0, 0.2, 0.3, 0.0, true, false);
        else if (test.first == Rest2)
            nonbonded->setParticleParameters(1, 0.2, 0.3, 0.0, 0.2, 0.3, 0.0, false, true);
        else if (test.first == ParameterMismatch)
            nonbonded->setParticleParameters(1, 0.2, 0.3, 0.0, 0.25, 0.3, 0.0, false, false);
        else if (test.first == ParticleOffset) {
            try {
                nonbonded->addParticleParameterOffset("lambdaA_charge", 1, 0.01, 0.0, 0.0);
            }
            catch (const OpenMMException& error) {
                require(std::string(error.what()).find("not implemented") != std::string::npos,
                        "Perturb rejected particle offsets with an unexpected explanation.");
                delete nonbonded;
                continue;
            }
        }
        else if (test.first == NoDirectSpace)
            nonbonded->setIncludeDirectSpace(false);
        if (test.first == CrossConstraint)
            system.addConstraint(0, 1, 0.5);
        system.addForce(nonbonded);
        if (test.first == CrossBond) {
            HarmonicBondForce* crossBond = new HarmonicBondForce();
            crossBond->addBond(0, 1, 0.5, 100.0);
            system.addForce(crossBond);
        }
        if (test.first == UnknownExternalForce) {
            CustomExternalForce* external = new CustomExternalForce("x");
            external->addParticle(1, vector<double>());
            system.addForce(external);
        }
        RelocationMC* relocation = new RelocationMC(300.0, 0.75, 0, 16);
        relocation->setUseCavityBias(false);
        relocation->setProtocolVersion(RelocationMC::MTM);
        relocation->setLigandAtoms({0});
        relocation->setLabileMolSiteParameters({{0.2, 0.3, 0.0}});
        relocation->setLabileMolReferencePositions({Vec3(0, 0, 0)});
        relocation->addLabileMol({1}, 1);
        system.addForce(relocation);
        VerletIntegrator integrator(0.001);
        try {
            Context context(system, integrator, platform);
        }
        catch (const std::exception& error) {
            require(std::string(error.what()).find(test.second) != std::string::npos,
                    "MTM unsupported-labile rejection returned the wrong explanation: "+std::string(error.what()));
            continue;
        }
        throw std::runtime_error("MTM Context accepted an unsupported labile particle configuration.");
    }
}

void testInnerCandidateRigidGeometryAcrossBoundary() {
    platform.setPropertyDefaultValue("UseCpuPme", "false");
    System system;
    for (int atom = 0; atom < 7; atom++)
        system.addParticle(atom == 0 ? 1.0 : (atom%3 == 1 ? 16.0 : 1.0));
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));

    const double oh = 0.09572;
    const double hx = -0.02399872;
    const double hy = 0.09266272;
    const double hh = std::sqrt((oh-hx)*(oh-hx)+hy*hy);
    for (int first : {1, 4}) {
        system.addConstraint(first, first+1, oh);
        system.addConstraint(first, first+2, oh);
        system.addConstraint(first+1, first+2, hh);
    }

    PerturbNonbondedForce* nonbonded = createNonbonded({0.0, -0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
    for (int first : {1, 4}) {
        nonbonded->addException(first, first+1, 0.0, 0.3, 0.0);
        nonbonded->addException(first, first+2, 0.0, 0.3, 0.0);
        nonbonded->addException(first+1, first+2, 0.0, 0.3, 0.0);
    }
    system.addForce(nonbonded);

    // The low four bytes of this double decode as a very large float.  This
    // radius therefore catches accidental double-address/float-kernel ABI use.
    const double sphereRadius = 0.01;
    RelocationMC* relocation = new RelocationMC(300.0, sphereRadius, 0, 8);
    relocation->setProtocolVersion(RelocationMC::Legacy);
    relocation->setRandomSeed(20260830);
    relocation->setUseCavityBias(false);
    relocation->setLigandAtoms({0});
    relocation->setLabileMolSiteParameters({{0.0, 0.3, 0.0}, {0.0, 0.3, 0.0}, {0.0, 0.3, 0.0}});
    relocation->setLabileMolReferencePositions({Vec3(0, 0, 0), Vec3(oh, 0, 0), Vec3(hx, hy, 0)});
    relocation->setOrientationsEnabled(false);
    relocation->addLabileMol({1, 2, 3}, 1);
    relocation->addLabileMol({4, 5, 6}, 4);
    system.addForce(relocation);

    VerletIntegrator integrator(0.001);
    integrator.setConstraintTolerance(1e-6);
    std::map<std::string, std::string> properties;
    properties["UseCpuPme"] = "false";
    Context context(system, integrator, platform, properties);
    relocation->setTemperature(305.0);
    relocation->setSphereRadius(0.012);
    relocation->setUseCavityBias(true);
    relocation->setOrientationsEnabled(true);
    requireThrowsContaining([&]() { relocation->updateParametersInContext(context); }, "reinitialization",
            "Changing orientationsEnabled on an initialized Context should be rejected.");
    // The failed update must not partially apply earlier scalar settings.
    relocation->setTemperature(300.0);
    relocation->setSphereRadius(sphereRadius);
    relocation->setUseCavityBias(false);
    relocation->setOrientationsEnabled(false);
    relocation->updateParametersInContext(context);
    relocation->setOrientationsEnabled(true);
    context.reinitialize(true);
    const Vec3 ligand(3.99, 2.0, 2.0);
    // Both source waters use a known 0.37-radian rotation about z, which is not
    // one of the finite prepared orientations.  Therefore every accepted target
    // exercises a nonidentity source-to-candidate rotation.  Sites are wrapped
    // individually to exercise source-frame reconstruction across PBC.
    const double sourceAngle = 0.37;
    auto rotatedAndWrapped = [=](const Vec3& anchor, const Vec3& offset) {
        Vec3 position = anchor+Vec3(std::cos(sourceAngle)*offset[0]-std::sin(sourceAngle)*offset[1],
                std::sin(sourceAngle)*offset[0]+std::cos(sourceAngle)*offset[1], offset[2]);
        for (int axis = 0; axis < 3; axis++)
            position[axis] -= std::floor(position[axis]/4.0)*4.0;
        return position;
    };
    const vector<Vec3> initialPositions = {ligand,
            rotatedAndWrapped(Vec3(0.02, 0.02, 1.0), Vec3(0, 0, 0)),
            rotatedAndWrapped(Vec3(0.02, 0.02, 1.0), Vec3(oh, 0, 0)),
            rotatedAndWrapped(Vec3(0.02, 0.02, 1.0), Vec3(hx, hy, 0)),
            rotatedAndWrapped(Vec3(3.98, 3.98, 3.0), Vec3(0, 0, 0)),
            rotatedAndWrapped(Vec3(3.98, 3.98, 3.0), Vec3(oh, 0, 0)),
            rotatedAndWrapped(Vec3(3.98, 3.98, 3.0), Vec3(hx, hy, 0))};
    const vector<Vec3> initialVelocities = {
            Vec3(0.15, -0.25, 0.35),
            Vec3(1.2, -0.4, 0.3), Vec3(-0.7, 1.1, 0.8), Vec3(0.2, -1.3, 0.5),
            Vec3(-0.6, 0.9, -0.2), Vec3(1.4, 0.3, -0.7), Vec3(-0.1, -1.2, 0.6)};
    context.setPositions(initialPositions);
    context.setVelocities(initialVelocities);
    const vector<Vec3> velocitiesBefore = context.getState(State::Velocities).getVelocities();
    context.getState(State::Energy);

    ExchangeResult result = relocation->attemptExchange(context);
    require(result.attempted && result.exchangeToInner && result.accepted,
            "Fixed-seed external-to-inner constrained-water relocation was not accepted.");
    require(result.newPositions.size() == 3, "Accepted constrained-water relocation returned the wrong site count.");
    Vec3 anchorDelta = result.newPositions[0]-ligand;
    for (int axis = 0; axis < 3; axis++)
        anchorDelta[axis] -= std::floor(anchorDelta[axis]/4.0+0.5)*4.0;
    require(anchorDelta.dot(anchorDelta) <= sphereRadius*sphereRadius+1e-8,
            "Inner candidate anchor escaped the configured sphere (possible scalar ABI mismatch).");
    const Vec3 candidateOh1 = result.newPositions[1]-result.newPositions[0];
    const Vec3 candidateOh2 = result.newPositions[2]-result.newPositions[0];
    const Vec3 candidateHh = result.newPositions[2]-result.newPositions[1];
    require(std::fabs(std::sqrt(candidateOh1.dot(candidateOh1))-oh) < 2e-6 &&
            std::fabs(std::sqrt(candidateOh2.dot(candidateOh2))-oh) < 2e-6 &&
            std::fabs(std::sqrt(candidateHh.dot(candidateHh))-hh) < 2e-6,
            "Inner candidate collapsed or failed to preserve raw rigid distances.");
    bool spansRawPeriodicImage = false;
    for (int axis = 0; axis < 3; axis++) {
        double minimum = result.newPositions[0][axis];
        double maximum = result.newPositions[0][axis];
        for (int site = 1; site < 3; site++) {
            minimum = std::min(minimum, result.newPositions[site][axis]);
            maximum = std::max(maximum, result.newPositions[site][axis]);
        }
        spansRawPeriodicImage |= minimum < 0.0 || maximum >= 4.0;
    }
    require(spansRawPeriodicImage,
            "Boundary regression did not make a rigid molecule span a raw periodic image.");

    const State committedState = context.getState(State::Positions | State::Velocities);
    const vector<Vec3>& committedVelocities = committedState.getVelocities();
    const int movedOffset = result.movedAtomIndices[0] == 1 ? 1 : 4;
    const int unmovedOffset = movedOffset == 1 ? 4 : 1;
    requireVecClose(committedVelocities[0], velocitiesBefore[0], 0.0,
            "Accepted relocation changed a non-labile atom velocity.");
    for (int site = 0; site < 3; site++)
        requireVecClose(committedVelocities[unmovedOffset+site], velocitiesBefore[unmovedOffset+site], 0.0,
                "Accepted relocation changed a non-moved labile molecule velocity.");

    const vector<double> waterMasses = {16.0, 1.0, 1.0};
    const vector<Vec3> sourceVelocities = {velocitiesBefore[movedOffset], velocitiesBefore[movedOffset+1],
                                           velocitiesBefore[movedOffset+2]};
    const vector<Vec3> updatedVelocities = {committedVelocities[movedOffset], committedVelocities[movedOffset+1],
                                            committedVelocities[movedOffset+2]};
    const Vec3 sourceCOM = massWeightedCOM(waterMasses, sourceVelocities);
    requireVecClose(massWeightedCOM(waterMasses, updatedVelocities), sourceCOM, 2e-6,
            "Accepted relocation did not preserve unequal-mass COM velocity and linear momentum.");
    requireClose(relativeKineticEnergy(waterMasses, updatedVelocities),
            relativeKineticEnergy(waterMasses, sourceVelocities), 2e-6,
            "Accepted relocation did not preserve relative kinetic energy.");
    const PreparedOrientations prepared;
    int matchedOrientation = -1;
    for (int orientation = 0; orientation < prepared.size(); orientation++) {
        const Vec3 expectedOh1 = prepared.rotate(orientation, Vec3(oh, 0.0, 0.0));
        const Vec3 expectedOh2 = prepared.rotate(orientation, Vec3(hx, hy, 0.0));
        const Vec3 firstDifference = expectedOh1-candidateOh1;
        const Vec3 secondDifference = expectedOh2-candidateOh2;
        if (firstDifference.dot(firstDifference) < 1e-10 && secondDifference.dot(secondDifference) < 1e-10) {
            matchedOrientation = orientation;
            break;
        }
    }
    require(matchedOrientation >= 0,
            "Accepted water candidate did not match one prepared orientation on both independent vectors.");
    const std::array<Vec3, 3>& target = prepared.getRotationMatrix(matchedOrientation);
    const vector<Vec3> movedSourcePositions = {initialPositions[movedOffset], initialPositions[movedOffset+1],
                                                initialPositions[movedOffset+2]};
    const Vec3 testBox[3] = {Vec3(4.0, 0.0, 0.0), Vec3(0.0, 4.0, 0.0), Vec3(0.0, 0.0, 4.0)};
    const std::array<Vec3, 3> expectedRotation = RelocationMCVelocity::computeRotationBetweenConformations(
            movedSourcePositions, result.newPositions, testBox, 0);
    const bool nonidentityVelocityRotation =
            (expectedRotation[0]-Vec3(1.0, 0.0, 0.0)).dot(expectedRotation[0]-Vec3(1.0, 0.0, 0.0)) > 1e-10 ||
            (expectedRotation[1]-Vec3(0.0, 1.0, 0.0)).dot(expectedRotation[1]-Vec3(0.0, 1.0, 0.0)) > 1e-10 ||
            (expectedRotation[2]-Vec3(0.0, 0.0, 1.0)).dot(expectedRotation[2]-Vec3(0.0, 0.0, 1.0)) > 1e-10;
    require(nonidentityVelocityRotation,
            "Manual multi-site regression did not exercise a nonidentity source-to-candidate rotation.");
    for (int site = 0; site < 3; site++)
        requireVecClose(updatedVelocities[site], sourceCOM+
                (expectedRotation[0]*(sourceVelocities[site]-sourceCOM)[0]+
                 expectedRotation[1]*(sourceVelocities[site]-sourceCOM)[1]+
                 expectedRotation[2]*(sourceVelocities[site]-sourceCOM)[2]), 2e-6,
                "Accepted manual relocation did not apply the exact source-to-candidate velocity rotation.");
    for (const std::pair<int, int>& pair : {std::make_pair(0, 1), std::make_pair(0, 2),
            std::make_pair(1, 2)}) {
        Vec3 sourceBond = initialPositions[movedOffset+pair.second]-initialPositions[movedOffset+pair.first];
        for (int axis = 0; axis < 3; axis++)
            sourceBond[axis] -= std::floor(sourceBond[axis]/4.0+0.5)*4.0;
        const Vec3 sourceRelativeVelocity = sourceVelocities[pair.second]-sourceVelocities[pair.first];
        const Vec3 targetBond = result.newPositions[pair.second]-result.newPositions[pair.first];
        const Vec3 targetRelativeVelocity = updatedVelocities[pair.second]-updatedVelocities[pair.first];
        const Vec3 fittedSourceBond = RelocationMCVelocity::applyRotation(expectedRotation, sourceBond);
        requireClose(fittedSourceBond.dot(targetRelativeVelocity), sourceBond.dot(sourceRelativeVelocity), 5e-6,
                "Best-fit velocity rotation did not preserve its transformed bond velocity dot product.");
        const Vec3 fitError = targetBond-fittedSourceBond;
        const double errorBound = std::sqrt(fitError.dot(fitError)*targetRelativeVelocity.dot(targetRelativeVelocity))+1e-8;
        require(std::fabs(targetBond.dot(targetRelativeVelocity)-sourceBond.dot(sourceRelativeVelocity)) <= errorBound,
                "Accepted-coordinate constraint derivative exceeded its best-fit geometric error bound.");
    }

    integrator.step(2);
    const State state = context.getState(State::Positions | State::Energy);
    require(std::isfinite(state.getPotentialEnergy()),
            "MD after a cross-boundary constrained relocation produced non-finite energy.");
    const vector<Vec3>& positions = state.getPositions();
    for (const Vec3& position : positions)
        require(std::isfinite(position[0]) && std::isfinite(position[1]) && std::isfinite(position[2]),
                "MD after a cross-boundary constrained relocation produced non-finite positions.");
    const int oxygen = result.movedAtomIndices[0];
    const int hydrogen1 = result.movedAtomIndices[1];
    const int hydrogen2 = result.movedAtomIndices[2];
    const Vec3 finalOh1 = positions[hydrogen1]-positions[oxygen];
    const Vec3 finalOh2 = positions[hydrogen2]-positions[oxygen];
    const Vec3 finalHh = positions[hydrogen2]-positions[hydrogen1];
    require(std::fabs(std::sqrt(finalOh1.dot(finalOh1))-oh) < 2e-5 &&
            std::fabs(std::sqrt(finalOh2.dot(finalOh2))-oh) < 2e-5 &&
            std::fabs(std::sqrt(finalHh.dot(finalHh))-hh) < 2e-5,
            "MD did not preserve raw constrained geometry after a cross-boundary relocation.");

    // Exercise the same multi-site strategy through updateContextState().  A
    // no-op CustomIntegrator advances steps without obscuring the MC velocity
    // transform with force integration.
    relocation->setExchangeFrequency(1);
    CustomIntegrator automaticIntegrator(0.001);
    automaticIntegrator.addUpdateContextState();
    Context automaticContext(system, automaticIntegrator, platform, properties);
    automaticContext.setPositions(initialPositions);
    automaticContext.setVelocities(initialVelocities);
    automaticIntegrator.step(1);
    vector<Vec3> beforeAutomaticVelocities, afterAutomaticVelocities;
    int automaticMovedOffset = -1;
    for (int attempt = 0; attempt < 20 && automaticMovedOffset < 0; attempt++) {
        const State before = automaticContext.getState(State::Positions | State::Velocities);
        automaticIntegrator.step(1);
        const State after = automaticContext.getState(State::Positions | State::Velocities);
        int movedThisAttempt = -1;
        for (int offset : {1, 4}) {
            bool changed = false;
            for (int site = 0; site < 3; site++) {
                const Vec3 delta = after.getPositions()[offset+site]-before.getPositions()[offset+site];
                changed |= delta.dot(delta) > 1e-8;
            }
            if (changed) {
                require(movedThisAttempt < 0, "Automatic relocation changed more than one labile molecule.");
                movedThisAttempt = offset;
            }
        }
        if (movedThisAttempt < 0) {
            for (int atom = 0; atom < 7; atom++)
                requireVecClose(after.getVelocities()[atom], before.getVelocities()[atom], 0.0,
                        "Rejected automatic multi-site relocation changed a velocity.");
        }
        else {
            automaticMovedOffset = movedThisAttempt;
            beforeAutomaticVelocities = before.getVelocities();
            afterAutomaticVelocities = after.getVelocities();
        }
    }
    require(automaticMovedOffset > 0, "No automatic multi-site relocation was accepted within 20 fixed-seed attempts.");
    const int automaticUnmovedOffset = automaticMovedOffset == 1 ? 4 : 1;
    requireVecClose(afterAutomaticVelocities[0], beforeAutomaticVelocities[0], 0.0,
            "Automatic multi-site relocation changed a non-labile atom velocity.");
    for (int site = 0; site < 3; site++)
        requireVecClose(afterAutomaticVelocities[automaticUnmovedOffset+site],
                beforeAutomaticVelocities[automaticUnmovedOffset+site], 0.0,
                "Automatic multi-site relocation changed the unselected molecule velocity.");
    const vector<Vec3> automaticSourceVelocities = {
        beforeAutomaticVelocities[automaticMovedOffset],
        beforeAutomaticVelocities[automaticMovedOffset+1],
        beforeAutomaticVelocities[automaticMovedOffset+2]};
    const vector<Vec3> automaticTargetVelocities = {
        afterAutomaticVelocities[automaticMovedOffset],
        afterAutomaticVelocities[automaticMovedOffset+1],
        afterAutomaticVelocities[automaticMovedOffset+2]};
    requireVecClose(massWeightedCOM(waterMasses, automaticTargetVelocities),
            massWeightedCOM(waterMasses, automaticSourceVelocities), 2e-6,
            "Automatic multi-site relocation did not preserve COM velocity.");
    requireClose(relativeKineticEnergy(waterMasses, automaticTargetVelocities),
            relativeKineticEnergy(waterMasses, automaticSourceVelocities), 2e-6,
            "Automatic multi-site relocation did not preserve relative kinetic energy.");
}

void testUnsupportedLabileExceptionsRejected() {
    auto build = [&](double internalChargeProduct, double internalEpsilon, bool crossException) {
        System* system = new System();
        for (int atom = 0; atom < 4; atom++)
            system->addParticle(1.0);
        system->setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
        PerturbNonbondedForce* nonbonded = createNonbonded({0.0, 0.2, -0.2, 0.1});
        nonbonded->addException(1, 2, internalChargeProduct, 0.3, internalEpsilon);
        if (crossException)
            nonbonded->addException(1, 3, 0.0, 0.3, 0.0);
        system->addForce(nonbonded);
        RelocationMC* relocation = new RelocationMC(300.0, 0.75, 0, 4);
        relocation->setProtocolVersion(RelocationMC::Legacy);
        relocation->setLigandAtoms({0});
        relocation->setLabileMolSiteParameters({{0.2, 0.3, 0.0}, {-0.2, 0.3, 0.0}});
        relocation->setLabileMolReferencePositions({Vec3(0, 0, 0), Vec3(0.1, 0, 0)});
        relocation->addLabileMol({1, 2}, 1);
        system->addForce(relocation);
        return system;
    };
    {
        std::unique_ptr<System> system(build(0.0, 0.0, true));
        VerletIntegrator integrator(0.001);
        try {
            Context context(*system, integrator, platform);
        }
        catch (const OpenMMException& error) {
            const string message = error.what();
            require(message.find("exception") != string::npos,
                    "Unsupported labile exception rejection did not explain the exception contract.");
            return;
        }
        throw std::runtime_error("CUDA RelocationMC accepted an unsupported cross-molecule exception.");
    }
}

void testRejectedMovePreservesCoordinates() {
    platform.setPropertyDefaultValue("UseCpuPme", "false");
    const std::string logPath = createTemporaryLogPath("rejected-format");
    System system;
    for (int i = 0; i < 2; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    system.addForce(createNonbonded({0.0, 1e-6}));
    RelocationMC* relocation = new RelocationMC(300.0, 1e-5, 0, 1);
    relocation->setProtocolVersion(RelocationMC::Legacy);
    relocation->setUseCavityBias(true);
    relocation->setRandomSeed(17);
    relocation->setVerbose(true);
    relocation->setLogFileName(logPath);
    relocation->setLigandAtoms({0});
    relocation->setLabileMolSiteParameters({{0.0, 0.3, 0.0}});
    relocation->setLabileMolReferencePositions({Vec3(0, 0, 0)});
    relocation->addLabileMol({1}, 1);
    system.addForce(relocation);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    const vector<Vec3> initial = {Vec3(2, 2, 2), Vec3(0.25, 0.25, 0.25)};
    const vector<Vec3> initialVelocities = {Vec3(1.2, -0.4, 0.3), Vec3(-0.7, 1.1, 0.8)};
    context.setPositions(initial);
    context.setVelocities(initialVelocities);
    const vector<Vec3> velocitiesBefore = context.getState(State::Velocities).getVelocities();
    context.getState(State::Energy);
    ExchangeResult result = relocation->attemptExchange(context);
    require(result.attempted && !result.accepted,
            "Tiny-cavity CUDA proposal should be deterministically rejected.");
    const State finalState = context.getState(State::Positions | State::Velocities);
    const vector<Vec3>& finalPositions = finalState.getPositions();
    const vector<Vec3>& finalVelocities = finalState.getVelocities();
    for (int atom = 0; atom < 2; atom++) {
        const Vec3 delta = finalPositions[atom]-initial[atom];
        require(delta.dot(delta) == 0.0, "A rejected CUDA relocation changed Context positions.");
        requireVecClose(finalVelocities[atom], velocitiesBefore[atom], 0.0,
                "A rejected CUDA relocation changed Context velocities.");
    }
    const std::string log = readTextFile(logPath);
    require(log.find("-----1------\n") == 0,
            "The first rejected attempt did not begin with the exact numbered separator.");
    require(countOccurrences(log, "-----1------\n") == 1,
            "The first rejected attempt wrote the wrong number of separators.");
    require(countOccurrences(log, "attempted=1 accepted=0") == 1,
            "Rejected attempt status was not written exactly once.");
    requireAbsent(log, "initPotentialEnergy_kJ_per_mol=",
            "A rejected attempt logged the initial full Context energy.");
    requireAbsent(log, "updatedPotentialEnergy_kJ_per_mol=",
            "A rejected attempt logged an updated full Context energy.");
    requireAbsent(log, "deleteMoleculeEnergy_kJ_per_mol=",
            "A rejected attempt logged the delete decision energy.");
    requireAbsent(log, "insertMoleculeEnergy_kJ_per_mol=",
            "A rejected attempt logged the insert decision energy.");
    require(!log.empty() && log.back() == '\n', "Rejected attempt log was not newline terminated.");
    std::remove(logPath.c_str());
}

void testNonverboseAttemptDoesNotLog() {
    const std::string logPath = createTemporaryLogPath("nonverbose");
    {
        RelocationMC* relocation;
        std::unique_ptr<System> system = createRefreshTestSystem(relocation, false, logPath);
        VerletIntegrator integrator(0.001);
        Context context(*system, integrator, platform);
        context.setPositions({Vec3(2.0, 2.0, 2.0), Vec3(2.25, 2.0, 2.0),
                              Vec3(0.25, 0.25, 0.25), Vec3(3.2, 3.0, 2.8)});
        relocation->attemptExchange(context);
    }
    std::ifstream nonverboseLog(logPath);
    require(!nonverboseLog.good(), "Nonverbose RelocationMC unexpectedly created a log file.");
    std::remove(logPath.c_str());
}

void testAcceptedAttemptEnergyLog() {
    platform.setPropertyDefaultValue("UseCpuPme", "false");
    const std::string logPath = createTemporaryLogPath("accepted-format");
    RelocationMC* relocation;
    std::unique_ptr<System> system = createRefreshTestSystem(relocation, true, logPath);
    VerletIntegrator integrator(0.001);
    Context context(*system, integrator, platform);
    context.setPositions({Vec3(2.0, 2.0, 2.0), Vec3(2.25, 2.0, 2.0),
                          Vec3(0.25, 0.25, 0.25), Vec3(3.2, 3.0, 2.8)});

    const double initialPotentialEnergy = getPotentialEnergy(context);
    const ExchangeResult first = relocation->attemptExchange(context);
    require(first.attempted && first.accepted,
            "Fixed-seed accepted logging regression did not accept its first proposal.");
    const double updatedPotentialEnergy = getPotentialEnergy(context);

    const std::string firstLog = readTextFile(logPath);
    require(firstLog.find("-----1------\n") == 0,
            "The first accepted attempt did not begin with the exact numbered separator.");
    require(countOccurrences(firstLog, "-----1------\n") == 1,
            "The first accepted attempt wrote the wrong number of separators.");
    require(countOccurrences(firstLog, "attempted=1 accepted=1") == 1,
            "Accepted attempt status was not written exactly once.");
    const std::string initialKey = "initPotentialEnergy_kJ_per_mol=";
    const std::string updatedKey = "updatedPotentialEnergy_kJ_per_mol=";
    const std::string deleteKey = "deleteMoleculeEnergy_kJ_per_mol=";
    const std::string insertKey = "insertMoleculeEnergy_kJ_per_mol=";
    require(countOccurrences(firstLog, initialKey) == 1 &&
            countOccurrences(firstLog, updatedKey) == 1 &&
            countOccurrences(firstLog, deleteKey) == 1 &&
            countOccurrences(firstLog, insertKey) == 1,
            "Accepted attempt did not write each requested energy exactly once.");
    requireClose(getLoggedValue(firstLog, initialKey), initialPotentialEnergy, 2e-12,
            "Logged initial potential energy did not equal the real pre-commit Context energy.");
    requireClose(getLoggedValue(firstLog, updatedKey), updatedPotentialEnergy, 2e-12,
            "Logged updated potential energy did not equal the real post-commit Context energy.");
    requireClose(getLoggedValue(firstLog, deleteKey), first.deletedMoleculeEnergy, 2e-12,
            "Logged delete energy was not the decision energy returned by ExchangeResult.");
    requireClose(getLoggedValue(firstLog, insertKey), first.candidateMoleculeEnergy, 2e-12,
            "Logged insert energy was not the decision energy returned by ExchangeResult.");
    require(!firstLog.empty() && firstLog.back() == '\n',
            "Accepted attempt log was not newline terminated.");

    const ExchangeResult second = relocation->attemptExchange(context);
    require(second.attempted, "Second logging regression proposal was not attempted.");
    const std::string completeLog = readTextFile(logPath);
    require(countOccurrences(completeLog, "-----1------\n") == 1 &&
            countOccurrences(completeLog, "-----2------\n") == 1,
            "Attempt separators did not advance monotonically across calls.");
    require(completeLog.find("-----2------\n") > completeLog.find("-----1------\n"),
            "The second attempt separator was not appended after the first record.");
    require(!completeLog.empty() && completeLog.back() == '\n',
            "Multi-attempt log was not newline terminated.");
    std::remove(logPath.c_str());
}

void testAutomaticFrequencyAndInvalidation() {
    platform.setPropertyDefaultValue("UseCpuPme", "false");
    const std::string logPath = createTemporaryLogPath("automatic-refresh");
    System system;
    for (int i = 0; i < 3; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    system.addForce(createNonbonded({0.0, 1e-6, -1e-6}));
    RelocationMC* relocation = new RelocationMC(300.0, 0.75, 2, 128);
    relocation->setProtocolVersion(RelocationMC::Legacy);
    relocation->setRandomSeed(20260829);
    relocation->setUseCavityBias(false);
    relocation->setVerbose(true);
    relocation->setLogFileName(logPath);
    relocation->setLigandAtoms({0});
    relocation->setLabileMolSiteParameters({{0.0, 0.3, 0.0}});
    relocation->setLabileMolReferencePositions({Vec3(0, 0, 0)});
    relocation->addLabileMol({1}, 1);
    relocation->addLabileMol({2}, 2);
    system.addForce(relocation);

    // Advance step count without modifying velocities, with an explicit
    // updateContextState operation so the due relocation executes at step two.
    CustomIntegrator integrator(0.001);
    integrator.addUpdateContextState();
    Context context(system, integrator, platform);
    const vector<Vec3> initial = {Vec3(2, 2, 2), Vec3(2.25, 2, 2), Vec3(0.25, 0.25, 0.25)};
    const vector<Vec3> initialVelocities = {Vec3(0.3, -0.2, 0.1), Vec3(1.0, 2.0, 3.0),
                                            Vec3(-1.5, 0.7, -0.4)};
    context.setPositions(initial);
    context.setVelocities(initialVelocities);
    require(!relocation->isExchangeDue(context), "Automatic relocation should not run before the first step.");
    integrator.step(1);
    require(!relocation->isExchangeDue(context), "Frequency-two relocation should not be due at step one.");
    integrator.step(1);
    require(relocation->isExchangeDue(context), "Frequency-two relocation should be due at step two.");
    const vector<Vec3> velocitiesBeforeAutomatic = context.getState(State::Velocities).getVelocities();
    integrator.step(1);
    require(!relocation->isExchangeDue(context), "The step-two automatic relocation should execute at most once.");
    const State state = context.getState(State::Positions | State::Velocities | State::Forces | State::Energy);
    require(std::isfinite(state.getPotentialEnergy()),
            "Forces and energy were not recomputed after an automatic accepted move.");
    bool moved = false;
    int movedAtom = -1;
    for (int atom = 1; atom < 3; atom++) {
        const Vec3 delta = state.getPositions()[atom]-initial[atom];
        if (delta.dot(delta) > 1e-8) {
            moved = true;
            movedAtom = atom;
        }
    }
    require(moved, "The fixed-seed automatic CUDA relocation was not accepted.");
    requireVecClose(state.getVelocities()[0], velocitiesBeforeAutomatic[0], 0.0,
            "Automatic accepted relocation changed a non-moved atom velocity.");
    requireVecClose(state.getVelocities()[movedAtom], velocitiesBeforeAutomatic[movedAtom], 0.0,
            "Single-site automatic relocation should preserve its COM velocity.");
    const int otherAtom = movedAtom == 1 ? 2 : 1;
    requireVecClose(state.getVelocities()[otherAtom], velocitiesBeforeAutomatic[otherAtom], 0.0,
            "Automatic accepted relocation changed the other labile molecule velocity.");
    const std::string log = readTextFile(logPath);
    require(countOccurrences(log, "-----1------\n") == 1,
            "Automatic relocation did not write exactly one numbered attempt separator.");
    require(countOccurrences(log, "attempted=") == 1,
            "Automatic relocation produced more than one attempt log entry at one due step.");
    require(countOccurrences(log, "initPotentialEnergy_kJ_per_mol=") == 1 &&
            countOccurrences(log, "updatedPotentialEnergy_kJ_per_mol=") == 1 &&
            countOccurrences(log, "deleteMoleculeEnergy_kJ_per_mol=") == 1 &&
            countOccurrences(log, "insertMoleculeEnergy_kJ_per_mol=") == 1,
            "Automatic accepted relocation did not write all four energy fields.");
    std::remove(logPath.c_str());
}

void benchmarkRelocationAttemptScaling() {
    for (int candidateSites : {1, 4, 16, 32}) {
        System system;
        const int numMolecules = 16;
        const int numParticles = 1+numMolecules;
        for (int atom = 0; atom < numParticles; atom++)
            system.addParticle(1.0);
        system.setDefaultPeriodicBoxVectors(Vec3(8, 0, 0), Vec3(0, 8, 0), Vec3(0, 0, 8));
        vector<double> charges(numParticles, 0.0);
        for (int atom = 1; atom < numParticles; atom++)
            charges[atom] = atom%2 == 0 ? 1e-6 : -1e-6;
        system.addForce(createNonbonded(charges));
        RelocationMC* relocation = new RelocationMC(300.0, 1.0, 0, candidateSites);
        relocation->setProtocolVersion(RelocationMC::Legacy);
        relocation->setRandomSeed(20260829);
        relocation->setUseCavityBias(false);
        relocation->setLigandAtoms({0});
        relocation->setLabileMolSiteParameters({{1e-6, 0.3, 0.0}});
        relocation->setLabileMolReferencePositions({Vec3(0, 0, 0)});
        for (int molecule = 0; molecule < numMolecules; molecule++)
            relocation->addLabileMol({molecule+1}, molecule+1);
        system.addForce(relocation);
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, platform);
        vector<Vec3> positions(numParticles);
        positions[0] = Vec3(4, 4, 4);
        for (int atom = 1; atom < numParticles; atom++)
            positions[atom] = atom <= numMolecules/2 ? Vec3(4+0.04*atom, 4, 4) :
                    Vec3(0.3+0.35*(atom-numMolecules/2), 0.5, 0.5);
        context.setPositions(positions);
        context.getState(State::Energy);
        relocation->attemptExchange(context);
        vector<double> timings;
        for (int sample = 0; sample < 7; sample++) {
            context.getState(State::Energy);
            const auto start = std::chrono::steady_clock::now();
            relocation->attemptExchange(context);
            context.getState(State::Energy);
            const auto stop = std::chrono::steady_clock::now();
            timings.push_back(std::chrono::duration<double, std::micro>(stop-start).count());
        }
        std::sort(timings.begin(), timings.end());
        std::cout << "Relocation attempt benchmark: sites=" << candidateSites
                  << " orientations=60 K=" << candidateSites*60
                  << " particles=" << numParticles << " labileMolecules=" << numMolecules
                  << " DisablePmeStream=" << platform.getPropertyValue(context, "DisablePmeStream")
                  << " median_us=" << timings[timings.size()/2] << std::endl;
    }
}

void benchmarkSameContextMdAndRelocation() {
    const int candidateSites = 16;
    System system;
    const int numMolecules = 16;
    const int numParticles = 1+numMolecules;
    for (int atom = 0; atom < numParticles; atom++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(8, 0, 0), Vec3(0, 8, 0), Vec3(0, 0, 8));
    vector<double> charges(numParticles, 0.0);
    for (int atom = 1; atom < numParticles; atom++)
        charges[atom] = 1e-6;
    system.addForce(createNonbonded(charges));
    RelocationMC* relocation = new RelocationMC(300.0, 1.0, 0, candidateSites);
    relocation->setRandomSeed(20260829);
    relocation->setUseCavityBias(false);
    relocation->setProtocolVersion(RelocationMC::MTM);
    relocation->setLigandAtoms({0});
    relocation->setLabileMolSiteParameters({{1e-6, 0.3, 0.0}});
    relocation->setLabileMolReferencePositions({Vec3(0, 0, 0)});
    for (int molecule = 0; molecule < numMolecules; molecule++)
        relocation->addLabileMol({molecule+1}, molecule+1);
    system.addForce(relocation);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    vector<Vec3> positions(numParticles);
    positions[0] = Vec3(4, 4, 4);
    for (int atom = 1; atom < numParticles; atom++)
        positions[atom] = atom <= numMolecules/2 ? Vec3(4+0.04*atom, 4, 4) :
                Vec3(0.3+0.35*(atom-numMolecules/2), 0.5, 0.5);
    context.setPositions(positions);
    integrator.step(10);
    vector<double> mdTimings, mcTimings;
    for (int sample = 0; sample < 11; sample++) {
        auto start = std::chrono::steady_clock::now();
        integrator.step(10);
        context.getState(State::Energy);
        auto stop = std::chrono::steady_clock::now();
        mdTimings.push_back(std::chrono::duration<double, std::micro>(stop-start).count()/10.0);
        start = std::chrono::steady_clock::now();
        relocation->attemptExchange(context);
        context.getState(State::Energy);
        stop = std::chrono::steady_clock::now();
        mcTimings.push_back(std::chrono::duration<double, std::micro>(stop-start).count());
    }
    std::sort(mdTimings.begin(), mdTimings.end());
    std::sort(mcTimings.begin(), mcTimings.end());
    const double md = mdTimings[mdTimings.size()/2];
    const double mc = mcTimings[mcTimings.size()/2];
    std::cout << "Same-context MD/MC benchmark: particles=" << numParticles
              << " labileMolecules=" << numMolecules << " sites=" << candidateSites
              << " protocol=MTM K=" << candidateSites
              << " DisablePmeStream=" << platform.getPropertyValue(context, "DisablePmeStream")
              << " md_us_per_step=" << md << " mc_attempt_us=" << mc
              << " ratio=" << mc/md << " amortized_frequency_1000=" << mc/(1000*md) << std::endl;
}

void benchmarkDefaultMdPmeStream() {
    System system;
    const int numParticles = 2048;
    const double boxSize = 16.0;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    PerturbNonbondedForce* nonbonded = createNonbonded(vector<double>(numParticles, 0.0));
    for (int i = 0; i < numParticles; i++)
        nonbonded->setParticleParameters(i, i%2 == 0 ? 0.1 : -0.1, 0.3, 0.0,
                                         i%2 == 0 ? 0.1 : -0.1, 0.3, 0.0, false, false);
    system.addForce(nonbonded);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++) {
        const int x = i%16, y = (i/16)%16, z = i/256;
        positions[i] = Vec3((x+0.25)*boxSize/16, (y+0.25)*boxSize/16, (z+0.25)*boxSize/8);
    }
    context.setPositions(positions);
    integrator.step(10);
    vector<double> timings, refreshTimings;
    for (int sample = 0; sample < 51; sample++) {
        auto start = std::chrono::steady_clock::now();
        integrator.step(10);
        context.getState(State::Energy);
        auto stop = std::chrono::steady_clock::now();
        timings.push_back(std::chrono::duration<double, std::micro>(stop-start).count()/10.0);
        start = std::chrono::steady_clock::now();
        context.getState(State::Energy);
        stop = std::chrono::steady_clock::now();
        refreshTimings.push_back(std::chrono::duration<double, std::micro>(stop-start).count());
    }
    std::sort(timings.begin(), timings.end());
    std::sort(refreshTimings.begin(), refreshTimings.end());
    const double md = timings[timings.size()/2];
    const double refresh = refreshTimings[refreshTimings.size()/2];
    std::cout << "Normal MD PME benchmark: DisablePmeStream="
              << platform.getPropertyValue(context, "DisablePmeStream")
              << " particles=" << numParticles << " median_us_per_step=" << md
              << " full_energy_refresh_median_us=" << refresh
              << " refresh_step_equivalents=" << refresh/md << std::endl;
}

void testTriclinicRejectedOnCuda() {
    platform.setPropertyDefaultValue("UseCpuPme", "false");
    System system;
    for (int i = 0; i < 3; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0.2, 3, 0), Vec3(0, 0, 3));
    system.addForce(createNonbonded({0.0, 0.0, 0.0}));
    RelocationMC* relocation = new RelocationMC(300.0, 0.5, 0, 4);
    relocation->setLigandAtoms({0});
    relocation->setLabileMolSiteParameters({{0.0, 0.3, 0.0}});
    relocation->setLabileMolReferencePositions({Vec3(0, 0, 0)});
    relocation->addLabileMol({1}, 1);
    relocation->addLabileMol({2}, 2);
    system.addForce(relocation);
    VerletIntegrator integrator(0.001);
    try {
        Context context(system, integrator, platform);
    }
    catch (const std::exception& error) {
        require(std::string(error.what()).find("orthorhombic") != std::string::npos,
                std::string("Triclinic rejection did not explain the orthorhombic requirement; got: ")+error.what());
        return;
    }
    throw std::runtime_error("CUDA RelocationMC accepted a triclinic box.");
}

void testMtmContextPrecisionPositionChain() {
    System system;
    for (int atom = 0; atom < 4; atom++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    PerturbNonbondedForce* nonbonded = createNonbonded({0.0, 0.0, 0.0, 0.0});
    system.addForce(nonbonded);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    const vector<Vec3> input = {
        Vec3(1.000000000123, 1.999999999731, 3.000000000357),
        Vec3(3.900000000111, 2.100000000223, 1.700000000337),
        Vec3(0.100000000419, 2.100000000521, 1.700000000633),
        Vec3(3.900000000747, 2.300000000859, 1.700000000961)};
    context.setPositions(input);
    const vector<Vec3> effectiveInput = context.getState(State::Positions).getPositions();
    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(
            nonbonded->getCudaPlatformData(context));
    CudaContext& cu = *data.contexts[0];
    ContextSelector selector(cu);
    std::map<std::string, std::string> defines;
    defines["USE_PERIODIC"] = "1";
    defines["SQRT_ONE_4PI_EPS0"] = "1";
    CUmodule module = cu.createModule(CudaKernelSources::vectorOps+
            CudaRelocationMCKernelSources::relocation_mc, defines);
    CUfunction reorder = cu.getKernel(module, "reorderMtmPositionsContextPrecision");
    CUfunction gather = cu.getKernel(module, "gatherMtmRealPositionsContextPrecision");
    CUfunction gatherSelected = cu.getKernel(module, "gatherMtmSelectedMoleculeContextPrecision");
    CUfunction pack = cu.getKernel(module, "packMtmDeletedQueriesContextPrecision");
    CUfunction generate = cu.getKernel(module, "generateMtmCandidatePositionsContextPrecision");

    CudaArray orderedBase, orderedCorrection, gatheredBase, gatheredCorrection;
    CudaArray selectedBase, selectedCorrection, packedBase, packedCorrection;
    CudaArray candidateAnchorBase, candidateAnchorCorrection, candidateBase, candidateCorrection, candidateRegions;
    const int precisionCandidates = 64;
    candidateRegions.initialize<int>(cu, precisionCandidates, "precisionOracleCandidateRegions");
    if (cu.getUseDoublePrecision()) {
        orderedBase.initialize<mm_double4>(cu, cu.getPaddedNumAtoms(), "precisionOracleOrderedBase");
        gatheredBase.initialize<mm_double4>(cu, 3, "precisionOracleGatheredBase");
        selectedBase.initialize<mm_double4>(cu, 3, "precisionOracleSelectedBase");
        packedBase.initialize<mm_double4>(cu, 6, "precisionOraclePackedBase");
        candidateAnchorBase.initialize<mm_double4>(cu, precisionCandidates, "precisionOracleCandidateAnchors");
        candidateBase.initialize<mm_double4>(cu, precisionCandidates*3, "precisionOracleCandidates");
    }
    else {
        orderedBase.initialize<mm_float4>(cu, cu.getPaddedNumAtoms(), "precisionOracleOrderedBase");
        gatheredBase.initialize<mm_float4>(cu, 3, "precisionOracleGatheredBase");
        selectedBase.initialize<mm_float4>(cu, 3, "precisionOracleSelectedBase");
        packedBase.initialize<mm_float4>(cu, 6, "precisionOraclePackedBase");
        candidateAnchorBase.initialize<mm_float4>(cu, precisionCandidates, "precisionOracleCandidateAnchors");
        candidateBase.initialize<mm_float4>(cu, precisionCandidates*3, "precisionOracleCandidates");
    }
    CUdeviceptr nullCorrection = 0;
    CUdeviceptr contextCorrection = cu.getUseMixedPrecision() ? cu.getPosqCorrection().getDevicePointer() : nullCorrection;
    CUdeviceptr orderedCorrectionPointer = nullCorrection;
    CUdeviceptr gatheredCorrectionPointer = nullCorrection;
    CUdeviceptr selectedCorrectionPointer = nullCorrection;
    CUdeviceptr packedCorrectionPointer = nullCorrection;
    CUdeviceptr candidateAnchorCorrectionPointer = nullCorrection;
    CUdeviceptr candidateCorrectionPointer = nullCorrection;
    if (cu.getUseMixedPrecision()) {
        orderedCorrection.initialize<mm_float4>(cu, cu.getPaddedNumAtoms(), "precisionOracleOrderedCorrection");
        gatheredCorrection.initialize<mm_float4>(cu, 3, "precisionOracleGatheredCorrection");
        selectedCorrection.initialize<mm_float4>(cu, 3, "precisionOracleSelectedCorrection");
        packedCorrection.initialize<mm_float4>(cu, 6, "precisionOraclePackedCorrection");
        candidateAnchorCorrection.initialize<mm_float4>(cu, precisionCandidates, "precisionOracleCandidateAnchorCorrection");
        candidateCorrection.initialize<mm_float4>(cu, precisionCandidates*3, "precisionOracleCandidateCorrection");
        orderedCorrectionPointer = orderedCorrection.getDevicePointer();
        gatheredCorrectionPointer = gatheredCorrection.getDevicePointer();
        selectedCorrectionPointer = selectedCorrection.getDevicePointer();
        packedCorrectionPointer = packedCorrection.getDevicePointer();
        candidateAnchorCorrectionPointer = candidateAnchorCorrection.getDevicePointer();
        candidateCorrectionPointer = candidateCorrection.getDevicePointer();
    }
    int numAtoms = 4;
    void* reorderArgs[] = {&numAtoms, &cu.getPosq().getDevicePointer(), &contextCorrection,
            &cu.getAtomIndexArray().getDevicePointer(), &orderedBase.getDevicePointer(),
            &orderedCorrectionPointer};
    cu.executeKernel(reorder, reorderArgs, numAtoms);
    CudaArray indices, parameters, charges;
    indices.initialize<int>(cu, 3, "precisionOracleIndices");
    parameters.initialize<mm_float4>(cu, 3, "precisionOracleParameters");
    charges.initialize<float>(cu, 3, "precisionOracleCharges");
    indices.upload(vector<int>{1, 2, 3});
    parameters.upload(vector<mm_float4>(3, mm_float4(0, 0, 0, 0)));
    int count = 3;
    void* gatherArgs[] = {&count, &indices.getDevicePointer(), &orderedBase.getDevicePointer(),
            &orderedCorrectionPointer, &gatheredBase.getDevicePointer(), &gatheredCorrectionPointer};
    cu.executeKernel(gather, gatherArgs, count);
    int selectedMolecule = 0;
    void* selectedArgs[] = {&count, &selectedMolecule, &gatheredBase.getDevicePointer(),
            &gatheredCorrectionPointer, &parameters.getDevicePointer(), &selectedBase.getDevicePointer(),
            &selectedCorrectionPointer, &charges.getDevicePointer()};
    cu.executeKernel(gatherSelected, selectedArgs, count);

    int referenceSite = 0;
    float radiusSquaredFloat = 1.0f;
    double radiusSquaredDouble = 1.0;
    void* radiusSquaredPointer = cu.getUseMixedPrecision() || cu.getUseDoublePrecision() ?
            (void*) &radiusSquaredDouble : (void*) &radiusSquaredFloat;
    unsigned long long seed = 918273645ULL;
    unsigned long long attempt = 7;
    void* generateArgs[] = {const_cast<int*>(&precisionCandidates), &count, &referenceSite,
            radiusSquaredPointer, &seed, &attempt, &selectedBase.getDevicePointer(),
            &selectedCorrectionPointer, &selectedBase.getDevicePointer(), &selectedCorrectionPointer,
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &candidateAnchorBase.getDevicePointer(), &candidateAnchorCorrectionPointer,
            &candidateRegions.getDevicePointer(), &candidateBase.getDevicePointer(),
            &candidateCorrectionPointer};
    cu.executeKernel(generate, generateArgs, precisionCandidates);

    int candidateAtoms = 0;
    int realAtoms = 3;
    void* packArgs[] = {&count, &candidateAtoms, &realAtoms,
            &selectedBase.getDevicePointer(), &selectedCorrectionPointer,
            &selectedBase.getDevicePointer(), &selectedCorrectionPointer,
            &gatheredBase.getDevicePointer(), &gatheredCorrectionPointer,
            &packedBase.getDevicePointer(), &packedCorrectionPointer};
    cu.executeKernel(pack, packArgs, 6);

    vector<Vec3> actual(6);
    if (cu.getUseDoublePrecision()) {
        vector<mm_double4> values;
        packedBase.download(values);
        for (int i = 0; i < 6; i++)
            actual[i] = Vec3(values[i].x, values[i].y, values[i].z);
    }
    else {
        vector<mm_float4> values;
        packedBase.download(values);
        vector<mm_float4> corrections(6, mm_float4(0, 0, 0, 0));
        if (cu.getUseMixedPrecision())
            packedCorrection.download(corrections);
        for (int i = 0; i < 6; i++)
            actual[i] = Vec3((double) values[i].x+corrections[i].x,
                    (double) values[i].y+corrections[i].y,
                    (double) values[i].z+corrections[i].z);
    }
    const double tolerance = cu.getUseDoublePrecision() || cu.getUseMixedPrecision() ? 2e-12 : 2e-6;
    if (cu.getUseMixedPrecision()) {
        vector<mm_float4> rawBase, rawCorrection, gatheredBaseValues, gatheredCorrectionValues;
        vector<int> atomOrder;
        cu.getPosq().download(rawBase);
        cu.getPosqCorrection().download(rawCorrection);
        cu.getAtomIndexArray().download(atomOrder);
        gatheredBase.download(gatheredBaseValues);
        gatheredCorrection.download(gatheredCorrectionValues);
        std::cout << std::setprecision(17);
        for (int host = 1; host <= 3; host++) {
            int internal = -1;
            for (int index = 0; index < numAtoms; index++)
                if (atomOrder[index] == host)
                    internal = index;
            const mm_float4 rb = rawBase[internal];
            const mm_float4 rc = rawCorrection[internal];
            const mm_float4 gb = gatheredBaseValues[host-1];
            const mm_float4 gc = gatheredCorrectionValues[host-1];
            std::cout << "MTM mixed position chain host=" << host << " internal=" << internal
                      << " rawBase=(" << rb.x << ',' << rb.y << ',' << rb.z << ')'
                      << " rawCorrection=(" << rc.x << ',' << rc.y << ',' << rc.z << ')'
                      << " rawSum=(" << (double)rb.x+rc.x << ',' << (double)rb.y+rc.y << ',' << (double)rb.z+rc.z << ')'
                      << " gatheredBase=(" << gb.x << ',' << gb.y << ',' << gb.z << ')'
                      << " gatheredCorrection=(" << gc.x << ',' << gc.y << ',' << gc.z << ')'
                      << " reconstructed=(" << actual[host-1][0] << ',' << actual[host-1][1] << ',' << actual[host-1][2] << ')'
                      << " expectedEffective=(" << effectiveInput[host][0] << ',' << effectiveInput[host][1]
                      << ',' << effectiveInput[host][2] << ')' << std::endl;
        }
    }
    for (int i = 0; i < 3; i++) {
        requireVecClose(actual[i], effectiveInput[i+1], tolerance,
                "Context-precision selected-D chain did not reconstruct the actual Context coordinate.");
        requireVecClose(actual[i+3], effectiveInput[i+1], tolerance,
                "Context-precision packed query did not preserve the selected-D coordinate.");
    }
    vector<Vec3> generated(precisionCandidates*3);
    if (cu.getUseDoublePrecision()) {
        vector<mm_double4> values;
        candidateBase.download(values);
        for (int i = 0; i < (int) values.size(); i++)
            generated[i] = Vec3(values[i].x, values[i].y, values[i].z);
    }
    else {
        vector<mm_float4> values;
        candidateBase.download(values);
        vector<mm_float4> corrections(values.size(), mm_float4(0, 0, 0, 0));
        if (cu.getUseMixedPrecision())
            candidateCorrection.download(corrections);
        for (int i = 0; i < (int) values.size(); i++)
            generated[i] = Vec3((double) values[i].x+corrections[i].x,
                    (double) values[i].y+corrections[i].y,
                    (double) values[i].z+corrections[i].z);
    }
    auto sourceDistance = [&](int first, int second) {
        Vec3 delta = effectiveInput[first+1]-effectiveInput[second+1];
        for (int axis = 0; axis < 3; axis++)
            delta[axis] -= std::floor(delta[axis]/4.0+0.5)*4.0;
        return std::sqrt(delta.dot(delta));
    };
    const double source01 = sourceDistance(0, 1);
    const double source02 = sourceDistance(0, 2);
    const double source12 = sourceDistance(1, 2);
    for (int candidate = 0; candidate < precisionCandidates; candidate++) {
        const Vec3& p0 = generated[candidate*3];
        const Vec3& p1 = generated[candidate*3+1];
        const Vec3& p2 = generated[candidate*3+2];
        const Vec3 d01 = p1-p0;
        const Vec3 d02 = p2-p0;
        const Vec3 d12 = p2-p1;
        requireClose(std::sqrt(d01.dot(d01)), source01, tolerance*10,
                "Context-precision Haar generation changed source pair distance 0-1.");
        requireClose(std::sqrt(d02.dot(d02)), source02, tolerance*10,
                "Context-precision Haar generation changed source pair distance 0-2.");
        requireClose(std::sqrt(d12.dot(d12)), source12, tolerance*10,
                "Context-precision Haar generation changed source pair distance 1-2.");
    }
    if (cu.getUseMixedPrecision()) {
        vector<mm_float4> corrections;
        packedCorrection.download(corrections);
        bool hasCorrection = false;
        for (const mm_float4& value : corrections)
            hasCorrection = hasCorrection || value.x != 0.0f || value.y != 0.0f || value.z != 0.0f;
        require(hasCorrection, "Mixed-precision position chain silently discarded all correction bits.");
    }
}

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        const bool loggingOnly = std::getenv("RELOCATIONMC_LOGGING_ONLY") != nullptr;
        const bool nonfiniteOnly = std::getenv("RELOCATIONMC_NONFINITE_ONLY") != nullptr;
        const bool runSlow = std::getenv("RELOCATIONMC_RUN_SLOW_TESTS") != nullptr;
        const bool benchmarkOnly = std::getenv("RELOCATIONMC_BENCHMARK_ONLY") != nullptr;
        if (benchmarkOnly) {
            benchmarkSameContextMdAndRelocation();
            return 0;
        }
        if (nonfiniteOnly) {
            testGpuNonfiniteProposalsFailClosed();
            testNonfiniteDecisionPreservesContextState();
            return 0;
        }
        if (loggingOnly) {
            testRejectedMovePreservesCoordinates();
            testNonverboseAttemptDoesNotLog();
            testAcceptedAttemptEnergyLog();
            testAutomaticFrequencyAndInvalidation();
            return 0;
        }
        testPreparedOrientations();
        testGpuProposalOracles();
        testMtmContextPrecisionPositionChain();
        testMtmGpuOracles();
        testGpuNonfiniteProposalsFailClosed();
        testNonfiniteDecisionPreservesContextState();
        testMinimalCudaProtocol();
        testMtmFullAttemptAndStateSafety();
        testMtmUnsupportedConfigurationsRejected();
        testInnerCandidateRigidGeometryAcrossBoundary();
        testUnsupportedLabileExceptionsRejected();
        testRejectedMovePreservesCoordinates();
        testNonverboseAttemptDoesNotLog();
        testAcceptedAttemptEnergyLog();
        testAutomaticFrequencyAndInvalidation();
        if (runSlow) {
            benchmarkRelocationAttemptScaling();
            benchmarkSameContextMdAndRelocation();
            benchmarkDefaultMdPmeStream();
        }
        testTriclinicRejectedOnCuda();
        std::cout << "Done" << std::endl;
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "exception: " << error.what() << std::endl;
        return 1;
    }
}
