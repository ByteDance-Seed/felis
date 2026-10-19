/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2024 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "TestCudaPerturbNonbondedForce.h"
#include "TestPerturbNonbondedForce.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/XmlSerializer.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/internal/ContextImpl.h"
#include <cuda.h>
#include <algorithm>
#include <string>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
//#include <gperftools/profiler.h>

static CudaArrayDevicePointer uploadContextPrecisionPositions(CudaContext& cu,
        const std::vector<mm_float4>& positions, CudaArray& positionBase,
        CudaArray& positionCorrection, const std::string& name,
        const std::vector<mm_double4>* doublePrecisionPositions = NULL) {
    if (cu.getUseDoublePrecision()) {
        positionBase.initialize<mm_double4>(cu, positions.size(), name+"Base");
        if (doublePrecisionPositions != NULL) {
            ASSERT_EQUAL(positions.size(), doublePrecisionPositions->size());
            positionBase.upload(*doublePrecisionPositions);
        }
        else {
            std::vector<mm_double4> values(positions.size());
            for (int i = 0; i < (int) positions.size(); i++)
                values[i] = mm_double4(positions[i].x, positions[i].y, positions[i].z, positions[i].w);
            positionBase.upload(values);
        }
        return 0;
    }
    positionBase.initialize<mm_float4>(cu, positions.size(), name+"Base");
    if (!cu.getUseMixedPrecision()) {
        positionBase.upload(positions);
        return 0;
    }
    positionCorrection.initialize<mm_float4>(cu, positions.size(), name+"Correction");
    std::vector<mm_float4> baseValues(positions.size()), correctionValues(positions.size());
    for (int i = 0; i < (int) positions.size(); i++) {
        const float x = std::floor(positions[i].x*2.0f)*0.5f;
        const float y = std::floor(positions[i].y*2.0f)*0.5f;
        const float z = std::floor(positions[i].z*2.0f)*0.5f;
        baseValues[i] = mm_float4(x, y, z, positions[i].w);
        correctionValues[i] = mm_float4(positions[i].x-x, positions[i].y-y,
                positions[i].z-z, 0.0f);
    }
    positionBase.upload(baseValues);
    positionCorrection.upload(correctionValues);
    return positionCorrection.getDevicePointer();
}

static void assertInvalidContextPrecisionCorrectionThrows(const std::function<void()>& operation) {
    bool threw = false;
    try {
        operation();
    }
    catch (const OpenMMException&) {
        threw = true;
    }
    ASSERT(threw);
}

void testLargeSystem() {
    const int numMolecules = 1600;
    const int numParticles = numMolecules*2;
    const double cutoff = 2.0;
    const double boxSize = 40.0;
    const double tol = 2e-3;
    ReferencePlatform reference;
    // Register the kernel factory with the Reference platform.
    ReferencePerturbNonbondedKernelFactory* factory = new ReferencePerturbNonbondedKernelFactory();
    reference.registerKernelFactory(PerturbNonbondedPlugin::CalcPerturbNonbondedForceKernel::Name(), factory);
    
    System system;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    PerturbNonbondedForce* nonbonded = new PerturbNonbondedForce();
    HarmonicBondForce* bonds = new HarmonicBondForce();
    vector<Vec3> positions(numParticles);
    vector<Vec3> velocities(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);

    for (int i = 0; i < numMolecules; i++) {
        if (i < numMolecules/2) {
            nonbonded->addParticle(-1.0, 0.2, 0.1, 0, 1.0, 0.0, false, false);
            nonbonded->addParticle(1.0, 0.1, 0.1, 0, 1.0, 0.0, false, false);
        }
        else {
            nonbonded->addParticle(-1.0, 0.2, 0.2, 0, 1.0, 0.0, false, false);
            nonbonded->addParticle(1.0, 0.1, 0.2, 0, 1.0, 0.0, false, false);
        }
        positions[2*i] = Vec3(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
        positions[2*i+1] = Vec3(positions[2*i][0]+1.0, positions[2*i][1], positions[2*i][2]);
        velocities[2*i] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt));
        velocities[2*i+1] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt));
        bonds->addBond(2*i, 2*i+1, 1.0, 0.1);
        nonbonded->addException(2*i, 2*i+1, 0.0, 0.15, 0.0);
    }

    nonbonded->setSoftcoreAlpha(1.0);
    nonbonded->addGlobalParameter("lambdaA_charge", 1.0);
    nonbonded->addGlobalParameter("lambdaB_charge", 0.0);
    nonbonded->addGlobalParameter("lambdaC_charge", 0.0);
    nonbonded->addGlobalParameter("rest2_scale", 1.0);
    nonbonded->addGlobalParameter("lambdaA_vdw", 1.0);
    nonbonded->addGlobalParameter("lambdaB_vdw", 0.0);
    // Try with no cutoffs and make sure it agrees with the Reference platform.

    nonbonded->setNonbondedMethod(PerturbNonbondedForce::NoCutoff);
    system.addForce(nonbonded);
    system.addForce(bonds);
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context context(system, integrator1, platform);
    Context referenceContext(system, integrator2, reference);
    context.setPositions(positions);
    context.setVelocities(velocities);
    referenceContext.setPositions(positions);
    referenceContext.setVelocities(velocities);
    State state = context.getState(State::Positions | State::Velocities | State::Forces | State::Energy);
    State referenceState = referenceContext.getState(State::Positions | State::Velocities | State::Forces | State::Energy);
    for (int i = 0; i < numParticles; i++) {
        ASSERT_EQUAL_VEC(state.getPositions()[i], referenceState.getPositions()[i], tol);
        ASSERT_EQUAL_VEC(state.getVelocities()[i], referenceState.getVelocities()[i], tol);
        ASSERT_EQUAL_VEC(state.getForces()[i], referenceState.getForces()[i], tol);
    }
    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), referenceState.getPotentialEnergy(), tol);

    // Now try cutoffs but not periodic boundary conditions.

    nonbonded->setNonbondedMethod(PerturbNonbondedForce::CutoffNonPeriodic);
    nonbonded->setCutoffDistance(cutoff);
    context.reinitialize(true);
    referenceContext.reinitialize(true);
    state = context.getState(State::Positions | State::Velocities | State::Forces | State::Energy);
    referenceState = referenceContext.getState(State::Positions | State::Velocities | State::Forces | State::Energy);
    for (int i = 0; i < numParticles; i++) {
        ASSERT_EQUAL_VEC(state.getPositions()[i], referenceState.getPositions()[i], tol);
        ASSERT_EQUAL_VEC(state.getVelocities()[i], referenceState.getVelocities()[i], tol);
        ASSERT_EQUAL_VEC(state.getForces()[i], referenceState.getForces()[i], tol);
    }
    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), referenceState.getPotentialEnergy(), tol);

    // Now do the same thing with periodic boundary conditions.

    nonbonded->setNonbondedMethod(PerturbNonbondedForce::CutoffPeriodic);
    context.reinitialize(true);
    referenceContext.reinitialize(true);
    state = context.getState(State::Positions | State::Velocities | State::Forces | State::Energy);
    referenceState = referenceContext.getState(State::Positions | State::Velocities | State::Forces | State::Energy);
    for (int i = 0; i < numParticles; i++) {
        double dx = state.getPositions()[i][0]-referenceState.getPositions()[i][0];
        double dy = state.getPositions()[i][1]-referenceState.getPositions()[i][1];
        double dz = state.getPositions()[i][2]-referenceState.getPositions()[i][2];
        ASSERT_EQUAL_TOL(dx-floor(dx/boxSize+0.5)*boxSize, 0, tol);
        ASSERT_EQUAL_TOL(dy-floor(dy/boxSize+0.5)*boxSize, 0, tol);
        ASSERT_EQUAL_TOL(dz-floor(dz/boxSize+0.5)*boxSize, 0, tol);
        ASSERT_EQUAL_VEC(state.getVelocities()[i], referenceState.getVelocities()[i], tol);
        ASSERT_EQUAL_VEC(state.getForces()[i], referenceState.getForces()[i], tol);
    }
    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), referenceState.getPotentialEnergy(), tol);
}


void testChangingParameters() {
    const int numMolecules = 600;
    const int numParticles = numMolecules*2;
    const double cutoff = 2.0;
    const double boxSize = 20.0;
    const double tol = 2e-3;
    ReferencePlatform reference;
    // Register the kernel factory with the Reference platform.
    ReferencePerturbNonbondedKernelFactory* factory = new ReferencePerturbNonbondedKernelFactory();
    reference.registerKernelFactory(PerturbNonbondedPlugin::CalcPerturbNonbondedForceKernel::Name(), factory);
    
    System system;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    PerturbNonbondedForce* nonbonded = new PerturbNonbondedForce();
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);

    for (int i = 0; i < numMolecules; i++) {
        if (i < numMolecules/2) {
            nonbonded->addParticle(-1.0, 0.2, 0.1, 0, 1.0, 0.0, false, false);
            nonbonded->addParticle(1.0, 0.1, 0.1, 0, 1.0, 0.0, false, false);
        }
        else {
            nonbonded->addParticle(-1.0, 0.2, 0.2, 0, 1.0, 0.0, false, false);
            nonbonded->addParticle(1.0, 0.1, 0.2, 0, 1.0, 0.0, false, false);
        }
        positions[2*i] = Vec3(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
        positions[2*i+1] = Vec3(positions[2*i][0]+1.0, positions[2*i][1], positions[2*i][2]);
        system.addConstraint(2*i, 2*i+1, 1.0);
        nonbonded->addException(2*i, 2*i+1, 0.0, 0.15, 0.0);
    }
    nonbonded->setNonbondedMethod(PerturbNonbondedForce::PME);
    nonbonded->setCutoffDistance(cutoff);
    nonbonded->setSoftcoreAlpha(1.0);
    nonbonded->addGlobalParameter("lambdaA_charge", 1.0);
    nonbonded->addGlobalParameter("lambdaB_charge", 0.0);
    nonbonded->addGlobalParameter("lambdaC_charge", 0.0);
    nonbonded->addGlobalParameter("rest2_scale", 1.0);
    nonbonded->addGlobalParameter("lambdaA_vdw", 1.0);
    nonbonded->addGlobalParameter("lambdaB_vdw", 0.0);

    system.addForce(nonbonded);
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    
    // See if the forces and energies match the Reference platform.
    
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context context(system, integrator1, platform);
    Context referenceContext(system, integrator2, reference);
    context.setPositions(positions);
    referenceContext.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    State referenceState = referenceContext.getState(State::Forces | State::Energy);
    for (int i = 0; i < numParticles; i++)
        ASSERT_EQUAL_VEC(state.getForces()[i], referenceState.getForces()[i], tol);
    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), referenceState.getPotentialEnergy(), tol);
    
    // Now modify parameters and see if they still agree.

    for (int i = 0; i < numParticles; i += 5) {
        double chargeA, sigmaA, epsilonA;
        double chargeB, sigmaB, epsilonB;
        bool isDummy, isREST2;

        nonbonded->getParticleParameters(i, chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2);
        nonbonded->setParticleParameters(i, 1.5*chargeA, 1.1*sigmaA, 1.7*epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2);
    }
    nonbonded->updateParametersInContext(context);
    nonbonded->updateParametersInContext(referenceContext);
    state = context.getState(State::Forces | State::Energy);
    referenceState = referenceContext.getState(State::Forces | State::Energy);
    for (int i = 0; i < numParticles; i++)
        ASSERT_EQUAL_VEC(state.getForces()[i], referenceState.getForces()[i], tol);
    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), referenceState.getPotentialEnergy(), tol);
}

void testDeterministicForces() {
    // Check that the CudaDeterministicForces property works correctly.
    
    const int numParticles = 1000;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(6, 0, 0), Vec3(2.1, 6, 0), Vec3(-1.5, -0.5, 6));
    PerturbNonbondedForce *nonbonded = new PerturbNonbondedForce();
    nonbonded->setNonbondedMethod(PerturbNonbondedForce::PME);
    nonbonded->setSoftcoreAlpha(1.0);
    nonbonded->addGlobalParameter("lambdaA_charge", 1.0);
    nonbonded->addGlobalParameter("lambdaB_charge", 0.0);
    nonbonded->addGlobalParameter("lambdaC_charge", 0.0);
    nonbonded->addGlobalParameter("rest2_scale", 1.0);
    nonbonded->addGlobalParameter("lambdaA_vdw", 1.0);
    nonbonded->addGlobalParameter("lambdaB_vdw", 0.0);

    system.addForce(nonbonded);
    vector<Vec3> positions;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(i%2 == 0 ? 1 : -1, 1, 0, 0, 1.0, 0.0, false, false);
        positions.push_back(Vec3(genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5)*6);
    }
    VerletIntegrator integrator(0.001);
    map<string, string> properties;
    properties[CudaPlatform::CudaDeterministicForces()] = "true";
    Context context(system, integrator, platform, properties);
    context.setPositions(positions);
    State state1 = context.getState(State::Forces);
    State state2 = context.getState(State::Forces);
    
    // All forces should be *exactly* equal.
    
    for (int i = 0; i < numParticles; i++) {
        ASSERT_EQUAL(state1.getForces()[i][0], state2.getForces()[i][0]);
        ASSERT_EQUAL(state1.getForces()[i][1], state2.getForces()[i][1]);
        ASSERT_EQUAL(state1.getForces()[i][2], state2.getForces()[i][2]);
    }
}

bool canRunHugeTest() {
    // Create a minimal context just to see which device is being used.

    System system;
    system.addParticle(1.0);
    VerletIntegrator integrator(1.0);
    Context context(system, integrator, platform);
    int deviceIndex = stoi(platform.getPropertyValue(context, CudaPlatform::CudaDeviceIndex()));

    // Find out how much memory the device has.

    CUdevice device;
    cuDeviceGet(&device, deviceIndex);
    size_t memory;
    cuDeviceTotalMem(&memory, device);

    // Only run the huge test if the device has at least 4 GB of memory.

    return (memory >= 4L*(1<<30));
}

void testPerformance(){
    string system_file="../tests/data/normal_system.xml";
    std::ifstream inputFile(system_file);
    System* system = XmlSerializer::deserialize<System>(inputFile);
    
    string init_state_file="../tests/data/normal_state.xml";
    std::ifstream inputFile2(init_state_file);
    State* state = XmlSerializer::deserialize<State>(inputFile2);
    vector<Vec3> positions = state->getPositions();

    LangevinMiddleIntegrator integrator(300, 1, 0.001);
    Context normal_context(*system, integrator, platform);

    normal_context.setPositions(positions);

    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    //ProfilerStart("Normal.prof");
    //integrator.setIntegrationForceGroups(1<<1);
    integrator.step(1000);
    //ProfilerStop();
    
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Normal context step time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us" << endl;

    string perturb_system_file="../tests/data/perturb_system.xml";
    std::ifstream inputFile3(perturb_system_file);
    System* perturb_system = XmlSerializer::deserialize<System>(inputFile3);

    LangevinMiddleIntegrator perturb_integrator(300, 1, 0.001);
    Context perturb_context(*perturb_system, perturb_integrator, platform);
    perturb_context.setPositions(positions);

    start = chrono::steady_clock::now();
    //ProfilerStart("Perturb.prof");
    //perturb_integrator.setIntegrationForceGroups(1<<7);
    perturb_integrator.step(1000);
    //ProfilerStop();
    end = chrono::steady_clock::now();
    cout << "Perturb context step time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us" << endl;
}

void testSysB(){
    string system_file="../tests/data/sysB.xml";
    std::ifstream inputFile(system_file);
    System* system = XmlSerializer::deserialize<System>(inputFile);

    string normal_system_file="../tests/data/normal_sysB.xml";
    std::ifstream inputFile3(normal_system_file);
    System* normal_system = XmlSerializer::deserialize<System>(inputFile3);
    
    string init_state_file="../tests/data/sysB_state.xml";
    std::ifstream inputFile2(init_state_file);
    State* state = XmlSerializer::deserialize<State>(inputFile2);
    vector<Vec3> positions = state->getPositions();
    vector<Vec3> boxVectors;
    boxVectors.push_back(Vec3(6.7930, 0, 0));
    boxVectors.push_back(Vec3(0, 6.7760, 0));
    boxVectors.push_back(Vec3(0, 0, 8.2590));

    LangevinMiddleIntegrator integrator(300, 1, 0.001);
    Context sysB_context(*system, integrator, platform);

    LangevinMiddleIntegrator normal_integrator(300, 1, 0.001);
    Context normal_context(*normal_system, normal_integrator, platform);

    normal_context.setPositions(positions);
    normal_context.setPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    sysB_context.setPositions(positions);
    sysB_context.setPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);

    State sysB_state = sysB_context.getState(State::Forces | State::Energy);
    State normal_state = normal_context.getState(State::Forces | State::Energy);
    
    std::cout<<"SysB energy: "<<sysB_state.getPotentialEnergy()<<std::endl;
    std::cout<<"Normal energy: "<<normal_state.getPotentialEnergy()<<std::endl;
}

void testBlockPrunedParticleEnergyQueries(bool disablePmeStream) {
    const int numParticles = 4096;
    const int numQueries = 256;
    const bool runSlow = std::getenv("PERTURB_RUN_SLOW_TESTS") != nullptr;
    const int benchmarkIterations = runSlow ? 101 : 5;
    const double boxSize = 20.0;
    const double cutoff = 1.0;
    const double dielectric = 78.3;

    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    PerturbNonbondedForce* nonbonded = new PerturbNonbondedForce();
    nonbonded->setNonbondedMethod(PerturbNonbondedForce::PME);
    nonbonded->setCutoffDistance(cutoff);
    nonbonded->setReactionFieldDielectric(dielectric);
    nonbonded->setUseDispersionCorrection(false);
    nonbonded->setReciprocalSpaceForceGroup(1);
    nonbonded->addGlobalParameter("lambdaA_charge", 1.0);
    nonbonded->addGlobalParameter("lambdaB_charge", 0.0);
    nonbonded->addGlobalParameter("lambdaC_charge", 0.0);
    nonbonded->addGlobalParameter("rest2_scale", 1.0);
    nonbonded->addGlobalParameter("lambdaA_vdw", 1.0);
    nonbonded->addGlobalParameter("lambdaB_vdw", 0.0);
    std::vector<Vec3> systemPositions(numParticles);
    for (int index = 0; index < numParticles; index++) {
        system.addParticle(1.0);
        const double charge = index%2 == 0 ? 0.2 : -0.2;
        const double sigma = index == 37 ? 0.331546 : 0.3;
        const double epsilon = index == 37 ? 0.412979 : 0.0;
        const bool isDummy = index == 37;
        nonbonded->addParticle(charge, sigma, epsilon, isDummy ? 0.0 : charge,
                sigma, isDummy ? 0.0 : epsilon, isDummy, false);
        const int x = index%16;
        const int y = (index/16)%16;
        const int z = index/256;
        systemPositions[index] = Vec3((x+0.25)*boxSize/16, (y+0.25)*boxSize/16, (z+0.25)*boxSize/16);
    }
    system.addForce(nonbonded);
    VerletIntegrator integrator(0.001);
    std::map<std::string, std::string> properties;
    properties["DisablePmeStream"] = disablePmeStream ? "true" : "false";
    Context context(system, integrator, platform, properties);
    context.setPositions(systemPositions);
    context.getState(State::Energy);

    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(
            nonbonded->getCudaPlatformData(context));
    CudaContext& cu = *data.contexts[0];
    ContextSelector selector(cu);
    CudaArray positions, parameters, excludedStarts, excludedCounts, excludedAtoms, forces, energies;
    CudaArray symmetricAtomicEnergies, symmetricAtomicSum;
    positions.initialize<mm_float4>(cu, numQueries, "queryPositions");
    parameters.initialize<mm_float4>(cu, numQueries, "queryParameters");
    excludedStarts.initialize<int>(cu, numQueries, "excludedStarts");
    excludedCounts.initialize<int>(cu, numQueries, "excludedCounts");
    excludedAtoms.initialize<int>(cu, numQueries, "excludedAtoms");
    forces.initialize<mm_float4>(cu, numQueries, "queryForces");
    energies.initialize<float>(cu, numQueries, "queryEnergies");
    const int mixedElementSize = (cu.getUseMixedPrecision() || cu.getUseDoublePrecision()) ? sizeof(double) : sizeof(float);
    symmetricAtomicEnergies.initialize(cu, numParticles, mixedElementSize, "symmetricAtomicEnergies");
    symmetricAtomicSum.initialize(cu, 1, mixedElementSize, "symmetricAtomicSum");
    nonbonded->getSymmetricPMEReciprocalAtomicEnergiesCuda(context,
            symmetricAtomicEnergies.getDevicePointer(), symmetricAtomicSum.getDevicePointer());
    double symmetricSum = 0.0;
    if (mixedElementSize == sizeof(double)) {
        std::vector<double> sum;
        symmetricAtomicSum.download(sum);
        symmetricSum = sum[0];
    }
    else {
        std::vector<float> sum;
        symmetricAtomicSum.download(sum);
        symmetricSum = sum[0];
    }
    const double reciprocalGroupEnergy = context.getState(State::Energy, false, 1<<1).getPotentialEnergy();
    double analyticSelf = 0.0;
    double alphaForClosure;
    int closureX, closureY, closureZ;
    nonbonded->getPMEParametersInContext(context, alphaForClosure, closureX, closureY, closureZ);
    for (int atom = 0; atom < numParticles; atom++) {
        const double charge = atom%2 == 0 ? 0.2 : -0.2;
        analyticSelf -= ONE_4PI_EPS0*alphaForClosure/std::sqrt(M_PI)*charge*charge;
    }
    const double meshReciprocalEnergy = reciprocalGroupEnergy-analyticSelf;
    ASSERT_EQUAL_TOL(meshReciprocalEnergy, symmetricSum, cu.getUseDoublePrecision() ? 2e-6 : 2e-4);

    // Validate linear source->query PME scratch maps without touching the main grid.
    // For a partition full=D+B, psi_full-psi_D must equal psi_B.
    const int deletedCount = 17;
    const int backgroundCount = numParticles-deletedCount;
    const int scratchQueryCount = 41;
    CudaArray allPositions, allCharges, deletedPositions, deletedCharges, backgroundPositions,
            backgroundCharges, scratchQueryPositions, fullPotentials, deletedPotentials, backgroundPotentials;
    allPositions.initialize<mm_float4>(cu, numParticles, "allSourcePositions");
    allCharges.initialize<float>(cu, numParticles, "allSourceCharges");
    deletedPositions.initialize<mm_float4>(cu, deletedCount, "deletedSourcePositions");
    deletedCharges.initialize<float>(cu, deletedCount, "deletedSourceCharges");
    backgroundPositions.initialize<mm_float4>(cu, backgroundCount, "backgroundSourcePositions");
    backgroundCharges.initialize<float>(cu, backgroundCount, "backgroundSourceCharges");
    scratchQueryPositions.initialize<mm_float4>(cu, scratchQueryCount, "scratchQueryPositions");
    fullPotentials.initialize<float>(cu, scratchQueryCount, "fullScratchPotentials");
    deletedPotentials.initialize<float>(cu, scratchQueryCount, "deletedScratchPotentials");
    backgroundPotentials.initialize<float>(cu, scratchQueryCount, "backgroundScratchPotentials");
    std::vector<mm_float4> allSourcePositionValues(numParticles), deletedPositionValues(deletedCount),
            backgroundPositionValues(backgroundCount), scratchPositionValues(scratchQueryCount);
    std::vector<float> allChargeValues(numParticles), deletedChargeValues(deletedCount),
            backgroundChargeValues(backgroundCount);
    for (int atom = 0; atom < numParticles; atom++) {
        const Vec3& p = systemPositions[atom];
        allSourcePositionValues[atom] = mm_float4((float)p[0], (float)p[1], (float)p[2], 0.0f);
        allChargeValues[atom] = atom%2 == 0 ? 0.2f : -0.2f;
        if (atom < deletedCount) {
            deletedPositionValues[atom] = allSourcePositionValues[atom];
            deletedChargeValues[atom] = allChargeValues[atom];
        }
        else {
            backgroundPositionValues[atom-deletedCount] = allSourcePositionValues[atom];
            backgroundChargeValues[atom-deletedCount] = allChargeValues[atom];
        }
    }
    for (int query = 0; query < scratchQueryCount; query++)
        scratchPositionValues[query] = allSourcePositionValues[(query*97+11)%numParticles];
    allPositions.upload(allSourcePositionValues);
    allCharges.upload(allChargeValues);
    deletedPositions.upload(deletedPositionValues);
    deletedCharges.upload(deletedChargeValues);
    backgroundPositions.upload(backgroundPositionValues);
    backgroundCharges.upload(backgroundChargeValues);
    scratchQueryPositions.upload(scratchPositionValues);
    CudaArray scratchSourceBase, scratchSourceCorrection, scratchQueryBase, scratchQueryCorrection,
            contextFullPotentials;
    const CudaArrayDevicePointer scratchSourceCorrectionPointer = uploadContextPrecisionPositions(
            cu, allSourcePositionValues, scratchSourceBase, scratchSourceCorrection,
            "scratchContextSource");
    const CudaArrayDevicePointer scratchQueryCorrectionPointer = uploadContextPrecisionPositions(
            cu, scratchPositionValues, scratchQueryBase, scratchQueryCorrection,
            "scratchContextQuery");
    contextFullPotentials.initialize<float>(cu, scratchQueryCount, "contextFullScratchPotentials");
    nonbonded->getPMEPotentialFromParticlesCuda(context, numParticles,
            allPositions.getDevicePointer(), allCharges.getDevicePointer(), scratchQueryCount,
            scratchQueryPositions.getDevicePointer(), fullPotentials.getDevicePointer());
    nonbonded->getPMEPotentialFromParticlesCuda(context, deletedCount,
            deletedPositions.getDevicePointer(), deletedCharges.getDevicePointer(), scratchQueryCount,
            scratchQueryPositions.getDevicePointer(), deletedPotentials.getDevicePointer());
    nonbonded->getPMEPotentialFromParticlesCuda(context, backgroundCount,
            backgroundPositions.getDevicePointer(), backgroundCharges.getDevicePointer(), scratchQueryCount,
            scratchQueryPositions.getDevicePointer(), backgroundPotentials.getDevicePointer());
    nonbonded->getPMEPotentialFromParticlesCudaContextPrecision(context, numParticles,
            scratchSourceBase.getDevicePointer(), scratchSourceCorrectionPointer,
            allCharges.getDevicePointer(), scratchQueryCount, scratchQueryBase.getDevicePointer(),
            scratchQueryCorrectionPointer, contextFullPotentials.getDevicePointer());
    std::vector<float> fullPotentialValues, deletedPotentialValues, backgroundPotentialValues,
            contextFullPotentialValues;
    fullPotentials.download(fullPotentialValues);
    deletedPotentials.download(deletedPotentialValues);
    backgroundPotentials.download(backgroundPotentialValues);
    contextFullPotentials.download(contextFullPotentialValues);
    const double linearTolerance = cu.getUseDoublePrecision() ? 2e-5 : 2e-3;
    for (int query = 0; query < scratchQueryCount; query++) {
        ASSERT_EQUAL_TOL(backgroundPotentialValues[query],
                fullPotentialValues[query]-deletedPotentialValues[query], linearTolerance);
        ASSERT_EQUAL_TOL(fullPotentialValues[query], contextFullPotentialValues[query], linearTolerance);
    }
    const double reciprocalEnergyAfterScratch = context.getState(State::Energy, false, 1<<1).getPotentialEnergy();
    ASSERT_EQUAL_TOL(reciprocalGroupEnergy, reciprocalEnergyAfterScratch, 1e-6);

    // Each candidate self term must equal a separate one-candidate scratch call.
    const int selfCandidateCount = 3;
    const int selfAtomsPerCandidate = 4;
    const int selfAtomCount = selfCandidateCount*selfAtomsPerCandidate;
    CudaArray selfPositions, selfCharges, selfEnergies, selfReferencePotentials;
    selfPositions.initialize<mm_float4>(cu, selfAtomCount, "candidateSelfPositions");
    selfCharges.initialize<float>(cu, selfAtomCount, "candidateSelfCharges");
    selfEnergies.initialize<float>(cu, selfCandidateCount, "candidateSelfEnergies");
    selfReferencePotentials.initialize<float>(cu, selfAtomsPerCandidate, "candidateSelfReferencePotentials");
    std::vector<mm_float4> selfPositionValues(selfAtomCount);
    std::vector<float> selfChargeValues(selfAtomCount);
    for (int index = 0; index < selfAtomCount; index++) {
        const int candidate = index/selfAtomsPerCandidate;
        const int site = index%selfAtomsPerCandidate;
        selfPositionValues[index] = mm_float4(1.0f+0.7f*candidate+0.04f*site,
                2.0f+0.3f*candidate-0.03f*site, 3.0f+0.2f*candidate+0.02f*site, 0.0f);
        selfChargeValues[index] = site%2 == 0 ? 0.21f : -0.21f;
    }
    selfPositions.upload(selfPositionValues);
    selfCharges.upload(selfChargeValues);
    CudaArray selfContextBase, selfContextCorrection, selfContextEnergies;
    const CudaArrayDevicePointer selfContextCorrectionPointer = uploadContextPrecisionPositions(
            cu, selfPositionValues, selfContextBase, selfContextCorrection, "candidateSelfContextPosition");
    selfContextEnergies.initialize<float>(cu, selfCandidateCount, "candidateSelfContextEnergies");
    nonbonded->getPMECandidateSelfEnergiesCuda(context, selfCandidateCount, selfAtomsPerCandidate,
            selfPositions.getDevicePointer(), selfCharges.getDevicePointer(), selfEnergies.getDevicePointer());
    std::vector<float> selfEnergyValues;
    selfEnergies.download(selfEnergyValues);
    nonbonded->getPMECandidateSelfEnergiesCudaContextPrecision(context,
            selfCandidateCount, selfAtomsPerCandidate, selfContextBase.getDevicePointer(),
            selfContextCorrectionPointer, selfCharges.getDevicePointer(),
            selfContextEnergies.getDevicePointer());
    std::vector<float> selfContextEnergyValues;
    selfContextEnergies.download(selfContextEnergyValues);
    for (int candidate = 0; candidate < selfCandidateCount; candidate++) {
        ASSERT_EQUAL_TOL(selfEnergyValues[candidate], selfContextEnergyValues[candidate], linearTolerance);
        const CUdeviceptr candidatePositionPointer = selfPositions.getDevicePointer()+
                (CUdeviceptr)candidate*selfAtomsPerCandidate*sizeof(mm_float4);
        const CUdeviceptr candidateChargePointer = selfCharges.getDevicePointer()+
                (CUdeviceptr)candidate*selfAtomsPerCandidate*sizeof(float);
        nonbonded->getPMEPotentialFromParticlesCuda(context, selfAtomsPerCandidate,
                (CudaArrayDevicePointer)candidatePositionPointer, (CudaArrayDevicePointer)candidateChargePointer,
                selfAtomsPerCandidate, (CudaArrayDevicePointer)candidatePositionPointer,
                selfReferencePotentials.getDevicePointer());
        std::vector<float> referencePotentials;
        selfReferencePotentials.download(referencePotentials);
        double referenceSelf = 0.0;
        for (int site = 0; site < selfAtomsPerCandidate; site++)
            referenceSelf += 0.5*std::sqrt(ONE_4PI_EPS0)*selfChargeValues[candidate*selfAtomsPerCandidate+site]*referencePotentials[site];
        ASSERT_EQUAL_TOL(referenceSelf, selfEnergyValues[candidate], 2e-5);
    }

    std::vector<mm_float4> queryPositions(numQueries), queryParameters(numQueries);
    std::vector<int> starts(numQueries), counts(numQueries, 1), exclusions(numQueries);
    for (int query = 0; query < numQueries; query++) {
        const int atom = (query*37)%numParticles;
        const Vec3& p = systemPositions[atom];
        queryPositions[query] = mm_float4((float) p[0], (float) p[1], (float) p[2], 0.0f);
        queryParameters[query] = mm_float4(0.35f, 0.15f, 0.0f, 0.0f);
        starts[query] = query;
        exclusions[query] = atom;
    }
    const Vec3 closePosition = systemPositions[37]-Vec3(0.095, 0.0, 0.0);
    queryPositions[numQueries-1] = mm_float4((float)closePosition[0], (float)closePosition[1],
            (float)closePosition[2], 0.0f);
    queryParameters[numQueries-1] = mm_float4(-0.834f, 0.5f*0.315061f,
            2.0f*std::sqrt(0.636386f), 0.0f);
    std::vector<mm_double4> doubleQueryPositions(numQueries);
    for (int query = 0; query < numQueries; query++) {
        const double tail = (1+(query%7))*1e-10;
        doubleQueryPositions[query] = mm_double4((double) queryPositions[query].x+tail,
                (double) queryPositions[query].y-0.5*tail,
                (double) queryPositions[query].z+0.25*tail, 0.0);
    }
    if (cu.getUseDoublePrecision())
        ASSERT((double) (float) doubleQueryPositions[0].x != doubleQueryPositions[0].x);
    positions.upload(queryPositions);
    parameters.upload(queryParameters);
    // ContextPrecision queries are an opt-in corrected-coordinate view.  The
    // unsuffixed calls below continue to consume their legacy float4 inputs.
    CudaArray contextPositionBase, contextPositionCorrection;
    const CudaArrayDevicePointer contextPositionCorrectionPointer = uploadContextPrecisionPositions(
            cu, queryPositions, contextPositionBase, contextPositionCorrection, "queryContextPosition",
            cu.getUseDoublePrecision() ? &doubleQueryPositions : NULL);
    excludedStarts.upload(starts);
    excludedCounts.upload(counts);
    excludedAtoms.upload(exclusions);

    nonbonded->getInteractionEnergyForRealParticlesCuda(context, numQueries,
            positions.getDevicePointer(), parameters.getDevicePointer(), excludedStarts.getDevicePointer(),
            excludedCounts.getDevicePointer(), excludedAtoms.getDevicePointer(),
            forces.getDevicePointer(), energies.getDevicePointer());
    std::vector<float> result;
    energies.download(result);

    const double k = std::pow(cutoff, -3.0)*(dielectric-1.0)/(2.0*dielectric+1.0);
    const double c = (1.0/cutoff)*(3.0*dielectric)/(2.0*dielectric+1.0);
    for (int query = 0; query < numQueries; query++) {
        Vec3 queryPosition(queryPositions[query].x, queryPositions[query].y, queryPositions[query].z);
        double expected = 0.0;
        for (int atom = 0; atom < numParticles; atom++) {
            if (atom == exclusions[query])
                continue;
            Vec3 delta = systemPositions[atom]-queryPosition;
            for (int axis = 0; axis < 3; axis++)
                delta[axis] -= std::floor(delta[axis]/boxSize+0.5)*boxSize;
            const double r2 = delta.dot(delta);
            if (r2 > 1e-12 && r2 < cutoff*cutoff) {
                const double invR = 1.0/std::sqrt(r2);
                const double charge = atom%2 == 0 ? 0.2 : -0.2;
                expected += ONE_4PI_EPS0*0.35*charge*(invR+k*r2-c);
            }
        }
        if (query != numQueries-1)
            ASSERT_EQUAL_TOL(expected, result[query], 2e-4);
    }

    // The same block-pruned kernel is separately compiled for candidate PME
    // direct space.  Compare it to the analytic erfc(alpha*r)/r sum.
    nonbonded->getDirectEnergyForCandidateParticlesCuda(context, numQueries,
            positions.getDevicePointer(), parameters.getDevicePointer(),
            forces.getDevicePointer(), energies.getDevicePointer());
    std::vector<float> candidateDirect;
    energies.download(candidateDirect);
    nonbonded->getDirectEnergyForCandidateParticlesCudaContextPrecision(context, numQueries,
            contextPositionBase.getDevicePointer(), contextPositionCorrectionPointer,
            parameters.getDevicePointer(), forces.getDevicePointer(), energies.getDevicePointer());
    std::vector<float> contextCandidateDirect;
    energies.download(contextCandidateDirect);
    for (int query = 0; query < numQueries; query++)
        ASSERT_EQUAL_TOL(candidateDirect[query], contextCandidateDirect[query],
                std::max(5e-4, 5e-6*std::fabs(candidateDirect[query])));

    CudaArray legacyPmePotentials, contextPmePotentials;
    legacyPmePotentials.initialize<float>(cu, numQueries, "legacyPmePotentials");
    contextPmePotentials.initialize<float>(cu, numQueries, "contextPmePotentials");
    nonbonded->getPMEPotentialAtPositionsCuda(context, numQueries,
            positions.getDevicePointer(), legacyPmePotentials.getDevicePointer());
    nonbonded->getPMEPotentialAtPositionsCudaContextPrecision(context, numQueries,
            contextPositionBase.getDevicePointer(), contextPositionCorrectionPointer,
            contextPmePotentials.getDevicePointer());
    std::vector<float> legacyPmePotentialValues, contextPmePotentialValues;
    legacyPmePotentials.download(legacyPmePotentialValues);
    contextPmePotentials.download(contextPmePotentialValues);
    for (int query = 0; query < numQueries; query++)
        ASSERT_EQUAL_TOL(legacyPmePotentialValues[query], contextPmePotentialValues[query], linearTolerance);

    const int subsetCount = 7;
    CudaArray subsetIndices, legacySubsetEnergies, contextSubsetEnergies;
    subsetIndices.initialize<int>(cu, subsetCount, "contextPrecisionSubsetIndices");
    legacySubsetEnergies.initialize<float>(cu, numQueries, "legacySubsetEnergies");
    contextSubsetEnergies.initialize<float>(cu, numQueries, "contextSubsetEnergies");
    std::vector<int> subsetValues = {3, 17, 41, 127, 511, 1023, 2047};
    subsetIndices.upload(subsetValues);
    nonbonded->getDirectEnergyForCandidateSubsetCuda(context, numQueries,
            positions.getDevicePointer(), parameters.getDevicePointer(), subsetCount,
            subsetIndices.getDevicePointer(), legacySubsetEnergies.getDevicePointer());
    nonbonded->getDirectEnergyForCandidateSubsetCudaContextPrecision(context, numQueries,
            contextPositionBase.getDevicePointer(), contextPositionCorrectionPointer,
            parameters.getDevicePointer(), subsetCount, subsetIndices.getDevicePointer(),
            contextSubsetEnergies.getDevicePointer());
    std::vector<float> legacySubsetValues, contextSubsetValues;
    legacySubsetEnergies.download(legacySubsetValues);
    contextSubsetEnergies.download(contextSubsetValues);
    for (int query = 0; query < numQueries; query++)
        ASSERT_EQUAL_TOL(legacySubsetValues[query], contextSubsetValues[query],
                std::max(3e-4, 5e-6*std::fabs(legacySubsetValues[query])));

    // Enforce the public pointer contract on every ContextPrecision query.  Mixed
    // precision requires a correction array, while single and double require null.
    const CudaArrayDevicePointer invalidCorrectionPointer = cu.getUseMixedPrecision() ?
            0 : positions.getDevicePointer();
    assertInvalidContextPrecisionCorrectionThrows([&]() {
        nonbonded->getDirectEnergyForCandidateParticlesCudaContextPrecision(context, numQueries,
                contextPositionBase.getDevicePointer(), invalidCorrectionPointer,
                parameters.getDevicePointer(), forces.getDevicePointer(), energies.getDevicePointer());
    });
    assertInvalidContextPrecisionCorrectionThrows([&]() {
        nonbonded->getPMEPotentialAtPositionsCudaContextPrecision(context, numQueries,
                contextPositionBase.getDevicePointer(), invalidCorrectionPointer,
                contextPmePotentials.getDevicePointer());
    });
    assertInvalidContextPrecisionCorrectionThrows([&]() {
        nonbonded->getPMEPotentialFromParticlesCudaContextPrecision(context, numParticles,
                scratchSourceBase.getDevicePointer(), invalidCorrectionPointer,
                allCharges.getDevicePointer(), scratchQueryCount, scratchQueryBase.getDevicePointer(),
                scratchQueryCorrectionPointer, contextFullPotentials.getDevicePointer());
    });
    assertInvalidContextPrecisionCorrectionThrows([&]() {
        nonbonded->getPMEPotentialFromParticlesCudaContextPrecision(context, numParticles,
                scratchSourceBase.getDevicePointer(), scratchSourceCorrectionPointer,
                allCharges.getDevicePointer(), scratchQueryCount, scratchQueryBase.getDevicePointer(),
                invalidCorrectionPointer, contextFullPotentials.getDevicePointer());
    });
    assertInvalidContextPrecisionCorrectionThrows([&]() {
        nonbonded->getPMECandidateSelfEnergiesCudaContextPrecision(context,
                selfCandidateCount, selfAtomsPerCandidate, selfContextBase.getDevicePointer(),
                invalidCorrectionPointer, selfCharges.getDevicePointer(),
                selfContextEnergies.getDevicePointer());
    });
    assertInvalidContextPrecisionCorrectionThrows([&]() {
        nonbonded->getDirectEnergyForCandidateSubsetCudaContextPrecision(context, numQueries,
                contextPositionBase.getDevicePointer(), invalidCorrectionPointer,
                parameters.getDevicePointer(), subsetCount, subsetIndices.getDevicePointer(),
                contextSubsetEnergies.getDevicePointer());
    });

    double alpha;
    int gridX, gridY, gridZ;
    nonbonded->getPMEParametersInContext(context, alpha, gridX, gridY, gridZ);
    for (int query = 0; query < numQueries; query++) {
        Vec3 queryPosition(queryPositions[query].x, queryPositions[query].y, queryPositions[query].z);
        const double queryCharge = queryParameters[query].x;
        const double queryHalfSigma = queryParameters[query].y;
        const double queryTwiceSqrtEpsilon = queryParameters[query].z;
        double expected = 0.0;
        for (int atom = 0; atom < numParticles; atom++) {
            Vec3 delta = systemPositions[atom]-queryPosition;
            for (int axis = 0; axis < 3; axis++)
                delta[axis] -= std::floor(delta[axis]/boxSize+0.5)*boxSize;
            const double r2 = delta.dot(delta);
            if (r2 > 1e-12 && r2 < cutoff*cutoff) {
                const double r = std::sqrt(r2);
                double chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB;
                bool isDummy, isRest2;
                nonbonded->getParticleParameters(atom, chargeA, sigmaA, epsilonA,
                        chargeB, sigmaB, epsilonB, isDummy, isRest2);
                expected += ONE_4PI_EPS0*queryCharge*chargeA*std::erfc(alpha*r)/r;
                if (queryTwiceSqrtEpsilon != 0.0 && epsilonA != 0.0) {
                    const double sigma = queryHalfSigma+0.5*sigmaA;
                    const double epsilon = queryTwiceSqrtEpsilon*2.0*std::sqrt(epsilonA);
                    const double v6 = std::pow(sigma/r, 6);
                    expected += epsilon*(v6*v6-v6);
                }
            }
        }
        const double tolerance = std::max(3e-4, 3e-6*std::fabs(expected));
        ASSERT_EQUAL_TOL(expected, candidateDirect[query], tolerance);
        if (query == numQueries-1)
            ASSERT(expected > 1e6);
    }

    // Exclusion arrays contain valid data, so this also verifies that output
    // buffers are overwritten rather than accumulated across launches.
    std::vector<float> sentinel(numQueries, 1234.0f);
    energies.upload(sentinel);
    nonbonded->getInteractionEnergyForRealParticlesCuda(context, numQueries,
            positions.getDevicePointer(), parameters.getDevicePointer(), excludedStarts.getDevicePointer(),
            excludedCounts.getDevicePointer(), excludedAtoms.getDevicePointer(),
            forces.getDevicePointer(), energies.getDevicePointer());
    std::vector<float> repeated;
    energies.download(repeated);
    for (int query = 0; query < numQueries; query++)
        ASSERT_EQUAL_TOL(result[query], repeated[query], 1e-6);

    // Warm up before timing and force completion with a blocking download on
    // every sample.  This excludes Context construction and JIT compilation.
    auto measureMedian = [&](bool useBlockPruning) {
        nonbonded->setCudaParticleQueryBlockPruningForTesting(context, useBlockPruning);
        for (int iteration = 0; iteration < 10; iteration++) {
            nonbonded->getInteractionEnergyForRealParticlesCuda(context, numQueries,
                    positions.getDevicePointer(), parameters.getDevicePointer(), excludedStarts.getDevicePointer(),
                    excludedCounts.getDevicePointer(), excludedAtoms.getDevicePointer(),
                    forces.getDevicePointer(), energies.getDevicePointer());
            energies.download(repeated);
        }
        std::vector<double> timings;
        timings.reserve(benchmarkIterations);
        for (int iteration = 0; iteration < benchmarkIterations; iteration++) {
            const auto start = std::chrono::steady_clock::now();
            nonbonded->getInteractionEnergyForRealParticlesCuda(context, numQueries,
                    positions.getDevicePointer(), parameters.getDevicePointer(), excludedStarts.getDevicePointer(),
                    excludedCounts.getDevicePointer(), excludedAtoms.getDevicePointer(),
                    forces.getDevicePointer(), energies.getDevicePointer());
            energies.download(repeated);
            const auto stop = std::chrono::steady_clock::now();
            timings.push_back(std::chrono::duration<double, std::micro>(stop-start).count());
        }
        std::sort(timings.begin(), timings.end());
        return timings[timings.size()/2];
    };
    const double prunedMedianUs = measureMedian(true);
    const double exhaustiveMedianUs = measureMedian(false);
    nonbonded->setCudaParticleQueryBlockPruningForTesting(context, true);
    const double fullScanPairs = (double) numParticles*numQueries;
    const double cutoffVolumeFraction = (4.0*M_PI*cutoff*cutoff*cutoff/3.0)/(boxSize*boxSize*boxSize);
    std::cout << "RF query benchmark: system=" << numParticles
              << " queries=" << numQueries << " pruned_median_us=" << prunedMedianUs
              << " exhaustive_block_median_us=" << exhaustiveMedianUs
              << " speedup=" << exhaustiveMedianUs/prunedMedianUs
              << " full_scan_pairs=" << fullScanPairs
              << " expected_cutoff_pairs=" << fullScanPairs*cutoffVolumeFraction << std::endl;
}

void testRealPmeDirectSoftcoreTypeSymmetry() {
    const int numWaterAtoms = 3;
    const double boxSize = 3.0;
    const double cutoff = 1.0;
    const double softcoreAlpha = 0.5;
    const double lambdaA = 0.1;
    const double lambdaB = 0.2;

    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    PerturbNonbondedForce* nonbonded = new PerturbNonbondedForce();
    nonbonded->setNonbondedMethod(PerturbNonbondedForce::PME);
    nonbonded->setCutoffDistance(cutoff);
    nonbonded->setEwaldErrorTolerance(1e-5);
    nonbonded->setUseDispersionCorrection(false);
    nonbonded->setSoftcoreAlpha(softcoreAlpha);
    nonbonded->addGlobalParameter("lambdaA_charge", 1.0);
    nonbonded->addGlobalParameter("lambdaB_charge", 0.0);
    nonbonded->addGlobalParameter("lambdaC_charge", 0.0);
    nonbonded->addGlobalParameter("rest2_scale", 1.0);
    nonbonded->addGlobalParameter("lambdaA_vdw", lambdaA);
    nonbonded->addGlobalParameter("lambdaB_vdw", lambdaB);

    const double waterCharges[numWaterAtoms] = {-0.834, 0.417, 0.417};
    const double waterSigmas[numWaterAtoms] = {0.315061, 1.0, 1.0};
    const double waterEpsilons[numWaterAtoms] = {0.636386, 0.0, 0.0};
    const double dummyChargeA = 0.5;
    const double dummySigmaA = 0.331546;
    const double dummyEpsilonA = 0.412979;
    const double dummySigmaB = 0.3;
    const double dummyEpsilonB = 0.2;
    for (int atom = 0; atom < numWaterAtoms; atom++) {
        system.addParticle(atom == 0 ? 15.999 : 1.008);
        nonbonded->addParticle(waterCharges[atom], waterSigmas[atom], waterEpsilons[atom],
                waterCharges[atom], waterSigmas[atom], waterEpsilons[atom], false, false);
    }
    system.addParticle(12.0);
    nonbonded->addParticle(dummyChargeA, dummySigmaA, dummyEpsilonA,
            0.0, dummySigmaB, dummyEpsilonB, true, false);
    for (int atom1 = 0; atom1 < numWaterAtoms; atom1++)
        for (int atom2 = atom1+1; atom2 < numWaterAtoms; atom2++)
            nonbonded->addException(atom1, atom2, 0.0, 0.1, 0.0);
    system.addForce(nonbonded);

    const double ohDistance = 0.09572;
    const double hohAngle = 104.52*M_PI/180.0;
    std::vector<Vec3> positions;
    positions.push_back(Vec3(1.0, 1.0, 1.0));
    positions.push_back(Vec3(1.0+ohDistance, 1.0, 1.0));
    positions.push_back(Vec3(1.0+ohDistance*std::cos(hohAngle),
            1.0+ohDistance*std::sin(hohAngle), 1.0));
    positions.push_back(Vec3(1.0, 1.0, 1.085));

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.getState(State::Energy);

    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(
            nonbonded->getCudaPlatformData(context));
    CudaContext& cu = *data.contexts[0];
    ContextSelector selector(cu);
    CudaArray realAtoms, energies;
    realAtoms.initialize<int>(cu, numWaterAtoms, "softcoreRealAtoms");
    energies.initialize<float>(cu, numWaterAtoms, "softcoreRealEnergies");
    std::vector<int> realAtomIndices = {0, 1, 2};
    realAtoms.upload(realAtomIndices);
    // This legacy API deliberately reads the base posq representation, matching
    // the coordinates used by the main Context nonbonded energy evaluation.
    nonbonded->getDirectEnergyForRealParticlesPmeCuda(context, numWaterAtoms,
            realAtoms.getDevicePointer(), numWaterAtoms, energies.getDevicePointer());
    std::vector<float> actual;
    energies.download(actual);

    double alpha;
    int gridX, gridY, gridZ;
    nonbonded->getPMEParametersInContext(context, alpha, gridX, gridY, gridZ);
    double cpuOracleTotal = 0.0;
    double unsoftenedTotal = 0.0;
    for (int atom = 0; atom < numWaterAtoms; atom++) {
        Vec3 delta = positions.back()-positions[atom];
        const double r2 = delta.dot(delta);
        const double r = std::sqrt(r2);
        const double coulomb = ONE_4PI_EPS0*waterCharges[atom]*dummyChargeA*std::erfc(alpha*r)/r;
        const double sigmaA = 0.5*(waterSigmas[atom]+dummySigmaA);
        const double sigmaB = 0.5*(waterSigmas[atom]+dummySigmaB);
        const double epsilonA = 4.0*std::sqrt(waterEpsilons[atom]*dummyEpsilonA);
        const double epsilonB = 4.0*std::sqrt(waterEpsilons[atom]*dummyEpsilonB);
        const double r6 = r2*r2*r2;
        const double sigmaA6 = std::pow(sigmaA, 6);
        const double sigmaB6 = std::pow(sigmaB, 6);
        const double softcoreA6 = sigmaA6/(softcoreAlpha*(1.0-lambdaA)*(1.0-lambdaA)*sigmaA6+r6);
        const double softcoreB6 = sigmaB6/(softcoreAlpha*(1.0-lambdaB)*(1.0-lambdaB)*sigmaB6+r6);
        const double expected = coulomb+
                lambdaA*epsilonA*(softcoreA6*softcoreA6-softcoreA6)+
                lambdaB*epsilonB*(softcoreB6*softcoreB6-softcoreB6);
        const double unsoftenedA6 = sigmaA6/r6;
        // This is the pre-fix ordinary-query branch: it ignored the dummy
        // source type and used the unscaled A-state Lennard-Jones energy.
        const double unsoftened = coulomb+epsilonA*(unsoftenedA6*unsoftenedA6-unsoftenedA6);
        const double tolerance = std::max(2e-3, 3e-5*std::fabs(expected));
        ASSERT_EQUAL_TOL(expected, actual[atom], tolerance);
        cpuOracleTotal += expected;
        unsoftenedTotal += unsoftened;
    }
    double actualTotal = 0.0;
    for (float energy : actual)
        actualTotal += energy;
    ASSERT_EQUAL_TOL(cpuOracleTotal, actualTotal, std::max(5e-3, 3e-5*std::fabs(cpuOracleTotal)));
    ASSERT(cpuOracleTotal < 0.0);
    ASSERT(unsoftenedTotal > 1e6);
    ASSERT(std::fabs(cpuOracleTotal) < 1e3);
    std::cout << "Real PME direct softcore symmetry: gpu_total=" << actualTotal
              << " cpu_oracle=" << cpuOracleTotal
              << " unsoftened_control=" << unsoftenedTotal << std::endl;
}

void runPlatformTests() {
    //testSysB();
    //testPerformance();
    testLargeSystem();
    testReordering();
    testDeterministicForces();
    if (canRunHugeTest())
        testHugeSystem();
    testBlockPrunedParticleEnergyQueries(false);
    testBlockPrunedParticleEnergyQueries(true);
    testRealPmeDirectSoftcoreTypeSymmetry();

}
