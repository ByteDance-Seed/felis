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

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/HarmonicBondForce.h"
#include "PerturbNonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "sfmt/SFMT.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace OpenMM;
using namespace std;
using namespace PerturbNonbondedPlugin;

const double TOL = 1e-5;

void testCoulomb() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    PerturbNonbondedForce* forceField = new PerturbNonbondedForce();
    forceField->addParticle(0.5, 1, 0, 0.5, 1, 0, false, false);
    forceField->addParticle(-1.5, 1, 0, -1.5, 1, 0, false, false);

    forceField->setSoftcoreAlpha(1.0);
    forceField->addGlobalParameter("lambdaA_charge", 1.0);
    forceField->addGlobalParameter("lambdaB_charge", 0.0);
    forceField->addGlobalParameter("lambdaC_charge", 0.0);
    forceField->addGlobalParameter("rest2_scale", 1.0);
    forceField->addGlobalParameter("lambdaA_vdw", 1.0);
    forceField->addGlobalParameter("lambdaB_vdw", 0.0);

    system.addForce(forceField);
    ASSERT(!forceField->usesPeriodicBoundaryConditions());
    ASSERT(!system.usesPeriodicBoundaryConditions());
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(2, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    double force = ONE_4PI_EPS0*(-0.75)/4.0;
    ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(-0.75)/2.0, state.getPotentialEnergy(), TOL);
}

void testLJ() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    PerturbNonbondedForce* forceField = new PerturbNonbondedForce();
    forceField->addParticle(0, 1.2, 1, 0, 1.2, 1, false, false);
    forceField->addParticle(0, 1.4, 2, 0, 1.4, 2, false, false);
    forceField->setSoftcoreAlpha(1.0);
    forceField->addGlobalParameter("lambdaA_charge", 1.0);
    forceField->addGlobalParameter("lambdaB_charge", 0.0);
    forceField->addGlobalParameter("lambdaC_charge", 0.0);
    forceField->addGlobalParameter("rest2_scale", 1.0);
    forceField->addGlobalParameter("lambdaA_vdw", 1.0);
    forceField->addGlobalParameter("lambdaB_vdw", 0.0);

    system.addForce(forceField);
    ASSERT(!forceField->usesPeriodicBoundaryConditions());
    ASSERT(!system.usesPeriodicBoundaryConditions());
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(2, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    double x = 1.3/2.0;
    double eps = SQRT_TWO;
    double force = 4.0*eps*(12*std::pow(x, 12.0)-6*std::pow(x, 6.0))/2.0;
    ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_TOL(4.0*eps*(std::pow(x, 12.0)-std::pow(x, 6.0)), state.getPotentialEnergy(), TOL);
}

void testExclusionsAnd14() {
    System system;
    PerturbNonbondedForce* nonbonded = new PerturbNonbondedForce();
    for (int i = 0; i < 5; ++i) {
        system.addParticle(1.0);
        nonbonded->addParticle(0, 1.5, 0, 0, 1.5, 0, false, false);
    }

    nonbonded->setSoftcoreAlpha(1.0);
    nonbonded->addGlobalParameter("lambdaA_charge", 1.0);
    nonbonded->addGlobalParameter("lambdaB_charge", 0.0);
    nonbonded->addGlobalParameter("lambdaC_charge", 0.0);
    nonbonded->addGlobalParameter("rest2_scale", 1.0);
    nonbonded->addGlobalParameter("lambdaA_vdw", 1.0);
    nonbonded->addGlobalParameter("lambdaB_vdw", 0.0);

    vector<pair<int, int> > bonds = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    nonbonded->createExceptionsFromBonds(bonds, 0.0, 0.0);
    int first14, second14;
    for (int i = 0; i < nonbonded->getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        nonbonded->getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if ((particle1 == 0 && particle2 == 3) || (particle1 == 3 && particle2 == 0))
            first14 = i;
        if ((particle1 == 1 && particle2 == 4) || (particle1 == 4 && particle2 == 1))
            second14 = i;
    }
    system.addForce(nonbonded);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    for (int i = 1; i < 5; ++i) {

        // Test LJ forces

        vector<Vec3> positions(5);
        const double r = 1.0;
        for (int j = 0; j < 5; ++j) {
            nonbonded->setParticleParameters(j, 0, 1.5, 0, 0, 1.5, 0, false, false);
            positions[j] = Vec3(0, j, 0);
        }
        nonbonded->setParticleParameters(0, 0, 1.5, 1, 0, 1.5, 1, false, false);
        nonbonded->setParticleParameters(i, 0, 1.5, 1, 0, 1.5, 1, false, false);
        nonbonded->setExceptionParameters(first14, 0, 3, 0, 1.5, 0.0);
        nonbonded->setExceptionParameters(second14, 1, 4, 0, 1.5, 0.0);
        positions[i] = Vec3(r, 0, 0);
        context.reinitialize();
        context.setPositions(positions);
        State state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces = state.getForces();
        double x = 1.5/r; // sigma = 1.5, epsilon = 1.0
        double eps = 1.0;
        double force = 4.0*eps*(12*std::pow(x, 12.0)-6*std::pow(x, 6.0))/r;
        double energy = 4.0*eps*(std::pow(x, 12.0)-std::pow(x, 6.0));
        if (i == 3) {
            force *= 0.0;
            energy *= 0.0;
        }
        if (i < 3) {
            force = 0;
            energy = 0;
        }
        ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[0], TOL);
        ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[i], TOL);
        ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), TOL);

        // Test Coulomb forces

        nonbonded->setParticleParameters(0, 2, 1.5, 0, 2, 1.5, 0, false, false);
        nonbonded->setParticleParameters(i, 2, 1.5, 0, 2, 1.5, 0, false, false);
        nonbonded->setExceptionParameters(first14, 0, 3, 0, 1.5, 0);
        nonbonded->setExceptionParameters(second14, 1, 4, 0, 1.5, 0);
        context.reinitialize();
        context.setPositions(positions);
        state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces2 = state.getForces();
        force = ONE_4PI_EPS0*4/(r*r);
        energy = ONE_4PI_EPS0*4/r;
        if (i == 3) {
            force *= 0.0;
            energy *= 0.0;
        }
        if (i < 3) {
            force = 0;
            energy = 0;
        }
        ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces2[0], TOL);
        ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces2[i], TOL);
        ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), TOL);
    }
}

void testCutoff() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    PerturbNonbondedForce* forceField = new PerturbNonbondedForce();
    forceField->addParticle(1.0, 1, 0, 1.0, 1, 0, false, false);
    forceField->addParticle(1.0, 1, 0, 1.0, 1, 0, false, false);
    forceField->addParticle(1.0, 1, 0, 1.0, 1, 0, false, false);
    forceField->setNonbondedMethod(PerturbNonbondedForce::CutoffNonPeriodic);

    forceField->setSoftcoreAlpha(1.0);
    forceField->addGlobalParameter("lambdaA_charge", 1.0);
    forceField->addGlobalParameter("lambdaB_charge", 0.0);
    forceField->addGlobalParameter("lambdaC_charge", 0.0);
    forceField->addGlobalParameter("rest2_scale", 1.0);
    forceField->addGlobalParameter("lambdaA_vdw", 1.0);
    forceField->addGlobalParameter("lambdaB_vdw", 0.0);


    const double cutoff = 2.9;
    forceField->setCutoffDistance(cutoff);
    const double eps = 50.0;
    forceField->setReactionFieldDielectric(eps);
    system.addForce(forceField);
    ASSERT(!forceField->usesPeriodicBoundaryConditions());
    ASSERT(!system.usesPeriodicBoundaryConditions());
    Context context(system, integrator, platform);
    vector<Vec3> positions(3);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 2, 0);
    positions[2] = Vec3(0, 3, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    const double krf = (1.0/(cutoff*cutoff*cutoff))*(eps-1.0)/(2.0*eps+1.0);
    const double crf = (1.0/cutoff)*(3.0*eps)/(2.0*eps+1.0);
    const double force1 = ONE_4PI_EPS0*(1.0)*(0.25-2.0*krf*2.0);
    const double force2 = ONE_4PI_EPS0*(1.0)*(1.0-2.0*krf*1.0);
    ASSERT_EQUAL_VEC(Vec3(0, -force1, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(0, force1-force2, 0), forces[1], TOL);
    ASSERT_EQUAL_VEC(Vec3(0, force2, 0), forces[2], TOL);
    const double energy1 = ONE_4PI_EPS0*(1.0)*(0.5+krf*4.0-crf);
    const double energy2 = ONE_4PI_EPS0*(1.0)*(1.0+krf*1.0-crf);
    ASSERT_EQUAL_TOL(energy1+energy2, state.getPotentialEnergy(), TOL);
}

void testCutoff14() {
    System system;
    VerletIntegrator integrator(0.01);
    PerturbNonbondedForce* nonbonded = new PerturbNonbondedForce();
    nonbonded->setNonbondedMethod(PerturbNonbondedForce::CutoffNonPeriodic);
    for (int i = 0; i < 5; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(0, 1.5, 0, 0, 1.5, 0, false, false);
    }

    nonbonded->setSoftcoreAlpha(1.0);
    nonbonded->addGlobalParameter("lambdaA_charge", 1.0);
    nonbonded->addGlobalParameter("lambdaB_charge", 0.0);
    nonbonded->addGlobalParameter("lambdaC_charge", 0.0);
    nonbonded->addGlobalParameter("rest2_scale", 1.0);
    nonbonded->addGlobalParameter("lambdaA_vdw", 1.0);
    nonbonded->addGlobalParameter("lambdaB_vdw", 0.0);

    const double cutoff = 3.5;
    nonbonded->setCutoffDistance(cutoff);
    const double eps = 30.0;
    nonbonded->setReactionFieldDielectric(eps);
    vector<pair<int, int> > bonds = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    nonbonded->createExceptionsFromBonds(bonds, 0.0, 0.0);
    int first14, second14;
    for (int i = 0; i < nonbonded->getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        nonbonded->getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if ((particle1 == 0 && particle2 == 3) || (particle1 == 3 && particle2 == 0))
            first14 = i;
        if ((particle1 == 1 && particle2 == 4) || (particle1 == 4 && particle2 == 1))
            second14 = i;
    }
    system.addForce(nonbonded);
    ASSERT(!nonbonded->usesPeriodicBoundaryConditions());
    ASSERT(!system.usesPeriodicBoundaryConditions());
    Context context(system, integrator, platform);
    vector<Vec3> positions(5);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(1, 0, 0);
    positions[2] = Vec3(2, 0, 0);
    positions[3] = Vec3(3, 0, 0);
    positions[4] = Vec3(4, 0, 0);
    context.setPositions(positions);
    for (int i = 1; i < 5; ++i) {

        // Test LJ forces

        nonbonded->setParticleParameters(0, 0, 1.5, 1, 0, 1.5, 1, false, false);
        for (int j = 1; j < 5; ++j)
            nonbonded->setParticleParameters(j, 0, 1.5, 0, 0, 1.5, 0, false, false);
        nonbonded->setParticleParameters(i, 0, 1.5, 1, 0, 1.5, 1, false, false);
        nonbonded->setExceptionParameters(first14, 0, 3, 0, 1.5, 0.0);
        nonbonded->setExceptionParameters(second14, 1, 4, 0, 1.5, 0.0);
        context.reinitialize(true);
        State state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces = state.getForces();
        double r = positions[i][0];
        double x = 1.5/r;
        double e = 1.0;
        double force = 4.0*e*(12*std::pow(x, 12.0)-6*std::pow(x, 6.0))/r;
        double energy = 4.0*e*(std::pow(x, 12.0)-std::pow(x, 6.0));
        if (i == 3) {
            force *= 0.0;
            energy *= 0.0;
        }
        if (i < 3 || r > cutoff) {
            force = 0;
            energy = 0;
        }
        ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[0], TOL);
        ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[i], TOL);
        ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), TOL);

        // Test Coulomb forces

        const double q = 0.7;
        nonbonded->setParticleParameters(0, q, 1.5, 0, q, 1.5, 0, false, false);
        nonbonded->setParticleParameters(i, q, 1.5, 0, q, 1.5, 0, false, false);
        nonbonded->setExceptionParameters(first14, 0, 3,  0, 1.5, 0);
        nonbonded->setExceptionParameters(second14, 1, 4, 0, 1.5, 0);
        context.reinitialize(true);
        state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces2 = state.getForces();
        force = ONE_4PI_EPS0*q*q/(r*r);
        energy = ONE_4PI_EPS0*q*q/r;
        if (i == 3) {
            force *= 0.0;
            energy *= 0.0;
        }
        if (i < 3 || r > cutoff) {
            force = 0;
            energy = 0;
        }
        ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces2[0], TOL);
        ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces2[i], TOL);
        ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), TOL);
    }
}

void testPeriodic() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    PerturbNonbondedForce* nonbonded = new PerturbNonbondedForce();
    nonbonded->addParticle(1.0, 1, 0, 1.0, 1, 0, false, false);
    nonbonded->addParticle(1.0, 1, 0, 1.0, 1, 0, false, false);
    nonbonded->addParticle(1.0, 1, 0, 1.0, 1, 0, false, false);
    nonbonded->addException(0, 1, 0.0, 1.0, 0.0);
    nonbonded->setNonbondedMethod(PerturbNonbondedForce::CutoffPeriodic);

    nonbonded->setSoftcoreAlpha(1.0);
    nonbonded->addGlobalParameter("lambdaA_charge", 1.0);
    nonbonded->addGlobalParameter("lambdaB_charge", 0.0);
    nonbonded->addGlobalParameter("lambdaC_charge", 0.0);
    nonbonded->addGlobalParameter("rest2_scale", 1.0);
    nonbonded->addGlobalParameter("lambdaA_vdw", 1.0);
    nonbonded->addGlobalParameter("lambdaB_vdw", 0.0);

    const double cutoff = 2.0;
    nonbonded->setCutoffDistance(cutoff);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    system.addForce(nonbonded);
    ASSERT(nonbonded->usesPeriodicBoundaryConditions());
    ASSERT(system.usesPeriodicBoundaryConditions());
    Context context(system, integrator, platform);
    vector<Vec3> positions(3);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(2, 0, 0);
    positions[2] = Vec3(3, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    const double eps = 78.3;
    const double krf = (1.0/(cutoff*cutoff*cutoff))*(eps-1.0)/(2.0*eps+1.0);
    const double crf = (1.0/cutoff)*(3.0*eps)/(2.0*eps+1.0);
    const double force = ONE_4PI_EPS0*(1.0)*(1.0-2.0*krf*1.0);
    ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), forces[2], TOL);
    ASSERT_EQUAL_TOL(2*ONE_4PI_EPS0*(1.0)*(1.0+krf*1.0-crf), state.getPotentialEnergy(), TOL);
}

void testPeriodicExceptions() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    PerturbNonbondedForce* nonbonded = new PerturbNonbondedForce();
    nonbonded->addParticle(1.0, 1, 0, 1.0, 1, 0, false, false);
    nonbonded->addParticle(1.0, 1, 0, 1.0, 1, 0, false, false);
    //nonbonded->addException(0, 1, 0.0, 1.0, 0.0);
    nonbonded->setNonbondedMethod(PerturbNonbondedForce::CutoffPeriodic);
    
    nonbonded->setSoftcoreAlpha(1.0);
    nonbonded->addGlobalParameter("lambdaA_charge", 1.0);
    nonbonded->addGlobalParameter("lambdaB_charge", 0.0);
    nonbonded->addGlobalParameter("lambdaC_charge", 0.0);
    nonbonded->addGlobalParameter("rest2_scale", 1.0);
    nonbonded->addGlobalParameter("lambdaA_vdw", 1.0);
    nonbonded->addGlobalParameter("lambdaB_vdw", 0.0);

    const double cutoff = 2.0;
    nonbonded->setCutoffDistance(cutoff);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    system.addForce(nonbonded);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(3, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    vector<Vec3> forces = state.getForces();

    double rfDielectric = nonbonded->getReactionFieldDielectric();
    double krf = pow(cutoff, -3.0)*(rfDielectric-1.0)/(2.0*rfDielectric+1.0);
    double crf = (1.0/cutoff)*(3.0*rfDielectric)/(2.0*rfDielectric+1.0);
    double r = 1.0, inverseR = 1.0/r;
    double force = ONE_4PI_EPS0 *(inverseR - 2.0 * krf * r * r);

    ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0 * (inverseR + krf * r * r - crf), state.getPotentialEnergy(), TOL);
}

void testTriclinic() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    Vec3 a(3.1, 0, 0);
    Vec3 b(0.4, 3.5, 0);
    Vec3 c(-0.1, -0.5, 4.0);
    system.setDefaultPeriodicBoxVectors(a, b, c);
    VerletIntegrator integrator(0.01);
    PerturbNonbondedForce* nonbonded = new PerturbNonbondedForce();
    nonbonded->addParticle(1.0, 1, 0, 1.0, 1, 0, false, false);
    nonbonded->addParticle(1.0, 1, 0, 1.0, 1, 0, false, false);
    nonbonded->setNonbondedMethod(PerturbNonbondedForce::CutoffPeriodic);

    nonbonded->setSoftcoreAlpha(1.0);
    nonbonded->addGlobalParameter("lambdaA_charge", 1.0);
    nonbonded->addGlobalParameter("lambdaB_charge", 0.0);
    nonbonded->addGlobalParameter("lambdaC_charge", 0.0);
    nonbonded->addGlobalParameter("rest2_scale", 1.0);
    nonbonded->addGlobalParameter("lambdaA_vdw", 1.0);
    nonbonded->addGlobalParameter("lambdaB_vdw", 0.0);

    const double cutoff = 1.5;
    nonbonded->setCutoffDistance(cutoff);
    system.addForce(nonbonded);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    const double eps = 78.3;
    const double krf = (1.0/(cutoff*cutoff*cutoff))*(eps-1.0)/(2.0*eps+1.0);
    const double crf = (1.0/cutoff)*(3.0*eps)/(2.0*eps+1.0);
    for (int iteration = 0; iteration < 50; iteration++) {
        // Generate random positions for the two particles.

        positions[0] = a*genrand_real2(sfmt) + b*genrand_real2(sfmt) + c*genrand_real2(sfmt);
        positions[1] = a*genrand_real2(sfmt) + b*genrand_real2(sfmt) + c*genrand_real2(sfmt);
        context.setPositions(positions);

        // Loop over all possible periodic copies and find the nearest one.

        Vec3 delta;
        double distance2 = 100.0;
        for (int i = -1; i < 2; i++)
            for (int j = -1; j < 2; j++)
                for (int k = -1; k < 2; k++) {
                    Vec3 d = positions[1]-positions[0]+a*i+b*j+c*k;
                    if (d.dot(d) < distance2) {
                        delta = d;
                        distance2 = d.dot(d);
                    }
                }
        double distance = sqrt(distance2);

        // See if the force and energy are correct.

        State state = context.getState(State::Forces | State::Energy);
        if (distance >= cutoff) {
            ASSERT_EQUAL(0.0, state.getPotentialEnergy());
            ASSERT_EQUAL_VEC(Vec3(0, 0, 0), state.getForces()[0], 0);
            ASSERT_EQUAL_VEC(Vec3(0, 0, 0), state.getForces()[1], 0);
        }
        else {
            const Vec3 force = delta*ONE_4PI_EPS0*(-1.0/(distance*distance*distance)+2.0*krf);
            ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(1.0/distance+krf*distance*distance-crf), state.getPotentialEnergy(), 1e-4);
            ASSERT_EQUAL_VEC(force, state.getForces()[0], 1e-4);
            ASSERT_EQUAL_VEC(-force, state.getForces()[1], 1e-4);
        }
    }
}


void testHugeSystem(double tol=1e-5) {
    // Create a system with over 3 million particles.
    
    const int gridSize = 150;
    const int numParticles = gridSize*gridSize*gridSize;
    const double spacing = 0.3;
    const double boxSize = gridSize*spacing;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    PerturbNonbondedForce* force = new PerturbNonbondedForce();

    force->setSoftcoreAlpha(1.0);
    force->addGlobalParameter("lambdaA_charge", 1.0);
    force->addGlobalParameter("lambdaB_charge", 0.0);
    force->addGlobalParameter("lambdaC_charge", 0.0);
    force->addGlobalParameter("rest2_scale", 1.0);
    force->addGlobalParameter("lambdaA_vdw", 1.0);
    force->addGlobalParameter("lambdaB_vdw", 0.0);

    system.addForce(force);
    force->setNonbondedMethod(PerturbNonbondedForce::CutoffPeriodic);
    force->setCutoffDistance(1.0);
    //force->setUseSwitchingFunction(false);
    //force->setSwitchingDistance(0.9);
    vector<Vec3> positions;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            for (int k = 0; k < gridSize; k++) {
                system.addParticle(1.0);
                force->addParticle(0.0, 0.1, 1.0, 0.0, 0.1, 1.0, false, false);
                positions.push_back(Vec3(i*spacing+genrand_real2(sfmt)*0.1, j*spacing+genrand_real2(sfmt)*0.1, k*spacing+genrand_real2(sfmt)*0.1));
            }
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);

    // Compute the norm of the force.

    State state = context.getState(State::Forces);
    double norm = 0.0;
    for (int i = 0; i < numParticles; ++i) {
        Vec3 f = state.getForces()[i];
        norm += f[0]*f[0] + f[1]*f[1] + f[2]*f[2];
    }
    norm = std::sqrt(norm);
    
    // Take a small step in the direction of the energy gradient and see whether the potential energy changes by the expected amount.

    const double delta = 0.3;
    double step = 0.5*delta/norm;
    vector<Vec3> positions2(numParticles), positions3(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        Vec3 p = positions[i];
        Vec3 f = state.getForces()[i];
        positions2[i] = Vec3(p[0]-f[0]*step, p[1]-f[1]*step, p[2]-f[2]*step);
        positions3[i] = Vec3(p[0]+f[0]*step, p[1]+f[1]*step, p[2]+f[2]*step);
    }
    context.setPositions(positions2);
    State state2 = context.getState(State::Energy);
    context.setPositions(positions3);
    State state3 = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(state2.getPotentialEnergy(), state3.getPotentialEnergy()+norm*delta, tol)
}

void testDispersionCorrection() {
    // Create a box full of identical particles.

    int gridSize = 5;
    int numParticles = gridSize*gridSize*gridSize;
    double boxSize = gridSize*0.7;
    double cutoff = boxSize/3;
    System system;
    VerletIntegrator integrator(0.01);
    PerturbNonbondedForce* nonbonded = new PerturbNonbondedForce();
    vector<Vec3> positions(numParticles);
    int index = 0;
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            for (int k = 0; k < gridSize; k++) {
                system.addParticle(1.0);
                nonbonded->addParticle(0, 1.1, 0.5, 0, 1.1, 0.5, false, false);
                positions[index] = Vec3(i*boxSize/gridSize, j*boxSize/gridSize, k*boxSize/gridSize);
                index++;
            }
    nonbonded->setNonbondedMethod(PerturbNonbondedForce::CutoffPeriodic);
    nonbonded->setCutoffDistance(cutoff);

    nonbonded->setSoftcoreAlpha(1.0);
    nonbonded->addGlobalParameter("lambdaA_charge", 1.0);
    nonbonded->addGlobalParameter("lambdaB_charge", 0.0);
    nonbonded->addGlobalParameter("lambdaC_charge", 0.0);
    nonbonded->addGlobalParameter("rest2_scale", 1.0);
    nonbonded->addGlobalParameter("lambdaA_vdw", 1.0);
    nonbonded->addGlobalParameter("lambdaB_vdw", 0.0);

    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    system.addForce(nonbonded);

    // See if the correction has the correct value.

    Context context(system, integrator, platform);
    context.setPositions(positions);
    double energy1 = context.getState(State::Energy).getPotentialEnergy();
    nonbonded->setUseDispersionCorrection(false);
    context.reinitialize();
    context.setPositions(positions);
    double energy2 = context.getState(State::Energy).getPotentialEnergy();
    double term1 = (0.5*pow(1.1, 12)/pow(cutoff, 9))/9;
    double term2 = (0.5*pow(1.1, 6)/pow(cutoff, 3))/3;
    double expected = 8*M_PI*numParticles*numParticles*(term1-term2)/(boxSize*boxSize*boxSize);
    ASSERT_EQUAL_TOL(expected, energy1-energy2, 1e-4);

    // Now modify half the particles to be different, and see if it is still correct.

    int numType2 = 0;
    for (int i = 0; i < numParticles; i += 2) {
        nonbonded->setParticleParameters(i, 0, 1, 1, 0, 1, 1, false, false);
        numType2++;
    }
    int numType1 = numParticles-numType2;
    nonbonded->updateParametersInContext(context);
    energy2 = context.getState(State::Energy).getPotentialEnergy();
    nonbonded->setUseDispersionCorrection(true);
    context.reinitialize();
    context.setPositions(positions);
    energy1 = context.getState(State::Energy).getPotentialEnergy();
    term1 = ((numType1*(numType1+1))/2)*(0.5*pow(1.1, 12)/pow(cutoff, 9))/9;
    term2 = ((numType1*(numType1+1))/2)*(0.5*pow(1.1, 6)/pow(cutoff, 3))/3;
    term1 += ((numType2*(numType2+1))/2)*(1*pow(1.0, 12)/pow(cutoff, 9))/9;
    term2 += ((numType2*(numType2+1))/2)*(1*pow(1.0, 6)/pow(cutoff, 3))/3;
    double combinedSigma = 0.5*(1+1.1);
    double combinedEpsilon = sqrt(1*0.5);
    term1 += (numType1*numType2)*(combinedEpsilon*pow(combinedSigma, 12)/pow(cutoff, 9))/9;
    term2 += (numType1*numType2)*(combinedEpsilon*pow(combinedSigma, 6)/pow(cutoff, 3))/3;
    term1 /= (numParticles*(numParticles+1))/2;
    term2 /= (numParticles*(numParticles+1))/2;
    expected = 8*M_PI*numParticles*numParticles*(term1-term2)/(boxSize*boxSize*boxSize);
    ASSERT_EQUAL_TOL(expected, energy1-energy2, 1e-4);
}


void testSwitchingFunction(PerturbNonbondedForce::PerturbNonbondedMethod method) {
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(6, 0, 0), Vec3(0, 6, 0), Vec3(0, 0, 6));
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    PerturbNonbondedForce* nonbonded = new PerturbNonbondedForce();
    nonbonded->addParticle(0, 1.2, 1, 0.2, 1.2, 1, false, false);
    nonbonded->addParticle(0, 1.4, 2, 0, 1.4, 2, false, false);
    nonbonded->setNonbondedMethod(method);
    nonbonded->setCutoffDistance(2.0);
    nonbonded->setUseSwitchingFunction(true);
    nonbonded->setSwitchingDistance(1.5);
    nonbonded->setUseDispersionCorrection(false);

    nonbonded->setSoftcoreAlpha(1.0);
    nonbonded->addGlobalParameter("lambdaA_charge", 1.0);
    nonbonded->addGlobalParameter("lambdaB_charge", 0.0);
    nonbonded->addGlobalParameter("lambdaC_charge", 0.0);
    nonbonded->addGlobalParameter("rest2_scale", 1.0);
    nonbonded->addGlobalParameter("lambdaA_vdw", 1.0);
    nonbonded->addGlobalParameter("lambdaB_vdw", 0.0);

    system.addForce(nonbonded);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    double eps = SQRT_TWO;
    
    // Compute the interaction at various distances.
    
    for (double r = 1.0; r < 2.5; r += 0.1) {
        positions[1] = Vec3(r, 0, 0);
        context.setPositions(positions);
        State state = context.getState(State::Forces | State::Energy);
        
        // See if the energy is correct.
        
        double x = 1.3/r;
        double expectedEnergy = 4.0*eps*(std::pow(x, 12.0)-std::pow(x, 6.0));
        double switchValue;
        if (r <= 1.5)
            switchValue = 1;
        else if (r >= 2.0)
            switchValue = 0;
        else {
            double t = (r-1.5)/0.5;
            switchValue = 1+t*t*t*(-10+t*(15-t*6));
        }
        ASSERT_EQUAL_TOL(switchValue*expectedEnergy, state.getPotentialEnergy(), TOL);
        
        // See if the force is the gradient of the energy.
        
        double delta = 1e-3;
        positions[1] = Vec3(r-delta, 0, 0);
        context.setPositions(positions);
        double e1 = context.getState(State::Energy).getPotentialEnergy();
        positions[1] = Vec3(r+delta, 0, 0);
        context.setPositions(positions);
        double e2 = context.getState(State::Energy).getPotentialEnergy();
        ASSERT_EQUAL_TOL((e2-e1)/(2*delta), state.getForces()[0][0], 1e-3);
    }
}

void testTwoForces() {
    // Create a system with two NonbondedForces.
    
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    PerturbNonbondedForce* nb1 = new PerturbNonbondedForce();
    nb1->addParticle(-1.5, 1, 1.2, -1.5, 1, 1.2, false, false);
    nb1->addParticle(0.5, 1, 1.0, 0.5, 1, 1.0, false, false);

    nb1->setSoftcoreAlpha(1.0);
    nb1->addGlobalParameter("lambdaA_charge", 1.0);
    nb1->addGlobalParameter("lambdaB_charge", 0.0);
    nb1->addGlobalParameter("lambdaC_charge", 0.0);
    nb1->addGlobalParameter("rest2_scale", 1.0);
    nb1->addGlobalParameter("lambdaA_vdw", 1.0);
    nb1->addGlobalParameter("lambdaB_vdw", 0.0);

    system.addForce(nb1);
    PerturbNonbondedForce* nb2 = new PerturbNonbondedForce();
    nb2->addParticle(0.4, 1.4, 0.5, 0.4, 1.4, 0.5, false, false);
    nb2->addParticle(0.3, 1.8, 1.0, 0.3, 1.8, 1.0, false, false);
    nb2->setForceGroup(1);

    nb2->setSoftcoreAlpha(1.0);
    nb2->addGlobalParameter("lambdaA_charge", 1.0);
    nb2->addGlobalParameter("lambdaB_charge", 0.0);
    nb2->addGlobalParameter("lambdaC_charge", 0.0);
    nb2->addGlobalParameter("rest2_scale", 1.0);
    nb2->addGlobalParameter("lambdaA_vdw", 1.0);
    nb2->addGlobalParameter("lambdaB_vdw", 0.0);

    system.addForce(nb2);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(1.5, 0, 0);
    context.setPositions(positions);
    State state1 = context.getState(State::Energy, false, 1<<0);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(-1.5*0.5)/1.5 + 4.0*sqrt(1.2*1.0)*(pow(1.0/1.5, 12.0)-pow(1.0/1.5, 6.0)), state1.getPotentialEnergy(), TOL);
    State state2 = context.getState(State::Energy, false, 1<<1);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(0.4*0.3)/1.5 + 4.0*sqrt(0.5*1.0)*(pow(1.6/1.5, 12.0)-pow(1.6/1.5, 6.0)), state2.getPotentialEnergy(), TOL);
    State state = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(state1.getPotentialEnergy()+state2.getPotentialEnergy(), state.getPotentialEnergy(), TOL);
    
    // Try modifying them and see if they're still correct.
    
    nb1->setParticleParameters(0, -1.2, 1.1, 1.4, -1.2, 1.1, 1.4, false, false);
    nb1->updateParametersInContext(context);
    nb2->setParticleParameters(0, 0.5, 1.6, 0.6, 0.5, 1.6, 0.6, false, false);
    nb2->updateParametersInContext(context);
    state1 = context.getState(State::Energy, false, 1<<0);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(-1.2*0.5)/1.5 + 4.0*sqrt(1.4*1.0)*(pow(1.05/1.5, 12.0)-pow(1.05/1.5, 6.0)), state1.getPotentialEnergy(), TOL);
    state2 = context.getState(State::Energy, false, 1<<1);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(0.5*0.3)/1.5 + 4.0*sqrt(0.6*1.0)*(pow(1.7/1.5, 12.0)-pow(1.7/1.5, 6.0)), state2.getPotentialEnergy(), TOL);
    
    // Make sure it also works with PME.
    nb1->setForceGroup(0);
    nb1->setReciprocalSpaceForceGroup(1);
    nb2->setForceGroup(2);
    nb2->setReciprocalSpaceForceGroup(3);

    nb1->setNonbondedMethod(PerturbNonbondedForce::PME);
    nb2->setNonbondedMethod(PerturbNonbondedForce::PME);
    context.reinitialize(true);
    State nb1_direct = context.getState(State::Energy, false, 1<<0);
    State nb1_recip = context.getState(State::Energy, false, 1<<1);
    State nb2_direct = context.getState(State::Energy, false, 1<<2);
    State nb2_recip = context.getState(State::Energy, false, 1<<3);
    
    state = context.getState(State::Energy);    
    ASSERT_EQUAL_TOL(nb1_direct.getPotentialEnergy()+nb2_direct.getPotentialEnergy() + nb1_recip.getPotentialEnergy()+nb2_recip.getPotentialEnergy(), state.getPotentialEnergy(), TOL);
}


void testEwaldExceptions() {
    // Create a minimal system using LJPME.

    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    PerturbNonbondedForce* force = new PerturbNonbondedForce();

    force->setSoftcoreAlpha(1.0);
    force->addGlobalParameter("lambdaA_charge", 1.0);
    force->addGlobalParameter("lambdaB_charge", 0.0);
    force->addGlobalParameter("lambdaC_charge", 0.0);
    force->addGlobalParameter("rest2_scale", 1.0);
    force->addGlobalParameter("lambdaA_vdw", 1.0);
    force->addGlobalParameter("lambdaB_vdw", 0.0);

    system.addForce(force);
    force->setNonbondedMethod(PerturbNonbondedForce::LJPME);
    force->setCutoffDistance(1.0);
    force->addParticle(1.0, 0.5, 1.0, 1.0, 0.5, 1.0, false, false);
    force->addParticle(1.0, 0.5, 1.0, 1.0, 0.5, 1.0, false, false);
    force->addParticle(-1.0, 0.5, 1.0, -1.0, 0.5, 1.0, false, false);
    force->addParticle(-1.0, 0.5, 1.0, -1.0, 0.5, 1.0, false, false);
    vector<Vec3> positions = {
        Vec3(0, 0, 0),
        Vec3(1.5, 0, 0),
        Vec3(0, 0.5, 0.5),
        Vec3(0.2, 1.3, 0)
    };
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    
    // Compute the energy.
    
    double e1 = context.getState(State::Energy).getPotentialEnergy();

    // Add a periodic exception and see if the energy changes by the correct amount.

    force->addException(0, 1, 0.0, 0.8, 0.0);
    force->setExceptionsUsePeriodicBoundaryConditions(true);
    context.reinitialize(true);
    double e2 = context.getState(State::Energy).getPotentialEnergy();
    double r = 0.5;
    double expectedChange = ONE_4PI_EPS0*(0.2-1.0)/r + 4*2.0*(pow(0.8/r, 12)-pow(0.8/r, 6)) - 4*1.0*(pow(0.5/r, 12)-pow(0.5/r, 6));
    ASSERT_EQUAL_TOL(expectedChange, e2-e1, 1e-5);
}

void testDirectAndReciprocal() {
    // Create a minimal system with direct space and reciprocal space in different force groups.

    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    PerturbNonbondedForce* force = new PerturbNonbondedForce();

    force->setSoftcoreAlpha(1.0);
    force->addGlobalParameter("lambdaA_charge", 1.0);
    force->addGlobalParameter("lambdaB_charge", 0.0);
    force->addGlobalParameter("lambdaC_charge", 0.0);
    force->addGlobalParameter("rest2_scale", 1.0);
    force->addGlobalParameter("lambdaA_vdw", 1.0);
    force->addGlobalParameter("lambdaB_vdw", 0.0);

    system.addForce(force);
    force->setNonbondedMethod(PerturbNonbondedForce::PME);
    force->setCutoffDistance(1.0);
    force->setReciprocalSpaceForceGroup(1);
    force->addParticle(1.0, 0.5, 1.0, 1.0, 0.5, 1.0, false, false);
    force->addParticle(1.0, 0.5, 1.0, 1.0, 0.5, 1.0, false, false);
    force->addParticle(-1.0, 0.5, 1.0, -1.0, 0.5, 1.0, false, false);
    force->addParticle(-1.0, 0.5, 1.0, -1.0, 0.5, 1.0, false, false);
    force->addException(0, 2, 0.0, 0.5, 0.0);
    vector<Vec3> positions = {
        Vec3(0, 0, 0),
        Vec3(1.5, 0, 0),
        Vec3(0, 0.5, 0.5),
        Vec3(0.2, 1.3, 0)
    };
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    
    // Compute the direct and reciprocal space energies together and separately.
    
    double e1 = context.getState(State::Energy).getPotentialEnergy();
    double e2 = context.getState(State::Energy, true, 1<<0).getPotentialEnergy();
    double e3 = context.getState(State::Energy, true, 1<<1).getPotentialEnergy();
    ASSERT_EQUAL_TOL(e1, e2+e3, 1e-5);
    ASSERT(e2 != 0);
    ASSERT(e3 != 0);
    
    // Completely disable the direct space calculation.
    
    force->setIncludeDirectSpace(false);
    context.reinitialize(true);
    double e4 = context.getState(State::Energy).getPotentialEnergy();
    ASSERT_EQUAL_TOL(e3, e4, 1e-5);
}

void testReordering() {
    // Check that reordering of atoms doesn't alter their positions.
    
    const int numParticles = 200;
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
        nonbonded->addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, false);
        positions.push_back(Vec3(genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5)*20);
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    integrator.step(1);
    State state = context.getState(State::Positions | State::Velocities);
    for (int i = 0; i < numParticles; i++) {
        ASSERT_EQUAL_VEC(positions[i], state.getPositions()[i], 1e-6);
    }
}

void runPlatformTests();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);

        
        testCoulomb();
        testLJ();
        testExclusionsAnd14();
        testCutoff();
        testCutoff14();
        testPeriodic();
        testPeriodicExceptions();
        testTriclinic();
        testDispersionCorrection();
        

        // Switching function is not supported.
        /*testSwitchingFunction(PerturbNonbondedForce::CutoffNonPeriodic);
        testSwitchingFunction(PerturbNonbondedForce::PME);*/

        testTwoForces();
        //testEwaldExceptions(); // We don't test Ewald exceptions because LJPME is not implemented.
        testDirectAndReciprocal();
        
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
