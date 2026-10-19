/* -------------------------------------------------------------------------- *
 *                              OpenMMPerturbNonbonded                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded by the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
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

#include "ReferencePerturbNonbondedKernels.h"
#include "ReferencePerturbLJCoulombIxn.h"
#include "openmm/reference/ReferenceLJCoulomb14.h"
#include "openmm/reference/ReferenceBondForce.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include "openmm/internal/ContextImpl.h"
#include "internal/PerturbNonbondedForceImpl.h"
#include <iostream>

using namespace PerturbNonbondedPlugin;
using namespace OpenMM;
using namespace std;

double evalIntegral(double r, double rs, double rc, double sigma) {
    // Compute the indefinite integral of the LJ interaction multiplied by the switching function.
    // This is a large and somewhat horrifying expression, though it does grow on you if you look
    // at it long enough.  Perhaps it could be simplified further, but I got tired of working on it.
    
    double A = 1/(rc-rs);
    double A2 = A*A;
    double A3 = A2*A;
    double sig2 = sigma*sigma;
    double sig6 = sig2*sig2*sig2;
    double rs2 = rs*rs;
    double rs3 = rs*rs2;
    double r2 = r*r;
    double r3 = r*r2;
    double r4 = r*r3;
    double r5 = r*r4;
    double r6 = r*r5;
    double r9 = r3*r6;
    return sig6*A3*((
        sig6*(
            + rs3*28*(6*rs2*A2 + 15*rs*A + 10)
            - r*rs2*945*(rs2*A2 + 2*rs*A + 1)
            + r2*rs*1080*(2*rs2*A2 + 3*rs*A + 1)
            - r3*420*(6*rs2*A2 + 6*rs*A + 1)
            + r4*756*(2*rs*A2 + A)
            - r5*378*A2)
        -r6*(
            + rs3*84*(6*rs2*A2 + 15*rs*A + 10)
            - r*rs2*3780*(rs2*A2 + 2*rs*A + 1)
            + r2*rs*7560*(2*rs2*A2 + 3*rs*A + 1))
        )/(252*r9)
     - log(r)*10*(6*rs2*A2 + 6*rs*A + 1)
     + r*15*(2*rs*A2 + A)
     - r2*3*A2
    );
}

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->positions;
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->forces;
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return data->periodicBoxVectors;
}

ReferenceCalcPerturbNonbondedForceKernel::~ReferenceCalcPerturbNonbondedForceKernel() {
    if (neighborList != NULL)
        delete neighborList;
}

void ReferenceCalcPerturbNonbondedForceKernel::initialize(const System& system, const PerturbNonbondedForce& force) {

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionsWithOffsets.insert(exception);
    }
    numParticles = force.getNumParticles();
    exclusions.resize(numParticles);
    vector<int> nb14s;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        exclusions[particle1].insert(particle2);
        exclusions[particle2].insert(particle1);
        if (chargeProd != 0.0 || epsilon != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end()) {
            nb14Index[i] = nb14s.size();
            nb14s.push_back(i);
        }
    }

    // Build the arrays.

    num14 = nb14s.size();
    bonded14IndexArray.resize(num14, vector<int>(2));
    bonded14ParamArray.resize(num14, vector<double>(3));
    particleParamArray.resize(numParticles, vector<double>(3));
    baseParticleParams.resize(numParticles);
    baseExceptionParams.resize(num14);    

    particleCoulombParams.resize(numParticles, array<double, 3>());
    particleSigmaEpsilonParams.resize(numParticles, array<double, 4>());
    particleTypeParams.resize(numParticles, array<bool, 2>());
    perturbSigmaEpsilonParams_raw.resize(numParticles, array<double, 4>());

    for (int i = 0; i < numParticles; ++i){
        //force.getParticleParameters(i, baseParticleParams[i][0], baseParticleParams[i][1], baseParticleParams[i][2]);
        double chargeA, sigmaA, epsilonA;
        double chargeB, sigmaB, epsilonB;
        bool isDummy, isREST2;
        force.getParticleParameters(i, chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2);
        
        double chargeC = 0.0;
        if(isDummy){
            chargeC = 0;
        }
        else{
            chargeC = (chargeA + chargeB) * 0.5;
        }
        particleCoulombParams[i] = {chargeA, chargeB, chargeC};
        particleTypeParams[i] = {isDummy, isREST2};
        particleSigmaEpsilonParams[i] = {0.5 * sigmaA, 2.0 * sqrt(epsilonA), 0.5 * sigmaB, 2.0 * sqrt(epsilonB)};
        perturbSigmaEpsilonParams_raw[i] = {sigmaA, epsilonA, sigmaB, epsilonB};
    }

    softcore_alpha = force.getSoftcoreAlpha();
       
    for (int i = 0; i < num14; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(nb14s[i], particle1, particle2, baseExceptionParams[i][0], baseExceptionParams[i][1], baseExceptionParams[i][2]);
        bonded14IndexArray[i][0] = particle1;
        bonded14IndexArray[i][1] = particle2;
    }
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        particleParamOffsets[make_pair(param, particle)] = {charge, sigma, epsilon};
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionParamOffsets[make_pair(param, nb14Index[exception])] = {charge, sigma, epsilon};
    }
    nonbondedMethod = PerturbNonbondedForce::PerturbNonbondedMethod(force.getNonbondedMethod());
    nonbondedCutoff = force.getCutoffDistance();
    if (nonbondedMethod == PerturbNonbondedForce::NoCutoff) {
        neighborList = NULL;
        useSwitchingFunction = false;
    }
    else {
        neighborList = new NeighborList();
        useSwitchingFunction = force.getUseSwitchingFunction();
        switchingDistance = force.getSwitchingDistance();
    }
    if (nonbondedMethod == PerturbNonbondedForce::Ewald) {
        double alpha;
        PerturbNonbondedForceImpl::calcEwaldParameters(system, force, alpha, kmax[0], kmax[1], kmax[2]);
        ewaldAlpha = alpha;
    }
    else if (nonbondedMethod == PerturbNonbondedForce::PME) {
        double alpha;
        PerturbNonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSize[0], gridSize[1], gridSize[2], false);
        ewaldAlpha = alpha;
    }
    else if (nonbondedMethod == PerturbNonbondedForce::LJPME) {
        double alpha;
        PerturbNonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSize[0], gridSize[1], gridSize[2], false);
        ewaldAlpha = alpha;
        PerturbNonbondedForceImpl::calcPMEParameters(system, force, alpha, dispersionGridSize[0], dispersionGridSize[1], dispersionGridSize[2], true);
        ewaldDispersionAlpha = alpha;
        useSwitchingFunction = false;
    }
    if (nonbondedMethod == PerturbNonbondedForce::NoCutoff ||
            nonbondedMethod == PerturbNonbondedForce::CutoffNonPeriodic)
        exceptionsArePeriodic = false;
    else
        exceptionsArePeriodic = force.getExceptionsUsePeriodicBoundaryConditions();
    rfDielectric = force.getReactionFieldDielectric();
    if (force.getUseDispersionCorrection()){
        dispersionCoefficient = 0.0;
        useDispersionCorrection = true;
    }
    else
        dispersionCoefficient = 0.0;
}

double ReferenceCalcPerturbNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    //computeParameters(context);
    computePerturbParameters(context);
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    double energy = 0;
    ReferencePerturbLJCoulombIxn clj;
    bool periodic = (nonbondedMethod == PerturbNonbondedForce::CutoffPeriodic);
    bool ewald  = (nonbondedMethod == PerturbNonbondedForce::Ewald);
    bool pme  = (nonbondedMethod == PerturbNonbondedForce::PME);
    bool ljpme = (nonbondedMethod == PerturbNonbondedForce::LJPME);
    if (nonbondedMethod != PerturbNonbondedForce::NoCutoff) {
        computeNeighborListVoxelHash(*neighborList, numParticles, posData, exclusions, extractBoxVectors(context), periodic || ewald || pme || ljpme, nonbondedCutoff, 0.0);
        clj.setUseCutoff(nonbondedCutoff, *neighborList, rfDielectric);
    }
    if (periodic || ewald || pme || ljpme) {
        Vec3* boxVectors = extractBoxVectors(context);
        double minAllowedSize = 1.999999*nonbondedCutoff;
        if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize)
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
        clj.setPeriodic(boxVectors);
        clj.setPeriodicExceptions(exceptionsArePeriodic);
    }
    if (ewald)
        clj.setUseEwald(ewaldAlpha, kmax[0], kmax[1], kmax[2]);
    if (pme)
        clj.setUsePME(ewaldAlpha, gridSize);
    if (ljpme){
        clj.setUsePME(ewaldAlpha, gridSize);
        clj.setUseLJPME(ewaldDispersionAlpha, dispersionGridSize);
    }
    if (useSwitchingFunction)
        clj.setUseSwitchingFunction(switchingDistance);
    clj.calculatePairIxn(
        numParticles, posData, 
        particleSigmaEpsilonParamsInRun,
        particleChargesInRun,
        particleTypeParams, 
        exclusions, 
        forceData, 
        includeEnergy ? &energy : NULL, 
        includeDirect, 
        includeReciprocal,
        softcore_alpha,
        lambdaA_vdw,
        lambdaB_vdw);
    if (includeDirect) {
        ReferenceBondForce refBondForce;
        ReferenceLJCoulomb14 nonbonded14;
        if (exceptionsArePeriodic) {
            Vec3* boxVectors = extractBoxVectors(context);
            nonbonded14.setPeriodic(boxVectors);
        }
        refBondForce.calculateForce(num14, bonded14IndexArray, posData, bonded14ParamArray, forceData, includeEnergy ? &energy : NULL, nonbonded14);
        if (periodic || ewald || pme) {
            Vec3* boxVectors = extractBoxVectors(context);
            double volume = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
            // This funcion would update EcorrA and EcorrB with latest rest2_scale.
            double Ecorr = 0.0;
            if (useDispersionCorrection){
                this->getPerturbDispersionCorrections(context);
                Ecorr = lambdaA_vdw * EcorrA + lambdaB_vdw * EcorrB;
                Ecorr = Ecorr / (lambdaA_vdw + lambdaB_vdw);
            }
            
            
            if (useDispersionCorrection)
                energy += Ecorr / volume;
            //energy += dispersionCoefficient/(boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2]);
        }
    }
    return energy;
}

void ReferenceCalcPerturbNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const PerturbNonbondedForce& force, int firstParticle, int lastParticle, int firstException, int lastException) {
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (force.getNumParticleParameterOffsets() != particleParamOffsets.size())
        throw OpenMMException("updateParametersInContext: The number of particle parameter offsets has changed");
    if (force.getNumExceptionParameterOffsets() != exceptionParamOffsets.size())
        throw OpenMMException("updateParametersInContext: The number of exception parameter offsets has changed");
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        if (particleParamOffsets.find(make_pair(param, particle)) == particleParamOffsets.end())
            throw OpenMMException("updateParametersInContext: The parameter or particle index of a particle parameter offset has changed");
    }

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionsWithOffsets.insert(exception);
        if (exceptionParamOffsets.find(make_pair(param, nb14Index[exception])) == exceptionParamOffsets.end())
            throw OpenMMException("updateParametersInContext: The parameter or exception index of an exception parameter offset has changed");
    }
    vector<int> nb14s;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if (nb14Index.find(i) == nb14Index.end()) {
            if (chargeProd != 0.0 || epsilon != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end())
                throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");
        }
        else
            nb14s.push_back(i);
    }
    if (nb14s.size() != num14)
        throw OpenMMException("updateParametersInContext: The number of non-excluded exceptions has changed");

    // Record the values.

    for (int i = firstParticle; i <= lastParticle; ++i){
        //force.getParticleParameters(i, baseParticleParams[i][0], baseParticleParams[i][1], baseParticleParams[i][2]);
        double chargeA, sigmaA, epsilonA;
        double chargeB, sigmaB, epsilonB;
        bool isDummy, isREST2;
        force.getParticleParameters(i, chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2);
        particleTypeParams[i][0] = isDummy;
        particleTypeParams[i][1] = isREST2;

        if (isDummy){
            double chargeC = 0.0;
            particleCoulombParams[i] = {chargeA, chargeB, chargeC};
        } else {
            double chargeC = (chargeA + chargeB) * 0.5;
            particleCoulombParams[i] = {chargeA, chargeB, chargeC};
        }
        particleSigmaEpsilonParams[i] = {0.5 * sigmaA, 2.0 * sqrt(epsilonA), 0.5 * sigmaB, 2.0 * sqrt(epsilonB)};
    }
    updatePerturbParamsFlag = true;

    for (int i = 0; i < num14; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(nb14s[i], particle1, particle2, baseExceptionParams[i][0], baseExceptionParams[i][1], baseExceptionParams[i][2]);
        bonded14IndexArray[i][0] = particle1;
        bonded14IndexArray[i][1] = particle2;
    }
    particleParamOffsets.clear();
    exceptionParamOffsets.clear();
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        particleParamOffsets[make_pair(param, particle)] = {charge, sigma, epsilon};
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionParamOffsets[make_pair(param, nb14Index[exception])] = {charge, sigma, epsilon};
    }
    
    // Recompute the coefficient for the dispersion correction.

    PerturbNonbondedForce::PerturbNonbondedMethod method = force.getNonbondedMethod();
    if (force.getUseDispersionCorrection() && (method == PerturbNonbondedForce::CutoffPeriodic || method == PerturbNonbondedForce::Ewald || method == PerturbNonbondedForce::PME)){
        //dispersionCoefficient = PerturbNonbondedForceImpl::calcDispersionCorrection(context.getSystem(), force);
        useDispersionCorrection = true;
        dispersionCoefficient = 0.0;
    }
}

void ReferenceCalcPerturbNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != PerturbNonbondedForce::PME &&
            nonbondedMethod != PerturbNonbondedForce::LJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME or LJPME");
    alpha = ewaldAlpha;
    nx = gridSize[0];
    ny = gridSize[1];
    nz = gridSize[2];
}

void ReferenceCalcPerturbNonbondedForceKernel::getInteractionEnergyForRealParticlesCuda(
        ContextImpl&, int, CudaArrayDevicePointer, CudaArrayDevicePointer, CudaArrayDevicePointer,
        CudaArrayDevicePointer, CudaArrayDevicePointer, CudaArrayDevicePointer, CudaArrayDevicePointer) {
    throw OpenMMException("CUDA particle energy queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getDirectEnergyForCandidateParticlesCuda(
        ContextImpl&, int, CudaArrayDevicePointer, CudaArrayDevicePointer,
        CudaArrayDevicePointer, CudaArrayDevicePointer) {
    throw OpenMMException("CUDA particle energy queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getDirectEnergyForCandidateParticlesCudaContextPrecision(
        ContextImpl&, int, CudaArrayDevicePointer, CudaArrayDevicePointer,
        CudaArrayDevicePointer, CudaArrayDevicePointer, CudaArrayDevicePointer) {
    throw OpenMMException("CUDA context-precision particle energy queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getPMEPotentialAtPositionsCuda(
        ContextImpl&, int, CudaArrayDevicePointer, CudaArrayDevicePointer) {
    throw OpenMMException("CUDA PME potential queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getPMEPotentialAtPositionsCudaContextPrecision(
        ContextImpl&, int, CudaArrayDevicePointer, CudaArrayDevicePointer,
        CudaArrayDevicePointer) {
    throw OpenMMException("CUDA context-precision PME potential queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getSymmetricPMEReciprocalAtomicEnergiesCuda(
        ContextImpl&, CudaArrayDevicePointer, CudaArrayDevicePointer) {
    throw OpenMMException("CUDA PME reciprocal atomic energy queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getPMEPotentialFromParticlesCuda(
        ContextImpl&, int, CudaArrayDevicePointer, CudaArrayDevicePointer,
        int, CudaArrayDevicePointer, CudaArrayDevicePointer) {
    throw OpenMMException("CUDA PME scratch potential queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getPMEPotentialFromParticlesCudaContextPrecision(
        ContextImpl&, int, CudaArrayDevicePointer, CudaArrayDevicePointer,
        CudaArrayDevicePointer, int, CudaArrayDevicePointer, CudaArrayDevicePointer,
        CudaArrayDevicePointer) {
    throw OpenMMException("CUDA context-precision PME scratch potential queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getPMECandidateSelfEnergiesCuda(
        ContextImpl&, int, int, CudaArrayDevicePointer, CudaArrayDevicePointer,
        CudaArrayDevicePointer) {
    throw OpenMMException("CUDA PME candidate self energy queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getPMECandidateSelfEnergiesCudaContextPrecision(
        ContextImpl&, int, int, CudaArrayDevicePointer, CudaArrayDevicePointer,
        CudaArrayDevicePointer, CudaArrayDevicePointer) {
    throw OpenMMException("CUDA context-precision PME candidate self energy queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getDirectEnergyForCandidateSubsetCuda(
        ContextImpl&, int, CudaArrayDevicePointer, CudaArrayDevicePointer,
        int, CudaArrayDevicePointer, CudaArrayDevicePointer) {
    throw OpenMMException("CUDA PME candidate-subset direct queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getDirectEnergyForCandidateSubsetCudaContextPrecision(
        ContextImpl&, int, CudaArrayDevicePointer, CudaArrayDevicePointer,
        CudaArrayDevicePointer, int, CudaArrayDevicePointer, CudaArrayDevicePointer) {
    throw OpenMMException("CUDA context-precision PME candidate-subset direct queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getDirectEnergyForRealParticlesPmeCuda(
        ContextImpl&, int, CudaArrayDevicePointer, int, CudaArrayDevicePointer) {
    throw OpenMMException("CUDA PME real-particle direct queries require the CUDA Platform.");
}

void ReferenceCalcPerturbNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != PerturbNonbondedForce::LJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using LJPME");
    alpha = ewaldDispersionAlpha;
    nx = dispersionGridSize[0];
    ny = dispersionGridSize[1];
    nz = dispersionGridSize[2];
}

void ReferenceCalcPerturbNonbondedForceKernel::computeParameters(ContextImpl& context) {
    // Compute particle parameters.

    vector<double> charges(numParticles), sigmas(numParticles), epsilons(numParticles);
    for (int i = 0; i < numParticles; i++) {
        charges[i] = baseParticleParams[i][0];
        sigmas[i] = baseParticleParams[i][1];
        epsilons[i] = baseParticleParams[i][2];
    }
    for (auto& offset : particleParamOffsets) {
        double value = context.getParameter(offset.first.first);
        int index = offset.first.second;
        charges[index] += value*offset.second[0];
        sigmas[index] += value*offset.second[1];
        epsilons[index] += value*offset.second[2];
    }
    for (int i = 0; i < numParticles; i++) {
        particleParamArray[i][0] = 0.5*sigmas[i];
        particleParamArray[i][1] = 2.0*sqrt(epsilons[i]);
        particleParamArray[i][2] = charges[i];
    }

    // Compute exception parameters.

    charges.resize(num14);
    sigmas.resize(num14);
    epsilons.resize(num14);
    for (int i = 0; i < num14; i++) {
        charges[i] = baseExceptionParams[i][0];
        sigmas[i] = baseExceptionParams[i][1];
        epsilons[i] = baseExceptionParams[i][2];
    }
    for (auto& offset : exceptionParamOffsets) {
        double value = context.getParameter(offset.first.first);
        int index = offset.first.second;
        charges[index] += value*offset.second[0];
        sigmas[index] += value*offset.second[1];
        epsilons[index] += value*offset.second[2];
    }
    for (int i = 0; i < num14; i++) {
        bonded14ParamArray[i][0] = sigmas[i];
        bonded14ParamArray[i][1] = 4.0*epsilons[i];
        bonded14ParamArray[i][2] = charges[i];
    }
}


void ReferenceCalcPerturbNonbondedForceKernel::computePerturbParameters(ContextImpl& context) {
    // Compute particle parameters.
    lambdaA_charge = context.getParameter("lambdaA_charge");
    lambdaB_charge = context.getParameter("lambdaB_charge");
    lambdaC_charge = context.getParameter("lambdaC_charge");
    lambdaA_vdw = context.getParameter("lambdaA_vdw");
    lambdaB_vdw = context.getParameter("lambdaB_vdw");
    rest2_scale = context.getParameter("rest2_scale");
    
    if (lambdaA_charge != lambdaA_charge_old || lambdaB_charge != lambdaB_charge_old || lambdaC_charge != lambdaC_charge_old || rest2_scale != rest2_scale_old) {
        lambdaA_charge_old = lambdaA_charge;
        lambdaB_charge_old = lambdaB_charge;
        lambdaC_charge_old = lambdaC_charge;
        lambdaA_vdw_old = lambdaA_vdw;
        lambdaB_vdw_old = lambdaB_vdw;
        rest2_scale_old = rest2_scale;
        sqrt_rest2_scale = sqrt(rest2_scale);
        updatePerturbParamsFlag = true;
    } else {
    }

    if (!updatePerturbParamsFlag){
        updatePerturbParamsFlag = false;
        return;
    } else {
        // Interpolate particle charges.
        particleChargesInRun.resize(numParticles);
        particleSigmaEpsilonParamsInRun.resize(numParticles);
        for (int i = 0; i < numParticles; i++) {
            particleChargesInRun[i] = lambdaA_charge * particleCoulombParams[i][0] + lambdaB_charge * particleCoulombParams[i][1] + lambdaC_charge * particleCoulombParams[i][2];       
            particleSigmaEpsilonParamsInRun[i] = particleSigmaEpsilonParams[i];

            bool isDummy = particleTypeParams[i][0];
            bool isREST2 = particleTypeParams[i][1];
            if (isREST2){
                particleChargesInRun[i] *= sqrt_rest2_scale;
                particleSigmaEpsilonParamsInRun[i][1] *= sqrt_rest2_scale;
                particleSigmaEpsilonParamsInRun[i][3] *= sqrt_rest2_scale;
            }
        }
    }
    
}


void ReferenceCalcPerturbNonbondedForceKernel::getPerturbDispersionCorrections(ContextImpl& context) {
    if (nonbondedMethod == PerturbNonbondedForce::NoCutoff || nonbondedMethod == PerturbNonbondedForce::CutoffNonPeriodic){
        EcorrA = 0.0;
        EcorrB = 0.0;
        return;
    }

    if (!useDispersionCorrection)
        throw OpenMMException("PerturbNonbondedForce: Dispersion correction requested, but not supported.");
    
    // Record sigma and epsilon for every particle, including the default value
    // for every offset parameter.
    vector<double> sigmaA(numParticles), epsilonA(numParticles);
    vector<double> sigmaB(numParticles), epsilonB(numParticles);
    for (int i = 0; i < numParticles; i++) {
        sigmaA[i] = perturbSigmaEpsilonParams_raw[i][0];
        epsilonA[i] = perturbSigmaEpsilonParams_raw[i][1];
        sigmaB[i] = perturbSigmaEpsilonParams_raw[i][2];
        epsilonB[i] = perturbSigmaEpsilonParams_raw[i][3];
        
        bool isREST2 = particleTypeParams[i][1];
        
        if (isREST2){
            epsilonA[i] *= rest2_scale;
            epsilonB[i] *= rest2_scale;
        }
    }

    EcorrA = this->calcDispersionCorrection(sigmaA, epsilonA);
    EcorrB = this->calcDispersionCorrection(sigmaB, epsilonB);
    return;
}

double ReferenceCalcPerturbNonbondedForceKernel::calcDispersionCorrection(const vector<double>& sigmaList, const vector<double>& epsilonList) {
    // Identify all particle classes (defined by sigma and epsilon), and count the number of
    // particles in each class.

    map<pair<double, double>, int> classCounts;
    for (int i = 0; i < numParticles; i++) {
        pair<double, double> key = make_pair(sigmaList[i], epsilonList[i]);
        map<pair<double, double>, int>::iterator entry = classCounts.find(key);
        if (entry == classCounts.end())
            classCounts[key] = 1;
        else
            entry->second++;
    }

    // Loop over all pairs of classes to compute the coefficient.

    double sum1 = 0, sum2 = 0, sum3 = 0;
    for (map<pair<double, double>, int>::const_iterator entry = classCounts.begin(); entry != classCounts.end(); ++entry) {
        double sigma = entry->first.first;
        double epsilon = entry->first.second;
        double count = (double) entry->second;
        count *= (count + 1) / 2;
        double sigma2 = sigma*sigma;
        double sigma6 = sigma2*sigma2*sigma2;
        sum1 += count*epsilon*sigma6*sigma6;
        sum2 += count*epsilon*sigma6;
        if (useSwitchingFunction)
            sum3 += count*epsilon*(evalIntegral(nonbondedCutoff, switchingDistance, nonbondedCutoff, sigma)-evalIntegral(switchingDistance, switchingDistance, nonbondedCutoff, sigma));
    }
    for (map<pair<double, double>, int>::const_iterator class1 = classCounts.begin(); class1 != classCounts.end(); ++class1)
        for (map<pair<double, double>, int>::const_iterator class2 = classCounts.begin(); class2 != class1; ++class2) {
            double sigma = 0.5*(class1->first.first+class2->first.first);
            double epsilon = sqrt(class1->first.second*class2->first.second);
            double count = (double) class1->second;
            count *= (double) class2->second;
            double sigma2 = sigma*sigma;
            double sigma6 = sigma2*sigma2*sigma2;
            sum1 += count*epsilon*sigma6*sigma6;
            sum2 += count*epsilon*sigma6;
            if (useSwitchingFunction)
                sum3 += count*epsilon*(evalIntegral(nonbondedCutoff, switchingDistance, nonbondedCutoff, sigma)-evalIntegral(switchingDistance, switchingDistance, nonbondedCutoff, sigma));
        }

    double numInteractions = (numParticles*(numParticles+1))/2;
    sum1 /= numInteractions;
    sum2 /= numInteractions;
    sum3 /= numInteractions;
    return 8*numParticles*numParticles*M_PI*(sum1/(9*pow(nonbondedCutoff, 9))-sum2/(3*pow(nonbondedCutoff, 3))+sum3);
}
