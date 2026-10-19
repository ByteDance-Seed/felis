/* -------------------------------------------------------------------------- *

 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "openmm/Context.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/common/ComputeVectorTypes.h"
#include "openmm/common/CommonKernelUtilities.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/ComputeProgram.h"
#include "openmm/common/ComputeKernel.h"
#include "openmm/common/ComputeEvent.h"
#include "openmm/cuda/CudaNonbondedUtilities.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iterator>
#include <set>
#include "CommonKernelSources.h"
#include "CudaKernelSources.h"
#include "CudaPerturbNonbondedKernelSources.h"
#include "CudaPerturbNonbondedKernels.h"
#include "internal/PerturbNonbondedForceImpl.h"


using namespace OpenMM;
using namespace std;
using namespace PerturbNonbondedPlugin;

static void validateContextPrecisionPositionCorrection(const CudaContext& cu,
        CudaArrayDevicePointer correction, const char* label) {
    if (cu.getUseMixedPrecision() && correction == 0)
        throw OpenMMException(string(label)+" requires a float4 correction array in mixed precision.");
    if (!cu.getUseMixedPrecision() && correction != 0)
        throw OpenMMException(string(label)+" requires a zero correction pointer in single or double precision.");
}

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

#define CHECK_RESULT(result, prefix) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        m<<prefix<<": "<<CudaContext::getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }

static void getCudaPmeParameters(CudaContext& cu, bool& usePmeQueue, bool& useFixedPointChargeSpreading) {
    usePmeQueue = (!cu.getPlatformData().disablePmeStream && !cu.getPlatformData().useCpuPme);
    useFixedPointChargeSpreading = cu.getUseDoublePrecision() || cu.getPlatformData().deterministicForces;
}

class CudaCalcPerturbNonbondedForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const PerturbNonbondedForce& force) : force(force) {
        particleOffset.resize(force.getNumParticles(), -1);
        exceptionOffset.resize(force.getNumExceptions(), -1);
        for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
            string parameter;
            int particleIndex;
            double chargeScale, sigmaScale, epsilonScale;
            force.getParticleParameterOffset(i, parameter, particleIndex, chargeScale, sigmaScale, epsilonScale);
            particleOffset[particleIndex] = i;
        }
        for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
            string parameter;
            int exceptionIndex;
            double chargeProdScale, sigmaScale, epsilonScale;
            force.getExceptionParameterOffset(i, parameter, exceptionIndex, chargeProdScale, sigmaScale, epsilonScale);
            exceptionOffset[exceptionIndex] = i;
        }
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double chargeA_1, sigmaA_1, epsilonA_1, chargeB_1, sigmaB_1, epsilonB_1;
        double chargeA_2, sigmaA_2, epsilonA_2, chargeB_2, sigmaB_2, epsilonB_2;
        bool isDummy_1, isREST2_1;
        bool isDummy_2, isREST2_2;

        force.getParticleParameters(particle1, chargeA_1, sigmaA_1, epsilonA_1, chargeB_1, sigmaB_1, epsilonB_1, isDummy_1, isREST2_1);
        force.getParticleParameters(particle2, chargeA_2, sigmaA_2, epsilonA_2, chargeB_2, sigmaB_2, epsilonB_2, isDummy_2, isREST2_2);
        return (chargeA_1 == chargeA_2 && sigmaA_1 == sigmaA_2 && epsilonA_1 == epsilonA_2 && chargeB_1 == chargeB_2 && sigmaB_1 == sigmaB_2 && epsilonB_1 == epsilonB_2 && isDummy_1 == isDummy_2 && isREST2_1 == isREST2_2);
    }
    int getNumParticleGroups() {
        return force.getNumExceptions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(index, particle1, particle2, chargeProd, sigma, epsilon);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        if (exceptionOffset[group1] != -1 || exceptionOffset[group2] != -1) {
            if (exceptionOffset[group1] == -1 || exceptionOffset[group2] == -1)
                return false;
            string parameter1, parameter2;
            int exceptionIndex1, exceptionIndex2;
            double chargeProdScale1, sigmaScale1, epsilonScale1, chargeProdScale2, sigmaScale2, epsilonScale2;
            force.getExceptionParameterOffset(exceptionOffset[group1], parameter1, exceptionIndex1, chargeProdScale1, sigmaScale1, epsilonScale1);
            force.getExceptionParameterOffset(exceptionOffset[group2], parameter2, exceptionIndex2, chargeProdScale2, sigmaScale2, epsilonScale2);
            if (parameter1 != parameter2 || chargeProdScale1 != chargeProdScale2 || sigmaScale1 != sigmaScale2 || epsilonScale1 != epsilonScale2)
                return false;
        }
        int particle1, particle2;
        double chargeProd1, chargeProd2, sigma1, sigma2, epsilon1, epsilon2;
        force.getExceptionParameters(group1, particle1, particle2, chargeProd1, sigma1, epsilon1);
        force.getExceptionParameters(group2, particle1, particle2, chargeProd2, sigma2, epsilon2);
        return (chargeProd1 == chargeProd2 && sigma1 == sigma2 && epsilon1 == epsilon2);
    }
private:
    const PerturbNonbondedForce& force;
    vector<int> particleOffset, exceptionOffset;
};

class CudaCalcPerturbNonbondedForceKernel::PmeIO : public CalcPmeReciprocalForceKernel::IO {
public:
    PmeIO(CudaContext& cu, ComputeKernel addForcesKernel) : cu(cu), addForcesKernel(addForcesKernel) {
        forceTemp.initialize<mm_float4>(cu, cu.getNumAtoms(), "PmeForce");
        addForcesKernel->addArg(forceTemp);
        addForcesKernel->addArg();
    }
    float* getPosq() {
        ContextSelector selector(cu);
        cu.getPosq().download(posq);
        return (float*) &posq[0];
    }
    void setForce(float* force) {
        forceTemp.upload(force);
        addForcesKernel->setArg(1, cu.getLongForceBuffer());
        addForcesKernel->execute(cu.getNumAtoms());
        //addForcesKernel.setArg(1, cu.getLongForceBuffer());
        //addForcesKernel.execute(cu.getNumAtoms());
    }
    void setChargeDerivatives(float* chargeDerivatives) {
    }
private:
    CudaContext& cu;
    vector<mm_float4> posq;
    CudaArray forceTemp;
    ComputeKernel addForcesKernel;
};

class CudaCalcPerturbNonbondedForceKernel::PmePreComputation : public CudaContext::ForcePreComputation {
public:
    PmePreComputation(CudaContext& cu, Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : cu(cu), pme(pme), io(io) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        Vec3 boxVectors[3];
        cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        pme.getAs<CalcPmeReciprocalForceKernel>().beginComputation(io, boxVectors, includeEnergy, true, false);
    }
private:
    CudaContext& cu;
    Kernel pme;
    CalcPmeReciprocalForceKernel::IO& io;
};

class CudaCalcPerturbNonbondedForceKernel::PmePostComputation : public CudaContext::ForcePostComputation {
public:
    PmePostComputation(Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : pme(pme), io(io) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        return pme.getAs<CalcPmeReciprocalForceKernel>().finishComputation(io);
    }
private:
    Kernel pme;
    CalcPmeReciprocalForceKernel::IO& io;
};

class CudaCalcPerturbNonbondedForceKernel::SyncQueuePreComputation : public CudaContext::ForcePreComputation {
public:
    SyncQueuePreComputation(CudaContext& cu, ComputeQueue queue, ComputeEvent event, int forceGroup) : cu(cu), queue(queue), event(event), forceGroup(forceGroup) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            event->enqueue();
            event->queueWait(queue);
        }
    }
private:
    CudaContext& cu;
    ComputeQueue queue;
    ComputeEvent event;
    int forceGroup;
};

class CudaCalcPerturbNonbondedForceKernel::SyncQueuePostComputation : public CudaContext::ForcePostComputation {
public:
    SyncQueuePostComputation(CudaContext& cu, ComputeEvent event, CudaArray& pmeEnergyBuffer, int forceGroup) : cu(cu), event(event),
            pmeEnergyBuffer(pmeEnergyBuffer), forceGroup(forceGroup) {
    }
    void setKernel(ComputeKernel kernel) {
        addEnergyKernel = kernel;
        addEnergyKernel->addArg(pmeEnergyBuffer);
        addEnergyKernel->addArg(cu.getEnergyBuffer());
        addEnergyKernel->addArg((int) pmeEnergyBuffer.getSize());
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            event->wait();
            if (includeEnergy)
                addEnergyKernel->execute(pmeEnergyBuffer.getSize());
        }
        return 0.0;
    }
private:
    CudaContext& cu;
    ComputeEvent event;
    ComputeKernel addEnergyKernel;
    CudaArray& pmeEnergyBuffer;
    int forceGroup;
};

CudaCalcPerturbNonbondedForceKernel::~CudaCalcPerturbNonbondedForceKernel() {
    ContextSelector selector(cu);
    if (pmeio != NULL)
        delete pmeio;
}

void CudaCalcPerturbNonbondedForceKernel::initialize(const System& system, const PerturbNonbondedForce& force) {
    if (cu.getNumContexts() != 1)
        throw OpenMMException("PerturbNonbonded CUDA particle queries require a single CUDA device.");
    bool usePmeQueue, useFixedPointChargeSpreading;
    getCudaPmeParameters(cu, usePmeQueue, useFixedPointChargeSpreading);
    commonInitialize(system, force, usePmeQueue, false, useFixedPointChargeSpreading, cu.getPlatformData().useCpuPme);
}

void CudaCalcPerturbNonbondedForceKernel::commonInitialize(const System& system, const PerturbNonbondedForce& force, bool usePmeQueue,
        bool deviceIsCpu, bool useFixedPointChargeSpreading, bool useCpuPme) {
    // This becomes true below only when a GPU PME grid, queue, and sync events
    // have all been created.  Non-PME contexts must not touch empty events.
    this->usePmeQueue = false;
    scratchPmeSpreadKernel = NULL;
    reduceCandidateSelfEnergyKernel = NULL;
    this->deviceIsCpu = deviceIsCpu;
    this->useFixedPointChargeSpreading = useFixedPointChargeSpreading;
    this->useCpuPme = useCpuPme;
    ContextSelector selector(cu);
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "nonbonded"+cu.intToString(forceIndex)+"_";

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionsWithOffsets.insert(exception);
    }
    vector<pair<int, int> > exclusions;
    vector<int> exceptions;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        exclusions.push_back(pair<int, int>(particle1, particle2));
        if (chargeProd != 0.0 || epsilon != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end()) {
            exceptionIndex[i] = exceptions.size();
            exceptions.push_back(i);
        }
    }

    // Initialize nonbonded interactions.

    numParticles = force.getNumParticles();
    vector<mm_float4> baseParticleParamVec(cu.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
    
    vector<mm_float4> perturbChargeParamVec(cu.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
    vector<mm_float4> perturbSigmaEpsilonParamVec(cu.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
    vector<int> ljAtomTypesVec(cu.getPaddedNumAtoms(), 0);

    perturbChargesParams_cpu.resize(cu.getPaddedNumAtoms());
    perturbSigmaEpsilonParams_cpu.resize(cu.getPaddedNumAtoms());
    perturbAtomTypesParams_cpu.resize(cu.getPaddedNumAtoms());
    
    vector<vector<int> > exclusionList(numParticles);
    hasCoulomb = false;
    hasLJ = false;
    for (int i = 0; i < numParticles; i++) {
        double chargeA, sigmaA, epsilonA;
        double chargeB, sigmaB, epsilonB;
        bool isDummy = false;
        bool isREST2 = false;

        force.getParticleParameters(i, chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2);
        
        ljAtomTypesVec[i] = isDummy ? 1 : 0;
        double chargeC = 0.0;
        float rest2_sign = isREST2 ? 1.0 : 0.0;
        if (isDummy){
            chargeC = 0;
        } else {
            chargeC = (chargeA + chargeB) * 0.5;
        }
        perturbChargeParamVec[i] = mm_float4(chargeA, chargeB, chargeC, rest2_sign);
        perturbSigmaEpsilonParamVec[i] = mm_float4(0.5 * sigmaA, 2 * sqrt(epsilonA), 0.5 * sigmaB, 2 * sqrt(epsilonB));

        perturbChargesParams_cpu[i] = {chargeA, chargeB, chargeC};
        perturbSigmaEpsilonParams_cpu[i] = {sigmaA, epsilonA, sigmaB, epsilonB};
        perturbAtomTypesParams_cpu[i] = {isDummy, isREST2};

        exclusionList[i].push_back(i);
        if (chargeA != 0.0)
            hasCoulomb = true;
        if (epsilonA != 0.0)
            hasLJ = true;
    }
    numParticles = force.getNumParticles();
    useSwitchingFunction = force.getUseSwitchingFunction();
    cutoff = force.getCutoffDistance();
    switchDist = force.getSwitchingDistance();
    enableCacheEnergyConstants = force.getEnableEnergyConstantCache();

    for (auto exclusion : exclusions) {
        exclusionList[exclusion.first].push_back(exclusion.second);
        exclusionList[exclusion.second].push_back(exclusion.first);
    }
    nonbondedMethod = CalcPerturbNonbondedForceKernel::PerturbNonbondedMethod(force.getNonbondedMethod());
    bool useCutoff = (nonbondedMethod != NoCutoff);
    bool usePeriodic = (nonbondedMethod != NoCutoff && nonbondedMethod != CutoffNonPeriodic);
    doLJPME = (nonbondedMethod == LJPME && hasLJ);
    usePosqCharges = hasCoulomb ? cu.requestPosqCharges() : false;
    map<string, string> defines;
    defines["HAS_COULOMB"] = (hasCoulomb ? "1" : "0");
    defines["HAS_LENNARD_JONES"] = (hasLJ ? "1" : "0");
    defines["USE_LJ_SWITCH"] = (useCutoff && force.getUseSwitchingFunction() ? "1" : "0");

    usePerturbDispersionCorrection = force.getUseDispersionCorrection() && hasLJ && usePeriodic;

    if (useCutoff) {
        // Compute the reaction field constants.

        double reactionFieldK = pow(force.getCutoffDistance(), -3.0)*(force.getReactionFieldDielectric()-1.0)/(2.0*force.getReactionFieldDielectric()+1.0);
        double reactionFieldC = (1.0 / force.getCutoffDistance())*(3.0*force.getReactionFieldDielectric())/(2.0*force.getReactionFieldDielectric()+1.0);
        defines["REACTION_FIELD_K"] = cu.doubleToString(reactionFieldK);
        defines["REACTION_FIELD_C"] = cu.doubleToString(reactionFieldC);

        // Compute the switching coefficients.

        if (force.getUseSwitchingFunction()) {
            defines["LJ_SWITCH_CUTOFF"] = cu.doubleToString(force.getSwitchingDistance());
            defines["LJ_SWITCH_C3"] = cu.doubleToString(10/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 3.0));
            defines["LJ_SWITCH_C4"] = cu.doubleToString(15/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 4.0));
            defines["LJ_SWITCH_C5"] = cu.doubleToString(6/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 5.0));
        }
    }
    //if (force.getUseDispersionCorrection() && cu.getContextIndex() == 0 && !doLJPME)
    //    dispersionCoefficient = PerturbNonbondedForceImpl::calcDispersionCorrection(system, force);
    //else
    dispersionCoefficient = 0.0;
    alpha = 0;
    ewaldSelfEnergy = 0.0;
    totalCharge = 0.0;
    map<string, string> paramsDefines;
    paramsDefines["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
    paramsDefines["EPSILON0"] = cu.doubleToString(EPSILON0);
    paramsDefines["WORK_GROUP_SIZE"] = cu.intToString(cu.ThreadBlockSize);
    paramsDefines["CHARGE_BUFFER_SIZE"] = cu.intToString(cu.getNumThreadBlocks());
    hasOffsets = (force.getNumParticleParameterOffsets() > 0 || force.getNumExceptionParameterOffsets() > 0);
    if (hasOffsets)
        paramsDefines["HAS_OFFSETS"] = "1";
    if (force.getNumParticleParameterOffsets() > 0)
        paramsDefines["HAS_PARTICLE_OFFSETS"] = "1";
    if (force.getNumExceptionParameterOffsets() > 0)
        paramsDefines["HAS_EXCEPTION_OFFSETS"] = "1";
    if (usePosqCharges)
        paramsDefines["USE_POSQ_CHARGES"] = "1";
    if (doLJPME)
        paramsDefines["INCLUDE_LJPME_EXCEPTIONS"] = "1";
    if (nonbondedMethod == Ewald) {
        // Compute the Ewald parameters.

        int kmaxx, kmaxy, kmaxz;
        PerturbNonbondedForceImpl::calcEwaldParameters(system, force, alpha, kmaxx, kmaxy, kmaxz);
        defines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        if (cu.getContextIndex() == 0) {
            paramsDefines["INCLUDE_EWALD"] = "1";
            paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cu.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
            for (int i = 0; i < numParticles; i++) {
                ewaldSelfEnergy -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
                totalCharge += baseParticleParamVec[i].x;
            }

            // Create the reciprocal space kernels.

            map<string, string> replacements;
            replacements["NUM_ATOMS"] = cu.intToString(numParticles);
            replacements["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
            replacements["KMAX_X"] = cu.intToString(kmaxx);
            replacements["KMAX_Y"] = cu.intToString(kmaxy);
            replacements["KMAX_Z"] = cu.intToString(kmaxz);
            replacements["EXP_COEFFICIENT"] = cu.doubleToString(-1.0/(4.0*alpha*alpha));
            replacements["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
            replacements["M_PI"] = cu.doubleToString(M_PI);
            ComputeProgram program = cu.compileProgram(CommonKernelSources::ewald, replacements);
            ewaldSumsKernel = program->createKernel("calculateEwaldCosSinSums");
            ewaldForcesKernel = program->createKernel("calculateEwaldForces");
            int elementSize = (cu.getUseDoublePrecision() ? sizeof(mm_double2) : sizeof(mm_float2));
            cosSinSums.initialize(cu, (2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1), elementSize, "cosSinSums");
        }
    }
    else if (((nonbondedMethod == PME || nonbondedMethod == LJPME) && hasCoulomb) || doLJPME) {
        // Compute the PME parameters.

        PerturbNonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSizeX, gridSizeY, gridSizeZ, false);
        gridSizeX = cu.findLegalFFTDimension(gridSizeX);
        gridSizeY = cu.findLegalFFTDimension(gridSizeY);
        gridSizeZ = cu.findLegalFFTDimension(gridSizeZ);
        if (doLJPME) {
            PerturbNonbondedForceImpl::calcPMEParameters(system, force, dispersionAlpha, dispersionGridSizeX,
                                                  dispersionGridSizeY, dispersionGridSizeZ, true);
            dispersionGridSizeX = cu.findLegalFFTDimension(dispersionGridSizeX);
            dispersionGridSizeY = cu.findLegalFFTDimension(dispersionGridSizeY);
            dispersionGridSizeZ = cu.findLegalFFTDimension(dispersionGridSizeZ);
        }
        defines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        defines["DO_LJPME"] = doLJPME ? "1" : "0";
        if (doLJPME) {
            defines["EWALD_DISPERSION_ALPHA"] = cu.doubleToString(dispersionAlpha);
            double invRCut6 = pow(force.getCutoffDistance(), -6);
            double dalphaR = dispersionAlpha * force.getCutoffDistance();
            double dar2 = dalphaR*dalphaR;
            double dar4 = dar2*dar2;
            double multShift6 = -invRCut6*(1.0 - exp(-dar2) * (1.0 + dar2 + 0.5*dar4));
            defines["INVCUT6"] = cu.doubleToString(invRCut6);
            defines["MULTSHIFT6"] = cu.doubleToString(multShift6);
        }
        if (cu.getContextIndex() == 0) {
            paramsDefines["INCLUDE_EWALD"] = "1";
            paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cu.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
            for (int i = 0; i < numParticles; i++) {
                ewaldSelfEnergy -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
                totalCharge += baseParticleParamVec[i].x;
            }
            if (doLJPME) {
                paramsDefines["INCLUDE_LJPME"] = "1";
                paramsDefines["LJPME_SELF_ENERGY_SCALE"] = cu.doubleToString(pow(dispersionAlpha, 6)/3.0);
                for (int i = 0; i < numParticles; i++)
                    ewaldSelfEnergy += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
            }
            pmeDefines["PME_ORDER"] = cu.intToString(PmeOrder);
            pmeDefines["NUM_ATOMS"] = cu.intToString(numParticles);
            pmeDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
            pmeDefines["NUM_INDICES"] = "0";
            pmeDefines["RECIP_EXP_FACTOR"] = cu.doubleToString(M_PI*M_PI/(alpha*alpha));
            pmeDefines["GRID_SIZE_X"] = cu.intToString(gridSizeX);
            pmeDefines["GRID_SIZE_Y"] = cu.intToString(gridSizeY);
            pmeDefines["GRID_SIZE_Z"] = cu.intToString(gridSizeZ);
            pmeDefines["EPSILON_FACTOR"] = cu.doubleToString(sqrt(ONE_4PI_EPS0));
            pmeDefines["M_PI"] = cu.doubleToString(M_PI);
            if (useFixedPointChargeSpreading)
                pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";
            if (deviceIsCpu)
                pmeDefines["DEVICE_IS_CPU"] = "1";
            if (useCpuPme && !doLJPME && usePosqCharges) {
                // Create the CPU PME kernel.

                try {
                    cpuPme = getPlatform().createKernel(CalcPmeReciprocalForceKernel::Name(), *cu.getContextImpl());
                    cpuPme.getAs<CalcPmeReciprocalForceKernel>().initialize(gridSizeX, gridSizeY, gridSizeZ, numParticles, {}, alpha, false);
                    ComputeProgram program = cu.compileProgram(CommonKernelSources::pme, pmeDefines);
                    ComputeKernel addForcesKernel = program->createKernel("addForces");
                    pmeio = new PmeIO(cu, addForcesKernel);
                    cu.addPreComputation(new PmePreComputation(cu, cpuPme, *pmeio));
                    cu.addPostComputation(new PmePostComputation(cpuPme, *pmeio));
                }
                catch (OpenMMException& ex) {
                    // The CPU PME plugin isn't available.
                }
            }
            if (pmeio == NULL) {
                // Create required data structures.

                int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
                int gridElements = gridSizeX*gridSizeY*gridSizeZ;
                if (doLJPME) {
                    gridElements = max(gridElements, dispersionGridSizeX*dispersionGridSizeY*dispersionGridSizeZ);
                }
                pmeGrid1.initialize(cu, gridElements, 2*elementSize, "pmeGrid1");
                pmeGrid2.initialize(cu, gridElements, 2*elementSize, "pmeGrid2");
                scratchPmeGrid1.initialize(cu, gridElements, 2*elementSize, "scratchPmeGrid1");
                scratchPmeGrid2.initialize(cu, gridElements, 2*elementSize, "scratchPmeGrid2");
                scratchPmePotentials.initialize<float>(cu, cu.getNumAtoms(), "scratchPmePotentials");
                if (useFixedPointChargeSpreading)
                    cu.addAutoclearBuffer(pmeGrid2);
                else
                    cu.addAutoclearBuffer(pmeGrid1);
                pmeBsplineModuliX.initialize(cu, gridSizeX, elementSize, "pmeBsplineModuliX");
                pmeBsplineModuliY.initialize(cu, gridSizeY, elementSize, "pmeBsplineModuliY");
                pmeBsplineModuliZ.initialize(cu, gridSizeZ, elementSize, "pmeBsplineModuliZ");
                if (doLJPME) {
                    pmeDispersionBsplineModuliX.initialize(cu, dispersionGridSizeX, elementSize, "pmeDispersionBsplineModuliX");
                    pmeDispersionBsplineModuliY.initialize(cu, dispersionGridSizeY, elementSize, "pmeDispersionBsplineModuliY");
                    pmeDispersionBsplineModuliZ.initialize(cu, dispersionGridSizeZ, elementSize, "pmeDispersionBsplineModuliZ");
                }
                pmeAtomGridIndex.initialize<mm_int2>(cu, numParticles, "pmeAtomGridIndex");
                int energyElementSize = (cu.getUseDoublePrecision() || cu.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
                pmeEnergyBuffer.initialize(cu, cu.getNumThreadBlocks()*CudaContext::ThreadBlockSize, energyElementSize, "pmeEnergyBuffer");
                cu.clearBuffer(pmeEnergyBuffer);
                sort = cu.createSort(new SortTrait(), cu.getNumAtoms());
                fft = cu.createFFT(gridSizeX, gridSizeY, gridSizeZ, true);
                if (doLJPME)
                    dispersionFft = cu.createFFT(dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, true);
                this->usePmeQueue = usePmeQueue;
                if (usePmeQueue) {
                    pmeDefines["USE_PME_STREAM"] = "1";
                    pmeQueue = cu.createQueue();
                    int recipForceGroup = force.getReciprocalSpaceForceGroup();
                    if (recipForceGroup < 0)
                        recipForceGroup = force.getForceGroup();
                    pmeSyncEvent = cu.createEvent();
                    paramsSyncEvent = cu.createEvent();
                    cu.addPreComputation(new SyncQueuePreComputation(cu, pmeQueue, pmeSyncEvent, recipForceGroup));
                    cu.addPostComputation(syncQueue = new SyncQueuePostComputation(cu, pmeSyncEvent, pmeEnergyBuffer, recipForceGroup));
                }

                // Initialize the b-spline moduli.

                for (int grid = 0; grid < 2; grid++) {
                    int xsize, ysize, zsize;
                    CudaArray *xmoduli, *ymoduli, *zmoduli;
                    if (grid == 0) {
                        xsize = gridSizeX;
                        ysize = gridSizeY;
                        zsize = gridSizeZ;
                        xmoduli = &pmeBsplineModuliX;
                        ymoduli = &pmeBsplineModuliY;
                        zmoduli = &pmeBsplineModuliZ;
                    }
                    else {
                        if (!doLJPME)
                            continue;
                        xsize = dispersionGridSizeX;
                        ysize = dispersionGridSizeY;
                        zsize = dispersionGridSizeZ;
                        xmoduli = &pmeDispersionBsplineModuliX;
                        ymoduli = &pmeDispersionBsplineModuliY;
                        zmoduli = &pmeDispersionBsplineModuliZ;
                    }
                    int maxSize = max(max(xsize, ysize), zsize);
                    vector<double> data(PmeOrder);
                    vector<double> ddata(PmeOrder);
                    vector<double> bsplines_data(maxSize);
                    data[PmeOrder-1] = 0.0;
                    data[1] = 0.0;
                    data[0] = 1.0;
                    for (int i = 3; i < PmeOrder; i++) {
                        double div = 1.0/(i-1.0);
                        data[i-1] = 0.0;
                        for (int j = 1; j < (i-1); j++)
                            data[i-j-1] = div*(j*data[i-j-2]+(i-j)*data[i-j-1]);
                        data[0] = div*data[0];
                    }

                    // Differentiate.

                    ddata[0] = -data[0];
                    for (int i = 1; i < PmeOrder; i++)
                        ddata[i] = data[i-1]-data[i];
                    double div = 1.0/(PmeOrder-1);
                    data[PmeOrder-1] = 0.0;
                    for (int i = 1; i < (PmeOrder-1); i++)
                        data[PmeOrder-i-1] = div*(i*data[PmeOrder-i-2]+(PmeOrder-i)*data[PmeOrder-i-1]);
                    data[0] = div*data[0];
                    for (int i = 0; i < maxSize; i++)
                        bsplines_data[i] = 0.0;
                    for (int i = 1; i <= PmeOrder; i++)
                        bsplines_data[i] = data[i-1];

                    // Evaluate the actual bspline moduli for X/Y/Z.

                    for (int dim = 0; dim < 3; dim++) {
                        int ndata = (dim == 0 ? xsize : dim == 1 ? ysize : zsize);
                        vector<double> moduli(ndata);
                        for (int i = 0; i < ndata; i++) {
                            double sc = 0.0;
                            double ss = 0.0;
                            for (int j = 0; j < ndata; j++) {
                                double arg = (2.0*M_PI*i*j)/ndata;
                                sc += bsplines_data[j]*cos(arg);
                                ss += bsplines_data[j]*sin(arg);
                            }
                            moduli[i] = sc*sc+ss*ss;
                        }
                        for (int i = 0; i < ndata; i++)
                            if (moduli[i] < 1.0e-7)
                                moduli[i] = (moduli[(i-1+ndata)%ndata]+moduli[(i+1)%ndata])*0.5;
                        if (dim == 0)
                            xmoduli->upload(moduli, true);
                        else if (dim == 1)
                            ymoduli->upload(moduli, true);
                        else
                            zmoduli->upload(moduli, true);
                    }
                }
            }
        }
    }

    // Add code to subtract off the reciprocal part of excluded interactions.

    if ((nonbondedMethod == Ewald || nonbondedMethod == PME || nonbondedMethod == LJPME) && pmeio == NULL) {
        int numContexts = cu.getNumContexts();
        int startIndex = cu.getContextIndex()*force.getNumExceptions()/numContexts;
        int endIndex = (cu.getContextIndex()+1)*force.getNumExceptions()/numContexts;
        int numExclusions = endIndex-startIndex;
        if (numExclusions > 0) {
            paramsDefines["HAS_EXCLUSIONS"] = "1";
            vector<vector<int> > atoms(numExclusions, vector<int>(2));
            exclusionAtoms.initialize<mm_int2>(cu, numExclusions, "exclusionAtoms");
            exclusionParams.initialize<mm_float4>(cu, numExclusions, "exclusionParams");
            vector<mm_int2> exclusionAtomsVec(numExclusions);
            for (int i = 0; i < numExclusions; i++) {
                int j = i+startIndex;
                exclusionAtomsVec[i] = mm_int2(exclusions[j].first, exclusions[j].second);
                atoms[i][0] = exclusions[j].first;
                atoms[i][1] = exclusions[j].second;
            }
            exclusionAtoms.upload(exclusionAtomsVec);
            map<string, string> replacements;
            replacements["PARAMS"] = cu.getBondedUtilities().addArgument(exclusionParams, "float4");
            replacements["EWALD_ALPHA"] = cu.doubleToString(alpha);
            replacements["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
            replacements["DO_LJPME"] = doLJPME ? "1" : "0";
            replacements["USE_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
            if (doLJPME)
                replacements["EWALD_DISPERSION_ALPHA"] = cu.doubleToString(dispersionAlpha);
            if (force.getIncludeDirectSpace())
                cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CommonKernelSources::pmeExclusions, replacements), force.getForceGroup());
        }
    }

    // Add the interaction to the default nonbonded kernel.

    //string source = cu.replaceStrings(CommonKernelSources::coulombLennardJones, defines);

    string perturb_coulomb_source = cu.replaceStrings(CudaPerturbNonbondedKernelSources::perturbCoulomb, defines);
    string perturb_lj_source = cu.replaceStrings(CudaPerturbNonbondedKernelSources::perturbLennardJones, defines);
    string perturb_direct_source = cu.replaceStrings(CudaPerturbNonbondedKernelSources::perturbDirectInteractions, defines);

    charges.initialize(cu, cu.getPaddedNumAtoms(), cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "charges");
    baseParticleParams.initialize<mm_float4>(cu, cu.getPaddedNumAtoms(), "baseParticleParams");
    baseParticleParams.upload(baseParticleParamVec);

    perturbChargeParams.initialize(cu, cu.getPaddedNumAtoms(), sizeof(mm_float4), "perturbChargeParam");
    perturbChargeParams.upload(perturbChargeParamVec);
    perturbSigmaEpsilonParams.initialize(cu, cu.getPaddedNumAtoms(), sizeof(mm_float4), "perturbSigmaEpsilonParam");
    perturbSigmaEpsilonParams.upload(perturbSigmaEpsilonParamVec);
    ljAtomTypes.initialize(cu, cu.getPaddedNumAtoms(), sizeof(int), "ljAtomType");
    ljAtomTypes.upload(ljAtomTypesVec);

    sigmaEpsilonA.initialize(cu, cu.getPaddedNumAtoms(), sizeof(mm_float2), "AsigmaEpsilon");
    sigmaEpsilonB.initialize(cu, cu.getPaddedNumAtoms(), sizeof(mm_float2), "BsigmaEpsilon");
    ljLambdaAB.initialize(cu, cu.getPaddedNumAtoms(), sizeof(mm_float2), "ljLambda");


    map<string, string> replacements;
    replacements["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
    if (usePosqCharges) {
        replacements["CHARGE1"] = "posq1.w";
        replacements["CHARGE2"] = "posq2.w";
    }
    else {
        replacements["CHARGE1"] = prefix+"charge1";
        replacements["CHARGE2"] = prefix+"charge2";
    }
    if (hasCoulomb && !usePosqCharges)
        cu.getNonbondedUtilities().addParameter(ComputeParameterInfo(charges, prefix+"charge", "real", 1));
    sigmaEpsilon.initialize<mm_float2>(cu, cu.getPaddedNumAtoms(), "sigmaEpsilon");
    if (hasLJ) {
        //replacements["SIGMA_EPSILON1"] = prefix+"sigmaEpsilon1";
        //replacements["SIGMA_EPSILON2"] = prefix+"sigmaEpsilon2";
        //cu.getNonbondedUtilities().addParameter(ComputeParameterInfo(sigmaEpsilonA, prefix+"AsigmaEpsilonA", "float", 2));
        replacements["A_SIGMA_EPSILON1"] = prefix+"AsigmaEpsilon1";
        replacements["A_SIGMA_EPSILON2"] = prefix+"AsigmaEpsilon2";
        replacements["B_SIGMA_EPSILON1"] = prefix+"BsigmaEpsilon1";
        replacements["B_SIGMA_EPSILON2"] = prefix+"BsigmaEpsilon2";
        replacements["LJ_ATOM_TYPE1"] = prefix+"ljAtomType1";
        replacements["LJ_ATOM_TYPE2"] = prefix+"ljAtomType2";
        replacements["LJ_LAMBDA1"] = prefix+"ljLambda1";
        replacements["LJ_LAMBDA2"] = prefix+"ljLambda2";
        replacements["SOFTCORE_ALPHA"] = cu.doubleToString(force.getSoftcoreAlpha());

        cu.getNonbondedUtilities().addParameter(ComputeParameterInfo(sigmaEpsilonA, prefix+"AsigmaEpsilon", "float", 2, false));
        cu.getNonbondedUtilities().addParameter(ComputeParameterInfo(sigmaEpsilonB, prefix+"BsigmaEpsilon", "float", 2, false));
        cu.getNonbondedUtilities().addParameter(ComputeParameterInfo(ljAtomTypes, prefix+"ljAtomType", "int", 1, true));
        cu.getNonbondedUtilities().addParameter(ComputeParameterInfo(ljLambdaAB, prefix+"ljLambda", "float", 2, false));
    }
    //source = cu.replaceStrings(source, replacements);

    perturb_coulomb_source = cu.replaceStrings(perturb_coulomb_source, replacements);
    perturb_lj_source = cu.replaceStrings(perturb_lj_source, replacements);
    perturb_direct_source = cu.replaceStrings(perturb_direct_source, replacements);

    int directForceGroup = force.getDirectSpaceForceGroup() < 0 ? force.getForceGroup() : force.getDirectSpaceForceGroup();
    int ljForceGroup = force.getLJForceGroup();

    //if (force.getIncludeDirectSpace())
    //    cu.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup(), numParticles > 3000, true);
    if (ljForceGroup < 0){
        // Direct source have Coulomb and LJ interactions, we don't need to split them since LJ force group is not set.
        if (force.getIncludeDirectSpace()) // Direct forces
            cu.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, perturb_direct_source, directForceGroup, numParticles > 3000, true);
    } else {
        if (force.getIncludeDirectSpace() && hasCoulomb) // Coulomb force group
            cu.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, perturb_coulomb_source, directForceGroup, numParticles > 3000, true);
        if (force.getIncludeDirectSpace() && force.getIncludeLJ() && hasLJ) // LJ force group
            cu.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, perturb_lj_source, ljForceGroup, numParticles > 3000, true);
    }

    // Initialize the exceptions.

    int numContexts = cu.getNumContexts();
    int startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0) {
        paramsDefines["HAS_EXCEPTIONS"] = "1";
        exceptionAtoms.resize(numExceptions);
        vector<vector<int> > atoms(numExceptions, vector<int>(2));
        exceptionParams.initialize<mm_float4>(cu, numExceptions, "exceptionParams");
        baseExceptionParams.initialize<mm_float4>(cu, numExceptions, "baseExceptionParams");
        vector<mm_float4> baseExceptionParamsVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], atoms[i][0], atoms[i][1], chargeProd, sigma, epsilon);
            baseExceptionParamsVec[i] = mm_float4(chargeProd, sigma, epsilon, 0);
            exceptionAtoms[i] = make_pair(atoms[i][0], atoms[i][1]);
        }
        baseExceptionParams.upload(baseExceptionParamsVec);
        map<string, string> replacements;
        replacements["APPLY_PERIODIC"] = (usePeriodic && force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0");
        replacements["PARAMS"] = cu.getBondedUtilities().addArgument(exceptionParams, "float4");
        if (force.getIncludeDirectSpace())
            cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CommonKernelSources::nonbondedExceptions, replacements), force.getForceGroup());
    }

    // Initialize parameter offsets.

    vector<vector<mm_float4> > particleOffsetVec(force.getNumParticles());
    vector<vector<mm_float4> > exceptionOffsetVec(numExceptions);
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        int paramIndex = cu.registerGlobalParam(param);
        paramIndices[param] = paramIndex;
        particleOffsetVec[particle].push_back(mm_float4(charge, sigma, epsilon, (float) paramIndex));
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        int index = exceptionIndex[exception];
        if (index < startIndex || index >= endIndex)
            continue;
        int paramIndex = cu.registerGlobalParam(param);
        paramIndices[param] = paramIndex;
        exceptionOffsetVec[index-startIndex].push_back(mm_float4(charge, sigma, epsilon, (float) paramIndex));
    }
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        paramValues[force.getGlobalParameterName(i)] = force.getGlobalParameterDefaultValue(i);
    particleParamOffsets.initialize<mm_float4>(cu, max(force.getNumParticleParameterOffsets(), 1), "particleParamOffsets");
    particleOffsetIndices.initialize<int>(cu, cu.getPaddedNumAtoms()+1, "particleOffsetIndices");
    vector<int> particleOffsetIndicesVec, exceptionOffsetIndicesVec;
    vector<mm_float4> p, e;
    for (int i = 0; i < particleOffsetVec.size(); i++) {
        particleOffsetIndicesVec.push_back(p.size());
        for (int j = 0; j < particleOffsetVec[i].size(); j++)
            p.push_back(particleOffsetVec[i][j]);
    }
    while (particleOffsetIndicesVec.size() < particleOffsetIndices.getSize())
        particleOffsetIndicesVec.push_back(p.size());
    for (int i = 0; i < exceptionOffsetVec.size(); i++) {
        exceptionOffsetIndicesVec.push_back(e.size());
        for (int j = 0; j < exceptionOffsetVec[i].size(); j++)
            e.push_back(exceptionOffsetVec[i][j]);
    }
    exceptionOffsetIndicesVec.push_back(e.size());
    if (force.getNumParticleParameterOffsets() > 0) {
        particleParamOffsets.upload(p);
        particleOffsetIndices.upload(particleOffsetIndicesVec);
    }
    exceptionParamOffsets.initialize<mm_float4>(cu, max((int) e.size(), 1), "exceptionParamOffsets");
    exceptionOffsetIndices.initialize<int>(cu, exceptionOffsetIndicesVec.size(), "exceptionOffsetIndices");
    if (e.size() > 0) {
        exceptionParamOffsets.upload(e);
        exceptionOffsetIndices.upload(exceptionOffsetIndicesVec);
    }
    chargeBuffer.initialize(cu, cu.getNumThreadBlocks(), cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "chargeBuffer");
    cu.clearBuffer(chargeBuffer);
    recomputeParams = true;

    // Initialize the kernel for updating parameters.

    ComputeProgram program = cu.compileProgram(CommonKernelSources::nonbondedParameters, paramsDefines);
    computeParamsKernel = program->createKernel("computeParameters");
    computeExclusionParamsKernel = program->createKernel("computeExclusionParameters");
    computePlasmaCorrectionKernel = program->createKernel("computePlasmaCorrection");

    /*if (usePerturbDispersionCorrection){
        self.getPerturbDispersionCorrections(EcorrA, EcorrB, rest2_scale);
    }*/
    // Cache selfEwald energy and perturb dispersion correction
    numReplicas = force.getNumReplicas();
    if (enableCacheEnergyConstants){
        lambdaSchedules = force.getLambdaSchedules();
        for(int ii = 0; ii < numReplicas; ii++){
            ewaldSelfEnergyCache[ii] = calcCachedEwaldSelfEnergy(
                                        perturbChargesParams_cpu, 
                                        perturbAtomTypesParams_cpu,
                                        lambdaSchedules["lambdaA_charge"][ii], 
                                        lambdaSchedules["lambdaB_charge"][ii], 
                                        lambdaSchedules["lambdaC_charge"][ii], 
                                        lambdaSchedules["rest2_scale"][ii]);
            if (usePerturbDispersionCorrection){
                EcorrCache[ii] = calcCachedPerturbDispersionCorrection(
                                        perturbSigmaEpsilonParams_cpu, 
                                        perturbAtomTypesParams_cpu, 
                                        lambdaSchedules["rest2_scale"][ii], 
                                        lambdaSchedules["lambdaA_vdw"][ii], 
                                        lambdaSchedules["lambdaB_vdw"][ii]);
            }
        }
    }
    if (!enableCacheEnergyConstants && usePerturbDispersionCorrection){
        paramsDefines["USE_PERTURB_DISPERSION_CORRECTION"] = usePerturbDispersionCorrection ? "1" : "0";
    }    

    ComputeProgram perturbModule = cu.compileProgram(CudaPerturbNonbondedKernelSources::perturbParams, paramsDefines);
    updatePerturbParamsKernel = perturbModule->createKernel("updatePerturbParameters");
    perturbParamsUpdateFlag = true;

    // Compile zero-copy particle energy evaluators.  Query positions and parameters
    // are packed float4 arrays so their layout is independent of Context precision.
    map<string, string> queryDefines = defines;
    queryDefines["NUM_ATOMS"] = cu.intToString(numParticles);
    queryDefines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
    queryDefines["CUTOFF_SQUARED"] = cu.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    const double paddedQueryCutoff = cu.getNonbondedUtilities().padCutoff(force.getCutoffDistance());
    queryDefines["PADDED_CUTOFF"] = cu.doubleToString(paddedQueryCutoff);
    queryDefines["PADDED_CUTOFF_SQUARED"] = cu.doubleToString(paddedQueryCutoff*paddedQueryCutoff);
    double queryReactionFieldK = pow(force.getCutoffDistance(), -3.0)*(force.getReactionFieldDielectric()-1.0)/
            (2.0*force.getReactionFieldDielectric()+1.0);
    double queryReactionFieldC = (1.0/force.getCutoffDistance())*(3.0*force.getReactionFieldDielectric())/
            (2.0*force.getReactionFieldDielectric()+1.0);
    queryDefines["REACTION_FIELD_K"] = cu.doubleToString(queryReactionFieldK);
    queryDefines["REACTION_FIELD_C"] = cu.doubleToString(queryReactionFieldC);
    queryDefines["SOFTCORE_ALPHA"] = cu.doubleToString(force.getSoftcoreAlpha());
    queryDefines["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
    queryDefines["HAS_COULOMB"] = hasCoulomb ? "1" : "0";
    queryDefines["HAS_LENNARD_JONES"] = (hasLJ && force.getIncludeLJ()) ? "1" : "0";
    queryDefines["USE_POSQ_CHARGES"] = usePosqCharges ? "1" : "0";
    queryDefines["USE_CUTOFF"] = "1";
    queryDefines["USE_PERIODIC"] = "1";
    queryDefines.erase("USE_EWALD");
    CUmodule realQueryModule = cu.createModule(CudaKernelSources::vectorOps+
            CudaPerturbNonbondedKernelSources::particleEnergyQueries, queryDefines);
    realParticleEnergyKernel = cu.getKernel(realQueryModule, "computeParticleInteractionEnergies");

    queryDefines["USE_EWALD"] = "1";
    queryDefines["EWALD_ALPHA"] = cu.doubleToString(alpha);
    queryDefines["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
    CUmodule candidateQueryModule = cu.createModule(CudaKernelSources::vectorOps+
            CudaPerturbNonbondedKernelSources::particleEnergyQueries, queryDefines);
    candidateParticleEnergyKernel = cu.getKernel(candidateQueryModule, "computeParticleInteractionEnergies");
    candidateSubsetEnergyKernel = cu.getKernel(candidateQueryModule, "computeCandidateSubsetEnergies");
    realPmeDirectEnergyKernel = cu.getKernel(candidateQueryModule, "computeRealPmeDirectEnergies");
    map<string, string> contextQueryDefines = queryDefines;
    contextQueryDefines["CONTEXT_PRECISION_POSITIONS"] = "1";
    CUmodule contextCandidateQueryModule = cu.createModule(CudaKernelSources::vectorOps+
            CudaPerturbNonbondedKernelSources::particleEnergyQueries, contextQueryDefines);
    candidateParticleEnergyContextPrecisionKernel = cu.getKernel(contextCandidateQueryModule,
            "computeParticleInteractionEnergies");
    candidateSubsetEnergyContextPrecisionKernel = cu.getKernel(contextCandidateQueryModule,
            "computeCandidateSubsetEnergies");

    if (pmeGrid1.isInitialized() && pmeio == NULL) {
        map<string, string> potentialDefines = pmeDefines;
        potentialDefines["SQRT_ONE_4PI_EPS0"] = cu.doubleToString(sqrt(ONE_4PI_EPS0));
        if (usePosqCharges)
            potentialDefines["USE_POSQ_CHARGES"] = "1";
        CUmodule potentialModule = cu.createModule(CudaKernelSources::vectorOps+
                CudaPerturbNonbondedKernelSources::particleEnergyQueries, potentialDefines);
        CUmodule scratchModule = cu.createModule(CudaKernelSources::vectorOps+
                CudaPerturbNonbondedKernelSources::pmeScratchQueries, potentialDefines);
        map<string, string> contextPotentialDefines = potentialDefines;
        contextPotentialDefines["CONTEXT_PRECISION_POSITIONS"] = "1";
        CUmodule contextPotentialModule = cu.createModule(CudaKernelSources::vectorOps+
                CudaPerturbNonbondedKernelSources::particleEnergyQueries, contextPotentialDefines);
        CUmodule contextScratchModule = cu.createModule(CudaKernelSources::vectorOps+
                CudaPerturbNonbondedKernelSources::pmeScratchQueries, contextPotentialDefines);
        pmePotentialKernel = cu.getKernel(potentialModule, "interpolatePmePotentialAtPositions");
        pmePotentialContextPrecisionKernel = cu.getKernel(contextPotentialModule,
                "interpolatePmePotentialAtPositions");
        scratchPmeSpreadKernel = cu.getKernel(scratchModule, "spreadScratchPmeCharges");
        scratchPmeSpreadContextPrecisionKernel = cu.getKernel(contextScratchModule,
                "spreadScratchPmeCharges");
        reduceCandidateSelfEnergyKernel = cu.getKernel(scratchModule, "reduceScratchCandidateSelfEnergy");
        pmeAtomicEnergyKernel = cu.getKernel(potentialModule, "computePmeReciprocalAtomicEnergies");
        sumMixedValuesKernel = cu.getKernel(potentialModule, "sumMixedValues");
    }
    else {
        pmePotentialKernel = NULL;
        pmePotentialContextPrecisionKernel = NULL;
        scratchPmeSpreadKernel = NULL;
        scratchPmeSpreadContextPrecisionKernel = NULL;
        reduceCandidateSelfEnergyKernel = NULL;
        scratchPmeConvolutionKernel = ComputeKernel();
        pmeAtomicEnergyKernel = NULL;
        sumMixedValuesKernel = NULL;
    }

    info = new ForceInfo(force);
    cu.addForce(info);
}

double CudaCalcPerturbNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    ContextSelector selector(cu);
    if (!hasInitializedKernel) {
        hasInitializedKernel = true;
        computeParamsKernel->addArg(cu.getEnergyBuffer());
        computeParamsKernel->addArg();
        computeParamsKernel->addArg(cu.getGlobalParamValues());
        computeParamsKernel->addArg(cu.getPaddedNumAtoms());
        computeParamsKernel->addArg(baseParticleParams);
        computeParamsKernel->addArg(cu.getPosq());
        computeParamsKernel->addArg(charges);
        computeParamsKernel->addArg(sigmaEpsilon);
        computeParamsKernel->addArg(particleParamOffsets);
        computeParamsKernel->addArg(particleOffsetIndices);
        computeParamsKernel->addArg(chargeBuffer);

        updatePerturbParamsKernel->addArg(cu.getEnergyBuffer());
        updatePerturbParamsKernel->addArg(); // computeSelfEnergy
        updatePerturbParamsKernel->addArg(); // computePerturbDispersionCorrection
        updatePerturbParamsKernel->addArg(cu.getGlobalParamValues());
        updatePerturbParamsKernel->addArg(cu.getPaddedNumAtoms());
        updatePerturbParamsKernel->addArg(perturbChargeParams);
        updatePerturbParamsKernel->addArg(perturbSigmaEpsilonParams);
        updatePerturbParamsKernel->addArg(sigmaEpsilonA);
        updatePerturbParamsKernel->addArg(sigmaEpsilonB);
        updatePerturbParamsKernel->addArg(cu.getPosq());
        updatePerturbParamsKernel->addArg(charges);
        updatePerturbParamsKernel->addArg(ljLambdaAB);
        updatePerturbParamsKernel->addArg(particleParamOffsets);
        updatePerturbParamsKernel->addArg(particleOffsetIndices);
        // TODO: compact these lambda values to a single vector
        updatePerturbParamsKernel->addArg(); // lambdaA_charge
        updatePerturbParamsKernel->addArg(); // lambdaB_charge
        updatePerturbParamsKernel->addArg(); // lambdaC_charge
        updatePerturbParamsKernel->addArg(); // sqrt_rest2
        updatePerturbParamsKernel->addArg(); // lambdaA_vdw
        updatePerturbParamsKernel->addArg(); // lambdaB_vdw
        updatePerturbParamsKernel->addArg(); // Ecorr

        if (exceptionParams.isInitialized()) {
            computeParamsKernel->addArg((int) exceptionParams.getSize());
            computeParamsKernel->addArg(baseExceptionParams);
            computeParamsKernel->addArg(exceptionParams);
            computeParamsKernel->addArg(exceptionParamOffsets);
            computeParamsKernel->addArg(exceptionOffsetIndices);
        }
        if (exclusionParams.isInitialized()) {
            computeExclusionParamsKernel->addArg(cu.getPosq());
            computeExclusionParamsKernel->addArg(charges);
            computeExclusionParamsKernel->addArg(sigmaEpsilon);
            computeExclusionParamsKernel->addArg((int) exclusionParams.getSize());
            computeExclusionParamsKernel->addArg(exclusionAtoms);
            computeExclusionParamsKernel->addArg(exclusionParams);
        }
        computePlasmaCorrectionKernel->addArg(chargeBuffer);
        computePlasmaCorrectionKernel->addArg(cu.getEnergyBuffer());
        if (cu.getUseDoublePrecision())
            computePlasmaCorrectionKernel->addArg(alpha);
        else
            computePlasmaCorrectionKernel->addArg((float) alpha);
        computePlasmaCorrectionKernel->addArg();
        if (cosSinSums.isInitialized()) {
            ewaldSumsKernel->addArg(cu.getEnergyBuffer());
            ewaldSumsKernel->addArg(cu.getPosq());
            ewaldSumsKernel->addArg(cosSinSums);
            ewaldSumsKernel->addArg();
            ewaldForcesKernel->addArg(cu.getLongForceBuffer());
            ewaldForcesKernel->addArg(cu.getPosq());
            ewaldForcesKernel->addArg(cosSinSums);
            ewaldForcesKernel->addArg();
        }
        if (pmeGrid1.isInitialized()) {
            // Create kernels for Coulomb PME.

            map<string, string> replacements;
            replacements["CHARGE"] = (usePosqCharges ? "pos.w" : "charges[atom]");
            ComputeProgram program = cu.compileProgram(cu.replaceStrings(CommonKernelSources::pme, replacements), pmeDefines);
            pmeGridIndexKernel = program->createKernel("findAtomGridIndex");
            pmeSpreadChargeKernel = program->createKernel("gridSpreadCharge");
            pmeConvolutionKernel = program->createKernel("reciprocalConvolution");
            scratchPmeConvolutionKernel = program->createKernel("reciprocalConvolution");
            pmeEvalEnergyKernel = program->createKernel("gridEvaluateEnergy");
            pmeInterpolateForceKernel = program->createKernel("gridInterpolateForce");
            pmeGridIndexKernel->addArg(cu.getPosq());
            pmeGridIndexKernel->addArg(pmeAtomGridIndex);
            for (int i = 0; i < 8; i++)
                pmeGridIndexKernel->addArg();
            pmeSpreadChargeKernel->addArg(cu.getPosq());
            if (useFixedPointChargeSpreading)
                pmeSpreadChargeKernel->addArg(pmeGrid2);
            else
                pmeSpreadChargeKernel->addArg(pmeGrid1);
            for (int i = 0; i < 8; i++)
                pmeSpreadChargeKernel->addArg();
            pmeSpreadChargeKernel->addArg(pmeAtomGridIndex);
            pmeSpreadChargeKernel->addArg(charges);
            pmeConvolutionKernel->addArg(pmeGrid2);
            pmeConvolutionKernel->addArg(pmeBsplineModuliX);
            pmeConvolutionKernel->addArg(pmeBsplineModuliY);
            pmeConvolutionKernel->addArg(pmeBsplineModuliZ);
            scratchPmeConvolutionKernel->addArg(scratchPmeGrid2);
            scratchPmeConvolutionKernel->addArg(pmeBsplineModuliX);
            scratchPmeConvolutionKernel->addArg(pmeBsplineModuliY);
            scratchPmeConvolutionKernel->addArg(pmeBsplineModuliZ);
            for (int i = 0; i < 3; i++) {
                pmeConvolutionKernel->addArg();
                scratchPmeConvolutionKernel->addArg();
            }
            pmeEvalEnergyKernel->addArg(pmeGrid2);
            if (usePmeQueue)
                pmeEvalEnergyKernel->addArg(pmeEnergyBuffer);
            else
                pmeEvalEnergyKernel->addArg(cu.getEnergyBuffer());
            pmeEvalEnergyKernel->addArg(pmeBsplineModuliX);
            pmeEvalEnergyKernel->addArg(pmeBsplineModuliY);
            pmeEvalEnergyKernel->addArg(pmeBsplineModuliZ);
            for (int i = 0; i < 3; i++)
                pmeEvalEnergyKernel->addArg();
            pmeInterpolateForceKernel->addArg(cu.getPosq());
            pmeInterpolateForceKernel->addArg(cu.getLongForceBuffer());
            pmeInterpolateForceKernel->addArg(pmeGrid1);
            for (int i = 0; i < 8; i++)
                pmeInterpolateForceKernel->addArg();
            pmeInterpolateForceKernel->addArg(pmeAtomGridIndex);
            pmeInterpolateForceKernel->addArg(charges);
            if (useFixedPointChargeSpreading) {
                pmeFinishSpreadChargeKernel = program->createKernel("finishSpreadCharge");
                pmeFinishSpreadChargeKernel->addArg(pmeGrid2);
                pmeFinishSpreadChargeKernel->addArg(pmeGrid1);
            }
            if (usePmeQueue)
                syncQueue->setKernel(program->createKernel("addEnergy"));

            if (doLJPME) {
                // Create kernels for LJ PME.

                pmeDefines["EWALD_ALPHA"] = cu.doubleToString(dispersionAlpha);
                pmeDefines["GRID_SIZE_X"] = cu.intToString(dispersionGridSizeX);
                pmeDefines["GRID_SIZE_Y"] = cu.intToString(dispersionGridSizeY);
                pmeDefines["GRID_SIZE_Z"] = cu.intToString(dispersionGridSizeZ);
                pmeDefines["EPSILON_FACTOR"] = "1";
                pmeDefines["RECIP_EXP_FACTOR"] = cu.doubleToString(M_PI*M_PI/(dispersionAlpha*dispersionAlpha));
                pmeDefines["USE_LJPME"] = "1";
                pmeDefines["CHARGE_FROM_SIGEPS"] = "1";
                program = cu.compileProgram(CommonKernelSources::pme, pmeDefines);
                pmeDispersionGridIndexKernel = program->createKernel("findAtomGridIndex");
                pmeDispersionSpreadChargeKernel = program->createKernel("gridSpreadCharge");
                pmeDispersionConvolutionKernel = program->createKernel("reciprocalConvolution");
                pmeDispersionEvalEnergyKernel = program->createKernel("gridEvaluateEnergy");
                pmeDispersionInterpolateForceKernel = program->createKernel("gridInterpolateForce");
                pmeDispersionGridIndexKernel->addArg(cu.getPosq());
                pmeDispersionGridIndexKernel->addArg(pmeAtomGridIndex);
                for (int i = 0; i < 8; i++)
                    pmeDispersionGridIndexKernel->addArg();
                pmeDispersionSpreadChargeKernel->addArg(cu.getPosq());
                if (useFixedPointChargeSpreading)
                    pmeDispersionSpreadChargeKernel->addArg(pmeGrid2);
                else
                    pmeDispersionSpreadChargeKernel->addArg(pmeGrid1);
                for (int i = 0; i < 8; i++)
                    pmeDispersionSpreadChargeKernel->addArg();
                pmeDispersionSpreadChargeKernel->addArg(pmeAtomGridIndex);
                pmeDispersionSpreadChargeKernel->addArg(sigmaEpsilon);
                pmeDispersionConvolutionKernel->addArg(pmeGrid2);
                pmeDispersionConvolutionKernel->addArg(pmeDispersionBsplineModuliX);
                pmeDispersionConvolutionKernel->addArg(pmeDispersionBsplineModuliY);
                pmeDispersionConvolutionKernel->addArg(pmeDispersionBsplineModuliZ);
                for (int i = 0; i < 3; i++)
                    pmeDispersionConvolutionKernel->addArg();
                pmeDispersionEvalEnergyKernel->addArg(pmeGrid2);
                if (usePmeQueue)
                    pmeDispersionEvalEnergyKernel->addArg(pmeEnergyBuffer);
                else
                    pmeDispersionEvalEnergyKernel->addArg(cu.getEnergyBuffer());
                pmeDispersionEvalEnergyKernel->addArg(pmeDispersionBsplineModuliX);
                pmeDispersionEvalEnergyKernel->addArg(pmeDispersionBsplineModuliY);
                pmeDispersionEvalEnergyKernel->addArg(pmeDispersionBsplineModuliZ);
                for (int i = 0; i < 3; i++)
                    pmeDispersionEvalEnergyKernel->addArg();
                pmeDispersionInterpolateForceKernel->addArg(cu.getPosq());
                pmeDispersionInterpolateForceKernel->addArg(cu.getLongForceBuffer());
                pmeDispersionInterpolateForceKernel->addArg(pmeGrid1);
                for (int i = 0; i < 8; i++)
                    pmeDispersionInterpolateForceKernel->addArg();
                pmeDispersionInterpolateForceKernel->addArg(pmeAtomGridIndex);
                pmeDispersionInterpolateForceKernel->addArg(sigmaEpsilon);
                if (useFixedPointChargeSpreading) {
                    pmeDispersionFinishSpreadChargeKernel = program->createKernel("finishSpreadCharge");
                    pmeDispersionFinishSpreadChargeKernel->addArg(pmeGrid2);
                    pmeDispersionFinishSpreadChargeKernel->addArg(pmeGrid1);
                }
            }
        }
    }

    // Update particle and exception parameters.

    for (auto param : paramValues) {
        double value = context.getParameter(param.first);
        if (value != param.second) {
            paramValues[param.first] = value;
            recomputeParams = true;
        }
    }
  
    if (recomputeParams) {
        // Check if global parameters have changed.
        if (!enableCacheEnergyConstants){
            lambdaA_charge = context.getParameter("lambdaA_charge");
            lambdaB_charge = context.getParameter("lambdaB_charge");
            lambdaC_charge = context.getParameter("lambdaC_charge");
            rest2_scale = context.getParameter("rest2_scale");
            lambdaA_vdw = context.getParameter("lambdaA_vdw");
            lambdaB_vdw = context.getParameter("lambdaB_vdw");    
        } else {
            replica_id = (int) context.getParameter("replica_id");
            lambdaA_charge = lambdaSchedules["lambdaA_charge"][replica_id];
            lambdaB_charge = lambdaSchedules["lambdaB_charge"][replica_id];
            lambdaC_charge = lambdaSchedules["lambdaC_charge"][replica_id];
            rest2_scale = lambdaSchedules["rest2_scale"][replica_id];
            lambdaA_vdw = lambdaSchedules["lambdaA_vdw"][replica_id];
            lambdaB_vdw = lambdaSchedules["lambdaB_vdw"][replica_id];
        }

        if (lambdaA_charge != lambdaA_charge_old || lambdaB_charge != lambdaB_charge_old || lambdaC_charge != lambdaC_charge_old || rest2_scale != rest2_scale_old || lambdaA_vdw != lambdaA_vdw_old || lambdaB_vdw != lambdaB_vdw_old) {
            perturbParamsUpdateFlag = true;
            lambdaA_charge_old = lambdaA_charge;
            lambdaB_charge_old = lambdaB_charge;
            lambdaC_charge_old = lambdaC_charge;
            rest2_scale_old = rest2_scale;
            lambdaA_vdw_old = lambdaA_vdw;
            lambdaB_vdw_old = lambdaB_vdw;
            if(enableCacheEnergyConstants){
                replica_id_old = replica_id;
            }
        }
    }

    double energy = 0.0;
    double sqrt_rest2 = sqrt(rest2_scale);
    double Ecorr = 0.0;
    /* TODO: Plasma correction shall be included in calcCachedEwaldSelfEnergy.
    if (includeReciprocal && (pmeGrid1.isInitialized() || cosSinSums.isInitialized())) {
        Vec3 a, b, c;
        cu.getPeriodicBoxVectors(a, b, c);
        double volume = a[0]*b[1]*c[2];
        //energy = ewaldSelfEnergy - totalCharge*totalCharge/(8*EPSILON0*volume*alpha*alpha);
        // TODO: Add totalCharge*totalCharge/(8*EPSILON0*volume*alpha*alpha) term to updatePerturbParamsKernel->setArg(19, Ecorr);
    }*/

    if (recomputeParams || perturbParamsUpdateFlag || includeEnergy) {
        //computeParamsKernel->setArg(1, (int) (includeEnergy && includeReciprocal));
        //computeParamsKernel->execute(cu.getNumAtoms());
        // Make selfEnergy in reciprocal space only.
        // Make perturb dispersion correction in direct space only.
        int computeSelfEnergy = includeReciprocal && (!enableCacheEnergyConstants); 
        int computePerturbDispersionCorrection = includeDirect && usePerturbDispersionCorrection && (!enableCacheEnergyConstants);
        // Calculate the perturb dispersion correction on-the-fly when cached energy is not used.
        if (!enableCacheEnergyConstants && includeDirect){
            if (usePerturbDispersionCorrection){
                Vec3 a, b, c;
                cu.getPeriodicBoxVectors(a, b, c);
                double volume = a[0]*b[1]*c[2];
                double inverseVolume = 1.0/volume;
                // Since epsilon parameters are influenced by rest2_scale, 
                // we need to update EcorrA and EcorrB with new version of rest2_scale.
                vector<double> EcorrAB;
                if (hasLJ && (EcorrA == 0.0 || EcorrB == 0.0))
                    EcorrAB = this->getPerturbDispersionCorrections();
                
                Ecorr = lambdaA_vdw * EcorrAB[0] + lambdaB_vdw * EcorrAB[1];
                Ecorr = Ecorr / (lambdaA_vdw + lambdaB_vdw);
                Ecorr = (Ecorr * inverseVolume) / cu.getPaddedNumAtoms();
            }
        } else {
            Ecorr = 0.0;
        } 
        
        if (enableCacheEnergyConstants && includeDirect) {
            if (usePerturbDispersionCorrection){
                Vec3 a, b, c;
                cu.getPeriodicBoxVectors(a, b, c);
                energy += EcorrCache[replica_id] / (a[0]*b[1]*c[2]);
            }
        }

        if (enableCacheEnergyConstants && includeReciprocal) {
            energy += ewaldSelfEnergyCache[replica_id];
        }

        if (recomputeParams || perturbParamsUpdateFlag || ((includeDirect || includeReciprocal) && !enableCacheEnergyConstants)){
            updatePerturbParamsKernel->setArg(1, computeSelfEnergy);
            updatePerturbParamsKernel->setArg(2, computePerturbDispersionCorrection);
            updatePerturbParamsKernel->setArg(14, lambdaA_charge);
            updatePerturbParamsKernel->setArg(15, lambdaB_charge);
            updatePerturbParamsKernel->setArg(16, lambdaC_charge);
            updatePerturbParamsKernel->setArg(17, sqrt_rest2);
            updatePerturbParamsKernel->setArg(18, lambdaA_vdw);
            updatePerturbParamsKernel->setArg(19, lambdaB_vdw);
            updatePerturbParamsKernel->setArg(20, Ecorr);
            updatePerturbParamsKernel->execute(cu.getPaddedNumAtoms());
        }
        
        if (exclusionParams.isInitialized())
            computeExclusionParamsKernel->execute(exclusionParams.getSize());
        if (usePmeQueue) {
            paramsSyncEvent->enqueue();
            paramsSyncEvent->queueWait(pmeQueue);
        }
        recomputeParams = false;
        perturbParamsUpdateFlag = false;
    }

    // Do reciprocal space calculations.

    if (cosSinSums.isInitialized() && includeReciprocal) {
        Vec3 a, b, c;
        cu.getPeriodicBoxVectors(a, b, c);
        if (cu.getUseDoublePrecision()) {
            ewaldSumsKernel->setArg(3, mm_double4(a[0], b[1], c[2], 0));
            ewaldForcesKernel->setArg(3, mm_double4(a[0], b[1], c[2], 0));
        }
        else {
            ewaldSumsKernel->setArg(3, mm_float4((float) a[0], (float) b[1], (float) c[2], 0));
            ewaldForcesKernel->setArg(3, mm_float4((float) a[0], (float) b[1], (float) c[2], 0));
        }
        ewaldSumsKernel->execute(cosSinSums.getSize());
        ewaldForcesKernel->execute(cu.getNumAtoms());
    }
    if (pmeGrid1.isInitialized() && includeReciprocal) {
        if (usePmeQueue)
            cu.setCurrentQueue(pmeQueue);

        // Invert the periodic box vectors.

        Vec3 boxVectors[3];
        cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
        double scale = 1.0/determinant;
        mm_double4 recipBoxVectors[3];
        recipBoxVectors[0] = mm_double4(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0, 0);
        recipBoxVectors[1] = mm_double4(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0, 0);
        recipBoxVectors[2] = mm_double4((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale, -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale, 0);
        mm_float4 recipBoxVectorsFloat[3];
        for (int i = 0; i < 3; i++)
            recipBoxVectorsFloat[i] = mm_float4((float) recipBoxVectors[i].x, (float) recipBoxVectors[i].y, (float) recipBoxVectors[i].z, 0);

        // Execute the reciprocal space kernels.

        if (hasCoulomb) {
            if (stepsToSort <= 0 || doLJPME || cu.getNumAtoms() > 15000) {
                setPeriodicBoxArgs(cu, pmeGridIndexKernel, 2);
                if (cu.getUseDoublePrecision()) {
                    pmeGridIndexKernel->setArg(7, recipBoxVectors[0]);
                    pmeGridIndexKernel->setArg(8, recipBoxVectors[1]);
                    pmeGridIndexKernel->setArg(9, recipBoxVectors[2]);
                }
                else {
                    pmeGridIndexKernel->setArg(7, recipBoxVectorsFloat[0]);
                    pmeGridIndexKernel->setArg(8, recipBoxVectorsFloat[1]);
                    pmeGridIndexKernel->setArg(9, recipBoxVectorsFloat[2]);
                }
                pmeGridIndexKernel->execute(cu.getNumAtoms());
                sort->sort(pmeAtomGridIndex);
                stepsToSort = 3;
            }
            else
                stepsToSort--;
            setPeriodicBoxArgs(cu, pmeSpreadChargeKernel, 2);
            if (cu.getUseDoublePrecision()) {
                pmeSpreadChargeKernel->setArg(7, recipBoxVectors[0]);
                pmeSpreadChargeKernel->setArg(8, recipBoxVectors[1]);
                pmeSpreadChargeKernel->setArg(9, recipBoxVectors[2]);
            }
            else {
                pmeSpreadChargeKernel->setArg(7, recipBoxVectorsFloat[0]);
                pmeSpreadChargeKernel->setArg(8, recipBoxVectorsFloat[1]);
                pmeSpreadChargeKernel->setArg(9, recipBoxVectorsFloat[2]);
            }
            pmeSpreadChargeKernel->execute(cu.getNumAtoms());
            if (useFixedPointChargeSpreading)
                pmeFinishSpreadChargeKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            fft->execFFT(pmeGrid1, pmeGrid2, true);
            if (cu.getUseDoublePrecision()) {
                pmeConvolutionKernel->setArg<mm_double4>(4, recipBoxVectors[0]);
                pmeConvolutionKernel->setArg<mm_double4>(5, recipBoxVectors[1]);
                pmeConvolutionKernel->setArg<mm_double4>(6, recipBoxVectors[2]);
                pmeEvalEnergyKernel->setArg<mm_double4>(5, recipBoxVectors[0]);
                pmeEvalEnergyKernel->setArg<mm_double4>(6, recipBoxVectors[1]);
                pmeEvalEnergyKernel->setArg<mm_double4>(7, recipBoxVectors[2]);
            }
            else {
                pmeConvolutionKernel->setArg<mm_float4>(4, recipBoxVectorsFloat[0]);
                pmeConvolutionKernel->setArg<mm_float4>(5, recipBoxVectorsFloat[1]);
                pmeConvolutionKernel->setArg<mm_float4>(6, recipBoxVectorsFloat[2]);
                pmeEvalEnergyKernel->setArg<mm_float4>(5, recipBoxVectorsFloat[0]);
                pmeEvalEnergyKernel->setArg<mm_float4>(6, recipBoxVectorsFloat[1]);
                pmeEvalEnergyKernel->setArg<mm_float4>(7, recipBoxVectorsFloat[2]);
            }
            if (includeEnergy)
                pmeEvalEnergyKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            fft->execFFT(pmeGrid2, pmeGrid1, false);
            setPeriodicBoxArgs(cu, pmeInterpolateForceKernel, 3);
            if (cu.getUseDoublePrecision()) {
                pmeInterpolateForceKernel->setArg(8, recipBoxVectors[0]);
                pmeInterpolateForceKernel->setArg(9, recipBoxVectors[1]);
                pmeInterpolateForceKernel->setArg(10, recipBoxVectors[2]);
            }
            else {
                pmeInterpolateForceKernel->setArg(8, recipBoxVectorsFloat[0]);
                pmeInterpolateForceKernel->setArg(9, recipBoxVectorsFloat[1]);
                pmeInterpolateForceKernel->setArg(10, recipBoxVectorsFloat[2]);
            }
            if (deviceIsCpu)
                pmeInterpolateForceKernel->execute(cu.getNumThreadBlocks(), 1);
            else
                pmeInterpolateForceKernel->execute(cu.getNumAtoms());
        }

        if (doLJPME && hasLJ) {
            setPeriodicBoxArgs(cu, pmeDispersionGridIndexKernel, 2);
            if (cu.getUseDoublePrecision()) {
                pmeDispersionGridIndexKernel->setArg(7, recipBoxVectors[0]);
                pmeDispersionGridIndexKernel->setArg(8, recipBoxVectors[1]);
                pmeDispersionGridIndexKernel->setArg(9, recipBoxVectors[2]);
            }
            else {
                pmeDispersionGridIndexKernel->setArg(7, recipBoxVectorsFloat[0]);
                pmeDispersionGridIndexKernel->setArg(8, recipBoxVectorsFloat[1]);
                pmeDispersionGridIndexKernel->setArg(9, recipBoxVectorsFloat[2]);
            }
            pmeDispersionGridIndexKernel->execute(cu.getNumAtoms());
            sort->sort(pmeAtomGridIndex);
            if (useFixedPointChargeSpreading)
                cu.clearBuffer(pmeGrid2);
            else
                cu.clearBuffer(pmeGrid1);
            setPeriodicBoxArgs(cu, pmeDispersionSpreadChargeKernel, 2);
            if (cu.getUseDoublePrecision()) {
                pmeDispersionSpreadChargeKernel->setArg(7, recipBoxVectors[0]);
                pmeDispersionSpreadChargeKernel->setArg(8, recipBoxVectors[1]);
                pmeDispersionSpreadChargeKernel->setArg(9, recipBoxVectors[2]);
            }
            else {
                pmeDispersionSpreadChargeKernel->setArg(7, recipBoxVectorsFloat[0]);
                pmeDispersionSpreadChargeKernel->setArg(8, recipBoxVectorsFloat[1]);
                pmeDispersionSpreadChargeKernel->setArg(9, recipBoxVectorsFloat[2]);
            }
            pmeDispersionSpreadChargeKernel->execute(cu.getNumAtoms());
            if (useFixedPointChargeSpreading)
                pmeDispersionFinishSpreadChargeKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            dispersionFft->execFFT(pmeGrid1, pmeGrid2, true);
            if (cu.getUseDoublePrecision()) {
                pmeDispersionConvolutionKernel->setArg(4, recipBoxVectors[0]);
                pmeDispersionConvolutionKernel->setArg(5, recipBoxVectors[1]);
                pmeDispersionConvolutionKernel->setArg(6, recipBoxVectors[2]);
                pmeDispersionEvalEnergyKernel->setArg(5, recipBoxVectors[0]);
                pmeDispersionEvalEnergyKernel->setArg(6, recipBoxVectors[1]);
                pmeDispersionEvalEnergyKernel->setArg(7, recipBoxVectors[2]);
            }
            else {
                pmeDispersionConvolutionKernel->setArg(4, recipBoxVectorsFloat[0]);
                pmeDispersionConvolutionKernel->setArg(5, recipBoxVectorsFloat[1]);
                pmeDispersionConvolutionKernel->setArg(6, recipBoxVectorsFloat[2]);
                pmeDispersionEvalEnergyKernel->setArg(5, recipBoxVectorsFloat[0]);
                pmeDispersionEvalEnergyKernel->setArg(6, recipBoxVectorsFloat[1]);
                pmeDispersionEvalEnergyKernel->setArg(7, recipBoxVectorsFloat[2]);
            }
            if (!hasCoulomb)
                cu.clearBuffer(pmeEnergyBuffer);
            if (includeEnergy)
                pmeDispersionEvalEnergyKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            pmeDispersionConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            dispersionFft->execFFT(pmeGrid2, pmeGrid1, false);
            setPeriodicBoxArgs(cu, pmeDispersionInterpolateForceKernel, 3);
            if (cu.getUseDoublePrecision()) {
                pmeDispersionInterpolateForceKernel->setArg(8, recipBoxVectors[0]);
                pmeDispersionInterpolateForceKernel->setArg(9, recipBoxVectors[1]);
                pmeDispersionInterpolateForceKernel->setArg(10, recipBoxVectors[2]);
            }
            else {
                pmeDispersionInterpolateForceKernel->setArg(8, recipBoxVectorsFloat[0]);
                pmeDispersionInterpolateForceKernel->setArg(9, recipBoxVectorsFloat[1]);
                pmeDispersionInterpolateForceKernel->setArg(10, recipBoxVectorsFloat[2]);
            }
            if (deviceIsCpu)
                pmeDispersionInterpolateForceKernel->execute(cu.getNumThreadBlocks(), 1);
            else
                pmeDispersionInterpolateForceKernel->execute(cu.getNumAtoms());
        }
        if (usePmeQueue) {
            pmeSyncEvent->enqueue();
            cu.restoreDefaultQueue();
        }
    }
    /*
    if (dispersionCoefficient != 0.0 && includeDirect) {
        Vec3 a, b, c;
        cu.getPeriodicBoxVectors(a, b, c);
        energy += dispersionCoefficient/(a[0]*b[1]*c[2]);
    }
    */
    return energy;
}

void CudaCalcPerturbNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const PerturbNonbondedForce& force, int firstParticle, int lastParticle, int firstException, int lastException) {
    // Make sure the new parameters are acueptable.

    ContextSelector selector(cu);
    if (force.getNumParticles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (!hasCoulomb || !hasLJ) {
        for (int i = 0; i < force.getNumParticles(); i++) {
            double chargeA, sigmaA, epsilonA;
            double chargeB, sigmaB, epsilonB;
            bool isDummy, isREST2;
            force.getParticleParameters(i, chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2);
            if (!hasCoulomb && (chargeA != 0.0 || chargeB != 0.0))
                throw OpenMMException("updateParametersInContext: The nonbonded force kernel does not include Coulomb interactions, because all charges were originally 0");
            if (!hasLJ && (epsilonA != 0.0 || epsilonB != 0.0))
                throw OpenMMException("updateParametersInContext: The nonbonded force kernel does not include Lennard-Jones interactions, because all epsilons were originally 0");
        }
    }
    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionsWithOffsets.insert(exception);
    }
    vector<int> exceptions;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if (exceptionIndex.find(i) == exceptionIndex.end()) {
            if (chargeProd != 0.0 || epsilon != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end())
                throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");
        }
        else
            exceptions.push_back(i);
    }
    int numContexts = cu.getNumContexts();
    int startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions != exceptionAtoms.size())
        throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");

    // Record the per-particle parameters.

    if (firstParticle <= lastParticle) {
        vector<mm_float4> baseParticleParamVec(cu.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
        vector<mm_float4> perturbChargeParamVec(cu.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
        vector<mm_float4> perturbSigmaEpsilonParamVec(cu.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
        vector<int> ljAtomTypesVec(cu.getPaddedNumAtoms(), 0); 

        perturbChargesParams_cpu.resize(cu.getPaddedNumAtoms());
        perturbSigmaEpsilonParams_cpu.resize(cu.getPaddedNumAtoms());
        perturbAtomTypesParams_cpu.resize(cu.getPaddedNumAtoms());

        for (int i = 0; i < force.getNumParticles(); i++) {
            double chargeA, sigmaA, epsilonA;
            double chargeB, sigmaB, epsilonB;
            bool isDummy, isREST2;
            force.getParticleParameters(i, chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2);
            //baseParticleParamVec[i] = mm_float4(chargeA, sigmaA, epsilonA, 0);

            ljAtomTypesVec[i] = isDummy ? 1 : 0;
            double chargeC = 0.0;
            float rest2_sign = 0.0;
            if (isDummy){
                chargeC = 0;
                rest2_sign = isREST2 ? 1.0 : 0.0;
            } else {
                chargeC = (chargeA + chargeB) * 0.5;
                rest2_sign = isREST2 ? 1.0 : 0.0;
            }
            perturbChargeParamVec[i] = mm_float4(chargeA, chargeB, chargeC, rest2_sign);
            perturbSigmaEpsilonParamVec[i] = mm_float4(0.5 * sigmaA, 2 * sqrt(epsilonA), 0.5 * sigmaB, 2 * sqrt(epsilonB));

            perturbChargesParams_cpu[i] = {chargeA, chargeB, chargeC};
            perturbSigmaEpsilonParams_cpu[i] = {sigmaA, epsilonA, sigmaB, epsilonB};
            perturbAtomTypesParams_cpu[i] = {isDummy, isREST2};
        }
        baseParticleParams.uploadSubArray(&baseParticleParamVec[firstParticle], firstParticle, lastParticle-firstParticle+1);
        perturbChargeParams.uploadSubArray(&perturbChargeParamVec[firstParticle], firstParticle, lastParticle-firstParticle+1);
        perturbSigmaEpsilonParams.uploadSubArray(&perturbSigmaEpsilonParamVec[firstParticle], firstParticle, lastParticle-firstParticle+1);
        ljAtomTypes.uploadSubArray(&ljAtomTypesVec[firstParticle], firstParticle, lastParticle-firstParticle+1);

        // Compute the self energy.

        ewaldSelfEnergy = 0.0;
        totalCharge = 0.0;
        if (nonbondedMethod == Ewald || nonbondedMethod == PME || nonbondedMethod == LJPME) {
            if (cu.getContextIndex() == 0) {
                for (int i = 0; i < force.getNumParticles(); i++) {
                    ewaldSelfEnergy -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
                    totalCharge += baseParticleParamVec[i].x;
                    if (doLJPME)
                        ewaldSelfEnergy += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
                }
            }
        }
    }

    // Record the exceptions.

    if (firstException <= lastException && numExceptions > 0) {
        vector<mm_float4> baseExceptionParamsVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            int particle1, particle2;
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], particle1, particle2, chargeProd, sigma, epsilon);
            if (make_pair(particle1, particle2) != exceptionAtoms[i])
                throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");
            baseExceptionParamsVec[i] = mm_float4(chargeProd, sigma, epsilon, 0);
        }
        baseExceptionParams.upload(baseExceptionParamsVec);
    }

    // Record parameter offsets.

    vector<vector<mm_float4> > particleOffsetVec(force.getNumParticles());
    vector<vector<mm_float4> > exceptionOffsetVec(numExceptions);
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        auto paramIndex = paramIndices.find(param);
        if (paramIndex == paramIndices.end())
            throw OpenMMException("updateParametersInContext: The parameter of a particle parameter offset has changed");
        particleOffsetVec[particle].push_back(mm_float4(charge, sigma, epsilon, paramIndex->second));
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        int index = exceptionIndex[exception];
        if (index < startIndex || index >= endIndex)
            continue;
        auto paramIndex = paramIndices.find(param);
        if (paramIndex == paramIndices.end())
            throw OpenMMException("updateParametersInContext: The parameter of an exception parameter offset has changed");
        exceptionOffsetVec[index-startIndex].push_back(mm_float4(charge, sigma, epsilon, paramIndex->second));
    }
    if (max(force.getNumParticleParameterOffsets(), 1) != particleParamOffsets.getSize())
        throw OpenMMException("updateParametersInContext: The number of particle parameter offsets has changed");
    vector<mm_float4> p, e;
    for (int i = 0; i < particleOffsetVec.size(); i++)
        for (int j = 0; j < particleOffsetVec[i].size(); j++)
            p.push_back(particleOffsetVec[i][j]);
    for (int i = 0; i < exceptionOffsetVec.size(); i++)
        for (int j = 0; j < exceptionOffsetVec[i].size(); j++)
            e.push_back(exceptionOffsetVec[i][j]);
    if (force.getNumParticleParameterOffsets() > 0)
        particleParamOffsets.upload(p);
    if (max((int) e.size(), 1) != exceptionParamOffsets.getSize())
        throw OpenMMException("updateParametersInContext: The number of exception parameter offsets has changed");
    if (e.size() > 0)
        exceptionParamOffsets.upload(e);

    // Compute other values.

    if (force.getUseDispersionCorrection() && cu.getContextIndex() == 0 && (nonbondedMethod == CutoffPeriodic || nonbondedMethod == Ewald || nonbondedMethod == PME)){
        //dispersionCoefficient = PerturbNonbondedForceImpl::calcDispersionCorrection(context.getSystem(), force);
        //self.getPerturbDispersionCorrections(EcorrA, EcorrB, rest2_scale);
        usePerturbDispersionCorrection = true;
    }

    enableCacheEnergyConstants = force.getEnableEnergyConstantCache();

    // Cache selfEwald energy and perturb dispersion correction
    numReplicas = force.getNumReplicas();
    if (enableCacheEnergyConstants){
        lambdaSchedules = force.getLambdaSchedules();
        for(int ii = 0; ii < numReplicas; ii++){
            ewaldSelfEnergyCache[ii] = calcCachedEwaldSelfEnergy(
                                        perturbChargesParams_cpu, 
                                        perturbAtomTypesParams_cpu,
                                        lambdaSchedules["lambdaA_charge"][ii], 
                                        lambdaSchedules["lambdaB_charge"][ii], 
                                        lambdaSchedules["lambdaC_charge"][ii], 
                                        lambdaSchedules["rest2_scale"][ii]);
            if (usePerturbDispersionCorrection){
                EcorrCache[ii] = calcCachedPerturbDispersionCorrection(
                                        perturbSigmaEpsilonParams_cpu, 
                                        perturbAtomTypesParams_cpu, 
                                        lambdaSchedules["rest2_scale"][ii], 
                                        lambdaSchedules["lambdaA_vdw"][ii], 
                                        lambdaSchedules["lambdaB_vdw"][ii]);
            }
        }
    }

    cu.invalidateMolecules(info, firstParticle <= lastParticle || force.getNumParticleParameterOffsets() > 0,
                           firstException <= lastException || force.getNumExceptionParameterOffsets() > 0);
    recomputeParams = true;
    perturbParamsUpdateFlag = true;
}

void CudaCalcPerturbNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != PME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    if (useCpuPme)
        cpuPme.getAs<CalcPmeReciprocalForceKernel>().getPMEParameters(alpha, nx, ny, nz);
    else {
        alpha = this->alpha;
        nx = gridSizeX;
        ny = gridSizeY;
        nz = gridSizeZ;
    }
}

void CudaCalcPerturbNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != LJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    if (useCpuPme)
        //cpuPme.getAs<CalcPmeReciprocalForceKernel>().getLJPMEParameters(alpha, nx, ny, nz);
        throw OpenMMException("getPMEParametersInContext: CPUPME has not been implemented for LJPME yet.");
    else {
        alpha = this->dispersionAlpha;
        nx = dispersionGridSizeX;
        ny = dispersionGridSizeY;
        nz = dispersionGridSizeZ;
    }
}


void CudaCalcPerturbNonbondedForceKernel::getInteractionEnergyForRealParticlesCuda(
        ContextImpl&, int numRealAtoms, CudaArrayDevicePointer positions,
        CudaArrayDevicePointer parameters, CudaArrayDevicePointer excludedAtomStarts,
        CudaArrayDevicePointer excludedAtomCounts, CudaArrayDevicePointer excludedAtoms,
        CudaArrayDevicePointer forces, CudaArrayDevicePointer energies) {
    ContextSelector selector(cu);
    if (cu.getNumContexts() != 1)
        throw OpenMMException("Particle energy queries currently require a single CUDA device.");
    if (numRealAtoms < 0)
        throw OpenMMException("The real-particle query count cannot be negative.");
    if (numRealAtoms == 0)
        return;
    int hasExclusions = 1;
    int startBlockIndex = 0;
    int numBlocks = cu.getNumAtomBlocks();
    CUdeviceptr positionPointer = static_cast<CUdeviceptr>(positions);
    CUdeviceptr parameterPointer = static_cast<CUdeviceptr>(parameters);
    CUdeviceptr excludedStartsPointer = static_cast<CUdeviceptr>(excludedAtomStarts);
    CUdeviceptr excludedCountsPointer = static_cast<CUdeviceptr>(excludedAtomCounts);
    CUdeviceptr excludedAtomsPointer = static_cast<CUdeviceptr>(excludedAtoms);
    CUdeviceptr forcePointer = static_cast<CUdeviceptr>(forces);
    CUdeviceptr energyPointer = static_cast<CUdeviceptr>(energies);
    void* args[] = {&numRealAtoms, cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &cu.getPosq().getDevicePointer(), &charges.getDevicePointer(),
            &sigmaEpsilonA.getDevicePointer(), &sigmaEpsilonB.getDevicePointer(),
            &ljAtomTypes.getDevicePointer(), &positionPointer, &parameterPointer,
            &excludedStartsPointer, &excludedCountsPointer, &excludedAtomsPointer,
            &cu.getAtomIndexArray().getDevicePointer(), &hasExclusions,
            &lambdaA_vdw, &lambdaB_vdw, &startBlockIndex, &numBlocks, &useParticleQueryBlockPruning,
            &cu.getNonbondedUtilities().getBlockCenters().getDevicePointer(),
            &cu.getNonbondedUtilities().getBlockBoundingBoxes().getDevicePointer(),
            &forcePointer, &energyPointer};
    cu.executeKernel(realParticleEnergyKernel, args, numRealAtoms);
}

void CudaCalcPerturbNonbondedForceKernel::getDirectEnergyForCandidateParticlesCuda(
        ContextImpl&, int numCandidateAtoms, CudaArrayDevicePointer positions,
        CudaArrayDevicePointer parameters, CudaArrayDevicePointer forces,
        CudaArrayDevicePointer energies) {
    ContextSelector selector(cu);
    if (cu.getNumContexts() != 1)
        throw OpenMMException("Particle energy queries currently require a single CUDA device.");
    if (nonbondedMethod != PME)
        throw OpenMMException("Candidate PME energy queries require PerturbNonbondedForce::PME.");
    if (numCandidateAtoms < 0)
        throw OpenMMException("The candidate-particle query count cannot be negative.");
    if (numCandidateAtoms == 0)
        return;
    int hasExclusions = 0;
    int startBlockIndex = 0;
    int numBlocks = cu.getNumAtomBlocks();
    CUdeviceptr positionPointer = static_cast<CUdeviceptr>(positions);
    CUdeviceptr parameterPointer = static_cast<CUdeviceptr>(parameters);
    CUdeviceptr forcePointer = static_cast<CUdeviceptr>(forces);
    CUdeviceptr energyPointer = static_cast<CUdeviceptr>(energies);
    CUdeviceptr unused = 0;
    void* args[] = {&numCandidateAtoms, cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &cu.getPosq().getDevicePointer(), &charges.getDevicePointer(),
            &sigmaEpsilonA.getDevicePointer(), &sigmaEpsilonB.getDevicePointer(),
            &ljAtomTypes.getDevicePointer(), &positionPointer, &parameterPointer,
            &unused, &unused, &unused, &cu.getAtomIndexArray().getDevicePointer(), &hasExclusions,
            &lambdaA_vdw, &lambdaB_vdw, &startBlockIndex, &numBlocks, &useParticleQueryBlockPruning,
            &cu.getNonbondedUtilities().getBlockCenters().getDevicePointer(),
            &cu.getNonbondedUtilities().getBlockBoundingBoxes().getDevicePointer(),
            &forcePointer, &energyPointer};
    cu.executeKernel(candidateParticleEnergyKernel, args, numCandidateAtoms);
}

void CudaCalcPerturbNonbondedForceKernel::getDirectEnergyForCandidateParticlesCudaContextPrecision(
        ContextImpl&, int numCandidateAtoms, CudaArrayDevicePointer positionBase,
        CudaArrayDevicePointer positionCorrection, CudaArrayDevicePointer parameters,
        CudaArrayDevicePointer forces, CudaArrayDevicePointer energies) {
    ContextSelector selector(cu);
    if (cu.getNumContexts() != 1)
        throw OpenMMException("Context-precision particle energy queries currently require a single CUDA device.");
    if (nonbondedMethod != PME)
        throw OpenMMException("Context-precision candidate PME energy queries require PerturbNonbondedForce::PME.");
    if (numCandidateAtoms < 0)
        throw OpenMMException("The context-precision candidate-particle query count cannot be negative.");
    if (numCandidateAtoms == 0)
        return;
    validateContextPrecisionPositionCorrection(cu, positionCorrection,
            "Context-precision candidate-particle queries");
    int hasExclusions = 0;
    int startBlockIndex = 0;
    int numBlocks = cu.getNumAtomBlocks();
    CUdeviceptr positionBasePointer = static_cast<CUdeviceptr>(positionBase);
    CUdeviceptr positionCorrectionPointer = static_cast<CUdeviceptr>(positionCorrection);
    CUdeviceptr contextCorrectionPointer = cu.getUseMixedPrecision() ?
            cu.getPosqCorrection().getDevicePointer() : 0;
    CUdeviceptr parameterPointer = static_cast<CUdeviceptr>(parameters);
    CUdeviceptr forcePointer = static_cast<CUdeviceptr>(forces);
    CUdeviceptr energyPointer = static_cast<CUdeviceptr>(energies);
    CUdeviceptr unused = 0;
    void* args[] = {&numCandidateAtoms, cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &cu.getPosq().getDevicePointer(), &contextCorrectionPointer, &charges.getDevicePointer(),
            &sigmaEpsilonA.getDevicePointer(), &sigmaEpsilonB.getDevicePointer(),
            &ljAtomTypes.getDevicePointer(), &positionBasePointer, &positionCorrectionPointer,
            &parameterPointer, &unused, &unused, &unused, &cu.getAtomIndexArray().getDevicePointer(),
            &hasExclusions, &lambdaA_vdw, &lambdaB_vdw, &startBlockIndex, &numBlocks,
            &useParticleQueryBlockPruning, &cu.getNonbondedUtilities().getBlockCenters().getDevicePointer(),
            &cu.getNonbondedUtilities().getBlockBoundingBoxes().getDevicePointer(),
            &forcePointer, &energyPointer};
    cu.executeKernel(candidateParticleEnergyContextPrecisionKernel, args, numCandidateAtoms);
}

void CudaCalcPerturbNonbondedForceKernel::getDirectEnergyForCandidateSubsetCuda(
        ContextImpl&, int numCandidateAtoms, CudaArrayDevicePointer positions,
        CudaArrayDevicePointer parameters, int numSourceAtoms,
        CudaArrayDevicePointer sourceAtomIndices, CudaArrayDevicePointer energies) {
    ContextSelector selector(cu);
    if (cu.getNumContexts() != 1)
        throw OpenMMException("Candidate-subset direct queries currently require a single CUDA device.");
    if (nonbondedMethod != PME)
        throw OpenMMException("Candidate-subset direct queries require PerturbNonbondedForce::PME.");
    if (numCandidateAtoms < 0 || numSourceAtoms < 0)
        throw OpenMMException("Candidate-subset query counts cannot be negative.");
    if (numCandidateAtoms == 0)
        return;
    CUdeviceptr positionPointer = static_cast<CUdeviceptr>(positions);
    CUdeviceptr parameterPointer = static_cast<CUdeviceptr>(parameters);
    CUdeviceptr sourcePointer = static_cast<CUdeviceptr>(sourceAtomIndices);
    CUdeviceptr energyPointer = static_cast<CUdeviceptr>(energies);
    void* args[] = {&numCandidateAtoms, &numSourceAtoms,
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &cu.getPosq().getDevicePointer(), &charges.getDevicePointer(),
            &sigmaEpsilonA.getDevicePointer(), &sigmaEpsilonB.getDevicePointer(),
            &ljAtomTypes.getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(),
            &positionPointer, &parameterPointer, &sourcePointer,
            &lambdaA_vdw, &lambdaB_vdw, &energyPointer};
    cu.executeKernel(candidateSubsetEnergyKernel, args, numCandidateAtoms);
}

void CudaCalcPerturbNonbondedForceKernel::getDirectEnergyForCandidateSubsetCudaContextPrecision(
        ContextImpl&, int numCandidateAtoms, CudaArrayDevicePointer positionBase,
        CudaArrayDevicePointer positionCorrection, CudaArrayDevicePointer parameters,
        int numSourceAtoms, CudaArrayDevicePointer sourceAtomIndices,
        CudaArrayDevicePointer energies) {
    ContextSelector selector(cu);
    if (cu.getNumContexts() != 1)
        throw OpenMMException("Context-precision candidate-subset direct queries currently require a single CUDA device.");
    if (nonbondedMethod != PME)
        throw OpenMMException("Context-precision candidate-subset direct queries require PerturbNonbondedForce::PME.");
    if (numCandidateAtoms < 0 || numSourceAtoms < 0)
        throw OpenMMException("Context-precision candidate-subset query counts cannot be negative.");
    if (numCandidateAtoms == 0)
        return;
    validateContextPrecisionPositionCorrection(cu, positionCorrection,
            "Context-precision candidate-subset queries");
    CUdeviceptr positionBasePointer = static_cast<CUdeviceptr>(positionBase);
    CUdeviceptr positionCorrectionPointer = static_cast<CUdeviceptr>(positionCorrection);
    CUdeviceptr contextCorrectionPointer = cu.getUseMixedPrecision() ?
            cu.getPosqCorrection().getDevicePointer() : 0;
    CUdeviceptr parameterPointer = static_cast<CUdeviceptr>(parameters);
    CUdeviceptr sourcePointer = static_cast<CUdeviceptr>(sourceAtomIndices);
    CUdeviceptr energyPointer = static_cast<CUdeviceptr>(energies);
    void* args[] = {&numCandidateAtoms, &numSourceAtoms,
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &cu.getPosq().getDevicePointer(), &contextCorrectionPointer, &charges.getDevicePointer(),
            &sigmaEpsilonA.getDevicePointer(), &sigmaEpsilonB.getDevicePointer(),
            &ljAtomTypes.getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(),
            &positionBasePointer, &positionCorrectionPointer, &parameterPointer, &sourcePointer,
            &lambdaA_vdw, &lambdaB_vdw, &energyPointer};
    cu.executeKernel(candidateSubsetEnergyContextPrecisionKernel, args, numCandidateAtoms);
}

void CudaCalcPerturbNonbondedForceKernel::getDirectEnergyForRealParticlesPmeCuda(
        ContextImpl&, int numRealAtoms, CudaArrayDevicePointer realAtomIndices,
        int numAtomsPerMolecule, CudaArrayDevicePointer energies) {
    ContextSelector selector(cu);
    if (cu.getNumContexts() != 1)
        throw OpenMMException("Real PME direct queries currently require a single CUDA device.");
    if (nonbondedMethod != PME)
        throw OpenMMException("Real PME direct queries require PerturbNonbondedForce::PME.");
    if (numRealAtoms < 0 || numAtomsPerMolecule <= 0 || numRealAtoms%numAtomsPerMolecule != 0)
        throw OpenMMException("Real PME direct query counts are invalid.");
    if (numRealAtoms == 0)
        return;
    CUdeviceptr indexPointer = static_cast<CUdeviceptr>(realAtomIndices);
    CUdeviceptr energyPointer = static_cast<CUdeviceptr>(energies);
    void* args[] = {&numRealAtoms, &numAtomsPerMolecule,
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &cu.getPosq().getDevicePointer(), &charges.getDevicePointer(),
            &sigmaEpsilonA.getDevicePointer(), &sigmaEpsilonB.getDevicePointer(),
            &ljAtomTypes.getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(),
            &indexPointer, &lambdaA_vdw, &lambdaB_vdw, &energyPointer};
    cu.executeKernel(realPmeDirectEnergyKernel, args, numRealAtoms);
}

void CudaCalcPerturbNonbondedForceKernel::getPMEPotentialFromParticlesCuda(
        ContextImpl&, int numSourceParticles, CudaArrayDevicePointer sourcePositions,
        CudaArrayDevicePointer sourceCharges, int numQueryPositions,
        CudaArrayDevicePointer queryPositions, CudaArrayDevicePointer queryPotentials) {
    ContextSelector selector(cu);
    if (cu.getNumContexts() != 1)
        throw OpenMMException("PME scratch potential queries currently require a single CUDA device.");
    if (nonbondedMethod != PME || pmePotentialKernel == NULL || !scratchPmeSpreadKernel)
        throw OpenMMException("PME scratch potential queries require GPU PME with PerturbNonbondedForce::PME.");
    if (numSourceParticles <= 0 || numQueryPositions < 0)
        throw OpenMMException("PME scratch query counts must be positive for sources and nonnegative for queries.");
    if (numQueryPositions == 0)
        return;
    if (usePmeQueue)
        pmeSyncEvent->queueWait(cu.getCurrentQueue());
    Vec3 boxVectors[3];
    cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    const double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
    const double scale = 1.0/determinant;
    mm_double4 reciprocal[3] = {
        mm_double4(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0, 0),
        mm_double4(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0, 0),
        mm_double4((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale,
                -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale, 0)
    };
    mm_float4 reciprocalFloat[3];
    for (int i = 0; i < 3; i++)
        reciprocalFloat[i] = mm_float4((float) reciprocal[i].x, (float) reciprocal[i].y,
                (float) reciprocal[i].z, 0);
    CUdeviceptr sourcePositionPointer = static_cast<CUdeviceptr>(sourcePositions);
    CUdeviceptr sourceChargePointer = static_cast<CUdeviceptr>(sourceCharges);
    CUdeviceptr queryPositionPointer = static_cast<CUdeviceptr>(queryPositions);
    CUdeviceptr queryPotentialPointer = static_cast<CUdeviceptr>(queryPotentials);
    cu.clearBuffer(scratchPmeGrid1);
    cu.clearBuffer(scratchPmeGrid2);
    void* spreadArgs[] = {&numSourceParticles, &sourcePositionPointer, &sourceChargePointer,
            &scratchPmeGrid1.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
            cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
            cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            cu.getUseDoublePrecision() ? (void*) &reciprocal[0] : (void*) &reciprocalFloat[0],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[1] : (void*) &reciprocalFloat[1],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[2] : (void*) &reciprocalFloat[2]};
    cu.executeKernel(scratchPmeSpreadKernel, spreadArgs, numSourceParticles);
    fft->execFFT(scratchPmeGrid1, scratchPmeGrid2, true);
    if (cu.getUseDoublePrecision()) {
        scratchPmeConvolutionKernel->setArg<mm_double4>(4, reciprocal[0]);
        scratchPmeConvolutionKernel->setArg<mm_double4>(5, reciprocal[1]);
        scratchPmeConvolutionKernel->setArg<mm_double4>(6, reciprocal[2]);
    }
    else {
        scratchPmeConvolutionKernel->setArg<mm_float4>(4, reciprocalFloat[0]);
        scratchPmeConvolutionKernel->setArg<mm_float4>(5, reciprocalFloat[1]);
        scratchPmeConvolutionKernel->setArg<mm_float4>(6, reciprocalFloat[2]);
    }
    scratchPmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
    fft->execFFT(scratchPmeGrid2, scratchPmeGrid1, false);
    void* gatherArgs[] = {&numQueryPositions, &scratchPmeGrid1.getDevicePointer(),
            &queryPositionPointer, &queryPotentialPointer, cu.getPeriodicBoxSizePointer(),
            cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
            cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            cu.getUseDoublePrecision() ? (void*) &reciprocal[0] : (void*) &reciprocalFloat[0],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[1] : (void*) &reciprocalFloat[1],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[2] : (void*) &reciprocalFloat[2]};
    cu.executeKernel(pmePotentialKernel, gatherArgs, numQueryPositions);
}

void CudaCalcPerturbNonbondedForceKernel::getPMEPotentialFromParticlesCudaContextPrecision(
        ContextImpl&, int numSourceParticles, CudaArrayDevicePointer sourcePositionBase,
        CudaArrayDevicePointer sourcePositionCorrection, CudaArrayDevicePointer sourceCharges,
        int numQueryPositions, CudaArrayDevicePointer queryPositionBase,
        CudaArrayDevicePointer queryPositionCorrection, CudaArrayDevicePointer queryPotentials) {
    ContextSelector selector(cu);
    if (cu.getNumContexts() != 1)
        throw OpenMMException("Context-precision PME scratch potential queries currently require a single CUDA device.");
    if (nonbondedMethod != PME || pmePotentialContextPrecisionKernel == NULL ||
            !scratchPmeSpreadContextPrecisionKernel)
        throw OpenMMException("Context-precision PME scratch potential queries require GPU PME with PerturbNonbondedForce::PME.");
    if (numSourceParticles <= 0 || numQueryPositions < 0)
        throw OpenMMException("Context-precision PME scratch query counts must be positive for sources and nonnegative for queries.");
    if (numQueryPositions == 0)
        return;
    validateContextPrecisionPositionCorrection(cu, sourcePositionCorrection,
            "Context-precision PME scratch source positions");
    validateContextPrecisionPositionCorrection(cu, queryPositionCorrection,
            "Context-precision PME scratch query positions");
    if (usePmeQueue)
        pmeSyncEvent->queueWait(cu.getCurrentQueue());
    Vec3 boxVectors[3];
    cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    const double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
    const double scale = 1.0/determinant;
    mm_double4 reciprocal[3] = {
        mm_double4(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0, 0),
        mm_double4(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0, 0),
        mm_double4((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale,
                -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale, 0)
    };
    mm_float4 reciprocalFloat[3];
    for (int i = 0; i < 3; i++)
        reciprocalFloat[i] = mm_float4((float) reciprocal[i].x, (float) reciprocal[i].y,
                (float) reciprocal[i].z, 0);
    CUdeviceptr sourcePositionBasePointer = static_cast<CUdeviceptr>(sourcePositionBase);
    CUdeviceptr sourcePositionCorrectionPointer = static_cast<CUdeviceptr>(sourcePositionCorrection);
    CUdeviceptr sourceChargePointer = static_cast<CUdeviceptr>(sourceCharges);
    CUdeviceptr queryPositionBasePointer = static_cast<CUdeviceptr>(queryPositionBase);
    CUdeviceptr queryPositionCorrectionPointer = static_cast<CUdeviceptr>(queryPositionCorrection);
    CUdeviceptr queryPotentialPointer = static_cast<CUdeviceptr>(queryPotentials);
    cu.clearBuffer(scratchPmeGrid1);
    cu.clearBuffer(scratchPmeGrid2);
    void* spreadArgs[] = {&numSourceParticles, &sourcePositionBasePointer,
            &sourcePositionCorrectionPointer, &sourceChargePointer,
            &scratchPmeGrid1.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
            cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
            cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            cu.getUseDoublePrecision() ? (void*) &reciprocal[0] : (void*) &reciprocalFloat[0],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[1] : (void*) &reciprocalFloat[1],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[2] : (void*) &reciprocalFloat[2]};
    cu.executeKernel(scratchPmeSpreadContextPrecisionKernel, spreadArgs, numSourceParticles);
    fft->execFFT(scratchPmeGrid1, scratchPmeGrid2, true);
    if (cu.getUseDoublePrecision()) {
        scratchPmeConvolutionKernel->setArg<mm_double4>(4, reciprocal[0]);
        scratchPmeConvolutionKernel->setArg<mm_double4>(5, reciprocal[1]);
        scratchPmeConvolutionKernel->setArg<mm_double4>(6, reciprocal[2]);
    }
    else {
        scratchPmeConvolutionKernel->setArg<mm_float4>(4, reciprocalFloat[0]);
        scratchPmeConvolutionKernel->setArg<mm_float4>(5, reciprocalFloat[1]);
        scratchPmeConvolutionKernel->setArg<mm_float4>(6, reciprocalFloat[2]);
    }
    scratchPmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
    fft->execFFT(scratchPmeGrid2, scratchPmeGrid1, false);
    void* gatherArgs[] = {&numQueryPositions, &scratchPmeGrid1.getDevicePointer(),
            &queryPositionBasePointer, &queryPositionCorrectionPointer, &queryPotentialPointer,
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(),
            cu.getPeriodicBoxVecZPointer(),
            cu.getUseDoublePrecision() ? (void*) &reciprocal[0] : (void*) &reciprocalFloat[0],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[1] : (void*) &reciprocalFloat[1],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[2] : (void*) &reciprocalFloat[2]};
    cu.executeKernel(pmePotentialContextPrecisionKernel, gatherArgs, numQueryPositions);
}

void CudaCalcPerturbNonbondedForceKernel::getPMECandidateSelfEnergiesCuda(
        ContextImpl& context, int numCandidates, int numAtomsPerCandidate,
        CudaArrayDevicePointer candidatePositions, CudaArrayDevicePointer candidateCharges,
        CudaArrayDevicePointer energies) {
    if (numCandidates < 0 || numAtomsPerCandidate <= 0)
        throw OpenMMException("PME candidate self energy counts are invalid.");
    if (numCandidates == 0)
        return;
    if (numAtomsPerCandidate > scratchPmePotentials.getSize())
        throw OpenMMException("A PME candidate contains more atoms than the Context.");
    const CUdeviceptr basePositionPointer = static_cast<CUdeviceptr>(candidatePositions);
    const CUdeviceptr baseChargePointer = static_cast<CUdeviceptr>(candidateCharges);
    const CUdeviceptr baseEnergyPointer = static_cast<CUdeviceptr>(energies);
    for (int candidate = 0; candidate < numCandidates; candidate++) {
        CUdeviceptr positionPointer = basePositionPointer+
                (CUdeviceptr) candidate*numAtomsPerCandidate*sizeof(mm_float4);
        CUdeviceptr chargePointer = baseChargePointer+
                (CUdeviceptr) candidate*numAtomsPerCandidate*sizeof(float);
        getPMEPotentialFromParticlesCuda(context, numAtomsPerCandidate,
                (CudaArrayDevicePointer) positionPointer, (CudaArrayDevicePointer) chargePointer,
                numAtomsPerCandidate, (CudaArrayDevicePointer) positionPointer,
                (CudaArrayDevicePointer) scratchPmePotentials.getDevicePointer());
        CUdeviceptr outputPointer = baseEnergyPointer+(CUdeviceptr) candidate*sizeof(float);
        void* reduceArgs[] = {&numAtomsPerCandidate, &chargePointer,
                &scratchPmePotentials.getDevicePointer(), &outputPointer};
        cu.executeKernel(reduceCandidateSelfEnergyKernel, reduceArgs, 1, 1);
    }
}

void CudaCalcPerturbNonbondedForceKernel::getPMECandidateSelfEnergiesCudaContextPrecision(
        ContextImpl& context, int numCandidates, int numAtomsPerCandidate,
        CudaArrayDevicePointer candidatePositionBase,
        CudaArrayDevicePointer candidatePositionCorrection,
        CudaArrayDevicePointer candidateCharges, CudaArrayDevicePointer energies) {
    if (numCandidates < 0 || numAtomsPerCandidate <= 0)
        throw OpenMMException("Context-precision PME candidate self energy counts are invalid.");
    if (numCandidates == 0)
        return;
    validateContextPrecisionPositionCorrection(cu, candidatePositionCorrection,
            "Context-precision PME candidate self queries");
    if (numAtomsPerCandidate > scratchPmePotentials.getSize())
        throw OpenMMException("A context-precision PME candidate contains more atoms than the Context.");
    const CUdeviceptr basePositionPointer = static_cast<CUdeviceptr>(candidatePositionBase);
    const CUdeviceptr baseCorrectionPointer = static_cast<CUdeviceptr>(candidatePositionCorrection);
    const CUdeviceptr baseChargePointer = static_cast<CUdeviceptr>(candidateCharges);
    const CUdeviceptr baseEnergyPointer = static_cast<CUdeviceptr>(energies);
    const int positionElementSize = cu.getUseDoublePrecision() ? sizeof(mm_double4) : sizeof(mm_float4);
    for (int candidate = 0; candidate < numCandidates; candidate++) {
        CUdeviceptr positionPointer = basePositionPointer+
                (CUdeviceptr) candidate*numAtomsPerCandidate*positionElementSize;
        CUdeviceptr correctionPointer = cu.getUseMixedPrecision() ? baseCorrectionPointer+
                (CUdeviceptr) candidate*numAtomsPerCandidate*sizeof(mm_float4) : 0;
        CUdeviceptr chargePointer = baseChargePointer+
                (CUdeviceptr) candidate*numAtomsPerCandidate*sizeof(float);
        getPMEPotentialFromParticlesCudaContextPrecision(context, numAtomsPerCandidate,
                (CudaArrayDevicePointer) positionPointer, (CudaArrayDevicePointer) correctionPointer,
                (CudaArrayDevicePointer) chargePointer, numAtomsPerCandidate,
                (CudaArrayDevicePointer) positionPointer, (CudaArrayDevicePointer) correctionPointer,
                (CudaArrayDevicePointer) scratchPmePotentials.getDevicePointer());
        CUdeviceptr outputPointer = baseEnergyPointer+(CUdeviceptr) candidate*sizeof(float);
        void* reduceArgs[] = {&numAtomsPerCandidate, &chargePointer,
                &scratchPmePotentials.getDevicePointer(), &outputPointer};
        cu.executeKernel(reduceCandidateSelfEnergyKernel, reduceArgs, 1, 1);
    }
}

void CudaCalcPerturbNonbondedForceKernel::getSymmetricPMEReciprocalAtomicEnergiesCuda(
        ContextImpl&, CudaArrayDevicePointer energies, CudaArrayDevicePointer sum) {
    ContextSelector selector(cu);
    if (cu.getNumContexts() != 1)
        throw OpenMMException("PME reciprocal atomic energy queries currently require a single CUDA device.");
    if (nonbondedMethod != PME || pmeAtomicEnergyKernel == NULL)
        throw OpenMMException("PME reciprocal atomic energy queries require GPU PME with PerturbNonbondedForce::PME.");
    if (!hasInitializedKernel)
        throw OpenMMException("Evaluate the Context energy once before requesting PME reciprocal atomic energies.");
    if (usePmeQueue)
        pmeSyncEvent->queueWait(cu.getCurrentQueue());
    Vec3 boxVectors[3];
    cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    const double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
    const double scale = 1.0/determinant;
    mm_double4 reciprocal[3] = {
        mm_double4(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0, 0),
        mm_double4(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0, 0),
        mm_double4((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale,
                -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale, 0)
    };
    mm_float4 reciprocalFloat[3];
    for (int i = 0; i < 3; i++)
        reciprocalFloat[i] = mm_float4((float) reciprocal[i].x, (float) reciprocal[i].y,
                (float) reciprocal[i].z, 0);
    CUdeviceptr energyPointer = static_cast<CUdeviceptr>(energies);
    CUdeviceptr sumPointer = static_cast<CUdeviceptr>(sum);
    int numAtoms = cu.getNumAtoms();
    void* args[] = {&numAtoms, &pmeGrid1.getDevicePointer(), &cu.getPosq().getDevicePointer(),
            &charges.getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &energyPointer,
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            cu.getUseDoublePrecision() ? (void*) &reciprocal[0] : (void*) &reciprocalFloat[0],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[1] : (void*) &reciprocalFloat[1],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[2] : (void*) &reciprocalFloat[2]};
    cu.executeKernel(pmeAtomicEnergyKernel, args, numAtoms);
    void* sumArgs[] = {&numAtoms, &energyPointer, &sumPointer};
    cu.executeKernel(sumMixedValuesKernel, sumArgs, 1, 1);
}

void CudaCalcPerturbNonbondedForceKernel::getPMEPotentialAtPositionsCuda(
        ContextImpl&, int numPositions, CudaArrayDevicePointer positions,
        CudaArrayDevicePointer potentials) {
    ContextSelector selector(cu);
    if (cu.getNumContexts() != 1)
        throw OpenMMException("PME potential queries currently require a single CUDA device.");
    if (nonbondedMethod != PME || pmePotentialKernel == NULL)
        throw OpenMMException("PME potential queries require GPU PME with PerturbNonbondedForce::PME.");
    if (!hasInitializedKernel)
        throw OpenMMException("Evaluate the Context energy once before requesting the PME potential.");
    if (usePmeQueue)
        pmeSyncEvent->queueWait(cu.getCurrentQueue());
    if (numPositions < 0)
        throw OpenMMException("The PME potential query count cannot be negative.");
    if (numPositions == 0)
        return;
    Vec3 boxVectors[3];
    cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    const double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
    const double scale = 1.0/determinant;
    mm_double4 reciprocal[3] = {
        mm_double4(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0, 0),
        mm_double4(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0, 0),
        mm_double4((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale,
                -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale, 0)
    };
    mm_float4 reciprocalFloat[3];
    for (int i = 0; i < 3; i++)
        reciprocalFloat[i] = mm_float4((float) reciprocal[i].x, (float) reciprocal[i].y,
                (float) reciprocal[i].z, 0);
    CUdeviceptr positionPointer = static_cast<CUdeviceptr>(positions);
    CUdeviceptr potentialPointer = static_cast<CUdeviceptr>(potentials);
    void* args[] = {&numPositions, &pmeGrid1.getDevicePointer(), &positionPointer, &potentialPointer,
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
            cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            cu.getUseDoublePrecision() ? (void*) &reciprocal[0] : (void*) &reciprocalFloat[0],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[1] : (void*) &reciprocalFloat[1],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[2] : (void*) &reciprocalFloat[2]};
    cu.executeKernel(pmePotentialKernel, args, numPositions);
}

void CudaCalcPerturbNonbondedForceKernel::getPMEPotentialAtPositionsCudaContextPrecision(
        ContextImpl&, int numPositions, CudaArrayDevicePointer positionBase,
        CudaArrayDevicePointer positionCorrection, CudaArrayDevicePointer potentials) {
    ContextSelector selector(cu);
    if (cu.getNumContexts() != 1)
        throw OpenMMException("Context-precision PME potential queries currently require a single CUDA device.");
    if (nonbondedMethod != PME || pmePotentialContextPrecisionKernel == NULL)
        throw OpenMMException("Context-precision PME potential queries require GPU PME with PerturbNonbondedForce::PME.");
    if (!hasInitializedKernel)
        throw OpenMMException("Evaluate the Context energy once before requesting the context-precision PME potential.");
    if (usePmeQueue)
        pmeSyncEvent->queueWait(cu.getCurrentQueue());
    if (numPositions < 0)
        throw OpenMMException("The context-precision PME potential query count cannot be negative.");
    if (numPositions == 0)
        return;
    validateContextPrecisionPositionCorrection(cu, positionCorrection,
            "Context-precision PME potential queries");
    Vec3 boxVectors[3];
    cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    const double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
    const double scale = 1.0/determinant;
    mm_double4 reciprocal[3] = {
        mm_double4(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0, 0),
        mm_double4(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0, 0),
        mm_double4((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale,
                -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale, 0)
    };
    mm_float4 reciprocalFloat[3];
    for (int i = 0; i < 3; i++)
        reciprocalFloat[i] = mm_float4((float) reciprocal[i].x, (float) reciprocal[i].y,
                (float) reciprocal[i].z, 0);
    CUdeviceptr positionBasePointer = static_cast<CUdeviceptr>(positionBase);
    CUdeviceptr positionCorrectionPointer = static_cast<CUdeviceptr>(positionCorrection);
    CUdeviceptr potentialPointer = static_cast<CUdeviceptr>(potentials);
    void* args[] = {&numPositions, &pmeGrid1.getDevicePointer(), &positionBasePointer,
            &positionCorrectionPointer, &potentialPointer, cu.getPeriodicBoxSizePointer(),
            cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
            cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            cu.getUseDoublePrecision() ? (void*) &reciprocal[0] : (void*) &reciprocalFloat[0],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[1] : (void*) &reciprocalFloat[1],
            cu.getUseDoublePrecision() ? (void*) &reciprocal[2] : (void*) &reciprocalFloat[2]};
    cu.executeKernel(pmePotentialContextPrecisionKernel, args, numPositions);
}

vector<double> CudaCalcPerturbNonbondedForceKernel::getPerturbDispersionCorrections() {
    if (nonbondedMethod == NoCutoff || nonbondedMethod == CutoffNonPeriodic){
        return vector<double>({0.0, 0.0});
    }

    if (!usePerturbDispersionCorrection)
        throw OpenMMException("PerturbNonbondedForce: Dispersion correction requested, but not supported.");
    
    // Record sigma and epsilon for every particle, including the default value
    // for every offset parameter.
    vector<double> sigmaA(numParticles), epsilonA(numParticles);
    vector<double> sigmaB(numParticles), epsilonB(numParticles);
    for (int i = 0; i < numParticles; i++) {
        sigmaA[i] = perturbSigmaEpsilonParams_cpu[i][0];
        epsilonA[i] = perturbSigmaEpsilonParams_cpu[i][1];
        sigmaB[i] = perturbSigmaEpsilonParams_cpu[i][2];
        epsilonB[i] = perturbSigmaEpsilonParams_cpu[i][3];
        
        bool isREST2 = perturbAtomTypesParams_cpu[i][1];
        
        if (isREST2){
            epsilonA[i] *= rest2_scale;
            epsilonB[i] *= rest2_scale;
        }
    }

    double EA = this->calcDispersionCorrection(sigmaA, epsilonA);
    double EB = this->calcDispersionCorrection(sigmaB, epsilonB);
    return vector<double>({EA, EB});
}

double CudaCalcPerturbNonbondedForceKernel::calcDispersionCorrection(vector<double> sigmaList, vector<double> epsilonList) {
    
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
            sum3 += count*epsilon*(evalIntegral(cutoff, switchDist, cutoff, sigma)-evalIntegral(switchDist, switchDist, cutoff, sigma));
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
                sum3 += count*epsilon*(evalIntegral(cutoff, switchDist, cutoff, sigma)-evalIntegral(switchDist, switchDist, cutoff, sigma));
        }
    double numParticles_64_bit = (double) numParticles;
    double numInteractions = (numParticles_64_bit*(numParticles_64_bit+1))/2;
    sum1 /= numInteractions;
    sum2 /= numInteractions;
    sum3 /= numInteractions;
    double raw_dispersion = sum1/(9*pow(cutoff, 9))-sum2/(3*pow(cutoff, 3))+sum3;
    return 8 * numParticles_64_bit * numParticles_64_bit * M_PI * raw_dispersion;
}


double CudaCalcPerturbNonbondedForceKernel::calcCachedEwaldSelfEnergy(
    std::vector<std::array<double, 3>> chargesParams,
    std::vector<std::array<bool, 2>> particleTypes, 
    double lambdaA, double lambdaB, double lambdaC, double rest2_scale) {
    double sqrt_rest2 = sqrt(rest2_scale);
    double ewaldSelfEnergy = 0.0;
    double ewaldScale = ONE_4PI_EPS0 * alpha / sqrt(M_PI);
    for (int ii = 0; ii < numParticles; ii++){
        double charge = lambdaA*chargesParams[ii][0]+lambdaB*chargesParams[ii][1]+lambdaC*chargesParams[ii][2];
        if (particleTypes[ii][1]) // REST2 particle
            charge *= sqrt_rest2;  
        ewaldSelfEnergy -= ewaldScale * charge * charge;
    }
    return ewaldSelfEnergy;
}

double CudaCalcPerturbNonbondedForceKernel::calcCachedPerturbDispersionCorrection(
    std::vector<std::array<double, 4>> sigmaEpsilonParams,
    std::vector<std::array<bool, 2>> particleTypes,
    double rest2_scale, double lambdaA, double lambdaB) {
    if (nonbondedMethod == NoCutoff || nonbondedMethod == CutoffNonPeriodic){
        return 0.0;
    }

    if (!usePerturbDispersionCorrection)
        throw OpenMMException("PerturbNonbondedForce: Dispersion correction requested, but not supported.");

    vector<double> EcorrAB = this->getPerturbDispersionCorrections();
    double EA = EcorrAB[0];
    double EB = EcorrAB[1];
    return (lambdaA*EA+lambdaB*EB) / (lambdaA + lambdaB);
}
