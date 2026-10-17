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

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/Messages.h"
#include "internal/PerturbNonbondedForceImpl.h"
#include "PerturbNonbondedKernels.h"
#include "openmm/kernels.h"
#include <cmath>
#include <map>
#include <sstream>
#include <algorithm>

using namespace OpenMM;
using namespace std;
using namespace PerturbNonbondedPlugin;

PerturbNonbondedForceImpl::PerturbNonbondedForceImpl(const PerturbNonbondedForce& owner) : owner(owner) {
    forceGroup = owner.getForceGroup();
    recipForceGroup = owner.getReciprocalSpaceForceGroup();
    if (recipForceGroup < 0)
        recipForceGroup = owner.getForceGroup();
    includeDirectSpace = owner.getIncludeDirectSpace();
}

PerturbNonbondedForceImpl::~PerturbNonbondedForceImpl() {
}

void PerturbNonbondedForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcPerturbNonbondedForceKernel::Name(), context);

    // Check for errors in the specification of exceptions.

    const OpenMM::System& system = context.getSystem();
    if (owner.getNumParticles() != system.getNumParticles())
        throw OpenMMException("PerturbNonbondedForce must have exactly as many particles as the System it belongs to.");
    if (owner.getUseSwitchingFunction()) {
        if (owner.getSwitchingDistance() < 0 || owner.getSwitchingDistance() >= owner.getCutoffDistance())
            throw OpenMMException("PerturbNonbondedForce: Switching distance must satisfy 0 <= r_switch < r_cutoff");
    }
    for (int i = 0; i < owner.getNumParticles(); i++) {
        double chargeA, sigmaA, epsilonA;
        double chargeB, sigmaB, epsilonB;
        bool isDummy, isREST2;
        owner.getParticleParameters(i, chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2);
        if (sigmaA < 0 || sigmaB < 0)
            throw OpenMMException("PerturbNonbondedForce: sigma for a particle cannot be negative");
        if (epsilonA < 0 || epsilonB < 0)
            throw OpenMMException("PerturbNonbondedForce: epsilon for a particle cannot be negative");
    }
    vector<set<int> > exceptions(owner.getNumParticles());
    for (int i = 0; i < owner.getNumExceptions(); i++) {
        int particle[2];
        double chargeProd, sigma, epsilon;
        owner.getExceptionParameters(i, particle[0], particle[1], chargeProd, sigma, epsilon);
        int minp = min(particle[0], particle[1]);
        int maxp = max(particle[0], particle[1]);
        for (int j = 0; j < 2; j++) {
            if (particle[j] < 0 || particle[j] >= owner.getNumParticles()) {
                stringstream msg;
                msg << "PerturbNonbondedForce: Illegal particle index for an exception: ";
                msg << particle[j];
                throw OpenMMException(msg.str());
            }
        }
        if (exceptions[minp].count(maxp) > 0) {
            stringstream msg;
            msg << "PerturbNonbondedForce: Multiple exceptions are specified for particles ";
            msg << particle[0];
            msg << " and ";
            msg << particle[1];
            throw OpenMMException(msg.str());
        }
        exceptions[minp].insert(maxp);
        if (sigma < 0)
            throw OpenMMException("PerturbNonbondedForce: sigma for an exception cannot be negative");
        if (epsilon < 0)
            throw OpenMMException("PerturbNonbondedForce: epsilon for an exception cannot be negative");
    }
    for (int i = 0; i < owner.getNumParticleParameterOffsets(); i++) {
        string parameter;
        int particleIndex;
        double chargeScale, sigmaScale, epsilonScale;
        owner.getParticleParameterOffset(i, parameter, particleIndex, chargeScale, sigmaScale, epsilonScale);
        if (particleIndex < 0 || particleIndex >= owner.getNumParticles()) {
            stringstream msg;
            msg << "PerturbNonbondedForce: Illegal particle index for a particle parameter offset: ";
            msg << particleIndex;
            throw OpenMMException(msg.str());
        }
    }
    for (int i = 0; i < owner.getNumExceptionParameterOffsets(); i++) {
        string parameter;
        int exceptionIndex;
        double chargeScale, sigmaScale, epsilonScale;
        owner.getExceptionParameterOffset(i, parameter, exceptionIndex, chargeScale, sigmaScale, epsilonScale);
        if (exceptionIndex < 0 || exceptionIndex >= owner.getNumExceptions()) {
            stringstream msg;
            msg << "PerturbNonbondedForce: Illegal exception index for an exception parameter offset: ";
            msg << exceptionIndex;
            throw OpenMMException(msg.str());
        }
    }
    if (owner.getNonbondedMethod() != PerturbNonbondedForce::NoCutoff && owner.getNonbondedMethod() != PerturbNonbondedForce::CutoffNonPeriodic) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("PerturbNonbondedForce: "+Messages::cutoffTooLarge);
        if (owner.getNonbondedMethod() == PerturbNonbondedForce::Ewald && (boxVectors[1][0] != 0.0 || boxVectors[2][0] != 0.0 || boxVectors[2][1] != 0))
            throw OpenMMException("PerturbNonbondedForce: Ewald is not supported with non-rectangular boxes.  Use PME instead.");
    }
    kernel.getAs<CalcPerturbNonbondedForceKernel>().initialize(context.getSystem(), owner);
}

double PerturbNonbondedForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    bool includeDirect = (includeDirectSpace && (groups&(1<<forceGroup)) != 0);
    bool includeReciprocal = ((groups&(1<<recipForceGroup)) != 0);
    includeEnergy = includeEnergy && ( includeDirect || includeReciprocal );
    return kernel.getAs<CalcPerturbNonbondedForceKernel>().execute(context, includeForces, includeEnergy, includeDirect, includeReciprocal);
}

map<string, double> PerturbNonbondedForceImpl::getDefaultParameters() {
    map<string, double> parameters;
    for (int i = 0; i < owner.getNumGlobalParameters(); i++)
        parameters[owner.getGlobalParameterName(i)] = owner.getGlobalParameterDefaultValue(i);
    return parameters;
}

std::vector<std::string> PerturbNonbondedForceImpl::getKernelNames() {
    return {CalcPerturbNonbondedForceKernel::Name()};
}

class PerturbNonbondedForceImpl::ErrorFunction {
public:
    virtual double getValue(int arg) const = 0;
};

class PerturbNonbondedForceImpl::EwaldErrorFunction : public ErrorFunction {
public:
    EwaldErrorFunction(double width, double alpha, double target) : width(width), alpha(alpha), target(target) {
    }
    double getValue(int arg) const {
        double temp = arg*M_PI/(width*alpha);
        return target-0.05*sqrt(width*alpha)*arg*exp(-temp*temp);
    }
private:
    double width, alpha, target;
};

void PerturbNonbondedForceImpl::calcEwaldParameters(const System& system, const PerturbNonbondedForce& force, double& alpha, int& kmaxx, int& kmaxy, int& kmaxz) {
    Vec3 boxVectors[3];
    system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    double tol = force.getEwaldErrorTolerance();
    alpha = (1.0/force.getCutoffDistance())*std::sqrt(-log(2.0*tol));
    kmaxx = findZero(EwaldErrorFunction(boxVectors[0][0], alpha, tol), 10);
    kmaxy = findZero(EwaldErrorFunction(boxVectors[1][1], alpha, tol), 10);
    kmaxz = findZero(EwaldErrorFunction(boxVectors[2][2], alpha, tol), 10);
    if (kmaxx%2 == 0)
        kmaxx++;
    if (kmaxy%2 == 0)
        kmaxy++;
    if (kmaxz%2 == 0)
        kmaxz++;
}

void PerturbNonbondedForceImpl::calcPMEParameters(const System& system, const PerturbNonbondedForce& force, double& alpha, int& xsize, int& ysize, int& zsize, bool lj) {
    if (lj)
        force.getLJPMEParameters(alpha, xsize, ysize, zsize);
    else
        force.getPMEParameters(alpha, xsize, ysize, zsize);
    if (alpha == 0.0) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double tol = force.getEwaldErrorTolerance();
        alpha = (1.0/force.getCutoffDistance())*std::sqrt(-log(2.0*tol));
        if (lj) {
            xsize = (int) ceil(alpha*boxVectors[0][0]/(3*pow(tol, 0.2)));
            ysize = (int) ceil(alpha*boxVectors[1][1]/(3*pow(tol, 0.2)));
            zsize = (int) ceil(alpha*boxVectors[2][2]/(3*pow(tol, 0.2)));
        }
        else {
            xsize = (int) ceil(2*alpha*boxVectors[0][0]/(3*pow(tol, 0.2)));
            ysize = (int) ceil(2*alpha*boxVectors[1][1]/(3*pow(tol, 0.2)));
            zsize = (int) ceil(2*alpha*boxVectors[2][2]/(3*pow(tol, 0.2)));
        }
        xsize = max(xsize, 6);
        ysize = max(ysize, 6);
        zsize = max(zsize, 6);
    }
}

int PerturbNonbondedForceImpl::findZero(const PerturbNonbondedForceImpl::ErrorFunction& f, int initialGuess) {
    int arg = initialGuess;
    double value = f.getValue(arg);
    if (value > 0.0) {
        while (value > 0.0 && arg > 0)
            value = f.getValue(--arg);
        return arg+1;
    }
    while (value < 0.0)
        value = f.getValue(++arg);
    return arg;
}

double PerturbNonbondedForceImpl::evalIntegral(double r, double rs, double rc, double sigma) {
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

double PerturbNonbondedForceImpl::calcDispersionCorrection(const System& system, const PerturbNonbondedForce& force) {
    throw OpenMMException("calcDispersionCorrection is deprecated in PerturbNonbondedForceImpl.");
}

void PerturbNonbondedForceImpl::updateParametersInContext(ContextImpl& context, int firstParticle, int lastParticle, int firstException, int lastException) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().copyParametersToContext(context, owner, firstParticle, lastParticle, firstException, lastException);
    context.systemChanged();
}

void PerturbNonbondedForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}

void PerturbNonbondedForceImpl::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getLJPMEParameters(alpha, nx, ny, nz);
}

void PerturbNonbondedForceImpl::getInteractionEnergyForRealParticlesCuda(ContextImpl& context,
        int numRealAtoms, CudaArrayDevicePointer positions, CudaArrayDevicePointer parameters,
        CudaArrayDevicePointer excludedAtomStarts, CudaArrayDevicePointer excludedAtomCounts,
        CudaArrayDevicePointer excludedAtoms, CudaArrayDevicePointer forces,
        CudaArrayDevicePointer energies) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getInteractionEnergyForRealParticlesCuda(context,
            numRealAtoms, positions, parameters, excludedAtomStarts, excludedAtomCounts,
            excludedAtoms, forces, energies);
}

void PerturbNonbondedForceImpl::getDirectEnergyForCandidateParticlesCuda(ContextImpl& context,
        int numCandidateAtoms, CudaArrayDevicePointer positions, CudaArrayDevicePointer parameters,
        CudaArrayDevicePointer forces, CudaArrayDevicePointer energies) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getDirectEnergyForCandidateParticlesCuda(context,
            numCandidateAtoms, positions, parameters, forces, energies);
}

void PerturbNonbondedForceImpl::getDirectEnergyForCandidateParticlesCudaContextPrecision(
        ContextImpl& context, int numCandidateAtoms, CudaArrayDevicePointer positionBase,
        CudaArrayDevicePointer positionCorrection, CudaArrayDevicePointer parameters,
        CudaArrayDevicePointer forces, CudaArrayDevicePointer energies) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getDirectEnergyForCandidateParticlesCudaContextPrecision(
            context, numCandidateAtoms, positionBase, positionCorrection, parameters, forces, energies);
}

void PerturbNonbondedForceImpl::getPMEPotentialAtPositionsCuda(ContextImpl& context, int numPositions,
        CudaArrayDevicePointer positions, CudaArrayDevicePointer potentials) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getPMEPotentialAtPositionsCuda(context,
            numPositions, positions, potentials);
}

void PerturbNonbondedForceImpl::getPMEPotentialAtPositionsCudaContextPrecision(
        ContextImpl& context, int numPositions, CudaArrayDevicePointer positionBase,
        CudaArrayDevicePointer positionCorrection, CudaArrayDevicePointer potentials) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getPMEPotentialAtPositionsCudaContextPrecision(
            context, numPositions, positionBase, positionCorrection, potentials);
}

void PerturbNonbondedForceImpl::getSymmetricPMEReciprocalAtomicEnergiesCuda(ContextImpl& context,
        CudaArrayDevicePointer energies, CudaArrayDevicePointer sum) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getSymmetricPMEReciprocalAtomicEnergiesCuda(context, energies, sum);
}

void PerturbNonbondedForceImpl::getPMEPotentialFromParticlesCuda(ContextImpl& context,
        int numSourceParticles, CudaArrayDevicePointer sourcePositions,
        CudaArrayDevicePointer sourceCharges, int numQueryPositions,
        CudaArrayDevicePointer queryPositions, CudaArrayDevicePointer queryPotentials) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getPMEPotentialFromParticlesCuda(
            context, numSourceParticles, sourcePositions, sourceCharges,
            numQueryPositions, queryPositions, queryPotentials);
}

void PerturbNonbondedForceImpl::getPMEPotentialFromParticlesCudaContextPrecision(
        ContextImpl& context, int numSourceParticles,
        CudaArrayDevicePointer sourcePositionBase,
        CudaArrayDevicePointer sourcePositionCorrection,
        CudaArrayDevicePointer sourceCharges, int numQueryPositions,
        CudaArrayDevicePointer queryPositionBase,
        CudaArrayDevicePointer queryPositionCorrection,
        CudaArrayDevicePointer queryPotentials) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getPMEPotentialFromParticlesCudaContextPrecision(
            context, numSourceParticles, sourcePositionBase, sourcePositionCorrection,
            sourceCharges, numQueryPositions, queryPositionBase,
            queryPositionCorrection, queryPotentials);
}

void PerturbNonbondedForceImpl::getPMECandidateSelfEnergiesCuda(ContextImpl& context,
        int numCandidates, int numAtomsPerCandidate,
        CudaArrayDevicePointer candidatePositions, CudaArrayDevicePointer candidateCharges,
        CudaArrayDevicePointer energies) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getPMECandidateSelfEnergiesCuda(
            context, numCandidates, numAtomsPerCandidate, candidatePositions, candidateCharges, energies);
}

void PerturbNonbondedForceImpl::getPMECandidateSelfEnergiesCudaContextPrecision(
        ContextImpl& context, int numCandidates, int numAtomsPerCandidate,
        CudaArrayDevicePointer candidatePositionBase,
        CudaArrayDevicePointer candidatePositionCorrection,
        CudaArrayDevicePointer candidateCharges, CudaArrayDevicePointer energies) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getPMECandidateSelfEnergiesCudaContextPrecision(
            context, numCandidates, numAtomsPerCandidate, candidatePositionBase,
            candidatePositionCorrection, candidateCharges, energies);
}

void PerturbNonbondedForceImpl::getDirectEnergyForCandidateSubsetCuda(ContextImpl& context,
        int numCandidateAtoms, CudaArrayDevicePointer candidatePositions,
        CudaArrayDevicePointer candidateParameters, int numSourceAtoms,
        CudaArrayDevicePointer sourceAtomIndices, CudaArrayDevicePointer energies) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getDirectEnergyForCandidateSubsetCuda(
            context, numCandidateAtoms, candidatePositions, candidateParameters,
            numSourceAtoms, sourceAtomIndices, energies);
}

void PerturbNonbondedForceImpl::getDirectEnergyForCandidateSubsetCudaContextPrecision(
        ContextImpl& context, int numCandidateAtoms,
        CudaArrayDevicePointer candidatePositionBase,
        CudaArrayDevicePointer candidatePositionCorrection,
        CudaArrayDevicePointer candidateParameters, int numSourceAtoms,
        CudaArrayDevicePointer sourceAtomIndices, CudaArrayDevicePointer energies) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getDirectEnergyForCandidateSubsetCudaContextPrecision(
            context, numCandidateAtoms, candidatePositionBase, candidatePositionCorrection,
            candidateParameters, numSourceAtoms, sourceAtomIndices, energies);
}

void PerturbNonbondedForceImpl::getDirectEnergyForRealParticlesPmeCuda(ContextImpl& context,
        int numRealAtoms, CudaArrayDevicePointer realAtomIndices,
        int numAtomsPerMolecule, CudaArrayDevicePointer energies) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().getDirectEnergyForRealParticlesPmeCuda(
            context, numRealAtoms, realAtomIndices, numAtomsPerMolecule, energies);
}

void PerturbNonbondedForceImpl::setParticleQueryBlockPruning(bool enabled) {
    kernel.getAs<CalcPerturbNonbondedForceKernel>().setParticleQueryBlockPruning(enabled);
}
