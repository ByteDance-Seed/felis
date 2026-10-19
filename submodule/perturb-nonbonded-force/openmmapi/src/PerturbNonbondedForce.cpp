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

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "PerturbNonbondedForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "internal/PerturbNonbondedForceImpl.h"
#include <cmath>
#include <map>
#include <sstream>
#include <utility>
#include <iostream>

using namespace OpenMM;
using namespace std;
using namespace PerturbNonbondedPlugin;

PerturbNonbondedForce::PerturbNonbondedForce() : nonbondedMethod(NoCutoff), cutoffDistance(1.0), switchingDistance(-1.0), rfDielectric(78.3),
        ewaldErrorTol(5e-4), alpha(0.0), dalpha(0.0), useSwitchingFunction(false), useDispersionCorrection(true), exceptionsUsePeriodic(false), recipForceGroup(-1),
        includeDirectSpace(true), nx(0), ny(0), nz(0), dnx(0), dny(0), dnz(0), numContexts(0) {
}

PerturbNonbondedForce::PerturbNonbondedMethod PerturbNonbondedForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void PerturbNonbondedForce::setNonbondedMethod(PerturbNonbondedMethod method) {
    if (method < 0 || method > 5)
        throw OpenMMException("PerturbNonbondedForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

double PerturbNonbondedForce::getCutoffDistance() const {
    return cutoffDistance;
}

void PerturbNonbondedForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

bool PerturbNonbondedForce::getUseSwitchingFunction() const {
    return useSwitchingFunction;
}

void PerturbNonbondedForce::setUseSwitchingFunction(bool use) {
    if (use){
        throw OpenMMException("PerturbNonbondedForce: LJ with Switching Function is not supported.");
    } //    useSwitchingFunction = use; 
}

double PerturbNonbondedForce::getSwitchingDistance() const {
    return switchingDistance;
}

void PerturbNonbondedForce::setSwitchingDistance(double distance) {
    switchingDistance = distance;
}

double PerturbNonbondedForce::getReactionFieldDielectric() const {
    return rfDielectric;
}

void PerturbNonbondedForce::setReactionFieldDielectric(double dielectric) {
    rfDielectric = dielectric;
}

double PerturbNonbondedForce::getEwaldErrorTolerance() const {
    return ewaldErrorTol;
}

void PerturbNonbondedForce::setEwaldErrorTolerance(double tol) {
    ewaldErrorTol = tol;
}

void PerturbNonbondedForce::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = this->nx;
    ny = this->ny;
    nz = this->nz;
}

void PerturbNonbondedForce::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->dalpha;
    nx = this->dnx;
    ny = this->dny;
    nz = this->dnz;
}

void PerturbNonbondedForce::setPMEParameters(double alpha, int nx, int ny, int nz) {
    this->alpha = alpha;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
}

void PerturbNonbondedForce::setLJPMEParameters(double alpha, int nx, int ny, int nz) {
    this->dalpha = alpha;
    this->dnx = nx;
    this->dny = ny;
    this->dnz = nz;
}

void PerturbNonbondedForce::getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const PerturbNonbondedForceImpl&>(getImplInContext(context)).getPMEParameters(alpha, nx, ny, nz);
}

void PerturbNonbondedForce::getLJPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const PerturbNonbondedForceImpl&>(getImplInContext(context)).getLJPMEParameters(alpha, nx, ny, nz);
}

int PerturbNonbondedForce::addParticle(double chargeA, double sigmaA, double epsilonA, double chargeB, double sigmaB, double epsilonB, bool isDummy, bool isRest2) {
    particles.push_back(ParticleInfo(chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isRest2));
    return particles.size()-1;
}

void PerturbNonbondedForce::getParticleParameters(int index, double& chargeA, double& sigmaA, double& epsilonA, double& chargeB, double& sigmaB, double& epsilonB, bool& isDummy, bool& isRest2) const {
    ASSERT_VALID_INDEX(index, particles);
    chargeA = particles[index].chargeA;
    sigmaA = particles[index].sigmaA;
    epsilonA = particles[index].epsilonA;
    chargeB = particles[index].chargeB;
    sigmaB = particles[index].sigmaB;
    epsilonB = particles[index].epsilonB;
    isDummy = particles[index].isDummy;
    isRest2 = particles[index].isRest2;
}

void PerturbNonbondedForce::setParticleParameters(int index, double chargeA, double sigmaA, double epsilonA, double chargeB, double sigmaB, double epsilonB, bool isDummy, bool isRest2) {
    ASSERT_VALID_INDEX(index, particles);
    particles[index].chargeA = chargeA;
    particles[index].sigmaA = sigmaA;
    particles[index].epsilonA = epsilonA;
    particles[index].chargeB = chargeB;
    particles[index].sigmaB = sigmaB;
    particles[index].epsilonB = epsilonB;
    particles[index].isDummy = isDummy;
    particles[index].isRest2 = isRest2;
    
    if (numContexts > 0) {
        firstChangedParticle = min(index, firstChangedParticle);
        lastChangedParticle = max(index, lastChangedParticle);
        //throw OpenMMException("PerturbNonbondedForce: setParticleParameters is not implemented for multiple contexts");
    }
}

int PerturbNonbondedForce::addException(int particle1, int particle2, double chargeProd, double sigma, double epsilon, bool replace) {
    map<pair<int, int>, int>::iterator iter = exceptionMap.find(pair<int, int>(particle1, particle2));
    if (chargeProd != 0.0){
        throw OpenMMException("PerturbNonbondedForce: chargeProd non-zero exception not supported. Please used CustomBondForce for this purpose.");
    }
    if (epsilon != 0.0){
        throw OpenMMException("PerturbNonbondedForce: epsilon non-zero exception not supported. Please used CustomBondForce for this purpose.");
    }

    int newIndex;
    if (iter == exceptionMap.end())
        iter = exceptionMap.find(pair<int, int>(particle2, particle1));
    if (iter != exceptionMap.end()) {
        if (!replace) {
            stringstream msg;
            msg << "PerturbNonbondedForce: There is already an exception for particles ";
            msg << particle1;
            msg << " and ";
            msg << particle2;
            throw OpenMMException(msg.str());
        }
        exceptions[iter->second] = ExceptionInfo(particle1, particle2, chargeProd, sigma, epsilon);
        newIndex = iter->second;
        exceptionMap.erase(iter->first);
    }
    else {
        exceptions.push_back(ExceptionInfo(particle1, particle2, chargeProd, sigma, epsilon));
        newIndex = exceptions.size()-1;
    }
    exceptionMap[pair<int, int>(particle1, particle2)] = newIndex;
    return newIndex;
}
void PerturbNonbondedForce::getExceptionParameters(int index, int& particle1, int& particle2, double& chargeProd, double& sigma, double& epsilon) const {
    ASSERT_VALID_INDEX(index, exceptions);
    particle1 = exceptions[index].particle1;
    particle2 = exceptions[index].particle2;
    chargeProd = exceptions[index].chargeProd;
    sigma = exceptions[index].sigma;
    epsilon = exceptions[index].epsilon;
}

void PerturbNonbondedForce::setExceptionParameters(int index, int particle1, int particle2, double chargeProd, double sigma, double epsilon) {
    if (chargeProd != 0.0){
        throw OpenMMException("PerturbNonbondedForce: chargeProd non-zero exception not supported. Please used CustomBondForce for this purpose.");
    }
    if (epsilon != 0.0){
        throw OpenMMException("PerturbNonbondedForce: epsilon non-zero exception not supported. Please used CustomBondForce for this purpose.");
    }
    
    ASSERT_VALID_INDEX(index, exceptions);
    exceptions[index].particle1 = particle1;
    exceptions[index].particle2 = particle2;
    exceptions[index].chargeProd = chargeProd;
    exceptions[index].sigma = sigma;
    exceptions[index].epsilon = epsilon;
    if (numContexts > 0) {
        firstChangedException = min(index, firstChangedException);
        lastChangedException = max(index, lastChangedException);
    }}

ForceImpl* PerturbNonbondedForce::createImpl() const {
    if (numContexts == 0) {
        // Begin tracking changes to particles and exceptions.
        firstChangedParticle = particles.size();
        lastChangedParticle = -1;
        firstChangedException = exceptions.size();
        lastChangedException = -1;
    }
    numContexts++;
    return new PerturbNonbondedForceImpl(*this);
}

void PerturbNonbondedForce::createExceptionsFromBonds(const vector<pair<int, int> >& bonds, double coulomb14Scale, double lj14Scale) {
    for (auto& bond : bonds)
        if (bond.first < 0 || bond.second < 0 || bond.first >= particles.size() || bond.second >= particles.size())
            throw OpenMMException("createExceptionsFromBonds: Illegal particle index in list of bonds");

    // Find particles separated by 1, 2, or 3 bonds.

    vector<set<int> > exclusions(particles.size());
    vector<set<int> > bonded12(exclusions.size());
    for (auto& bond : bonds) {
        bonded12[bond.first].insert(bond.second);
        bonded12[bond.second].insert(bond.first);
    }
    for (int i = 0; i < (int) exclusions.size(); ++i)
        addExclusionsToSet(bonded12, exclusions[i], i, i, 2);

    // Find particles separated by 1 or 2 bonds and create the exceptions.

    for (int i = 0; i < (int) exclusions.size(); ++i) {
        set<int> bonded13;
        addExclusionsToSet(bonded12, bonded13, i, i, 1);
        for (int j : exclusions[i]) {
            if (j < i) {
                if (bonded13.find(j) == bonded13.end()) {
                    // This is a 1-4 interaction.

                    const ParticleInfo& particle1 = particles[j];
                    const ParticleInfo& particle2 = particles[i];
                    const double chargeProd = coulomb14Scale*particle1.chargeA*particle2.chargeA;
                    const double sigma = 0.5*(particle1.sigmaA+particle2.sigmaA);
                    const double epsilon = lj14Scale*std::sqrt(particle1.epsilonA*particle2.epsilonA);

                    if (chargeProd != 0.0 || epsilon != 0.0){
                        throw OpenMMException("PerturbNonbondedForce: chargeProd or epsilon non-zero exception not supported. Please used CustomBondForce for this purpose.");
                    }                  
                    addException(j, i, chargeProd, sigma, epsilon);
                }
                else {
                    // This interaction should be completely excluded.

                    addException(j, i, 0.0, 1.0, 0.0);
                }
            }
        }
    }
}

void PerturbNonbondedForce::addExclusionsToSet(const vector<set<int> >& bonded12, set<int>& exclusions, int baseParticle, int fromParticle, int currentLevel) const {
    for (int i : bonded12[fromParticle]) {
        if (i != baseParticle)
            exclusions.insert(i);
        if (currentLevel > 0)
            addExclusionsToSet(bonded12, exclusions, baseParticle, i, currentLevel-1);
    }
}

int PerturbNonbondedForce::addGlobalParameter(const string& name, double defaultValue) {
    globalParameters.push_back(GlobalParameterInfo(name, defaultValue));
    return globalParameters.size()-1;
}

const string& PerturbNonbondedForce::getGlobalParameterName(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].name;
}

void PerturbNonbondedForce::setGlobalParameterName(int index, const string& name) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].name = name;
}

double PerturbNonbondedForce::getGlobalParameterDefaultValue(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].defaultValue;
}

void PerturbNonbondedForce::setGlobalParameterDefaultValue(int index, double defaultValue) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].defaultValue = defaultValue;
}

int PerturbNonbondedForce::getGlobalParameterIndex(const std::string& parameter) const {
    for (int i = 0; i < globalParameters.size(); i++)
        if (globalParameters[i].name == parameter)
            return i;
    throw OpenMMException("There is no global parameter called '"+parameter+"'");
}

int PerturbNonbondedForce::addParticleParameterOffset(const std::string& parameter, int particleIndex, double chargeScale, double sigmaScale, double epsilonScale) {
    throw OpenMMException("PerturbNonbondedForce: addParticleParameterOffset is not implemented");
    particleOffsets.push_back(ParticleOffsetInfo(getGlobalParameterIndex(parameter), particleIndex, chargeScale, sigmaScale, epsilonScale));
    return particleOffsets.size()-1;
}

void PerturbNonbondedForce::getParticleParameterOffset(int index, std::string& parameter, int& particleIndex, double& chargeScale, double& sigmaScale, double& epsilonScale) const {
    throw OpenMMException("PerturbNonbondedForce: getParticleParameterOffset is not implemented");
    ASSERT_VALID_INDEX(index, particleOffsets);
    parameter = globalParameters[particleOffsets[index].parameter].name;
    particleIndex = particleOffsets[index].particle;
    chargeScale = particleOffsets[index].chargeScale;
    sigmaScale = particleOffsets[index].sigmaScale;
    epsilonScale = particleOffsets[index].epsilonScale;
}

void PerturbNonbondedForce::setParticleParameterOffset(int index, const std::string& parameter, int particleIndex, double chargeScale, double sigmaScale, double epsilonScale) {
    throw OpenMMException("PerturbNonbondedForce: setParticleParameterOffset is not implemented");
    ASSERT_VALID_INDEX(index, particleOffsets);
    particleOffsets[index].parameter = getGlobalParameterIndex(parameter);
    particleOffsets[index].particle = particleIndex;
    particleOffsets[index].chargeScale = chargeScale;
    particleOffsets[index].sigmaScale = sigmaScale;
    particleOffsets[index].epsilonScale = epsilonScale;
}

int PerturbNonbondedForce::addExceptionParameterOffset(const std::string& parameter, int exceptionIndex, double chargeProdScale, double sigmaScale, double epsilonScale) {
    throw OpenMMException("PerturbNonbondedForce: addExceptionParameterOffset is not implemented");
    exceptionOffsets.push_back(ExceptionOffsetInfo(getGlobalParameterIndex(parameter), exceptionIndex, chargeProdScale, sigmaScale, epsilonScale));
    return exceptionOffsets.size()-1;
}

void PerturbNonbondedForce::getExceptionParameterOffset(int index, std::string& parameter, int& exceptionIndex, double& chargeProdScale, double& sigmaScale, double& epsilonScale) const {
    throw OpenMMException("PerturbNonbondedForce: getExceptionParameterOffset is not implemented");
    ASSERT_VALID_INDEX(index, exceptionOffsets);
    parameter = globalParameters[exceptionOffsets[index].parameter].name;
    exceptionIndex = exceptionOffsets[index].exception;
    chargeProdScale = exceptionOffsets[index].chargeProdScale;
    sigmaScale = exceptionOffsets[index].sigmaScale;
    epsilonScale = exceptionOffsets[index].epsilonScale;
}

void PerturbNonbondedForce::setExceptionParameterOffset(int index, const std::string& parameter, int exceptionIndex, double chargeProdScale, double sigmaScale, double epsilonScale) {
    throw OpenMMException("PerturbNonbondedForce: setExceptionParameterOffset is not implemented");
    ASSERT_VALID_INDEX(index, exceptionOffsets);
    exceptionOffsets[index].parameter = getGlobalParameterIndex(parameter);
    exceptionOffsets[index].exception = exceptionIndex;
    exceptionOffsets[index].chargeProdScale = chargeProdScale;
    exceptionOffsets[index].sigmaScale = sigmaScale;
    exceptionOffsets[index].epsilonScale = epsilonScale;
}

int PerturbNonbondedForce::getReciprocalSpaceForceGroup() const {
    return recipForceGroup;
}

void PerturbNonbondedForce::setReciprocalSpaceForceGroup(int group) {
    if (group < -1 || group > 31)
        throw OpenMMException("Force group must be between -1 and 31");
    recipForceGroup = group;
}

bool PerturbNonbondedForce::getIncludeDirectSpace() const {
    return includeDirectSpace;
}

void PerturbNonbondedForce::setIncludeDirectSpace(bool include) {
    includeDirectSpace = include;
}

void PerturbNonbondedForce::updateParametersInContext(Context& context) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context),
            firstChangedParticle, lastChangedParticle, firstChangedException, lastChangedException);
    if (numContexts == 1) {
        // We just updated the only existing context for this force, so we can reset
        // the tracking of changed particles and exceptions.
        firstChangedParticle = particles.size();
        lastChangedParticle = -1;
        firstChangedException = exceptions.size();
        lastChangedException = -1;
    }
}

bool PerturbNonbondedForce::getExceptionsUsePeriodicBoundaryConditions() const {
    return exceptionsUsePeriodic;
}

void PerturbNonbondedForce::setExceptionsUsePeriodicBoundaryConditions(bool periodic) {
    exceptionsUsePeriodic = periodic;
}

int PerturbNonbondedForce::getNumParticles() const {
    return particles.size();
}

int PerturbNonbondedForce::getNumExceptions() const {
    return exceptions.size();
}

int PerturbNonbondedForce::getNumGlobalParameters() const {
    return globalParameters.size();
}

int PerturbNonbondedForce::getNumParticleParameterOffsets() const {
    return particleOffsets.size();
}

int PerturbNonbondedForce::getNumExceptionParameterOffsets() const {
    return exceptionOffsets.size();
}


void PerturbNonbondedForce::setSoftcoreAlpha(double alpha) {
    if (!std::isfinite(alpha) || alpha < 0.0)
        throw OpenMMException("PerturbNonbondedForce: softcore alpha must be finite and nonnegative");
    softcore_alpha = alpha;
}
double PerturbNonbondedForce::getSoftcoreAlpha() const {
    return softcore_alpha;
}

void PerturbNonbondedForce::setIncludeLJ(bool includeLJ) {
    this->includeLJ = includeLJ;
}

bool PerturbNonbondedForce::getIncludeLJ() const {
    return this->includeLJ;
}

void PerturbNonbondedForce::setLJForceGroup(int ljForceGroup) {
    if (ljForceGroup < -1 || ljForceGroup > 31)
        throw OpenMMException("PerturbNonbondedForce: LJ force group must be between -1 and 31");
    this->ljForceGroup = ljForceGroup;
}

int PerturbNonbondedForce::getLJForceGroup() const {
    return this->ljForceGroup;
}

void PerturbNonbondedForce::setDirectSpaceForceGroup(int directSpaceForceGroup) {
    if (directSpaceForceGroup < -1 || directSpaceForceGroup > 31)
        throw OpenMMException("PerturbNonbondedForce: direct-space force group must be between -1 and 31");
    this->directSpaceForceGroup = directSpaceForceGroup;
}

int PerturbNonbondedForce::getDirectSpaceForceGroup() const {
    return this->directSpaceForceGroup;
}

void PerturbNonbondedForce::enableEnergyConstantCache(bool enable) {
    cacheEnergyConstants = enable;
}

bool PerturbNonbondedForce::getEnableEnergyConstantCache() const {
    return cacheEnergyConstants;
}

void PerturbNonbondedForce::getInteractionEnergyForRealParticlesCuda(Context& context, int numRealAtoms,
        CudaArrayDevicePointer positions, CudaArrayDevicePointer parameters,
        CudaArrayDevicePointer excludedAtomStarts, CudaArrayDevicePointer excludedAtomCounts,
        CudaArrayDevicePointer excludedAtoms, CudaArrayDevicePointer forces,
        CudaArrayDevicePointer energies) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getInteractionEnergyForRealParticlesCuda(
            getContextImpl(context), numRealAtoms, positions, parameters, excludedAtomStarts,
            excludedAtomCounts, excludedAtoms, forces, energies);
}

void PerturbNonbondedForce::getDirectEnergyForCandidateParticlesCuda(Context& context,
        int numCandidateAtoms, CudaArrayDevicePointer positions, CudaArrayDevicePointer parameters,
        CudaArrayDevicePointer forces, CudaArrayDevicePointer energies) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getDirectEnergyForCandidateParticlesCuda(
            getContextImpl(context), numCandidateAtoms, positions, parameters, forces, energies);
}

void PerturbNonbondedForce::getDirectEnergyForCandidateParticlesCudaContextPrecision(
        Context& context, int numCandidateAtoms, CudaArrayDevicePointer positionBase,
        CudaArrayDevicePointer positionCorrection, CudaArrayDevicePointer parameters,
        CudaArrayDevicePointer forces, CudaArrayDevicePointer energies) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getDirectEnergyForCandidateParticlesCudaContextPrecision(
            getContextImpl(context), numCandidateAtoms, positionBase, positionCorrection,
            parameters, forces, energies);
}

void PerturbNonbondedForce::getPMEPotentialAtPositionsCuda(Context& context, int numPositions,
        CudaArrayDevicePointer positions, CudaArrayDevicePointer potentials) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getPMEPotentialAtPositionsCuda(
            getContextImpl(context), numPositions, positions, potentials);
}

void PerturbNonbondedForce::getPMEPotentialAtPositionsCudaContextPrecision(Context& context,
        int numPositions, CudaArrayDevicePointer positionBase,
        CudaArrayDevicePointer positionCorrection, CudaArrayDevicePointer potentials) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getPMEPotentialAtPositionsCudaContextPrecision(
            getContextImpl(context), numPositions, positionBase, positionCorrection, potentials);
}

void PerturbNonbondedForce::getSymmetricPMEReciprocalAtomicEnergiesCuda(Context& context,
        CudaArrayDevicePointer energies, CudaArrayDevicePointer sum) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getSymmetricPMEReciprocalAtomicEnergiesCuda(
            getContextImpl(context), energies, sum);
}

void PerturbNonbondedForce::getPMEPotentialFromParticlesCuda(Context& context,
        int numSourceParticles, CudaArrayDevicePointer sourcePositions,
        CudaArrayDevicePointer sourceCharges, int numQueryPositions,
        CudaArrayDevicePointer queryPositions, CudaArrayDevicePointer queryPotentials) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getPMEPotentialFromParticlesCuda(
            getContextImpl(context), numSourceParticles, sourcePositions, sourceCharges,
            numQueryPositions, queryPositions, queryPotentials);
}

void PerturbNonbondedForce::getPMEPotentialFromParticlesCudaContextPrecision(
        Context& context, int numSourceParticles, CudaArrayDevicePointer sourcePositionBase,
        CudaArrayDevicePointer sourcePositionCorrection, CudaArrayDevicePointer sourceCharges,
        int numQueryPositions, CudaArrayDevicePointer queryPositionBase,
        CudaArrayDevicePointer queryPositionCorrection, CudaArrayDevicePointer queryPotentials) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getPMEPotentialFromParticlesCudaContextPrecision(
            getContextImpl(context), numSourceParticles, sourcePositionBase, sourcePositionCorrection,
            sourceCharges, numQueryPositions, queryPositionBase, queryPositionCorrection, queryPotentials);
}

void PerturbNonbondedForce::getPMECandidateSelfEnergiesCuda(Context& context,
        int numCandidates, int numAtomsPerCandidate,
        CudaArrayDevicePointer candidatePositions, CudaArrayDevicePointer candidateCharges,
        CudaArrayDevicePointer energies) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getPMECandidateSelfEnergiesCuda(
            getContextImpl(context), numCandidates, numAtomsPerCandidate,
            candidatePositions, candidateCharges, energies);
}

void PerturbNonbondedForce::getPMECandidateSelfEnergiesCudaContextPrecision(
        Context& context, int numCandidates, int numAtomsPerCandidate,
        CudaArrayDevicePointer candidatePositionBase,
        CudaArrayDevicePointer candidatePositionCorrection,
        CudaArrayDevicePointer candidateCharges, CudaArrayDevicePointer energies) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getPMECandidateSelfEnergiesCudaContextPrecision(
            getContextImpl(context), numCandidates, numAtomsPerCandidate,
            candidatePositionBase, candidatePositionCorrection, candidateCharges, energies);
}

void PerturbNonbondedForce::getDirectEnergyForCandidateSubsetCuda(Context& context,
        int numCandidateAtoms, CudaArrayDevicePointer candidatePositions,
        CudaArrayDevicePointer candidateParameters, int numSourceAtoms,
        CudaArrayDevicePointer sourceAtomIndices, CudaArrayDevicePointer energies) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getDirectEnergyForCandidateSubsetCuda(
            getContextImpl(context), numCandidateAtoms, candidatePositions, candidateParameters,
            numSourceAtoms, sourceAtomIndices, energies);
}

void PerturbNonbondedForce::getDirectEnergyForCandidateSubsetCudaContextPrecision(
        Context& context, int numCandidateAtoms, CudaArrayDevicePointer candidatePositionBase,
        CudaArrayDevicePointer candidatePositionCorrection,
        CudaArrayDevicePointer candidateParameters, int numSourceAtoms,
        CudaArrayDevicePointer sourceAtomIndices, CudaArrayDevicePointer energies) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getDirectEnergyForCandidateSubsetCudaContextPrecision(
            getContextImpl(context), numCandidateAtoms, candidatePositionBase,
            candidatePositionCorrection, candidateParameters, numSourceAtoms,
            sourceAtomIndices, energies);
}

void PerturbNonbondedForce::getDirectEnergyForRealParticlesPmeCuda(Context& context,
        int numRealAtoms, CudaArrayDevicePointer realAtomIndices,
        int numAtomsPerMolecule, CudaArrayDevicePointer energies) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).getDirectEnergyForRealParticlesPmeCuda(
            getContextImpl(context), numRealAtoms, realAtomIndices, numAtomsPerMolecule, energies);
}

void* PerturbNonbondedForce::getCudaPlatformData(Context& context) const {
    getImplInContext(context);
    return getContextImpl(context).getPlatformData();
}

void PerturbNonbondedForce::setCudaParticleQueryBlockPruningForTesting(Context& context, bool enabled) {
    dynamic_cast<PerturbNonbondedForceImpl&>(getImplInContext(context)).setParticleQueryBlockPruning(enabled);
}

void PerturbNonbondedForce::setLambdaSchedules(
        const std::vector<double>& lambdaA_charge_schedule, 
        const std::vector<double>& lambdaB_charge_schedule,
        const std::vector<double>& lambdaC_charge_schedule,
        const std::vector<double>& lambdaA_vdw_schedule,
        const std::vector<double>& lambdaB_vdw_schedule,
        const std::vector<double>& rest2_scale_schedule) {
    const size_t scheduleSize = lambdaA_charge_schedule.size();
    if (scheduleSize == 0)
        throw OpenMMException("PerturbNonbondedForce: lambda schedules cannot be empty");
    if (lambdaB_charge_schedule.size() != scheduleSize ||
            lambdaC_charge_schedule.size() != scheduleSize ||
            lambdaA_vdw_schedule.size() != scheduleSize ||
            lambdaB_vdw_schedule.size() != scheduleSize ||
            rest2_scale_schedule.size() != scheduleSize)
        throw OpenMMException("PerturbNonbondedForce: all lambda schedules must have the same length");
    const std::vector<const std::vector<double>*> schedules = {
            &lambdaA_charge_schedule, &lambdaB_charge_schedule, &lambdaC_charge_schedule,
            &lambdaA_vdw_schedule, &lambdaB_vdw_schedule, &rest2_scale_schedule};
    for (const auto* schedule : schedules)
        for (double value : *schedule)
            if (!std::isfinite(value))
                throw OpenMMException("PerturbNonbondedForce: lambda schedules must contain only finite values");
    for (double value : rest2_scale_schedule)
        if (value < 0.0)
            throw OpenMMException("PerturbNonbondedForce: REST2 scale values must be nonnegative");
    numReplicas = static_cast<int>(scheduleSize);
    lambdaSchedules["lambdaA_charge"].resize(numReplicas);  
    lambdaSchedules["lambdaB_charge"].resize(numReplicas);
    lambdaSchedules["lambdaC_charge"].resize(numReplicas);
    lambdaSchedules["lambdaA_vdw"].resize(numReplicas);
    lambdaSchedules["lambdaB_vdw"].resize(numReplicas);
    lambdaSchedules["rest2_scale"].resize(numReplicas);

    lambdaSchedules["lambdaA_charge"] = lambdaA_charge_schedule;
    lambdaSchedules["lambdaB_charge"] = lambdaB_charge_schedule;
    lambdaSchedules["lambdaC_charge"] = lambdaC_charge_schedule;
    lambdaSchedules["lambdaA_vdw"] = lambdaA_vdw_schedule;
    lambdaSchedules["lambdaB_vdw"] = lambdaB_vdw_schedule;
    lambdaSchedules["rest2_scale"] = rest2_scale_schedule;
}

std::map<string, vector<double>> PerturbNonbondedForce::getLambdaSchedules() const {
    if (lambdaSchedules.empty())
        throw OpenMMException("lambdaSchedules is empty");
    return lambdaSchedules;
}
int PerturbNonbondedForce::getNumReplicas() const {
    return numReplicas;
}
