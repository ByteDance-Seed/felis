/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2010-2021 Stanford University and the Authors.      *
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

#include "PerturbNonbondedForce.h"
#include "PerturbNonbondedForceProxy.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/Force.h"
#include <sstream>
#include <iostream>

using namespace PerturbNonbondedPlugin;
using namespace OpenMM;
using namespace std;

PerturbNonbondedForceProxy::PerturbNonbondedForceProxy() : SerializationProxy("PerturbNonbondedForce") {
}

void PerturbNonbondedForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 5);
    const PerturbNonbondedForce& force = *reinterpret_cast<const PerturbNonbondedForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setIntProperty("method", (int) force.getNonbondedMethod());
    node.setDoubleProperty("cutoff", force.getCutoffDistance());
    node.setBoolProperty("useSwitchingFunction", force.getUseSwitchingFunction());
    node.setDoubleProperty("switchingDistance", force.getSwitchingDistance());
    node.setDoubleProperty("ewaldTolerance", force.getEwaldErrorTolerance());
    node.setDoubleProperty("rfDielectric", force.getReactionFieldDielectric());
    node.setIntProperty("dispersionCorrection", force.getUseDispersionCorrection());
    node.setIntProperty("exceptionsUsePeriodic", force.getExceptionsUsePeriodicBoundaryConditions());
    node.setBoolProperty("includeDirectSpace", force.getIncludeDirectSpace());
    node.setBoolProperty("includeLJ", force.getIncludeLJ());
    node.setIntProperty("directSpaceForceGroup", force.getDirectSpaceForceGroup());
    node.setIntProperty("ljForceGroup", force.getLJForceGroup());
    node.setBoolProperty("enableEnergyCache", force.getEnableEnergyConstantCache());
    double alpha;
    int nx, ny, nz;
    force.getPMEParameters(alpha, nx, ny, nz);
    node.setDoubleProperty("alpha", alpha);
    node.setIntProperty("nx", nx);
    node.setIntProperty("ny", ny);
    node.setIntProperty("nz", nz);
    force.getLJPMEParameters(alpha, nx, ny, nz);
    node.setDoubleProperty("ljAlpha", alpha);
    node.setIntProperty("ljnx", nx);
    node.setIntProperty("ljny", ny);
    node.setIntProperty("ljnz", nz);
    node.setIntProperty("recipForceGroup", force.getReciprocalSpaceForceGroup());
    SerializationNode& globalParams = node.createChildNode("GlobalParameters");
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalParams.createChildNode("Parameter").setStringProperty("name", force.getGlobalParameterName(i)).setDoubleProperty("default", force.getGlobalParameterDefaultValue(i));
    SerializationNode& particleOffsets = node.createChildNode("ParticleOffsets");
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        int particle;
        double chargeScale, sigmaScale, epsilonScale;
        string parameter;
        force.getParticleParameterOffset(i, parameter, particle, chargeScale, sigmaScale, epsilonScale);
        particleOffsets.createChildNode("Offset").setStringProperty("parameter", parameter).setIntProperty("particle", particle).setDoubleProperty("q", chargeScale).setDoubleProperty("sig", sigmaScale).setDoubleProperty("eps", epsilonScale);
    }
    SerializationNode& exceptionOffsets = node.createChildNode("ExceptionOffsets");
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        int exception;
        double chargeProdScale, sigmaScale, epsilonScale;
        string parameter;
        force.getExceptionParameterOffset(i, parameter, exception, chargeProdScale, sigmaScale, epsilonScale);
        exceptionOffsets.createChildNode("Offset").setStringProperty("parameter", parameter).setIntProperty("exception", exception).setDoubleProperty("q", chargeProdScale).setDoubleProperty("sig", sigmaScale).setDoubleProperty("eps", epsilonScale);
    }
    SerializationNode& particles = node.createChildNode("Particles");
    for (int i = 0; i < force.getNumParticles(); i++) {
        double chargeA, sigmaA, epsilonA;
        double chargeB, sigmaB, epsilonB;
        bool isDummy, isREST2;
        force.getParticleParameters(i, chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2);
        particles.createChildNode("Particle").setDoubleProperty("qA", chargeA).setDoubleProperty("sigA", sigmaA).setDoubleProperty("epsA", epsilonA).setDoubleProperty("qB", chargeB).setDoubleProperty("sigB", sigmaB).setDoubleProperty("epsB", epsilonB).setBoolProperty("isDummy", isDummy).setBoolProperty("isREST2", isREST2);
    }

    if (force.getEnableEnergyConstantCache()){
        SerializationNode& numReplicas = node.createChildNode("NumReplicas");
        numReplicas.setIntProperty("numReplicas", force.getNumReplicas());
        SerializationNode& lambdaSchedulesNode = node.createChildNode("LambdaSchedules");
        map<string, vector<double>> lambdaSchedules = force.getLambdaSchedules();
        for (auto& schedule : lambdaSchedules) {
            lambdaSchedulesNode.createChildNode("Schedule").setStringProperty("name", schedule.first);
            for (int i = 0; i < schedule.second.size(); i++)
                lambdaSchedulesNode.createChildNode(schedule.first+to_string(i)).setDoubleProperty("value", schedule.second[i]);
        }
    }
    
    SerializationNode& softcoreAlpha = node.createChildNode("SoftcoreAlpha");
    softcoreAlpha.setDoubleProperty("alpha", force.getSoftcoreAlpha());

    SerializationNode& exceptions = node.createChildNode("Exceptions");
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        exceptions.createChildNode("Exception").setIntProperty("p1", particle1).setIntProperty("p2", particle2).setDoubleProperty("q", chargeProd).setDoubleProperty("sig", sigma).setDoubleProperty("eps", epsilon);
    }
}

void* PerturbNonbondedForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version < 1 || version > 5)
        throw OpenMMException("Unsupported version number");
    PerturbNonbondedForce* force = new PerturbNonbondedForce();
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setNonbondedMethod((PerturbNonbondedForce::PerturbNonbondedMethod) node.getIntProperty("method"));
        force->setCutoffDistance(node.getDoubleProperty("cutoff"));
        force->setUseSwitchingFunction(node.getBoolProperty("useSwitchingFunction", false));
        force->setSwitchingDistance(node.getDoubleProperty("switchingDistance", -1.0));
        force->setEwaldErrorTolerance(node.getDoubleProperty("ewaldTolerance"));
        force->setReactionFieldDielectric(node.getDoubleProperty("rfDielectric"));
        force->setUseDispersionCorrection(node.getIntProperty("dispersionCorrection"));
        force->enableEnergyConstantCache(node.getBoolProperty("enableEnergyCache", false));
        if (node.hasProperty("includeDirectSpace"))
            force->setIncludeDirectSpace(node.getBoolProperty("includeDirectSpace"));
        if (version >= 5) {
            force->setIncludeLJ(node.getBoolProperty("includeLJ", true));
            force->setDirectSpaceForceGroup(node.getIntProperty("directSpaceForceGroup", -1));
            force->setLJForceGroup(node.getIntProperty("ljForceGroup", -1));
        }
        double alpha = node.getDoubleProperty("alpha", 0.0);
        int nx = node.getIntProperty("nx", 0);
        int ny = node.getIntProperty("ny", 0);
        int nz = node.getIntProperty("nz", 0);
        force->setPMEParameters(alpha, nx, ny, nz);
        if (version >= 2) {
            alpha = node.getDoubleProperty("ljAlpha", 0.0);
            nx = node.getIntProperty("ljnx", 0);
            ny = node.getIntProperty("ljny", 0);
            nz = node.getIntProperty("ljnz", 0);
            force->setLJPMEParameters(alpha, nx, ny, nz);
        }
        force->setReciprocalSpaceForceGroup(node.getIntProperty("recipForceGroup", -1));
        if (version >= 3) {
            const SerializationNode& globalParams = node.getChildNode("GlobalParameters");
            for (auto& parameter : globalParams.getChildren())
                force->addGlobalParameter(parameter.getStringProperty("name"), parameter.getDoubleProperty("default"));
            const SerializationNode& particleOffsets = node.getChildNode("ParticleOffsets");
            for (auto& offset : particleOffsets.getChildren())
                force->addParticleParameterOffset(offset.getStringProperty("parameter"), offset.getIntProperty("particle"), offset.getDoubleProperty("q"), offset.getDoubleProperty("sig"), offset.getDoubleProperty("eps"));
            const SerializationNode& exceptionOffsets = node.getChildNode("ExceptionOffsets");
            for (auto& offset : exceptionOffsets.getChildren())
                force->addExceptionParameterOffset(offset.getStringProperty("parameter"), offset.getIntProperty("exception"), offset.getDoubleProperty("q"), offset.getDoubleProperty("sig"), offset.getDoubleProperty("eps"));
        }
        if (version >= 4)
            force->setExceptionsUsePeriodicBoundaryConditions(node.getIntProperty("exceptionsUsePeriodic"));
        const SerializationNode& particles = node.getChildNode("Particles");
        for (auto& particle : particles.getChildren())
            force->addParticle(particle.getDoubleProperty("qA"), particle.getDoubleProperty("sigA"), particle.getDoubleProperty("epsA"), particle.getDoubleProperty("qB"), particle.getDoubleProperty("sigB"), particle.getDoubleProperty("epsB"), particle.getBoolProperty("isDummy"), particle.getBoolProperty("isREST2"));
        
        const SerializationNode& softcoreAlpha = node.getChildNode("SoftcoreAlpha");
        force->setSoftcoreAlpha(softcoreAlpha.getDoubleProperty("alpha"));

        if (force->getEnableEnergyConstantCache()){
            int numReplicas = node.getChildNode("NumReplicas").getIntProperty("numReplicas", 0);
            vector<double> lambdaA_charge_schedule(numReplicas);
            vector<double> lambdaB_charge_schedule(numReplicas);
            vector<double> lambdaC_charge_schedule(numReplicas);
            vector<double> lambdaA_vdw_schedule(numReplicas);
            vector<double> lambdaB_vdw_schedule(numReplicas);
            vector<double> rest2_scale_schedule(numReplicas);

            const SerializationNode& lambdaSchedulesNode = node.getChildNode("LambdaSchedules");
            for (int i = 0; i < numReplicas; i++){
                lambdaA_charge_schedule[i] = lambdaSchedulesNode.getChildNode("lambdaA_charge"+to_string(i)).getDoubleProperty("value");
                lambdaB_charge_schedule[i] = lambdaSchedulesNode.getChildNode("lambdaB_charge"+to_string(i)).getDoubleProperty("value");
                lambdaC_charge_schedule[i] = lambdaSchedulesNode.getChildNode("lambdaC_charge"+to_string(i)).getDoubleProperty("value");
                lambdaA_vdw_schedule[i] = lambdaSchedulesNode.getChildNode("lambdaA_vdw"+to_string(i)).getDoubleProperty("value");
                lambdaB_vdw_schedule[i] = lambdaSchedulesNode.getChildNode("lambdaB_vdw"+to_string(i)).getDoubleProperty("value");
                rest2_scale_schedule[i] = lambdaSchedulesNode.getChildNode("rest2_scale"+to_string(i)).getDoubleProperty("value");
            }
            force->setLambdaSchedules(lambdaA_charge_schedule, lambdaB_charge_schedule, lambdaC_charge_schedule, lambdaA_vdw_schedule, lambdaB_vdw_schedule, rest2_scale_schedule);
        }
        const SerializationNode& exceptions = node.getChildNode("Exceptions");
        for (auto& exception : exceptions.getChildren())
            force->addException(exception.getIntProperty("p1"), exception.getIntProperty("p2"), exception.getDoubleProperty("q"), exception.getDoubleProperty("sig"), exception.getDoubleProperty("eps"));
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
