// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

%module(package="RelocationMCPlugin") RelocationMCPlugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include "std_string.i"
%include "std_vector.i"
%include "stdint.i"

%typemap(out) std::vector<Vec3>* {
    $result = copyVVec3ToList(*$1);
}

%{
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include "RelocationMC.h"
#include <numpy/arrayobject.h>

namespace OpenMM {

PyObject* copyVVec3ToList(const std::vector<Vec3>& values) {
    PyObject* list = PyList_New(values.size());
    PyObject* module = PyImport_AddModule("openmm");
    PyObject* vec3 = PyObject_GetAttrString(module, "Vec3");
    for (int i = 0; i < static_cast<int>(values.size()); i++) {
        const Vec3& value = values[i];
        PyObject* args = Py_BuildValue("(d,d,d)", value[0], value[1], value[2]);
        PyObject* item = PyObject_CallObject(vec3, args);
        Py_DECREF(args);
        PyList_SET_ITEM(list, i, item);
    }
    Py_DECREF(vec3);
    return list;
}

int isNumpyAvailable() {
    static bool initialized = false;
    static bool available = false;
    if (!initialized) {
        initialized = true;
        available = (_import_array() >= 0);
    }
    return available;
}

} // namespace OpenMM

using namespace OpenMM;
%}

%exception {
    try {
        $action
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

namespace std {
    %template(IntVector) vector<int>;
}

namespace OpenMM {
namespace RelocationMCPlugin {

struct LabileMolSiteParameters {
    LabileMolSiteParameters(double charge = 0.0, double sigma = 0.0, double epsilon = 0.0);
    double charge;
    double sigma;
    double epsilon;
};

struct ExchangeResult {
    ExchangeResult();
    bool attempted;
    bool accepted;
    bool exchangeToInner;
    int movedLabileMol;
    double deletedMoleculeEnergy;
    double candidateMoleculeEnergy;
    double deletionProbability;
    double insertionProbability;
    int protocolVersion;
    double candidateSelectionProbability;
    double acceptanceProbability;
    double deletedSourceScore;
    double candidateSourceScore;
    double sourceLogSum;
    double proposedSourceLogSum;
    double forwardLogWeightSum;
    double reverseLogWeightSum;
    double logAcceptanceRatio;
    double energyDelta;
    double energyResidual;
    double isolatedReciprocalDelta;
    double deletedDirectEnergy;
    double deletedFullMeshCrossEnergy;
    double deletedMeshSelfEnergy;
    double deletedMeshRemovalEnergy;
    double candidateEnvironmentDirectEnergy;
    double candidateFullMeshCrossEnergy;
    double candidateDeletedMeshCrossEnergy;
    double candidateMeshSelfEnergy;
    std::vector<Vec3> newPositions;
    std::vector<int> movedAtomIndices;
};

} // namespace RelocationMCPlugin
} // namespace OpenMM

namespace std {
    %template(LabileMolSiteParametersVector) vector<OpenMM::RelocationMCPlugin::LabileMolSiteParameters>;
}

namespace OpenMM {
namespace RelocationMCPlugin {

class RelocationMC : public OpenMM::Force {
public:
    enum ProtocolVersion {
        Legacy = 1,
        MTM = 2
    };
    RelocationMC(double temperature = 300.0, double sphereRadius = 0.5,
            int exchangeFrequency = 100, int numCandidateSitesPerExchange = 300);
    void setProtocolVersion(ProtocolVersion protocol);
    ProtocolVersion getProtocolVersion() const;
    void setTemperature(double temperature);
    double getTemperature() const;
    void setSphereRadius(double radius);
    double getSphereRadius() const;
    void setExchangeFrequency(int frequency);
    int getExchangeFrequency() const;
    void setNumCandidateSitesPerExchange(int numCandidateSites);
    int getNumCandidateSitesPerExchange() const;
    void setRandomSeed(uint64_t seed);
    uint64_t getRandomSeed() const;
    void setLigandAtoms(const std::vector<int>& ligandAtoms);
    const std::vector<int>& getLigandAtoms() const;
    void setLabileMolSiteParameters(const std::vector<LabileMolSiteParameters>& parameters);
    const std::vector<LabileMolSiteParameters>& getLabileMolSiteParameters() const;
    void addLabileMol(const std::vector<int>& atomIndices, int referenceAtom);
    int getNumLabileMols() const;
    int getNumAtomsPerLabileMol() const;
    void setLabileMolReferencePositions(const std::vector<Vec3>& positions);
    const std::vector<Vec3>& getLabileMolReferencePositions() const;
    void setOrientationsEnabled(bool enabled);
    bool getOrientationsEnabled() const;
    void setUseCavityBias(bool enabled);
    bool getUseCavityBias() const;
    void setUseLegacyAsymmetricEnergies(bool enabled);
    bool getUseLegacyAsymmetricEnergies() const;
    void setVerbose(bool verbose);
    bool getVerbose() const;
    void setLogFileName(const std::string& fileName);
    const std::string& getLogFileName() const;
    bool isExchangeDue(const OpenMM::Context& context) const;
    ExchangeResult attemptExchange(OpenMM::Context& context);
    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const;
};

%extend RelocationMC {
    static OpenMM::RelocationMCPlugin::RelocationMC& cast(OpenMM::Force& force) {
        return dynamic_cast<OpenMM::RelocationMCPlugin::RelocationMC&>(force);
    }
    static bool isinstance(OpenMM::Force& force) {
        return dynamic_cast<OpenMM::RelocationMCPlugin::RelocationMC*>(&force) != NULL;
    }
}

} // namespace RelocationMCPlugin
} // namespace OpenMM
