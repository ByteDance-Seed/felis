// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#include "RelocationMC.h"
#include "internal/RelocationMCImpl.h"
#include <openmm/OpenMMException.h>
#include <algorithm>
#include <cmath>
#include <set>

using namespace OpenMM;
using namespace OpenMM::RelocationMCPlugin;
using std::vector;

RelocationMC::RelocationMC(double temperature, double sphereRadius, int exchangeFrequency,
        int numCandidateSitesPerExchange) : temperature(temperature), sphereRadius(sphereRadius),
        exchangeFrequency(exchangeFrequency), numCandidateSitesPerExchange(numCandidateSitesPerExchange),
        protocolVersion(MTM), randomSeed(0), orientationsEnabled(true), useCavityBias(false),
        useLegacyAsymmetricEnergies(false), verbose(false) {
    setName("RelocationMC");
    setTemperature(temperature);
    setSphereRadius(sphereRadius);
    setExchangeFrequency(exchangeFrequency);
    setNumCandidateSitesPerExchange(numCandidateSitesPerExchange);
}

void RelocationMC::setProtocolVersion(ProtocolVersion protocol) {
    if (protocol != Legacy && protocol != MTM)
        throw OpenMMException("Unsupported RelocationMC protocol version.");
    if (protocol == MTM) {
        if (!orientationsEnabled)
            throw OpenMMException("RelocationMC MTM requires Haar orientations to be enabled.");
        if (useCavityBias)
            throw OpenMMException("RelocationMC MTM uses whole-box proposals and does not permit a separate cavity-volume bias.");
        if (useLegacyAsymmetricEnergies)
            throw OpenMMException("RelocationMC MTM requires environment-only exact-PME energies.");
    }
    protocolVersion = protocol;
}

RelocationMC::ProtocolVersion RelocationMC::getProtocolVersion() const {
    return protocolVersion;
}

void RelocationMC::setTemperature(double value) {
    if (!std::isfinite(value) || value <= 0.0)
        throw OpenMMException("RelocationMC temperature must be finite and positive.");
    temperature = value;
}

double RelocationMC::getTemperature() const {
    return temperature;
}

void RelocationMC::setSphereRadius(double value) {
    if (!std::isfinite(value) || value <= 0.0)
        throw OpenMMException("RelocationMC sphere radius must be finite and positive.");
    sphereRadius = value;
}

double RelocationMC::getSphereRadius() const {
    return sphereRadius;
}

void RelocationMC::setExchangeFrequency(int value) {
    if (value < 0)
        throw OpenMMException("RelocationMC exchange frequency cannot be negative.");
    exchangeFrequency = value;
}

int RelocationMC::getExchangeFrequency() const {
    return exchangeFrequency;
}

void RelocationMC::setNumCandidateSitesPerExchange(int value) {
    if (value <= 0)
        throw OpenMMException("RelocationMC candidate-site count must be positive.");
    numCandidateSitesPerExchange = value;
}

int RelocationMC::getNumCandidateSitesPerExchange() const {
    return numCandidateSitesPerExchange;
}

void RelocationMC::setRandomSeed(uint64_t seed) {
    randomSeed = seed;
}

uint64_t RelocationMC::getRandomSeed() const {
    return randomSeed;
}

void RelocationMC::setLigandAtoms(const vector<int>& atoms) {
    if (atoms.empty())
        throw OpenMMException("RelocationMC requires at least one ligand atom.");
    std::set<int> uniqueAtoms;
    for (int atom : atoms) {
        if (atom < 0)
            throw OpenMMException("RelocationMC ligand atom indices cannot be negative.");
        if (!uniqueAtoms.insert(atom).second)
            throw OpenMMException("RelocationMC ligand atom indices must be unique.");
    }
    ligandAtoms = atoms;
}

const vector<int>& RelocationMC::getLigandAtoms() const {
    return ligandAtoms;
}

void RelocationMC::setLabileMolSiteParameters(const vector<LabileMolSiteParameters>& parameters) {
    if (parameters.empty())
        throw OpenMMException("RelocationMC requires at least one labile-molecule site parameter set.");
    for (const auto& p : parameters) {
        if (!std::isfinite(p.charge) || !std::isfinite(p.sigma) || !std::isfinite(p.epsilon) ||
                p.sigma < 0.0 || p.epsilon < 0.0)
            throw OpenMMException("RelocationMC site parameters must be finite, with nonnegative sigma and epsilon.");
    }
    if (!labileMols.empty() && parameters.size() != labileMols.front().atomIndices.size())
        throw OpenMMException("RelocationMC site-parameter count must match every labile molecule.");
    if (!labileMolReferencePositions.empty() && parameters.size() != labileMolReferencePositions.size())
        throw OpenMMException("RelocationMC site-parameter count must match reference positions.");
    labileMolSiteParameters = parameters;
}

const vector<LabileMolSiteParameters>& RelocationMC::getLabileMolSiteParameters() const {
    return labileMolSiteParameters;
}

void RelocationMC::addLabileMol(const vector<int>& atomIndices, int referenceAtom) {
    if (atomIndices.empty())
        throw OpenMMException("RelocationMC labile molecule cannot be empty.");
    if (std::find(atomIndices.begin(), atomIndices.end(), referenceAtom) == atomIndices.end())
        throw OpenMMException("RelocationMC reference atom must belong to its labile molecule.");
    std::set<int> uniqueAtoms;
    for (int atom : atomIndices) {
        if (atom < 0)
            throw OpenMMException("RelocationMC labile-molecule atom indices cannot be negative.");
        if (!uniqueAtoms.insert(atom).second)
            throw OpenMMException("RelocationMC labile-molecule atom indices must be unique.");
        for (const auto& molecule : labileMols)
            if (std::find(molecule.atomIndices.begin(), molecule.atomIndices.end(), atom) != molecule.atomIndices.end())
                throw OpenMMException("A particle cannot belong to more than one RelocationMC labile molecule.");
    }
    if (!labileMols.empty() && atomIndices.size() != labileMols.front().atomIndices.size())
        throw OpenMMException("Every RelocationMC labile molecule must have the same number of sites.");
    if (!labileMolSiteParameters.empty() && atomIndices.size() != labileMolSiteParameters.size())
        throw OpenMMException("Every labile molecule must have one atom per configured site parameter.");
    if (!labileMolReferencePositions.empty() && atomIndices.size() != labileMolReferencePositions.size())
        throw OpenMMException("Every labile molecule must have one atom per configured reference position.");
    labileMols.push_back({atomIndices, referenceAtom});
}

int RelocationMC::getNumLabileMols() const {
    return static_cast<int>(labileMols.size());
}

void RelocationMC::getLabileMolParameters(int index, vector<int>& atomIndices, int& referenceAtom) const {
    if (index < 0 || index >= getNumLabileMols())
        throw OpenMMException("RelocationMC labile-molecule index is out of range.");
    atomIndices = labileMols[index].atomIndices;
    referenceAtom = labileMols[index].referenceAtom;
}

const vector<LabileMol>& RelocationMC::getLabileMols() const {
    return labileMols;
}

int RelocationMC::getNumAtomsPerLabileMol() const {
    if (!labileMols.empty())
        return static_cast<int>(labileMols.front().atomIndices.size());
    return static_cast<int>(labileMolSiteParameters.size());
}

void RelocationMC::setLabileMolReferencePositions(const vector<Vec3>& positions) {
    if (positions.empty())
        throw OpenMMException("RelocationMC reference positions cannot be empty.");
    for (const Vec3& position : positions)
        if (!std::isfinite(position[0]) || !std::isfinite(position[1]) || !std::isfinite(position[2]))
            throw OpenMMException("RelocationMC reference positions must be finite.");
    if (getNumAtomsPerLabileMol() != 0 && positions.size() != static_cast<size_t>(getNumAtomsPerLabileMol()))
        throw OpenMMException("RelocationMC reference-position count must match the molecule site count.");
    labileMolReferencePositions = positions;
}

const vector<Vec3>& RelocationMC::getLabileMolReferencePositions() const {
    return labileMolReferencePositions;
}

void RelocationMC::setOrientationsEnabled(bool enabled) {
    if (!enabled && protocolVersion == MTM)
        throw OpenMMException("RelocationMC MTM requires Haar orientations to be enabled.");
    orientationsEnabled = enabled;
}

bool RelocationMC::getOrientationsEnabled() const {
    return orientationsEnabled;
}

void RelocationMC::setUseCavityBias(bool enabled) {
    if (enabled && protocolVersion == MTM)
        throw OpenMMException("RelocationMC MTM uses whole-box proposals and does not permit a separate cavity-volume bias.");
    useCavityBias = enabled;
}

bool RelocationMC::getUseCavityBias() const {
    return useCavityBias;
}

void RelocationMC::setUseLegacyAsymmetricEnergies(bool enabled) {
    if (enabled && protocolVersion == MTM)
        throw OpenMMException("RelocationMC MTM requires environment-only exact-PME energies.");
    useLegacyAsymmetricEnergies = enabled;
}

bool RelocationMC::getUseLegacyAsymmetricEnergies() const {
    return useLegacyAsymmetricEnergies;
}

void RelocationMC::setVerbose(bool value) {
    verbose = value;
}

bool RelocationMC::getVerbose() const {
    return verbose;
}

void RelocationMC::setLogFileName(const std::string& fileName) {
    logFileName = fileName;
}

const std::string& RelocationMC::getLogFileName() const {
    return logFileName;
}

bool RelocationMC::isExchangeDue(const Context& context) const {
    return dynamic_cast<const RelocationMCImpl&>(getImplInContext(context)).isExchangeDue(context);
}

ExchangeResult RelocationMC::attemptExchange(Context& context) {
    return dynamic_cast<RelocationMCImpl&>(getImplInContext(context)).attemptExchange(context);
}

void RelocationMC::updateParametersInContext(Context& context) {
    dynamic_cast<RelocationMCImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

ForceImpl* RelocationMC::createImpl() const {
    return new RelocationMCImpl(*this);
}
