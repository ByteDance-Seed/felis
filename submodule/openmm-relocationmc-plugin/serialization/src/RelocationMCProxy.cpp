// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#include "RelocationMC.h"
#include "RelocationMCProxy.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationNode.h"

using namespace OpenMM;
using namespace OpenMM::RelocationMCPlugin;

RelocationMCProxy::RelocationMCProxy() : SerializationProxy("RelocationMC") {
}

void RelocationMCProxy::serialize(const void* object, SerializationNode& node) const {
    const RelocationMC& relocation = *reinterpret_cast<const RelocationMC*>(object);
    node.setIntProperty("version", 4);
    node.setIntProperty("protocolVersion", static_cast<int>(relocation.getProtocolVersion()));
    node.setIntProperty("forceGroup", relocation.getForceGroup());
    node.setStringProperty("name", relocation.getName());
    node.setDoubleProperty("temperature", relocation.getTemperature());
    node.setDoubleProperty("sphereRadius", relocation.getSphereRadius());
    node.setIntProperty("exchangeFrequency", relocation.getExchangeFrequency());
    node.setIntProperty("numCandidateSitesPerExchange", relocation.getNumCandidateSitesPerExchange());
    node.setLongProperty("randomSeed", static_cast<long long>(relocation.getRandomSeed()));
    node.setBoolProperty("orientationsEnabled", relocation.getOrientationsEnabled());
    node.setBoolProperty("useCavityBias", relocation.getUseCavityBias());
    node.setBoolProperty("useLegacyAsymmetricEnergies", relocation.getUseLegacyAsymmetricEnergies());
    node.setBoolProperty("verbose", relocation.getVerbose());
    node.setStringProperty("logFileName", relocation.getLogFileName());

    SerializationNode& ligandAtoms = node.createChildNode("LigandAtoms");
    for (int atom : relocation.getLigandAtoms())
        ligandAtoms.createChildNode("Atom").setIntProperty("index", atom);

    SerializationNode& siteParameters = node.createChildNode("LabileMolSiteParameters");
    for (const auto& params : relocation.getLabileMolSiteParameters()) {
        siteParameters.createChildNode("Site")
                .setDoubleProperty("charge", params.charge)
                .setDoubleProperty("sigma", params.sigma)
                .setDoubleProperty("epsilon", params.epsilon);
    }

    SerializationNode& referencePositions = node.createChildNode("LabileMolReferencePositions");
    for (const auto& position : relocation.getLabileMolReferencePositions()) {
        referencePositions.createChildNode("Position")
                .setDoubleProperty("x", position[0])
                .setDoubleProperty("y", position[1])
                .setDoubleProperty("z", position[2]);
    }

    SerializationNode& molecules = node.createChildNode("LabileMols");
    for (const auto& molecule : relocation.getLabileMols()) {
        SerializationNode& molNode = molecules.createChildNode("LabileMol")
                .setIntProperty("referenceAtom", molecule.referenceAtom);
        for (int atom : molecule.atomIndices)
            molNode.createChildNode("Atom").setIntProperty("index", atom);
    }
}

void* RelocationMCProxy::deserialize(const SerializationNode& node) const {
    const int version = node.getIntProperty("version");
    if (version < 2 || version > 4)
        throw OpenMMException("Unsupported RelocationMC serialization version.");

    RelocationMC* relocation = new RelocationMC(node.getDoubleProperty("temperature"),
            node.getDoubleProperty("sphereRadius"), node.getIntProperty("exchangeFrequency"),
            node.getIntProperty("numCandidateSitesPerExchange"));
    try {
        relocation->setForceGroup(node.getIntProperty("forceGroup", 0));
        relocation->setName(node.getStringProperty("name", relocation->getName()));
        relocation->setRandomSeed(static_cast<uint64_t>(node.getLongProperty("randomSeed", 0)));
        // Schema versions 2 and 3 predate an explicit protocol.  They always
        // deserialize to the historical behavior rather than inheriting the
        // current default from the installed plugin.
        const int protocol = version >= 4 ? node.getIntProperty("protocolVersion") :
                static_cast<int>(RelocationMC::Legacy);
        relocation->setProtocolVersion(static_cast<RelocationMC::ProtocolVersion>(protocol));
        relocation->setOrientationsEnabled(node.getBoolProperty("orientationsEnabled", true));
        relocation->setUseCavityBias(node.getBoolProperty("useCavityBias", protocol == RelocationMC::Legacy));
        relocation->setUseLegacyAsymmetricEnergies(node.getBoolProperty("useLegacyAsymmetricEnergies", false));
        relocation->setVerbose(node.getBoolProperty("verbose", false));
        relocation->setLogFileName(node.getStringProperty("logFileName", ""));

        std::vector<int> ligandAtoms;
        for (const auto& atomNode : node.getChildNode("LigandAtoms").getChildren())
            ligandAtoms.push_back(atomNode.getIntProperty("index"));
        relocation->setLigandAtoms(ligandAtoms);

        std::vector<LabileMolSiteParameters> siteParameters;
        for (const auto& siteNode : node.getChildNode("LabileMolSiteParameters").getChildren())
            siteParameters.emplace_back(siteNode.getDoubleProperty("charge"),
                    siteNode.getDoubleProperty("sigma"), siteNode.getDoubleProperty("epsilon"));
        relocation->setLabileMolSiteParameters(siteParameters);

        std::vector<Vec3> referencePositions;
        for (const auto& positionNode : node.getChildNode("LabileMolReferencePositions").getChildren())
            referencePositions.emplace_back(positionNode.getDoubleProperty("x"),
                    positionNode.getDoubleProperty("y"), positionNode.getDoubleProperty("z"));
        relocation->setLabileMolReferencePositions(referencePositions);

        for (const auto& molNode : node.getChildNode("LabileMols").getChildren()) {
            std::vector<int> atomIndices;
            for (const auto& atomNode : molNode.getChildren())
                atomIndices.push_back(atomNode.getIntProperty("index"));
            relocation->addLabileMol(atomIndices, molNode.getIntProperty("referenceAtom"));
        }
    }
    catch (...) {
        delete relocation;
        throw;
    }
    return relocation;
}
