// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#include "RelocationMC.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace OpenMM;
using namespace OpenMM::RelocationMCPlugin;

extern "C" void registerRelocationMCSerializationProxies();

static void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Function>
static void requireThrows(Function function, const char* message) {
    try {
        function();
    }
    catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

static void configureTopology(RelocationMC& force) {
    force.setLigandAtoms({0, 1});
    force.setLabileMolSiteParameters({{-0.8, 0.3, 0.5}, {0.4, 0.0, 0.0}});
    force.setLabileMolReferencePositions({Vec3(0, 0, 0), Vec3(0.1, 0, 0)});
    force.addLabileMol({2, 3}, 2);
    force.addLabileMol({4, 5}, 4);
}

static std::string serialize(const RelocationMC& force) {
    std::stringstream buffer;
    XmlSerializer::serialize<RelocationMC>(&force, "Force", buffer);
    return buffer.str();
}

static RelocationMC* deserialize(const std::string& xml) {
    std::stringstream buffer(xml);
    return XmlSerializer::deserialize<RelocationMC>(buffer);
}

static void replaceFirst(std::string& text, const std::string& from, const std::string& to) {
    const size_t position = text.find(from);
    if (position == std::string::npos)
        throw std::runtime_error("Expected XML property was not found: "+from);
    text.replace(position, from.size(), to);
}

static void removeProperty(std::string& text, const std::string& name) {
    const std::string marker = " "+name+"=\"";
    const size_t begin = text.find(marker);
    if (begin == std::string::npos)
        throw std::runtime_error("Expected XML property was not found: "+name);
    const size_t valueEnd = text.find('"', begin+marker.size());
    if (valueEnd == std::string::npos)
        throw std::runtime_error("Malformed XML property: "+name);
    text.erase(begin, valueEnd-begin+1);
}

static void testLegacyRoundTrip() {
    RelocationMC force(310.0, 0.7, 25, 17);
    force.setProtocolVersion(RelocationMC::Legacy);
    force.setName("legacy-relocation");
    force.setForceGroup(7);
    force.setRandomSeed(42);
    configureTopology(force);
    force.setOrientationsEnabled(false);
    force.setUseCavityBias(false);
    force.setUseLegacyAsymmetricEnergies(true);
    force.setVerbose(true);
    force.setLogFileName("relocation.log");

    const std::string xml = serialize(force);
    require(xml.find("version=\"4\"") != std::string::npos,
            "RelocationMC did not serialize schema version 4.");
    require(xml.find("protocolVersion=\"1\"") != std::string::npos,
            "Legacy protocol was not serialized explicitly.");
    RelocationMC* copy = deserialize(xml);

    require(copy->getProtocolVersion() == RelocationMC::Legacy,
            "Legacy protocol did not survive serialization.");
    require(copy->getTemperature() == 310.0, "Temperature did not survive serialization.");
    require(copy->getSphereRadius() == 0.7, "Sphere radius did not survive serialization.");
    require(copy->getExchangeFrequency() == 25, "Frequency did not survive serialization.");
    require(copy->getNumCandidateSitesPerExchange() == 17, "Candidate count did not survive serialization.");
    require(copy->getRandomSeed() == 42, "Random seed did not survive serialization.");
    require(copy->getForceGroup() == 7 && copy->getName() == "legacy-relocation",
            "Force metadata did not survive serialization.");
    require(copy->getLigandAtoms().size() == 2, "Ligand atoms did not survive serialization.");
    require(copy->getLabileMolSiteParameters().size() == 2, "Site parameters did not survive serialization.");
    require(copy->getLabileMolReferencePositions().size() == 2, "Reference positions did not survive serialization.");
    require(copy->getNumLabileMols() == 2, "Labile molecules did not survive serialization.");
    require(!copy->getOrientationsEnabled(), "Orientation setting did not survive serialization.");
    require(!copy->getUseCavityBias(), "Cavity-bias setting did not survive serialization.");
    require(copy->getUseLegacyAsymmetricEnergies(), "Legacy-energy setting did not survive serialization.");
    require(copy->getVerbose(), "Verbose setting did not survive serialization.");
    require(copy->getLogFileName() == "relocation.log", "Log filename did not survive serialization.");
    delete copy;
}

static std::string createMtmXml() {
    RelocationMC force(300.0, 0.6, 10, 11);
    configureTopology(force);
    force.setUseCavityBias(false);
    force.setProtocolVersion(RelocationMC::MTM);
    return serialize(force);
}

static void testMtmRoundTrip() {
    const std::string xml = createMtmXml();
    require(xml.find("protocolVersion=\"2\"") != std::string::npos,
            "MTM protocol was not serialized explicitly.");
    RelocationMC* copy = deserialize(xml);
    require(copy->getProtocolVersion() == RelocationMC::MTM,
            "MTM protocol did not survive serialization.");
    require(copy->getOrientationsEnabled() && !copy->getUseCavityBias() &&
                    !copy->getUseLegacyAsymmetricEnergies(),
            "MTM compatibility settings did not survive serialization.");
    delete copy;
}

static void testOldSchemasSelectLegacy() {
    const std::string current = createMtmXml();
    for (int version : {2, 3}) {
        std::string old = current;
        replaceFirst(old, "version=\"4\"", "version=\""+std::to_string(version)+"\"");
        removeProperty(old, "protocolVersion");
        RelocationMC* copy = deserialize(old);
        require(copy->getProtocolVersion() == RelocationMC::Legacy,
                "An old serialization schema did not select the legacy protocol.");
        delete copy;
    }
}

static void testInvalidProtocolSerializationRejected() {
    std::string missing = createMtmXml();
    removeProperty(missing, "protocolVersion");
    requireThrows([&]() {
        RelocationMC* force = deserialize(missing);
        delete force;
    }, "Schema v4 without protocolVersion should be rejected.");

    std::string unknown = createMtmXml();
    replaceFirst(unknown, "protocolVersion=\"2\"", "protocolVersion=\"99\"");
    requireThrows([&]() {
        RelocationMC* force = deserialize(unknown);
        delete force;
    }, "An unknown serialized protocolVersion should be rejected.");
}

int main() {
    try {
        registerRelocationMCSerializationProxies();
        testLegacyRoundTrip();
        testMtmRoundTrip();
        testOldSchemasSelectLegacy();
        testInvalidProtocolSerializationRejected();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
