// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#ifndef OPENMM_RELOCATIONMCIMPL_H_
#define OPENMM_RELOCATIONMCIMPL_H_

#include "RelocationMC.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace OpenMM {
namespace RelocationMCPlugin {

/** Internal implementation for the force-free RelocationMC protocol. */
class RelocationMCImpl : public ForceImpl {
public:
    explicit RelocationMCImpl(const RelocationMC& owner);
    ~RelocationMCImpl() override;

    void initialize(ContextImpl& context) override;
    const RelocationMC& getOwner() const override {
        return owner;
    }
    void updateContextState(ContextImpl& context, bool& forcesInvalid) override;
    double calcForcesAndEnergy(ContextImpl&, bool, bool, int) override {
        return 0.0;
    }
    std::map<std::string, double> getDefaultParameters() override;
    std::vector<std::string> getKernelNames() override;

    bool isExchangeDue(const Context& context) const;
    ExchangeResult attemptExchange(Context& context);
    void updateParametersInContext(ContextImpl& context);

private:
    void validateConfiguration(ContextImpl& context) const;
    ExchangeResult performExchange(ContextImpl& context, bool* forcesInvalid = nullptr);
    void writeLog(const ExchangeResult& result, double initialPotentialEnergy,
            double updatedPotentialEnergy);

    const RelocationMC& owner;
    Kernel kernel;
    RelocationMC::ProtocolVersion protocolVersion;
    long long lastAutomaticAttemptStep;
    int numAttempted;
    int numInvoked;
    std::stringstream logStream;
    std::ofstream logFile;
};

} // namespace RelocationMCPlugin
} // namespace OpenMM

#endif
