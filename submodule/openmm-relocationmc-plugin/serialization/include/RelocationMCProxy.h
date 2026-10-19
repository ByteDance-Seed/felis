// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#ifndef OPENMM_RELOCATIONMC_PROXY_H_
#define OPENMM_RELOCATIONMC_PROXY_H_

#include "internal/windowsExportRelocationMC.h"
#include "openmm/serialization/SerializationProxy.h"

namespace OpenMM {

class OPENMM_EXPORT_RELOCATIONMC RelocationMCProxy : public SerializationProxy {
public:
    RelocationMCProxy();
    void serialize(const void* object, SerializationNode& node) const override;
    void* deserialize(const SerializationNode& node) const override;
};

} // namespace OpenMM

#endif
