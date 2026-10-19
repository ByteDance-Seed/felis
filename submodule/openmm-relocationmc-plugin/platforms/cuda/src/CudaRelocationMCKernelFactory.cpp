// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#include <exception>
#include "CudaRelocationMCKernelFactory.h"
#include "CudaRelocationMCKernels.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <vector>

using namespace OpenMM;
using namespace OpenMM::RelocationMCPlugin;
using namespace std;

namespace {
void registerRelocationMCFactories(Platform& platform) {
    CudaRelocationMCKernelFactory* factory = new CudaRelocationMCKernelFactory();
    platform.registerKernelFactory(CalcRelocationMCKernel::Name(), factory);
}
}

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        registerRelocationMCFactories(platform);
    }
    catch (const std::exception&) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerRelocationMCCudaKernelFactories() {
    Platform* platform;
    try {
        platform = &Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        platform = new CudaPlatform();
        Platform::registerPlatform(platform);
    }
    registerRelocationMCFactories(*platform);
}

KernelImpl* CudaRelocationMCKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
    // For multiple devices, we need to create a parallel kernel
    if (data.contexts.size() > 1){
        throw OpenMMException("Parallel RelocationMC kernel is not implemented yet");
    }
    // For single device, we create a single kernel
    CudaContext& cu = *data.contexts[0];
    if (name == CalcRelocationMCKernel::Name())
        return new CudaCalcRelocationMCKernel(name, platform, cu, context.getSystem());
    throw OpenMMException((std::string("Tried to create RelocationMC kernel with illegal kernel name '")+name+"'").c_str());
}
