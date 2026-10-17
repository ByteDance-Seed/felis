// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#ifdef WIN32
#define _USE_MATH_DEFINES
#endif

#include "openmm/cuda/CudaPlatform.h"
#include <iostream>
#include <string>

extern "C" void registerPerturbNonbondedCudaKernelFactories();
extern "C" void registerRelocationMCCudaKernelFactories();

OpenMM::Platform& initializePlatform() {
    registerPerturbNonbondedCudaKernelFactories();
    registerRelocationMCCudaKernelFactories();
    return OpenMM::Platform::getPlatformByName("CUDA");
}

OpenMM::Platform& platform = initializePlatform();

void initializeTests(int argc, char* argv[]) {
    if (argc > 1)
        platform.setPropertyDefaultValue("Precision", std::string(argv[1]));
    if (argc > 2)
        platform.setPropertyDefaultValue("DeviceIndex", std::string(argv[2]));
    if (argc > 3)
        platform.setPropertyDefaultValue("DisablePmeStream", std::string(argv[3]));
}
