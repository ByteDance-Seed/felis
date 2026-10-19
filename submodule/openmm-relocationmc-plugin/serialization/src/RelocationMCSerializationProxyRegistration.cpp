// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#ifdef WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "RelocationMC.h"
#include "RelocationMCProxy.h"
#include "openmm/serialization/SerializationProxy.h"

#if defined(WIN32)
    extern "C" OPENMM_EXPORT_RELOCATIONMC void registerRelocationMCSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerRelocationMCSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerRelocationMCSerializationProxies();
#endif

using namespace OpenMM;
using namespace OpenMM::RelocationMCPlugin;

extern "C" OPENMM_EXPORT_RELOCATIONMC void registerRelocationMCSerializationProxies() {
    SerializationProxy::registerProxy(typeid(RelocationMC), new RelocationMCProxy());
}
