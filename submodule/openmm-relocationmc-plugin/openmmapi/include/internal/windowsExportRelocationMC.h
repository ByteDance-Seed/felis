/*
 * Adapted from the OpenMM example plugin
 * (https://github.com/openmm/openmmexampleplugin), MIT License.
 * Portions copyright (c) 2014-2021 Stanford University and the Authors.
 * See THIRD_PARTY_NOTICES for the full MIT notice.
 * Modifications copyright (c) 2026 Bytedance Ltd. and/or its affiliates
 * SPDX-License-Identifier: CC-BY-NC-4.0 AND MIT
 */

#ifndef OPENMM_WINDOWSEXPORTRELOCATIONMC_H_
#define OPENMM_WINDOWSEXPORTRELOCATIONMC_H_

#ifdef _MSC_VER
    #pragma warning(disable:4996)
    #pragma warning(disable:4251)
    #if defined(RELOCATIONMC_BUILDING_SHARED_LIBRARY)
        #define OPENMM_EXPORT_RELOCATIONMC __declspec(dllexport)
    #elif defined(RELOCATIONMC_BUILDING_STATIC_LIBRARY) || defined(RELOCATIONMC_USE_STATIC_LIBRARIES)
        #define OPENMM_EXPORT_RELOCATIONMC
    #else
        #define OPENMM_EXPORT_RELOCATIONMC __declspec(dllimport)
    #endif
#else
    #define OPENMM_EXPORT_RELOCATIONMC
#endif

#endif
