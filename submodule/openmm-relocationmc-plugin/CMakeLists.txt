# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

cmake_minimum_required(VERSION 3.18)
project(OpenMMRelocationMCPlugin VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)

set(OPENMM_DIR "/usr/local/openmm" CACHE PATH "OpenMM installation prefix")
set(PERTURB_NONBONDED_INSTALL_DIR "${OPENMM_DIR}" CACHE PATH
    "Prefix containing a pre-installed PerturbNonbondedPlugin")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_path(OPENMM_INCLUDE_DIR openmm/Context.h
    HINTS "${OPENMM_DIR}/include"
    REQUIRED)
find_path(OPENMM_CUDA_INCLUDE_DIR openmm/cuda/CudaKernelSources.h
    HINTS "${OPENMM_DIR}/include"
    REQUIRED)
find_library(OPENMM_LIBRARY OpenMM
    HINTS "${OPENMM_DIR}/${CMAKE_INSTALL_LIBDIR}" "${OPENMM_DIR}/lib"
    REQUIRED)
add_library(OpenMM::OpenMM UNKNOWN IMPORTED)
set_target_properties(OpenMM::OpenMM PROPERTIES
    IMPORTED_LOCATION "${OPENMM_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OPENMM_INCLUDE_DIR}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${OPENMM_DIR}" CACHE PATH "Installation prefix" FORCE)
endif()

file(GLOB RELOCATIONMC_API_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include/internal/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/serialization/include/*.h")
file(GLOB RELOCATIONMC_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/serialization/src/*.cpp")

set(RELOCATIONMC_LIBRARY_NAME RelocationMCPlugin)
set(SHARED_RELOCATIONMC_TARGET ${RELOCATIONMC_LIBRARY_NAME})
add_library(${SHARED_RELOCATIONMC_TARGET} SHARED ${RELOCATIONMC_SOURCES} ${RELOCATIONMC_API_HEADERS})
target_compile_definitions(${SHARED_RELOCATIONMC_TARGET} PRIVATE RELOCATIONMC_BUILDING_SHARED_LIBRARY)
target_include_directories(${SHARED_RELOCATIONMC_TARGET}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/serialization/include")
target_link_libraries(${SHARED_RELOCATIONMC_TARGET} PUBLIC OpenMM::OpenMM)
set_target_properties(${SHARED_RELOCATIONMC_TARGET} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    INSTALL_RPATH "$ORIGIN")

install(TARGETS ${SHARED_RELOCATIONMC_TARGET}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY openmmapi/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY serialization/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")

if(BUILD_TESTING)
    add_subdirectory(serialization/tests)
    add_subdirectory(tests)
endif()

set(PERTURB_NONBONDED_HEADER "${PERTURB_NONBONDED_INSTALL_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/PerturbNonbondedForce.h")
set(PERTURB_NONBONDED_INTERNAL_HEADER "${PERTURB_NONBONDED_INSTALL_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/internal/PerturbNonbondedForceImpl.h")
set(PERTURB_NONBONDED_CORE_LIBRARY "${PERTURB_NONBONDED_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/libPerturbNonbondedPlugin${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(PERTURB_NONBONDED_CUDA_LIBRARY "${PERTURB_NONBONDED_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/plugins/libPerturbNonbondedForceCUDA${CMAKE_SHARED_LIBRARY_SUFFIX}")
foreach(required_file
        "${PERTURB_NONBONDED_HEADER}"
        "${PERTURB_NONBONDED_INTERNAL_HEADER}"
        "${PERTURB_NONBONDED_CORE_LIBRARY}"
        "${PERTURB_NONBONDED_CUDA_LIBRARY}")
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR
            "RelocationMC requires a pre-installed CUDA-enabled PerturbNonbondedPlugin. "
            "Missing: ${required_file}. Install PerturbNonbondedPlugin first or set "
            "PERTURB_NONBONDED_INSTALL_DIR to its installation prefix.")
    endif()
endforeach()

find_package(CUDAToolkit REQUIRED)
find_library(OPENMM_CUDA_LIBRARY OpenMMCUDA
    HINTS "${OPENMM_DIR}/${CMAKE_INSTALL_LIBDIR}/plugins" "${OPENMM_DIR}/lib/plugins"
    REQUIRED)
add_subdirectory(platforms/cuda)

option(BUILD_PYTHON_WRAPPERS "Build the Python API" ON)
if(BUILD_PYTHON_WRAPPERS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
    find_program(SWIG_EXECUTABLE swig REQUIRED)
    add_subdirectory(python)
endif()
