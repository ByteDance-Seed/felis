# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.18)
project(OpenMMPerturbNonbondedPlugin VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)

set(OPENMM_DIR "/usr/local/openmm" CACHE PATH "OpenMM installation prefix")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_path(OPENMM_INCLUDE_DIR openmm/Context.h
    HINTS "${OPENMM_DIR}/include"
    REQUIRED)
find_library(OPENMM_LIBRARY OpenMM
    HINTS "${OPENMM_DIR}/${CMAKE_INSTALL_LIBDIR}" "${OPENMM_DIR}/lib"
    REQUIRED)
add_library(OpenMM UNKNOWN IMPORTED GLOBAL)
set_target_properties(OpenMM PROPERTIES
    IMPORTED_LOCATION "${OPENMM_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OPENMM_INCLUDE_DIR}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${OPENMM_DIR}" CACHE PATH "Installation prefix" FORCE)
endif()

file(GLOB PERTURB_API_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include/internal/*.h")
file(GLOB PERTURB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/serialization/src/*.cpp")

set(PERTURB_LIBRARY_NAME PerturbNonbondedPlugin)
set(SHARED_PERTURB_TARGET ${PERTURB_LIBRARY_NAME})
add_library(${SHARED_PERTURB_TARGET} SHARED ${PERTURB_SOURCES} ${PERTURB_API_HEADERS})
target_compile_definitions(${SHARED_PERTURB_TARGET} PRIVATE PERTURBNONBONDED_BUILDING_SHARED_LIBRARY)
target_include_directories(${SHARED_PERTURB_TARGET}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/serialization/include")
target_link_libraries(${SHARED_PERTURB_TARGET} PUBLIC OpenMM)
set_target_properties(${SHARED_PERTURB_TARGET} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    INSTALL_RPATH "$ORIGIN")

install(TARGETS ${SHARED_PERTURB_TARGET}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY openmmapi/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")

if(BUILD_TESTING)
    add_subdirectory(serialization/tests)
endif()

add_subdirectory(platforms/reference)

find_package(CUDAToolkit REQUIRED)
find_library(OPENMM_CUDA_LIBRARY OpenMMCUDA
    HINTS "${OPENMM_DIR}/${CMAKE_INSTALL_LIBDIR}/plugins" "${OPENMM_DIR}/lib/plugins"
    REQUIRED)
add_library(OpenMMCUDA UNKNOWN IMPORTED GLOBAL)
set_target_properties(OpenMMCUDA PROPERTIES
    IMPORTED_LOCATION "${OPENMM_CUDA_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OPENMM_INCLUDE_DIR}")
set(CUDA_LIBRARIES CUDA::cuda_driver)
set(CUDA_TOOLKIT_INCLUDE ${CUDAToolkit_INCLUDE_DIRS})
add_subdirectory(platforms/cuda)

option(NONBONDED_BUILD_PYTHON_WRAPPERS "Build the Python API" ON)
if(NONBONDED_BUILD_PYTHON_WRAPPERS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    find_program(SWIG_EXECUTABLE swig REQUIRED)
    add_subdirectory(python)
endif()
