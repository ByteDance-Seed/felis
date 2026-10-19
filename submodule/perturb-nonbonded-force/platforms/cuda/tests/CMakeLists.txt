# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: LGPL-3.0-or-later

# Make sure we find these locally before looking in OpenMM/include if
# OpenMM was previously installed there.
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Build the tests

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/openmmapi/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/serialization/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/platforms/reference/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/platforms/reference/src)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/platforms/cuda/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/platforms/cuda/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../../openmmapi/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/tests)

MACRO(TEST_CUDA_PERTURB_NONBONDED_PLUGIN_TARGET target)
    ADD_EXECUTABLE(${target} ${target}.cpp ${CMAKE_SOURCE_DIR}/platforms/reference/src/ReferencePerturbNonbondedKernels.cpp ${CMAKE_SOURCE_DIR}/platforms/reference/src/ReferencePerturbNonbondedKernelFactory.cpp ${CMAKE_SOURCE_DIR}/platforms/reference/src/ReferencePerturbLJCoulombIxn.cpp)
    TARGET_LINK_LIBRARIES(${target} OpenMM OpenMMCUDA ${PERTURB_LIBRARY_NAME} ${PERTURB_CUDA_LIBRARY_NAME} ${OPENMM_PERTURB_REFERENCE_LIBRARY_NAME} CUDA::cuda_driver)
    SET_TARGET_PROPERTIES(${target} PROPERTIES COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}" LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")
    #ADD_TEST(${target} ${target})
    ADD_TEST(NAME ${target}Single COMMAND ${target} "single")
    ADD_TEST(NAME ${target}Mixed COMMAND ${target} "mixed")
    ADD_TEST(NAME ${target}Double COMMAND ${target} "double")
ENDMACRO(TEST_CUDA_PERTURB_NONBONDED_PLUGIN_TARGET)

TEST_CUDA_PERTURB_NONBONDED_PLUGIN_TARGET(TestCudaPerturbNonbondedForce)
