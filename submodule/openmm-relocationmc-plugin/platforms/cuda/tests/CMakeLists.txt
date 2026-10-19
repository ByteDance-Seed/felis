# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

INCLUDE_DIRECTORIES(${CUDAToolkit_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/openmmapi/include)
INCLUDE_DIRECTORIES(${PERTURB_NONBONDED_INSTALL_DIR}/${CMAKE_INSTALL_INCLUDEDIR})

ADD_EXECUTABLE(TestCudaRelocationMC TestCudaRelocationMC.cpp)
TARGET_LINK_LIBRARIES(TestCudaRelocationMC PRIVATE
    ${SHARED_RELOCATIONMC_TARGET}
    ${SHARED_TARGET}
    "${PERTURB_NONBONDED_CORE_LIBRARY}"
    "${PERTURB_NONBONDED_CUDA_LIBRARY}"
    "${OPENMM_CUDA_LIBRARY}"
    ${RELOCATIONMC_OPENMM_TARGET}
    CUDA::cuda_driver)

ADD_TEST(NAME TestCudaRelocationMCSinglePmeStreamEnabled COMMAND TestCudaRelocationMC single 0 false)
ADD_TEST(NAME TestCudaRelocationMCMixedPmeStreamEnabled COMMAND TestCudaRelocationMC mixed 0 false)
ADD_TEST(NAME TestCudaRelocationMCDoublePmeStreamEnabled COMMAND TestCudaRelocationMC double 0 false)
ADD_TEST(NAME TestCudaRelocationMCSinglePmeStreamDisabled COMMAND TestCudaRelocationMC single 0 true)
ADD_TEST(NAME TestCudaRelocationMCMixedPmeStreamDisabled COMMAND TestCudaRelocationMC mixed 0 true)
ADD_TEST(NAME TestCudaRelocationMCDoublePmeStreamDisabled COMMAND TestCudaRelocationMC double 0 true)
