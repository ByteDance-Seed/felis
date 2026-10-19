# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

ADD_EXECUTABLE(TestRelocationMC TestRelocationMC.cpp)
TARGET_INCLUDE_DIRECTORIES(TestRelocationMC PRIVATE ${CMAKE_SOURCE_DIR}/platforms/cuda/include)
TARGET_LINK_LIBRARIES(TestRelocationMC PRIVATE RelocationMCPlugin ${RELOCATIONMC_OPENMM_TARGET})
ADD_TEST(NAME TestRelocationMC COMMAND TestRelocationMC)

ADD_EXECUTABLE(TestMTMAugmentedPool TestMTMAugmentedPool.cpp)
ADD_TEST(NAME TestMTMAugmentedPool COMMAND TestMTMAugmentedPool)
