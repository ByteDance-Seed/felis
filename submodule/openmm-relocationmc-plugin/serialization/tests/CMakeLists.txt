# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

ADD_EXECUTABLE(TestSerializeRelocationMC TestSerializeRelocationMC.cpp)
TARGET_LINK_LIBRARIES(TestSerializeRelocationMC PRIVATE RelocationMCPlugin ${RELOCATIONMC_OPENMM_TARGET})
ADD_TEST(NAME TestSerializeRelocationMC COMMAND TestSerializeRelocationMC)
