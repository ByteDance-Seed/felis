# Adapted from the OpenMM example plugin
# (https://github.com/openmm/openmmexampleplugin), MIT License.
# Portions copyright (c) 2014-2021 Stanford University and the Authors.
# See THIRD_PARTY_NOTICES for the full MIT notice.
# Modifications copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT

#
# Testing
#

# Automatically create tests using files named "Test*.cpp"
FILE(GLOB TEST_PROGS "*Test*.cpp")
FOREACH(TEST_PROG ${TEST_PROGS})
    GET_FILENAME_COMPONENT(TEST_ROOT ${TEST_PROG} NAME_WE)

    # Link with shared library

    ADD_EXECUTABLE(${TEST_ROOT} ${TEST_PROG})
    TARGET_LINK_LIBRARIES(${TEST_ROOT} ${SHARED_PERTURB_TARGET} ${SHARED_REFERENCE_TARGET})
    TARGET_INCLUDE_DIRECTORIES(${TEST_ROOT} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    SET_TARGET_PROPERTIES(${TEST_ROOT} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
    ADD_TEST(${TEST_ROOT} ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT})
    
ENDFOREACH(TEST_PROG ${TEST_PROGS})
