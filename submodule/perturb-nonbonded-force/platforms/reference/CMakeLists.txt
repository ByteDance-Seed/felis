# Adapted from the OpenMM example plugin
# (https://github.com/openmm/openmmexampleplugin), MIT License.
# Portions copyright (c) 2014-2021 Stanford University and the Authors.
# See THIRD_PARTY_NOTICES for the full MIT notice.
# Modifications copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT

#---------------------------------------------------
# OpenMM PerturbNonbonded Plugin Reference Platform
#----------------------------------------------------

# Collect up information about the version of the OpenMM library we're building
# and make it available to the code so it can be built into the binaries.

SET(OPENMM_PERTURB_REFERENCE_LIBRARY_NAME PerturbNonbondedForceReference)

SET(SHARED_REFERENCE_TARGET ${OPENMM_PERTURB_REFERENCE_LIBRARY_NAME})


# These are all the places to search for header files which are
# to be part of the API.
SET(API_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include/internal")

# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)

# collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)

FILE(GLOB_RECURSE src_files  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.c)
FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/reference/include)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/reference/src)

# Create the library

INCLUDE_DIRECTORIES(${REFERENCE_INCLUDE_DIR})

ADD_LIBRARY(${SHARED_REFERENCE_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})

TARGET_LINK_LIBRARIES(${SHARED_REFERENCE_TARGET} OpenMM)
TARGET_LINK_LIBRARIES(${SHARED_REFERENCE_TARGET} ${PERTURB_LIBRARY_NAME})
SET_TARGET_PROPERTIES(${SHARED_REFERENCE_TARGET} PROPERTIES
    COMPILE_FLAGS "-DOPENMM_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    INSTALL_RPATH "$ORIGIN/..")

INSTALL(TARGETS ${SHARED_REFERENCE_TARGET} DESTINATION ${CMAKE_INSTALL_LIBDIR}/plugins)
IF(BUILD_TESTING)
    ADD_SUBDIRECTORY(tests)
ENDIF()
