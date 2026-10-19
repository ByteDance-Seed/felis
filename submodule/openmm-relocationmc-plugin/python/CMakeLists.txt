# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

FIND_PACKAGE(Python3 REQUIRED COMPONENTS NumPy)

SET(RELOCATIONMC_SWIG_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/RelocationMCPluginWrapper.cpp")
SET(RELOCATIONMC_PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/RelocationMCPlugin")
SET(RELOCATIONMC_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages" CACHE PATH
    "Prefix-relative destination for the installed RelocationMC Python package")
OPTION(RELOCATIONMC_INSTALL_PYTHON_WRAPPERS "Install the Python API with cmake --install" ON)

ADD_CUSTOM_COMMAND(
    OUTPUT "${RELOCATIONMC_SWIG_WRAPPER}" "${RELOCATIONMC_PYTHON_PACKAGE_DIR}/RelocationMCPlugin.py"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RELOCATIONMC_PYTHON_PACKAGE_DIR}"
    COMMAND "${SWIG_EXECUTABLE}" -c++ -python
        -I"${OPENMM_DIR}/include"
        -I"${CMAKE_SOURCE_DIR}/openmmapi/include"
        -outdir "${RELOCATIONMC_PYTHON_PACKAGE_DIR}"
        -o "${RELOCATIONMC_SWIG_WRAPPER}"
        "${CMAKE_CURRENT_SOURCE_DIR}/relocationmc_plugin.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/relocationmc_plugin.i"
            "${CMAKE_SOURCE_DIR}/openmmapi/include/RelocationMC.h")

ADD_LIBRARY(_RelocationMCPlugin MODULE "${RELOCATIONMC_SWIG_WRAPPER}")
TARGET_INCLUDE_DIRECTORIES(_RelocationMCPlugin PRIVATE
    "${CMAKE_SOURCE_DIR}/openmmapi/include"
    "${OPENMM_DIR}/include")
TARGET_LINK_LIBRARIES(_RelocationMCPlugin PRIVATE RelocationMCPlugin Python3::Module Python3::NumPy)
SET_TARGET_PROPERTIES(_RelocationMCPlugin PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${RELOCATIONMC_PYTHON_PACKAGE_DIR}"
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR};${OPENMM_DIR}/${CMAKE_INSTALL_LIBDIR}")

ADD_CUSTOM_TARGET(PythonInstall ALL
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/RelocationMCPlugin/__init__.py" "${RELOCATIONMC_PYTHON_PACKAGE_DIR}/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/RelocationMCPlugin/testPluginInstallation.py" "${RELOCATIONMC_PYTHON_PACKAGE_DIR}/testPluginInstallation.py"
    DEPENDS _RelocationMCPlugin)

IF(RELOCATIONMC_INSTALL_PYTHON_WRAPPERS)
    INSTALL(TARGETS _RelocationMCPlugin LIBRARY DESTINATION "${RELOCATIONMC_PYTHON_INSTALL_DIR}/RelocationMCPlugin")
    INSTALL(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/RelocationMCPlugin/__init__.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/RelocationMCPlugin/testPluginInstallation.py"
        "${RELOCATIONMC_PYTHON_PACKAGE_DIR}/RelocationMCPlugin.py"
        DESTINATION "${RELOCATIONMC_PYTHON_INSTALL_DIR}/RelocationMCPlugin")
ENDIF()
