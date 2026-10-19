# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: LGPL-3.0-or-later

set(PERTURB_SWIG_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/PerturbNonbondedPluginWrapper.cpp")
set(PERTURB_PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/PerturbNonbondedPlugin")
set(PERTURB_NONBONDED_PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    CACHE PATH "Destination for the installed PerturbNonbonded Python package")

add_custom_command(
    OUTPUT "${PERTURB_SWIG_WRAPPER}" "${PERTURB_PYTHON_PACKAGE_DIR}/PerturbNonbondedPlugin.py"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PERTURB_PYTHON_PACKAGE_DIR}"
    COMMAND "${SWIG_EXECUTABLE}" -python -c++
        -I"${OPENMM_INCLUDE_DIR}"
        -I"${CMAKE_SOURCE_DIR}/openmmapi/include"
        -outdir "${PERTURB_PYTHON_PACKAGE_DIR}"
        -o "${PERTURB_SWIG_WRAPPER}"
        "${CMAKE_CURRENT_SOURCE_DIR}/perturb_nonbonded_plugin.i"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/perturb_nonbonded_plugin.i"
        "${CMAKE_SOURCE_DIR}/openmmapi/include/PerturbNonbondedForce.h")

add_library(_PerturbNonbondedPlugin MODULE "${PERTURB_SWIG_WRAPPER}")
target_include_directories(_PerturbNonbondedPlugin PRIVATE
    "${CMAKE_SOURCE_DIR}/openmmapi/include"
    "${OPENMM_INCLUDE_DIR}")
target_link_libraries(_PerturbNonbondedPlugin PRIVATE
    ${SHARED_PERTURB_TARGET}
    OpenMM
    Python3::Module)
set_target_properties(_PerturbNonbondedPlugin PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${PERTURB_PYTHON_PACKAGE_DIR}"
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR};${OPENMM_DIR}/${CMAKE_INSTALL_LIBDIR}")

add_custom_target(PythonInstall ALL
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/PerturbNonbondedPlugin/__init__.py"
        "${PERTURB_PYTHON_PACKAGE_DIR}/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/PerturbNonbondedPlugin/testPluginInstallation.py"
        "${PERTURB_PYTHON_PACKAGE_DIR}/testPluginInstallation.py"
    DEPENDS _PerturbNonbondedPlugin)

install(TARGETS _PerturbNonbondedPlugin
    LIBRARY DESTINATION "${PERTURB_NONBONDED_PYTHON_INSTALL_DIR}/PerturbNonbondedPlugin")
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/PerturbNonbondedPlugin/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerturbNonbondedPlugin/testPluginInstallation.py"
    "${PERTURB_PYTHON_PACKAGE_DIR}/PerturbNonbondedPlugin.py"
    DESTINATION "${PERTURB_NONBONDED_PYTHON_INSTALL_DIR}/PerturbNonbondedPlugin")
