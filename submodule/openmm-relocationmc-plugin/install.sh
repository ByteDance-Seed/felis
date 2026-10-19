#!/usr/bin/env bash
# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

set -euo pipefail

PROJECT="RelocationMCPlugin"
SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-"${SOURCE_DIR}/build"}
OPENMM_DIR=${OPENMM_DIR:-}
PERTURB_NONBONDED_DIR=${PERTURB_NONBONDED_DIR:-}
INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX:-}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-}
PYTHON_INSTALL_DIR=${PYTHON_INSTALL_DIR:-}
BUILD_JOBS=${BUILD_JOBS:-}
BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
CUDA_TEST_MODE=auto
BUILD_PYTHON=on
RUN_TESTS=on
CMAKE_EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Build, test, and install RelocationMCPlugin from source.

CUDA and a CUDA-enabled PerturbNonbondedPlugin installation are required.

Usage: ./install.sh [options] [-- extra-cmake-arguments]

Options:
  --openmm-dir PATH       OpenMM installation prefix. By default, detect it
                          from the selected Python environment.
  --perturb-prefix PATH   PerturbNonbonded installation prefix. Defaults to
                          the native installation prefix, then OPENMM_DIR.
  --prefix PATH           Native installation prefix. Defaults to OPENMM_DIR.
  --build-dir PATH        CMake build directory. Defaults to ./build.
  --python PATH           Python executable. Defaults to python3.
  --python-install-dir P  Python package destination. Defaults to the selected
                          interpreter's site-packages directory.
  --jobs N                Number of parallel build jobs.
  --build-type TYPE       CMake build type. Defaults to Release.
  --cuda-tests MODE       Run CUDA tests: auto, on, or off.
  --no-python             Do not build or install the Python wrapper.
  --skip-tests            Do not run C++ or Python smoke tests.
  -h, --help              Show this help message.

Environment equivalents:
  OPENMM_DIR, PERTURB_NONBONDED_DIR, CMAKE_INSTALL_PREFIX, BUILD_DIR,
  PYTHON_EXECUTABLE, PYTHON_INSTALL_DIR, BUILD_JOBS, and CMAKE_BUILD_TYPE.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

normalize_mode() {
    case "$1" in
        auto|AUTO) echo auto ;;
        on|ON|yes|YES|true|TRUE|1) echo on ;;
        off|OFF|no|NO|false|FALSE|0) echo off ;;
        *) die "expected auto, on, or off; got '$1'" ;;
    esac
}

while (($#)); do
    case "$1" in
        --openmm-dir) (($# >= 2)) || die "$1 requires a path"; OPENMM_DIR=$2; shift 2 ;;
        --perturb-prefix) (($# >= 2)) || die "$1 requires a path"; PERTURB_NONBONDED_DIR=$2; shift 2 ;;
        --prefix) (($# >= 2)) || die "$1 requires a path"; INSTALL_PREFIX=$2; shift 2 ;;
        --build-dir) (($# >= 2)) || die "$1 requires a path"; BUILD_DIR=$2; shift 2 ;;
        --python) (($# >= 2)) || die "$1 requires a path"; PYTHON_EXECUTABLE=$2; shift 2 ;;
        --python-install-dir) (($# >= 2)) || die "$1 requires a path"; PYTHON_INSTALL_DIR=$2; shift 2 ;;
        --jobs) (($# >= 2)) || die "$1 requires a positive integer"; BUILD_JOBS=$2; shift 2 ;;
        --build-type) (($# >= 2)) || die "$1 requires a value"; BUILD_TYPE=$2; shift 2 ;;
        --cuda-tests) (($# >= 2)) || die "$1 requires auto, on, or off"; CUDA_TEST_MODE=$(normalize_mode "$2"); shift 2 ;;
        --no-python) BUILD_PYTHON=off; shift ;;
        --skip-tests) RUN_TESTS=off; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; CMAKE_EXTRA_ARGS=("$@"); break ;;
        *) die "unknown option '$1'" ;;
    esac
done

command -v cmake >/dev/null 2>&1 || die "cmake is required"
if [[ -z "${PYTHON_EXECUTABLE}" ]]; then
    PYTHON_EXECUTABLE=$(command -v python3 || command -v python || true)
elif [[ "${PYTHON_EXECUTABLE}" != */* ]]; then
    PYTHON_EXECUTABLE=$(command -v "${PYTHON_EXECUTABLE}" || true)
fi
[[ -n "${PYTHON_EXECUTABLE}" && -x "${PYTHON_EXECUTABLE}" ]] || die "a working Python executable is required"

if [[ -z "${OPENMM_DIR}" ]]; then
    OPENMM_DIR=$("${PYTHON_EXECUTABLE}" - <<'PY' 2>/dev/null || true
import os
from openmm import version
print(os.path.dirname(os.path.abspath(version.openmm_library_path)))
PY
    )
fi
[[ -d "${OPENMM_DIR}/include/openmm" ]] || die "OpenMM headers were not found under ${OPENMM_DIR}/include"
[[ -d "${OPENMM_DIR}/lib" ]] || die "OpenMM libraries were not found under ${OPENMM_DIR}/lib"
INSTALL_PREFIX=${INSTALL_PREFIX:-${OPENMM_DIR}}

has_perturb_installation() {
    local prefix=$1
    [[ -f "${prefix}/include/PerturbNonbondedForce.h" ]] &&
        [[ -f "${prefix}/include/internal/PerturbNonbondedForceImpl.h" ]] &&
        [[ -f "${prefix}/lib/libPerturbNonbondedPlugin.so" ]] &&
        [[ -f "${prefix}/lib/plugins/libPerturbNonbondedForceCUDA.so" ]]
}

if [[ -z "${PERTURB_NONBONDED_DIR}" ]]; then
    for candidate in "${INSTALL_PREFIX}" "${OPENMM_DIR}"; do
        if has_perturb_installation "${candidate}"; then
            PERTURB_NONBONDED_DIR=${candidate}
            break
        fi
    done
fi
[[ -n "${PERTURB_NONBONDED_DIR}" ]] || \
    die "RelocationMC requires a pre-installed CUDA-enabled PerturbNonbondedPlugin; install it first or pass --perturb-prefix"
has_perturb_installation "${PERTURB_NONBONDED_DIR}" || \
    die "incomplete PerturbNonbondedPlugin installation under ${PERTURB_NONBONDED_DIR}; required headers, core library, and CUDA plugin must all be installed"

if command -v ldd >/dev/null 2>&1; then
    PERTURB_LDD_OUTPUT=$(LD_LIBRARY_PATH="${PERTURB_NONBONDED_DIR}/lib:${PERTURB_NONBONDED_DIR}/lib/plugins:${OPENMM_DIR}/lib:${OPENMM_DIR}/lib/plugins:${LD_LIBRARY_PATH:-}" \
        ldd "${PERTURB_NONBONDED_DIR}/lib/plugins/libPerturbNonbondedForceCUDA.so" 2>&1) || \
        die "the pre-installed PerturbNonbonded CUDA plugin could not be inspected"
    [[ "${PERTURB_LDD_OUTPUT}" != *"not found"* ]] || \
        die "the pre-installed PerturbNonbonded CUDA plugin has unresolved shared-library dependencies"
fi

if [[ -z "${BUILD_JOBS}" ]]; then
    BUILD_JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
fi
[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"

if [[ "${BUILD_PYTHON}" == on ]]; then
    command -v swig >/dev/null 2>&1 || die "swig is required to build the Python wrapper"
    if [[ -z "${PYTHON_INSTALL_DIR}" ]]; then
        PYTHON_INSTALL_DIR=$("${PYTHON_EXECUTABLE}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
    fi
fi

command -v nvcc >/dev/null 2>&1 || die "CUDA is required, but nvcc was not found"
[[ -f "${OPENMM_DIR}/lib/plugins/libOpenMMCUDA.so" ]] || \
    die "CUDA is required, but ${OPENMM_DIR}/lib/plugins/libOpenMMCUDA.so was not found"

CUDA_TEST_VALUE=OFF
if [[ "${RUN_TESTS}" == on && "${CUDA_TEST_MODE}" != off ]]; then
    if "${PYTHON_EXECUTABLE}" - <<'PY' >/dev/null 2>&1
import openmm
openmm.Platform.loadPluginsFromDirectory(openmm.Platform.getDefaultPluginsDirectory())
openmm.Platform.getPlatformByName("CUDA")
PY
    then
        CUDA_TEST_VALUE=ON
    elif [[ "${CUDA_TEST_MODE}" == on ]]; then
        die "CUDA tests were requested, but the OpenMM CUDA platform cannot load"
    fi
fi

CMAKE_ARGS=(
    -S "${SOURCE_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
    -DOPENMM_DIR="${OPENMM_DIR}"
    -DPERTURB_NONBONDED_INSTALL_DIR="${PERTURB_NONBONDED_DIR}"
    -DBUILD_TESTING="$([[ "${RUN_TESTS}" == on ]] && echo ON || echo OFF)"
    -DBUILD_PYTHON_WRAPPERS="$([[ "${BUILD_PYTHON}" == on ]] && echo ON || echo OFF)"
)
if [[ "${BUILD_PYTHON}" == on ]]; then
    CMAKE_ARGS+=(
        "-DPython3_EXECUTABLE=${PYTHON_EXECUTABLE}"
        "-DRELOCATIONMC_PYTHON_INSTALL_DIR=${PYTHON_INSTALL_DIR}"
    )
fi
CUDA_COMPILER=$(command -v nvcc)
CMAKE_ARGS+=("-DCUDAToolkit_ROOT=$(cd "$(dirname "${CUDA_COMPILER}")/.." && pwd)")
CMAKE_ARGS+=("${CMAKE_EXTRA_ARGS[@]}")

echo "Installing ${PROJECT}"
echo "  OpenMM:       ${OPENMM_DIR}"
echo "  Perturb:      ${PERTURB_NONBONDED_DIR}"
echo "  Prefix:       ${INSTALL_PREFIX}"
echo "  Build:        ${BUILD_DIR}"
echo "  Python:       ${BUILD_PYTHON}"
echo "  CUDA build:   required"
echo "  CUDA tests:   ${CUDA_TEST_VALUE}"

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"
if [[ "${RUN_TESTS}" == on ]]; then
    if [[ "${CUDA_TEST_VALUE}" == ON ]]; then
        ctest --test-dir "${BUILD_DIR}" --output-on-failure
    else
        ctest --test-dir "${BUILD_DIR}" --output-on-failure -E Cuda
    fi
fi
cmake --install "${BUILD_DIR}"

if [[ "${RUN_TESTS}" == on && "${BUILD_PYTHON}" == on ]]; then
    unset OPENMM_PLUGIN_DIR
    export RELOCATIONMC_PLUGIN_DIR="${INSTALL_PREFIX}/lib/plugins"
    export RELOCATIONMC_SMOKE_LOAD_CUDA="$([[ "${CUDA_TEST_VALUE}" == ON ]] && echo 1 || echo 0)"
    if [[ "$(uname -s)" == Darwin ]]; then
        export DYLD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${INSTALL_PREFIX}/lib/plugins:${PERTURB_NONBONDED_DIR}/lib:${PERTURB_NONBONDED_DIR}/lib/plugins:${OPENMM_DIR}/lib:${OPENMM_DIR}/lib/plugins:${DYLD_LIBRARY_PATH:-}"
    else
        export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${INSTALL_PREFIX}/lib/plugins:${PERTURB_NONBONDED_DIR}/lib:${PERTURB_NONBONDED_DIR}/lib/plugins:${OPENMM_DIR}/lib:${OPENMM_DIR}/lib/plugins:${LD_LIBRARY_PATH:-}"
    fi
    "${PYTHON_EXECUTABLE}" -m RelocationMCPlugin.testPluginInstallation
fi

echo "${PROJECT} was installed successfully."
