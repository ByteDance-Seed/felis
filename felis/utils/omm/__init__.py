# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import openmm
from packaging import version

_openmm_version = version.parse(openmm.__version__)
if not version.parse("8.5") <= _openmm_version < version.parse("8.6"):
    raise RuntimeError(f"OpenMM version >=8.5,<8.6 is required, but found {openmm.__version__}")

del openmm, version, _openmm_version
