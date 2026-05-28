# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import openmm
from packaging import version

if version.parse(openmm.__version__) < version.parse("8.3"):
    raise RuntimeError(f"OpenMM version >= 8.3 is required, but found {openmm.__version__}")

del openmm, version
