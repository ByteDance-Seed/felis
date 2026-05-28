# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys


def install_shims_openmm() -> None:

    # openmm (shim)
    openmm_mod = importlib.import_module("felis.tests.testkit.openmm")
    sys.modules["openmm"] = openmm_mod

    # openmm.app (shim)
    app_mod = importlib.import_module("felis.tests.testkit.openmm.app")
    sys.modules["openmm.app"] = app_mod
    openmm_mod.app = app_mod

    # openmm.unit (real)
    from .unit import _load_real_openmm_unit
    unit_mod = _load_real_openmm_unit()
    openmm_mod.unit = unit_mod
