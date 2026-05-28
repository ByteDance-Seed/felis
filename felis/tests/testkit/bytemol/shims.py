# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys


def install_shims_bytemol() -> None:

    # bytemol.core (shim)
    core_mod = importlib.import_module("felis.tests.testkit.bytemol.core")
    sys.modules["bytemol.core"] = core_mod

    # bytemol.toolkit.system_builder (shim)
    system_builder_mod = importlib.import_module("felis.tests.testkit.bytemol.toolkit.system_builder")
    sys.modules["bytemol.toolkit.system_builder"] = system_builder_mod
