# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys


def install_shims_openmmtools() -> None:

    # openmmtools (shim)
    otools_mod = importlib.import_module("felis.tests.testkit.openmmtools")
    sys.modules["openmmtools"] = otools_mod
