# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_real_openmm_unit() -> ModuleType:
    """Load the real `openmm.unit` module without importing the real `openmm` package."""
    dist = importlib.metadata.distribution("openmm")
    unit_file = None
    for f in dist.files or []:
        p = str(f)
        if p.endswith("openmm/unit/__init__.py"):
            unit_file = dist.locate_file(f)
            break

    unit_path = Path(unit_file)
    spec = importlib.util.spec_from_file_location(
        "openmm.unit",
        unit_path,
        submodule_search_locations=[str(unit_path.parent)],
    )
    mod = importlib.util.module_from_spec(spec)

    sys.modules["openmm.unit"] = mod
    spec.loader.exec_module(mod)
    return mod
