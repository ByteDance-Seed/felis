# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class MinimizeRelaxOption(Enum):
    """Define the minimize/relax strategy option.

    Allow constructing an enum instance from multiple user-facing formats:

    - `int`: Use the underlying numeric value (e.g., `1`).
    - `str` of `int`: Use a digit string (e.g., `"1"`).
    - Case-insensitive name: Use the option name regardless of case
      (e.g., `L_BFGS`, `l_bfgs`).
    - Alias name: Accept common aliases (e.g., `bfgs` maps to `l_bfgs`,
      `FIRE2.0` maps to `FIRE2`).

    This behavior is implemented via the `_missing_` hook.
    """

    none = 0
    l_bfgs = 1
    heating = 2
    brownian = 3
    fire2 = 4

    @classmethod
    def _missing_(cls, value):
        mapping = {
            0: cls.none,
            1: cls.l_bfgs,
            2: cls.heating,
            3: cls.brownian,
            4: cls.fire2,
        }
        fields_low = [f.name.lower() for f in cls]

        if isinstance(value, int):
            if value in mapping:
                return mapping[value]
            return None

        if isinstance(value, str):
            v = value.strip()
            if v.isdigit():
                iv = int(v)
                if iv in mapping:
                    return mapping[iv]
                return None
            low = v.lower()
            if low == "bfgs":
                return cls.l_bfgs
            elif low == "fire2.0":
                return cls.fire2
            if low in fields_low:
                return cls[low]
            return None

        return None


class IntegratorNameOption(Enum):
    LangevinMiddleIntegrator = "LangevinMiddleIntegrator"
    BrownianIntegrator = "BrownianIntegrator"
    FIRE2 = "FIRE2"

    @classmethod
    def _missing_(cls, value):
        mapping = dict([(f.name.lower(), f) for f in cls])
        if isinstance(value, str):
            v = value.strip()
            low = v.lower()
            if low in mapping.keys():
                return mapping[low]
            return None
        return None
