# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any


def find_ideal_geometry(histogram_result: dict[str, Any]) -> dict[str, float]:
    """Find ideal Boresch geometry parameters from histogram peak locations."""
    binmax = histogram_result["binmax"]
    theta0 = round(binmax[0], 1)
    r0 = round(binmax[1], 3)
    alpha0 = round(binmax[2], 1)
    phi0 = round(binmax[3], 1)
    beta0 = round(binmax[4], 1)
    gamma0 = round(binmax[5], 1)
    return {
        "r0": float(r0),
        "theta0": float(theta0),
        "phi0": float(phi0),
        "alpha0": float(alpha0),
        "beta0": float(beta0),
        "gamma0": float(gamma0),
    }
