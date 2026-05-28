# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

def test_find_ideal_geometry_rounding_rules():
    from felis.protocols.boresch.find_ideal_geometry import find_ideal_geometry

    # The first 6 entries correspond to theta, r, alpha, phi, beta, gamma in that order.
    histogram_result = {
        "binmax": [
            12.34,
            0.123456,
            98.76,
            -179.99,
            0.04,
            359.96,
            0.0,
            0.0,
            0.0,
        ]
    }
    ideal = find_ideal_geometry(histogram_result)
    assert ideal == {
        "r0": 0.123,
        "theta0": 12.3,
        "phi0": -180.0,
        "alpha0": 98.8,
        "beta0": 0.0,
        "gamma0": 360.0,
    }
