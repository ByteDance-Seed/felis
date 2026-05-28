# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from felis.protocols.boresch.dg_correction import BR_correction_2006, BR_correction_2023


def test_br_correction_harmonic_coeff_equivalence():
    # For both formulas: harmonic=False applies a coeff=2 internally.
    # Therefore: BR(harmonic=False, K) == BR(harmonic=True, 2*K).
    T = 300.0
    r0_nm = 0.3
    theta0_deg = 100.0
    alpha0_deg = 80.0
    Kr, Ka, Kd = 10.0, 20.0, 30.0

    got_2006 = BR_correction_2006(False, T, r0_nm, theta0_deg, 0.0, 0.0, 0.0, 0.0, Kr, Ka, Kd)
    exp_2006 = BR_correction_2006(True, T, r0_nm, theta0_deg, 0.0, 0.0, 0.0, 0.0, 2 * Kr, 2 * Ka, 2 * Kd)
    assert got_2006 == pytest.approx(exp_2006, rel=1e-12, abs=1e-12)

    got_2023 = BR_correction_2023(False, T, r0_nm, theta0_deg, 0.0, alpha0_deg, 0.0, 0.0, Kr, Ka, Kd)
    exp_2023 = BR_correction_2023(True, T, r0_nm, theta0_deg, 0.0, alpha0_deg, 0.0, 0.0, 2 * Kr, 2 * Ka, 2 * Kd)
    assert got_2023 == pytest.approx(exp_2023, rel=1e-12, abs=1e-12)


def test_br_correction_increases_with_kr():
    T = 300.0
    r0_nm = 0.3
    theta0_deg = 100.0
    alpha0_deg = 80.0
    Ka, Kd = 20.0, 30.0

    c1_2006 = BR_correction_2006(True, T, r0_nm, theta0_deg, 0.0, 0.0, 0.0, 0.0, 10.0, Ka, Kd)
    c2_2006 = BR_correction_2006(True, T, r0_nm, theta0_deg, 0.0, 0.0, 0.0, 0.0, 40.0, Ka, Kd)
    assert c2_2006 > c1_2006

    c1_2023 = BR_correction_2023(True, T, r0_nm, theta0_deg, 0.0, alpha0_deg, 0.0, 0.0, 10.0, Ka, Kd)
    c2_2023 = BR_correction_2023(True, T, r0_nm, theta0_deg, 0.0, alpha0_deg, 0.0, 0.0, 40.0, Ka, Kd)
    assert c2_2023 > c1_2023


def test_br_correction_regression_values():
    # A small regression check to catch accidental formula/units changes.
    T = 300.0
    r0_nm = 0.3
    theta0_deg = 100.0
    alpha0_deg = 80.0
    Kr, Ka, Kd = 10.0, 20.0, 30.0

    c2006 = BR_correction_2006(True, T, r0_nm, theta0_deg, 0.0, 0.0, 0.0, 0.0, Kr, Ka, Kd)
    c2023 = BR_correction_2023(True, T, r0_nm, theta0_deg, 0.0, alpha0_deg, 0.0, 0.0, Kr, Ka, Kd)

    assert c2006 == pytest.approx(8.875541079956676, rel=1e-12, abs=1e-12)
    assert c2023 == pytest.approx(10.131193864045196, rel=1e-12, abs=1e-12)
    assert c2023 > c2006
