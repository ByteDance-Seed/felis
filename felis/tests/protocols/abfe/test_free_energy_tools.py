# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from felis.protocols.abfe.free_energy_tools import calc_bar
from felis.protocols.abfe.free_energy_tools import calc_raw_dG
from felis.protocols.abfe.free_energy_tools import fermi_dirac


def test_calc_raw_dG_constant_input_has_zero_stdev():
    x = np.full(20, 3.25)
    dG, stdev = calc_raw_dG(x)

    assert dG == pytest.approx(3.25)
    assert stdev == pytest.approx(0.0)


def test_calc_raw_dG_shift_invariance():
    x = np.array([0.0, 0.5, 1.0, 2.0, -1.5])
    c = 420
    dG1, stdev1 = calc_raw_dG(x)
    dG2, stdev2 = calc_raw_dG(x + c)

    assert dG2 == pytest.approx(dG1 + c)
    assert stdev2 == pytest.approx(stdev1)


def test_calc_raw_dG_matches_direct_formula_for_small_values():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.0, scale=0.2, size=50)

    dG, stdev = calc_raw_dG(x)
    expected = -np.log(np.mean(np.exp(-x)))

    assert dG == pytest.approx(expected)
    assert np.isfinite(stdev)
    assert stdev >= 0.0


def test_fermi_dirac_symmetry_and_bounds():
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    fx = fermi_dirac(x)
    fnx = fermi_dirac(-x)

    np.testing.assert_allclose(fx + fnx, np.ones_like(x), rtol=0, atol=1e-12)
    assert np.all(fx >= 0.0)
    assert np.all(fx <= 1.0)


def test_calc_bar_returns_zero_for_symmetric_inputs():
    # If ui == vj and nu == nv, c0 == 0 is an exact fixed point.
    ui = np.linspace(-2.0, 2.0, 101)
    vj = ui.copy()
    c0, stdev = calc_bar(ui, vj)

    assert c0 == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(stdev)
    assert stdev >= 0.0


def test_calc_bar_returns_finite_for_basic_case():
    ui = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    vj = np.array([-0.1, 0.2, -0.3, 0.4, -0.5])
    c0, stdev = calc_bar(ui, vj)

    assert np.isfinite(c0)
    assert np.isfinite(stdev)
    assert stdev >= 0.0


def test_calc_bar_logs_warning_when_not_converged(monkeypatch, caplog):
    import felis.protocols.abfe.free_energy_tools as fe

    call_count = {"n": 0}

    def fake_fermi_dirac(x):
        # Force a constant non-zero Newton step (dc) so the solver never
        # satisfies abs(dc) < eps and hits the max-iteration warning branch.
        call_count["n"] += 1
        val = 0.8 if call_count["n"] % 2 == 1 else 0.2
        x_arr = np.asarray(x, dtype=float)
        return np.full_like(x_arr, val, dtype=float)

    monkeypatch.setattr(fe, "fermi_dirac", fake_fermi_dirac)

    ui = np.zeros(5)
    vj = np.zeros(7)
    with caplog.at_level("WARNING"):
        c0, stdev = fe.calc_bar(ui, vj)

    assert "bar did not converge" in caplog.text
    assert np.isfinite(c0)
    assert np.isfinite(stdev)
