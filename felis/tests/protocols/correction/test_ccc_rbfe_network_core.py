# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Cover core cycle-closure behaviors for the CCC RBFE solver."""

import numpy as np


def test_triangle_cycle_closure_is_enforced(make_ccc_edge_df, ccc_runner, get_ccc_edge_ddg):
    """Verify exact closure and gauge fixing on a minimal triangle network.

    This test exercises the smallest non-trivial connected graph so that the
    solver must reconcile one inconsistent cycle while still producing node
    free energies whose component-wise gauge sums to zero.

    Args:
        make_ccc_edge_df: Fixture that builds the RBFE edge table used by the
            solver.
        ccc_runner: Fixture that invokes `run_ccc_from_edge_df` with the
            standard ligand, ddG, and sigma column names.
        get_ccc_edge_ddg: Fixture that extracts one corrected edge ddG from the
            solver output.
    """
    df = make_ccc_edge_df(
        lig1=["A", "B", "A"],
        lig2=["B", "C", "C"],
        ddg=[1.0, 1.0, 2.2],
        sigma=[0.1, 0.1, 0.1],
    )
    res = ccc_runner(df)
    out = res.edge_df

    y_ab = get_ccc_edge_ddg(out, "A", "B")
    y_bc = get_ccc_edge_ddg(out, "B", "C")
    y_ac = get_ccc_edge_ddg(out, "A", "C")

    # Closure for A->B + B->C - A->C must be exactly 0 up to numerical tolerance.
    assert abs((y_ab + y_bc) - y_ac) < 1e-10

    # Sum-to-zero gauge per component
    assert abs(float(res.node_df["dG (kcal/mol)"].sum())) < 1e-10


def test_paper_easy_example(make_ccc_edge_df, ccc_runner, get_ccc_edge_ddg):
    """Reproduce the closed-form three-edge example from the CCC paper.

    The values come from the Appendix A toy system, where the corrected edge
    weights can be written analytically as the raw values minus one third of the
    cycle residual.

    Args:
        make_ccc_edge_df: Fixture that builds the RBFE edge table used by the
            solver.
        ccc_runner: Fixture that invokes `run_ccc_from_edge_df` with the
            standard ligand, ddG, and sigma column names.
        get_ccc_edge_ddg: Fixture that extracts one corrected edge ddG from the
            solver output.
    """
    # Appendix A in paper
    # dx.doi.org/10.1021/ct300911a | J. Chem. Theory Comput. 2013, 9, 1282-1293
    df = make_ccc_edge_df(
        lig1=["2", "3", "1"],
        lig2=["1", "2", "3"],
        ddg=[1.0, 2.0, -4.0],
        sigma=[0.1, 0.1, 0.1],
    )
    res = ccc_runner(df)
    out = res.edge_df

    y_21 = get_ccc_edge_ddg(out, "2", "1")
    y_32 = get_ccc_edge_ddg(out, "3", "2")
    y_13 = get_ccc_edge_ddg(out, "1", "3")
    assert abs(y_21 + y_32 + y_13) < 1e-10
    delta = 1.0 + 2.0 - 4.0
    assert np.isclose(y_21, 1.0 - delta / 3.0, atol=1e-10)
    assert np.isclose(y_32, 2.0 - delta / 3.0, atol=1e-10)
    assert np.isclose(y_13, -4.0 - delta / 3.0, atol=1e-10)
