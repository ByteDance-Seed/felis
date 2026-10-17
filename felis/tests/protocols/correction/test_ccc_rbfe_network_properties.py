# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Exercise higher-level invariants of the CCC RBFE solver."""

import numpy as np


def test_cycle_closure_correction_is_idempotent(make_ccc_edge_df, ccc_runner, get_ccc_edge_ddg):
    """Verify that reapplying CCC leaves a corrected cycle-consistent network unchanged.

    Start from a triangle network with a non-zero cycle residual so the first
    correction must modify the edge ddGs. Then rebuild a second-pass input table
    from the first pass `CCC ddG (kcal/mol)` output and rerun CCC. This checks
    the two idempotence guarantees that matter here: the corrected edge ddGs stay
    identical, and the recovered node `dG (kcal/mol)` values also stay identical.
    Keep an explicit closure assertion on the second pass as a guard that the
    rerun remains cycle-consistent.

    The second-pass table must rename `Lig 1` / `Lig 2` / `CCC ddG (kcal/mol)`
    back to the raw input schema expected by `run_ccc_from_edge_df`, instead of
    feeding the first-pass output columns directly.

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

    res1 = ccc_runner(df)

    # Rebuild a fresh raw-input table from the corrected edge output before the
    # second pass, instead of reusing the first-pass output schema directly.
    df_second_pass = res1.edge_df[["Lig 1", "Lig 2", "CCC ddG (kcal/mol)", "sigma"]].rename(columns={
        "Lig 1": "Ligand 1",
        "Lig 2": "Ligand 2",
        "CCC ddG (kcal/mol)": "ddG (kcal/mol)",
    })
    res2 = ccc_runner(df_second_pass)

    edges1 = res1.edge_df.sort_values(["Lig 1", "Lig 2"]).reset_index(drop=True)
    edges2 = res2.edge_df.sort_values(["Lig 1", "Lig 2"]).reset_index(drop=True)
    # Idempotence on the edge space: rerunning CCC must preserve corrected ddGs.
    np.testing.assert_array_equal(edges1[["Lig 1", "Lig 2"]].to_numpy(), edges2[["Lig 1", "Lig 2"]].to_numpy())
    np.testing.assert_allclose(edges1["CCC ddG (kcal/mol)"], edges2["CCC ddG (kcal/mol)"], atol=1e-10)

    nodes1 = res1.node_df.sort_values("Ligand").reset_index(drop=True)
    nodes2 = res2.node_df.sort_values("Ligand").reset_index(drop=True)
    # Idempotence on the node space: the inferred node potentials must also match.
    np.testing.assert_array_equal(nodes1[["Ligand"]].to_numpy(), nodes2[["Ligand"]].to_numpy())
    np.testing.assert_allclose(nodes1["dG (kcal/mol)"], nodes2["dG (kcal/mol)"], atol=1e-10)

    y_ab = get_ccc_edge_ddg(edges2, "A", "B")
    y_bc = get_ccc_edge_ddg(edges2, "B", "C")
    y_ac = get_ccc_edge_ddg(edges2, "A", "C")
    # The second pass should still satisfy exact cycle closure.
    assert abs((y_ab + y_bc) - y_ac) < 1e-10


def test_weighting_downweights_large_sigma_edges(make_ccc_edge_df, ccc_runner, get_ccc_edge_ddg):
    """Verify that large-sigma measurements absorb more of the CCC correction.

    The A->B edge is assigned a much larger uncertainty than the other two
    edges, so weighted least squares should shift that edge more aggressively
    while still keeping the corrected cycle exactly closed.

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
        ddg=[4.0, 1.0, 2.0],
        sigma=[10.0, 0.1, 0.1],
    )
    res = ccc_runner(df)
    out = res.edge_df

    # Compare raw-minus-corrected residuals to confirm the noisiest edge moves most.
    row_ab = out[(out["Lig 1"] == "A") & (out["Lig 2"] == "B")].iloc[0]
    row_ac = out[(out["Lig 1"] == "A") & (out["Lig 2"] == "C")].iloc[0]
    row_bc = out[(out["Lig 1"] == "B") & (out["Lig 2"] == "C")].iloc[0]
    resid_ab = float(row_ab["ddG (kcal/mol)"] - row_ab["CCC ddG (kcal/mol)"])
    resid_ac = float(row_ac["ddG (kcal/mol)"] - row_ac["CCC ddG (kcal/mol)"])
    resid_bc = float(row_bc["ddG (kcal/mol)"] - row_bc["CCC ddG (kcal/mol)"])

    assert abs(resid_ab) > abs(resid_ac)
    assert abs(resid_ab) > abs(resid_bc)

    # Still enforces closure
    y_ab = get_ccc_edge_ddg(out, "A", "B")
    y_bc = get_ccc_edge_ddg(out, "B", "C")
    y_ac = get_ccc_edge_ddg(out, "A", "C")
    assert abs((y_ab + y_bc) - y_ac) < 1e-10
