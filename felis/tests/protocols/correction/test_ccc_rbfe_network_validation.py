# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Check input-validation paths for the CCC RBFE solver."""

import math

import pytest


def test_disconnected_components_raise_error(make_ccc_edge_df, ccc_runner):
    """Verify that CCC rejects graphs with more than one connected component.

    Args:
        make_ccc_edge_df: Fixture that builds the RBFE edge table used by the
            solver.
        ccc_runner: Fixture that invokes `run_ccc_from_edge_df` with the
            standard ligand, ddG, and sigma column names.
    """
    df = make_ccc_edge_df(
        lig1=["A", "B", "A", "D"],
        lig2=["B", "C", "C", "E"],
        ddg=[1.0, 1.0, 2.1, -0.3],
        sigma=[0.2, 0.2, 0.2, 0.1],
    )
    with pytest.raises(ValueError, match="Expected exactly 1 connected component"):
        ccc_runner(df)


def test_duplicate_directed_edges_raise_error(make_ccc_edge_df, ccc_runner):
    """Verify that CCC rejects repeated directed ligand transformations.

    Args:
        make_ccc_edge_df: Fixture that builds the RBFE edge table used by the
            solver.
        ccc_runner: Fixture that invokes `run_ccc_from_edge_df` with the
            standard ligand, ddG, and sigma column names.
    """
    df = make_ccc_edge_df(
        lig1=["A", "A"],
        lig2=["B", "B"],
        ddg=[1.0, 1.1],
        sigma=[0.1, 0.1],
    )
    with pytest.raises(ValueError, match="Duplicate directed edges"):
        ccc_runner(df)


@pytest.mark.parametrize(
    ("ddg", "sigma", "match"),
    [
        ([1.0, math.inf], [0.1, 0.1], "ddG column contains NaN/inf"),
        ([1.0, -math.inf], [0.1, 0.1], "ddG column contains NaN/inf"),
        ([1.0, 2.0], [0.1, math.inf], "sigma/stderr column contains NaN/inf"),
    ],
)
def test_non_finite_inputs_raise_error(make_ccc_edge_df, ccc_runner, ddg, sigma, match):
    """Verify that CCC rejects infinite ddG and sigma values.

    Args:
        make_ccc_edge_df: Fixture that builds the RBFE edge table used by the
            solver.
        ccc_runner: Fixture that invokes `run_ccc_from_edge_df` with the
            standard ligand, ddG, and sigma column names.
        ddg: Edge ddG values containing a non-finite entry or not.
        sigma: Edge sigma values containing a non-finite entry or not.
        match: Expected error message pattern.
    """
    df = make_ccc_edge_df(
        lig1=["A", "B"],
        lig2=["B", "C"],
        ddg=ddg,
        sigma=sigma,
    )
    with pytest.raises(ValueError, match=match):
        ccc_runner(df)
