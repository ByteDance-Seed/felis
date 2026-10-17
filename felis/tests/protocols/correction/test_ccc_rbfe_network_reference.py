# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Regress the CCC RBFE solver against the archived reference CSV suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bytemol.utils import get_data_file_path


def _iter_ccc_reference_csv_paths() -> list[Path]:
    """Return the sorted reference CSV paths for CCC regression coverage.

    The helper resolves the repository test-data root through
    `get_data_file_path` so the parameterized regression suite remains stable
    after moving this test file across directories.

    Returns:
        list[Path]: Sorted CSV paths under `felis/tests/testdata/ccc`.
    """
    testdata_root = Path(get_data_file_path("tests/testdata/README.md", "felis")).parent
    root = testdata_root / "ccc"
    return sorted(root.rglob("*.csv"))


@pytest.mark.parametrize("csv_path", _iter_ccc_reference_csv_paths())
def test_ccc_batch_against_reference_csv(csv_path: Path, ccc_runner):
    """Compare inferred node potentials against archived CCC edge references.
    Data originally provided by https://github.com/schrodinger/public_binding_free_energy_benchmark/tree/main/21_4_results/edge_predictions

    The reference CSVs store expected corrected edge values in kcal/mol. This
    test refits node potentials from the raw Bennett ddGs, reconstructs each
    corrected edge from the node map, and checks that the reconstructed values
    match the archived outputs within the historical tolerance.

    Args:
        csv_path: Path to one archived CCC regression CSV.
        ccc_runner: Fixture that invokes `run_ccc_from_edge_df` with the
            standard CCC column configuration.

    Raises:
        AssertionError: If the CSV schema is missing expected ligand or ddG
            columns, if duplicate directed edges are present, or if any ligand
            cannot be mapped back to an inferred node potential.
    """
    df = pd.read_csv(csv_path)

    # Accept both in-house and legacy ligand column naming conventions.
    if {"Lig 1", "Lig 2"}.issubset(df.columns):
        lig1_col = "Lig 1"
        lig2_col = "Lig 2"
    elif {"Ligand1", "Ligand2"}.issubset(df.columns):
        lig1_col = "Ligand1"
        lig2_col = "Ligand2"
    else:
        raise AssertionError(f"Unexpected ligand columns in {csv_path}: {df.columns.tolist()}")

    ddg_col = "Bennett ddG (kcal/mol)"
    # Reference CSV CCC values are generated under an equal-sigma (unweighted) assumption.
    # To match the reference precisely, feed a constant sigma into the solver.
    sigma_col = "__sigma"
    expected_col = "CCC ddG (kcal/mol)"
    missing = [c for c in [ddg_col, expected_col] if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing columns in {csv_path}: {missing}")

    df = df.copy()
    assert not df.duplicated([lig1_col, lig2_col]).any(), f"Duplicate directed edges in {csv_path}"

    # Fit node potentials using only rows with numeric Bennett ddG.
    ddg_num = pd.to_numeric(df[ddg_col], errors="coerce")
    df_fit = df.loc[ddg_num.notna(), :].copy()
    # janssen_bace/bace_p3_arg368_in_morecorehops_out.csv has one row of nan Bennett ddG values
    # merck/shp2_manual_213_exp_out.csv has one row of nan Bennett ddG values
    # misc/cdk8_koehler_213_out.csv has two rows of nan Bennett ddG values
    # Exclude these rows from fitting.
    df_fit = df_fit[~df_fit[ddg_col].isna()]
    df_fit[sigma_col] = 1.0  # this value does not matter when all edges share the same error

    # Reuse the original ligand labels on output so the node-to-edge mapping below is direct.
    res = ccc_runner(
        df_fit,
        lig1_col=lig1_col,
        lig2_col=lig2_col,
        ddg_col=ddg_col,
        sigma_col=sigma_col,
        out_lig1_col=lig1_col,
        out_lig2_col=lig2_col,
        out_ddg_col=ddg_col,
        out_sigma_col=sigma_col,
        debug=False,
    )

    lig1 = df[lig1_col].astype(str)
    lig2 = df[lig2_col].astype(str)
    x = res.node_df.set_index("Ligand")["dG (kcal/mol)"]
    x1 = lig1.map(x)
    x2 = lig2.map(x)
    assert not x1.isna().any() and not x2.isna().any(), f"Missing node potential for some ligands in {csv_path}"

    # actual_edge_raw calculated by in-house ccc
    actual_edge_raw = (x2 - x1).to_numpy(dtype=float)
    # expected_edge_raw provided by reference csv
    expected_edge_raw = df[expected_col].to_numpy(dtype=float)
    np.testing.assert_allclose(actual_edge_raw - expected_edge_raw, 0.0, atol=1e-2)
