# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd
import pytest

from felis.protocols.correction.ccc_rbfe_network import run_ccc_from_edge_df


@pytest.fixture(name="make_ccc_edge_df")
def fixture_make_ccc_edge_df():

    def _make_ccc_edge_df(lig1, lig2, ddg, sigma) -> pd.DataFrame:
        return pd.DataFrame({
            "Ligand 1": lig1,
            "Ligand 2": lig2,
            "ddG (kcal/mol)": ddg,
            "sigma": sigma,
        })

    return _make_ccc_edge_df


@pytest.fixture(name="ccc_runner")
def fixture_ccc_runner():

    def _ccc_runner(df: pd.DataFrame, **kwargs):
        options = {
            "lig1_col": "Ligand 1",
            "lig2_col": "Ligand 2",
            "ddg_col": "ddG (kcal/mol)",
            "sigma_col": "sigma",
            "out_ddg_col": "ddG (kcal/mol)",
            "out_sigma_col": "sigma",
        }
        options.update(kwargs)
        return run_ccc_from_edge_df(df, **options)

    return _ccc_runner


@pytest.fixture(name="get_ccc_edge_ddg")
def fixture_get_ccc_edge_ddg():

    def _get_ccc_edge_ddg(edge_df: pd.DataFrame, lig1: str, lig2: str) -> float:
        return float(edge_df[(edge_df["Lig 1"] == lig1) & (edge_df["Lig 2"] == lig2)]["CCC ddG (kcal/mol)"].iloc[0])

    return _get_ccc_edge_ddg
