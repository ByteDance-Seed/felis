# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for group-level filtered dG aggregation.

This module exercises the ABFE/RBFE "grouping" statistics helpers in
`felis.protocols.stats.main_dg_group_all_runs`.

The functions under test take ligand-state predictions from multiple runs and
produce one aggregated dG per benchmark group (e.g., by selecting the minimum
state per group, or by running `PostCorrections` for a more precise merge).
"""

import json

import numpy as np
from pandas import DataFrame
import pytest

from felis.protocols.stats.main_dg_group_all_runs import _compare_filtered_dGsv1
from felis.protocols.stats.main_dg_group_all_runs import _map_postcorrected_results_to_groups
from felis.protocols.stats.main_dg_group_all_runs import compare_filtered_dGs_v2
from felis.protocols.stats.main_dg_group_all_runs import compare_filtered_dGs_v3
from felis.protocols.stats.main_dg_group_all_runs import compare_full_dGs


def _get_test_date():
    bm_full = DataFrame({
        "group_id": ["A", "B", "C", "C", "C", "C"],
        "Ligand_name": ["A", "B", "C", "C1", "C2", "C3"],
        "exp_dG_kcal_mol": [-1.0, -2.0, -3.0, -3.0, -3.0, -3.0]
    })

    r1 = DataFrame({"ligand": ["C1", "C3", "C", "B", "C2", "A"], "dG(kcal/mol)": [-3.0, -3.0, -3.2, -2.1, -3.4, -1.1]})
    r2 = DataFrame({"ligand": ["C3", "C", "B", "C2", "C1", "A"], "dG(kcal/mol)": [-2.8, -3.6, -2.2, -3.3, -3.1, -1.2]})
    r3 = DataFrame({"ligand": ["C1", "B", "C2", "A", "C3", "C"], "dG(kcal/mol)": [-3.2, -2.3, -3.5, -1.3, -3.4, -3.1]})

    r1min = DataFrame({"ligand": ["C", "B", "A"], "dG(kcal/mol)": [-3.4, -2.1, -1.1]})
    r2min = DataFrame({"ligand": ["A", "B", "C"], "dG(kcal/mol)": [-1.2, -2.2, -3.6]})
    r3min = DataFrame({"ligand": ["B", "C", "A"], "dG(kcal/mol)": [-2.3, -3.5, -1.3]})

    bm1_exp = DataFrame({"group_id": ["B", "C", "A"], "exp_dG_kcal_mol": [-2.0, -3.0, -1.0]})
    return bm_full, bm1_exp, [[r1, r1min], [r2, r2min], [r3, r3min]]


def _write_correction_config(tmp_path, name: str, config: dict) -> str:
    out_path = tmp_path / name
    out_path.write_text(json.dumps(config), encoding="utf-8")
    return str(out_path)


def _rotamer_corrected_dg(dg_values: list[float], rt: float = 0.596) -> float:
    np_dg = np.asarray(dg_values, dtype=float)
    dg_base = float(np.min(np_dg))
    return dg_base - rt * np.log(np.exp(-(np_dg - dg_base) / rt).sum())


def test_abfe_filtered_dG():
    """Test that legacy and current "min" filtering paths produce expected means.

    Functions under test:
        - `_compare_filtered_dGsv1`: aggregates already-filtered per-group tables.
        - `compare_full_dGs` + `compare_filtered_dGs_v2(select_method="min")`: the
          newer path that starts from full ligand-state tables, selects one state
          per group, and then aggregates across runs.

    What this test validates:
        - The returned filtered table preserves the benchmark group ordering.
        - The reported `avg/dG` column matches the returned mean vector.
        - The computed mean values match a hand-checked expected result for this
          small synthetic dataset.

    How it tests:
        - Builds a synthetic benchmark with one group (C) having multiple ligand
          states (C, C1, C2, C3).
        - Provides three synthetic runs and, for v1 only, the pre-filtered
          per-group selections for each run.
        - Compares the mean aggregated dG values against fixed expected arrays.
    """

    bm_full, bm1_exp, df_runs = _get_test_date()

    # v1: assumes input run tables are already filtered to one row per group.
    df1_compare_filtered, _np_dG_exp, np1_dG_mean, _np_dG_runs, _mean_metrics, _list_of_metrics = _compare_filtered_dGsv1(
        bm1_exp, [df_filtered for _, df_filtered in df_runs])

    assert (df1_compare_filtered["ligand"].values == bm1_exp["group_id"].values).all()
    assert np.isclose(df1_compare_filtered["avg/dG"].values, np1_dG_mean).all()
    assert np.isclose(np1_dG_mean, [-2.2, -3.5, -1.2]).all()

    select_method = "min"

    # v2: starts from full ligand-state tables, builds a full comparison table,
    # and then selects one state per group using `select_method`.
    df_comparedG, _, _, _ = compare_full_dGs(bm_full, [df_full for df_full, _ in df_runs])
    df2_compare_filtered, _np_dG_exp, np2_dG_mean, _np_dG_std, _np_dG_runs, _mean_metrics, _list_of_metrics = compare_filtered_dGs_v2(
        bm_full, bm1_exp, df_comparedG, select_method, [df_full for df_full, _ in df_runs])

    assert (df2_compare_filtered["ligand"].values == bm1_exp["group_id"].values).all()
    assert np.isclose(df2_compare_filtered["avg/dG"].values, np2_dG_mean).all()
    assert np.isclose(np2_dG_mean, [-2.2, -3.4, -1.2]).all()


def test_compare_filtered_dGs_v3_min_matches_v2(tmp_path):
    """Ensure `compare_filtered_dGs_v3(..., select_method="min")` matches v2.

    Functions under test:
        - `compare_filtered_dGs_v2(select_method="min")`
        - `compare_filtered_dGs_v3(select_method="min")`

    What this test validates:
        - v3 keeps backwards-compatible behavior for the `min` selection policy.
        - All returned outputs (dataframe, numpy arrays, and metrics) are
          identical between v2 and v3.

    How it tests:
        - Builds a synthetic benchmark and three synthetic runs.
        - Runs both v2 and v3 with the same inputs.
        - Uses exact dataframe equality and `assert_allclose` for numeric arrays.
    """
    bm_full, bm1_exp, df_runs = _get_test_date()
    df_comparedG, _, _, _ = compare_full_dGs(bm_full, [df_full for df_full, _ in df_runs])

    v2_result = compare_filtered_dGs_v2(bm_full, bm1_exp, df_comparedG, "min", [df_full for df_full, _ in df_runs])
    v3_result = compare_filtered_dGs_v3(bm_full,
                                        bm1_exp,
                                        df_comparedG,
                                        "min", [df_full for df_full, _ in df_runs],
                                        str(tmp_path),
                                        correction_config_json=None)

    df_v2, np_exp_v2, np_mean_v2, np_std_v2, np_runs_v2, mean_metrics_v2, list_metrics_v2 = v2_result
    df_v3, np_exp_v3, np_mean_v3, np_std_v3, np_runs_v3, mean_metrics_v3, list_metrics_v3 = v3_result

    assert df_v2.equals(df_v3)
    np.testing.assert_allclose(np_exp_v2, np_exp_v3)
    np.testing.assert_allclose(np_mean_v2, np_mean_v3)
    np.testing.assert_allclose(np_std_v2, np_std_v3)
    np.testing.assert_allclose(np_runs_v2, np_runs_v3)
    assert mean_metrics_v2 == mean_metrics_v3
    assert list_metrics_v2 == list_metrics_v3


def test_compare_filtered_dGs_v3_precise_applies_post_corrections(tmp_path):
    """Test that `select_method="precise"` runs `PostCorrections` per run.

    Function under test:
        - `compare_filtered_dGs_v3(select_method="precise")`

    Feature under test:
        In "precise" mode, each run's full ligand-state table is converted into a
        nodewise `PostCorrections` input table (`ligand`, `avg/dG`, `exp`).
        `PostCorrections` is then executed with the provided correction config,
        and the corrected values are mapped back to benchmark group ids.

    How it tests:
        - Writes a rotamer correction config that merges the four C states
          (C, C1, C2, C3) into one effective node.
        - Computes the expected corrected dG for that merged node using the same
          Boltzmann integration formula (`_rotamer_corrected_dg`).
        - Asserts per-run corrected vectors in the `r{n}` columns match expected
          values.
        - Asserts the reported mean/std/range match numpy reductions of the same
          expected per-run vectors.
    """
    bm_full, bm1_exp, df_runs = _get_test_date()
    df_comparedG, _, _, _ = compare_full_dGs(bm_full, [df_full for df_full, _ in df_runs])

    # Correction config: treat the four C states as rotamers/conformers of one
    # group so `PostCorrections` merges them into a single effective node.
    config_path = _write_correction_config(tmp_path, "rotamer.json", {"rotamer": [["C", "C1", "C2", "C3"]]})

    df_compare_filtered, np_dG_exp, np_dG_mean, np_dG_std, np_dG_runs, mean_metrics, list_of_metrics = compare_filtered_dGs_v3(
        bm_full, bm1_exp, df_comparedG, "precise", [df_full for df_full, _ in df_runs], str(tmp_path), config_path)

    assert (df_compare_filtered["ligand"].values == bm1_exp["group_id"].values).all()

    for col in ["r1", "r2", "r3"]:
        assert col in df_compare_filtered.columns

    expected_run_vectors = np.array([
        [-2.1, _rotamer_corrected_dg([-3.2, -3.0, -3.4, -3.0]), -1.1],
        [-2.2, _rotamer_corrected_dg([-3.6, -2.8, -3.3, -3.1]), -1.2],
        [-2.3, _rotamer_corrected_dg([-3.1, -3.2, -3.5, -3.4]), -1.3],
    ])
    expected_mean = np.mean(expected_run_vectors, axis=0)
    expected_std = np.std(expected_run_vectors, ddof=1, axis=0)
    expected_range = np.max(expected_run_vectors, axis=0) - np.min(expected_run_vectors, axis=0)

    np.testing.assert_allclose(df_compare_filtered["r1"].to_numpy(dtype=float), expected_run_vectors[0])
    np.testing.assert_allclose(df_compare_filtered["r2"].to_numpy(dtype=float), expected_run_vectors[1])
    np.testing.assert_allclose(df_compare_filtered["r3"].to_numpy(dtype=float), expected_run_vectors[2])
    np.testing.assert_allclose(df_compare_filtered["avg/dG"].to_numpy(dtype=float), expected_mean)
    np.testing.assert_allclose(df_compare_filtered["stdev/dG"].to_numpy(dtype=float), expected_std)
    np.testing.assert_allclose(df_compare_filtered["range/dG"].to_numpy(dtype=float), expected_range)
    np.testing.assert_allclose(np_dG_runs, expected_run_vectors)
    np.testing.assert_allclose(np_dG_mean, expected_mean)
    np.testing.assert_allclose(np_dG_std, expected_std)
    assert np_dG_exp.shape == (3,)
    assert len(list_of_metrics) == 3
    assert np.isfinite(mean_metrics.RMSE)


def test_compare_filtered_dGs_v3_precise_requires_config(tmp_path):
    """Test that `select_method="precise"` fails without a correction config.

    Function under test:
        - `compare_filtered_dGs_v3(select_method="precise")`

    What this test validates:
        - "precise" mode requires a valid `correction_config_json` path.
        - The code fails fast with a clear error message when it is omitted.
    """
    bm_full, bm1_exp, df_runs = _get_test_date()
    df_comparedG, _, _, _ = compare_full_dGs(bm_full, [df_full for df_full, _ in df_runs])

    with pytest.raises(ValueError, match="correction_config_json"):
        compare_filtered_dGs_v3(
            bm_full,
            bm1_exp,
            df_comparedG,
            "precise",
            [df_full for df_full, _ in df_runs],
            str(tmp_path),
            correction_config_json=None,
        )


def test_map_postcorrected_results_to_groups_rejects_duplicate_group_id():
    """Test that mapping corrected results back to groups rejects duplicates.

    Function under test:
        - `_map_postcorrected_results_to_groups`

    What this test validates:
        - Each benchmark group id must map to exactly one corrected row.
        - If multiple corrected rows correspond to the same benchmark group, the
          mapping is ambiguous and should raise `ValueError`.

    How it tests:
        - Constructs a corrected result table where two ligands (C1, C2) belong
          to the same benchmark group "C".
        - Expects a clear error indicating duplicate group assignment.
    """
    bm_full, _bm1_exp, _df_runs = _get_test_date()
    corrected_df = DataFrame({
        "ligand": ["C1", "C2"],
        "avg/dG": [-3.1, -3.4],
    })

    with pytest.raises(ValueError, match="Multiple corrected rows share group_id 'C'"):
        _map_postcorrected_results_to_groups(bm_full, corrected_df)


def test_compare_filtered_dGs_v3_precise_rejects_ambiguous_corrected_group(tmp_path):
    """Test that "precise" mode rejects correction outputs spanning multiple groups.

    Function under test:
        - `compare_filtered_dGs_v3(select_method="precise")`

    What this test validates:
        - The corrected nodewise output from `PostCorrections` must map cleanly
          to benchmark groups: one corrected row per group.
        - If the correction config merges only a subset of states within a
          benchmark group, the mapping back to groups becomes ambiguous (multiple
          corrected rows for the same group id) and should be rejected.

    How it tests:
        - Uses a rotamer config that merges only two of the C states (C1, C2),
          leaving other C states unmerged.
        - Expects `_map_postcorrected_results_to_groups` to detect duplicate group
          id "C" and raise `ValueError`.
    """
    bm_full, bm1_exp, df_runs = _get_test_date()
    df_comparedG, _, _, _ = compare_full_dGs(bm_full, [df_full for df_full, _ in df_runs])
    config_path = _write_correction_config(tmp_path, "ambiguous.json", {"rotamer": [["C1", "C2"]]})

    with pytest.raises(ValueError, match="Multiple corrected rows share group_id 'C'"):
        compare_filtered_dGs_v3(
            bm_full,
            bm1_exp,
            df_comparedG,
            "precise",
            [df_full for df_full, _ in df_runs],
            str(tmp_path),
            config_path,
        )
