# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Self-tests for correction unit-test helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from felis.tests.protocols.correction import post_corrections_helpers as helpers


def test_correction_unit_tests_do_not_import_testkit():
    """Guard correction unit tests from depending on artifact-only helpers."""
    correction_dir = Path(__file__).resolve().parent
    forbidden_imports = ["felis.tests." + "testkit", "tests/" + "testkit"]
    offenders = []
    for py_file in sorted(correction_dir.glob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        if any(forbidden in text for forbidden in forbidden_imports):
            offenders.append(py_file.name)

    assert offenders == []


def test_resolve_workflow_output_path_accepts_absolute_relative_and_basename(tmp_path: Path):
    """Verify workflow output resolution without changing numeric thresholds.

    The helper should accept absolute paths, output-dir-relative paths, and
    basename fallbacks. No expected failure is associated with this self-test.

    Args:
        tmp_path (Path): Temporary directory containing a synthetic final TSV.
    """
    final_path = tmp_path / "result.tsv"
    final_path.write_text("ligand\tavg/dG\nA\t0.0\n", encoding="utf-8")

    assert helpers.resolve_workflow_output_path(tmp_path, str(final_path)) == final_path
    assert helpers.resolve_workflow_output_path(tmp_path, "result.tsv") == final_path
    assert helpers.resolve_workflow_output_path(tmp_path, str(tmp_path / "nested" / "result.tsv")) == final_path


def test_resolve_workflow_output_path_raises_on_unresolvable_path(tmp_path: Path):
    """Verify the resolver raises when no candidate path exists.

    Args:
        tmp_path (Path): Temporary directory with no matching files.
    """
    with pytest.raises(AssertionError, match="Could not resolve workflow output path"):
        helpers.resolve_workflow_output_path(tmp_path, "nonexistent.tsv")

    with pytest.raises(AssertionError, match="Could not resolve workflow output path"):
        helpers.resolve_workflow_output_path(tmp_path, str(tmp_path / "subdir" / "nonexistent.tsv"))


def test_assert_post_correction_node_df_rejects_bad_schema_and_values():
    """Validate node-table schema checks used by correction workflow tests.

    A minimal valid table should pass, while missing `avg/dG`, non-finite
    kcal/mol values, empty DataFrames, empty ligand labels, and non-numeric
    dG values are expected failures raised as `AssertionError`.
    """
    helpers.assert_post_correction_node_df(pd.DataFrame({"ligand": ["A"], "avg/dG": [0.0]}))

    with pytest.raises(AssertionError, match="Missing required columns"):
        helpers.assert_post_correction_node_df(pd.DataFrame({"ligand": ["A"]}))

    with pytest.raises(AssertionError, match="non-finite avg/dG"):
        helpers.assert_post_correction_node_df(pd.DataFrame({"ligand": ["A"], "avg/dG": [np.inf]}))

    with pytest.raises(AssertionError, match="empty"):
        helpers.assert_post_correction_node_df(pd.DataFrame(columns=["ligand", "avg/dG"]))

    with pytest.raises(AssertionError, match="empty ligand label"):
        helpers.assert_post_correction_node_df(pd.DataFrame({"ligand": [""], "avg/dG": [0.0]}))

    with pytest.raises(AssertionError, match="non-numeric avg/dG"):
        helpers.assert_post_correction_node_df(pd.DataFrame({"ligand": ["A"], "avg/dG": ["N/A"]}))


def test_treatment_group_representatives_detect_unmerged_reference_values():
    """Verify reference representative selection detects unmerged groups.

    Rotamer, tautomer-population, and pKa pairs should map to shared treatment
    groups. The expected failure is an `AssertionError` when reference dG values
    in kcal/mol differ inside a group beyond the unchanged `1e-2` tolerance.
    """
    cfg = {
        "rotamer": [["r1", "r2"]],
        "pka_tautomer": {
            "population": {
                "p1:p2": [0.25, 0.75],
            },
            "pKa_list": [["d", "h", 7.0]],
        },
    }
    ref_order = ["r1", "r2", "p1", "p2", "d", "h", "solo"]
    group_map = helpers.build_treatment_groups(cfg, ref_order)

    assert group_map["r1"] == group_map["r2"]
    assert group_map["p1"] == group_map["p2"]
    assert group_map["d"] == group_map["h"]
    assert group_map["solo"] not in {group_map["r1"], group_map["p1"], group_map["d"]}

    ref_pred = pd.Series({"r1": 1.0, "r2": 1.0, "p1": 2.0, "p2": 2.2, "d": 3.0, "h": 3.0, "solo": 4.0})
    with pytest.raises(AssertionError, match="Reference Pred dG values differ within the same treatment group"):
        helpers.pick_group_representatives(ref_pred, group_map, ref_order, equal_tol=1e-2)


def test_pick_group_representatives_selects_min_dg_within_tolerance():
    """Verify representative selection picks the ligand with minimum dG when values are within tolerance."""
    cfg = {
        "rotamer": [["r1", "r2"]],
        "pka_tautomer": {
            "pKa_list": [["d", "h", 7.0]],
        },
    }
    ref_order = ["r1", "r2", "d", "h", "solo"]
    group_map = helpers.build_treatment_groups(cfg, ref_order)

    # r1/r2 differ by 0.005 (< 1e-2 tol), d/h are equal, solo is alone
    ref_pred = pd.Series({"r1": 1.005, "r2": 1.000, "d": 3.0, "h": 3.0, "solo": 4.0})
    selected = helpers.pick_group_representatives(ref_pred, group_map, ref_order, equal_tol=1e-2)

    assert set(selected) == {"r2", "d", "solo"}  # r2 has min dG in rotamer group, d is first in pKa group


def test_pick_group_representatives_all_solos():
    """Verify representative selection when every ligand is in its own group."""
    ref_order = ["a", "b", "c"]
    group_map = {"a": "g_a", "b": "g_b", "c": "g_c"}
    ref_pred = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    selected = helpers.pick_group_representatives(ref_pred, group_map, ref_order, equal_tol=1e-2)
    assert set(selected) == {"a", "b", "c"}


def test_build_treatment_groups_empty_config():
    """Verify treatment group building with no rotamer/pka/population config."""
    ref_order = ["a", "b", "c"]
    group_map = helpers.build_treatment_groups({}, ref_order)
    assert len(set(group_map.values())) == 3  # each ligand in its own group
    assert group_map["a"] != group_map["b"] != group_map["c"]


def test_build_treatment_groups_empty_rotamer_list():
    """Verify treatment group building skips empty rotamer sublists."""
    cfg = {"rotamer": [["a", "b"], [], ["c"]]}
    ref_order = ["a", "b", "c", "d"]
    group_map = helpers.build_treatment_groups(cfg, ref_order)
    assert group_map["a"] == group_map["b"]
    assert group_map["c"] != group_map["a"]  # single-element rotamer group is its own
    assert group_map["d"] not in {group_map["a"], group_map["c"]}


def test_expand_inhouse_output_to_parts_preserves_structured_ligand_parts():
    """Check expansion of in-memory RBFE ligand-part mappings.

    Structured `ligand_parts` values must expand merged rows back to each input
    ligand without reading stringified CSV round-trips. No expected failure is
    associated with this valid mapping case.
    """
    out_df = pd.DataFrame({
        "ligand": ["A+B", "C"],
        "avg/dG": [1.25, -1.25],
        "ligand_parts": [["A", "B"], "C"],
    })

    expanded = helpers.expand_inhouse_output_to_parts(out_df)

    assert expanded.to_dict() == {"A": 1.25, "B": 1.25, "C": -1.25}


def test_expand_inhouse_output_to_parts_rejects_missing_columns():
    """Verify expansion raises when required columns are absent."""
    with pytest.raises(AssertionError, match="Unexpected output schema"):
        helpers.expand_inhouse_output_to_parts(pd.DataFrame({"ligand": ["A"], "avg/dG": [0.0]}))

    with pytest.raises(AssertionError, match="Unexpected output schema"):
        helpers.expand_inhouse_output_to_parts(pd.DataFrame({"ligand": ["A"], "ligand_parts": ["A"]}))


def test_expand_inhouse_output_to_parts_rejects_stringified_containers():
    """Verify expansion rejects stringified Python-repr containers from CSV round-trips."""
    out_df = pd.DataFrame({
        "ligand": ["A+B"],
        "avg/dG": [1.0],
        "ligand_parts": ["['A', 'B']"],
    })
    with pytest.raises(AssertionError, match="Unexpected stringified ligand_parts"):
        helpers.expand_inhouse_output_to_parts(out_df)


def test_expand_inhouse_output_to_parts_rejects_inconsistent_dg():
    """Verify expansion raises when the same part gets conflicting dG values."""
    out_df = pd.DataFrame({
        "ligand": ["A+B", "A+C"],
        "avg/dG": [1.0, 2.0],
        "ligand_parts": [["A", "B"], ["A", "C"]],
    })
    with pytest.raises(AssertionError, match="Inconsistent expanded dG"):
        helpers.expand_inhouse_output_to_parts(out_df)


def test_expand_inhouse_output_to_parts_handles_nested_ligand_parts():
    """Verify expansion handles nested list-of-lists in ligand_parts."""
    out_df = pd.DataFrame({
        "ligand": ["A+B", "C"],
        "avg/dG": [1.5, -1.5],
        "ligand_parts": [["A", ["B1", "B2"]], "C"],
    })
    expanded = helpers.expand_inhouse_output_to_parts(out_df)
    assert expanded.to_dict() == {"A": 1.5, "B1": 1.5, "B2": 1.5, "C": -1.5}


def test_assert_sum0_keeps_main_branch_strict_threshold():
    """Verify the sum-to-zero helper enforces the main-branch threshold.

    Values whose sum is smaller than `1e-2` should pass. The expected failure is
    an `AssertionError` for values whose sum reaches the strict `1e-2` threshold
    used by the shared RBFE reference comparator.
    """
    helpers.assert_sum0(np.array([0.006, -0.002]), tol=1e-2, context="self-test")

    with pytest.raises(AssertionError, match="not sum-to-zero"):
        helpers.assert_sum0(np.array([0.02, -0.01]), tol=1e-2, context="self-test")


def test_assert_sum0_boundary_at_tolerance():
    """Verify sum-to-zero rejects values whose absolute sum equals the tolerance."""
    with pytest.raises(AssertionError, match="not sum-to-zero"):
        helpers.assert_sum0(np.array([0.01, 0.0]), tol=1e-2, context="boundary")

    with pytest.raises(AssertionError, match="not sum-to-zero"):
        helpers.assert_sum0(np.array([-0.01, 0.0]), tol=1e-2, context="boundary")

    # Just below tolerance should pass
    helpers.assert_sum0(np.array([0.009, -0.004]), tol=1e-2, context="boundary")
