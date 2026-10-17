# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Check synthetic RBFE pKa/tautomer post-correction behaviors."""

import math
from pathlib import Path

import pytest

from felis.protocols.correction.post_corrections import RBPostCorrections
from felis.tests.protocols.correction.post_corrections_helpers import ALL_TAUTOMERS_CONFIGS
from felis.tests.protocols.correction.post_corrections_helpers import assert_single_merged_ligand_result
from felis.tests.protocols.correction.post_corrections_helpers import build_synthetic_rbfe_edges
from felis.tests.protocols.correction.post_corrections_helpers import expected_all_tautomers_merged_dg
from felis.tests.protocols.correction.post_corrections_helpers import expected_shared_a_minus_merged_dg
from felis.tests.protocols.correction.post_corrections_helpers import expected_shared_ha_merged_dg
from felis.tests.protocols.correction.post_corrections_helpers import read_autosaved_results
from felis.tests.protocols.correction.post_corrections_helpers import run_synthetic_rbfe_pka_taut_correct
from felis.tests.protocols.correction.post_corrections_helpers import SHARED_HA_CONFIGS
from felis.tests.protocols.correction.post_corrections_helpers import synthetic_component_dg_map


@pytest.mark.parametrize("config", SHARED_HA_CONFIGS)
def test_rbfe_pka_taut_correct_unifies_shared_ha_component(tmp_path: Path, config: dict):
    """Verify that shared-HA RBFE states collapse into one merged ligand.

    Args:
        tmp_path (Path): Temporary directory for synthetic input and output
            files.
        config (dict): One equivalent pKa/population configuration for the
            shared-HA component.
    """
    pka_rt = 0.596
    pc = run_synthetic_rbfe_pka_taut_correct(tmp_path, config, pka_rt=pka_rt, stem="shared_ha")
    df = read_autosaved_results(pc)
    assert_single_merged_ligand_result(df, ligand="a1+a2+a3", expected_dg=expected_shared_ha_merged_dg(pka_rt))


def test_rbfe_pka_taut_correct_unifies_shared_a_minus_component(tmp_path: Path):
    """Verify that the synthetic shared-A- RBFE component merges correctly.

    Args:
        tmp_path (Path): Temporary directory for synthetic input and output
            files.
    """
    config = {"pka_tautomer": {"pH": 7.0, "pKa_list": [["a1", "a3", 7.0], ["a1", "a2", 7.0 - math.log10(3.0)]]}}

    pka_rt = 0.596
    pc = run_synthetic_rbfe_pka_taut_correct(tmp_path, config, pka_rt=pka_rt, stem="shared_ha")
    df = read_autosaved_results(pc)
    assert_single_merged_ligand_result(df, ligand="a1+a2+a3", expected_dg=expected_shared_a_minus_merged_dg(pka_rt))


@pytest.mark.parametrize("config", ALL_TAUTOMERS_CONFIGS)
def test_rbfe_pka_taut_correct_unifies_all_tautomers(tmp_path: Path, config: dict):
    """Verify that same-charge RBFE tautomer groups merge to one ligand.

    Args:
        tmp_path (Path): Temporary directory for synthetic input and output
            files.
        config (dict): Population-only tautomer configuration for the merged
            component.
    """
    pka_rt = 0.596
    pc = run_synthetic_rbfe_pka_taut_correct(tmp_path, config, pka_rt=pka_rt, stem="shared_ha")
    df = read_autosaved_results(pc)
    assert_single_merged_ligand_result(df, ligand="a1+a2+a3", expected_dg=expected_all_tautomers_merged_dg(pka_rt))


def test_rbfe_pka_taut_correct_raises_on_inconsistent_population_vs_pka_overlap(tmp_path: Path):
    """Verify that contradictory population and pKa links are rejected.

    Args:
        tmp_path (Path): Temporary directory for synthetic input and output
            files.
    """
    cfg = {
        "pka_tautomer": {
            "pH": 7.0,
            "pKa_list": [["a1", "a3", 7.0], ["a2", "a3", 7.1]],
            "population": {
                "a1:a2": [0.75, 0.25],
            },
        }
    }

    with pytest.raises(ValueError, match="Inconsistent pKa/tautomer component"):
        run_synthetic_rbfe_pka_taut_correct(tmp_path, cfg, stem="overlap_inconsistent")


def test_rbfe_pka_taut_correct_raises_on_more_than_two_charge_levels(tmp_path: Path):
    """Verify that RBFE pKa correction rejects more than two charge levels.

    Args:
        tmp_path (Path): Temporary directory for synthetic input and output
            files.
    """
    cfg = {
        "pka_tautomer": {
            "pH": 7.0,
            "pKa_list": [["a1", "a2", 7.0], ["a2", "a3", 7.0]],
        }
    }

    with pytest.raises(ValueError, match=r"^Only support up to 2 charge levels \(A- and HA\)."):
        run_synthetic_rbfe_pka_taut_correct(tmp_path, cfg, stem="three_charge_levels")


def test_rbfe_pka_taut_correct_raises_on_many_to_many_topology(tmp_path: Path):
    """Verify that unsupported many-to-many RBFE topologies fail fast.

    Args:
        tmp_path (Path): Temporary directory for synthetic input and output
            files.
    """
    cfg = {
        "pka_tautomer": {
            "pH": 7.0,
            "pKa_list": [["a1", "a3", 7.0]],
            "population": {
                "a1:a2": [0.75, 0.25],
                "a3:a4": [0.6, 0.4],
            },
        }
    }

    dg_map = {"a1": -1.0, "a2": 0.0, "a3": 1.0, "a4": 1.5}
    edges = build_synthetic_rbfe_edges(dg_map)
    edges_path = tmp_path / "many_to_many.csv"
    edges.to_csv(edges_path, index=False)

    pc = RBPostCorrections(str(edges_path), cfg, output_dir=str(tmp_path), rt=0.596, pka_rt=0.596)
    # RBFE pKa correction consumes the graph representation, so build it
    # explicitly before calling `pka_taut_correct` directly in isolation.
    pc._build_graph_from_consistent_edges(pc._base_edges)

    with pytest.raises(NotImplementedError, match="Supported cases are only"):
        pc.pka_taut_correct(debug=False)


@pytest.mark.parametrize("bad_name", [None, "", "   "])
@pytest.mark.parametrize("col", ["Lig 1", "Lig 2"])
def test_rbfe_rejects_missing_ligand_names(tmp_path: Path, col: str, bad_name):
    """Verify that empty ligand endpoints are rejected instead of becoming "nan".

    Args:
        tmp_path (Path): Temporary directory for synthetic input files.
        col (str): Endpoint column that receives the missing ligand name.
        bad_name: Missing or blank ligand name written to the input CSV.
    """
    edges = build_synthetic_rbfe_edges(synthetic_component_dg_map())
    edges[col] = edges[col].astype(object)
    edges.loc[0, col] = bad_name
    edges_path = tmp_path / "missing_ligand.csv"
    edges.to_csv(edges_path, index=False)

    with pytest.raises(ValueError, match=f"Missing ligand names in column '{col}'"):
        RBPostCorrections(str(edges_path), {}, output_dir=str(tmp_path))
