# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Check synthetic AB pKa/tautomer post-correction behaviors."""

import math
from pathlib import Path

import pandas as pd
import pytest

from felis.protocols.correction.post_corrections import PostCorrections
from felis.tests.protocols.correction.post_corrections_helpers import ALL_TAUTOMERS_CONFIGS
from felis.tests.protocols.correction.post_corrections_helpers import assert_single_merged_ligand_result
from felis.tests.protocols.correction.post_corrections_helpers import build_synthetic_nodewise_df
from felis.tests.protocols.correction.post_corrections_helpers import expected_all_tautomers_merged_dg
from felis.tests.protocols.correction.post_corrections_helpers import expected_shared_a_minus_merged_dg
from felis.tests.protocols.correction.post_corrections_helpers import expected_shared_ha_merged_dg
from felis.tests.protocols.correction.post_corrections_helpers import read_autosaved_results
from felis.tests.protocols.correction.post_corrections_helpers import run_synthetic_ab_pka_taut_correct
from felis.tests.protocols.correction.post_corrections_helpers import SHARED_HA_CONFIGS


@pytest.mark.parametrize("config", SHARED_HA_CONFIGS)
def test_ab_pka_taut_correct_unifies_shared_ha_component(tmp_path: Path, config: dict):
    """Verify that shared-HA synthetic states collapse into one merged ligand.

    Args:
        tmp_path (Path): Temporary directory for synthetic input and output
            files.
        config (dict): One equivalent pKa/population configuration for the
            shared-HA component.
    """
    pka_rt = 0.596
    pc = run_synthetic_ab_pka_taut_correct(tmp_path, config, pka_rt=pka_rt, stem="ab_shared_ha")
    df = read_autosaved_results(pc)
    assert_single_merged_ligand_result(df, ligand="a1+a2+a3", expected_dg=expected_shared_ha_merged_dg(pka_rt))


def test_ab_pka_taut_correct_unifies_shared_a_minus_component(tmp_path: Path):
    """Verify that the synthetic shared-A- state also collapses to one ligand.

    Args:
        tmp_path (Path): Temporary directory for synthetic input and output
            files.
    """
    config = {"pka_tautomer": {"pH": 7.0, "pKa_list": [["a1", "a3", 7.0], ["a1", "a2", 7.0 - math.log10(3.0)]]}}

    pka_rt = 0.596
    pc = run_synthetic_ab_pka_taut_correct(tmp_path, config, pka_rt=pka_rt, stem="ab_shared_a_minus")
    df = read_autosaved_results(pc)
    assert_single_merged_ligand_result(df, ligand="a1+a2+a3", expected_dg=expected_shared_a_minus_merged_dg(pka_rt))


@pytest.mark.parametrize("config", ALL_TAUTOMERS_CONFIGS)
def test_ab_pka_taut_correct_unifies_all_tautomers(tmp_path: Path, config: dict):
    """Verify that same-charge tautomer groups merge to one effective ligand.

    Args:
        tmp_path (Path): Temporary directory for synthetic input and output
            files.
        config (dict): Population-only tautomer configuration for the merged
            component.
    """
    pka_rt = 0.596
    pc = run_synthetic_ab_pka_taut_correct(tmp_path, config, pka_rt=pka_rt, stem="ab_same_charge")
    df = read_autosaved_results(pc)
    assert_single_merged_ligand_result(df, ligand="a1+a2+a3", expected_dg=expected_all_tautomers_merged_dg(pka_rt))


def test_ab_pka_taut_correct_raises_on_inconsistent_population_vs_pka_overlap(tmp_path: Path):
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
        run_synthetic_ab_pka_taut_correct(tmp_path, cfg, stem="ab_overlap_inconsistent")


def test_ab_pka_taut_correct_raises_on_more_than_two_charge_levels(tmp_path: Path):
    """Verify that the AB helper rejects more than two charge levels.

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
        run_synthetic_ab_pka_taut_correct(tmp_path, cfg, stem="ab_three_charge_levels")


def test_ab_pka_taut_correct_raises_on_many_to_many_topology(tmp_path: Path):
    """Verify that unsupported many-to-many pKa/tautomer topologies fail.

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
    # Build a node-wise table because the AB workflow starts from ligand dGs
    # instead of an RBFE edge network.
    df = build_synthetic_nodewise_df(dg_map)
    csv_path = tmp_path / "ab_many_to_many.tsv"
    df.to_csv(csv_path, sep="\t", index=False)
    pc = PostCorrections(str(csv_path), cfg, output_dir=str(tmp_path), rt=0.596, pka_rt=0.596)

    with pytest.raises(NotImplementedError, match="Supported cases are only"):
        pc.pka_taut_correct(debug=False)


def test_ab_conf_correct_propagates_exp(tmp_path: Path):
    """Verify rotamer merging preserves identical experimental dG values.

    The ABFE conformer correction merges two synthetic rotamer states with the
    same experimental value in kcal/mol. The expected successful outcome is one
    merged ligand row whose `exp` column keeps that shared value exactly within
    the existing `1e-12` absolute tolerance; no expected failure is associated
    with this valid input.

    Args:
        tmp_path (Path): Temporary directory for the synthetic node-wise input
            TSV and post-correction outputs.
    """
    df = pd.DataFrame({
        "ligand": ["a1", "a2"],
        "avg/dG": [0.0, 1.0],
        "exp": [-10.5, -10.5],
    })
    csv_path = tmp_path / "ab_rotamer_exp.tsv"
    df.to_csv(csv_path, sep="\t", index=False)

    cfg = {"rotamer": [["a1", "a2"]]}
    pc = PostCorrections(str(csv_path), cfg, output_dir=str(tmp_path), rt=0.596, pka_rt=0.596)
    pc.conf_correct(debug=False)

    out = pc.get_results().set_index("ligand")
    assert "a1+a2" in out.index
    assert "exp" in out.columns
    assert math.isclose(float(out.loc["a1+a2", "exp"]), -10.5, rel_tol=0.0, abs_tol=1e-12)


def test_ab_conf_correct_raises_on_conflicting_exp(tmp_path: Path):
    """Verify rotamer merging rejects conflicting experimental dG values.

    The ABFE conformer correction receives two rotamer states whose `exp` values
    differ in kcal/mol. The expected failure is a `ValueError` reporting
    differing experimental values, because the merged state cannot carry a
    single unambiguous experimental reference.

    Args:
        tmp_path (Path): Temporary directory for the synthetic node-wise input
            TSV and post-correction outputs.
    """
    df = pd.DataFrame({
        "ligand": ["a1", "a2"],
        "avg/dG": [0.0, 1.0],
        "exp": [-10.5, -9.0],
    })
    csv_path = tmp_path / "ab_rotamer_exp_conflict.tsv"
    df.to_csv(csv_path, sep="\t", index=False)

    cfg = {"rotamer": [["a1", "a2"]]}
    pc = PostCorrections(str(csv_path), cfg, output_dir=str(tmp_path), rt=0.596, pka_rt=0.596)

    with pytest.raises(ValueError, match=r"differing experimental values"):
        pc.conf_correct(debug=False)
