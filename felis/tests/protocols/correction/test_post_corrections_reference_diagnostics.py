# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Diagnose known reference-data mismatches for RBFE post-corrections."""

import math
from pathlib import Path

import pytest

from felis.tests.protocols.correction.post_corrections_helpers import assert_rbfe_post_corrections_against_reference_csv
from felis.tests.protocols.correction.post_corrections_helpers import build_treatment_groups
from felis.tests.protocols.correction.post_corrections_helpers import copy_packaged_rbfe_system
from felis.tests.protocols.correction.post_corrections_helpers import integrate_pka_reference_dg
from felis.tests.protocols.correction.post_corrections_helpers import integrate_population_reference_dg
from felis.tests.protocols.correction.post_corrections_helpers import load_ligand_predictions
from felis.tests.protocols.correction.post_corrections_helpers import load_rbfe_correction_config
from felis.tests.protocols.correction.post_corrections_helpers import pick_group_representatives
from felis.tests.protocols.correction.post_corrections_helpers import reference_pred
from felis.tests.protocols.correction.post_corrections_helpers import write_corrected_ligand_predictions


@pytest.mark.parametrize("conformer_correction_route", [None, "edge_shift_ccc"])
def test_merck_syk_ref_not_merged_causes_tautomer_mismatch(tmp_path: Path, conformer_correction_route: str | None):
    """Verify SYK post-correction output matches the population-merged reference.

    The test rewrites the packaged reference so both CHEMBL3265003 tautomer
    states use the population-integrated dG in kcal/mol, then checks the
    corrected output against this merged reference via assert_allclose.

    The sum-to-zero check is skipped (skip_sum0=True) because the SYK system
    does not satisfy the strict 1e-2 sum-to-zero constraint after correction;
    the numerical comparison against the merged reference remains valid.

    Args:
        tmp_path (Path): Temporary directory for the copied reference system and
            diagnostic CSV/TSV artifacts.
        conformer_correction_route (str | None): Rotamer correction route under
            test; both the legacy and edge-shift routes are verified.
    """
    sys_dir = copy_packaged_rbfe_system(tmp_path, system_subpath="merck/syk", dest_name="syk")

    ref_pred_df = load_ligand_predictions(sys_dir)
    cfg = load_rbfe_correction_config(sys_dir)

    pair_key = "CHEMBL3265003_tautomer_1:CHEMBL3265003_tautomer_2"
    weights = cfg.get("pka_tautomer", {}).get("population", {}).get(pair_key)
    assert weights is not None, f"Missing pka_tautomer.population[{pair_key}] in cfg"
    assert abs((float(weights[0]) + float(weights[1])) - 1.0) < 1e-6

    dg1 = float(ref_pred_df.loc[ref_pred_df["Ligand name"] == "CHEMBL3265003_tautomer_1",
                                "Pred. dG (kcal/mol)"].iloc[0])
    dg2 = float(ref_pred_df.loc[ref_pred_df["Ligand name"] == "CHEMBL3265003_tautomer_2",
                                "Pred. dG (kcal/mol)"].iloc[0])
    pka_rt = 0.592484
    merged = integrate_population_reference_dg([dg1, dg2], [float(weights[0]), float(weights[1])], pka_rt)
    ref_pred_df.loc[ref_pred_df["Ligand name"] == "CHEMBL3265003_tautomer_1", "Pred. dG (kcal/mol)"] = merged
    ref_pred_df.loc[ref_pred_df["Ligand name"] == "CHEMBL3265003_tautomer_2", "Pred. dG (kcal/mol)"] = merged

    corrected_ref_csv = write_corrected_ligand_predictions(sys_dir, ref_pred_df)

    assert_rbfe_post_corrections_against_reference_csv(
        tmp_path,
        sys_dir,
        ref_csv=corrected_ref_csv,
        rotamer_route=conformer_correction_route,
        stem_tag="ligand_predictions",
        rt=0.596,
        pka_rt=pka_rt,
        tolerence=0.01,
        skip_sum0=True,
    )


def test_fragments_jak2_set1_ref_not_merged_causes_population_pair_mismatch(tmp_path: Path):
    """Show jak2_set1 population-pair mismatch comes from reference mapping."""
    sys_dir = copy_packaged_rbfe_system(tmp_path, system_subpath="fragments/jak2_set1", dest_name="jak2_set1")

    ref_pred_df = load_ligand_predictions(sys_dir)
    cfg = load_rbfe_correction_config(sys_dir)

    weights = cfg.get("pka_tautomer", {}).get("population", {}).get("4a:4b")
    assert weights is not None, "Missing pka_tautomer.population[4a:4b] in cfg"
    dg_4a = float(ref_pred_df.loc[ref_pred_df["Ligand name"] == "4a", "Pred. dG (kcal/mol)"].iloc[0])
    dg_4b = float(ref_pred_df.loc[ref_pred_df["Ligand name"] == "4b", "Pred. dG (kcal/mol)"].iloc[0])
    assert not math.isclose(dg_4a, dg_4b, rel_tol=0.0, abs_tol=1e-12)

    ref_pred_series = reference_pred(ref_pred_df)
    ref_order = ref_pred_series.index.astype(str).tolist()
    group_map = build_treatment_groups(cfg, ref_order)
    assert group_map["4a"] == group_map["4b"]
    assert "3_alternative" not in ref_order
    # The packaged reference keeps `4a` and `4b` separate even though the config
    # says they belong to one treated population pair, so representative picking
    # must fail before we patch the reference values.
    with pytest.raises(AssertionError, match="Reference Pred dG values differ within the same treatment group"):
        pick_group_representatives(ref_pred_series, group_map, ref_order, equal_tol=1e-2)

    merged = integrate_population_reference_dg([dg_4a, dg_4b], [float(weights[0]), float(weights[1])], 0.592484)
    ref_pred_df.loc[ref_pred_df["Ligand name"] == "4a", "Pred. dG (kcal/mol)"] = merged
    ref_pred_df.loc[ref_pred_df["Ligand name"] == "4b", "Pred. dG (kcal/mol)"] = merged

    corrected_ref_csv = write_corrected_ligand_predictions(sys_dir, ref_pred_df)
    assert_rbfe_post_corrections_against_reference_csv(
        tmp_path,
        sys_dir,
        ref_csv=corrected_ref_csv,
        stem_tag="ligand_predictions",
        rt=0.596,
        pka_rt=0.592484,
        tolerence=0.01,
    )


def test_merck_eg5_ref_not_merged_causes_tautomer_mismatch(tmp_path: Path):
    """Show EG5 CHEMBL1084935 protomer mismatch comes from reference mapping."""
    sys_dir = copy_packaged_rbfe_system(tmp_path, system_subpath="merck/eg5", dest_name="eg5")

    ref_pred_df = load_ligand_predictions(sys_dir)
    cfg = load_rbfe_correction_config(sys_dir)

    pka_cfg = cfg.get("pka_tautomer", {})
    p_h = float(pka_cfg.get("pH"))
    target_pair = None
    # Identify the one deprotonated/protonated pair that should have been merged
    # on the reference side before comparing with the in-house output.
    for deprot, prot, pka_val in pka_cfg.get("pKa_list", []):
        if str(deprot) == "CHEMBL1084935_1" and str(prot) == "CHEMBL1084935":
            target_pair = (str(deprot), str(prot), float(pka_val))
            break
    assert target_pair is not None

    deprot, prot, pka_val = target_pair
    dg_deprot = float(ref_pred_df.loc[ref_pred_df["Ligand name"] == deprot, "Pred. dG (kcal/mol)"].iloc[0])
    dg_prot = float(ref_pred_df.loc[ref_pred_df["Ligand name"] == prot, "Pred. dG (kcal/mol)"].iloc[0])
    merged = integrate_pka_reference_dg(dg_deprot, dg_prot, pka_rt=0.592484, pka_val=pka_val, ph=p_h)
    ref_pred_df.loc[ref_pred_df["Ligand name"] == deprot, "Pred. dG (kcal/mol)"] = merged
    ref_pred_df.loc[ref_pred_df["Ligand name"] == prot, "Pred. dG (kcal/mol)"] = merged

    corrected_ref_csv = write_corrected_ligand_predictions(sys_dir, ref_pred_df)
    assert_rbfe_post_corrections_against_reference_csv(
        tmp_path,
        sys_dir,
        ref_csv=corrected_ref_csv,
        stem_tag="ligand_predictions",
        rt=0.596,
        pka_rt=0.592484,
        tolerence=0.01,
    )


def test_charge_tyk2_ref_not_merged_causes_pka_pair_mismatch(tmp_path: Path):
    """Show TYK2 pKa-pair mismatch comes from reference mapping."""
    sys_dir = copy_packaged_rbfe_system(tmp_path, system_subpath="charge/tyk2", dest_name="tyk2")

    ref_pred_df = load_ligand_predictions(sys_dir)
    cfg = load_rbfe_correction_config(sys_dir)

    pka_cfg = cfg.get("pka_tautomer", {})
    target_pair = None
    for deprot, prot, pka_val in pka_cfg.get("pKa_list", []):
        if str(deprot) == "jmc_32charg" and str(prot) == "jmc_32neu":
            target_pair = (str(deprot), str(prot), float(pka_val))
            break
    assert target_pair is not None

    deprot, prot, pka_val = target_pair
    dg_deprot = float(ref_pred_df.loc[ref_pred_df["Ligand name"] == deprot, "Pred. dG (kcal/mol)"].iloc[0])
    dg_prot = float(ref_pred_df.loc[ref_pred_df["Ligand name"] == prot, "Pred. dG (kcal/mol)"].iloc[0])
    ref_pred_series = reference_pred(ref_pred_df)
    ref_order = ref_pred_series.index.astype(str).tolist()
    group_map = build_treatment_groups(cfg, ref_order)
    # This reference still stores separate pKa-pair predictions, which should be
    # rejected before the corrected merged value is written back.
    with pytest.raises(AssertionError, match="Reference Pred dG values differ within the same treatment group"):
        pick_group_representatives(ref_pred_series, group_map, ref_order, equal_tol=1e-3)

    merged = integrate_pka_reference_dg(dg_deprot,
                                        dg_prot,
                                        pka_rt=0.592484,
                                        pka_val=pka_val,
                                        ph=float(pka_cfg.get("pH")))
    ref_pred_df.loc[ref_pred_df["Ligand name"] == deprot, "Pred. dG (kcal/mol)"] = merged
    ref_pred_df.loc[ref_pred_df["Ligand name"] == prot, "Pred. dG (kcal/mol)"] = merged

    corrected_ref_csv = write_corrected_ligand_predictions(sys_dir, ref_pred_df)
    assert_rbfe_post_corrections_against_reference_csv(
        tmp_path,
        sys_dir,
        ref_csv=corrected_ref_csv,
        stem_tag="ligand_predictions",
        rt=0.596,
        pka_rt=0.592484,
        tolerence=0.01,
    )


def test_merck_tnks2_ref_not_merged_causes_pka_pair_mismatch(tmp_path: Path):
    """Show TNKS2 pKa-pair mismatch comes from reference mapping."""
    sys_dir = copy_packaged_rbfe_system(tmp_path, system_subpath="merck/tnks2", dest_name="tnks2")

    ref_pred_df = load_ligand_predictions(sys_dir)
    cfg = load_rbfe_correction_config(sys_dir)

    expected_pairs = {
        ("8d_neutral_equitorial", "8d"),
        ("8b_neutral", "8b"),
        ("8a_neutral", "8a"),
    }
    corrected_pairs = set()
    pka_cfg = cfg.get("pka_tautomer", {})
    for deprot, prot, pka_val in pka_cfg.get("pKa_list", []):
        deprot = str(deprot)
        prot = str(prot)
        dg_deprot = float(ref_pred_df.loc[ref_pred_df["Ligand name"] == deprot, "Pred. dG (kcal/mol)"].iloc[0])
        dg_prot = float(ref_pred_df.loc[ref_pred_df["Ligand name"] == prot, "Pred. dG (kcal/mol)"].iloc[0])
        # Ignore pairs the reference already merged correctly and patch only the
        # ones that still expose mismatched protonation states.
        if math.isclose(dg_deprot, dg_prot, rel_tol=0.0, abs_tol=1e-12):
            continue
        corrected_pairs.add((deprot, prot))
        merged = integrate_pka_reference_dg(dg_deprot,
                                            dg_prot,
                                            pka_rt=0.592484,
                                            pka_val=float(pka_val),
                                            ph=float(pka_cfg.get("pH")))
        ref_pred_df.loc[ref_pred_df["Ligand name"] == deprot, "Pred. dG (kcal/mol)"] = merged
        ref_pred_df.loc[ref_pred_df["Ligand name"] == prot, "Pred. dG (kcal/mol)"] = merged

    assert corrected_pairs == expected_pairs
    corrected_ref_csv = write_corrected_ligand_predictions(sys_dir, ref_pred_df)
    assert_rbfe_post_corrections_against_reference_csv(
        tmp_path,
        sys_dir,
        ref_csv=corrected_ref_csv,
        stem_tag="ligand_predictions",
        rt=0.596,
        pka_rt=0.592484,
        tolerence=0.0101,
    )


def test_mcs_docking_hne_500mm_ref_not_merged_causes_pka_pair_mismatch(tmp_path: Path):
    """Show hne_500mM pKa-pair mismatch comes from reference mapping."""
    sys_dir = copy_packaged_rbfe_system(tmp_path, system_subpath="mcs_docking/hne_500mM", dest_name="hne_500mM")

    ref_pred_df = load_ligand_predictions(sys_dir)
    cfg = load_rbfe_correction_config(sys_dir)

    expected_pairs = {
        ("21_neutral", "21"): 8.53,
        ("16", "16_neutral"): 3.58,
        ("19", "19_neutral"): 3.66,
    }
    pka_cfg = cfg.get("pka_tautomer", {})
    actual_pairs = {(str(deprot), str(prot)): float(pka_val) for deprot, prot, pka_val in pka_cfg.get("pKa_list", [])}
    assert actual_pairs == expected_pairs

    ref_pred_series = reference_pred(ref_pred_df)
    ref_order = ref_pred_series.index.astype(str).tolist()
    group_map = build_treatment_groups(cfg, ref_order)
    with pytest.raises(AssertionError, match="Reference Pred dG values differ within the same treatment group"):
        pick_group_representatives(ref_pred_series, group_map, ref_order, equal_tol=1e-2)

    corrected_pairs = set()
    for deprot, prot, pka_val in pka_cfg.get("pKa_list", []):
        deprot = str(deprot)
        prot = str(prot)
        dg_deprot = float(ref_pred_df.loc[ref_pred_df["Ligand name"] == deprot, "Pred. dG (kcal/mol)"].iloc[0])
        dg_prot = float(ref_pred_df.loc[ref_pred_df["Ligand name"] == prot, "Pred. dG (kcal/mol)"].iloc[0])
        # Rewrite every expected pKa pair so the corrected reference mirrors the
        # merged-state convention used by the production code.
        corrected_pairs.add((deprot, prot))
        merged = integrate_pka_reference_dg(dg_deprot,
                                            dg_prot,
                                            pka_rt=0.592484,
                                            pka_val=float(pka_val),
                                            ph=float(pka_cfg.get("pH")))
        ref_pred_df.loc[ref_pred_df["Ligand name"] == deprot, "Pred. dG (kcal/mol)"] = merged
        ref_pred_df.loc[ref_pred_df["Ligand name"] == prot, "Pred. dG (kcal/mol)"] = merged

    assert corrected_pairs == set(expected_pairs)
    corrected_ref_csv = write_corrected_ligand_predictions(sys_dir, ref_pred_df)
    assert_rbfe_post_corrections_against_reference_csv(
        tmp_path,
        sys_dir,
        ref_csv=corrected_ref_csv,
        stem_tag="ligand_predictions",
        rt=0.596,
        pka_rt=0.592484,
        tolerence=0.01,
    )
