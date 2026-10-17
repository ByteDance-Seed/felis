# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Run batch and system-level RBFE post-corrections reference regressions."""

from contextlib import nullcontext
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from felis.protocols.correction.post_corrections import PostCorrections
from felis.protocols.correction.post_corrections import RBPostCorrections
from felis.tests.protocols.correction.post_corrections_helpers import assert_ab_post_correction_artifacts
from felis.tests.protocols.correction.post_corrections_helpers import assert_post_correction_node_df
from felis.tests.protocols.correction.post_corrections_helpers import assert_rbfe_post_corrections_against_reference_csv
from felis.tests.protocols.correction.post_corrections_helpers import iter_rbfe_post_correction_system_dirs
from felis.tests.protocols.correction.post_corrections_helpers import rbfe_post_correction_root
from felis.tests.protocols.correction.post_corrections_helpers import resolve_workflow_output_path
from felis.tests.protocols.correction.post_corrections_helpers import safe_sys_stem_for_tmp

A2A_HIP278_PKA_RT_ROTAMER_CASES = [
    (0.592484, "edge_shift_ccc", nullcontext()),
    (0.592484, None, nullcontext()),
    (0.596, "edge_shift_ccc", pytest.raises(AssertionError)),
    (0.596, None, pytest.raises(AssertionError)),
]

# Registry of RBFE batch reference systems expected to fail for known reasons.
#
# Keys are paths relative to ``rbfe_post_correction_root()`` (POSIX style).
# Each value is a human-readable reason consumed by ``pytest.mark.xfail``.
# Adding a system here makes the expected failure visible at parametrization
# time instead of being buried inside the regression helper.
RBFE_BATCH_EXPECTED_FAILURES: dict[str, str] = {
    "janssen_bace/bace_keranen_p2": ("bace_keranen_p2 has a rotamer group with no valid nodes in the graph; "
                                     "RBPostCorrections.run() raises ValueError before completing."),
    "charge/tyk2": ("charge/tyk2 reference Pred dG values differ within the same treatment group "
                    "(see test_charge_tyk2_ref_not_merged_causes_pka_pair_mismatch)."),
    "fragments/jak2_set1": ("fragments/jak2_set1 reference Pred dG values differ within the same treatment group "
                            "(see test_fragments_jak2_set1_ref_not_merged_causes_population_pair_mismatch)."),
    "mcs_docking/hne_500mM": ("mcs_docking/hne_500mM reference Pred dG values differ within the same treatment group "
                              "(see test_mcs_docking_hne_500mm_ref_not_merged_causes_pka_pair_mismatch)."),
    "merck/cdk8": ("merck/cdk8 reference is missing ligand 28 in the fep+ input edge_predictions.csv; "
                   "in-house vs reference ligand dG diff exceeds 0.01 kcal/mol (max ~0.057 kcal/mol)."),
    "merck/eg5": ("merck/eg5 reference provides CHEMBL1084935 vs CHEMBL1084935_1 with predict dG before pKa correction "
                  "(see test_merck_eg5_ref_not_merged_causes_tautomer_mismatch)."),
    "merck/syk": ("merck/syk has unequal CHEMBL3265003_tautomer_1 and CHEMBL3265003_tautomer_2 pred dG values "
                  "(see test_merck_syk_ref_not_merged_causes_tautomer_mismatch); "
                  "also fails the in-house sum-to-zero check."),
    "merck/tnks2": ("merck/tnks2 reference provides three pKa pairs with predict dG before pKa correction "
                    "(see test_merck_tnks2_ref_not_merged_causes_pka_pair_mismatch)."),
    "waterset/throm_nozob_hip75":
        ("waterset/throm_nozob_hip75 reference has two known errors "
         "(see test_rbfe_post_corrections_throm_nozob_hip75_bug_locator_against_reference_csv)."),
}


def _rbfe_batch_params():
    root = rbfe_post_correction_root()
    for sys_dir in iter_rbfe_post_correction_system_dirs():
        sys_rel = sys_dir.relative_to(root).as_posix()
        marks = []
        reason = RBFE_BATCH_EXPECTED_FAILURES.get(sys_rel)
        if reason is not None:
            marks.append(pytest.mark.xfail(strict=True, reason=reason))
        yield pytest.param(sys_dir, marks=marks, id=sys_rel)


# Systems with an expected ABFE workflow failure.
#
# Currently this list is minimal: systems are only added when the ABFE
# workflow cannot run to completion even after CCC cycle-closure fallback.
ABFE_WORKFLOW_EXPECTED_FAILURE_SUFFIXES: list[str] = [
    # System where a configured rotamer group has no valid nodes in the
    # ABFE graph (e.g. ['28_in', '28_out'] in bace_keranen_p2), causing
    # PostCorrections.conf_correct to raise during the real workflow.
    "janssen_bace/bace_keranen_p2",
]


def _abfe_workflow_expectation_for_system(sys_dir: Path):
    """Return the expected ABFE workflow outcome for a packaged system.

    These correspond to per-system diagnostics and explicit xfail branches
    in the RBFE reference regression helper where edge ddGs and/or reference
    mapping are intentionally inconsistent or under investigation.
    """

    sys_rel = str(sys_dir).replace("\\", "/")
    for suffix in ABFE_WORKFLOW_EXPECTED_FAILURE_SUFFIXES:
        if sys_rel.endswith(suffix):
            return pytest.raises(ValueError, match=r"Rotamer list .* contains no valid nodes in the graph\."), True
    return nullcontext(), False


def _build_abfe_nodewise_from_ccc(tmp_path: Path, sys_dir: Path, *, rt: float, pka_rt: float) -> Path:
    """Derive ABFE-style nodewise dG from CCC ddG edge_predictions.

    - Reads edge_predictions.csv under sys_dir.
    - Uses RBPostCorrections graph machinery to integrate CCC ddGs into
      nodewise dG values.
    - Writes a TSV with columns ('ligand', 'avg/dG') (and optional 'exp')
      under tmp_path and returns its path.
    """

    edges_path = sys_dir / "edge_predictions.csv"
    cfg_path = sys_dir / "rbfe_correction_config.json"

    assert edges_path.exists(), f"Missing edge_predictions.csv in {sys_dir}"
    assert cfg_path.exists(), f"Missing rbfe_correction_config.json in {sys_dir}"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    rb_pc = RBPostCorrections(str(edges_path), cfg, output_dir=str(tmp_path), rt=rt, pka_rt=pka_rt)
    # Prefer the fast path that integrates from already cycle-consistent
    # edges. If the CCC ddG network is inconsistent, fall back to a single
    # CCC cycle-closure pass on the base edges and rebuild a consistent
    # nodewise dG graph from the resulting node_df.
    try:
        rb_pc._build_graph_from_consistent_edges(rb_pc._base_edges, tol=1e-3)
    except ValueError:
        res_ccc = rb_pc._run_ccc(rb_pc._base_edges, debug=False)
        rb_pc._build_graph_from_node_df(res_ccc.node_df)

    node_df = rb_pc.get_results()

    ab_df = node_df[["ligand", "avg/dG"]].copy()

    stem = safe_sys_stem_for_tmp(sys_dir) + "__ab_from_ccc"
    ab_tsv = tmp_path / f"{stem}.tsv"
    ab_df.to_csv(ab_tsv, sep="\t", index=False)

    return ab_tsv


@pytest.mark.parametrize("sys_dir", list(_rbfe_batch_params()))
def test_rbfe_post_corrections_batch_against_reference_csv(tmp_path: Path, sys_dir: Path):
    """Run the shared RBFE reference regression across every packaged system.

    Args:
        tmp_path (Path): Temporary directory for generated diagnostics and
            corrected outputs.
        sys_dir (Path): One packaged RBFE reference system directory.
    """
    assert_rbfe_post_corrections_against_reference_csv(tmp_path, sys_dir, tolerence=0.01, rt=0.596, pka_rt=0.592484)


@pytest.mark.parametrize("sys_dir", iter_rbfe_post_correction_system_dirs())
def test_abfe_post_corrections_batch_from_rbfe_ccc(tmp_path: Path, sys_dir: Path):
    """Run ABFE post-corrections workflow as a smoke test per RBFE system.

    For each packaged RBFE reference system:
    - Integrate CCC ddGs from edge_predictions.csv into nodewise dG.
    - Use the shared RBFE correction config to run the ABFE workflow
      end-to-end on the derived nodewise inputs.
    - Assert that the workflow completes without error and emits the
      expected ABFE artifacts.
    """

    rt = 0.596
    pka_rt = 0.592484
    expectation, expects_failure = _abfe_workflow_expectation_for_system(sys_dir)

    ab_tsv = _build_abfe_nodewise_from_ccc(tmp_path, sys_dir, rt=rt, pka_rt=pka_rt)

    cfg_path = sys_dir / "rbfe_correction_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # The ABFE smoke test only cares that the workflow runs to completion.
    # After CCC cycle-closure and/or node merging, ligand labels in the
    # nodewise dG table may no longer match exp_ref keys one-to-one (for
    # example "39charg+39neu"). Build a shallow copy of the config and
    # drop exp_ref so PostCorrections does not enforce strict experimental
    # value consistency here; this test focuses on the post-corrections
    # pipeline itself.
    cfg_for_ab = dict(cfg)
    cfg_for_ab.pop("exp_ref", None)

    pc = PostCorrections(str(ab_tsv), cfg_for_ab, output_dir=str(tmp_path), rt=rt, pka_rt=pka_rt)
    with expectation:
        out_path = pc.run(debug=True)
    if expects_failure:
        return

    full_out = resolve_workflow_output_path(tmp_path, out_path)
    df = pd.read_csv(full_out, sep="\t")
    assert_post_correction_node_df(df)

    assert_ab_post_correction_artifacts(tmp_path, Path(ab_tsv).stem)


def test_rbfe_post_corrections_scyt_dehyd_against_reference_csv(tmp_path: Path):
    """Verify the `scyt_dehyd` case with Bennett ddGs treated as CCC inputs.

    Args:
        tmp_path (Path): Temporary directory for the copied system and
            generated diagnostics.
    """
    sys_dir = rbfe_post_correction_root() / "waterset" / "scyt_dehyd"
    shutil.copytree(sys_dir, tmp_path / "scyt_dehyd")
    sys_dir = tmp_path / "scyt_dehyd"

    in_csv = sys_dir / "edge_predictions.csv"
    in_df = pd.read_csv(in_csv)
    # The historical reference was produced from Bennett ddGs, so rewrite the
    # copied fixture to feed those values through the CCC input column.
    in_df.rename(columns={"Bennett ddG (kcal/mol)": "CCC ddG (kcal/mol)"}, inplace=True)
    in_df.to_csv(in_csv, index=False)

    assert_rbfe_post_corrections_against_reference_csv(
        tmp_path,
        sys_dir,
        in_csv=in_csv,
        tolerence=0.01,
        rt=0.596,
        pka_rt=0.596,
    )


@pytest.mark.parametrize("conformer_correction_route", [None, "edge_shift_ccc"])
def test_rbfe_post_corrections_throm_nozob_hip75_bug_locator_against_reference_csv(
        tmp_path: Path, conformer_correction_route: str | None):
    """Check the throm_nozob_hip75 case after correcting known reference issues."""
    rt = 0.596
    sys_dir = rbfe_post_correction_root() / "waterset" / "throm_nozob_hip75"
    shutil.copytree(sys_dir, tmp_path / "throm_nozob_hip75")
    sys_dir = tmp_path / "throm_nozob_hip75"

    ref_csv = sys_dir / "ligand_predictions.csv"
    df = pd.read_csv(ref_csv)
    cfg = json.loads((sys_dir / "rbfe_correction_config.json").read_text(encoding="utf-8"))
    # Reconstruct the reference-side rotamer entropy correction so the expected
    # ligand dGs match the in-house merged-state convention.
    for i, row in df.iterrows():
        state = row["Ligand name"]
        pred_dg = row["Pred. dG (kcal/mol)"]
        for grp in cfg.get("rotamer"):
            states = [str(x) for x in grp]
            if state in states:
                df.loc[i, "Pred. dG (kcal/mol)"] = pred_dg - rt * np.log(len(grp))
                break

    input_edges_df = pd.read_csv(sys_dir / "edge_predictions.csv")
    ddg_5_7b_original = input_edges_df.loc[(input_edges_df["Lig 1"] == "5") & (input_edges_df["Lig 2"] == "7b"),
                                           "CCC ddG (kcal/mol)"].item()
    ddg_5_7f_original = input_edges_df.loc[(input_edges_df["Lig 1"] == "5") & (input_edges_df["Lig 2"] == "7f"),
                                           "CCC ddG (kcal/mol)"].item()
    base_dg_5 = df.loc[df["Ligand name"] == "5", "Pred. dG (kcal/mol)"].values[0]
    # Anchor the problematic ligands back to ligand `5` using the original edge
    # predictions to mirror the intended reference values exactly.
    df.loc[df["Ligand name"] == "7b", "Pred. dG (kcal/mol)"] = base_dg_5 + ddg_5_7b_original
    df.loc[df["Ligand name"] == "7f", "Pred. dG (kcal/mol)"] = base_dg_5 + ddg_5_7f_original

    corrected_csv = sys_dir / "ligand_predictions_corrected.csv"
    df.to_csv(corrected_csv, index=False)

    assert_rbfe_post_corrections_against_reference_csv(
        tmp_path,
        sys_dir,
        ref_csv=corrected_csv,
        rotamer_route=conformer_correction_route,
        stem_tag=ref_csv.stem,
        tolerence=0.01,
    )


@pytest.mark.parametrize(("pka_rt", "rotamer_route", "expectation"), A2A_HIP278_PKA_RT_ROTAMER_CASES)
def test_gpcrs_a2a_hip278_rotamer_routes_and_pka_pairs_diagnostics(tmp_path: Path, pka_rt: float,
                                                                   rotamer_route: str | None, expectation):
    """Diagnose the HIP278 A2A mismatch across (pka_rt, rotamer_route) pairings
    using the original CCC-based edge_predictions input.
    Identify the mismatch comes from the pka_rt value instead of the rotamer_route."""
    sys_dir = rbfe_post_correction_root() / "gpcrs" / "a2a"
    with expectation:
        assert_rbfe_post_corrections_against_reference_csv(
            tmp_path,
            sys_dir,
            rotamer_route=rotamer_route,
            rt=0.596,
            pka_rt=pka_rt,
            stem_tag="a2a",
            tolerence=0.01,
        )


@pytest.mark.parametrize(("pka_rt", "rotamer_route", "expectation"), A2A_HIP278_PKA_RT_ROTAMER_CASES)
def test_gpcrs_a2a_hip278_check_ddg_initial(tmp_path: Path, pka_rt: float, rotamer_route: str | None, expectation):
    """Diagnose the HIP278 A2A mismatch when Bennett edge ddGs are treated as
    the initial CCC input, isolating the effect of the input ddG source from
    the downstream reference-side corrections."""
    sys_dir = rbfe_post_correction_root() / "gpcrs" / "a2a"
    shutil.copytree(sys_dir, tmp_path / "a2a")
    sys_dir = tmp_path / "a2a"
    in_csv = sys_dir / "edge_predictions.csv"
    in_df = pd.read_csv(in_csv)
    # For HIP278 A2A, the historical reference was produced from Bennett ddGs.
    # Rewrite the copied fixture to feed those values through the CCC input
    # column so this variant can isolate whether the mismatch originates from
    # the initial edge ddG source rather than the later reference-side
    # corrections.
    in_df.rename(columns={"Bennett ddG (kcal/mol)": "CCC ddG (kcal/mol)"}, inplace=True)
    in_df.to_csv(in_csv, index=False)

    with expectation:
        # First, run with an explicit in_csv pointing at the Bennett-as-CCC
        # modified edge_predictions. This ensures we are definitely consuming
        # the substituted input when computing the post-corrections.
        assert_rbfe_post_corrections_against_reference_csv(
            tmp_path,
            sys_dir,
            in_csv=in_csv,
            rotamer_route=rotamer_route,
            rt=0.596,
            pka_rt=pka_rt,
            stem_tag="a2a",
            tolerence=0.01,
        )

        # Then, run again without in_csv so the helper discovers the default
        # edge_predictions.csv path. This checks that the outcome depends only
        # on pka_rt and rotamer_route choices and not on how the input CSV path
        # is supplied.
        assert_rbfe_post_corrections_against_reference_csv(
            tmp_path,
            sys_dir,
            rotamer_route=rotamer_route,
            rt=0.596,
            pka_rt=pka_rt,
            stem_tag="a2a",
            tolerence=0.01,
        )
