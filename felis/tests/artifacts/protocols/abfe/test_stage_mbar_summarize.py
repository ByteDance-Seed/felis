# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from felis.tests.testkit import assert_manifest
from felis.tests.testkit.artifacts import (make_abfe_input_config, make_artifact_stage_context, seed_empty_files,
                                           seed_text_files)


def test_stage_mbar_summarize_artifacts(tmp_path: Path, monkeypatch):

    from felis.protocols.abfe.config_types import ABStage, ABStageExecutionBase
    from felis.protocols.boresch.main_boresch_restraints import BORESCH_CFG_JSON

    ctx = make_artifact_stage_context(monkeypatch, tmp_path)
    job_root = ctx.job_root
    lig_stem = ctx.lig_stem

    tmp_sdf = tmp_path / f"{lig_stem}.sdf"
    sdf_abs = str(tmp_sdf.absolute())

    acfg = make_abfe_input_config(
        outdir=str(tmp_path),
        sdffile=sdf_abs,
    )

    # gpu0: 0 1 2 3
    # gpu1:       3 4 5 6
    ndev = 2
    nstates = 7

    # mbar

    boresch_json = job_root / f"prepare/{BORESCH_CFG_JSON}"
    sysA_done = job_root / "progress/sysA.done"
    sysB_done = job_root / "progress/sysB.done"
    a_nc = [job_root / f"trj/a{idx}.nc" for idx in range(ndev)]
    b_nc = [job_root / f"trj/b{idx}.nc" for idx in range(ndev)]
    seed_text_files({
        boresch_json: (r"""{"boresch": {"ligatoms": [0,1,2], "proatoms": [1000,1001,1002],
"r_theta_phi": [90.0,90.0,90.0], "alpha_beta_gamma": [90.0,90.0,90.0], "k_r_a_dih_kcal": [2.0,80.0,80.0]}}"""),
        sysA_done: f'{{"n_cuda_devices": {ndev}}}',
        sysB_done: f'{{"n_cuda_devices": {ndev}}}',
    })
    seed_empty_files(a_nc + b_nc)

    exec_mbar = ABStageExecutionBase.create(
        ABStage.mbar,
        acfg=acfg,
        planned_stages=[ABStage.mbar],
        job_context=ctx.job_context,
    )
    exec_mbar.exec()
    expected_rel_files = [
        f"prepare/{BORESCH_CFG_JSON}",
        "progress/mbar.done",
        "progress/sysA.done",
        "progress/sysB.done",
        "analysis/A_converge_table.tsv",
        "analysis/B_converge_table.tsv",
        "analysis/A_fe_table.tsv",
        "analysis/B_fe_table.tsv",
        "analysis/R_fe_table.tsv",
        "analysis/_converge_Abar_Abar.png",
        "analysis/_converge_Bbar_Bbar.png",
    ]
    for idx in range(ndev):
        expected_rel_files.append(f"trj/a{idx}.nc")
        expected_rel_files.append(f"trj/b{idx}.nc")
    for idx in range(nstates - 1):
        expected_rel_files.append(f"analysis/_A{idx:02d}_A{idx+1:02d}.png")
        expected_rel_files.append(f"analysis/_B{idx:02d}_B{idx+1:02d}.png")
        expected_rel_files.append(f"analysis/_converge_A{idx:02d}_A{idx+1:02d}.png")
        expected_rel_files.append(f"analysis/_converge_B{idx:02d}_B{idx+1:02d}.png")

    assert_manifest(job_root, expected_rel_files=expected_rel_files)

    # summarize

    sysA_lam = job_root / "prepare/sysA_lam.json"
    sysB_lam = job_root / "prepare/sysB_lam.json"
    seed_text_files({
        sysA_lam: (r"""
{"ab": {"lam_list": [[1.0,1.0,0.0], [1.0,0.05,0.0], [1.0,0.0,0.0], [0.999,0.0,0.0], [0.996,0.0,0.0], [0.99,0.0,0.0], [0.0,0.0,0.0]]}}"""
                  ),
        sysB_lam: (r"""
{"ab": {"lam_list": [[0.0,0.0,1.0], [0.999,0.0,1.0], [1.0,0.0,1.0], [1.0,0.98,1.0], [1.0,1.0,1.0], [1.0,1.0,0.75], [1.0,1.0,0.0]]}}"""
                  ),
    })

    exec_summarize = ABStageExecutionBase.create(ABStage.summarize,
                                                 acfg,
                                                 planned_stages=[ABStage.summarize],
                                                 job_context=ctx.job_context)
    exec_summarize.exec()

    expected_rel_files = [x for x in expected_rel_files if not x.startswith("analysis/")]
    expected_rel_files.extend([
        "prepare/sysA_lam.json",
        "prepare/sysB_lam.json",
        "progress/summarize.done",
        "analysis/sys_A_fe_merge.png",
        "analysis/sys_B_fe_merge.png",
        "analysis/sys_A_converge_merge.png",
        "analysis/sys_B_converge_merge.png",
        "analysis/A_converge_table.tsv",
        "analysis/B_converge_table.tsv",
        "analysis/A_fe_table.tsv",
        "analysis/B_fe_table.tsv",
        "analysis/R_fe_table.tsv",
        "analysis/sys_abfe.tsv",
    ])
    assert_manifest(job_root, expected_rel_files=expected_rel_files)
