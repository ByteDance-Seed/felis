# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from felis.tests.testkit import assert_manifest
from felis.tests.testkit.artifacts import boresch_npt_manifest
from felis.tests.testkit.artifacts import make_abfe_input_config
from felis.tests.testkit.artifacts import make_artifact_stage_context
from felis.tests.testkit.artifacts import seed_empty_files


def test_stage_boresch_em_artifacts(tmp_path: Path, monkeypatch):

    from felis.configs import MinimizeRelaxOption
    from felis.protocols.abfe.config_types import ABStage
    from felis.protocols.abfe.config_types import ABStageExecutionBase

    ctx = make_artifact_stage_context(monkeypatch, tmp_path)
    job_root = ctx.job_root

    sysB_top = job_root / "prepare/sysB.top"
    sysB_gro = job_root / "prepare/sysB.gro"
    seed_empty_files([sysB_top, sysB_gro])

    acfg = make_abfe_input_config(outdir=str(tmp_path))

    exec_boresch_em = ABStageExecutionBase.create(
        ABStage.boresch_em,
        acfg=acfg,
        planned_stages=[ABStage.boresch_em],
        job_context=ctx.job_context,
    )
    exec_boresch_em.exec()

    em_version = acfg.md_pro_em_version
    expected_rel_files = [
        "trj/sysB_boresch_em.pdb",
        "trj/sysB_boresch_em.dynlog",
        str(sysB_top.relative_to(job_root)),
        str(sysB_gro.relative_to(job_root)),
        f"prepare/sysB_no_rstrn_em{em_version}.pdb",
        "progress/boresch_em.done",
    ]
    if em_version not in (MinimizeRelaxOption.l_bfgs.value, MinimizeRelaxOption.heating.value):
        expected_rel_files.append(f"trj/sysB_boresch_em.em{em_version}.csv")

    assert_manifest(
        job_root,
        expected_rel_files=expected_rel_files,
    )


def test_stage_boresch_npt_artifacts(tmp_path: Path, monkeypatch):

    np = 4

    from felis.protocols.abfe.config_types import ABStage
    from felis.protocols.abfe.config_types import ABStageExecutionBase

    ctx = make_artifact_stage_context(monkeypatch, tmp_path)
    job_root = ctx.job_root

    sysB_top = job_root / "prepare/sysB.top"
    sysB_pdb = job_root / "trj/sysB_boresch_em.pdb"
    seed_empty_files([sysB_top, sysB_pdb])

    acfg = make_abfe_input_config(outdir=str(tmp_path))

    exec_boresch_npt = ABStageExecutionBase.create(
        ABStage.boresch_npt,
        acfg=acfg,
        planned_stages=[ABStage.boresch_npt],
        job_context=ctx.job_context,
    )
    exec_boresch_npt.exec()

    expected_rel_files = [
        "trj/sysB_boresch_em.pdb",
        "prepare/sysB.top",
        "progress/boresch_npt.done",
    ]
    expected_rel_files.extend(boresch_npt_manifest("sysB_boresch_npt", np=np))

    assert_manifest(job_root, expected_rel_files)
