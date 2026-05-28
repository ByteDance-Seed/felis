# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

from felis.tests.testkit import assert_manifest
from felis.tests.testkit.artifacts import make_abfe_input_config, make_artifact_stage_context


def test_stage_makebox_artifacts(tmp_path: Path, monkeypatch):

    from felis.protocols.abfe.config_types import ABStage, ABStageExecutionBase

    ctx = make_artifact_stage_context(monkeypatch, tmp_path)
    job_root = ctx.job_root
    lig_stem = ctx.lig_stem

    tmp_sdf = Path(f"{lig_stem}.sdf")
    tmp_sdf.touch()

    tmp_itp = tmp_sdf.with_suffix(".itp")
    tmp_itp.touch()

    acfg = make_abfe_input_config(
        outdir=str(tmp_path),
        sdffile=str(tmp_sdf),
        itpfile=str(tmp_itp),
    )

    exec_makebox = ABStageExecutionBase.create(
        ABStage.makebox,
        acfg,
        str(tmp_sdf.absolute()),
        planned_stages=[ABStage.makebox],
        job_context=ctx.job_context,
    )
    exec_makebox.exec()

    shutil.rmtree(job_root / "prepare/outA")
    shutil.rmtree(job_root / "prepare/outB")
    assert_manifest(
        job_root,
        expected_rel_files=[
            f"prepare/{lig_stem}.sdf",
            f"prepare/{lig_stem}.itp",
            "prepare/sysA.gro",
            "prepare/sysA.top",
            "prepare/sysA_ab_ligatoms.json",
            "prepare/sysA_atom_ids.json",
            "prepare/sysA_posres.json",
            "prepare/sysB.gro",
            "prepare/sysB.top",
            "prepare/sysB_ab_ligatoms.json",
            "prepare/sysB_atom_ids.json",
            "prepare/sysB_posres.json",
            "progress/makebox.done",
        ],
    )
