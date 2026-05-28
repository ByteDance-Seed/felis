# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from felis.tests.testkit import assert_manifest
from felis.tests.testkit.artifacts import (make_abfe_input_config, make_artifact_stage_context,
                                           replica_exchange_manifest, seed_empty_files, seed_text_files)


def test_stage_sysA_em_artifacts(tmp_path: Path, monkeypatch):

    from felis.protocols.abfe.config_types import ABStage, ABStageExecutionBase

    ctx = make_artifact_stage_context(monkeypatch, tmp_path)
    job_root = ctx.job_root

    sysA_posres = job_root / "prepare/sysA_posres.json"
    sysA_top = job_root / "prepare/sysA.top"
    sysA_gro = job_root / "prepare/sysA.gro"
    seed_text_files({sysA_posres: "{}"})
    seed_empty_files([sysA_top, sysA_gro])

    acfg = make_abfe_input_config(outdir=str(tmp_path))

    exec_sysA_em = ABStageExecutionBase.create(
        ABStage.sysA_em,
        acfg=acfg,
        planned_stages=[ABStage.sysA_em],
        job_context=ctx.job_context,
    )
    exec_sysA_em.exec()

    assert_manifest(
        job_root,
        expected_rel_files=[
            str(sysA_posres.relative_to(job_root)),
            str(sysA_top.relative_to(job_root)),
            str(sysA_gro.relative_to(job_root)),
            "prepare/sysA_em.pdb",
            "trj/sysA_em.pdb",
            "trj/sysA_em.dynlog",
            "progress/sysA_em.done",
        ],
    )


def test_stage_sysA_artifacts(tmp_path: Path, monkeypatch):

    np = 4
    ndev = 8

    from felis.protocols.abfe.config_types import ABStage, ABStageExecutionBase

    ctx = make_artifact_stage_context(monkeypatch, tmp_path)
    job_root = ctx.job_root
    lig_stem = ctx.lig_stem

    tmp_itp = tmp_path / f"{lig_stem}.itp"
    tmp_sdf = tmp_path / f"{lig_stem}.sdf"
    sdf_abs = str(tmp_sdf.absolute())

    ab_ligatoms = job_root / "prepare/sysA_ab_ligatoms.json"
    atom_ids = job_root / "prepare/sysA_atom_ids.json"
    sysA_top = job_root / "prepare/sysA.top"
    sysA_pdb = job_root / "prepare/sysA_em.pdb"
    seed_text_files({ab_ligatoms: "{}", atom_ids: "{}"})
    seed_empty_files([sysA_top, sysA_pdb])

    acfg = make_abfe_input_config(
        outdir=str(tmp_path),
        sdffile=str(tmp_sdf),
        itpfile=str(tmp_itp),
        elamrecipe="e29",
        vlamrecipe="v45",
        reslamrecipe="r02",
    )

    exec_sysA = ABStageExecutionBase.create(
        ABStage.sysA,
        acfg=acfg,
        sdf_abs=sdf_abs,
        n_cuda_devices=ndev,
        np=np,
        planned_stages=[ABStage.sysA],
        job_context=ctx.job_context,
    )
    exec_sysA.exec()

    expected_rel_files = [
        str(sysA_top.relative_to(job_root)),
        str(sysA_pdb.relative_to(job_root)),
        str(ab_ligatoms.relative_to(job_root)),
        str(atom_ids.relative_to(job_root)),
        "prepare/sysA_lam.json",
        "progress/sysA.done",
    ]
    expected_rel_files.extend(replica_exchange_manifest(prefixes=["a"], ndev=ndev, np=np))
    assert_manifest(job_root, expected_rel_files)
