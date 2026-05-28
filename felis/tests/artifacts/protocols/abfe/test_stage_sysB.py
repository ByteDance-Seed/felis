# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from felis.tests.testkit import assert_manifest
from felis.tests.testkit.artifacts import make_abfe_input_config
from felis.tests.testkit.artifacts import make_artifact_stage_context
from felis.tests.testkit.artifacts import replica_exchange_manifest
from felis.tests.testkit.artifacts import seed_empty_files
from felis.tests.testkit.artifacts import seed_text_files


def test_stage_sysB_em_artifacts(tmp_path: Path, monkeypatch):

    from felis.configs import MinimizeRelaxOption
    from felis.protocols.abfe.config_types import ABStage
    from felis.protocols.abfe.config_types import ABStageExecutionBase
    from felis.protocols.boresch.main_boresch_restraints import BORESCH_CFG_JSON

    ctx = make_artifact_stage_context(monkeypatch, tmp_path)
    job_root = ctx.job_root

    boresch_cfg = job_root / f"prepare/{BORESCH_CFG_JSON}"
    sysB_posres = job_root / "prepare/sysB_posres.json"
    sysB_top = job_root / "prepare/sysB.top"
    sysB_gro = job_root / "prepare/sysB.gro"

    seed_text_files({boresch_cfg: "{}", sysB_posres: "{}"})
    seed_empty_files([sysB_top, sysB_gro])

    acfg = make_abfe_input_config(outdir=str(tmp_path))

    exec_sysB_em = ABStageExecutionBase.create(
        ABStage.sysB_em,
        acfg=acfg,
        planned_stages=[ABStage.sysB_em],
        job_context=ctx.job_context,
    )
    exec_sysB_em.exec()

    em_version = acfg.md_pro_em_version
    expected_rel_files = [
        str(boresch_cfg.relative_to(job_root)),
        str(sysB_posres.relative_to(job_root)),
        str(sysB_top.relative_to(job_root)),
        str(sysB_gro.relative_to(job_root)),
        "trj/sysB_boresch_filtered.pdb",
        "trj/sysB_boresch_filtered.dynlog",
        "trj/sysB_em.pdb",
        "trj/sysB_em.dynlog",
        "prepare/sysB_em.pdb",
        f"prepare/sysB_rstrn_em{em_version}.pdb",
        "prepare/sysB_boresch_refined.pdb",
        "progress/sysB_em.done",
    ]
    if em_version not in (MinimizeRelaxOption.l_bfgs.value, MinimizeRelaxOption.heating.value):
        expected_rel_files.append(f"trj/sysB_em.em{em_version}.csv")
        expected_rel_files.append(f"trj/sysB_boresch_filtered.em{em_version}.csv")

    assert_manifest(
        job_root,
        expected_rel_files=expected_rel_files,
    )


def test_stage_sysB_artifacts(tmp_path: Path, monkeypatch):

    np = 4
    ndev = 8

    from felis.protocols.abfe.config_types import ABStage
    from felis.protocols.abfe.config_types import ABStageExecutionBase
    from felis.protocols.boresch.main_boresch_restraints import BORESCH_CFG_JSON

    ctx = make_artifact_stage_context(monkeypatch, tmp_path)
    job_root = ctx.job_root
    lig_stem = ctx.lig_stem

    boresch_cfg = job_root / f"prepare/{BORESCH_CFG_JSON}"
    sysB_ligatoms = job_root / "prepare/sysB_ab_ligatoms.json"
    sysB_atom_ids = job_root / "prepare/sysB_atom_ids.json"
    sysB_top = job_root / "prepare/sysB.top"
    sysB_pdb = job_root / "prepare/sysB_em.pdb"

    seed_text_files({boresch_cfg: "{}", sysB_ligatoms: "{}", sysB_atom_ids: "{}"})
    seed_empty_files([sysB_top, sysB_pdb])

    tmp_itp = tmp_path / f"{lig_stem}.itp"
    tmp_sdf = tmp_path / f"{lig_stem}.sdf"
    sdf_abs = str(tmp_sdf.absolute())

    acfg = make_abfe_input_config(
        outdir=str(tmp_path),
        sdffile=str(tmp_sdf),
        itpfile=str(tmp_itp),
        elamrecipe="e29",
        vlamrecipe="v45",
        reslamrecipe="r02",
    )

    exec_sysB = ABStageExecutionBase.create(
        ABStage.sysB,
        acfg=acfg,
        sdf_abs=sdf_abs,
        n_cuda_devices=ndev,
        np=np,
        planned_stages=[ABStage.sysB],
        job_context=ctx.job_context,
    )
    exec_sysB.exec()

    expected_rel_files = [
        str(sysB_top.relative_to(job_root)),
        str(sysB_pdb.relative_to(job_root)),
        str(sysB_ligatoms.relative_to(job_root)),
        str(sysB_atom_ids.relative_to(job_root)),
        str(boresch_cfg.relative_to(job_root)),
        "prepare/sysB_lam.json",
        "progress/sysB.done",
    ]
    expected_rel_files.extend(replica_exchange_manifest(prefixes=["b"], ndev=ndev, np=np))

    assert_manifest(job_root, expected_rel_files)
