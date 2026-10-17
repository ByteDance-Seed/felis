# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PARTIAL_PROTEIN_SCHEDULE = [
    [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5],
]


def _make_stage_context(tmp_path, monkeypatch, ligand_charge=0):
    from felis.protocols.abfe.job_context import JobContext

    monkeypatch.chdir(tmp_path)
    job_context = JobContext("ligand0")
    prepare_dir = Path(job_context.prepare_dir)
    prepare_dir.mkdir(parents=True, exist_ok=True)

    for filename in [
            "sysA.top",
            "sysA_em.pdb",
            "sysA_ab_ligatoms.json",
            "sysA_atom_ids.json",
            "sysB.top",
            "sysB_em.pdb",
            "sysB_ab_ligatoms.json",
            "sysB_atom_ids.json",
            "boresch_cfg.json",
    ]:
        (prepare_dir / filename).touch()

    ligand_path = tmp_path / "ligand0.sdf"
    ligand_path.touch()
    itp_path = tmp_path / "ligand0.itp"
    itp_path.touch()

    class _Mol:
        pass

    monkeypatch.setattr("rdkit.Chem.SDMolSupplier", lambda *_args, **_kwargs: [_Mol()])
    monkeypatch.setattr("rdkit.Chem.GetFormalCharge", lambda _mol: ligand_charge)

    return job_context, ligand_path, itp_path


def test_sysA_v2_threads_hmr_to_repex_command(tmp_path, monkeypatch):
    pytest.importorskip("rdkit.Chem")
    from felis.protocols.abfe.config_types import ABFEInputConfig
    from felis.protocols.abfe.config_types import ABStage
    from felis.protocols.abfe.config_types import ABStageExecutionBase
    import felis.protocols.abfe_v2.stage_sysA as stage_sysA

    calls = []
    job_context, ligand_path, itp_path = _make_stage_context(tmp_path, monkeypatch, ligand_charge=0)

    monkeypatch.setattr(stage_sysA, "get_visible_cuda_devices", lambda: [0])
    monkeypatch.setattr(stage_sysA, "_select_sysA_lambdas", lambda *_args, **_kwargs: [[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    monkeypatch.setattr(stage_sysA, "plot_lambda_schedule", lambda **_kwargs: None)

    def _fake_build_cuda_mps_command(cmd, gpu_id=None):
        calls.append(("build", cmd, gpu_id))
        return cmd

    def _fake_run_one_call(cmd, working_dir=None, extra_envs=None):
        calls.append(("run", cmd, working_dir, extra_envs))
        return 0

    monkeypatch.setattr(job_context, "build_cuda_mps_command", _fake_build_cuda_mps_command)
    monkeypatch.setattr(job_context, "run_one_call", _fake_run_one_call)

    acfg = ABFEInputConfig(
        outdir=str(tmp_path),
        sdffile=str(ligand_path),
        itpfile=str(itp_path),
        stages=["sysA_v2"],
        md_integrator_dt_ps=0.004,
        md_integrator_hmr=3.0,
        rmc_enable=True,
    )
    acfg.check()

    stage = ABStageExecutionBase.create(
        ABStage.sysA_v2,
        acfg=acfg,
        sdf_abs=str(ligand_path),
        n_cuda_devices=1,
        planned_stages=[ABStage.sysA_v2],
        job_context=job_context,
    )
    stage.exec()

    build_call = next(call for call in calls if call[0] == "build")
    assert "f:integrator.dt_ps:0.004" in build_call[1]
    assert "f:integrator.hmr:3.0" in build_call[1]
    assert not any("relocation_mc" in arg for arg in build_call[1])


def test_sysB_v2_threads_hmr_to_repex_command(tmp_path, monkeypatch):
    pytest.importorskip("rdkit.Chem")
    from felis.protocols.abfe.config_types import ABFEInputConfig
    from felis.protocols.abfe.config_types import ABStage
    from felis.protocols.abfe.config_types import ABStageExecutionBase
    import felis.protocols.abfe_v2.stage_sysB as stage_sysB

    calls = []
    job_context, ligand_path, itp_path = _make_stage_context(tmp_path, monkeypatch, ligand_charge=1)

    monkeypatch.setattr(stage_sysB, "get_visible_cuda_devices", lambda: [0])
    monkeypatch.setattr(
        stage_sysB,
        "_select_sysB_lambdas",
        lambda *_args, **_kwargs: [[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]],
    )
    monkeypatch.setattr(stage_sysB, "plot_lambda_schedule", lambda **_kwargs: None)

    def _fake_build_cuda_mps_command(cmd, gpu_id=None):
        calls.append(("build", cmd, gpu_id))
        return cmd

    def _fake_run_one_call(cmd, working_dir=None, extra_envs=None):
        calls.append(("run", cmd, working_dir, extra_envs))
        return 0

    monkeypatch.setattr(job_context, "build_cuda_mps_command", _fake_build_cuda_mps_command)
    monkeypatch.setattr(job_context, "run_one_call", _fake_run_one_call)

    acfg = ABFEInputConfig(
        outdir=str(tmp_path),
        sdffile=str(ligand_path),
        itpfile=str(itp_path),
        stages=["sysB_v2"],
        md_integrator_dt_ps=0.004,
        md_integrator_hmr=3.0,
        rmc_enable=True,
    )
    acfg.check()

    stage = ABStageExecutionBase.create(
        ABStage.sysB_v2,
        acfg=acfg,
        sdf_abs=str(ligand_path),
        n_cuda_devices=1,
        planned_stages=[ABStage.sysB_v2],
        job_context=job_context,
    )
    stage.exec()

    build_call = next(call for call in calls if call[0] == "build")
    assert "f:integrator.dt_ps:0.004" in build_call[1]
    assert "f:integrator.hmr:3.0" in build_call[1]
    assert "i:relocation_mc.enable:1" in build_call[1]
    assert "i:relocation_mc.num_candidate_sites:16384" in build_call[1]


@pytest.mark.parametrize("schedule_source", ["configured", "optimized"])
def test_sysB_v2_rejects_partial_restraint_endpoint_before_launch(
    tmp_path,
    monkeypatch,
    schedule_source,
):
    pytest.importorskip("rdkit.Chem")
    from felis.protocols.abfe.config_types import ABFEInputConfig
    from felis.protocols.abfe.config_types import ABStage
    from felis.protocols.abfe.config_types import ABStageExecutionBase
    import felis.protocols.abfe_v2.stage_sysB as stage_sysB

    job_context, ligand_path, itp_path = _make_stage_context(tmp_path, monkeypatch, ligand_charge=0)
    configured_path = tmp_path / "configured_sysB_lam.json"
    if schedule_source == "configured":
        configured_path.write_text(json.dumps({"ab": {"lam_list": PARTIAL_PROTEIN_SCHEDULE}}))
        pro_lam_json = str(configured_path)
    else:
        optimized_path = Path(job_context.prepare_dir) / stage_sysB.SYSB_OPT_LAM_JSON
        optimized_path.write_text(json.dumps({"ab": {"lam_list": PARTIAL_PROTEIN_SCHEDULE}}))
        pro_lam_json = None

    monkeypatch.setattr(stage_sysB, "get_visible_cuda_devices", lambda: [0])
    monkeypatch.setattr(
        job_context,
        "run_one_call",
        lambda *_args, **_kwargs: pytest.fail("partial schedule reached HREMD launch"),
    )

    acfg = ABFEInputConfig(
        outdir=str(tmp_path),
        sdffile=str(ligand_path),
        itpfile=str(itp_path),
        pro_lam_json=pro_lam_json,
        stages=["sysB_v2"],
    )
    acfg.check()
    stage = ABStageExecutionBase.create(
        ABStage.sysB_v2,
        acfg=acfg,
        sdf_abs=str(ligand_path),
        n_cuda_devices=1,
        planned_stages=[ABStage.sysB_v2],
        job_context=job_context,
    )

    with pytest.raises(ValueError, match="analytical Boresch correction.*full restraint endpoint"):
        stage.exec()


def test_sysB_optimizer_rejects_partial_restraint_lambda_end(tmp_path, monkeypatch):
    pytest.importorskip("rdkit.Chem")
    from felis.protocols.abfe.config_types import ABFEInputConfig
    from felis.protocols.abfe.config_types import ABStage
    from felis.protocols.abfe.config_types import ABStageExecutionBase

    job_context, ligand_path, _itp_path = _make_stage_context(tmp_path, monkeypatch, ligand_charge=0)

    with pytest.raises(ValueError, match="restraint_lambda_end.*full restraint endpoint"):
        ABStageExecutionBase.create(
            ABStage.sysB_opt_lambdas_v2,
            acfg=ABFEInputConfig(),
            sdf_abs=str(ligand_path),
            n_cuda_devices=1,
            restraint_lambda_end=0.5,
            planned_stages=[ABStage.sysB_opt_lambdas_v2],
            job_context=job_context,
        )


@pytest.mark.parametrize("schedule_source", ["configured", "optimized"])
def test_sysB_optimizer_skip_rejects_partial_schedule(tmp_path, monkeypatch, schedule_source):
    pytest.importorskip("rdkit.Chem")
    from felis.protocols.abfe.config_types import ABFEInputConfig
    from felis.protocols.abfe.config_types import ABStage
    from felis.protocols.abfe.config_types import ABStageExecutionBase
    import felis.protocols.abfe_v2.stage_sysB as stage_sysB

    job_context, ligand_path, _itp_path = _make_stage_context(tmp_path, monkeypatch, ligand_charge=0)
    configured_path = tmp_path / "configured_sysB_lam.json"
    if schedule_source == "configured":
        configured_path.write_text(json.dumps({"ab": {"lam_list": PARTIAL_PROTEIN_SCHEDULE}}))
        pro_lam_json = str(configured_path)
    else:
        optimized_path = Path(job_context.prepare_dir) / stage_sysB.SYSB_OPT_LAM_JSON
        optimized_path.write_text(json.dumps({"ab": {"lam_list": PARTIAL_PROTEIN_SCHEDULE}}))
        pro_lam_json = None

    acfg = ABFEInputConfig(pro_lam_json=pro_lam_json, outdir=str(tmp_path))
    acfg.check()
    stage = ABStageExecutionBase.create(
        ABStage.sysB_opt_lambdas_v2,
        acfg=acfg,
        sdf_abs=str(ligand_path),
        n_cuda_devices=1,
        planned_stages=[ABStage.sysB_opt_lambdas_v2],
        job_context=job_context,
    )

    with pytest.raises(ValueError, match="analytical Boresch correction.*full restraint endpoint"):
        stage.exec()


def test_post_analysis_rejects_partial_restraint_endpoint_before_outputs(tmp_path):
    from felis.protocols.abfe.config_types import ABFEInputConfig
    from felis.protocols.abfe.config_types import ABStage
    from felis.protocols.abfe.config_types import ABStageExecutionBase

    prepare_dir = tmp_path / "prepare"
    prepare_dir.mkdir()
    (prepare_dir / "sysB_lam.json").write_text(json.dumps({"ab": {"lam_list": PARTIAL_PROTEIN_SCHEDULE}}))
    analysis_dir = tmp_path / "analysis"
    job_context = SimpleNamespace(
        prepare_dir=str(prepare_dir),
        analysis_dir=str(analysis_dir),
        trj_dir=str(tmp_path / "trj"),
    )
    stage = ABStageExecutionBase.create(
        ABStage.post_analysis_v2,
        acfg=ABFEInputConfig(),
        planned_stages=[ABStage.post_analysis_v2],
        job_context=job_context,
    )

    with pytest.raises(ValueError, match="analytical Boresch correction.*full restraint endpoint"):
        stage.exec()
    assert not analysis_dir.exists()
