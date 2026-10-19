# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

from pathlib import Path


def test_abfe_config_rejects_unknown_boresch_sampling_protocol(tmp_path):
    from felis.protocols.abfe.config_types import ABFEInputConfig
    import pytest

    acfg = ABFEInputConfig(
        sdffile="ligand0.sdf",
        itpfile="ligand0.itp",
        outdir=str(tmp_path),
        tmpdir=str(tmp_path),
        stages=["boresch_npt_v2"],
        boresch_sampling_protocol="bad_protocol",
    )

    with pytest.raises(ValueError, match="boresch_sampling_protocol"):
        acfg.check()


def test_abfe_config_rejects_unaligned_boresch_hremd_interval(tmp_path):
    from felis.protocols.abfe.config_types import ABFEInputConfig
    import pytest

    acfg = ABFEInputConfig(
        outdir=str(tmp_path),
        md_exchange_interval=1000,
        md_traj_interval=125000,
        md_checkpoint_interval=125000,
        md_boresch_traj_interval=2500,
        boresch_sampling_protocol="hremd_same_temperature",
    )

    with pytest.raises(
        ValueError,
        match="md_boresch_traj_interval must be a multiple of md_exchange_interval",
    ):
        acfg.check()


def test_abfe_config_allows_unaligned_boresch_long_npt_interval(tmp_path):
    from felis.protocols.abfe.config_types import ABFEInputConfig

    acfg = ABFEInputConfig(
        outdir=str(tmp_path),
        md_exchange_interval=1000,
        md_traj_interval=125000,
        md_checkpoint_interval=125000,
        md_boresch_traj_interval=2500,
        boresch_sampling_protocol="long_npt",
    )

    acfg.check()


def test_abfe_config_rejects_checkpoint_shorter_than_boresch_hremd_interval(
    tmp_path,
):
    from felis.protocols.abfe.config_types import ABFEInputConfig
    import pytest

    acfg = ABFEInputConfig(
        outdir=str(tmp_path),
        md_exchange_interval=250,
        md_traj_interval=1000,
        md_checkpoint_interval=1000,
        md_boresch_traj_interval=2000,
        boresch_sampling_protocol="hremd_same_temperature",
    )

    with pytest.raises(
        ValueError,
        match="md_checkpoint_interval must be >= md_boresch_traj_interval",
    ):
        acfg.check()


def _make_boresch_stage(tmp_path, monkeypatch, protocol, discard=7):
    from felis.protocols.abfe.config_types import ABFEInputConfig
    from felis.protocols.abfe.config_types import ABStage
    from felis.protocols.abfe.config_types import ABStageExecutionBase
    from felis.protocols.abfe.job_context import JobContext
    import felis.protocols.abfe_v2.stage_boresch as stage_boresch

    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stage_boresch, "get_visible_cuda_devices", lambda: [0, 1])

    def _fake_build_cuda_mps_command(self, cmd, gpu_id=None):
        calls.append(("mps", cmd, gpu_id))
        return ["wrapped", *cmd]

    def _fake_run_one_call(self, cmd, working_dir=None, extra_envs=None):
        calls.append(("run", cmd, working_dir, extra_envs))
        trj_dcd = Path(self.trj_dir) / "sysB_boresch_npt.dcd"
        trj_dcd.parent.mkdir(parents=True, exist_ok=True)
        trj_dcd.touch()
        return 0

    def _fake_discard(gk, trj, discard):
        calls.append(("discard", gk.filename.sys, trj, discard))

    def _fake_concat(gk, trj_list, out_trj, discard=0):
        calls.append(("concat", gk.filename.sys, trj_list, out_trj, discard))

    monkeypatch.setattr(JobContext, "build_cuda_mps_command", _fake_build_cuda_mps_command)
    monkeypatch.setattr(JobContext, "run_one_call", _fake_run_one_call)
    monkeypatch.setattr(stage_boresch, "discard_initial_trajectory_frames", _fake_discard)
    monkeypatch.setattr(stage_boresch, "concatenate_trajectory_files", _fake_concat)

    job_context = JobContext("ligand0")
    sys_top = Path(job_context.prepare_dir) / "sysB.top"
    sys_top.touch()

    acfg = ABFEInputConfig(
        sdffile="ligand0.sdf",
        outdir=str(tmp_path),
        tmpdir=str(tmp_path),
        stages=["boresch_npt_v2"],
        boresch_sampling_protocol=protocol,
    )

    stage = ABStageExecutionBase.create(
        ABStage.boresch_npt_v2,
        acfg=acfg,
        sdf_abs=str(tmp_path / "ligand0.sdf"),
        discard=discard,
        planned_stages=[ABStage.boresch_npt_v2],
        job_context=job_context,
    )
    return stage, job_context, calls


def _command_after(cmd, flag):
    return cmd[cmd.index(flag) + 1:]


def test_boresch_npt_v2_long_npt_discards_initial_frames(tmp_path, monkeypatch):
    stage, job_context, calls = _make_boresch_stage(
        tmp_path, monkeypatch, protocol="long_npt", discard=7
    )

    stage.exec()

    run_call = next(call for call in calls if call[0] == "run")
    run_cmd = run_call[1]
    assert "felis.app.dyn.replica_simulation" in run_cmd
    assert "i:integrator.nsnapshots:1200" in run_cmd
    assert "i:integrator.nstep_per_snapshot:2500" in run_cmd
    assert calls[-1] == (
        "discard",
        str(Path(job_context.prepare_dir) / "sysB.top"),
        str(Path(job_context.trj_dir) / "sysB_boresch_npt.dcd"),
        7,
    )
    assert (Path(job_context.progress_dir) / "boresch_npt_v2.done").is_file()


def test_boresch_npt_v2_threads_hmr_to_runtime_commands(tmp_path, monkeypatch):
    stage, _job_context, calls = _make_boresch_stage(
        tmp_path, monkeypatch, protocol="hremd_same_temperature", discard=0
    )
    stage._acfg.md_integrator_dt_ps = 0.004
    stage._acfg.md_integrator_hmr = 3.0

    stage.exec()

    mps_call = next(call for call in calls if call[0] == "mps")
    cmd = mps_call[1]
    assert "f:integrator.dt_ps:0.004" in cmd
    assert "f:integrator.hmr:3.0" in cmd


def test_boresch_npt_v2_hremd_same_temperature_protocol(tmp_path, monkeypatch):
    stage, job_context, calls = _make_boresch_stage(
        tmp_path, monkeypatch, protocol="hremd_same_temperature", discard=9
    )

    stage.exec()

    mps_call = next(call for call in calls if call[0] == "mps")
    cmd = mps_call[1]
    assert "felis.app.dyn.repex_v2" in cmd
    assert "prepare/sysB_boresch_lam.json" in _command_after(cmd, "--cfg")
    assert "i:integrator.nsnapshots:300" in cmd
    assert "i:integrator.nstep_per_snapshot:2500" in cmd
    assert _command_after(cmd, "--rextkv") == [
        "f:integrator.targetT_K:298.15",
        "f:integrator.targetT_K:298.15",
        "f:integrator.targetT_K:298.15",
        "f:integrator.targetT_K:298.15",
    ]
    assert mps_call[2] == [0, 1]

    concat_call = next(call for call in calls if call[0] == "concat")
    assert concat_call == (
        "concat",
        str(Path(job_context.prepare_dir) / "sysB.top"),
        [
            str(Path(job_context.trj_dir) / "sysB_boresch_npt_hremd" / f"replica_{idx}" / "traj.dcd")
            for idx in range(4)
        ],
        str(Path(job_context.trj_dir) / "sysB_boresch_npt.dcd"),
        9,
    )
    assert (Path(job_context.progress_dir) / "boresch_npt_v2.done").is_file()
