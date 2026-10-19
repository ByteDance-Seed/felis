# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from felis.app.tools import analysis4records
from felis.app.tools import optimize_protein_leg_lambdas as protein_optimizer
from felis.app.tools import optimize_water_leg_lambdas as water_optimizer
from felis.protocols.boresch.main_boresch_restraints import BORESCH_CFG_JSON
from felis.protocols.dyn import main_replica_exchange_v2


def _make_config():
    return SimpleNamespace(
        omm_tmp_directory="/tmp",
        dt_tkv=lambda: "f:integrator.dt_ps:0.004",
        hmr_tkv=lambda: "f:integrator.hmr:1.5",
    )


def _prepare_inputs(prepare_dir, prefix):
    names = [
        f"{prefix}.top",
        f"{prefix}_em.pdb",
        f"{prefix}_ab_ligatoms.json",
        f"{prefix}_atom_ids.json",
    ]
    if prefix == "sysB":
        names.append(BORESCH_CFG_JSON)
    for name in names:
        (prepare_dir / name).write_text("test")


def _patch_short_hremd(monkeypatch, module, captured):
    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "read_records", lambda _path: [{}])
    monkeypatch.setattr(
        module, "get_acceptance_ratios_from_records", lambda _records: np.array([0.5])
    )
    monkeypatch.setattr(
        module, "get_flow_drops_from_records", lambda _records: np.array([0.5])
    )
    monkeypatch.setattr(
        module, "get_round_trip_rate_from_records", lambda _records: 0.0
    )


@pytest.mark.parametrize(
    "module,prefix",
    [
        (water_optimizer, "sysA"),
        (protein_optimizer, "sysB"),
    ],
)
@pytest.mark.parametrize(
    "inherited_visibility,expected_visibility",
    [
        ("4,7", "7"),
        ("GPU-aaaaaaaa,GPU-bbbbbbbb", "GPU-bbbbbbbb"),
    ],
)
def test_no_mps_passes_selected_gpu_visibility(
    monkeypatch,
    tmp_path,
    module,
    prefix,
    inherited_visibility,
    expected_visibility,
):
    prepare_dir = tmp_path / "prepare"
    prepare_dir.mkdir()
    _prepare_inputs(prepare_dir, prefix)
    captured = {}
    _patch_short_hremd(monkeypatch, module, captured)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", inherited_visibility)
    selected_gpu_ids = module.resolve_cuda_device_ids(["1"])

    run_hremd = module.make_run_hremd_fn(
        opt_root=tmp_path / "opt",
        prepare_dir=prepare_dir,
        sdf_abs=tmp_path / "ligand.sdf",
        acfg=_make_config(),
        nsnapshots=2,
        traj_interval=10,
        exchange_interval=5,
        checkpoint_interval=10,
        gpu_ids=selected_gpu_ids,
        use_mps=False,
        no_plots=True,
    )
    schedule = np.zeros((2, 7), dtype=np.float64)
    if prefix == "sysB":
        schedule[-1, 6] = 1.0
    run_hremd(schedule)

    assert captured["command"][:3] == ["python3", "-m", "felis.app.dyn.repex_v2"]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == expected_visibility


@pytest.mark.parametrize(
    "visible_tokens,expected_indices",
    [
        ("4,7", ["0", "1"]),
        ("GPU-aaaaaaaa,GPU-bbbbbbbb", ["0", "1"]),
        ("", []),
    ],
)
def test_repex_uses_logical_indices_after_cuda_visibility(
    monkeypatch, tmp_path, visible_tokens, expected_indices
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_tokens)
    captured = {}
    gk = SimpleNamespace(
        integrator=SimpleNamespace(
            nsnapshots=2,
            nstep_per_snapshot=10,
            npt=1,
            targetT_K=298.15,
        ),
        openmm=SimpleNamespace(
            exchange_interval=5,
            checkpoint_interval=10,
            tmp_directory="/tmp",
            resume=False,
        ),
        dir=SimpleNamespace(trj=str(tmp_path)),
        filename=SimpleNamespace(stem="test"),
    )

    monkeypatch.setattr(
        main_replica_exchange_v2, "get_list_of_replicas", lambda _gks: [object()]
    )

    def fake_run_hremd(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(main_replica_exchange_v2, "run_hremd", fake_run_hremd)

    main_replica_exchange_v2.mainfunc([gk])

    assert captured["gpu_devices"] == expected_indices


def test_water_no_plots_skips_per_iteration_overlap(monkeypatch, tmp_path):
    prepare_dir = tmp_path / "prepare"
    prepare_dir.mkdir()
    _prepare_inputs(prepare_dir, "sysA")
    captured = {}
    _patch_short_hremd(monkeypatch, water_optimizer, captured)

    def fail_if_plotting_is_loaded():
        raise AssertionError("plotting should remain disabled")

    monkeypatch.setattr(
        water_optimizer, "_load_overlap_plotter", fail_if_plotting_is_loaded
    )

    run_hremd = water_optimizer.make_run_hremd_fn(
        opt_root=tmp_path / "opt",
        prepare_dir=prepare_dir,
        sdf_abs=tmp_path / "ligand.sdf",
        acfg=_make_config(),
        nsnapshots=2,
        traj_interval=10,
        exchange_interval=5,
        checkpoint_interval=10,
        gpu_ids=["0"],
        use_mps=False,
        no_plots=True,
    )
    run_hremd(np.zeros((2, 7), dtype=np.float64))

    assert not (tmp_path / "opt" / "iter_00" / "overlap_matrix.png").exists()
    assert not (tmp_path / "opt" / "iter_00" / "overlap_matrix.npy").exists()


def test_protein_optimizer_cli_rejects_partial_restraint_endpoint(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_protein_leg_lambdas",
            "--abfecfg",
            "abfe.yaml",
            "--job-name",
            "ligand0",
            "--restraint-lambda-end",
            "0.5",
        ],
    )

    with pytest.raises(SystemExit):
        protein_optimizer.parse_args()

    assert "analytical Boresch correction only supports a full restraint endpoint" in capsys.readouterr().err


def test_protein_optimizer_initial_json_rejects_partial_restraint_endpoint(tmp_path):
    lam_path = tmp_path / "partial_sysB_lam.json"
    lam_path.write_text(json.dumps({
        "ab": {
            "lam_list": [
                [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5],
            ],
        },
    }))

    with pytest.raises(ValueError, match="analytical Boresch correction.*full restraint endpoint"):
        protein_optimizer.load_initial_schedule(lam_path)


def test_analysis_cli_rejects_partial_restraint_endpoint_before_reading_records(tmp_path, monkeypatch):
    prepare_dir = tmp_path / "prepare"
    prepare_dir.mkdir()
    (prepare_dir / "sysB_lam.json").write_text(json.dumps({
        "ab": {
            "lam_list": [
                [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5],
            ],
        },
    }))
    monkeypatch.setattr(
        analysis4records,
        "read_records",
        lambda *_args, **_kwargs: pytest.fail("partial schedule reached record analysis"),
    )

    with pytest.raises(ValueError, match="analytical Boresch correction.*full restraint endpoint"):
        analysis4records.main([
            "--energy_matrix",
            str(tmp_path / "records.sqlite"),
            "--outdir",
            str(tmp_path / "analysis"),
            "--prepare_dir",
            str(prepare_dir),
        ])
