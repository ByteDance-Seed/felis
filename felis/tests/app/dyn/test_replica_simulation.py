# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging

import felis.app.dyn.replica_simulation as replica_simulation


def test_replica_simulation_uses_override_applied_config(monkeypatch):
    captured = {}

    class _State:

        def getPotentialEnergy(self):
            return 0.0

    class _Context:

        def getState(self, **_kwargs):
            return _State()

    class _Replica:

        def __init__(self, **kwargs):
            captured["replica_kwargs"] = kwargs
            self.context = _Context()

        def set_integrator(self, integrator):
            captured["integrator"] = integrator

        def set_num_replicas(self, _count):
            pass

        def set_platform(self, _platform, _properties):
            pass

        def init_context(self):
            pass

        def open_trajectory(self, path, dt_ps):
            captured["trajectory"] = (path, dt_ps)

        def step(self, _steps):
            pass

        def write_trajectory(self):
            pass

        def close_trajectory(self):
            pass

    def _get_system_topology(gk, **_kwargs):
        captured["system_gk"] = gk
        return object(), object()

    def _get_initial_pos_pbc(gk):
        captured["positions_gk"] = gk
        return object(), object()

    def _get_integrator(gk):
        captured["integrator_gk"] = gk
        return object()

    monkeypatch.setattr(replica_simulation.GlobalKeys, "check", lambda _self: None)
    monkeypatch.setattr(replica_simulation, "get_system_topology", _get_system_topology)
    monkeypatch.setattr(replica_simulation, "get_initial_pos_pbc", _get_initial_pos_pbc)
    monkeypatch.setattr(replica_simulation, "get_integrator", _get_integrator)
    monkeypatch.setattr(replica_simulation, "Replica", _Replica)
    monkeypatch.setattr(replica_simulation.Platform, "getPlatformByName", lambda _name: object())
    monkeypatch.setattr(replica_simulation, "setup_logger", lambda _path: logging.getLogger(__name__))
    monkeypatch.setattr(replica_simulation, "mkdir_and_get_filepath", lambda **_kwargs: "run.log")

    replica_simulation.mainfunc([
        "--rextkv",
        (
            "f:integrator.targetT_K:345.0,"
            "f:integrator.targetP_bar:2.5,"
            "f:integrator.dt_ps:0.005,"
            "i:integrator.randomseed:123,"
            "i:integrator.nsnapshots:0,"
            "s:dir.trj:override_trj"
        ),
    ])

    selected_gk = captured["system_gk"]
    assert captured["positions_gk"] is selected_gk
    assert captured["integrator_gk"] is selected_gk
    assert selected_gk.integrator.targetT_K == 345.0
    assert selected_gk.integrator.targetP_bar == 2.5
    assert selected_gk.integrator.dt_ps == 0.005
    assert selected_gk.integrator.randomseed == 123

    replica_kwargs = captured["replica_kwargs"]
    assert replica_kwargs["pressure"] == 2.5
    assert replica_kwargs["temperature"] == 345.0
    assert replica_kwargs["velocity_temperature_K"] == 345.0
    assert replica_kwargs["velocity_seed"] == 123
    assert captured["trajectory"] == ("override_trj/sysB_boresch_npt.dcd", 0.005)
