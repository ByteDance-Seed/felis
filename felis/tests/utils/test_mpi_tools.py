# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types


def _import_mpi_tools(monkeypatch, mpiplus_module):
    """Import `felis.utils.mpi_tools` with a controlled `mpiplus` module."""
    monkeypatch.setitem(sys.modules, "mpiplus", mpiplus_module)
    monkeypatch.delitem(sys.modules, "felis.utils.mpi_tools", raising=False)
    return importlib.import_module("felis.utils.mpi_tools")


def test_mpi_tools_falls_back_when_get_mpicomm_returns_none(monkeypatch):
    fake_mpiplus = types.SimpleNamespace(get_mpicomm=lambda: None)
    mpi_tools = _import_mpi_tools(monkeypatch, mpiplus_module=fake_mpiplus)

    assert mpi_tools.mpicomm.mpi_rank == 0
    assert mpi_tools.mpicomm.mpi_nproc == 1
    assert mpi_tools.mpicomm._comm is None

    # No-op synchronization and identity broadcast in the stub mode.
    assert mpi_tools.mpicomm.barrier() is None
    obj = {"k": "v"}
    assert mpi_tools.mpicomm.bcast(obj, root=0) is obj


def test_mpi_tools_uses_backend_comm_when_available(monkeypatch):

    class _FakeComm:

        def __init__(self):
            self.size = 4
            self.rank = 2
            self.barrier_called = False

        def barrier(self):
            self.barrier_called = True

        def bcast(self, data, root=0):
            return ("bcast", data, root)

    comm = _FakeComm()
    fake_mpiplus = types.SimpleNamespace(get_mpicomm=lambda: comm)

    mpi_tools = _import_mpi_tools(monkeypatch, mpiplus_module=fake_mpiplus)

    assert mpi_tools.mpicomm.mpi_nproc == 4
    assert mpi_tools.mpicomm.mpi_rank == 2
    assert mpi_tools.mpicomm._comm is comm

    mpi_tools.mpicomm.barrier()
    assert comm.barrier_called is True

    assert mpi_tools.mpicomm.bcast("x", root=1) == ("bcast", "x", 1)
