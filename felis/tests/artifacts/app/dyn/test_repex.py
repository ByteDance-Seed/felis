# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import contextlib
from pathlib import Path

import pytest

from felis.tests.testkit import assert_manifest
from felis.utils.mpi_tools import mpicomm


@pytest.mark.parametrize("nproc", [1, 4])
def test_repex_simulation_artifacts(monkeypatch, tmp_path: Path, nproc):

    stem = "case0"
    trj_dir = Path("trj")
    tmp_crd = Path(f"{stem}.gro")
    tmp_top = Path("protensystem.top")

    from felis.app.dyn.repex import mainfunc as cli_mainfunc
    from felis.protocols.dyn.main_replica_exchange import mainfunc as protocol_mainfunc

    with contextlib.chdir(tmp_path):

        tmp_crd.touch()
        tmp_top.touch()

        monkeypatch.setattr(mpicomm, "mpi_nproc", nproc)
        monkeypatch.setattr(mpicomm, "barrier", lambda: None)

        for rank in range(nproc):
            monkeypatch.setattr(mpicomm, "mpi_rank", rank)
            cli_mainfunc(
                argv=[
                    "--tkv",
                    f"s:dir.outbase:{tmp_path}",
                    f"s:dir.trj:{trj_dir}",
                    f"s:filename.stem:{stem}",
                    f"s:filename.sys:{tmp_top}",
                    f"s:filename.crd:{tmp_crd}",
                    "--rextkv",
                    "f:integrator.targetT_K:298.15",
                    "f:integrator.targetT_K:298.15",
                ],
                protocol_mainfunc=protocol_mainfunc,
            )

        expected_rel_files = {
            str(tmp_top),
            str(tmp_crd),
            f"{trj_dir}/{stem}.create_done",
            f"{trj_dir}/{stem}_checkpoint.nc",
            f"{trj_dir}/{stem}.nc",
        }
        for i in range(nproc):
            expected_rel_files.add(f"{trj_dir}/{stem}.rank{i:02d}.rexlog")

        assert_manifest(tmp_path, expected_rel_files=expected_rel_files)
