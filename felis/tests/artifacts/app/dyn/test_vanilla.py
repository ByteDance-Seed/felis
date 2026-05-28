# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import contextlib
from pathlib import Path

import pytest

from felis.tests.testkit import assert_manifest


@pytest.mark.parametrize("crd", ["gro", "pdb"])
@pytest.mark.parametrize("npt", [0, 1])
def test_vanilla_simulation_artifacts(tmp_path: Path, crd, npt):

    stem = "case0"
    trj_dir = Path("trj")
    tmp_crd = Path(f"{stem}.{crd}")
    tmp_top = Path("protensystem.top")

    from felis.app.dyn.vanilla import mainfunc as cli_mainfunc
    from felis.protocols.dyn.main_vanilla import mainfunc as protocol_mainfunc

    with contextlib.chdir(tmp_path):

        tmp_crd.touch()
        tmp_top.touch()

        cli_mainfunc(
            argv=[
                "--tkv",
                f"s:dir.outbase:{tmp_path}",
                f"s:dir.trj:{trj_dir}",
                f"s:filename.stem:{stem}",
                f"s:filename.sys:{tmp_top}",
                f"s:filename.crd:{tmp_crd}",
                f"i:integrator.npt:{npt}",
            ],
            protocol_mainfunc=protocol_mainfunc,
        )

        assert_manifest(trj_dir,
                        expected_rel_files={
                            f"{stem}.dynlog",
                            f"{stem}.csv",
                            f"{stem}.dcd",
                            f"{stem}.pdb",
                            f"{stem}.chk",
                        })


@pytest.mark.parametrize("crd", ["gro", "pdb"])
@pytest.mark.parametrize(("version_type", "minimize_version", "expected_rel_files"),
                         [("i", 1, {"case1_em.dynlog", "case1_em.pdb"}), ("i", 2, {"case1_em.dynlog", "case1_em.pdb"}),
                          ("i", 3, {"case1_em.dynlog", "case1_em.pdb", "case1_em.em3.csv"}),
                          ("s", "FIRE2.0", {"case1_em.dynlog", "case1_em.pdb", "case1_em.em4.csv"})])
def test_vanilla_minimization_artifacts(tmp_path: Path, crd, version_type, minimize_version, expected_rel_files):

    stem = "case1"
    trj_dir = Path("trj")
    tmp_crd = Path(f"{stem}.{crd}")
    tmp_top = Path("protensystem.top")

    from felis.app.dyn.vanilla import mainfunc as cli_mainfunc
    from felis.protocols.dyn.main_vanilla import mainfunc as protocol_mainfunc

    with contextlib.chdir(tmp_path):

        tmp_crd.touch()
        tmp_top.touch()

        cli_mainfunc(
            argv=[
                "--tkv",
                f"s:dir.outbase:{tmp_path}",
                f"s:dir.trj:{trj_dir}",
                f"s:filename.stem:{stem}_em",
                f"s:filename.sys:{tmp_top}",
                f"s:filename.crd:{tmp_crd}",
                "i:integrator.nstep_per_snapshot:50000",
                f"{version_type}:integrator.minimize:{minimize_version}",
            ],
            protocol_mainfunc=protocol_mainfunc,
        )

        assert_manifest(trj_dir, expected_rel_files)
