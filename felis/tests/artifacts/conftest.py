# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

_ARTIFACTS_ENABLED = os.environ.get("FELIS_TEST_ARTIFACTS") == "1"

# When the gating environment variable is not set to "1", skip collecting any
# tests under this directory rather than aborting the entire pytest run with a
# configuration error. This keeps unrelated test runs functional.
collect_ignore_glob = ["*"] if not _ARTIFACTS_ENABLED else []


def pytest_configure(config):
    if not _ARTIFACTS_ENABLED:
        return

    from felis.tests.testkit.bytemol.shims import install_shims_bytemol
    from felis.tests.testkit.openmm.shims import install_shims_openmm
    from felis.tests.testkit.openmmtools.shims import install_shims_openmmtools

    install_shims_openmm()
    install_shims_openmmtools()
    install_shims_bytemol()

    import felis.utils
    from felis.tests.testkit.mock import mock_extract_state_from_netdcd, mock_get_visible_cuda_devices
    ndev = 8  # Default number of mock devices
    felis.utils.get_visible_cuda_devices = mock_get_visible_cuda_devices(ndev)

    import felis.utils.omm.format_tools
    felis.utils.omm.format_tools.extract_state_from_netdcd = mock_extract_state_from_netdcd


def pytest_collection_modifyitems(config, items):
    """Centrally mark every test collected under this directory as ``artifact``.

    This avoids requiring each artifact test module to repeat the marker and lets
    contributors target ``-m "not artifact"`` or ``-m artifact`` reliably.
    """
    artifact_marker = pytest.mark.artifact
    conftest_dir = os.path.dirname(__file__)
    for item in items:
        try:
            item_path = str(item.path)
        except AttributeError:
            item_path = str(getattr(item, "fspath", ""))
        if item_path.startswith(conftest_dir):
            item.add_marker(artifact_marker)


@pytest.fixture(autouse=True)
def _patch_sys_modules_for_artifacts(monkeypatch):
    if not _ARTIFACTS_ENABLED:
        return

    from felis.tests.testkit import package_root

    tests_root = str(Path(package_root()) / "felis" / "tests")
    pythonpath_entries = [tests_root, package_root()]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(pythonpath_entries))
    monkeypatch.setenv("OMPI_MCA_pml", "ob1")
    monkeypatch.setenv("OMPI_MCA_btl", "self,vader")
    monkeypatch.setenv("OMPI_MCA_hwloc_base_binding_policy", "none")
