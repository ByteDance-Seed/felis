# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Site-level bootstrap for FELIS artifact tests.

This module is imported automatically by Python (via ``site``) when the
``felis/tests`` directory is present on ``PYTHONPATH``. Artifact tests use it to
install lightweight shims in child Python processes.

Enable artifact/contract-test mode with:
    - ``FELIS_TEST_ARTIFACTS=1``
"""

import os

if os.environ.get("FELIS_TEST_ARTIFACTS", "").strip() == "1":
    from felis.tests.testkit.openmm.shims import install_shims_openmm
    install_shims_openmm()

    from felis.tests.testkit.openmmtools.shims import install_shims_openmmtools
    install_shims_openmmtools()

    from felis.tests.testkit.bytemol.shims import install_shims_bytemol
    install_shims_bytemol()
