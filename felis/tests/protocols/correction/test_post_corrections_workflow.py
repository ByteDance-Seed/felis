# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exercise end-to-end smoke tests for post-corrections workflows."""

import json

import pandas as pd

from felis.protocols.correction.post_corrections import PostCorrections

from felis.tests.protocols.correction.post_corrections_helpers import assert_ab_post_correction_artifacts
from felis.tests.protocols.correction.post_corrections_helpers import assert_post_correction_node_df
from felis.tests.protocols.correction.post_corrections_helpers import post_corrections_testdata_root
from felis.tests.protocols.correction.post_corrections_helpers import resolve_workflow_output_path


def test_ab_workflow(tmp_path):
    """Verify that the AB workflow emits the expected post-correction artifacts.

    Args:
        tmp_path (Path): Temporary output directory used for the generated TSV
            artifacts.
    """
    root = post_corrections_testdata_root()
    config_path = root / "correction_config.json"
    csv_path = root / "compare-dG.tsv"
    with open(config_path, "r") as f:
        config = json.load(f)
    pc = PostCorrections(str(csv_path), config, output_dir=str(tmp_path), rt=0.596, pka_rt=0.596)
    out_path = pc.run(debug=True)

    resolved = resolve_workflow_output_path(tmp_path, out_path)
    df = pd.read_csv(resolved, sep="\t")
    assert_post_correction_node_df(df)

    assert_ab_post_correction_artifacts(tmp_path, csv_path.stem)