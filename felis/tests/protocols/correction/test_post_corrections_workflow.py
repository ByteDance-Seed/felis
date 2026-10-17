# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Exercise end-to-end smoke tests for post-corrections workflows."""

import json

import pandas as pd

from felis.protocols.correction.post_corrections import PostCorrections
from felis.protocols.correction.post_corrections import RBPostCorrections
from felis.tests.protocols.correction.post_corrections_helpers import assert_ab_post_correction_artifacts
from felis.tests.protocols.correction.post_corrections_helpers import assert_post_correction_node_df
from felis.tests.protocols.correction.post_corrections_helpers import assert_rbfe_post_correction_artifacts
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


def test_rb_workflow(tmp_path):
    """Verify the RBFE workflow with only symmetry and rotamer corrections.

    Args:
        tmp_path (Path): Temporary output directory used for the generated TSV
            artifacts.
    """
    edges = pd.DataFrame({
        "Lig 1": ["A", "B", "A"],
        "Lig 2": ["B", "C", "C"],
        "CCC ddG (kcal/mol)": [1.0, 1.0, 2.4],
    })
    edges_path = tmp_path / "rbfe_edges.csv"
    edges.to_csv(edges_path, index=False)

    cfg = {
        "symmetry": {
            "B": 2
        },
        "rotamer": [["A", "B"]],
    }

    pc = RBPostCorrections(str(edges_path), cfg, output_dir=str(tmp_path), rt=0.596, pka_rt=0.596)
    out_path = pc.run(debug=False)
    resolve_workflow_output_path(tmp_path, out_path)

    assert_rbfe_post_correction_artifacts(tmp_path, edges_path.stem)
