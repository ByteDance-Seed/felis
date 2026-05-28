# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path


class SystemBuilderConfig:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def prepare_system(config: SystemBuilderConfig):
    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "system.gro").touch()
    (outdir / "system.top").touch()
    atom_ids = {
        "ligands": {
            "LIG": [[0, 1, 2]]
        },
        "protein_heavy": {
            "PRO": [[3, 4, 5]]
        },
    }
    with open(outdir / "atom_ids.json", "w") as f:
        json.dump(atom_ids, f)
