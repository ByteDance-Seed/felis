# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np


def test_filter_and_mask_and_get_and_plot_histogram_writes_plot(tmp_path, monkeypatch):
    import felis.protocols.boresch.filter_mask_histogram_plot as mod

    # Only the second frame passes the VALID_ANGLE_RANGE (45..135) on theta.
    raw = {
        "r": np.array([0.5, 0.6, 0.7], dtype=float),
        "theta": np.array([30.0, 90.0, 150.0], dtype=float),
        "phi": np.array([-10.0, 20.0, -170.0], dtype=float),
        "alpha": np.array([90.0, 90.0, 90.0], dtype=float),
        "beta": np.array([-1.0, 2.0, -3.0], dtype=float),
        "gamma": np.array([4.0, -5.0, 6.0], dtype=float),
    }

    def _fake_savefig(path: str):
        Path(path).write_bytes(b"")

    monkeypatch.setattr(mod.plt, "savefig", _fake_savefig)

    out = mod.filter_and_mask_and_get_and_plot_histogram(str(tmp_path), raw, nbins=5)
    assert out["geom_mask"].tolist() == [False, True, False]
    assert out["normalized"]["phi360"].tolist() == [350.0, 20.0, 190.0]
    assert out["normalized"]["beta360"].tolist() == [359.0, 2.0, 357.0]
    assert out["normalized"]["gamma360"].tolist() == [4.0, 355.0, 6.0]

    assert len(out["binmax"]) == 9
    assert (tmp_path / "sys_boresch0.png").exists()
