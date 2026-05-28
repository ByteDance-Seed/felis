# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

def test_calculate_and_write_dg_to_file_formats_output(tmp_path, monkeypatch):
    import felis.protocols.boresch.calculate_and_write_dg as mod

    monkeypatch.setattr(mod, "BR_correction_2023", lambda *_args, **_kwargs: 2.0)

    ideal = {"r0": 0.1, "theta0": 90.0, "phi0": 0.0, "alpha0": 90.0, "beta0": 0.0, "gamma0": 0.0}
    correction_kcal = mod.calculate_and_write_dg_to_file(str(tmp_path), 300.0, ideal, 1.0, 2.0, 3.0)
    assert correction_kcal == 2.0

    outpath = tmp_path / "sysR_fe_table.txt"
    lines = outpath.read_text().splitlines()
    assert lines[0].startswith("StateA\tStateB\tdG+")
    assert lines[1].startswith("Sum\tkJ/mol\t8.3680\t\t-8.3680")
    assert lines[2].startswith("Sum\tkcal/mol\t2.0000\t\t-2.0000")
