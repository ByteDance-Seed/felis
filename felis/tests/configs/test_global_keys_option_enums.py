# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import contextlib
from pathlib import Path


def test_minimize_relax_option():

    from felis.configs import GlobalKeys
    from felis.configs import MinimizeRelaxOption

    op1 = MinimizeRelaxOption(0)
    assert op1.value == 0
    gk1 = GlobalKeys()
    gk1.update_by_tkv("i:integrator.minimize:00")
    gk1.check()
    assert gk1.integrator.minimize == op1.value

    op2 = MinimizeRelaxOption("1")
    assert op2.value == 1
    gk2 = GlobalKeys()
    gk2.update_by_tkv("s:integrator.minimize:001")
    gk2.check()
    assert gk2.integrator.minimize == op2.value

    op3 = MinimizeRelaxOption("l_bFGs")
    assert op3.value == MinimizeRelaxOption.l_bfgs.value
    gk3 = GlobalKeys()
    gk3.update_by_tkv("s:integrator.minimize:L_BfgS")
    gk3.check()
    assert gk3.integrator.minimize == op3.value

    op4 = MinimizeRelaxOption("BFGS")
    assert op4.value == MinimizeRelaxOption.l_bfgs.value
    gk4 = GlobalKeys()
    gk4.update_by_tkv("s:integrator.minimize:bfGs")
    gk4.check()
    assert gk4.integrator.minimize == op4.value


def test_integrator_name_option_case_insensitive():

    from felis.configs import IntegratorNameOption

    assert IntegratorNameOption("langevinmiddleintegrator") is IntegratorNameOption.LangevinMiddleIntegrator
    assert IntegratorNameOption("brownianintegrator") is IntegratorNameOption.BrownianIntegrator
    assert IntegratorNameOption("fire2") is IntegratorNameOption.FIRE2


def test_minimize_relax_option_in_abfeconfig(tmp_path: Path):

    from felis.configs import MinimizeRelaxOption
    from felis.protocols.abfe.config_types import ABFEInputConfig

    with contextlib.chdir(tmp_path):
        file1 = tmp_path / "f1.json"
        file1.write_text(r"{md_sol_em_version: BfGS, md_pro_em_version: 3}")
        abcfg1 = ABFEInputConfig.from_file(file1)
        abcfg1.outdir = str(tmp_path)
        abcfg1.check()
        assert abcfg1.md_sol_em_version == MinimizeRelaxOption.l_bfgs.value
        assert abcfg1.md_pro_em_version == MinimizeRelaxOption.brownian.value

        file2 = tmp_path / "f2.json"
        file2.write_text(r'{md_sol_em_version: BfGS, md_pro_em_version: "3"}')
        abcfg2 = ABFEInputConfig.from_file(file2)
        abcfg2.outdir = str(tmp_path)
        abcfg2.check()
        assert abcfg2.md_sol_em_version == MinimizeRelaxOption.l_bfgs.value
        assert abcfg2.md_pro_em_version == MinimizeRelaxOption.brownian.value

        file3 = tmp_path / "f3.yaml"
        file3.write_text(r"""md_sol_em_version: bFGS
md_pro_em_version: 3""")
        abcfg3 = ABFEInputConfig.from_file(file3)
        abcfg3.outdir = str(tmp_path)
        abcfg3.check()
        assert abcfg3.md_sol_em_version == MinimizeRelaxOption.l_bfgs.value
        assert abcfg3.md_pro_em_version == MinimizeRelaxOption.brownian.value

        file4 = tmp_path / "f4.yaml"
        file4.write_text(r'''md_sol_em_version: Bfgs
md_pro_em_version: "3"''')
        abcfg4 = ABFEInputConfig.from_file(file4)
        abcfg4.outdir = str(tmp_path)
        abcfg4.check()
        assert abcfg4.md_sol_em_version == MinimizeRelaxOption.l_bfgs.value
        assert abcfg4.md_pro_em_version == MinimizeRelaxOption.brownian.value
