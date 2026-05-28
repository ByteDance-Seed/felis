# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from felis.utils.topfile_tools import get_top_system_bonds_1direct, get_top_system_symbols


@pytest.fixture(scope="module", name="water_top_file")
def _water_top_file(tmp_path_factory) -> str:
    """Create a minimal GROMACS .top for a replicated water (H-O-H) system."""

    top_text = """
[ defaults ]
1           2            yes          0.5        0.8333

[ atomtypes ]
OW       8       15.9994   0.0     A       3.15075e-01 6.36386e-01
HW       H       1.0080    0.0     A       0.00000e+00 0.00000e+00

[ moleculetype ]
Water   2

[ atoms ]
1     HW    1      H-O-H    H1    1     0.417
2     OW    1      H-O-H    O     1    -0.834     15.9994
3     HW    1      H-O-H    H2    1     0.417

[ bonds ]
1     2   1       0.10    1000
2     3   1       0.10    1000

[ system ]
Water box

[ molecules ]
Water      2
"""
    out_dir = tmp_path_factory.mktemp("top")
    top_path = out_dir / "water_h-o-h.top"
    top_path.write_text(top_text)
    return str(top_path)


def test_get_top_system_symbols_water_h_o_h(water_top_file: str):
    symbols = get_top_system_symbols(water_top_file)
    assert symbols == ["H", "O", "H"] * 2


def test_get_top_system_bonds_base0_replicated_water(water_top_file: str):
    bonds = get_top_system_bonds_1direct(water_top_file, base=0)
    assert bonds == [(0, 1), (1, 2), (3, 4), (4, 5)]


def test_get_top_system_bonds_base1_replicated_water(water_top_file: str):
    bonds = get_top_system_bonds_1direct(water_top_file, base=1)
    assert bonds == [(1, 2), (2, 3), (4, 5), (5, 6)]


def test_get_top_system_bonds_rejects_invalid_base(water_top_file: str):
    with pytest.raises(ValueError, match=r"base must be 0 or 1"):
        get_top_system_bonds_1direct(water_top_file, base=2)
