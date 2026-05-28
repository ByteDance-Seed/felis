# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from felis.utils.element_guesser import guess_symbol


def test_guess_symbol_from_atomic_num_common_elements():
    """Test guessing symbol from atomic number for common elements."""
    assert guess_symbol(atomic_num=1) == "H"
    assert guess_symbol(atomic_num=6) == "C"
    assert guess_symbol(atomic_num=7) == "N"
    assert guess_symbol(atomic_num=8) == "O"
    assert guess_symbol(atomic_num=16) == "S"
    assert guess_symbol(atomic_num=17) == "Cl"


def test_guess_symbol_from_atomic_num_heavy_elements():
    """Test guessing symbol from atomic number for heavy elements."""
    assert guess_symbol(atomic_num=26) == "Fe"
    assert guess_symbol(atomic_num=30) == "Zn"
    assert guess_symbol(atomic_num=35) == "Br"
    assert guess_symbol(atomic_num=53) == "I"


def test_guess_symbol_from_atomic_mass_float():
    """Test guessing symbol from atomic mass as float."""
    assert guess_symbol(atomic_mass=1.01) == "H"
    assert guess_symbol(atomic_mass=12.01) == "C"
    assert guess_symbol(atomic_mass=14.01) == "N"
    assert guess_symbol(atomic_mass=16.00) == "O"
    assert guess_symbol(atomic_mass=32.07) == "S"


def test_guess_symbol_from_atomic_mass_string():
    """Test guessing symbol from atomic mass as string."""
    assert guess_symbol(atomic_mass="1.01") == "H"
    assert guess_symbol(atomic_mass="12.01") == "C"
    assert guess_symbol(atomic_mass="14.01") == "N"
    assert guess_symbol(atomic_mass="16.00") == "O"


def test_guess_symbol_from_atomic_mass_heavy_elements():
    """Test guessing symbol from atomic mass for heavy elements."""
    assert guess_symbol(atomic_mass=65.41) == "Zn"
    assert guess_symbol(atomic_mass=55.85) == "Fe"
    assert guess_symbol(atomic_mass=79.90) == "Br"
    assert guess_symbol(atomic_mass=126.90) == "I"


def test_guess_symbol_zinc_special_handling():
    """Test Zinc special handling for different atomic mass values."""
    # Zinc has special handling for multiple atomic mass values
    assert guess_symbol(atomic_mass=65.38) == "Zn"
    assert guess_symbol(atomic_mass=65.39) == "Zn"
    assert guess_symbol(atomic_mass=65.40) == "Zn"
    assert guess_symbol(atomic_mass=65.41) == "Zn"


def test_guess_symbol_both_inputs_matching():
    """Test guessing symbol when both atomic_num and atomic_mass match."""
    assert guess_symbol(atomic_num=6, atomic_mass=12.01) == "C"
    assert guess_symbol(atomic_num=8, atomic_mass=16.00) == "O"
    assert guess_symbol(atomic_num=30, atomic_mass=65.41) == "Zn"


def test_guess_symbol_both_inputs_mismatch():
    """Test that mismatching atomic_num and atomic_mass raises ValueError."""
    with pytest.raises(ValueError, match="Element symbols inferred from atomic_num and atomic_mass are not the same"):
        guess_symbol(atomic_num=6, atomic_mass=16.00)  # C vs O

    with pytest.raises(ValueError, match="Element symbols inferred from atomic_num and atomic_mass are not the same"):
        guess_symbol(atomic_num=1, atomic_mass=12.01)  # H vs C


def test_guess_symbol_both_none_raises():
    """Test that both inputs being None raises ValueError."""
    with pytest.raises(ValueError, match="atomic_num and atomic_mass are both None"):
        guess_symbol()


def test_guess_symbol_invalid_atomic_num():
    """Test that invalid atomic number raises KeyError."""
    # Valid atomic numbers start from 1
    with pytest.raises(KeyError):
        guess_symbol(atomic_num=999)  # Out of range

    with pytest.raises(KeyError):
        guess_symbol(atomic_num=-1)  # Negative


def test_guess_symbol_invalid_atomic_num_zero():
    """Test that atomic_num=0 is treated as falsy (neither valid nor raising KeyError)."""
    # atomic_num=0 is falsy in Python, so it's treated as if no atomic_num was provided
    # This results in the "both None" error when atomic_mass is also not provided
    with pytest.raises(ValueError, match="atomic_num and atomic_mass are both None"):
        guess_symbol(atomic_num=0)


def test_guess_symbol_invalid_atomic_mass():
    """Test that invalid atomic mass raises KeyError."""
    with pytest.raises(KeyError):
        guess_symbol(atomic_mass=999.99)

    with pytest.raises(KeyError):
        guess_symbol(atomic_mass="invalid")


def test_guess_symbol_invalid_atomic_mass_type():
    """Test that invalid atomic mass type raises TypeError."""
    with pytest.raises(TypeError, match="atomic_mass must be float or str"):
        guess_symbol(atomic_mass=123)  # int is not allowed


def test_guess_symbol_atomic_mass_none_treated_as_falsy():
    """Test that atomic_mass=None is treated as falsy (not a TypeError)."""
    # atomic_mass=None is the default and is treated as falsy
    # This results in the "both None" error when atomic_num is also not provided
    with pytest.raises(ValueError, match="atomic_num and atomic_mass are both None"):
        guess_symbol(atomic_mass=None)  # Same as default behavior


def test_periodic_table_weight_key_collision_raises_value_error(monkeypatch):
    """Cover the collision branch in the periodic-table map builder."""
    import felis.utils.element_guesser as element_guesser

    class _FakePeriodicTable:

        def GetElementSymbol(self, atomic_num: int) -> str:  # noqa: N802
            # Provide distinct symbols so the error message is informative.
            return {1: "X", 2: "Y"}.get(atomic_num, "Z")

        def GetAtomicWeight(self, atomic_num: int) -> float:  # noqa: N802
            # Force a collision: atomic numbers 1 and 2 have identical weights.
            return 10.0

    element_guesser._get_periodic_table_maps.cache_clear()
    monkeypatch.setattr(element_guesser.Chem, "GetPeriodicTable", lambda: _FakePeriodicTable())

    with pytest.raises(ValueError, match="not unique"):
        element_guesser._get_periodic_table_maps()

    element_guesser._get_periodic_table_maps.cache_clear()
