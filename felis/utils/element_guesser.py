# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Guess chemical element symbols from atomic number or atomic mass.

This module provides a small helper for resolving an element symbol (e.g.,
``"C"``) from either:

- an atomic number (Z), or
- an atomic mass value formatted to 2 decimal places.

The mapping is constructed from RDKit's periodic table and cached on first use.

Notes:
    - Atomic masses are matched on a 2-decimal string representation. When a
      float is provided, it is formatted with ``"{mass:.2f}"``.
    - To tolerate common rounding differences, the lookup table also accepts
      values within ``+/-0.009`` of the RDKit atomic weight when rounded to 2
      decimals.
    - Zinc (Zn) is patched to accept several commonly used weights, including
      OpenMM's ``65.409`` representation rounded to 2 decimals.
"""

import logging
from functools import lru_cache
from typing import Optional, Union

import rdkit.Chem as Chem

logger = logging.getLogger(__name__)

_PTABLE_LEN = 83  # H -- Bi | Po, At, Rn, ...


@lru_cache(maxsize=1)
def _get_periodic_table_maps() -> tuple[dict[int, tuple[int, str, str]], dict[str, tuple[int, str, str]]]:
    """Build and cache periodic-table lookup maps.

    Returns:
        tuple[dict[int, tuple[int, str, str]], dict[str, tuple[int, str, str]]]:
            - Map from atomic number to ``(atomic_num, symbol, weight_str)``.
            - Map from 2-decimal weight string to ``(atomic_num, symbol, weight_str)``.

    Raises:
        ValueError: If a 2-decimal weight key is not unique across elements.
    """
    ptable = Chem.GetPeriodicTable()
    atomnum_dict: dict[int, tuple[int, str, str]] = {}
    weight_dict: dict[str, tuple[int, str, str]] = {}

    for i in range(_PTABLE_LEN):
        atomic_num = i + 1
        symbol = ptable.GetElementSymbol(atomic_num)
        weight = ptable.GetAtomicWeight(atomic_num)
        weight_str = f"{weight:.2f}"

        atomnum_dict[atomic_num] = (atomic_num, symbol, weight_str)

        # Include a small tolerance around RDKit's atomic weight to tolerate
        # rounding differences between toolchains.
        for w_unique in {f"{weight - 0.009:.2f}", weight_str, f"{weight + 0.009:.2f}"}:
            if w_unique in weight_dict:
                existing = weight_dict[w_unique]
                raise ValueError(f"2-decimal atomic mass key {w_unique!r} is not unique: {existing[1]} vs {symbol}")
            weight_dict[w_unique] = (atomic_num, symbol, weight_str)

        if atomic_num == 30:
            # Zinc: 65.409(4), 2001; 65.39(2), 1983, RDKit; 65.38(1), 1971; https://www.ciaaw.org/pubs/TSAW-2001.pdf
            # In 2007, the recommended value was reverted to 65.38(2). https://ciaaw.org/zinc.htm
            # This patch is added for values commonly used in toolchains
            # (e.g. OpenMM's 65.409 rounded to 2 decimals).
            for w_unique_zn in ("65.38", "65.39", "65.40", "65.41"):
                weight_dict.setdefault(w_unique_zn, (atomic_num, symbol, weight_str))

    return atomnum_dict, weight_dict


def guess_symbol(atomic_num: Optional[int] = None, atomic_mass: Optional[Union[float, str]] = None) -> str:
    """Infer an element symbol from atomic number and/or atomic mass.

    If both `atomic_num` and `atomic_mass` are provided, they must resolve to
    the same element symbol or a `ValueError` is raised.

    Args:
        atomic_num (Optional[int]): Atomic number (Z). If provided, it must be
            a valid key in the precomputed periodic table map.
        atomic_mass (Optional[Union[float, str]]): Atomic mass. If a float is
            provided, it is formatted to 2 decimal places before lookup. If a
            string is provided, it must already be in the same 2-decimal form.

    Returns:
        str: The inferred element symbol (e.g., ``"H"``, ``"Cl"``).

    Raises:
        KeyError: If `atomic_num` or the 2-decimal mass key cannot be found in
            the internal lookup tables.
        TypeError: If `atomic_mass` is not a `float` or `str`.
        ValueError: If both inputs are `None`, or if both inputs are provided
            but resolve to different element symbols.
    """

    atomnum_dict, weight_dict = _get_periodic_table_maps()

    sym1, sym2 = None, None
    if atomic_num:
        _num, sym1, _wstr = atomnum_dict[atomic_num]
    if atomic_mass is not None:
        if isinstance(atomic_mass, str):
            mass_str = atomic_mass
        elif isinstance(atomic_mass, float):
            mass_str = f"{atomic_mass:.2f}"
        else:
            raise TypeError(f"atomic_mass must be float or str, not {type(atomic_mass)}")
        _num, sym2, _wstr = weight_dict[mass_str]

    if sym1 and sym2 is None:
        return sym1
    elif sym1 is None and sym2:
        return sym2
    elif sym1 and sym2:
        if sym1 == sym2:
            return sym1
        else:
            raise ValueError(
                f"Element symbols inferred from atomic_num and atomic_mass are not the same: {sym1} {sym2} {atomic_num} {atomic_mass}"
            )
    else:
        raise ValueError("atomic_num and atomic_mass are both None")
