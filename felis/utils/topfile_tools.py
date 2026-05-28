# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Utilities for extracting atom and bond data from GROMACS topology files.

This module provides small convenience wrappers around the `bytemol` topology
parser to extract information from a GROMACS ``.top`` file.

Currently supported helpers:

- Build a per-atom element-symbol list for the full system.
- Build a per-bond index list for the full system.

Notes
-----
- The heavy lifting is delegated to `bytemol.toolkit.gmxtool.topparse`.
- Element symbols are inferred via :func:`felis.utils.element_guesser.guess_symbol`.
"""

import logging

from bytemol.toolkit.gmxtool.topparse import TopoAtomTypes, TopoFullSystem
from felis.utils.element_guesser import guess_symbol

logger = logging.getLogger(__name__)


def get_top_system_symbols(top_file: str) -> list[str]:
    """Return per-atom element symbols for the full system described by a ``.top``.

    Parameters
    ----------
    top_file
        Path to a GROMACS topology file (``.top``).

    Returns
    -------
    list[str]
        A flat list of element symbols (e.g., ``["C", "H", "H", ...]``) in the
        same order as atoms in the topology-expanded system.

    Implementation details
    ----------------------
    - Reads the topology via :meth:`bytemol.toolkit.gmxtool.topparse.TopoFullSystem.from_file`.
    - For each atom, tries to use the atomic number (``at_num``) stored in the
      atom-type record; otherwise falls back to mass-based guessing.
    - Replicates per-molecule symbols by the molecule count ``m.nr``.

    Caveats
    -------
    - Element inference is heuristic when atomic numbers are missing.
    - The output is only as reliable as the topology's atom-type data.
    """
    sys_symbols = []
    topo = TopoFullSystem.from_file(top_file)
    tatp = TopoAtomTypes(topo.uuid)
    for im, m in enumerate(topo.molecules):
        tidx = topo.mol_to_topo_index[im]
        num_mols = m.nr
        itopo = topo.mol_topos[tidx]
        im_symbols = []

        for ia in itopo.atoms:
            iatomtype = ia.atype
            atomtype_record = tatp.atomtypes[tatp.type_to_index[iatomtype]]

            atomic_number = None
            if hasattr(atomtype_record, "at_num"):
                atomic_number = atomtype_record.at_num

            atomic_mass = None
            if hasattr(ia, "mass"):
                atomic_mass = ia.mass
            else:
                atomic_mass = atomtype_record.mass

            im_symbols.append(guess_symbol(atomic_number, atomic_mass))

        sys_symbols.extend(im_symbols * num_mols)
    return sys_symbols


def get_top_system_bonds_1direct(top_file: str, base: int = 0) -> list[tuple[int, int]]:
    """Return system bonds (undirected, unique per pair) as atom index tuples.

    The returned bond list is expanded to the *full system* by replicating each
    molecule's internal bonds by the molecule multiplicity ``m.nr``.

    Parameters
    ----------
    top_file
        Path to a GROMACS topology file (``.top``).
    base
        Output index base: ``0`` for 0-based indices, ``1`` for 1-based indices.

    Returns
    -------
    list[tuple[int, int]]
        A flat list of bonds as ``(i, j)`` atom indices. For each bond, the
        indices are ordered so that ``i <= j``.

    Notes
    -----
    - Input bond indices from `bytemol` are assumed to be 1-based; an internal
      offset is applied before adding the requested ``base``.
    - Only intra-molecule bonds described in the topology are included.
    """
    if base not in (0, 1):
        raise ValueError(f"base must be 0 or 1, not {base}")
    offset = -1
    atom_count = 0

    sys_bonds = []
    topo = TopoFullSystem.from_file(top_file)
    for im, m in enumerate(topo.molecules):
        tidx = topo.mol_to_topo_index[im]
        num_mols = m.nr
        itopo = topo.mol_topos[tidx]
        im_bonds = []
        for ib in itopo.bonds:
            ai = min(ib.ai, ib.aj) + offset
            aj = max(ib.ai, ib.aj) + offset
            im_bonds.append((ai, aj))
        for _ in range(num_mols):
            sys_bonds.extend([(i + base + atom_count, j + base + atom_count) for i, j in im_bonds])
            atom_count += itopo.natoms
    return sys_bonds
