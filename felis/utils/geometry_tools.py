# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Geometric calculation utilities for molecular structures.

This module provides functions to calculate geometric measurements
(bond lengths, angles, and dihedral angles) for molecular structures
using atomic positions.
"""
import logging
import math

from MDAnalysis.lib.distances import calc_angles, calc_bonds, calc_dihedrals

logger = logging.getLogger(__name__)


def get_geometries(list_of_geom_atoms: list[list[int]], pos, mda_dim):
    """Calculate geometric measurements for a list of atom groups.

    This function calculates bond lengths, bond angles, or dihedral angles
    depending on the number of atoms provided in each geometry group:
    - 2 atoms: bond length (using calc_bonds)
    - 3 atoms: bond angle (using calc_angles, result converted to degrees)
    - 4 atoms: dihedral angle (using calc_dihedrals, result converted to degrees)

    Args:
        list_of_geom_atoms: A list of atom index groups. Each inner list
            contains 2, 3, or 4 integer atom indices.
        pos: Position array containing atomic coordinates. Can be indexed
            by atom index to retrieve the coordinate array for that atom.
        mda_dim: MDAnalysis dimension parameter for periodic boundary
            conditions. Passed directly to calc_bonds.

    Returns:
        dict: A dictionary mapping atom index tuples to calculated geometry
        values:
        - For bonds: {(p1, p2): distance_value}
        - For angles: {(p1, p2, p3): angle_in_degrees}
        - For dihedrals: {(p1, p2, p3, p4): dihedral_in_degrees}

    Raises:
        IndexError: If an atom group has fewer than 2 atoms (when accessing
            geom_atoms[1]) or if an atom index is out of bounds for the pos array.
    """
    radian_to_degree = 180. / math.pi

    d = dict()
    for geom_atoms in list_of_geom_atoms:
        p1, p2 = geom_atoms[0], geom_atoms[1]
        ap1 = pos[p1]
        ap2 = pos[p2]
        if len(geom_atoms) >= 3:
            p3 = geom_atoms[2]
            ap3 = pos[p3]
        else:
            ap3 = None
        if len(geom_atoms) >= 4:
            p4 = geom_atoms[3]
            ap4 = pos[p4]
        else:
            ap4 = None

        if ap3 is None:
            ans = calc_bonds(ap1, ap2, mda_dim)
            d[(p1, p2)] = ans
        elif ap4 is None:
            ans = calc_angles(ap1, ap2, ap3) * radian_to_degree
            d[(p1, p2, p3)] = ans
        else:
            ans = calc_dihedrals(ap1, ap2, ap3, ap4) * radian_to_degree
            d[(p1, p2, p3, p4)] = ans
    return d
