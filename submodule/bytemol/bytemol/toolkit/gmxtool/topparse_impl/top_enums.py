# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, IntEnum


class NonbondedFunctionEnum(IntEnum):
    """
    nbfunc
    """
    LENNARD_JONES = 1
    # BUCKINGHAM = 2


class LJCombinationRuleEnum(IntEnum):
    """
    LJ = C6/r**6 - C12/r**12
       = 4 eps [(sig/r)**6 - (sig/r)**12]
    C6, C12 = 4*eps*sig**6, 4*eps*sig**12
    sig, eps = (C12/C6)**(1/6), C6**2/(4*C12)
    comb-rule
        vdw-def         comb-rule
    1   V,W = C6,C12    C6,C12 = geometric,geometric
    2   V,W = sig,eps   sig,eps = arithmetic,geometric
    3   V,W = sig,eps   C6,C12 = geometric,geometric

    Buckingham = A exp(-Br) - C/r**6
    comb-rule
    A,B,C = geometric,harmonic,geometric
    """
    # C6_C12_GEO_GEO = 1
    SIGMA_EPSILON = 2
    # SIGMA_EPSILON_C6_C12_GEO_GEO = 3


class BondTypeEnum(IntEnum):
    BOND = 1
    # G96 = 2
    # MORSE = 3
    # CUBIC = 4
    # CONNECTION = 5
    HARMONIC_POTENTIAL = 6
    # FENE = 7
    # TABULATED_EXCL = 8
    # TABULATED_WO_EXCL = 9
    # RESTRAINT = 10
    SOFTBOND = 11  # Soft-bond implementation according to JCTC 2017, 13, 42-54


class PairTypeEnum(IntEnum):
    EXTRA_LJ = 1
    EXTRA_COULOMB_LJ = 2


class PairNbTypeEnum(IntEnum):
    COULOMB_LJ = 1
    SOFTCORE_COULOMB_LJ = 2


class DihedralTypeEnum(IntEnum):
    PROPER = 1
    IMPROPER = 2
    RYCKAERT_BELLEMANS = 3
    PERIODIC_IMPROPER = 4
    # FOURIER = 5
    # TABULATED = 8
    MULTIPLE_PROPER = 9
    # RESTRICTED = 10
    # COMBINED_BENDING_TORSION = 11


class AngleTypeEnum(IntEnum):
    ANGLE = 1
    # G96 = 2
    # CROSS_BOND_BOND = 3
    # CROSS_BOND_ANGLE = 4
    # UREY_BRADLEY = 5
    # QUARTIC = 6
    # TABULATED = 8
    # LINEAR = 9
    # RESTRICTED_BENDING = 10


class VirtualSite2Enum(IntEnum):
    _2 = 1
    _2FD = 2


class VirtualSite3Enum(IntEnum):
    _3 = 1
    _3FD = 2
    _3FAD = 3
    _3OUT = 4


class VirtualSite4Enum(IntEnum):
    _4FDN = 2


class TopoRoundModeEnum(Enum):
    ON_READ = "r"
    ON_WRITE = "w"
    ON_READ_AND_WRITE = "rw"
    NEVER = ""
