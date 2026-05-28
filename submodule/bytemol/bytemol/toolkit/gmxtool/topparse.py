# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import decimal
import itertools
import logging
import os
from copy import deepcopy
from typing import Iterable, Tuple, Union
from uuid import uuid4

import numpy as np

from bytemol.toolkit.gmxtool.topparse_impl.top_enums import DihedralTypeEnum, PairTypeEnum, TopoRoundModeEnum
from bytemol.toolkit.gmxtool.topparse_impl.top_exceptions import TopoDuplicateNameMoleculeTypeException
from bytemol.toolkit.gmxtool.topparse_impl.top_records import (
    RecordAngle,
    RecordAtom,
    RecordAtomType,
    RecordBond,
    RecordDefaults,
    RecordDihedral,
    RecordDihedralRestraints,
    RecordExclusion,
    RecordMolecule,
    RecordMoleculeType,
    RecordPair,
    RecordPairNb,
    RecordPositionRestraints,
    Records,
    RecordSection,
    RecordSettle,
    RecordSystem,
    RecordText,
    RecordVirtualSite1,
    RecordVirtualSite2,
    RecordVirtualSite3,
    RecordVirtualSite4,
    TopoAtomTypes,
    TopoDefaults,
)

logger = logging.getLogger(__name__)

################
# molecule itp #
################


class TopoMolecule:

    def __init__(self,
                 system_uuid,
                 *,
                 round_on: Union[TopoRoundModeEnum, str] = TopoRoundModeEnum.ON_WRITE,
                 allow_round_diff: float = None,
                 override_atom_comment_with_qtot: bool = True):
        self.moleculetype: RecordMoleculeType = None
        self.atoms: list[RecordAtom] = []
        self.bonds: list[RecordBond] = []
        self.angles: list[RecordAngle] = []
        self.dihedrals: list[RecordDihedral] = []
        self.settles: list[RecordSettle] = []
        self.exclusions: list[RecordExclusion] = []
        self.pairs: list[RecordPair] = []
        self.pairs_nb: list[RecordPairNb] = []
        self.virtual_sites1: list[RecordVirtualSite1] = []
        self.virtual_sites2: list[RecordVirtualSite2] = []
        self.virtual_sites3: list[RecordVirtualSite3] = []
        self.virtual_sites4: list[RecordVirtualSite4] = []
        self.dihedral_restraints: list[RecordDihedralRestraints] = []
        self.position_restraints: list[RecordPositionRestraints] = []

        self.atom_to_type_index: list[int] = []
        self.system_uuid = system_uuid

        self.round_on: str = TopoRoundModeEnum(round_on).value
        self.allow_round_diff: float = allow_round_diff
        self.override_atom_comment_with_qtot: bool = override_atom_comment_with_qtot

    @property
    def molecule_type(self):
        return self.moleculetype

    @property
    def name(self) -> str:
        return self.moleculetype.name

    @property
    def nrexcl(self) -> int:
        return self.moleculetype.nrexcl

    @property
    def natoms(self) -> int:
        """
        Number of physical atoms and virtual sites.
        """
        return len(self.atoms)

    @property
    def nrealatoms(self) -> int:
        """
        Number of physical atoms.
        """
        ta = TopoAtomTypes(self.system_uuid)
        return len([idx for idx in self.atom_to_type_index if ta.atomtypes[idx].particle_type == "A"])

    @property
    def nvsites(self) -> int:
        """
        Number of virtual sites.
        """
        ta = TopoAtomTypes(self.system_uuid)
        return len([idx for idx in self.atom_to_type_index if ta.atomtypes[idx].particle_type == "V"])

    def __str__(self) -> str:
        if "w" in self.round_on:
            if self.allow_round_diff is None or self.allow_round_diff < 0:
                raise RuntimeError(f"Incorrect allow_round_diff (current: {self.allow_round_diff}).")
            elif self.allow_round_diff > 0:
                self.update_charge(partial_charges=None, symm=None, to_round=True)
            else:
                pass
        lines = []

        if self.moleculetype is not None:
            lines.append("")
            section = "moleculetype"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.moleculetype.annotate())
            lines.append(str(self.moleculetype))

        if self.natoms:
            lines.append("")
            section = "atoms"
            lines.append(str(RecordSection(section=section)))
            if self.override_atom_comment_with_qtot:
                qtot = 0.0
                a = self.atoms[0]
                lines.append(a.annotate() + " ; qtot" if hasattr(a, "charge") else "")
                for a in self.atoms:
                    if hasattr(a, "charge"):
                        qtot += a.charge
                        suffix = f" ; qtot {qtot:8.3f}"
                    else:
                        suffix = ""
                    lines.append(str(a) + suffix)
            else:
                lines.append(self.atoms[0].annotate())
                for a in self.atoms:
                    if a.comment is not None and a.comment.strip() != ";":
                        lines.append(str(a) + a.comment)
                    else:
                        lines.append(str(a))

        if len(self.bonds):
            lines.append("")
            section = "bonds"
            lines.append(str(RecordSection(section=section)))
            lines.append(max(self.bonds, key=lambda bond: bond.funct.value).annotate())
            for a in self.bonds:
                lines.append(str(a))

        if len(self.angles):
            lines.append("")
            section = "angles"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.angles[0].annotate())
            for a in self.angles:
                lines.append(str(a))

        if len(self.dihedrals):
            propers = []
            improps = []
            for a in self.dihedrals:
                if a.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.MULTIPLE_PROPER):
                    propers.append(a)
                elif a.funct in (DihedralTypeEnum.PERIODIC_IMPROPER, DihedralTypeEnum.IMPROPER):
                    improps.append(a)
                else:
                    a.raise_not_impl()

            if len(propers):
                lines.append("")
                section = "dihedrals"
                comment = " ; propers for gromacs 4.5 or higher, use funct 9"
                verbose = True
                lines.append(str(RecordSection(section=section, comment=comment, verbose=verbose)))
                lines.append(propers[0].annotate())
                for a in propers:
                    lines.append(str(a))

            if len(improps):
                lines.append("")
                section = "dihedrals"
                comment = " ; impropers"
                verbose = True
                lines.append(str(RecordSection(section=section, comment=comment, verbose=verbose)))
                lines.append(improps[0].annotate())
                for a in improps:
                    lines.append(str(a))

        if len(self.settles):
            lines.append("")
            section = "settles"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.settles[0].annotate())
            for a in self.settles:
                lines.append(str(a))

        if len(self.exclusions):
            lines.append("")
            section = "exclusions"
            lines.append(str(RecordSection(section=section)))
            for a in self.exclusions:
                lines.append(str(a))

        if len(self.pairs):
            lines.append("")
            section = "pairs"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.pairs[0].annotate())
            for a in self.pairs:
                lines.append(str(a))

        if len(self.pairs_nb):
            lines.append("")
            section = "pairs_nb"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.pairs_nb[0].annotate())
            for a in self.pairs_nb:
                lines.append(str(a))

        if len(self.virtual_sites1):
            lines.append("")
            section = "virtual_sites1"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.virtual_sites1[0].annotate())
            for a in self.virtual_sites1:
                lines.append(str(a))

        if len(self.virtual_sites2):
            lines.append("")
            section = "virtual_sites2"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.virtual_sites2[0].annotate())
            for a in self.virtual_sites2:
                lines.append(str(a))

        if len(self.virtual_sites3):
            lines.append("")
            section = "virtual_sites3"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.virtual_sites3[0].annotate())
            for a in self.virtual_sites3:
                lines.append(str(a))

        if len(self.virtual_sites4):
            lines.append("")
            section = "virtual_sites4"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.virtual_sites4[0].annotate())
            for a in self.virtual_sites4:
                lines.append(str(a))

        if len(self.dihedral_restraints):
            lines.append("")
            section = "dihedral_restraints"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.dihedral_restraints[0].annotate())
            for a in self.dihedral_restraints:
                lines.append(str(a))

        if len(self.position_restraints):
            lines.append("")
            section = "position_restraints"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.position_restraints[0].annotate())
            for a in self.position_restraints:
                lines.append(str(a))

        return "\n".join(lines)

    def _sort_bonds(self):
        for a in self.bonds:
            if a.ai > a.aj:
                a.ai, a.aj = a.aj, a.ai
        self.bonds.sort(key=RecordBond.cmp_key)

    def _sort_angles(self):
        for a in self.angles:
            if a.ai > a.ak:
                a.ai, a.ak = a.ak, a.ai  # aj does not change
        self.angles.sort(key=RecordAngle.cmp_key)

    def _sort_dihedrals(self):
        for a in self.dihedrals:
            if a.funct in (DihedralTypeEnum.PERIODIC_IMPROPER, DihedralTypeEnum.IMPROPER):
                continue  # do not modify improper
            if a.ai > a.al:
                a.ai, a.aj, a.ak, a.al = a.al, a.ak, a.aj, a.ai
        self.dihedrals.sort(key=RecordDihedral.cmp_key)

    def _sort_exclusions(self):
        self.exclusions.sort(key=RecordExclusion.cmp_key)

    def _sort_pairs(self):
        for pair in self.pairs:
            if pair.ai > pair.aj:
                pair.ai, pair.aj = pair.aj, pair.ai
                if pair.funct == PairTypeEnum.EXTRA_COULOMB_LJ:
                    pair.qi, pair.qj = pair.qj, pair.qi
        self.pairs.sort(key=RecordPair.cmp_key)

    def _sort_pairs_nb(self):
        for pair_nb in self.pairs_nb:
            if pair_nb.ai > pair_nb.aj:
                pair_nb.ai, pair_nb.aj = pair_nb.aj, pair_nb.ai
                pair_nb.qi, pair_nb.qj = pair_nb.qj, pair_nb.qi
        self.pairs_nb.sort(key=RecordPairNb.cmp_key)

    def _check_pair_exclusion(self):
        excl_set = set()
        for excl in self.exclusions:
            for aj in excl.aj_list:
                excl_set.add((excl.ai, aj))
        for pair in self.pairs:
            ai, aj = pair.ai, pair.aj
            if (ai, aj) in excl_set or (aj, ai) in excl_set:
                logger.warning(f"Excluded pair re-added by [ pairs ]: ({ai:>10},{aj:>10})")

    def _check_pair_nb_exclusion(self):
        """
        Any pair in [ pairs_nb ] must also be present in [ exclusions ]
        """
        excl_set = set((excl.ai, aj) for excl in self.exclusions for aj in excl.aj_list)
        for pair_nb in self.pairs_nb:
            ai, aj = pair_nb.ai, pair_nb.aj
            if (ai, aj) not in excl_set and (aj, ai) not in excl_set:
                logger.warning(f"A pair in [ pairs_nb ] is missing in [ exclusions ]: ({ai:>10},{aj:>10})")

    def add_record_atom(self, record: RecordAtom, atomtype: RecordAtomType = None) -> int:
        ta = TopoAtomTypes(self.system_uuid)
        if record.atype in ta.type_to_index.keys():
            self.atoms.append(record)
            self.atom_to_type_index.append(ta.type_to_index[record.atype])
        else:
            assert atomtype is not None and record.atype == atomtype.name
            logger.info(f"Adding new type {record.atype} to the [ atomtypes ] section.")
            idx = ta.add_record(atomtype)
            self.atoms.append(record)
            self.atom_to_type_index.append(idx)
        return self.natoms - 1

    def add_record_vsite(self, record: RecordAtom, vrecord) -> int:
        self.add_record_atom(record=record)
        if isinstance(vrecord, RecordVirtualSite1):
            self.virtual_sites1.append(vrecord)
        elif isinstance(vrecord, RecordVirtualSite2):
            self.virtual_sites2.append(vrecord)
        elif isinstance(vrecord, RecordVirtualSite3):
            self.virtual_sites3.append(vrecord)
        elif isinstance(vrecord, RecordVirtualSite4):
            self.virtual_sites4.append(vrecord)
        else:
            assert False
        return self.natoms - 1

    def determine_vsite_exclusions_and_pairs(self):
        # this code only works if gen-pairs is "yes"
        assert TopoDefaults(self.system_uuid).gen_pairs == "yes"

        nb1x = self.get_nb1x()
        b11, b12, b13, b14 = nb1x["11"], nb1x["12"], nb1x["13"], nb1x["14"]
        ptypes = self.get_ptypes()

        # exclusions
        ai_idx = dict((a.ai, idx) for idx, a in enumerate(self.exclusions))
        for a0, pt in enumerate(ptypes):
            if pt == "V":
                ai = a0 + 1
                a1_list = b11[a0] + b12[a0] + b13[a0] + b14[a0]
                a1_list.sort()

                if ai not in ai_idx.keys():
                    idx = len(self.exclusions)
                    self.exclusions.append(RecordExclusion(ai=ai))
                else:
                    idx = ai_idx[ai]

                for a1 in a1_list:
                    aj = a1 + 1
                    if aj != ai and aj not in self.exclusions[idx].aj_list:
                        self.exclusions[idx].aj_list.append(aj)

                self.exclusions[idx].aj_list.sort()

        # pairs (1-4)
        current_pairs = set()
        for a in self.pairs:
            current_pairs.add((a.ai, a.aj))
            current_pairs.add((a.aj, a.ai))

        for a0, a1_list in b14.items():
            ai = a0 + 1
            for a1 in a1_list:
                aj = a1 + 1
                if (ai, aj) not in current_pairs:
                    current_pairs.add((ai, aj))
                    current_pairs.add((aj, ai))
                    self.pairs.append(RecordPair(ai=ai, aj=aj, funct=PairTypeEnum.EXTRA_LJ))

        self._sort_pairs()

    def set_atomtype(self, atomidx: int, atype_or_typeidx: Tuple[str, int]):
        ta = TopoAtomTypes(self.system_uuid)
        if isinstance(atype_or_typeidx, int):
            self.atoms[atomidx].atype = ta.atomtypes[atype_or_typeidx].name
            self.atom_to_type_index[atomidx] = atype_or_typeidx
        elif isinstance(atype_or_typeidx, str):
            self.atoms[atomidx].atype = atype_or_typeidx
            self.atom_to_type_index[atomidx] = ta.type_to_index[atype_or_typeidx]
        else:
            assert False

    @staticmethod
    def average_by_symmetry(data, symm: dict[int, list[int]] = None):
        arr = np.copy(data)
        if symm is not None:
            for _, atoms in symm.items():
                where = np.full(len(data), False)
                where[atoms] = True
                mean = np.mean(data, where=where)
                for a in atoms:
                    arr[a] = mean
        return arr

    @staticmethod
    def round_list_sum_to_int(lst: list[float], symm: dict[int, list[int]] = None, maxdiff: float = 0.01):
        lst = TopoMolecule.average_by_symmetry(data=lst, symm=symm)

        # convert the list of floats to a list of Decimals rounded to 5 decimal places
        decimal_context = decimal.Context(prec=28, rounding=decimal.ROUND_HALF_DOWN)
        decimal_lst = [decimal_context.create_decimal(str(x)).quantize(decimal.Decimal("1.00000")) for x in lst]

        # apply the adjustment to the last number to ensure the sum is an integer
        decimal_sum = sum(decimal_lst)
        gap = (round(decimal_sum) - decimal_sum)
        if abs(gap) > abs(maxdiff):
            msg = f"Not Rounded when List Sum to Integer abs(gap) > abs(maxdiff) {abs(gap):6.3f} >{abs(maxdiff):6.3f}."
            if abs(maxdiff) > 0:
                logger.warning(msg)
            return [x for x in lst]

        if decimal_sum % 1 != 0:
            decimal_lst[-1] += gap

        float_lst = [float(x) for x in decimal_lst]
        before = round(sum(lst))
        after = round(sum(float_lst))
        assert before == after, f"Rounding error before {before} after {after}"
        return float_lst

    def get_ivatoms(self) -> dict[int, list[int]]:
        """
        0-based iatom and associated vatoms.
        """
        ai_avs = {}
        ptypes = self.get_ptypes()
        for i, pt in enumerate(ptypes):
            if pt == "A":
                ai_avs[i] = {i}
        for vs in (self.virtual_sites1, self.virtual_sites2, self.virtual_sites3, self.virtual_sites4):
            for a in vs:
                av, ai = a.av - 1, a.ai - 1
                ai_avs[ai].add(av)
        return dict((ai, sorted(list(avs))) for ai, avs in ai_avs.items())

    def get_doppelgangers(self, symm: dict[int, list[int]]) -> dict[int, int]:
        """
        doppelgangers[jatom] = iatom, or jatom if unique
        i/jatom are 0-based.
        """
        doppelgangers = {i: i for i in range(self.natoms)}
        for iatom, eq_list in symm.items():
            for jatom in eq_list:
                doppelgangers[jatom] = iatom
        return doppelgangers

    def get_nb1x(self) -> dict[str, dict[int, list[int]]]:
        """
        0-based 1-x connected neighboring particles.
        """
        nb1x = {}

        natoms = self.natoms
        nrexcl = self.nrexcl
        boo = {i: {i} for i in range(natoms)}

        b11 = {i: [i] for i in range(natoms)}
        if nrexcl >= 0:
            ai_avs = self.get_ivatoms()
            for _ai, avs in ai_avs.items():
                if len(avs) > 1:
                    avs2 = list(avs)
                    for ai, aj in itertools.combinations(avs2, 2):
                        if ai not in boo[aj]:
                            b11[aj].append(ai)
                            boo[aj].add(ai)
                        if aj not in boo[ai]:
                            b11[ai].append(aj)
                            boo[ai].add(aj)

            for _k, v in b11.items():
                v.sort()
        nb1x["11"] = b11

        b12 = {i: [] for i in range(natoms)}
        if nrexcl >= 1:
            for a in self.bonds:
                a0, a1 = a.ai - 1, a.aj - 1
                for p in itertools.product(b11[a0], b11[a1]):
                    ai, aj = p[0], p[1]
                    if aj not in boo[ai]:
                        b12[ai].append(aj)
                        boo[ai].add(aj)
                    if ai not in boo[aj]:
                        b12[aj].append(ai)
                        boo[aj].add(ai)
            for _k, v in b12.items():
                v.sort()
        nb1x["12"] = b12

        b13 = {i: [] for i in range(natoms)}
        if nrexcl >= 2:
            for k, v in b12.items():
                for w in v:
                    for x in b12[w]:
                        if x not in boo[k]:
                            b13[k].append(x)
                            boo[k].add(x)
                        if k not in boo[x]:
                            b13[x].append(k)
                            boo[x].add(k)
            for _k, v in b13.items():
                v.sort()
        nb1x["13"] = b13

        b14 = {i: [] for i in range(natoms)}
        if nrexcl >= 3:
            for k, v in b13.items():
                for w in v:
                    for x in b12[w]:
                        if x not in boo[k]:
                            b14[k].append(x)
                            boo[k].add(x)
                        if k not in boo[x]:
                            b14[x].append(k)
                            boo[x].add(k)
            for _k, v in b14.items():
                v.sort()
        nb1x["14"] = b14

        return nb1x

    def get_atypes(self) -> list[str]:
        return [a.atype for a in self.atoms]

    def get_ptypes(self) -> list[str]:
        ta = TopoAtomTypes(self.system_uuid)
        return [ta.atomtypes[type_idx].particle_type for type_idx in self.atom_to_type_index]

    def get_charges(self) -> list[float]:
        ta = TopoAtomTypes(self.system_uuid)
        return [
            a.charge if hasattr(a, "charge") else ta.atomtypes[type_idx].charge
            for a, type_idx in zip(self.atoms, self.atom_to_type_index)
        ]

    def get_masses(self) -> list[float]:
        ta = TopoAtomTypes(self.system_uuid)
        return [
            a.mass if hasattr(a, "mass") else ta.atomtypes[type_idx].mass
            for a, type_idx in zip(self.atoms, self.atom_to_type_index)
        ]

    def get_vw(self) -> list[Tuple[float, float]]:
        ta = TopoAtomTypes(self.system_uuid)
        return [(ta.atomtypes[type_idx].V, ta.atomtypes[type_idx].W) for type_idx in self.atom_to_type_index]

    def update_charge(self,
                      partial_charges: Iterable[float] = None,
                      symm: dict[int, list[int]] = None,
                      to_round: bool = False):
        if to_round:
            assert self.allow_round_diff is not None
            assert self.allow_round_diff >= 0.
            pchgs = self.round_list_sum_to_int(self.get_charges() if partial_charges is None else partial_charges, symm,
                                               self.allow_round_diff)
        elif partial_charges is None:
            return
        else:
            pchgs = partial_charges

        assert self.natoms == len(pchgs)
        for pc, atom in zip(pchgs, self.atoms):
            if hasattr(atom, "charge"):
                atom.charge = pc
            else:
                setattr(atom, "charge", pc)

    @classmethod
    def from_records(cls,
                     system_uuid,
                     records: list[RecordText],
                     sort_idx: bool,
                     *,
                     round_on: Union[TopoRoundModeEnum, str],
                     allow_round_diff: float,
                     override_atom_comment_with_qtot: bool = True):
        """
        Exit on finding the second [ moleculetype ].
        """

        tm = cls(system_uuid,
                 round_on=round_on,
                 allow_round_diff=allow_round_diff,
                 override_atom_comment_with_qtot=override_atom_comment_with_qtot)

        for r in records:
            # NOTE: skip [ intermolecular_interactions ] section when parsing top file to avoid counting the bond, angles and dihedrals
            # included in this section as terms of a particular molecule during parsing.
            if isinstance(r, RecordMoleculeType) or (isinstance(r, RecordSection) and
                                                     r.section == "intermolecular_interactions"):
                if tm.moleculetype is None:
                    tm.moleculetype = deepcopy(r)
                else:
                    break
            elif isinstance(r, RecordAtom):
                tm.add_record_atom(r)
            elif isinstance(r, RecordBond):
                if r.verbose is None:
                    r.verbose = True
                tm.bonds.append(r)
            elif isinstance(r, RecordAngle):
                if r.verbose is None:
                    r.verbose = True
                tm.angles.append(r)
            elif isinstance(r, RecordDihedral):
                if r.verbose is None:
                    r.verbose = True
                tm.dihedrals.append(r)
            elif isinstance(r, RecordSettle):
                tm.settles.append(r)
            elif isinstance(r, RecordExclusion):
                if r.verbose is None:
                    r.verbose = True
                tm.exclusions.append(r)
            elif isinstance(r, RecordPair):
                if r.verbose is None:
                    r.verbose = True
                tm.pairs.append(r)
            elif isinstance(r, RecordPairNb):
                if r.verbose is None:
                    r.verbose = True
                tm.pairs_nb.append(r)
            elif isinstance(r, RecordVirtualSite1):
                tm.virtual_sites1.append(r)
            elif isinstance(r, RecordVirtualSite2):
                tm.virtual_sites2.append(r)
            elif isinstance(r, RecordVirtualSite3):
                tm.virtual_sites3.append(r)
            elif isinstance(r, RecordVirtualSite4):
                tm.virtual_sites4.append(r)
            elif isinstance(r, RecordDihedralRestraints):
                tm.dihedral_restraints.append(r)
            elif isinstance(r, RecordPositionRestraints):
                tm.position_restraints.append(r)

        if sort_idx:
            tm._sort_bonds()
            tm._sort_angles()
            tm._sort_dihedrals()
            tm._sort_exclusions()
            tm._sort_pairs()
            tm._sort_pairs_nb()

        # tm._check_pair_exclusion()
        tm._check_pair_nb_exclusion()
        if "r" in tm.round_on:
            if allow_round_diff is None or allow_round_diff < 0:
                raise RuntimeError(f"Incorrect allow_round_diff (current: {allow_round_diff}).")
            elif allow_round_diff > 0:
                tm.update_charge(partial_charges=None, symm=None, to_round=True)
            # else: pass

        return tm


########################
# full system topology #
########################


class TopoFullSystem:

    def __del__(self):
        TopoDefaults.remove(self.uuid)
        TopoAtomTypes.remove(self.uuid)

    def __init__(self):
        """
        e.g.
        mol_topos:
            [Water, Na, Cl, Protein, Ligand]
        molecules:
            [Ligand*1, Protein*1, Na*3, Cl*4, Water*1000]
        mol_to_topo_index:
            [4, 3, 1, 2, 0]
        """
        self.remarks: list[RecordText] = []
        self.mol_topos: list[TopoMolecule] = []
        self.system: RecordSystem = RecordSystem.from_text("Full System")
        self.molecules: list[RecordMolecule] = []

        self.mol_to_topo_index: list[int] = []
        self.uuid = uuid4()

        TopoDefaults(self.uuid)
        TopoAtomTypes(self.uuid)

    @classmethod
    def from_file(cls,
                  path: str,
                  incdir: str = None,
                  sort_idx: bool = True,
                  allow_unknown: bool = False,
                  *,
                  round_on: Union[TopoRoundModeEnum, str] = TopoRoundModeEnum.ON_WRITE,
                  allow_round_diff: float = 0.05,
                  override_atom_comment_with_qtot: bool = True):
        records = Records.from_file(path=path, incdir=incdir, allow_unknown=allow_unknown)
        return cls.from_records(records=records.all,
                                sort_idx=sort_idx,
                                round_on=round_on,
                                allow_round_diff=allow_round_diff,
                                override_atom_comment_with_qtot=override_atom_comment_with_qtot)

    @classmethod
    def from_records(cls,
                     records: list[RecordText],
                     sort_idx: bool,
                     *,
                     round_on: Union[TopoRoundModeEnum, str] = TopoRoundModeEnum.ON_WRITE,
                     allow_round_diff: float = 0.05,
                     override_atom_comment_with_qtot: bool = True):
        tfs = cls()
        for r in records:
            if type(r) == RecordText:
                if r.text == "" and r.comment != " ; ":
                    tfs.remarks.append(r)
            else:
                break

        td = TopoDefaults(tfs.uuid)
        ta = TopoAtomTypes(tfs.uuid)
        for r in records:
            if isinstance(r, RecordDefaults):
                td.add_record(r)
            elif isinstance(r, RecordAtomType):
                ta.add_record(r)
            elif isinstance(r, RecordSystem):
                tfs.system = deepcopy(r)
            elif isinstance(r, RecordMolecule):
                tfs.molecules.append(r)

        for idx, r in enumerate(records):
            if isinstance(r, RecordMoleculeType):
                m = TopoMolecule.from_records(system_uuid=tfs.uuid,
                                              records=records[idx:],
                                              sort_idx=sort_idx,
                                              round_on=round_on,
                                              allow_round_diff=allow_round_diff,
                                              override_atom_comment_with_qtot=override_atom_comment_with_qtot)
                for im in tfs.mol_topos:
                    if m.name == im.name:
                        raise TopoDuplicateNameMoleculeTypeException(m.name)
                tfs.mol_topos.append(m)

        if len(tfs.molecules) == 0:
            assert len(tfs.mol_topos)
            for mol_topo in tfs.mol_topos:
                tfs.molecules.append(RecordMolecule.from_text(f"{mol_topo.name} 1"))

        for mol in tfs.molecules:
            for idx, mol_topo in enumerate(tfs.mol_topos):
                if mol.name == mol_topo.name:
                    tfs.mol_to_topo_index.append(idx)

        return tfs

    def _strs_mol_atp_itp(self, idx):
        """
        atp
            [ atomtypes ]
        itp
            [ moleculetypes ]
            [ atoms ]
            ...
        """

        idx = [idx] if isinstance(idx, int) else [i for i in idx]
        ta = TopoAtomTypes(self.uuid)

        pos, itp_lines = [], []
        for ii in idx:
            mol_idx = self.mol_to_topo_index[ii]
            tm = self.mol_topos[mol_idx]
            itp_lines.append(str(tm))
            pos += list(set(tm.atom_to_type_index))
        pos.sort()

        return ta.str_slice(pos), "\n".join(itp_lines)

    def str_mol_itp(self, idx) -> str:
        """
        ; remarks
        [ atomtypes ]
        [ moleculetypes ]
        [ atoms ]
        ...
        """

        atp_line, itp_line = self._strs_mol_atp_itp(idx)
        lines = []

        for a in self.remarks:
            lines.append(a.comment)
        lines.append(atp_line)
        lines.append(itp_line)
        return "\n".join(lines)

    def str_mol_atp_itp(self, idx) -> tuple[str]:
        """
        ; remarks
        [ atomtypes ]
        [ moleculetypes ]
        [ atoms ]
        ...
        """

        atp_line, itp_line = self._strs_mol_atp_itp(idx)
        atp_lines, itp_lines = [], []

        for a in self.remarks:
            atp_lines.append(a.comment)
        atp_lines.append(atp_line)
        for a in self.remarks:
            itp_lines.append(a.comment)
        itp_lines.append(itp_line)
        return "\n".join(atp_lines), "\n".join(itp_lines)

    def write_itp(self, itp_path: str, idx=0, separated_atp=False):
        if os.path.exists(itp_path):
            abspath = os.path.abspath(itp_path)
            logger.warning(f"File {abspath} exists and will be overwritten.")

        if separated_atp:
            atp_str, itp_str = self.str_mol_atp_itp(idx)
            logger.info(f"Writing itp: {itp_path}")
            with open(itp_path, "w") as fw:
                fw.write(itp_str)
                fw.write("\n")
            atp_path = itp_path[:-4] + '.atp'
            logger.info(f"Writing atp: {atp_path}")
            with open(atp_path, "w") as fw:
                fw.write(atp_str)
                fw.write("\n")

        else:
            itp_str = self.str_mol_itp(idx)
            logger.info(f"Writing itp: {itp_path}")
            with open(itp_path, "w") as fw:
                fw.write(itp_str)
                fw.write("\n")

    def str_system_top(self) -> str:
        """
        ; remarks
        [ defaults ]
        [ atomtypes ]
        ...
        [ system ]
        [ molecules ]
        """

        ta = TopoAtomTypes(self.uuid)
        lines = []

        for a in self.remarks:
            lines.append(a.comment)
        lines.append(str(TopoDefaults(self.uuid)))
        lines.append(str(ta))
        for a in self.mol_topos:
            lines.append(str(a))

        lines.append("")
        section = "system"
        lines.append(str(RecordSection(section=section)))
        lines.append(str(self.system))

        lines.append("")
        section = "molecules"
        lines.append(str(RecordSection(section=section)))
        lines.append(self.molecules[0].annotate())
        for a in self.molecules:
            lines.append(str(a))

        return "\n".join(lines)

    def write_top(self, top_path: str):
        top_str = self.str_system_top()
        if os.path.exists(top_path):
            abspath = os.path.abspath(top_path)
            logger.warning(f"File {abspath} exists and will be overwritten.")
        logger.info(f"Writing top: {top_path}")
        with open(top_path, "w") as fw:
            fw.write(top_str)
            fw.write("\n")

    def strs_system_top_atp_itp(self, atps: list[str], itps: list[str], mols: list[list[int]]):
        assert len(atps) == len(mols) and len(itps) == len(mols)

        # top string
        lines = []

        for a in self.remarks:
            lines.append(a.comment)
        lines.append(str(TopoDefaults(self.uuid)))

        lines.append("")
        for a in atps:
            lines.append(f'#include "{a}"')
        lines.append("")
        for a in itps:
            lines.append(f'#include "{a}"')

        lines.append("")
        section = "system"
        lines.append(str(RecordSection(section=section)))
        lines.append(str(self.system))

        lines.append("")
        section = "molecules"
        lines.append(str(RecordSection(section=section)))
        lines.append(self.molecules[0].annotate())
        for a in self.molecules:
            lines.append(str(a))

        # atp strings and atp strings
        atp_lines, itp_lines = [], []
        for idx in mols:
            atp, itp = self._strs_mol_atp_itp(idx=idx)
            atp_lines.append(atp)
            itp_lines.append(itp)

        return "\n".join(lines), atp_lines, itp_lines

    def write_top_atp_itp(self, top_path: str, atps: list[str], itps: list[str], mols: list[list[int]]):
        """
        Example

        [ molecules ]
        LIGAND         1
        PROTEIN        1
        WATER      10494
        NA            33
        CL            40
        CA             2

        tfs = TopoFullSystem.from_file(path=top_path)
        tfs.write_top_atp_itp(top_path="system.top",
            atps=["protein.atp", "water.atp", "ions.atp", "ligand.atp"],
            itps=["protein.itp", "water.itp", "ions.itp", "ligand.itp"],
            mols=[[1],           [2],         [3,4,5],    [0]])
        # Na, Cl, and Ca are written to "ions.atp" and "ions.itp".
        """
        top_str, atp_strs, itp_strs = self.strs_system_top_atp_itp(atps, itps, mols)
        for s, p in zip([top_str] + atp_strs + itp_strs, [top_path] + atps + itps):
            abspath = os.path.abspath(p)
            if os.path.exists(p):
                logger.warning(f"File {abspath} exists and will be overwritten.")
            with open(p, "w") as fw:
                fw.write(s)
                fw.write("\n")
