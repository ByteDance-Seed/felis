# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from copy import deepcopy
from typing import Optional, Tuple, Union

from bytemol.toolkit.gmxtool.topparse_impl.top_enums import (
    AngleTypeEnum,
    BondTypeEnum,
    DihedralTypeEnum,
    LJCombinationRuleEnum,
    NonbondedFunctionEnum,
    PairNbTypeEnum,
    PairTypeEnum,
    VirtualSite2Enum,
    VirtualSite3Enum,
    VirtualSite4Enum,
)
from bytemol.toolkit.gmxtool.topparse_impl.top_exceptions import TopoDuplicateNameAtomTypeException
from bytemol.toolkit.gmxtool.topparse_impl.top_factory import Factory

logger = logging.getLogger(__name__)

# pylint: disable=attribute-defined-outside-init
# this is necessary because these classes use hasattr and delattr to control its behavior.

###########
# records #
###########


class RecordText:

    def _check(self):
        pass

    def _init(self, **kwargs):
        """
        line: None or str. Passed in as "text=line".
            line will be stripped, then split by semicolon into text and comment.
            text will be split by whitespace into _fields.
        verbose: None or bool. Passed in as "verbose=verbose".
            In __str__, comment will be appended to the output only if True.
            If set to False, comment will never be print, unless verbose is set to True again.
        comment: None or str. Passed in as "comment=comment".
            If not None, this will overwrite the comment found in `line'.
        """

        self.line: str = kwargs.get("text", None)
        self.verbose: bool = kwargs.get("verbose", None)

        self._comment: str = None
        self.text: str = None
        self._fields: list[str] = []

        if isinstance(self.line, str):
            self.line = self.line.rstrip().lstrip()
            self._split(line=self.line)
        self.comment = kwargs.get("comment", self._comment)

    @property
    def comment(self):
        return self._comment

    @comment.setter
    def comment(self, _comment: Optional[str]):
        assert _comment is None or isinstance(_comment, str), TypeError("Comment must be a string or None.")

        if _comment is not None and not _comment.lstrip().startswith(";"):
            _comment = " ; " + _comment.strip()

        self._comment = _comment

    def _split(self, line: str):
        n = len(line)
        i = 0
        while i < n:
            if line[i] == ";":
                break
            i = i + 1
        end = i - 1
        while end > -1:
            if line[end] != " ":
                break
            end = end - 1

        self._comment = line[end + 1:]
        self.text = line[:end + 1]
        self._fields = self.text.split()

    #########
    # print #
    #########

    def _str_impl(self):
        return self.text

    def __str__(self) -> str:
        line = self._str_impl()
        if self.verbose and self.comment is not None and self.comment.strip() != ";":
            assert self.comment.lstrip().startswith(";"), ValueError("Comment must start with a semicolon.")

            line = line + self.comment
        return line

    ############
    # __init__ #
    ############

    def __init__(self, **kwargs):
        self._init(**kwargs)
        self._check()

    @classmethod
    def from_text(cls, text: str, **kwargs):
        r = super().__new__(cls)
        r._init(text=text, **kwargs)
        r._parse_line()
        r._check()
        return r

    def _parse_line(self):
        pass


class RecordInclude(RecordText):

    def _init(self, dirname: str, include: str = None, **kwargs):
        super()._init(**kwargs)

        self.dirname = dirname
        self.include = include
        self.abspath = None

    def _parse_line(self):
        fields = self._fields

        inc = fields[1]
        inc = inc[1:-1]
        self.include = inc
        if os.path.isabs(inc):
            self.abspath = self.include
        else:
            path = f"{self.dirname}/{inc}"
            path = os.path.abspath(path)
            self.abspath = path

    def _str_impl(self):
        return f"#include \"{self.include}\""


class RecordSection(RecordText):

    allowed_sections = (
        "defaults",
        "atomtypes",
        ##
        "moleculetype",
        "atoms",
        # "bondtypes",
        "bonds",
        # "angletypes",
        "angles",
        # "dihedraltypes",
        "dihedrals",
        "settles",
        "exclusions",
        # "pairtypes",
        "pairs",
        "pairs_nb",
        # "constrainttypes",
        # "constraints",
        # "nonbond_params",
        "virtual_sites1",
        "virtual_sites2",
        "virtual_sites3",
        "virtual_sites4",
        "dihedral_restraints",
        "position_restraints",
        # "virtual_sitesn",
        # "*restraints*",
        ##
        "system",
        "molecules",
    )

    def _check(self):
        if self.section is None:
            return
        if self.section not in self.allowed_sections:
            if self.allow_unknown:
                logger.warning(f"Ignoring section [ {self.section} ].")
            else:
                raise NotImplementedError(f"Section [ {self.section} ] is not supported.")

    def _init(self, section: str = None, allow_unknown: bool = False, **kwargs):
        super()._init(**kwargs)
        self.section = section
        self.allow_unknown = allow_unknown

    def _parse_line(self):
        fields = self._fields

        self.section = fields[1]

    def _str_impl(self):
        return f"[ {self.section} ]"


############
# defaults #
############


class RecordDefaults(RecordText):

    def _check(self):
        assert self.gen_pairs in ("yes", "no")
        assert 0 < self.fudge_lj <= 1
        assert 0 < self.fudge_qq <= 1

    def _init(self,
              nbfunc: NonbondedFunctionEnum = NonbondedFunctionEnum.LENNARD_JONES,
              comb_rule: LJCombinationRuleEnum = LJCombinationRuleEnum.SIGMA_EPSILON,
              gen_pairs: str = "yes",
              fudge_lj: float = 0.5,
              fudge_qq: float = 1.0 / 1.2,
              **kwargs):
        super()._init(**kwargs)

        self.nbfunc, self.comb_rule = nbfunc, comb_rule
        self.gen_pairs = gen_pairs
        self.fudge_lj, self.fudge_qq = fudge_lj, fudge_qq  # vdw 14 and charge 14 scales
        if isinstance(self.nbfunc, int):
            self.nbfunc = NonbondedFunctionEnum(self.nbfunc)
        if isinstance(self.comb_rule, int):
            self.comb_rule = LJCombinationRuleEnum(self.comb_rule)
        if isinstance(self.gen_pairs, str):
            self.gen_pairs = str.lower(self.gen_pairs)

    def _parse_line(self):
        fields = self._fields

        nfields = len(fields)
        self.nbfunc = NonbondedFunctionEnum(int(fields[0]))
        self.comb_rule = LJCombinationRuleEnum(int(fields[1]))
        self.gen_pairs = str.lower(fields[2])
        if nfields > 3:
            self.fudge_lj = float(fields[3])
        if nfields > 4:
            self.fudge_qq = float(fields[4])

    def annotate(self) -> str:
        return ";     nbfunc   comb-rule   gen-pairs     fudgeLJ     fudgeQQ"

    def _str_impl(self):
        return f"{self.nbfunc:>12}{self.comb_rule:>12}{self.gen_pairs:>12}{self.fudge_lj:>12.8f}{self.fudge_qq:>12.8f}"


class TopoDefaults(Factory):

    def __init__(self, uuid=None):
        if not self._initialized:
            self._initialized = True
            self._added = False
            self.item: RecordDefaults = RecordDefaults()

    def __str__(self) -> str:
        section = RecordSection(section="defaults")
        lines = []
        lines.append("")
        lines.append(str(section))
        lines.append(self.item.annotate())
        lines.append(str(self.item))
        return "\n".join(lines)

    def add_record(self, record: RecordDefaults) -> int:
        assert isinstance(record, RecordDefaults)
        if not self._added:
            self._added = True
            self.item = deepcopy(record)
            return 0
        else:
            raise RuntimeError("A [ defaults ] section has been parsed.")

    @property
    def nbfunc(self):
        return self.item.nbfunc

    @property
    def comb_rule(self):
        return self.item.comb_rule

    @property
    def gen_pairs(self):
        return self.item.gen_pairs

    @property
    def fudge_lj(self):
        return self.item.fudge_lj

    @property
    def fudge_qq(self):
        return self.item.fudge_qq


#############
# atomtypes #
#############


class RecordAtomType(RecordText):
    """
    type name; bonded type (optional); atomic number (optional); m; q; particle type; V; W

    alias (if comb_rule == SIGMA_EPSILON)
    V: sigma
    W: epsilon
    """

    def _check(self):
        if isinstance(self.particle_type, str):
            assert self.particle_type in ("A", "V")

    def _init(self,
              name: str = None,
              at_num: int = 0,
              mass: float = 0.0,
              charge: float = 0.0,
              ptype: str = "A",
              V: float = 1.0,
              W: float = 0.0,
              **kwargs):
        super()._init(**kwargs)

        self.name = name
        # self.bonded_type = None
        self.at_num = at_num
        self.mass, self.charge = mass, charge
        self.particle_type = ptype
        self.V, self.W = V, W

    def _parse_line(self):
        fields = self._fields

        nfields = len(fields)
        self.name = fields[0]
        b = 1
        if nfields == 8:
            self.bonded_type, self.at_num = fields[1], int(fields[2])
            b = 3
        elif nfields == 7:
            if fields[1].isdigit():
                self.at_num = int(fields[1])
            else:
                self.bonded_type = fields[1]
                delattr(self, "at_num")
            b = 2
        self.mass, self.charge = float(fields[b + 0]), float(fields[b + 1])
        self.particle_type = fields[b + 2]
        self.V, self.W = float(fields[b + 3]), float(fields[b + 4])

    def annotate(self, comb_rule: LJCombinationRuleEnum) -> str:
        line = ";name           bonded.type        at.num        mass        charge   ptype"
        if comb_rule == LJCombinationRuleEnum.SIGMA_EPSILON:
            return line + "           sigma         epsilon"
        else:
            return line + "           V(c6)          W(c12)"

    def _str_impl(self):
        line = f" {self.name:<15}"
        if hasattr(self, "bonded_type") and hasattr(self, "at_num"):
            line = line + f"{self.bonded_type:>15}{self.at_num:>10}"
        elif hasattr(self, "at_num"):
            line = line + f"               {self.at_num:>10}"
        elif hasattr(self, "bonded_type"):
            line = line + f"{self.bonded_type:>15}          "
        line = line + f"{self.mass:>12.4f}{self.charge:>14.5f}{self.particle_type:>8}{self.V:>16.5e}{self.W:>16.5e}"
        return line

    @property
    def sigma(self):
        return self.V

    @sigma.setter
    def sigma(self, value):
        self.V = value

    @property
    def epsilon(self):
        return self.W

    @epsilon.setter
    def epsilon(self, value):
        self.W = value


class TopoAtomTypes(Factory):

    def __init__(self, uuid=None):
        if not self._initialized:
            self._initialized = True
            self.atomtypes: list[RecordAtomType] = []
            self.type_to_index: dict[str, int] = {}

    @property
    def atom_types(self):
        return self.atomtypes

    def copy_from(self, ta):
        if self.uuid == ta.uuid:
            return
        self.atomtypes = [a for a in ta.atomtypes]
        self.type_to_index = deepcopy(ta.type_to_index)

    def add_record(self, record: RecordAtomType) -> int:
        """
        Return 0-based index where the new atomtype can be found.
        """
        assert isinstance(record, RecordAtomType)
        if record.name in self.type_to_index.keys():
            raise TopoDuplicateNameAtomTypeException(name=record.name)
        idx = len(self.atomtypes)
        self.atomtypes.append(record)
        self.type_to_index[record.name] = idx
        if record.verbose is None:
            self.atomtypes[idx].verbose = True
        return idx

    def rename_atomtype(self, oldname: str, newname: str) -> int:
        """
        Return 0-based index where the atom type name is renamed in-place.
        """
        if newname in self.type_to_index.keys():
            raise TopoDuplicateNameAtomTypeException(newname)
        idx = self.type_to_index[oldname]
        self.atomtypes[idx].name = newname
        self.type_to_index.pop(oldname)
        self.type_to_index[newname] = idx
        return idx

    def str_slice(self, select) -> str:
        lines = []
        if isinstance(select, slice):
            atps = self.atomtypes[select]
        else:
            atps = [self.atomtypes[i] for i in select]
        if len(atps):
            td = TopoDefaults(self.uuid)
            section = RecordSection(section="atomtypes")
            lines.append("")
            lines.append(str(section))
            lines.append(self.atomtypes[0].annotate(td.comb_rule))
            for a in atps:
                lines.append(str(a))
        return "\n".join(lines)

    def __str__(self) -> str:
        select = slice(0, None)
        return self.str_slice(select)


##########################################
# atoms, moleculetype, molecules, system #
##########################################


class RecordAtom(RecordText):

    def _check(self):
        # limited by gro file specs
        if isinstance(self.residue, str):
            if len(self.residue) > 5:
                logger.warning(f"Residue longer than 5 characters: {self.residue}")
        if isinstance(self.atom, str):
            if len(self.atom) > 5:
                logger.warning(f"Atom name longer than 5 characters: {self.atom}")

    def _init(self,
              nr: int = None,
              atype: str = None,
              resnr: int = None,
              residue: str = None,
              atom: str = None,
              cgnr: int = None,
              charge: float = None,
              mass: float = None,
              **kwargs):
        super()._init(**kwargs)

        self.nr, self.atype = nr, atype  # number, atomtype
        self.resnr, self.residue = resnr, residue  # residue number, residue name
        self.atom, self.cgnr = atom, cgnr  # atom name, charge group number
        if charge is not None:
            self.charge = charge
        if mass is not None:
            self.mass = mass

    def _parse_line(self):
        fields = self._fields

        nfields = len(fields)
        self.nr, self.atype = int(fields[0]), fields[1]
        self.resnr, self.residue = int(fields[2]), fields[3]
        self.atom, self.cgnr = fields[4], int(fields[5])
        if nfields > 6:
            self.charge = float(fields[6])
        if nfields > 7:
            self.mass = float(fields[7])

    def annotate(self) -> str:
        line = ";       nr            type      resi   res      atom      cgnr"
        if hasattr(self, "charge"):
            line = line + "      charge"
        if hasattr(self, "mass"):
            line = line + "          mass"
        return line

    def _str_impl(self):
        line = f"{self.nr:>10}{self.atype:>16}{self.resnr:>10}{self.residue:>6}{self.atom:>10}{self.cgnr:>10}"
        if hasattr(self, "charge"):
            line = line + f"{self.charge:>12.5f}"
        if hasattr(self, "mass"):
            line = line + f"{self.mass:>14.4f}"
        return line


class RecordMoleculeType(RecordText):

    def _init(self, name: str = None, nrexcl: int = None, **kwargs):
        super()._init(**kwargs)

        self.name, self.nrexcl = name, nrexcl

    def _parse_line(self):
        fields = self._fields

        self.name = fields[0]
        self.nrexcl = int(fields[1])

    def annotate(self) -> str:
        return ";name             nrexcl"

    def _str_impl(self):
        return f" {self.name:<15}{self.nrexcl:>8}"


class RecordMolecule(RecordText):

    def _init(self, name: str = None, nr: int = None, **kwargs):
        super()._init(**kwargs)

        self.name, self.nr = name, nr  # molecule name, number of molecules

    def _parse_line(self):
        fields = self._fields

        self.name, self.nr = fields[0], int(fields[1])

    def annotate(self) -> str:
        return ";molecule            nmols"

    def _str_impl(self):
        return f" {self.name:<15}{self.nr:>10}"


class RecordSystem(RecordText):

    def _init(self, title: str = None, **kwargs):
        super()._init(**kwargs)

        self.title = title

    def _parse_line(self):
        self.title = self.text

    def _str_impl(self):
        return self.title


############################
# bonds, angles, dihedrals #
# settles                  #
# exclusions, pairs        #
############################


class RecordBond(RecordText):

    def _init(self,
              ai: int = None,
              aj: int = None,
              funct: BondTypeEnum = None,
              c0: float = None,
              c1: float = None,
              c2: float = None,
              c3: float = None,
              **kwargs):
        super()._init(**kwargs)

        if isinstance(ai, int):
            self.ai = ai
        if isinstance(aj, int):
            self.aj = aj
        self.funct = funct
        self.c0, self.c1 = c0, c1
        if c2 is not None:
            self.c2 = c2
        if c3 is not None:
            self.c3 = c3
        if isinstance(self.funct, int):
            self.funct = BondTypeEnum(self.funct)

    def raise_not_impl(self):
        raise NotImplementedError(f"Bond funct {self.funct} ({self.funct.name}) is not implemented.")

    def _parse_line(self):
        fields = self._fields

        if fields[0].isdigit() and fields[1].isdigit():
            self.ai, self.aj = int(fields[0]), int(fields[1])
        self.funct = BondTypeEnum(int(fields[2]))
        if self.funct in (BondTypeEnum.BOND, BondTypeEnum.HARMONIC_POTENTIAL):
            assert len(fields) == 5
            self.c0, self.c1 = float(fields[3]), float(fields[4])
        elif self.funct == BondTypeEnum.SOFTBOND:
            assert len(fields) == 7
            self.c0, self.c1, self.c2, self.c3 = float(fields[3]), float(fields[4]), float(fields[5]), float(fields[6])
        else:
            self.raise_not_impl()

    def cmp_key(self):
        return (self.ai, self.aj)

    def annotate(self) -> str:
        if self.funct in (BondTypeEnum.BOND, BondTypeEnum.HARMONIC_POTENTIAL):
            return ";       ai        aj funct               r               k"
        elif self.funct == BondTypeEnum.SOFTBOND:
            return ";       ai        aj funct               r               k           alpha          lambda"
        else:
            self.raise_not_impl()

    def _str_impl(self):
        line = f"{self.ai:>10}{self.aj:>10}{self.funct:>6}"
        if self.funct in (BondTypeEnum.BOND, BondTypeEnum.HARMONIC_POTENTIAL):
            line += f"{self.c0:>16.6f}{self.c1:>16.6f}"
        elif self.funct == BondTypeEnum.SOFTBOND:
            line += f"{self.c0:>16.6f}{self.c1:>16.6f}{self.c2:>16.6f}{self.c3:>16.6f}"
        else:
            self.raise_not_impl()
        return line

    @property
    def b0(self):
        assert self.funct in (BondTypeEnum.BOND, BondTypeEnum.HARMONIC_POTENTIAL, BondTypeEnum.SOFTBOND)
        return self.c0

    @b0.setter
    def b0(self, value):
        assert self.funct in (BondTypeEnum.BOND, BondTypeEnum.HARMONIC_POTENTIAL, BondTypeEnum.SOFTBOND)
        self.c0 = value

    @property
    def kb(self):
        assert self.funct in (BondTypeEnum.BOND, BondTypeEnum.HARMONIC_POTENTIAL, BondTypeEnum.SOFTBOND)
        return self.c1

    @kb.setter
    def kb(self, value):
        assert self.funct in (BondTypeEnum.BOND, BondTypeEnum.HARMONIC_POTENTIAL, BondTypeEnum.SOFTBOND)
        self.c1 = value

    @property
    def alpha(self):
        assert self.funct in (BondTypeEnum.SOFTBOND,)
        return self.c2

    @alpha.setter
    def alpha(self, value):
        assert self.funct in (BondTypeEnum.SOFTBOND,)
        self.c2 = value

    @property
    def lam(self):
        assert self.funct in (BondTypeEnum.SOFTBOND,)
        return self.c3

    @lam.setter
    def lam(self, value):
        assert self.funct in (BondTypeEnum.SOFTBOND,)
        self.c3 = value


class RecordPair(RecordText):

    def _init(self,
              ai: int = None,
              aj: int = None,
              funct: PairTypeEnum = None,
              V: float = None,
              W: float = None,
              fudge_qq: float = None,
              qi: float = None,
              qj: float = None,
              **kwargs):
        super()._init(**kwargs)

        self.ai, self.aj = ai, aj
        self.funct = funct
        if V is not None:
            self.V = V
        if W is not None:
            self.W = W
        if fudge_qq is not None:
            self.fudge_qq = fudge_qq
        if qi is not None:
            self.qi = qi
        if qj is not None:
            self.qj = qj
        if isinstance(self.funct, int):
            self.funct = PairTypeEnum(self.funct)

    def get_params(self, td: TopoDefaults, vws: list[Tuple[float, float]],
                   charges: list[float]) -> Tuple[float, float, float, float, float]:
        fudge_lj = td.fudge_lj
        fudge_qq, qi, qj = td.fudge_qq, 0.0, 0.0
        v, w = 1.0, 0.0

        ai0, aj0 = self.ai - 1, self.aj - 1

        if hasattr(self, "V") and hasattr(self, "W"):
            v, w = self.V, self.W
        else:
            assert td.nbfunc == NonbondedFunctionEnum.LENNARD_JONES
            assert td.comb_rule == LJCombinationRuleEnum.SIGMA_EPSILON

            vi, wi = vws[ai0]
            vj, wj = vws[aj0]
            v = (vi + vj) / 2
            w = (wi * wj)**0.5
            w *= fudge_lj

        if hasattr(self, "fudge_qq") and hasattr(self, "qi") and hasattr(self, "qj"):
            fudge_qq, qi, qj = self.fudge_qq, self.qi, self.qj
        else:
            qi, qj = charges[ai0], charges[aj0]

        return (v, w, fudge_qq, qi, qj)

    def _parse_line(self):
        fields = self._fields

        nfields = len(fields)
        self.ai, self.aj = int(fields[0]), int(fields[1])
        self.funct = PairTypeEnum(int(fields[2]))
        if self.funct == PairTypeEnum.EXTRA_LJ:
            if nfields > 3:
                self.V, self.W = float(fields[3]), float(fields[4])
        elif self.funct == PairTypeEnum.EXTRA_COULOMB_LJ:
            self.fudge_qq, self.qi, self.qj = float(fields[3]), float(fields[4]), float(fields[5])
            self.V, self.W = float(fields[6]), float(fields[7])

    def cmp_key(self):
        return (self.ai, self.aj)

    def annotate(self) -> str:
        if self.funct == PairTypeEnum.EXTRA_LJ:
            if hasattr(self, "V") and hasattr(self, "W"):
                return ";       ai        aj funct               V               W"
            else:
                return ";       ai        aj funct"
        elif self.funct == PairTypeEnum.EXTRA_COULOMB_LJ:
            return ";       ai        aj funct         fudgeQQ              qi              qj               V               W"

    def _str_impl(self):
        line = f"{self.ai:>10}{self.aj:>10}{self.funct:>6}"
        if self.funct == PairTypeEnum.EXTRA_LJ:
            if hasattr(self, "V") and hasattr(self, "W"):
                return line + f"{self.V:>16.6f}{self.W:>16.6f}"
            else:
                return line
        elif self.funct == PairTypeEnum.EXTRA_COULOMB_LJ:
            return line + f"{self.fudge_qq:>16.6f}{self.qi:>16.6f}{self.qj:>16.6f}{self.V:>16.6f}{self.W:>16.6f}"
        else:
            self.raise_not_impl()


class RecordPairNb(RecordText):

    def _init(self,
              ai: int = None,
              aj: int = None,
              funct: PairNbTypeEnum = None,
              qi: float = None,
              qj: float = None,
              V: float = None,
              W: float = None,
              lam_vdw: float = None,
              **kwargs):
        super()._init(**kwargs)
        self.ai, self.aj = ai, aj
        self.funct = funct
        self.qi, self.qj = qi, qj
        self.V, self.W = V, W
        if lam_vdw is not None:
            self.lam_vdw = lam_vdw
        if isinstance(self.funct, int):
            self.funct = PairNbTypeEnum(self.funct)

    def raise_not_impl(self):
        raise NotImplementedError(f"PairNb funct {self.funct} ({self.funct.name}) is not implemented.")

    def _parse_line(self):
        fields = self._fields
        self.ai, self.aj = int(fields[0]), int(fields[1])
        self.funct = PairNbTypeEnum(int(fields[2]))
        self.qi, self.qj = float(fields[3]), float(fields[4])
        self.V, self.W = float(fields[5]), float(fields[6])
        if self.funct == PairNbTypeEnum.COULOMB_LJ:
            assert len(fields) == 7
        elif self.funct == PairNbTypeEnum.SOFTCORE_COULOMB_LJ:
            assert len(fields) == 8
            self.lam_vdw = float(fields[7])
        else:
            self.raise_not_impl()

    def cmp_key(self):
        return (self.ai, self.aj)

    def annotate(self) -> str:
        if self.funct == PairNbTypeEnum.COULOMB_LJ:
            return ";       ai        aj funct              qi              qj           sigma         epsilon"
        elif self.funct == PairNbTypeEnum.SOFTCORE_COULOMB_LJ:
            return ";       ai        aj funct              qi              qj           sigma         epsilon         lam_vdw"
        else:
            self.raise_not_impl()

    def _str_impl(self):
        line = f"{self.ai:>10}{self.aj:>10}{self.funct:>6}"
        if self.funct == PairNbTypeEnum.COULOMB_LJ:
            line += f"{self.qi:>16.6f}{self.qj:>16.6f}{self.V:>16.6f}{self.W:>16.6f}"
        elif self.funct == PairNbTypeEnum.SOFTCORE_COULOMB_LJ:
            line += f"{self.qi:>16.6f}{self.qj:>16.6f}{self.V:>16.6f}{self.W:>16.6f}{self.lam_vdw:>16.6f}"
        else:
            self.raise_not_impl()
        return line


class RecordAngle(RecordText):

    def _init(self,
              ai: int = None,
              aj: int = None,
              ak: int = None,
              funct: AngleTypeEnum = None,
              c0: float = None,
              c1: float = None,
              **kwargs):
        super()._init(**kwargs)

        if isinstance(ai, int):
            self.ai = ai
        if isinstance(aj, int):
            self.aj = aj
        if isinstance(ak, int):
            self.ak = ak
        self.funct = funct
        self.c0, self.c1 = c0, c1
        if isinstance(self.funct, int):
            self.funct = AngleTypeEnum(self.funct)

    def raise_not_impl(self):
        raise NotImplementedError(f"Angle funct {self.funct} ({self.funct.name}) is not implemented.")

    def _parse_line(self):
        fields = self._fields

        if fields[0].isdigit() and fields[1].isdigit() and fields[2].isdigit():
            self.ai, self.aj, self.ak = int(fields[0]), int(fields[1]), int(fields[2])
        self.funct = AngleTypeEnum(int(fields[3]))
        if self.funct == AngleTypeEnum.ANGLE:
            self.c0, self.c1 = float(fields[4]), float(fields[5])
        else:
            self.raise_not_impl()

    def cmp_key(self):
        return (self.ai, self.aj, self.ak)

    def annotate(self) -> str:
        if self.funct == AngleTypeEnum.ANGLE:
            return ";       ai        aj        ak funct           theta             cth"
        else:
            self.raise_not_impl()

    def _str_impl(self):
        if self.funct == AngleTypeEnum.ANGLE:
            line = f"{self.ai:>10}{self.aj:>10}{self.ak:>10}"
            line = line + f"{self.funct:>6}{self.c0:>16.6f}{self.c1:>16.6f}"
            return line
        else:
            self.raise_not_impl()

    @property
    def theta(self):
        assert self.funct in (AngleTypeEnum.ANGLE,)
        return self.c0

    @theta.setter
    def theta(self, value):
        assert self.funct in (AngleTypeEnum.ANGLE,)
        self.c0 = value

    @property
    def k(self):
        assert self.funct in (AngleTypeEnum.ANGLE,)
        return self.c1

    @k.setter
    def k(self, value):
        assert self.funct in (AngleTypeEnum.ANGLE,)
        self.c1 = value


class RecordDihedral(RecordText):

    def _check(self):
        if self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                          DihedralTypeEnum.MULTIPLE_PROPER):
            assert isinstance(self.c2, int)

    def _init(self,
              ai: int = None,
              aj: int = None,
              ak: int = None,
              al: int = None,
              funct: DihedralTypeEnum = None,
              c0: float = None,
              c1: float = None,
              c2: Union[int, float] = None,
              c3: float = None,
              c4: float = None,
              c5: float = None,
              **kwargs):
        super()._init(**kwargs)

        if isinstance(ai, int):
            self.ai = ai
        if isinstance(aj, int):
            self.aj = aj
        if isinstance(ak, int):
            self.ak = ak
        if isinstance(al, int):
            self.al = al
        self.funct = funct
        self.c0, self.c1, self.c2 = c0, c1, c2
        self.c3, self.c4, self.c5 = c3, c4, c5
        if isinstance(self.funct, int):
            self.funct = DihedralTypeEnum(self.funct)

    def raise_not_impl(self):
        raise NotImplementedError(f"Dihedral funct {self.funct} ({self.funct.name}) is not implemented.")

    def _parse_line(self):
        fields = self._fields

        if fields[0].isdigit() and fields[1].isdigit() and fields[2].isdigit() and fields[3].isdigit():
            self.ai, self.aj, self.ak, self.al = int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])
        self.funct = DihedralTypeEnum(int(fields[4]))
        if self.funct == DihedralTypeEnum.RYCKAERT_BELLEMANS:
            self.c0 = float(fields[5])
            self.c1 = float(fields[6])
            self.c2 = float(fields[7])
            self.c3 = float(fields[8])
            self.c4 = float(fields[9])
            self.c5 = float(fields[10])
        elif self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                            DihedralTypeEnum.MULTIPLE_PROPER):
            self.c0 = float(fields[5])
            self.c1 = float(fields[6])
            self.c2 = int(fields[7])
        elif self.funct in (DihedralTypeEnum.IMPROPER,):
            self.c0 = float(fields[5])
            self.c1 = float(fields[6])
        else:
            self.raise_not_impl()

    def cmp_key(self):
        return (self.ai, self.aj, self.ak, self.al)

    def annotate(self) -> str:
        line = ";       ai        aj        ak        al funct"
        if self.funct in (DihedralTypeEnum.IMPROPER,):
            line = line + "           phase               k"
            return line
        if self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                          DihedralTypeEnum.MULTIPLE_PROPER):
            line = line + "           phase               k        pn"
            return line
        elif self.funct in (DihedralTypeEnum.RYCKAERT_BELLEMANS,):
            line = line + "               c"
        else:
            self.raise_not_impl()

    def _str_impl(self):
        line = f"{self.ai:>10}{self.aj:>10}{self.ak:>10}{self.al:>10}"
        line = line + f"{self.funct:>6}"
        if self.funct == DihedralTypeEnum.RYCKAERT_BELLEMANS:
            return line + f"{self.c0:>16.6f}{self.c1:>16.6f}{self.c2:>16.6f}{self.c3:>16.6f}{self.c4:>16.6f}{self.c5:>16.6f}"
        elif self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                            DihedralTypeEnum.MULTIPLE_PROPER):
            return line + f"{self.c0:>16.6f}{self.c1:>16.6f}{self.c2:>10}"
        elif self.funct in (DihedralTypeEnum.IMPROPER,):
            return line + f"{self.c0:>16.6f}{self.c1:>16.6f}"
        else:
            self.raise_not_impl()

    @property
    def phi(self):
        assert self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                              DihedralTypeEnum.MULTIPLE_PROPER, DihedralTypeEnum.IMPROPER)
        return self.c0

    @phi.setter
    def phi(self, value):
        assert self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                              DihedralTypeEnum.MULTIPLE_PROPER, DihedralTypeEnum.IMPROPER)
        self.c0 = value

    @property
    def k(self):
        assert self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                              DihedralTypeEnum.MULTIPLE_PROPER, DihedralTypeEnum.IMPROPER)
        return self.c1

    @k.setter
    def k(self, value):
        assert self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                              DihedralTypeEnum.MULTIPLE_PROPER, DihedralTypeEnum.IMPROPER)
        self.c1 = value

    @property
    def multiplicity(self):
        assert self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                              DihedralTypeEnum.MULTIPLE_PROPER)
        return self.c2

    @multiplicity.setter
    def multiplicity(self, value):
        assert self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                              DihedralTypeEnum.MULTIPLE_PROPER)
        self.c2 = value


class RecordSettle(RecordText):

    def _check(self):
        assert self.funct == 1

    def _init(self, ao: int = None, funct: int = 1, doh: float = None, dhh: float = None, **kwargs):
        super()._init(**kwargs)

        self.idx = ao
        self.funct = funct
        self.doh, self.dhh = doh, dhh

    def _parse_line(self):
        fields = self._fields

        self.idx = int(fields[0])
        self.funct = int(fields[1])
        self.doh = float(fields[2])
        self.dhh = float(fields[3])

    def annotate(self) -> str:
        return ";             OW funct             dOH             dHH"

    def _str_impl(self):
        return f"{self.idx:>16}{self.funct:>6}{self.doh:>16.6f}{self.dhh:>16.6f}"


class RecordExclusion(RecordText):

    def _check(self):
        assert self.ai not in self.aj_list

    def _init(self, ai: int = None, aj_list: list[int] = None, **kwargs):
        super()._init(**kwargs)
        if aj_list is None:
            aj_list = []
        self.ai, self.aj_list = ai, [aj for aj in aj_list]
        self.aj_list.sort()

    def _parse_line(self):
        fields = self._fields

        self.ai = int(fields[0])
        self.aj_list = [int(aj) for aj in fields[1:]]
        self.aj_list.sort()

    def cmp_key(self):
        return self.ai

    def _str_impl(self):
        return " ".join([f"{j:>8}" for j in ([self.ai] + self.aj_list)])


#################
# virtual sites #
#################


class RecordVirtualSite1(RecordText):

    def _check(self):
        assert self.funct == 1

    def _init(self, av: int = None, ai: int = None, funct: int = 1, **kwargs):
        super()._init(**kwargs)

        self.av, self.ai = av, ai
        self.funct = funct

    def _parse_line(self):
        fields = self._fields

        self.av, self.ai = int(fields[0]), int(fields[1])
        self.funct = int(fields[2])

    def annotate(self) -> str:
        return ";     site         i funct"

    def _str_impl(self):
        return f"{self.av:>10}{self.ai:>10}{self.funct:>6}"


class RecordVirtualSite2(RecordText):

    def _init(self,
              av: int = None,
              ai: int = None,
              aj: int = None,
              funct: VirtualSite2Enum = None,
              c: float = None,
              **kwargs):
        super()._init(**kwargs)

        self.av, self.ai, self.aj = av, ai, aj
        self.funct = funct
        self.c = c
        if isinstance(self.funct, int):
            self.funct = VirtualSite2Enum(self.funct)

    def _parse_line(self):
        fields = self._fields

        self.av, self.ai, self.aj = int(fields[0]), int(fields[1]), int(fields[2])
        self.funct = VirtualSite2Enum(int(fields[3]))
        self.c = float(fields[4])

    def annotate(self) -> str:
        return ";     site        ij           funct          params"

    def _str_impl(self):
        return f"{self.av:>10}{self.ai:>10}{self.aj:>10}{self.funct:>6}{self.c:>16.6f}"


class RecordVirtualSite3(RecordText):

    def _init(self,
              av: int = None,
              ai: int = None,
              aj: int = None,
              ak: int = None,
              funct: VirtualSite3Enum = None,
              c0: float = None,
              c1: float = None,
              c2: float = None,
              **kwargs):
        super()._init(**kwargs)

        self.av, self.ai, self.aj, self.ak = av, ai, aj, ak
        self.funct = funct
        self.c0, self.c1, self.c2 = c0, c1, c2
        if isinstance(self.funct, int):
            self.funct = VirtualSite3Enum(self.funct)

    def _parse_line(self):
        fields = self._fields

        self.av, self.ai, self.aj, self.ak = int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])
        self.funct = VirtualSite3Enum(int(fields[4]))
        self.c0, self.c1 = float(fields[5]), float(fields[6])
        if self.funct == VirtualSite3Enum._3OUT:
            self.c2 = float(fields[7])

    def annotate(self) -> str:
        return ";     site       ijk                     funct          params"

    def _str_impl(self):
        line = f"{self.av:>10}{self.ai:>10}{self.aj:>10}{self.ak:>10}{self.funct:>6}{self.c0:>16.6f}{self.c1:>16.6f}"
        if self.funct == VirtualSite3Enum._3OUT:
            line = line + f"{self.c2:>16.6f}"
        return line


class RecordVirtualSite4(RecordText):

    def _init(self,
              av: int = None,
              ai: int = None,
              aj: int = None,
              ak: int = None,
              al: int = None,
              funct: VirtualSite4Enum = None,
              c0: float = None,
              c1: float = None,
              c2: float = None,
              **kwargs):
        super()._init(**kwargs)

        self.av, self.ai, self.aj, self.ak, self.al = av, ai, aj, ak, al
        self.funct = funct
        self.c0, self.c1, self.c2 = c0, c1, c2
        if isinstance(self.funct, int):
            self.funct = VirtualSite4Enum(self.funct)

    def _parse_line(self):
        fields = self._fields

        self.av = int(fields[0])
        self.ai, self.aj, self.ak, self.al = int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4])
        self.funct = VirtualSite4Enum(int(fields[5]))
        self.c0, self.c1, self.c2 = float(fields[6]), float(fields[7]), float(fields[8])

    def annotate(self) -> str:
        return ";     site      ijkl                               funct          params"

    def _str_impl(self):
        return f"{self.av:>10}{self.ai:>10}{self.aj:>10}{self.ak:>10}{self.al:>10}{self.funct:>6}{self.c0:>16.6f}{self.c1:>16.6f}{self.c2:>16.6f}"


class RecordDihedralRestraints(RecordText):

    def _check(self):
        assert self.funct == 1

    def _init(self,
              ai: int = None,
              aj: int = None,
              ak: int = None,
              al: int = None,
              funct: int = None,
              phi: float = None,
              fc: float = None,
              dphi: float = 0.,
              **kwargs):
        super()._init(**kwargs)
        if isinstance(ai, int):
            self.ai = ai
        if isinstance(aj, int):
            self.aj = aj
        if isinstance(ak, int):
            self.ak = ak
        if isinstance(al, int):
            self.al = al
        if isinstance(funct, int):
            self.funct = funct
        self.phi, self.fc, self.dphi = phi, fc, dphi

    def raise_not_impl(self):
        raise NotImplementedError(f"DihedralRestraints funct {self.funct} ({self.funct.name}) is not implemented.")

    def _parse_line(self):
        fields = self._fields
        if fields[0].isdigit() and fields[1].isdigit() and fields[2].isdigit() and fields[3].isdigit():
            self.ai, self.aj, self.ak, self.al = int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])
        self.funct = int(fields[4])
        self.phi = float(fields[5])
        self.dphi = float(fields[6])
        self.fc = float(fields[7])

    def cmp_key(self):
        return (self.ai, self.aj, self.ak, self.al)

    def annotate(self) -> str:
        line = ";       ai        aj        ak        al type           phi           dphi           fc"
        return line

    def _str_impl(self):
        line = f"{self.ai:>10}{self.aj:>10}{self.ak:>10}{self.al:>10}{self.funct:>6}{self.phi:>16.6f}{self.dphi:>16.6f}{self.fc:>16.6f}"
        return line


class RecordPositionRestraints(RecordText):

    def _check(self):
        assert self.funct == 1 or self.funct == 2

    def _init(
            self,
            ai: int = None,
            funct: int = None,
            g: int = None,  # direction
            r: float = None,  # radius
            k: float = None,  # force constant
            comment: str = None,
            **kwargs):
        super()._init(**kwargs)
        self.ai, self.funct, self.g = ai, funct, g
        self.r, self.k, self.comment = r, k, comment

    def raise_not_impl(self):
        raise NotImplementedError(f"PositionRestraints funct {self.funct} ({self.funct.name}) is not implemented.")

    def _parse_line(self):
        fields = self._fields
        self.ai, self.funct, self.g, self.r, self.k = int(fields[0]), int(fields[1]), int(fields[2]), float(
            fields[3]), float(fields[4])

    def cmp_key(self):
        return (self.ai, self.funct, self.g, self.r, self.k)

    def annotate(self) -> str:
        line = ";       ai        funct        g        r        k"
        return line

    def _str_impl(self):
        line = f"{self.ai:>10}{self.funct:>10}{self.g:>10}{self.r:>10}{self.k:>10}"
        return line


###############
# all records #
###############


class Records:

    def __init__(self):
        self.all: list[RecordText] = []

    def __iadd__(self, other):
        assert isinstance(other, Records)
        self.all.extend(other.all)
        return self

    @classmethod
    def from_file(cls, path: str, incdir: str, allow_unknown: bool):
        path = os.path.abspath(path)
        basename = os.path.basename(path)
        dirname = os.path.dirname(path) if incdir is None else incdir

        from_text = (
            RecordDefaults.from_text,
            RecordAtomType.from_text,
            ##
            RecordMoleculeType.from_text,
            RecordAtom.from_text,
            # RecordBondType.from_text,
            RecordBond.from_text,
            # RecordAngleType.from_text,
            RecordAngle.from_text,
            # RecordDihedralType.from_text,
            RecordDihedral.from_text,
            RecordSettle.from_text,
            RecordExclusion.from_text,
            # RecordPairType.from_text,
            RecordPair.from_text,
            RecordPairNb.from_text,
            # RecordConstraintType.from_text,
            # RecordConstraint.from_text,
            # RecordNonbondParam.from_text,
            RecordVirtualSite1.from_text,
            RecordVirtualSite2.from_text,
            RecordVirtualSite3.from_text,
            RecordVirtualSite4.from_text,
            RecordDihedralRestraints.from_text,
            RecordPositionRestraints.from_text,
            ##
            RecordSystem.from_text,
            RecordMolecule.from_text,
        )
        assert len(RecordSection.allowed_sections) == len(from_text)
        func_from_text = dict((k, v) for k, v in zip(RecordSection.allowed_sections, from_text))

        fs = cls()
        lineno = 0
        cur_section = None
        with open(path) as f:
            for raw in f:
                lineno += 1
                r0 = RecordText.from_text(text=raw)
                text = r0.text
                line = r0.line

                if text == "":
                    fs.all.append(r0)
                elif text.startswith("#include"):
                    r = RecordInclude.from_text(dirname=dirname,
                                                text=text)  # filter the comment after #include "file.path"
                    fs.all.append(r)
                    fs2 = cls.from_file(path=r.abspath, incdir=incdir, allow_unknown=allow_unknown)
                    fs += fs2
                elif text.startswith("#"):
                    if allow_unknown:
                        logger.warning(f"Ignoring {basename}:{lineno} {line}")
                    else:
                        raise RuntimeError(f"Cannot parse {basename}:{lineno} {line}")
                elif text.startswith("[ ") and text.endswith(" ]"):
                    r = RecordSection.from_text(allow_unknown=allow_unknown, text=line)
                    fs.all.append(r)
                    cur_section = r.section
                elif cur_section in ("atomtypes", "atoms"):
                    r = func_from_text[cur_section](text=line)
                    fs.all.append(r)
                elif cur_section in func_from_text.keys():  ### check here
                    r = func_from_text[cur_section](text=line)
                    fs.all.append(r)
                else:
                    if allow_unknown:
                        logger.warning(f"Ignoring {basename}:{lineno} {line}")
                    else:
                        raise RuntimeError(f"Cannot parse {basename}:{lineno} {line}")

        return fs
