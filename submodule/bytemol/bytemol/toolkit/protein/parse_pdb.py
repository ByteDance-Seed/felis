# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from typing import Optional

from networkx import get_node_attributes

from bytemol.toolkit.protein.helper.parse_residue_lib import get_amber_protein_res_template, get_ion_solvent_template
from bytemol.toolkit.protein.helper.pdb_output import (
    ATOMRecord,
    Component,
    FullPDBOutput,
    HETATMRecord,
    PDBData,
    PDBRecord,
    PureIonOutput,
    PureLigandOutput,
    PureProteinOutput,
    PureWaterOutput,
    TERRecord,
)
from bytemol.toolkit.protein.sanitizer import PureProteinSanitizer, PureWaterSanitizer
from bytemol.utils import temporary_cd

logger = logging.getLogger(__name__)

RESIDUE_ALT_NAMES = ['HIS', 'NMA']
SOLVENT_MOLECULES = ['HOH', 'WAT', 'TIP3', 'SPC', 'SOL']
RESTEMP_LIB = get_amber_protein_res_template()
ION_SOL_LIB = get_ion_solvent_template()
ACCEPTABLE_IONS = set([
    list(get_node_attributes(val['resGraph'], 'element').values())[0]
    for key, val in ION_SOL_LIB.items()
    if key not in SOLVENT_MOLECULES
])
TERMINAL_RES_NAMES = set([res for res, val in RESTEMP_LIB.items() if len(val['TailOrHead']) < 2])
CURRENTLY_ACCEPTABLE_PROTEIN_RESIDUES = list(RESTEMP_LIB.keys()) + RESIDUE_ALT_NAMES
# Ion residue names from openmmforcefields tip3p_standard.xml
IONS = [
    'AL', 'Ag', 'BA', 'BR', 'Be', 'CA', 'CD', 'CE', 'CL', 'CO', 'CR', 'CS', 'CU', 'Ce', 'Cr', 'Dy', 'EU', 'EU3', 'Er',
    'F', 'FE', 'FE2', 'GD3', 'HG', 'Hf', 'IN', 'IOD', 'K', 'LA', 'LI', 'LU', 'MG', 'MN', 'NA', 'NI', 'Nd', 'PB', 'PD',
    'PR', 'PT', 'Pu', 'RB', 'Ra', 'SM', 'SR', 'Sm', 'Sn', 'TB', 'Th', 'Tl', 'Tm', 'U4+', 'V2+', 'Y', 'YB2', 'ZN', 'Zr'
]

# NOTE: N-terminal residues to quick support TER in pdb file under ff14sb
NTERM_RESIDUES = [
    'NALA', 'NARG', 'NASN', 'NASP', 'NCYS', 'NCYX', 'NGLN', 'NGLU', 'NGLY', 'NHID', 'NHIE', 'NHIP', 'NILE', 'NLEU',
    'NLYS', 'NMET', 'NPHE', 'NPRO', 'NSER', 'NTHR', 'NTRP', 'NTYR', 'NVAL', "ACE"
]
NTERM_NUM_H = {
    res: list(get_node_attributes(RESTEMP_LIB[res]['resGraph'], "element").values()).count('H') for res in NTERM_RESIDUES
}


class OXTMissingException(Exception):
    pass


class HMissingException(Exception):
    pass


@dataclass
class ProteinCanonicalOutput:
    full_pdb_file: os.PathLike
    amino_acid_file: os.PathLike
    ion_file: Optional[os.PathLike]
    ligand_file: Optional[os.PathLike]
    water_file: Optional[os.PathLike]

    def get_non_aa_components(self) -> dict[str, os.PathLike]:
        return {
            k: v for k, v in asdict(self).items() if v is not None and k not in ('full_pdb_file', 'amino_acid_file')
        }


def _try_convert(value, cast_type, default=None):
    """
    Adapted from ParmEd/formats/pdb.py.
    Cast the input value to desired type.
    Parameters
    ----------
    value: input value to be cast into the target type
    cast_type: target type for the output

    Return
    ------
    Input value with the target type
    """
    try:
        return cast_type(value)
    except ValueError:
        return default


class PDBParser:
    """
    Parse a pdb file to each components including protein, metal, water and ligand
    """

    def __init__(self, pdb_file: Path, check_names: bool):
        self.pdb_file: Path = pdb_file
        self.root_dir: Path = self.pdb_file.parent
        self.pdb_name: str = pdb_file.stem
        self.pdb_data: PDBData = PDBData()

        self.check_names: bool = check_names
        self.has_metal: bool = False
        self.has_water: bool = False
        self.has_ligand: bool = False

        # NOTE: compatible with intraTER pdb file, later solve intra missing residue with better way
        self.intra_ter_lids = list()  # zero-based

    @property
    def all_records(self):
        return self.pdb_data.all_records

    @property
    def pdb_serial_number(self):
        return self.pdb_data.pdb_serial_number

    @property
    def ter_indices(self):
        return self.pdb_data.ter_indices

    @property
    def connect_info_set(self):
        return self.pdb_data.connect_info_set

    @property
    def connect_info(self):
        return self.pdb_data.connect_info

    @property
    def origin_protein_idx(self):
        return self.pdb_data.origin_protein_idx

    @property
    def origin_protein_wo_ter_idx(self):
        return self.pdb_data.origin_protein_wo_ter_idx

    @property
    def origin_water_idx(self):
        return self.pdb_data.origin_water_idx

    @property
    def origin_water_wo_ter_idx(self):
        return self.pdb_data.origin_water_wo_ter_idx

    @property
    def origin_metal_idx(self):
        return self.pdb_data.origin_metal_idx

    @property
    def origin_ligand_idx(self):
        return self.pdb_data.origin_ligand_idx

    @property
    def protein_records(self):
        return self.pdb_data.protein_records

    @property
    def protein_records_wo_ter(self):
        return self.pdb_data.protein_records_wo_ter

    @property
    def water_records(self):
        return self.pdb_data.water_records

    @property
    def water_records_wo_ter(self):
        return self.pdb_data.water_records_wo_ter

    @property
    def metal_records(self):
        return self.pdb_data.metal_records

    @property
    def ligand_records(self):
        return self.pdb_data.ligand_records

    def _check_atom_idx(self, line: str):
        strip_line = line.strip()
        if len(strip_line) > 5:
            raise RuntimeError(f'[PDBParser] currenly does not support atomidx with extend pos: {strip_line}')
        try:
            return int(strip_line)
        except:  # pylint: disable=W0707
            raise RuntimeError(f'[PDBParser] currently cannot parse other atomidx format: {strip_line}')

    def _parse_atom_record(self, line: str):
        record = {
            "atom_idx": line[6:12],  # pop later
            "atom_name": line[12:16],
            "alt_loc": line[16].strip(),  # pop later
            "res_name": line[17:20].strip(),
            "chain_id": line[21].strip(),
            "res_seq": int(line[22:26].strip()),
            "insertion_code": line[26].strip(),
            "x": float(line[30:38].strip()),
            "y": float(line[38:46].strip()),
            "z": float(line[46:54].strip()),
            "occupancy": float(line[54:60].strip()),
            "temp_factor": float(line[60:66].strip()),
            "element": line[76:78].strip(),
            "charge": line[78:80].strip()
        }

        # check atom idx
        atom_idx = record.pop("atom_idx", None)
        assert atom_idx is not None
        atom_idx = self._check_atom_idx(atom_idx)
        self.pdb_serial_number.append(atom_idx)

        # check alternative structure
        alt_loc = record.pop("alt_loc", None)
        if alt_loc:
            raise RuntimeError(
                f"[PDBParser] currently does not support alternative structure at atom idx {atom_idx}, please clean pdb."
            )

        # check element
        if not record["element"]:
            raise RuntimeError(f"[PDBParser] atom element symbol missing at atom idx {atom_idx}, please check pdb")

        # check insertion_code
        if not record["insertion_code"]:
            record.pop("insertion_code")

        return record

    def _parse_atom(self, line: str, lid: int):

        try:
            record = self._parse_atom_record(line)
        except Exception as e:
            raise RuntimeError(f"[PDBParser] failed to parse atom record at line {lid}, {e}") from e

        self.all_records.append(ATOMRecord(**record))  # pylint: disable=E1123

    def _parse_hetatom(self, line: str, lid: int):

        try:
            record = self._parse_atom_record(line)
        except Exception as e:
            raise RuntimeError(f"[PDBParser] failed to parse hetatom record at line {lid}, {e}") from e

        self.all_records.append(HETATMRecord(**record))  # pylint: disable=E1123

    def _parse_ter(self, line: str, lid: int):
        """
        Process TER records.

        Residue names in TER records will not be checked.
        """
        try:
            record = {
                "atom_idx": line[6:12],  # pop later
                "res_name": line[17:20].strip(),
                "chain_id": line[21].strip(),
                "res_seq": int(line[22:26].strip()),
                "insertion_code": line[26].strip()
            }
        except Exception as e:
            raise RuntimeError(f"[PDBParser] failed to parse ter record at line {lid}, {e}") from e

        # check atom idx
        atom_idx = record.pop("atom_idx", None)
        atom_idx = self._check_atom_idx(atom_idx)

        # check insertion_code
        if not record["insertion_code"]:
            record.pop("insertion_code")

        # check corresponding component
        resname_in_ter = record["res_name"]
        if resname_in_ter:
            if resname_in_ter in CURRENTLY_ACCEPTABLE_PROTEIN_RESIDUES:
                record['component'] = Component.protein
            elif resname_in_ter in IONS:
                record['component'] = Component.ion
            elif resname_in_ter in SOLVENT_MOLECULES:
                record['component'] = Component.water
            else:
                # NOTE: this part may not cover all cases.
                record['component'] = Component.ligand
        else:
            record['component'] = Component.protein  # to cover the case with empty res_name

        self.all_records.append(TERRecord(**record))  # pylint: disable=E1123

        # update last_ter_idx
        # actutally here we cannot use the atomidx as the ter index
        # since atomidx may not be continuous
        self.ter_indices.append(len(self.all_records) - 1)
        self.pdb_serial_number.append(atom_idx)

    def _parse_connect(self, line: str, lid: int):
        # Read CONECT record in pdb file.
        # According to the pdb file format: https://www.wwpdb.org/documentation/file-format-content/format33/sect10.html, each CONECT record contains the connectivity of one atom (origin_idx) with up to 4 other atoms (idx_1 to idx_4).
        # Note: the atom indices should NOT be read by splitting the line, because when atom index has 5 digits it might not be separated by space with other atom indices.
        origin_idx = _try_convert(line[6:11], int)
        idx_1 = _try_convert(line[11:16], int)
        idx_2 = _try_convert(line[16:21], int)
        idx_3 = _try_convert(line[21:26], int)
        idx_4 = _try_convert(line[26:31], int)
        if origin_idx is None or idx_1 is None:
            raise ValueError(
                f'[PDBParser] Invalid CONECT record: not enough atom indices in the following line {lid}:\n{line}')

        if origin_idx not in self.pdb_serial_number:
            logger.warning(f'[PDBParser] unseen atom serial {origin_idx} found in line: "{line.strip()}"')

        for at in (idx_1, idx_2, idx_3, idx_4):
            if not at:
                continue
            if at not in self.pdb_serial_number:
                logger.warning(f'[PDBParser] unseen atom serial {origin_idx} found in line: "{line.strip()}"')
            # check ligand bonds
            if (at in self.origin_ligand_idx and
                    origin_idx not in self.origin_ligand_idx) or (origin_idx in self.origin_ligand_idx and
                                                                  at not in self.origin_ligand_idx):
                raise ValueError(
                    f"[PDBParser] found covalent bond between ligand and other components. please check {origin_idx} and {at}"
                )
            # add all exist connect bond including ion
            bond = tuple(sorted((origin_idx, at)))
            self.connect_info_set.add(bond)

    def _check_protein_seg(self, segment: list[PDBRecord]):
        chain_id = set([record.chain_id for record in segment])
        res_names = set([record.res_name for record in segment])
        assert len(
            chain_id) == 1, f"Please double check PDB, unlawful due to multiple chain ids without TER: {chain_id}"
        for res_name in res_names:
            assert res_name in CURRENTLY_ACCEPTABLE_PROTEIN_RESIDUES, f"Please double check PDB with unsupported res_name: {res_name}"

        # init component
        for record in segment:
            record.component = Component.protein

    def _check_nonprotein_seg(self, segment: list[PDBRecord]):
        # no need to check chain id
        for i, record in enumerate(segment):
            res_name = record.res_name
            prev_res = segment[i - 1] if i > 0 else None
            if res_name in CURRENTLY_ACCEPTABLE_PROTEIN_RESIDUES:
                # current segment may be a protein residue starting with an unsupported residue, or missing TER records between different components
                raise ValueError(
                    f"Please double check PDB: an unsupported residue {prev_res.res_name} is followed by a protein residue {res_name} on chain {record.chain_id}. Please add TER record in between if {prev_res.res_name} is a non-protein component."
                )
            elif res_name in IONS:
                record.component = Component.ion
            elif res_name in SOLVENT_MOLECULES:
                record.component = Component.water
            else:
                # NOTE: this part may not cover all cases.
                record.component = Component.ligand

    def _check_segment(self, prev_ter_idx: int, cur_ter_idx: int, records: list[PDBRecord]):
        logger.info(f"start to check records between {prev_ter_idx} and {cur_ter_idx}")
        segment = records[prev_ter_idx:cur_ter_idx]
        logger.info(f"the last record is {segment[-1]}")
        if isinstance(segment[-1], TERRecord):  # deal with the last is TER
            segment.pop()
        if segment[0].res_name in CURRENTLY_ACCEPTABLE_PROTEIN_RESIDUES:
            # protein
            self._check_protein_seg(segment)
        else:
            # other components (water, ion, ligand)
            self._check_nonprotein_seg(segment)

    def _init_check_components(self):

        all_records = self.all_records
        ter_indices = deepcopy(self.ter_indices)
        ter_indices = list(map(lambda x: x + 1, ter_indices))
        assert len(all_records) > 0 and len(ter_indices) >= 0
        if len(ter_indices) == 0:
            # under this situation, we only accept that one chain protein
            # which means all records chain id should be the same and res name in the supported protein res
            self._check_protein_seg(all_records)

        else:
            # since we have ter, we want to make sure within each segment
            # if it is a protein chain, can only have one chain id
            # otherwise this segment cannot have protein residue
            previous_ter_idx = 0
            # deal with one ter
            if ter_indices[-1] != len(all_records):
                ter_indices.append(len(all_records))
            for ter_idx in ter_indices:
                self._check_segment(previous_ter_idx, ter_idx, all_records)
                previous_ter_idx = ter_idx

        self.pdb_data.assign_components = True

    def _prepare_component_idx(self):

        for idx, record in enumerate(self.all_records):
            component = record.component
            assert component is not None, f"record missing component at idx {idx}"
            if component == Component.protein:
                self.origin_protein_idx.append(idx)
            elif component == Component.ion:
                self.origin_metal_idx.append(idx)
                if not self.has_metal:
                    self.has_metal = True
            elif component == Component.water:
                self.origin_water_idx.append(idx)
                if not self.has_water:
                    self.has_water = True
            elif component == Component.ligand:
                self.origin_ligand_idx.append(idx)
                if not self.has_ligand:
                    self.has_ligand = True
            else:
                raise ValueError("Unknown component type!")

        # additional check
        num_protein_atoms = len(self.origin_protein_idx)
        logger.info(f"found {num_protein_atoms} protein atoms")
        num_metal_atoms = len(self.origin_metal_idx)
        logger.info(f"found {num_metal_atoms} metal atoms")
        num_water_atoms = len(self.origin_water_idx)
        logger.info(f"found {num_water_atoms} water atoms")
        num_lig_atoms = len(self.origin_ligand_idx)
        logger.info(f"found {num_lig_atoms} lig atoms")

        total_records = num_protein_atoms + num_metal_atoms + num_water_atoms + num_lig_atoms
        assert total_records == len(
            self.all_records), f"from component: {total_records} vs parsed: {len(self.all_records)}"

        # inital check water atom numbers
        num_water_atom_records = len(self.origin_water_wo_ter_idx)
        assert num_water_atom_records % 3 == 0, "Please double check PDB, possible missing water atoms."

    def _check_intra_missing_residue(self, lines: list[str]):

        prev_res_infos = []
        next_res_infos = []
        for intra_ter_lid in self.intra_ter_lids:
            prev_line = lines[intra_ter_lid - 1]
            prev_cid = prev_line[21].strip()
            prev_res_seq = int(prev_line[22:26].strip())
            prev_res_name = prev_line[17:20].strip()

            next_line = lines[intra_ter_lid + 1]
            next_cid = next_line[21].strip()
            next_res_seq = int(lines[intra_ter_lid + 1][22:26].strip())
            next_res_name = next_line[17:20].strip()

            logger.info(
                f"[PDBParser] found missing residue between {(prev_cid, prev_res_seq)} and {(next_cid, next_res_seq)}")

            prev_res_infos.append((prev_cid, prev_res_seq, prev_res_name))
            next_res_infos.append((next_cid, next_res_seq, next_res_name))

        grouped_records = {
            key: list(group)
            for key, group in groupby(self.all_records, key=lambda record: (record.chain_id, record.res_seq))
        }

        # check oxt exist in the prev residue (C-TERM)
        # NOTE: here we assume that the prev residue is capped by oxt after meastro
        oxt_missing_failed = []
        dup_atomname_failed = []
        for (cid, res_seq, _) in prev_res_infos:
            assert (cid, res_seq) in grouped_records.keys()
            records = grouped_records[(cid, res_seq)]
            records_atom_names = [record.atom_name.strip() for record in records]
            try:
                if "OXT" not in records_atom_names:
                    raise OXTMissingException(f"[PDBParser] OXT missing in the prev residue ({cid} {res_seq})")
                assert len(records_atom_names) == len(set(records_atom_names))
            except Exception as e:
                if isinstance(e, OXTMissingException):
                    logger.warning(f"[PDBParser] OXT missing in the residue ({cid} {res_seq})")
                    oxt_missing_failed.append((cid, res_seq))
                else:
                    atom_name_counter = Counter(records_atom_names)
                    dup = [atom_name for atom_name, count in atom_name_counter.items() if count > 1]
                    logger.warning(f"[PDBParser] found duplicated atom name {dup} in the residue ({cid} {res_seq})")
                    dup_atomname_failed.append((cid, res_seq))

        # check next residue (N-TERM)
        missing_H_failed = []
        unsupport_res_failed = []
        for (cid, res_seq, res_name) in next_res_infos:
            assert (cid, res_seq) in grouped_records.keys()
            records = grouped_records[(cid, res_seq)]
            nterm_res_name = "N" + res_name if res_name != "ACE" else res_name

            # step1: check res name is nterminal residue
            try:
                assert nterm_res_name in NTERM_RESIDUES
            except AssertionError:
                logger.warning(f"[PDBParser] found unsupported N-term residue ({cid} {res_seq} {res_name})")
                unsupport_res_failed.append((cid, res_seq, res_name))
                continue

            num_record_Hs = 0
            tgt_num_Hs = NTERM_NUM_H[nterm_res_name]
            record_atom_names = set()
            H_records = []
            dup_atomnames = []
            dup_atomname_elements = set()
            for record in records:
                if record.element == "H":
                    num_record_Hs += 1
                    H_records.append(record)
                atom_name = record.atom_name.strip()
                if atom_name in record_atom_names:
                    dup_atomnames.append(atom_name)
                    dup_atomname_elements.add(record.element)
                record_atom_names.add(atom_name)

            try:
                # step2: check any missing H
                if num_record_Hs != tgt_num_Hs:
                    raise HMissingException(f"[PDBParser] missing H in the N-term residue ({cid} {res_seq} {res_name})")

                # step3: check any dup atom_names
                assert len(dup_atomnames) == 0
            except Exception as e:
                if isinstance(e, HMissingException):
                    logger.warning(f"[PDBParser] missing H in the N-term residue ({cid} {res_seq} {res_name})")
                    missing_H_failed.append((cid, res_seq, res_name))
                else:
                    logger.warning(
                        f"[PDBParser] found duplicated atom name {dup_atomnames} in the N-term residue ({cid} {res_seq} {res_name})"
                    )
                    dup_atomname_failed.append((cid, res_seq, res_name))

                # rename H only if the dup element is H and no missing H is found
                if len(dup_atomname_elements) == 1 and list(dup_atomname_elements)[0] == "H" and len(
                        missing_H_failed) == 0:
                    for idx, record in enumerate(H_records, start=1):
                        old_atom_name = record.atom_name.strip()
                        new_atom_name = f" H{idx}"  # NOTE: the space to align with other atom names
                        logger.warning(
                            f"[PDBParser] rename atom name from {old_atom_name} to {new_atom_name} for N-term residue ({cid} {res_seq} {res_name})"
                        )
                        record.atom_name = new_atom_name

        logger.info("[PDBParser] check and update intra-TER done!")
        missing_flag = 0
        if len(oxt_missing_failed):
            missing_flag += 1
            logger.info(f"[PDBParser] found {len(oxt_missing_failed)} OXT missing, please fix: {oxt_missing_failed}.")
        if len(missing_H_failed):
            missing_flag += 1
            logger.info(f"[PDBParser] found {len(missing_H_failed)} missing H, please fix: {missing_H_failed}.")
        if len(unsupport_res_failed):
            missing_flag += 1
            logger.info(
                f"[PDBParser] found {len(unsupport_res_failed)} unsupported N-term residue, please fix: {unsupport_res_failed}."
            )
        if len(dup_atomname_failed):
            logger.info(
                f"[PDBParser] found {len(dup_atomname_failed)} duplicated atom name, please fix: {dup_atomname_failed}."
            )

        if missing_flag > 0:
            raise RuntimeError("[PDBParser] found missing OXT or missing H or unsupport_res_failed, please fix!")

    def _parse(self):
        method_dispatch = {
            "ATOM": self._parse_atom,
            "CONECT": self._parse_connect,
            "HETATM": self._parse_hetatom,
            "TER": self._parse_ter
        }

        with open(self.pdb_file, 'r') as f:
            lines = f.readlines()

        for lid, line in enumerate(lines):
            rec = line[:6].strip()
            if rec == "MODEL":
                raise ValueError("[PDBParser] multi-frames pdb is not supported!")
            if rec == "TER" and len(line.strip()) == 3:
                logger.info(f"[PDBParser] found intra-chain TER at line_id {lid}. Skip!")
                self.intra_ter_lids.append(lid)
                continue
            if rec in method_dispatch:
                method_dispatch[rec](line, lid)

        assert len(self.all_records) == len(
            self.pdb_serial_number), f"{len(self.all_records)} vs {len(self.pdb_serial_number)}"

        #check if there are duplicated pdb serial numbers
        pdb_serial_number_set = set(self.pdb_serial_number)
        serial_number_duplicates = [item for item in pdb_serial_number_set if self.pdb_serial_number.count(item) > 1]
        assert len(serial_number_duplicates) == 0, f"Found duplicated serial number in pdb: {serial_number_duplicates}"

        # check if there are discontinuity in the pdb serial number
        serial_n = self.pdb_data.pdb_serial_number
        break_point = [serial_n[i] for i in range(len(serial_n) - 1) if serial_n[i + 1] - serial_n[i] != 1]
        if len(break_point) != 0:
            logger.warning(
                f"[PDBParser] found discontinuity in the pdb serial number at indices: {', '.join(map(str, break_point))} "
            )

        if self.intra_ter_lids:
            logger.info("[PDBParser] check and update intra missing residue")
            self._check_intra_missing_residue(lines)

        # initialize & check record component
        self._init_check_components()

        # prepare original component serial idx
        self._prepare_component_idx()

        # sort conect set
        connect_info = sorted(self.connect_info_set)
        for bond in connect_info:
            at1, at2 = bond
            if at1 not in self.connect_info.keys():
                self.connect_info[at1] = [at2]
            else:
                if at2 not in self.connect_info[at1]:
                    self.connect_info[at1].append(at2)

        # sort val
        for val in self.connect_info.values():
            val.sort()

    def _update_connect_info(self, full_connect_info: set, record_type: Component):

        indices = None
        if record_type == Component.protein:
            indices = self.origin_protein_idx
        elif record_type == Component.water:
            indices = self.origin_water_idx

        assert indices is not None

        origin_indices = [self.pdb_serial_number[idx] for idx in indices]

        map_origin_set = set()
        for bond in full_connect_info:
            at0, at1 = bond
            origin_at0 = origin_indices[at0 - 1]
            origin_at1 = origin_indices[at1 - 1]

            map_origin_set.add(tuple(sorted((origin_at0, origin_at1))))

        connect_info_set = self.connect_info_set.union(map_origin_set)
        self.pdb_data.update_connect_info_set(connect_info_set)

        full_connect_info = sorted(connect_info_set)
        connect_info = self.pdb_data.clear_connect_info()
        for bond in full_connect_info:
            at0, at1 = bond
            if at0 not in connect_info.keys():
                connect_info[at0] = [at1]
            else:
                if at1 not in connect_info[at0]:
                    connect_info[at0].append(at1)

        # sort
        for val in connect_info.values():
            val.sort()

    def _update_resname_and_atomname(self, replace_names: dict):

        records = self.protein_records_wo_ter
        origin_indices = [self.pdb_serial_number[idx] for idx in self.origin_protein_wo_ter_idx]
        assert len(replace_names) == len(records) == len(
            origin_indices), f"{len(replace_names)} vs {len(records)} vs {len(origin_indices)}"

        for (serial_num, replace_name), record, origin_idx in zip(replace_names.items(), records, origin_indices):
            assert serial_num > 0, serial_num
            replace_res, replace_element = replace_name
            # origin_serial_num = self.pdb_data.map_idx_origin(serial_num, records)
            logger.info(f"[PDBParser] update atom {origin_idx} resname to {replace_res}, element to {replace_element}")
            record.res_name = replace_res
            record.atom_name = replace_element
            record.atname_updated = True

    def _update_structure_info(self, structure_info: dict, record_type: Component):

        indices = None
        records = None
        if record_type == Component.protein:
            indices = self.origin_protein_wo_ter_idx
            records = self.protein_records_wo_ter
        elif record_type == Component.water:
            indices = self.origin_water_wo_ter_idx
            records = self.water_records_wo_ter

        assert indices is not None and records is not None

        origin_indices = [self.pdb_serial_number[idx] for idx in indices]

        assert len(origin_indices) == len(records) == len(
            structure_info), f"{len(origin_indices)} vs {len(records)} vs {len(structure_info)}"
        for (serial_num, st), record in zip(structure_info.items(), records):
            assert serial_num > 0, serial_num
            assert record.element == st["element"], f"{record.element} vs {st['element']}"
            assert record.structure_type is None
            record.structure_type = st["structure_type"]

    def _sanitize_pure_protein(self, keep_debug: bool = False):

        with temporary_cd(self.root_dir):
            try:
                # step1: dump the pure protein pdb
                pure_protein_output = PureProteinOutput(self.pdb_data, self.pdb_name)
                # NOTE will write to testdata folder for unittests
                pure_protein_pdb_file = pure_protein_output.write_pdb(".")
                pure_protein_connect_set = pure_protein_output.connect_set

                # step2: sanitize and update
                pure_protein_sanitizer = PureProteinSanitizer(pure_protein_pdb_file,
                                                              pure_protein_connect_set,
                                                              check_names=self.check_names)
                full_connect_info = pure_protein_sanitizer.get_full_conect()
                pure_protein_struture_info = pure_protein_sanitizer.structure_info

                # step3: update connect info & strucure info
                self._update_connect_info(full_connect_info, Component.protein)
                self._update_structure_info(pure_protein_struture_info, Component.protein)

                # step5 (optional): update res name & atom
                if self.check_names:
                    self._update_resname_and_atomname(pure_protein_sanitizer.replace_names)

                logger.info(f"the value of keep_debug: {keep_debug}")
                if not keep_debug:
                    os.remove(pure_protein_pdb_file)
                    logger.info(f"{pure_protein_pdb_file} is removed!")
            except Exception as e:
                if isinstance(e, OSError):
                    logger.warning(f"failed to remove the file {pure_protein_pdb_file}.")
                else:
                    logger.warning(f"failed to sanitize pure protein due to {e}.")
                    raise e

    def _sanitize_metal(self):

        for idx in self.origin_metal_idx:
            metal_record = self.all_records[idx]
            if not isinstance(metal_record, TERRecord):  # skip TER record for element check
                # For example, fe, FE, and Fe are all acceptable
                assert metal_record.element.capitalize(
                ) in ACCEPTABLE_IONS, f"Please double check PDB at idx {idx} due to unacceptable ion element: {metal_record.element}"
                metal_record.structure_type = "ion"

    def _sanitize_water(self, keep_debug: bool = False):

        with temporary_cd(self.root_dir):
            try:
                pure_water_output = PureWaterOutput(self.pdb_data, self.pdb_name)
                pure_water_pdb_filename = pure_water_output.write_pdb(".")

                water_sanitizer = PureWaterSanitizer(pure_water_pdb_filename)
                water_connect_info = water_sanitizer.get_full_conect()
                water_structure_info = water_sanitizer.structure_info

                # update connect & structure_info
                self._update_connect_info(water_connect_info, Component.water)
                self._update_structure_info(water_structure_info, Component.water)

                if not keep_debug:
                    os.remove(pure_water_pdb_filename)
                    logger.info(f"{pure_water_pdb_filename} is removed!")
            except Exception as e:
                if isinstance(e, OSError):
                    logger.warning(f"failed to remove the file {pure_water_pdb_filename}.")
                else:
                    logger.warning(f"failed to sanitize water due to {e}.")
                    raise e

    @classmethod
    def parse_pdb(cls, pdb_file: str, *, check_names: bool = False, keep_debug: bool = False) -> "PDBParser":
        """
        classmethod to parse a protein pdb file into different components based on ATOM, HETATM, TER, CONECT records
        
        Parameters:
        ----------
        pdb_file: str
            the path to protein pdb file
        
        check_names: bool(optional), default `False`
            if `True`, update residue name and atom name for protonated residues. Mainly for ProPka
            
        keep_debug: bool(optional), default `False`
            if `True`, keep debug files along the process

        Returns:
        ----------
        pdb_inst: PDBParser
            the class instance 
        """

        pdb_file = Path(pdb_file).resolve()
        assert pdb_file.suffix == ".pdb" and pdb_file.exists(), pdb_file
        logger.info(f"[PDBParser] start to parse pdb at {pdb_file}")

        pdb_inst = cls(pdb_file, check_names)

        # parse and finish assign components
        pdb_inst._parse()

        # sanitize and update pure protein
        pdb_inst._sanitize_pure_protein(keep_debug)

        # sanitize and update metal (optional)
        if pdb_inst.has_metal:
            pdb_inst._sanitize_metal()

        # sanitize and udpate sol (optional)
        if pdb_inst.has_water:
            pdb_inst._sanitize_water(keep_debug)

        return pdb_inst

    def canonicalize(self, working_dir: str) -> ProteinCanonicalOutput:
        """
        Parse and canonicalize the protein pdb file into several components.
        
        Parameters:
        ----------
        working_dir: str
            the path to store pdb_file
            
        Returns:
        ----------
        A ProteinCanonicalOutput object containing the following components:
            - full_pdb_file: str
                the path to store the full pdb
            - amino_acid_file: str
                the path to store amino acids
            - ligand_file: str
                the path to store ligands
            - ion_file: str
                the path to store metal ions
            - water_file: str
                the path to store crystal waters
        """
        full_pdb_output = FullPDBOutput(self.pdb_data, self.pdb_name)
        full_pdb_file = full_pdb_output.write_pdb(working_dir)
        pure_protein_output = PureProteinOutput(self.pdb_data, self.pdb_name)
        amino_acid_file = pure_protein_output.write_pdb(working_dir)
        ligand_file = PureLigandOutput(self.pdb_data, self.pdb_name).write_pdb(working_dir) if self.has_ligand else None
        ion_file = PureIonOutput(self.pdb_data, self.pdb_name).write_pdb(working_dir) if self.has_metal else None
        water_file = PureWaterOutput(self.pdb_data, self.pdb_name).write_pdb(working_dir) if self.has_water else None

        return ProteinCanonicalOutput(
            full_pdb_file=full_pdb_file,
            amino_acid_file=amino_acid_file,
            ligand_file=ligand_file,
            ion_file=ion_file,
            water_file=water_file,
        )
