# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from bytemol.units.simple_unit import A_to_nm

logger = logging.getLogger(__name__)

MAX_CONNECT = 4  # max num of connect per line
SOL_NAME = "SOL"
ATOM_NUM_THREASHOLD = 100000


class Component(Enum):
    protein = auto()
    ion = auto()
    water = auto()
    ligand = auto()


class LigandIonConnectException(Exception):
    pass


@dataclass
class PDBRecord(ABC):

    @abstractmethod
    def to_string(self):
        pass


@dataclass
class TERRecord(PDBRecord):
    res_name: str
    chain_id: str
    res_seq: int

    # other info
    component: Component = Component.protein
    # default
    record_name: str = "TER"
    insertion_code: str = " "

    def to_string(self, atom_idx) -> str:
        ter = f"{self.record_name:<6}{atom_idx:>5}{self.res_name:>9}{self.chain_id:>2}{self.res_seq:>4}{self.insertion_code}"
        assert len(ter) == 27
        return ter


@dataclass
class CONECTRecord(PDBRecord):
    at0: int
    at1: int
    at2: Optional[int] = None
    at3: Optional[int] = None
    at4: Optional[int] = None

    record_name: str = "CONECT"

    def to_string(self):
        conect = f"{self.record_name:<6}{self.at0:>5}{self.at1:>5}"
        for at in (self.at2, self.at3, self.at4):
            if at is not None:
                conect += f"{at:>5}"

        return conect


# eq is default to be True
@dataclass
class ATOMRecord(PDBRecord):
    atom_name: str
    res_name: str
    chain_id: str
    res_seq: int
    x: float
    y: float
    z: float
    occupancy: float
    temp_factor: float
    element: str
    charge: str

    # other info
    component: Component = None
    structure_type: str = None
    atname_updated: bool = False

    # default
    record_name: str = "ATOM"
    insertion_code: str = " "

    def _adjust_atname(self):

        if len(self.atom_name) == 4:
            return self.atom_name
        elif self.atom_name[0].isdigit():
            return self.atom_name.ljust(4)
        else:
            return " " + self.atom_name.ljust(3)

    def to_string(self, atom_idx) -> str:
        if self.atname_updated:
            atom_name = self._adjust_atname()
        else:
            atom_name = self.atom_name
        atom = f"{self.record_name:<6}{atom_idx:>5} {atom_name:<4}{self.res_name:>4}{self.chain_id:>2}{self.res_seq:>4}{self.insertion_code}{self.x:>11}{self.y:>8}{self.z:>8}{self.occupancy:>6}{self.temp_factor:>6}{self.element:>12}{self.charge}"
        assert len(atom) in (78, 79, 80), f"atom record '{atom}', len: {len(atom)}"

        return atom


@dataclass
class HETATMRecord(ATOMRecord):
    record_name: str = "HETATM"


@dataclass
class PDBData:
    all_records: list[PDBRecord] = field(default_factory=list)
    connect_info_set: set = field(default_factory=set)
    connect_info: OrderedDict[int, list[int]] = field(default_factory=OrderedDict)

    pdb_serial_number: list[int] = field(default_factory=list)  # real origin atom serial number with ter
    origin_protein_idx: list[int] = field(default_factory=list)  # 0-based
    origin_metal_idx: list[int] = field(default_factory=list)  # 0-based
    origin_water_idx: list[int] = field(default_factory=list)  # 0-based
    origin_ligand_idx: list[int] = field(default_factory=list)  # 0-based
    ter_indices: list[int] = field(default_factory=list)  # 0-based

    assign_components: bool = False

    @property
    def origin_protein_wo_ter_idx(self):
        return list(filter(lambda x: x not in self.ter_indices, self.origin_protein_idx))

    @property
    def protein_records(self):
        assert self.assign_components
        return [record for record in self.all_records if record.component == Component.protein]

    @property
    def protein_records_wo_ter(self):
        assert self.assign_components
        return [
            record for record in self.all_records
            if (record.component == Component.protein and not isinstance(record, TERRecord))
        ]

    @property
    def origin_water_wo_ter_idx(self):
        return list(filter(lambda x: x not in self.ter_indices, self.origin_water_idx))

    @property
    def water_records(self):
        assert self.assign_components
        return [record for record in self.all_records if record.component == Component.water]

    @property
    def water_records_wo_ter(self):
        assert self.assign_components
        return [
            record for record in self.all_records
            if (record.component == Component.water and not isinstance(record, TERRecord))
        ]

    @property
    def metal_records(self):
        assert self.assign_components
        return [record for record in self.all_records if record.component == Component.ion]

    @property
    def ligand_records(self):
        assert self.assign_components
        return [record for record in self.all_records if record.component == Component.ligand]

    def update_connect_info_set(self, connect_info_set: set):
        self.connect_info_set = connect_info_set

    def clear_connect_info(self):
        self.connect_info = OrderedDict()
        return self.connect_info


class BasePDBOutput(ABC):

    @abstractmethod
    def write_pdb(self):
        pass


def _write_struct_info(json_filename: Path, records: list[PDBRecord]) -> os.PathLike:
    """
    Write structural info of PDB ATOM/HETATM records to a JSON file.
    """
    assert json_filename.suffix == ".json"
    struct_info = {}
    # Make sure that atom numbers in struct_info are consecutive
    atom_idx = 1  # 1-based index for ATOM and HETATM records (excluding TER)
    for record_idx, record in enumerate(records, start=1):
        if isinstance(record, TERRecord):
            continue
        assert record.structure_type is not None
        struct_info[record_idx] = {
            "atom_idx_1_based": atom_idx,
            "element": record.element,
            "structure_type": record.structure_type
        }
        atom_idx += 1

    with open(json_filename, 'w') as f:
        json.dump(struct_info, f, indent=2)

    return str(json_filename)


class PureWaterOutput(BasePDBOutput):

    def __init__(self, pdb_data: PDBData, pdb_name: str):
        self.pdb_data = pdb_data
        self._pdb_filename: str = f"{pdb_name}_crystal_waters.pdb"
        self._gro_filename: str = f"{pdb_name}_crystal_waters.gro"
        self.records: list[PDBRecord] = self.pdb_data.water_records

    def write_gro(self, working_dir: str):
        working_dir = Path(working_dir).resolve()
        assert working_dir.exists()
        filename = working_dir / self._gro_filename

        gro_line_template = "{:5d}{:<5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}"

        water_records = self.pdb_data.water_records_wo_ter
        water_groups = [water_records[i:i + 3] for i in range(0, len(water_records), 3)]
        water_atom_names = ("OW", "HW1", "HW2")
        total_atoms = 0
        lines = []
        max_box = 0
        box_buffer = 1.0  #nm

        for res_idx, water_group in enumerate(water_groups, start=1):
            for water_record, at_name in zip(water_group, water_atom_names):
                total_atoms += 1
                xx = A_to_nm(water_record.x)
                yy = A_to_nm(water_record.y)
                zz = A_to_nm(water_record.z)
                max_box = max(max_box, abs(xx), abs(yy), abs(zz))
                val = (res_idx, SOL_NAME, at_name, total_atoms % ATOM_NUM_THREASHOLD, xx, yy, zz)
                line = gro_line_template.format(*val)
                assert len(line) == 44, f"wrong length of gro line: {line}"
                line += "\n"
                lines.append(line)

        assert total_atoms == len(water_records), f"{total_atoms} vs {len(water_records)}"
        with open(filename, 'w') as f:
            f.write(f"Generated by PDBParser\n{total_atoms:<}\n")
            f.writelines(lines)
            f.write(f"{max_box+box_buffer:>10.5}" * 3 + "\n")

        return str(filename)

    def write_pdb(self, working_dir: str):
        working_dir = Path(working_dir).resolve()
        assert working_dir.exists()
        filename = working_dir / self._pdb_filename
        logger.info(f"writing pure water info to pdb: {filename}")

        lines = []
        water_records = self.records
        for idx, water_record in enumerate(water_records, start=1):
            line = water_record.to_string(idx)
            line += '\n'
            lines.append(line)

        with open(filename, 'w') as f:
            f.writelines(lines)
            f.write("END\n")

        return str(filename)


class PureProteinOutput(BasePDBOutput):

    def __init__(self, pdb_data: PDBData, pdb_name: str):
        self.pdb_data = pdb_data
        self._pdb_filename: str = f"{pdb_name}_pure_protein.pdb"
        self._json_filename: str = f"{pdb_name}_pure_protein_struct_info.json"
        self._connect_set: set = set()
        self.records: list[PDBRecord] = self.pdb_data.protein_records
        self.origin_indices: list[int] = [
            self.pdb_data.pdb_serial_number[idx] for idx in self.pdb_data.origin_protein_idx
        ]  # 1-based

    @property
    def connect_set(self):
        return self._connect_set

    def write_struct_info(self, working_dir: str):
        working_dir = Path(working_dir).resolve()
        assert working_dir.exists()
        filename = working_dir / self._json_filename

        return _write_struct_info(filename, self.records)

    def write_pdb(self, working_dir: str):
        working_dir = Path(working_dir).resolve()
        assert working_dir.exists()
        filename = working_dir / self._pdb_filename
        logger.info(f"writing pure protein info to pdb: {filename}")

        lines = []
        origin_indices = self.origin_indices
        # prepare ATOM, HETATM, and TER records
        for idx, record in enumerate(self.records, start=1):
            lines.append(record.to_string(idx))

        # prepare CONECT records
        pure_protein_connect_set = set()
        for at0, bonded_ats in self.pdb_data.connect_info.items():
            connect_dict = {}
            if not at0 in origin_indices:
                continue
            # convert to new idx
            new_at0 = origin_indices.index(at0) + 1
            connect_dict["at0"] = new_at0
            for idx, at in enumerate(bonded_ats, start=1):
                if at not in origin_indices:
                    continue
                key_name = f"at{idx}"
                # convert to new_at
                new_at = origin_indices.index(at) + 1
                connect_dict[key_name] = new_at

            assert len(connect_dict) < 6, connect_dict

            if len(connect_dict) < 2:
                logger.warning(f"incomplete connect info, skip {connect_dict}")
                continue

            for at in connect_dict.values():
                if at == new_at0:
                    continue
                pure_protein_connect_set.add((new_at0, at))
            lines.append(CONECTRecord(**connect_dict).to_string())

        # write
        with open(filename, "w") as f:
            f.write("\n".join(lines))
            f.write("\nEND\n")

        self._connect_set = pure_protein_connect_set

        return str(filename)


class MetalloproteinOutput(BasePDBOutput):
    """
    This class is used to get amino acids, metal ions, and optional water components
    of a PDBData object and combine them into a new pdb file.
    """

    def __init__(self, pdb_data: PDBData, pdb_name: str, with_water: bool):
        self.pdb_data = pdb_data
        self._pdb_filename = f"{pdb_name}.pdb"
        self._json_filename = f"{pdb_name}_struct_info.json"
        self.records = self.pdb_data.protein_records + self.pdb_data.metal_records + (self.pdb_data.water_records
                                                                                      if with_water else [])
        indices = self.pdb_data.origin_protein_idx + self.pdb_data.origin_metal_idx + (self.pdb_data.origin_water_idx
                                                                                       if with_water else [])
        self.origin_indices: list[int] = [self.pdb_data.pdb_serial_number[idx] for idx in indices]  # 1-based
        self.metal_indices_1_based: list[int] = [
            self.pdb_data.pdb_serial_number[idx] for idx in self.pdb_data.origin_metal_idx
        ]  # 1-based

    def write_struct_info(self, working_dir: str):
        working_dir = Path(working_dir).resolve()
        assert working_dir.exists()
        filename = working_dir / self._json_filename

        return _write_struct_info(filename, self.records)

    def write_pdb(self, working_dir: os.PathLike) -> os.PathLike:
        working_dir = Path(working_dir).resolve()
        assert working_dir.exists()
        filename = working_dir / self._pdb_filename
        logger.info(f"writing metalloprotein info to pdb: {filename}")

        lines = []
        origin_indices = self.origin_indices

        # prepare ATOM, HETATM, and TER records
        for idx, record in enumerate(self.records, start=1):
            lines.append(record.to_string(idx))

        # prepare CONECT records
        for at0, bonded_ats in self.pdb_data.connect_info.items():
            if at0 not in origin_indices:
                continue
            # Remove metal-related bonds defined in the input PDB file
            if at0 in self.metal_indices_1_based:
                continue
            new_at0 = origin_indices.index(at0) + 1

            connect_info = {"at0": new_at0}
            for idx, at in enumerate(bonded_ats, start=1):
                if at not in origin_indices:
                    continue
                # Remove metal-related bonds defined in the input PDB file
                if at in self.metal_indices_1_based:
                    continue
                key = f"at{idx}"
                new_at = origin_indices.index(at) + 1
                connect_info[key] = new_at

            if len(connect_info) <= 1:  # No bonds for at0
                continue

            # split connect info
            connect_info2 = None
            if len(connect_info) > 5:
                logger.warning(f"atom {at0} with new_idx {new_at0} has more than 4 bond, please check")

                connect_info2 = {"at0": new_at0}
                for idx in range(1, len(connect_info) - MAX_CONNECT):
                    old_key = f"at{idx + MAX_CONNECT}"
                    new_key = f"at{idx}"
                    connect_info2[new_key] = connect_info.pop(old_key)

            lines.append(CONECTRecord(**connect_info).to_string())
            if connect_info2:
                lines.append(CONECTRecord(**connect_info2).to_string())

        with open(filename, 'w') as f:
            f.write("\n".join(lines))
            f.write("\nEND\n")

        return str(filename)


class FullPDBOutput(BasePDBOutput):
    """
    This class is used to export the PDBData object into a new pdb file.
    If there are discontinuities in the original indices extracted from original pdb data,
    this class will fix the discontinuity and update the indices.
    The output pdb contains all existing components in the original pdb data,
    including protein, ligand, metal, and water.
    """

    def __init__(self, pdb_data: PDBData, pdb_name: str):
        self.pdb_data = pdb_data
        self._pdb_filename: str = f"{pdb_name}_full.pdb"
        self._json_filename: str = f"{pdb_name}_full_struct_info.json"
        self.records = self.pdb_data.all_records
        indices = range(len(self.pdb_data.all_records))
        self.origin_indices: list[int] = [self.pdb_data.pdb_serial_number[idx] for idx in indices]  # 1-based

    def write_struct_info(self, working_dir: str):
        working_dir = Path(working_dir).resolve()
        assert working_dir.exists()
        filename = working_dir / self._json_filename

        return _write_struct_info(filename, self.records)

    def write_pdb(self, working_dir: os.PathLike) -> os.PathLike:
        working_dir = Path(working_dir).resolve()
        assert working_dir.exists()
        filename = working_dir / self._pdb_filename
        logger.info(f'writing full pdb info to pdb: {filename}')

        lines = []
        origin_indices = self.origin_indices

        # prepare all record
        for idx, record in enumerate(self.pdb_data.all_records, start=1):
            lines.append(record.to_string(idx))

        # prepare connect and update idx
        for at0, bonded_ats in self.pdb_data.connect_info.items():
            # if there are discontinuity in the original indices, fix the discontinuity and update indices

            new_at0 = origin_indices.index(at0) + 1

            connect_info = {"at0": new_at0}
            for idx, at in enumerate(bonded_ats, start=1):
                # if there are discontinuity in the original indices, fix the discontinuity and update indices
                if at not in origin_indices:
                    continue
                key = f"at{idx}"
                new_at = origin_indices.index(at) + 1
                connect_info[key] = new_at

            if len(connect_info) <= 1:  # No bonds for at0
                logger.warning(f"incomplete connect info, skip {connect_info}")
                continue

            # split connect info
            connect_info2 = None
            if len(connect_info) > 5:
                logger.warning(f"atom {at0} with new_idx {new_at0} has more than 4 bond, please check")
                connect_info2 = {"at0": new_at0}
                for idx in range(1, len(connect_info) - MAX_CONNECT):
                    old_key = f"at{idx + MAX_CONNECT}"
                    new_key = f"at{idx}"
                    connect_info2[new_key] = connect_info.pop(old_key)

            lines.append(CONECTRecord(**connect_info).to_string())
            if connect_info2:
                lines.append(CONECTRecord(**connect_info2).to_string())

        with open(filename, 'w') as f:
            f.write("\n".join(lines))
            f.write("\nEND\n")

        return str(filename)


class PureLigandOutput(BasePDBOutput):

    def __init__(self, pdb_data: PDBData, pdb_name: str):
        self.pdb_data = pdb_data
        self._pdb_filename = f"{pdb_name}_ligand.pdb"
        self.records: list[PDBRecord] = self.pdb_data.ligand_records
        indices = self.pdb_data.origin_ligand_idx
        self.original_indices: list[int] = [self.pdb_data.pdb_serial_number[idx] for idx in indices]  # 1-based

    def write_pdb(self, working_dir: str, detect_ligand_ion_connect: bool = True):
        working_dir = Path(working_dir).resolve()
        assert working_dir.exists()
        filename = working_dir / self._pdb_filename

        lines = []
        original_indices = self.original_indices

        # prepare all records
        for idx, record in enumerate(self.records, start=1):
            lines.append(record.to_string(idx))

        for at0, bonded_ats in self.pdb_data.connect_info.items():
            if at0 not in original_indices:
                continue
            new_at0 = original_indices.index(at0) + 1

            connect_info = {"at0": new_at0}
            for idx, at in enumerate(bonded_ats, start=1):
                if at not in original_indices:
                    at_serial = self.pdb_data.pdb_serial_number.index(at)
                    at_component = self.pdb_data.all_records[at_serial].component.name
                    at_name = self.pdb_data.all_records[at_serial].atom_name.strip()
                    try:
                        if at_component == "ion":
                            raise LigandIonConnectException(
                                f"[PDBParser] found covalent bonds between ligand and {at_component} ({at_name}) between atoms {at0} and {at}"
                            )
                        elif at_component == "protein":
                            raise RuntimeError(
                                f"[PDBParser] found covalent bonds between ligand and {at_component} ({at_name}) between atoms {at0} and {at}"
                            )
                    except Exception as e:
                        if isinstance(e, LigandIonConnectException) and not detect_ligand_ion_connect:
                            logger.warning(
                                f"[PDBParser] found covalent bonds between ligand and {at_component} ({at_name}) between atoms {at0} and {at}"
                            )
                        else:
                            raise RuntimeError(e) from e
                    continue
                key = f"at{idx}"
                new_at = original_indices.index(at) + 1
                connect_info[key] = new_at

            # split connect info
            connect_info2 = None
            if len(connect_info) > 5:
                logger.warning(f"atom {at0} with new_idx {new_at0} has more than 4 bond, please check")

                connect_info2 = {"at0": new_at0}
                for idx in range(1, len(connect_info) - MAX_CONNECT):
                    old_key = f"at{idx + MAX_CONNECT}"
                    new_key = f"at{idx}"
                    connect_info2[new_key] = connect_info.pop(old_key)
            if len(connect_info) < 2:
                logger.warning(f"incomplete connect info, skip {connect_info}")
                continue

            lines.append(CONECTRecord(**connect_info).to_string())
            if connect_info2:
                lines.append(CONECTRecord(**connect_info2).to_string())

        with open(filename, 'w') as f:
            f.write("\n".join(lines))
            f.write("\nEND\n")

        return str(filename)


class PureIonOutput(BasePDBOutput):

    def __init__(self, pdb_data: PDBData, pdb_name: str):
        self.pdb_data = pdb_data
        self._pdb_filename = f"{pdb_name}_Ion.pdb"
        self.records: list[PDBRecord] = self.pdb_data.metal_records
        indices = self.pdb_data.origin_metal_idx
        self.origin_indices: list[int] = [self.pdb_data.pdb_serial_number[idx] for idx in indices]  # 1-based

    def write_pdb(self, working_dir: str):
        working_dir = Path(working_dir).resolve()
        assert working_dir.exists()
        filename = working_dir / self._pdb_filename

        lines = []
        records = self.records

        # prepare all record
        for idx, record in enumerate(records, start=1):
            lines.append(record.to_string(idx))

        with open(filename, 'w') as f:
            f.write("\n".join(lines))
            f.write("\nEND\n")

        return str(filename)
