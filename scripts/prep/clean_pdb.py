# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse

import gemmi

from bytemol.toolkit.protein.parse_pdb import CURRENTLY_ACCEPTABLE_PROTEIN_RESIDUES, _try_convert
from bytemol.utils import setup_default_logging

logger = setup_default_logging()


def _parse_pdb_serial(line: str) -> int | None:
    if line.startswith(("ATOM  ", "HETATM", "TER   ")):
        s = line[6:12].strip()
        if not s:
            return None
        return int(s)
    return None


def _replace_pdb_serial(line: str, new_serial: int) -> str:
    return line[:6] + f"{new_serial:5d}" + line[11:]


def _format_ter_line(serial: int, res_name: str, chain_id: str, res_seq: int, insertion_code: str) -> str:
    chain_char = chain_id[:1] if chain_id else " "
    icode_char = insertion_code[:1] if insertion_code else " "
    res = (res_name or "").strip()[:3].rjust(3)
    # refer to bytemol TERRecord.to_string()
    return f"TER   {serial:>5}{res:>9}{chain_char:>2}{res_seq:>4}{icode_char}\n"


def insert_ter(pdb_path: str) -> None:
    """
    Fix gemmi's pdb output bug by inserting TER records between disconnected segments.

    For ATOM->ATOM transitions within the same chain, TER insertion is decided by backbone bond distance
    (prefer C(prev)-N(next), fallback CA(prev)-CA(next)).
    For ATOM->HETATM/HETATM->ATOM transitions, TER insertion is decided by whether HETATM is a protein resname and if the chain id differs or resid discontinued.
    """
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    out: list[str] = []
    serial_map: dict[int, int] = {}
    inserted = 0

    last_primary: str | None = None
    last_hetatm_res_name = ""
    last_hetatm_chain_id = ""
    last_hetatm_res_seq = 0
    last_hetatm_icode = ""

    last_atom_res_name = ""
    last_atom_chain_id = ""
    last_atom_res_seq = 0
    last_atom_icode = ""
    last_atom_res_uid: tuple[str, int, str] | None = None
    last_atom_backbone: dict[str, tuple[float, float, float]] = {}

    for line in lines:
        rec = line[:6]

        if (rec == "ATOM  " and last_primary == "HETATM") or (rec == "HETATM" and last_primary == "ATOM  "):
            if rec == "HETATM":
                het_res_name = line[17:20].strip().upper()
                het_chain_id = line[21:22]
                het_res_seq = int(line[22:26].strip() or "0")

                atom_chain_id = last_atom_chain_id
                atom_res_seq = last_atom_res_seq

                ter_res_name = last_atom_res_name
                ter_chain_id = last_atom_chain_id
                ter_res_seq = last_atom_res_seq
                ter_icode = last_atom_icode
            else:
                het_res_name = (last_hetatm_res_name or "").strip().upper()
                het_chain_id = last_hetatm_chain_id
                het_res_seq = last_hetatm_res_seq

                atom_chain_id = line[21:22]
                atom_res_seq = int(line[22:26].strip() or "0")

                ter_res_name = last_hetatm_res_name
                ter_chain_id = last_hetatm_chain_id
                ter_res_seq = last_hetatm_res_seq
                ter_icode = last_hetatm_icode

            is_protein_hetatm = het_res_name in CURRENTLY_ACCEPTABLE_PROTEIN_RESIDUES or het_res_name == "TYS"
            need_ter = False
            if not is_protein_hetatm:
                need_ter = True
            else:
                if het_chain_id != atom_chain_id:
                    need_ter = True
                else:
                    if rec == "HETATM":
                        res_seq_gap = (het_res_seq > atom_res_seq + 1) or (het_res_seq < atom_res_seq)
                    else:
                        res_seq_gap = (atom_res_seq > het_res_seq + 1) or (atom_res_seq < het_res_seq)
                    if res_seq_gap:
                        need_ter = True
            if need_ter:
                old_serial = _parse_pdb_serial(line)
                if old_serial is None:
                    raise ValueError(f"{pdb_path}: {rec} record {line} has no serial number")

                ter_serial = old_serial + inserted
                out.append(_format_ter_line(ter_serial, ter_res_name, ter_chain_id, ter_res_seq, ter_icode))
                inserted += 1

        if rec == "ATOM  " and last_primary == "ATOM  ":
            cur_chain_id = line[21:22]
            cur_res_seq = int(line[22:26].strip() or "0")
            cur_icode = line[26:27]
            cur_res_uid = (cur_chain_id, cur_res_seq, cur_icode)
            cur_atom_name = line[12:16].strip()

            x = _try_convert(line[30:38], float)
            y = _try_convert(line[38:46], float)
            z = _try_convert(line[46:54], float)
            cur_xyz = None if (x is None or y is None or z is None) else (float(x), float(y), float(z))

            if last_atom_res_uid is not None and cur_chain_id == last_atom_chain_id and cur_res_uid != last_atom_res_uid:
                need_ter = False
                if cur_xyz is not None:
                    if cur_atom_name == "N" and "C" in last_atom_backbone:
                        px, py, pz = last_atom_backbone["C"]
                        dx = px - cur_xyz[0]
                        dy = py - cur_xyz[1]
                        dz = pz - cur_xyz[2]
                        need_ter = (dx * dx + dy * dy + dz * dz) > (1.8 * 1.8)
                    elif cur_atom_name == "CA" and "CA" in last_atom_backbone:
                        px, py, pz = last_atom_backbone["CA"]
                        dx = px - cur_xyz[0]
                        dy = py - cur_xyz[1]
                        dz = pz - cur_xyz[2]
                        need_ter = (dx * dx + dy * dy + dz * dz) > (4.5 * 4.5)
                    else:
                        need_ter = (cur_res_seq > last_atom_res_seq + 1) or (cur_res_seq < last_atom_res_seq)
                else:
                    need_ter = (cur_res_seq > last_atom_res_seq + 1) or (cur_res_seq < last_atom_res_seq)

                if need_ter:
                    old_serial = _parse_pdb_serial(line)
                    if old_serial is None:
                        raise ValueError(f"{pdb_path}: ATOM record {line} has no serial number")

                    ter_serial = old_serial + inserted
                    out.append(
                        _format_ter_line(
                            ter_serial,
                            last_atom_res_name,
                            last_atom_chain_id,
                            last_atom_res_seq,
                            last_atom_icode,
                        ))
                    inserted += 1

        if rec in ("ATOM  ", "HETATM", "TER   "):
            old_serial = _parse_pdb_serial(line)
            if old_serial is not None:
                new_serial = old_serial + inserted
                serial_map[old_serial] = new_serial
                line = _replace_pdb_serial(line, new_serial)
            else:
                raise ValueError(f"{pdb_path}: {rec} record {line} has no serial number")

        out.append(line)

        if rec == "HETATM":
            last_primary = "HETATM"
            last_hetatm_res_name = line[17:20].strip()
            last_hetatm_chain_id = line[21:22]
            last_hetatm_res_seq = int(line[22:26].strip() or "0")
            last_hetatm_icode = line[26:27]
        elif rec == "ATOM  ":
            last_primary = "ATOM  "
            cur_chain_id = line[21:22]
            cur_res_seq = int(line[22:26].strip() or "0")
            cur_icode = line[26:27]
            cur_res_uid = (cur_chain_id, cur_res_seq, cur_icode)

            if last_atom_res_uid is None or cur_res_uid != last_atom_res_uid:
                last_atom_backbone = {}

            atom_name = line[12:16].strip()
            if atom_name == "C" or atom_name == "CA":
                x = _try_convert(line[30:38], float)
                y = _try_convert(line[38:46], float)
                z = _try_convert(line[46:54], float)
                if x is not None and y is not None and z is not None:
                    last_atom_backbone[atom_name] = (float(x), float(y), float(z))

            last_atom_res_name = line[17:20].strip()
            last_atom_chain_id = cur_chain_id
            last_atom_res_seq = cur_res_seq
            last_atom_icode = cur_icode
            last_atom_res_uid = cur_res_uid
        elif rec in ("TER   ", "CONECT", "END   ", "END\n"):
            last_primary = rec

    if inserted == 0:
        return

    updated: list[str] = []
    for line in out:
        if not line.startswith("CONECT"):
            updated.append(line)
            continue

        origin_idx = _try_convert(line[6:11], int)
        idx_1 = _try_convert(line[11:16], int)
        idx_2 = _try_convert(line[16:21], int)
        idx_3 = _try_convert(line[21:26], int)
        idx_4 = _try_convert(line[26:31], int)
        if origin_idx is None or idx_1 is None:
            updated.append(line)
            continue

        partners = [idx_1, idx_2, idx_3, idx_4]
        new_origin = serial_map.get(origin_idx, origin_idx)
        new_partners = [serial_map.get(p, p) for p in partners if p is not None]

        l = "CONECT" + f"{new_origin:5d}"
        for p in new_partners:
            l += f"{p:5d}"
        updated.append(l + "\n")

    with open(pdb_path, "w") as f:
        f.writelines(updated)


def solve_alter_struct(protein_pdb: str) -> str:
    out_pdb = protein_pdb.replace(".pdb", "_no_alt.pdb")
    struct = gemmi.read_structure(protein_pdb)

    # 1. get old serial to coord map and connection info of old serial
    old_serial_to_coord = {}
    old_connections = []

    for model in struct:
        for chain in model:
            for res in chain:
                for atom in res:
                    # 使用 tuple(atom.pos) 将坐标转换为 (x, y, z) 元组作为 Key
                    # Gemmi 的 Position 对象直接转 tuple 是安全的
                    atom_id = tuple(atom.pos)
                    old_serial_to_coord[atom.serial] = atom_id

    # 备份连接信息 (保存为：[发起原子Serial, *伙伴原子Serials])
    # gemmi have problem read in connection info from pdb file
    # so we need to backup the connection info manually
    with open(protein_pdb, 'r') as f:
        for line in f:
            if line.startswith("CONECT"):
                # According to the pdb file format: https://www.wwpdb.org/documentation/file-format-content/format33/sect10.html,
                # each CONECT record contains the connectivity of one atom (origin_idx) with up to 4 other atoms (idx_1 to idx_4).
                origin_idx = _try_convert(line[6:11], int)
                idx_1 = _try_convert(line[11:16], int)
                idx_2 = _try_convert(line[16:21], int)
                idx_3 = _try_convert(line[21:26], int)
                idx_4 = _try_convert(line[26:31], int)
                if origin_idx is None or idx_1 is None:
                    raise ValueError(f'Invalid CONECT record: not enough atom indices in the following line\n{line}')
                if idx_4 is not None and len(line) > 32:
                    raise ValueError(f'Invalid CONECT record: too many atom indices in the following line\n{line}')

                # 1. 把所有可能的伙伴放入一个列表
                potential_partners = [idx_1, idx_2, idx_3, idx_4]

                # 2. 过滤掉 None 的伙伴
                valid_partners = [p for p in potential_partners if p is not None]

                # 3. 组合起来 (源原子 + 有效伙伴) 并添加
                if len(valid_partners) > 0:
                    old_connections.append([origin_idx] + valid_partners)

    # 2. clean alternative conformations
    # struct.remove_alternative_conformations()

    # 3. get atom coord to new serial map (this new serial should have TER taking up the serial number)
    # cryst1_record=False because bindingnet v2 pdb file does NOT have CRYST1 record
    # choose not to introduce gemmi default CRYST1 record
    struct.write_pdb(
        out_pdb,
        gemmi.PdbWriteOptions(minimal=False,
                              numbered_ter=True,
                              ssbond_records=False,
                              ter_records=True,
                              end_record=False,
                              cryst1_record=False))
    coord_to_new_serial = {}
    # this file is to get new serial to coord map
    new_serial_struct = gemmi.read_structure(out_pdb)
    for model in new_serial_struct:
        for chain in model:
            for res in chain:
                for atom in res:
                    pos = tuple(atom.pos)
                    assert pos not in coord_to_new_serial, f"duplicate coord {pos} in {protein_pdb}"
                    coord_to_new_serial[pos] = atom.serial

    # 4. update connection info based on new serial
    new_connections = []
    for conn in old_connections:
        central_atom = conn[0]
        neighbors = conn[1:]
        if central_atom not in old_serial_to_coord or old_serial_to_coord[central_atom] not in coord_to_new_serial:
            continue
        new_validated_neighbors = [
            coord_to_new_serial[old_serial_to_coord[old_serial]]
            for old_serial in neighbors
            if old_serial in old_serial_to_coord and old_serial_to_coord[old_serial] in coord_to_new_serial
        ]
        if len(new_validated_neighbors) == 0:
            continue
        new_conn = [coord_to_new_serial[old_serial_to_coord[central_atom]]]
        new_conn.extend(new_validated_neighbors)
        new_connections.append(new_conn)

    # 5. append new connection info to pdb file using pdb connect format
    with open(out_pdb, 'a') as f:
        for conn in new_connections:
            l = "CONECT"
            # 动态追加每一个原子序号，强制右对齐占5位
            for serial in conn:
                l += f"{serial:5d}"
            f.write(l + "\n")
        f.write("END\n")

    # 6. add the TER record if necessary
    insert_ter(out_pdb)

    return out_pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a protein pdb file.")
    parser.add_argument("--protein_pdb",
                        type=str,
                        default="protein_w_cofactors.pdb",
                        help="Path that contains protein pdb file.")
    args = parser.parse_args()

    solve_alter_struct(args.protein_pdb)
