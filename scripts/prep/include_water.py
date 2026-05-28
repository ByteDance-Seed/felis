# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil
import tempfile

from bytemol.toolkit.gmxtool import topparse
from bytemol.toolkit.system_builder.system_builder_tools import (GMX_BUILT_IN_SOLVENT_RESNAMES, _merge_atp_sections,
                                                                 combine_gro_lst)

crw_atp = r"""
[ atomtypes ]
;https://github.com/gromacs/gromacs/blob/main/share/top/amber99sb.ff/ffnonbonded.itp
;name  elem.num  mass  charge  ptype  sigma(nm)  epsilon(kJ/mol)
HW           1      1.008    0.0000  A   1.00000e-01  0.00000e+00
OW           8      16.00    0.0000  A   3.15061e-01  6.36386e-01
"""
crw_itp = r"""
;https://github.com/gromacs/gromacs/blob/main/share/top/amber99sb.ff/tip3p.itp
[ moleculetype ]
;mol.name  nr.excl
CRW        2
[ atoms ]
;id  at.type  res.nr  res.name  at.name  cg.nr  charge    mass
1    OW       1       CRW       OW       1      -0.834    16.00000
2    HW       1       CRW       HW1      1       0.417     1.00800
3    HW       1       CRW       HW2      1       0.417     1.00800
[ settles ]
; OW   funct   doh       dhh
1      1       0.09572   0.15139
[ exclusions ]
1   2   3
2   1   3
3   1   2
"""


def include_crystal_water_in_top(top_path: str, num: int, output_top: str, keep_tmp_files: bool = False):
    """
    top_path: .top topology file that does not include waters
    num: num of water molecules to be added to the system
    add water into the system, add ions definition into the system
    """
    assert isinstance(num, int) and num > 0

    lines = []

    # write atp to top file
    tfs = topparse.TopoFullSystem.from_file(top_path)
    lines.append(str(topparse.TopoDefaults(tfs.uuid)))

    tmpdirname = tempfile.mkdtemp()
    tmp_files = []

    tmp_crw = os.path.join(tmpdirname, "crw.itp")
    with open(tmp_crw, "w") as f:
        f.writelines(crw_atp + '\n')
        f.writelines(crw_itp)
    tmp_files.append(tmp_crw)

    topo_list = [top_path, tmp_crw]
    atp_section = _merge_atp_sections(topo_list)
    if not keep_tmp_files:
        shutil.rmtree(tmpdirname)
    lines.extend(atp_section)

    # write itp to top file
    for i in range(len(tfs.molecules)):
        _, _itp = tfs.str_mol_atp_itp(i)
        lines.append(_itp)

    # include tip3p itp
    lines.append(crw_itp)

    # now write system info
    lines.append("[ system ]")
    lines.append("system")
    lines.append("[ molecules ]")
    shouldnt_contain_restype = set(GMX_BUILT_IN_SOLVENT_RESNAMES)
    for i in range(len(tfs.molecules)):
        topo_index = tfs.mol_to_topo_index[i]
        molecule_type = tfs.molecules[i].name
        assert molecule_type == tfs.mol_topos[topo_index].moleculetype.name
        assert molecule_type != "SOL", "crystal water should not share 'SOL' molecule type with solvent"
        all_restypes = set([atom.residue for atom in tfs.mol_topos[topo_index].atoms])
        shouldnt_contain = all_restypes.intersection(shouldnt_contain_restype)
        assert len(
            shouldnt_contain
        ) == 0, f"residue types in moleculetype {molecule_type}: crystal water should not contain solvent restypes {shouldnt_contain}"
        lines.append(f"{molecule_type}    {tfs.molecules[i].nr}")

    lines.append(f"CRW    {num}")

    save_dir = os.path.dirname(os.path.abspath(output_top))
    os.makedirs(save_dir, exist_ok=True)
    with open(output_top, "w") as fw:
        for line in lines:
            fw.write(line + "\n")

    # to make better top format
    tfs = topparse.TopoFullSystem.from_file(output_top)
    tfs.write_top(output_top)

    return


def _get_num_crw(pure_crw_gro: str) -> int:
    # only accept crystal water gro file from labeled protein service
    with open(pure_crw_gro, "r") as f:
        lines = f.readlines()

    natoms = len(lines) - 3
    natoms_header = int(lines[1])
    assert natoms_header == natoms, f"pure_crw_gro {pure_crw_gro} has {natoms_header} atoms in header, which is not equal to {natoms} atoms"
    assert natoms % 3 == 0, f"pure_crw_gro {pure_crw_gro} has {natoms} atoms, which is not divisible by 3"
    return int(natoms / 3)


def construct_protein_w_crystal_water(protein_labeled_folder: str, output_path: str):
    assert os.path.exists(protein_labeled_folder)
    protein_gro = os.path.join(protein_labeled_folder, "protein_w_cofactors.gro")
    protein_top = os.path.join(protein_labeled_folder, "protein_w_cofactors.top")
    assert os.path.exists(protein_top)
    assert os.path.exists(protein_gro)
    crystal_gro = os.path.join(protein_labeled_folder, "protein_w_cofactors_crystal_waters_exclude_clash.gro")
    if not os.path.exists(crystal_gro):
        crystal_gro = os.path.join(protein_labeled_folder, "protein_w_cofactors_crystal_waters.gro")
    if os.path.exists(crystal_gro):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        protein_w_crw_gro = os.path.join(output_path, "protein_w_cofactors_w_crw.gro")
        protein_w_crw_top = os.path.join(output_path, "protein_w_cofactors_w_crw.top")
        combine_gro_lst([protein_gro, crystal_gro], protein_w_crw_gro)
        num_crw_molecule = _get_num_crw(crystal_gro)
        include_crystal_water_in_top(protein_top, num_crw_molecule, protein_w_crw_top)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_folder", type=str, default=".")
    args = parser.parse_args()
    assert os.path.exists(args.system_folder)
    protein_ff14sb = os.path.join(args.system_folder, "protein_ff14sb")
    assert os.path.exists(protein_ff14sb)
    construct_protein_w_crystal_water(protein_ff14sb, protein_ff14sb)
