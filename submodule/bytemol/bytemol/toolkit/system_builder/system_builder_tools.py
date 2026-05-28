# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import shutil
import subprocess
import tempfile

import ase
import numpy as np
from ase.io.gromacs import write_gromacs

from bytemol.toolkit.gmxtool import topparse
from bytemol.toolkit.protein.parse_pdb import CURRENTLY_ACCEPTABLE_PROTEIN_RESIDUES
from bytemol.utils import run_command_and_check

logger = logging.getLogger(__name__)

tip3p_atp = r"""
[ atomtypes ]
;https://github.com/gromacs/gromacs/blob/main/share/top/amber99sb.ff/ffnonbonded.itp
;name  elem.num  mass  charge  ptype  sigma(nm)  epsilon(kJ/mol)
HW           1      1.008    0.0000  A   1.00000e-01  0.00000e+00
OW           8      16.00    0.0000  A   3.15061e-01  6.36386e-01
"""

alchem_water_atp = r"""
[ atomtypes ]
;https://github.com/gromacs/gromacs/blob/main/share/top/amber99sb.ff/ffnonbonded.itp
;name  elem.num  mass  charge  ptype  sigma(nm)  epsilon(kJ/mol)
OW_ALW       1      1.008    0.0000  A   1.00000e-01  0.00000e+00
HW_ALW       8      16.00    0.0000  A   3.15061e-01  6.36386e-01
"""

tip3p_itp = r"""
;https://github.com/gromacs/gromacs/blob/main/share/top/amber99sb.ff/tip3p.itp
[ moleculetype ]
;mol.name  nr.excl
SOL        2
[ atoms ]
;id  at.type  res.nr  res.name  at.name  cg.nr  charge    mass
1    OW       1       SOL       OW       1      -0.834    16.00000
2    HW       1       SOL       HW1      1       0.417     1.00800
3    HW       1       SOL       HW2      1       0.417     1.00800
[ settles ]
; OW   funct   doh       dhh
1      1       0.09572   0.15139
[ exclusions ]
1   2   3
2   1   3
3   1   2
"""

# https://github.com/openmm/openmmforcefields/blob/main/openmmforcefields/ffxml/amber/tip3p_standard.xml
jctip3p_ions_atp = r"""
[ atomtypes ]
;name  elem.num  mass  charge  ptype  sigma(nm)  epsilon(kJ/mol)
  tip3pstdNa   11  22.989769  0.00  A     0.24392807     0.36584603
  tip3pstdCl   17  35.453200  0.00  A     0.4477657      0.14891274
"""

jctip3p_ions_itp = r"""
[ moleculetype ]
 NA   1
[ atoms ]
1     tip3pstdNa   1    NA     NA    1   +1.0
[ moleculetype ]
 CL   1
[ atoms ]
1     tip3pstdCl   1    CL     CL    1   -1.0
"""


class GMXError(Exception):
    '''general gmx error exception'''


def gmx_process(arguments: str, user_input: str = ""):
    ''' call gmx process '''
    assert isinstance(arguments, str) and arguments
    my_env = os.environ.copy()
    gmx_path = my_env.pop('GMX_PATH', 'gmx')

    try:
        run_command_and_check(" ".join([user_input, gmx_path, arguments]),
                              env={
                                  'GMX_NO_QUOTES': '1',
                                  'GMX_MAXBACKUP': '-1'
                              })
    except subprocess.CalledProcessError as exc:
        logger.exception('%s', exc)
        raise GMXError("GMX process failed") from exc

    return


def create_gas_gro(atoms: ase.Atoms, gro_file: str, resname: str):
    '''
    Convert small molecule atom coordinates in the input xyz file to a gro file.
    resname will be applied to all atoms.
    '''
    atoms.set_pbc([True] * 3)
    atoms.set_cell([20] * 3)
    atoms.set_array('residuenames', np.asarray([resname] * atoms.positions.shape[0]))
    write_gromacs(gro_file, atoms)


def mod_itp_moleculename_inplace(itp_file: str, new_molecule_name: str):
    with open(itp_file, 'r') as f_in:
        lines = f_in.readlines()

    reading_moleculetype = False

    for i, line in enumerate(lines):
        if line.strip().startswith('[ moleculetype ]'):
            reading_moleculetype = True
        elif reading_moleculetype:
            parts = line.split()
            if parts and parts[0][0] != ';':
                parts[0] = new_molecule_name
                lines[i] = '              '.join(parts) + '\n'
                break

    with open(itp_file, 'w') as f_out:
        f_out.writelines(lines)


def mod_itp_resname_inplace(itp_file: str, new_resname: str):
    assert len(new_resname) <= 5
    tfs = topparse.TopoFullSystem.from_file(itp_file)
    assert len(tfs.mol_topos) == 1
    topmol = tfs.mol_topos[0]
    for atom in topmol.atoms:
        atom.residue = new_resname
    with open(itp_file, 'w') as f:
        f.write(tfs.str_mol_itp(0))


def mod_itp_atomtype_inplace(itp_file: str, new_atomtype_prefix: str):
    print(f"\nmod itp file {itp_file} atomtype prefix to {new_atomtype_prefix}\n")
    tfs = topparse.TopoFullSystem.from_file(itp_file)
    assert len(tfs.mol_topos) == 1
    # update atomtypes in itp file [atoms] section
    topmol = tfs.mol_topos[0]
    for atom in topmol.atoms:
        atom.atype = new_atomtype_prefix + "_" + atom.atype
    # update atomtypes in itp file [atomtypes] section
    top_atypes = topparse.TopoAtomTypes(tfs.uuid)
    prev_atyps_copy = top_atypes.type_to_index.copy()
    for atomtype in prev_atyps_copy:
        top_atypes.rename_atomtype(atomtype, new_atomtype_prefix + "_" + atomtype)
    with open(itp_file, 'w') as f:
        f.write(tfs.str_mol_itp(0))


def get_prot_cofactor_molnames(top: str) -> tuple[list[str], list[str]]:
    tfs = topparse.TopoFullSystem.from_file(top)
    protein_molnames = []
    cofactor_molnames = []
    for i, mol in enumerate(tfs.molecules):
        moltop_index = tfs.mol_to_topo_index[i]
        moltop = tfs.mol_topos[moltop_index]
        first_atom = moltop.atoms[0]
        if first_atom.residue in CURRENTLY_ACCEPTABLE_PROTEIN_RESIDUES:
            # current molecule is protein
            if mol.name not in protein_molnames:
                protein_molnames.append(mol.name)
        else:
            if mol.name not in cofactor_molnames:
                cofactor_molnames.append(mol.name)
    return (protein_molnames, cofactor_molnames)


def get_ligand_cofactor_molnames(itps: list[str],
                                 ligand_or_cofactor_identities: list[str] = None) -> tuple[list[str], list[str]]:
    ligand_molnames = []
    cofactor_molnames = []
    if ligand_or_cofactor_identities is None:
        ligand_or_cofactor_identities = len(itps) * ["LIG"]
    for itp, lig_or_cof in zip(itps, ligand_or_cofactor_identities):
        tfs = topparse.TopoFullSystem.from_file(itp)
        assert len(tfs.molecules) == 1
        mol = tfs.molecules[0]
        if lig_or_cof == "LIG":
            if mol.name not in ligand_molnames:
                ligand_molnames.append(mol.name)
        else:
            if mol.name not in cofactor_molnames:
                cofactor_molnames.append(mol.name)
    return (ligand_molnames, cofactor_molnames)


def get_index_summary_file(top_path: str, out_json: str, molnames: dict[str, list[str]]):
    """
    input molnames: {
        "protein" : [protein_molnames]
        "cofactors" : [cofactor_molnames]
        "ligands" : [ligand_molnames]
    }

    output json atom id is 0-based
    """
    # input sanity check:
    molecule_types = set()
    for category in ["protein", "cofactors", "ligands"]:
        for moleculetype in molnames[category]:
            if moleculetype not in molecule_types:
                molecule_types.add(moleculetype)
            else:
                raise IOError(
                    f"duplicate molecule type {moleculetype} not allowed to share between protein, cofactors and ligands three different collections"
                )

    tfs = topparse.TopoFullSystem.from_file(top_path)
    summary_info = {
        "protein": {},
        "protein_backbone": {},
        "protein_heavy": {},
        "cofactors": {},
        "ligands": {},
        "others": {}
    }

    # write all atoms sections
    next_aid = 0
    for i, mol in enumerate(tfs.molecules):
        category = next((key for key, names in molnames.items() if mol.name in names), "others")

        if mol.name not in summary_info[category]:
            summary_info[category][mol.name] = []

        aid_lists = summary_info[category][mol.name]
        moltop_index = tfs.mol_to_topo_index[i]
        moltop = tfs.mol_topos[moltop_index]
        for _ in range(mol.nr):
            assert aid_lists is not None
            cur_molecule_aids = list(range(next_aid, next_aid + moltop.natoms))
            aid_lists.append(cur_molecule_aids)
            next_aid = next_aid + moltop.natoms

    # write protein_backbone and protein_heavyatoms
    next_aid = 0
    for i, mol in enumerate(tfs.molecules):
        if mol.name in molnames["protein"]:
            if mol.name not in summary_info["protein_backbone"]:
                summary_info["protein_backbone"][mol.name] = []
                summary_info["protein_heavy"][mol.name] = []

            prot_bb_lists = summary_info["protein_backbone"][mol.name]
            prot_heavy_lists = summary_info["protein_heavy"][mol.name]

        else:
            prot_bb_lists = None
            prot_heavy_lists = None

        moltop_index = tfs.mol_to_topo_index[i]
        moltop = tfs.mol_topos[moltop_index]
        for _ in range(mol.nr):
            cur_bb = []
            cur_heavy = []
            if prot_bb_lists is not None:
                for atom in moltop.atoms:
                    if atom.atom in ['CA', 'C', 'N']:
                        cur_bb.append(next_aid + atom.nr - 1)  # atom.nr is 1-based
                    if float(atom.mass) > 1.01:
                        cur_heavy.append(next_aid + atom.nr - 1)
                    elif float(atom.mass) < 1.00:
                        raise IOError("found atom mass < 1.00 amu, this is wierd, check input")

                prot_bb_lists.append(cur_bb)
                prot_heavy_lists.append(cur_heavy)

            next_aid = next_aid + moltop.natoms

    json_dir = os.path.dirname(os.path.abspath(out_json))
    os.makedirs(json_dir, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(summary_info, f, indent=4)


def read_gro_file(gro_path: str):

    with open(gro_path, 'r') as f_in:
        lines = f_in.readlines()
    title = lines[0]
    atom_count = int(lines[1].strip())
    atom_lines = lines[2:-1]
    box_dimension = lines[-1]

    return title, atom_count, atom_lines, box_dimension


def combine_gro_lst(gros: list[str], combined_gro: str):
    """only combine gro files, does not recalculate box dimensions"""
    assert gros and len(gros) >= 1
    if len(gros) == 1:
        shutil.copy(gros[0], combined_gro)
        return

    orig_gro = gros[0]
    orig_title, orig_atom_count, orig_lines, orig_box = read_gro_file(orig_gro)
    combined_atom_count = orig_atom_count
    combined_lines = orig_lines

    if len(gros) > 1:
        additional_gros = gros[1:]
        for additional_gro in additional_gros:
            _, new_atom_count, new_lines, _ = read_gro_file(additional_gro)
            combined_atom_count += new_atom_count

            #re-index the atoms in the additional gro file
            for i, _ in enumerate(new_lines):
                new_atom_index = int(i + orig_atom_count + 1)
                new_lines[i] = new_lines[i][:15] + f'{new_atom_index:>5d}' + new_lines[i][20:]

            combined_lines += new_lines

    # Use the box dimension in the orginal gro file
    combined_box_dimension = orig_box

    with open(combined_gro, 'w') as f_out:
        f_out.write(orig_title)
        f_out.write(f'{combined_atom_count}\n')
        f_out.writelines(combined_lines)
        f_out.write(combined_box_dimension)


def format_combined_gro(gro: str, top: str):
    # inplace revision
    # gro file created by combine_gro_lst has residue id messed up, fix it here
    # also update resname based on top provided
    assert os.path.exists(gro)
    assert os.path.exists(top)
    tfs = topparse.TopoFullSystem.from_file(top)

    # step 1. get residue natom list for all residue instance inside gro
    # record resname for each atom in top
    resname_atom_list = []
    residue_natom_list = []
    for i, mol in enumerate(tfs.molecules):
        resname_atom_list_in_mol = []
        residue_natom_list_in_mol = []
        moltop_index = tfs.mol_to_topo_index[i]
        moltop = tfs.mol_topos[moltop_index]
        cur_residue_natom = 0
        for aid, atom in enumerate(moltop.atoms):
            cur_residue_natom += 1
            resname_atom_list_in_mol.append(atom.residue)
            if aid == len(moltop.atoms) - 1 or atom.resnr != moltop.atoms[aid + 1].resnr:
                residue_natom_list_in_mol.append(cur_residue_natom)
                cur_residue_natom = 0
        for _ in range(mol.nr):
            residue_natom_list.extend(residue_natom_list_in_mol)
            resname_atom_list.extend(resname_atom_list_in_mol)

    # step 2. get atom index list (0-based) where these atoms are always the first atom inside the residue
    new_residue_first_atom_index = []
    accu_count = 0
    for natom in residue_natom_list:
        new_residue_first_atom_index.append(accu_count)
        accu_count += natom

    # step 3. rewrite residue id
    with open(gro, 'r') as f_in:
        lines = f_in.readlines()

    updated_lines = lines[:2]  # header and natom
    atom_lines = lines[2:-1]
    residue_index = 0  # index of new_residue_first_atom_index
    residue_id = "    1"
    for i, (atom_line, updated_resname) in enumerate(zip(atom_lines, resname_atom_list)):
        if residue_index < len(new_residue_first_atom_index) and i == new_residue_first_atom_index[residue_index]:
            residue_id = residue_index + 1
            residue_id = f"{residue_id:>5}"
            residue_id = residue_id[:5]
            residue_index += 1
        updated_resname = f"{updated_resname:<5}"
        updated_resname = updated_resname[:5]
        updated_atom_line = residue_id + updated_resname + atom_line[10:]
        updated_lines.append(updated_atom_line)

    updated_lines.append(lines[-1])  # box dimension

    with open(gro, 'w') as f_out:
        f_out.writelines(updated_lines)


def _from_itp_to_atpitp_files(itp_list: list[str]):
    """read in FF generated itp files, convert to itp, atp separated files
    itp_list: list of itp file names"""
    out_atps = []
    out_itps = []
    tmpdirname = tempfile.mkdtemp()
    for itp_file in itp_list:
        assert os.path.exists(itp_file)
        itp_fname = os.path.basename(itp_file)
        out_file = os.path.join(tmpdirname, f"tmp_{itp_fname}")
        ligtfs = topparse.TopoFullSystem.from_file(itp_file)
        ligtfs.write_itp(out_file, separated_atp=True)
        out_atps.append(out_file[:-4] + ".atp")
        out_itps.append(out_file)
    return (out_atps, out_itps)


def _merge_atp_sections(topo_files: list[str]):
    """
    combine all topology files atom type section
    different atp files can share same atom types
    """
    all_atomtype_records = dict()  # atomtype: atomtype line
    for p in topo_files:
        tfs = topparse.TopoFullSystem.from_file(p)
        ta = topparse.TopoAtomTypes(tfs.uuid)
        for record in ta.atomtypes:
            if record.name in all_atomtype_records.keys():
                newstr = str(record)
                oldstr = all_atomtype_records[record.name]
                if newstr != oldstr:
                    raise IOError(
                        f"same atom type name {record.name} in topo files {topo_files} have different params.")
            else:
                all_atomtype_records[record.name] = str(record)

    lines = [
        "[ atomtypes ]",
    ]
    for _, v in all_atomtype_records.items():
        lines.append(v)
    return lines


def _merge_itp_sections(itp_files_from_toparse: list[str]):
    lines = []
    for itp in itp_files_from_toparse:
        with open(itp, 'r') as f:
            for line in f:
                assert "atomtypes" not in lines, f"itp section should not be included in {itp} file"
                lines.append(line.strip())
    return lines


def _moltype_count_itp_files(itp_files):
    """process itp_files and write lines of molecule type count for [system] section"""
    if not itp_files:
        return

    lines = []
    count = 0
    for i, _itp in enumerate(itp_files):
        # Check if it is the same as the next ITP (to handle consecutive duplicates)
        if i < len(itp_files) - 1 and _itp == itp_files[i + 1]:
            count += 1
            continue

        count += 1
        tfs = topparse.TopoFullSystem.from_file(_itp)
        topo_index = tfs.mol_to_topo_index[0]
        assert tfs.molecules[0].name == tfs.mol_topos[topo_index].moleculetype.name
        lines.append(f"{tfs.molecules[0].name}    {count}")
        count = 0

    return lines


def create_combined_top(output_top: str,
                        prot_top: str = None,
                        itp_list: list[str] = None,
                        itp_identities: list[str] = None,
                        keep_tmp_files: bool = False):
    '''*********************** adapted from atm repo main_catmols func *************************
        combine topologies of molecules provided in prot_top and itp_list;
        itp_identities: list of identities for each itp file in itp_list, ["COF" | "LIG"]
        combine logic always follow LIG1, LIG2, protein_top(everything inside), COF1, COF2 ...
    '''

    # input sanity check
    assert output_top.endswith(".top")
    if itp_list is not None:
        if itp_identities is None:
            itp_identities = ["LIG"] * len(itp_list)
        assert len(itp_list) == len(
            itp_identities), f"len(itp_list) {len(itp_list)} != len(itp_identities) {len(itp_identities)}"
    else:
        assert itp_identities is None

    assert all([id in ["COF", "LIG"] for id in itp_identities
               ]), f"itp_identities should be either COF or LIG, got {itp_identities}"
    assert prot_top is not None or itp_list, "prot_top and itp_list can not be empty at the same time"

    # easy case
    if not itp_list:
        shutil.copy(prot_top, output_top)
        return

    tmp_file_list = []
    # data parsing
    topo_list = []  # .top and .itp topology files
    _itps = []  # topparse parsed itp section files for _merge_itp_sections
    tmpdirname = tempfile.mkdtemp()
    if prot_top is not None:
        assert os.path.exists(prot_top)
        topo_list.append(prot_top)

        protfs = topparse.TopoFullSystem.from_file(prot_top)
        prot_itps = []
        for i in range(len(protfs.molecules)):
            _, proitp = protfs.str_mol_atp_itp(i)
            proitp_file = os.path.join(tmpdirname, f"tmp_prot_{i}.itp")
            with open(proitp_file, "w") as f:
                f.write(proitp)
            prot_itps.append(proitp_file)
            tmp_file_list.append(proitp_file)

        prot_itps = list(dict.fromkeys(prot_itps))
        _itps.extend(prot_itps)

    if itp_list is not None:
        _itp_list = list(dict.fromkeys(itp_list))
        topo_list.extend(_itp_list)
        _, lig_itps = _from_itp_to_atpitp_files(_itp_list)
        _itps.extend(lig_itps)
        tmp_file_list.extend(lig_itps)
        tmp_file_list.extend([_itp[:-4] + ".atp" for _itp in lig_itps])

    # now generating topology file for this combined system
    lines = []
    lines.append(str(topparse.TopoDefaults()))

    # write atp section to top file
    atp_section = _merge_atp_sections(topo_list)
    lines.extend(atp_section)

    # write itp section to top file
    itp_section = _merge_itp_sections(_itps)
    lines.extend(itp_section)

    # now write system info
    lines.append("[ system ]")
    lines.append("system")
    lines.append("[ molecules ]")

    # write ligand top included molecule types
    lig_itps = [_itp for i, _itp in enumerate(itp_list) if itp_identities[i] == "LIG"]
    cof_itps = [_itp for i, _itp in enumerate(itp_list) if itp_identities[i] == "COF"]
    if len(lig_itps) > 0:
        _section_lines = _moltype_count_itp_files(lig_itps)
        lines.extend(_section_lines)

    # write protein top included molecule types
    if prot_top is not None:
        for i in range(len(protfs.molecules)):
            topo_index = protfs.mol_to_topo_index[i]
            assert protfs.molecules[i].name == protfs.mol_topos[topo_index].moleculetype.name
            lines.append(f"{protfs.molecules[i].name}    {protfs.molecules[i].nr}")

    # write cofactor top included molecule types
    if len(cof_itps) > 0:
        _section_lines = _moltype_count_itp_files(cof_itps)
        lines.extend(_section_lines)

    save_dir = os.path.dirname(os.path.abspath(output_top))
    os.makedirs(save_dir, exist_ok=True)
    with open(output_top, "w") as fw:
        for line in lines:
            fw.write(line + "\n")

    # to make better top format
    tfs = topparse.TopoFullSystem.from_file(output_top)
    tfs.write_top(output_top)

    if not keep_tmp_files:
        tmp_file_list = set(tmp_file_list)
        for f in tmp_file_list:
            os.remove(f)
        for f in tmp_file_list:
            p = os.path.dirname(f)
            if os.path.exists(p):
                shutil.rmtree(p)

    return


GMX_BUILT_IN_SOLVENT_RESNAMES = ["SOL", "WAT", "HOH", "OHH", "TIP", "T3P", "T4P", "T5P", "T3H"]


def include_tip3p_in_top(top_path: str, num: int, output_top: str, keep_tmp_files: bool = False):
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

    tmp_tip3p = os.path.join(tmpdirname, "tip3p.itp")
    with open(tmp_tip3p, "w") as f:
        f.writelines(tip3p_atp + '\n')
        f.writelines(tip3p_itp)
    tmp_files.append(tmp_tip3p)

    tmp_ion = os.path.join(tmpdirname, "ions.itp")
    with open(tmp_ion, "w") as f:
        f.writelines(jctip3p_ions_atp + '\n')
        f.writelines(jctip3p_ions_itp)
    tmp_files.append(tmp_ion)

    topo_list = [top_path, tmp_tip3p, tmp_ion]
    atp_section = _merge_atp_sections(topo_list)
    if not keep_tmp_files:
        shutil.rmtree(tmpdirname)
    lines.extend(atp_section)

    # write itp to top file
    for i in range(len(tfs.molecules)):
        _, _itp = tfs.str_mol_atp_itp(i)
        lines.append(_itp)

    # include tip3p itp
    lines.append(tip3p_itp)

    # include ions itp
    lines.append(jctip3p_ions_itp)

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

    lines.append(f"SOL    {num}")

    save_dir = os.path.dirname(os.path.abspath(output_top))
    os.makedirs(save_dir, exist_ok=True)
    with open(output_top, "w") as fw:
        for line in lines:
            fw.write(line + "\n")

    # to make better top format
    tfs = topparse.TopoFullSystem.from_file(output_top)
    tfs.write_top(output_top)

    return


def get_system_charge(top_file: str):
    tfs = topparse.TopoFullSystem.from_file(top_file)
    system_charge = 0.
    for i, molecule in enumerate(tfs.molecules):
        topo_index = tfs.mol_to_topo_index[i]
        molecule_topo = tfs.mol_topos[topo_index]
        molecule_charge = sum(molecule_topo.get_charges())
        system_charge += molecule.nr * molecule_charge
    return system_charge


def get_num_alchemwat(charge_needed: float, charge_range_per_alchem_water: tuple[float, float] = (-0.4, 0.6)) -> int:
    """
    charge_needed: total charge needed to neutralize the system
    charge_range_per_alchem_water: molecular total charge range of alchemical water molecules
    return num of alchemical water molecule num needed for current window
    """
    # one alchemical water molecule can help balance the charge ranges from [-0.4, 0.6]
    if charge_needed <= 0:
        idx = 0
    else:
        idx = 1
    charge_max_per_mol = charge_range_per_alchem_water[idx]
    num_alchem_water = charge_needed // charge_max_per_mol if np.isclose(
        charge_needed % charge_max_per_mol, 0.) else (charge_needed // charge_max_per_mol) + 1
    return int(num_alchem_water)


def get_alchemwater_topparse_itp_str(charge_needed: float, num_alchem_water: int) -> str:
    mol_charge = charge_needed / num_alchem_water
    if mol_charge >= 0.:
        positive_molcharge_shift = mol_charge
        negative_molcharge_shift = 0.
    else:
        positive_molcharge_shift = 0.
        negative_molcharge_shift = mol_charge / 2.0

    itp_str = f"""
;https://github.com/gromacs/gromacs/blob/main/share/top/amber99sb.ff/tip3p.itp
[ moleculetype ]
;mol.name  nr.excl
ALW        2
[ atoms ]
;id  at.type  res.nr  res.name  at.name  cg.nr  charge    mass
1    OW_ALW       1       ALW       ALO     1      {round(-0.834+positive_molcharge_shift, 5)}    16.00000
2    HW_ALW       1       ALW       ALH     1      {round(0.417+negative_molcharge_shift, 5)}     1.00800
3    HW_ALW       1       ALW       ALH     1      {round(0.417+negative_molcharge_shift, 5)}     1.00800
[ settles ]
; OW   funct   doh       dhh
1      1       0.09572   0.15139
[ exclusions ]
1   2   3
2   1   3
3   1   2
"""
    return itp_str


def include_alchemwater_in_top(top_path: str,
                               num: int,
                               alchem_water_topparse_itp: str,
                               output_top: str,
                               keep_tmp_files: bool = False):
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
    alw_itp_file = os.path.join(tmpdirname, "alw.itp")

    with open(alw_itp_file, "w") as f:
        f.writelines(alchem_water_atp + '\n')
        f.writelines(alchem_water_topparse_itp)

    topo_list = [top_path, alw_itp_file]
    atp_section = _merge_atp_sections(topo_list)
    if not keep_tmp_files:
        shutil.rmtree(tmpdirname)
    lines.extend(atp_section)

    # write itp to top file
    for i in range(len(tfs.molecules)):
        _, _itp = tfs.str_mol_atp_itp(i)
        lines.append(_itp)

    # include alchemical water itp
    lines.append(alchem_water_topparse_itp)

    # now write system info
    lines.append("[ system ]")
    lines.append("system")
    lines.append("[ molecules ]")
    for i in range(len(tfs.molecules)):
        topo_index = tfs.mol_to_topo_index[i]
        molecule_type = tfs.molecules[i].name
        assert molecule_type == tfs.mol_topos[topo_index].moleculetype.name
        if molecule_type == "SOL":
            prev_water_num = tfs.molecules[i].nr
            assert num < (
                prev_water_num
            ) / 20, f"alchemical water num not being negligible in bulk water; currently we need {num} alchemical waters"
            # now update water num; substitute $num water molecules to alchemical water molecules
            tfs.molecules[i].nr -= num
        lines.append(f"{molecule_type}    {tfs.molecules[i].nr}")

    if num > 0:
        lines.append(f"ALW    {num}")

    save_dir = os.path.dirname(os.path.abspath(output_top))
    os.makedirs(save_dir, exist_ok=True)
    with open(output_top, "w") as fw:
        for line in lines:
            fw.write(line + "\n")

    # to make better top format
    tfs = topparse.TopoFullSystem.from_file(output_top)
    tfs.write_top(output_top)

    return
