# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from typing import List, Tuple, Union

import ase.io as aio
import numpy as np

import bytemol.toolkit.gmxtool.topparse as topparse
import bytemol.toolkit.system_builder.system_builder_tools as systools
from bytemol.core import Molecule

logger = logging.getLogger(__name__)

genion_mdp = r"""
integrator                = md
nsteps                    = 0
dt                        = 0.00001
comm-grps                 = System
continuation              = yes
; Neighbor
verlet-buffer-tolerance   = -1
nstlist                   = 10
rlist                     = 0.9
rcoulomb                  = 0.8
rvdw                      = 0.8
; Coulomb
coulombtype               = Cut-off
coulomb-modifier          = None
; VDW
vdwtype                   = Cut-off
vdw-modifier              = None
; Periodic boundary conditions
pbc                       = xyz
"""


def check_legal_path(paths: Union[str, List[str]]):
    if isinstance(paths, str):
        paths = [
            paths,
        ]
    for path in paths:
        if not os.path.exists(path):
            raise IOError(f"{path} not exists")


def _reformat_gro_itp_files(sdf_or_xyz_files: List[str], itp_files: List[str], output_dir: str, start_name: str):
    """
    given sdf/xyz files and itp files with all identities to be ligands or all identities to be cofactors, 
    if all ligands identity: start_name = "M"
    if all cofactors identity: start_name = "C"
    generate itp files and gro files with following rules:
        1. rename itp files to MXX.itp and CXX.itp; XX both starts with 00, 01, 02, ...
        2. generate gro files with naming MXX.gro and CXX.gro by first converting sdf to xyz then to gro
        3. change moleculetype, resname in itp files to MXX and CXX
        4. change atomtype in itp files to have "MXX_" and "CXX_" prefix 
        to avoid shared atomtype naming with different FF params in different ligands, cofactors and protein
        5. itp_files can have duplicate values to allow multiple sdf files sharing same molecule type 
    
    return gro_files, itp_files
    """
    # 0. input sanity check
    assert start_name in ["M", "C"], RuntimeError(f"start_name {start_name} must be M or C")
    for struct_file in sdf_or_xyz_files:
        assert struct_file.endswith(".sdf") or struct_file.endswith(".xyz"), RuntimeError(
            f"{struct_file} must be .sdf or .xyz")
    os.makedirs(output_dir, exist_ok=True)

    # 1. generate itp files and gro files
    out_itp_files = []
    out_gro_files = []
    unique_itps = {}
    naming_mapping = {"sdf_xyz_mapping": {}, "itp_name_mapping": {}}
    for i, (struct_file, itp_file) in enumerate(zip(sdf_or_xyz_files, itp_files)):
        # 1.1 obtain reformate file naming: {lig_name}.gro, {itp_name}.itp
        if itp_file in unique_itps:
            itp_i = unique_itps[itp_file]
        else:
            itp_i = unique_itps[itp_file] = i
        lig_name = f"{start_name}{str(i).zfill(2)}"
        itp_name = f"{start_name}{str(itp_i).zfill(2)}"
        # 1.2 rename moleculename, resname, atomtype in itp files
        out_itp_path = os.path.join(output_dir, f"{itp_name}.itp")
        shutil.copy(itp_file, out_itp_path)
        # change molecule type naming from MOL to itp_name
        systools.mod_itp_moleculename_inplace(out_itp_path, itp_name)
        # change resname to itp_name, so that we can distinguish different ligand type within gro file
        systools.mod_itp_resname_inplace(out_itp_path, itp_name)
        # change atomtype to have "MXX_" and "CXX_" prefix
        systools.mod_itp_atomtype_inplace(out_itp_path, itp_name)

        # 1.3 convert sdf files to gro files via xyz file format
        # strictly conserve atom sequence in sdf file during file format transformation
        out_gro_path = os.path.join(output_dir, lig_name + ".gro")
        tmpxyzdirname = tempfile.mkdtemp()
        tmp_xyz_file = os.path.join(tmpxyzdirname, f"{lig_name}.xyz")
        if struct_file.endswith(".sdf"):
            mol = Molecule.from_sdf(struct_file)
            mol.to_xyz(tmp_xyz_file)
        else:
            shutil.copy(struct_file, tmp_xyz_file)
        atoms = aio.read(tmp_xyz_file, index=0)
        systools.create_gas_gro(atoms, out_gro_path, resname=itp_name)

        # 1.4 clean up tmp xyz file
        shutil.rmtree(tmpxyzdirname)
        # 1.5 add to output file list
        out_gro_files.append(out_gro_path)
        out_itp_files.append(out_itp_path)
        naming_mapping["sdf_xyz_mapping"][os.path.basename(struct_file)[:-4]] = lig_name
        naming_mapping["itp_name_mapping"][os.path.basename(itp_file)[:-4]] = itp_name

    # 2. output naming mapping json
    with open(os.path.join(output_dir, "small_molecule_name_mapping.json"), "w") as f:
        json.dump(naming_mapping, f, indent=4)

    logger.info(f"generated gro files: {out_gro_files}\n generated itp files {out_itp_files}")

    return (out_gro_files, out_itp_files)


def prepare_small_molecules(structures: List[str],
                            itps: List[str],
                            output_dir: str,
                            lig_cofactor_identities: List[str] = None) -> Tuple[List[str], List[str]]:
    """
       given small molecule structure list (xyz/sdf files) and topology files (itp files),
       with lig_cofactor_identities marking each small molecule's identity as either "LIG" or "COF"
       parse each ligand/cofactor structure to gro file and rename to MXX.gro or CXX.gro (XX: 00, 01, 02 ...)
       parse each ligand/cofactor itp with molecule type, resname and atomtype renamed
       output:
            (list of small molecule gro files, list of corresponding itp files)
       NOTE here multiple small molecules can share same itp files
       NOTE same type of molecules should stay together inside the ligand_itps list!!! [XXX, YYY, XXX] is NOT a valid itps input!!!!
    """
    ####################
    # 0. input sanity check
    #####################
    # 0.0 check if sufficient input files are provided
    assert len(structures) == len(itps), RuntimeError(
        'User should provide equal number of atom coordinates files (xyz/gro) and the forcefield files (itp) for small molecule MD preparation.'
    )
    check_legal_path(structures)
    check_legal_path(itps)
    assert len(itps) < 100  # limit due to M01, M02, MXX or C01, C02, CXX naming convention
    # 0.1 guard for wrong itp sequence input
    stored_itp_names = set()
    prev_itp = ""
    for itp in itps:
        if itp not in stored_itp_names:
            stored_itp_names.add(itp)
            prev_itp = itp
        elif prev_itp == itp:
            continue
        else:
            raise IOError(
                f"{itps} is invalid!! same type of molecules should stay together inside the ligand_itps list!!!")
    # 0.2 check if lig_cofactor_list is valid
    if lig_cofactor_identities is not None:
        assert len(lig_cofactor_identities) == len(itps), RuntimeError(
            'User should provide equal number of ligand names and ligand structures for small molecule MD preparation.')
        for name in lig_cofactor_identities:
            assert name in ["LIG", "COF"]
    else:
        lig_cofactor_identities = len(structures) * ["LIG"]

    # 0.3 convert data format for indexing
    lig_cofactor_identities = np.array(lig_cofactor_identities)
    structures = np.array(structures)
    itps = np.array(itps)

    os.makedirs(output_dir, exist_ok=True)

    ###################################
    # 1. get output gros and itps ready
    ###################################
    out_gros = np.array([""] * len(structures), dtype=object)
    out_itps = np.array([""] * len(structures), dtype=object)

    # 1.1 prepare itp and gro files for ligands
    logger.info(f"lig_cofactor_identities: {lig_cofactor_identities}")
    lig_locations = (lig_cofactor_identities == "LIG")
    if sum(lig_locations) > 0:
        lig_structures = structures[lig_locations]
        lig_itps = itps[lig_locations]
        out_lig_gros, out_lig_itps = _reformat_gro_itp_files(lig_structures, lig_itps, output_dir, "M")
        print(f"out_lig_gros: {out_lig_gros}\n lig_locations: {lig_locations}")

        out_gros[lig_locations] = out_lig_gros
        out_itps[lig_locations] = out_lig_itps

    # 1.2 prepare itp and gro files for cofactors
    cof_locations = (lig_cofactor_identities == "COF")
    if sum(cof_locations) > 0:
        cof_structures = structures[cof_locations]
        cof_itps = itps[cof_locations]
        out_cof_gros, out_cof_itps = _reformat_gro_itp_files(cof_structures, cof_itps, output_dir, "C")
        out_gros[cof_locations] = out_cof_gros
        out_itps[cof_locations] = out_cof_itps

    print(f"out_gros: {out_gros}")
    return (out_gros.tolist(), out_itps.tolist())


def prepare_protein(protein_structure: str, protein_top: str, output_dir: str):

    assert protein_structure[-4:].lower() in [".gro"], "only gro is allowed"
    check_legal_path(protein_structure)
    check_legal_path(protein_top)

    os.makedirs(output_dir, exist_ok=True)

    gro_path = os.path.join(output_dir, "protein.gro")
    top_path = os.path.join(output_dir, "protein.top")
    shutil.copy(protein_structure, gro_path)
    shutil.copy(protein_top, top_path)

    return (gro_path, top_path)


def _get_num_water(solute_gro: str, solvated_gro: str):
    with open(solute_gro, 'r') as f:
        num_lines_gas = sum(1 for _ in f)
    with open(solvated_gro, 'r') as f:
        num_lines_water = sum(1 for _ in f)
    assert (num_lines_water - num_lines_gas) % 3 == 0
    num_water = int((num_lines_water - num_lines_gas) / 3)
    return num_water


def add_water_ions_counterions(solute_gro: str, solute_top: str, conc: float, solvent_ions: Tuple[str, str],
                               output_dir: str) -> Tuple[str, str]:
    """solute_gro should NOT contain molecules with SOL moleculename, if there are crystalized water, rename it in gro and top file!!!!!"""

    os.makedirs(output_dir, exist_ok=True)
    system_gro = os.path.join(output_dir, "system.gro")
    system_top = os.path.join(output_dir, "system.top")
    shutil.copy(solute_top, system_top)

    # step 2.1 add water gro
    arguments = f"solvate -cp {solute_gro} -o {system_gro}"
    systools.gmx_process(arguments)

    # step 2.2 add water top
    num_water = _get_num_water(solute_gro, system_gro)
    systools.include_tip3p_in_top(solute_top, num_water, system_top)

    # step 2.3 add counter ion and generate ions for given concentration
    # counter ion type should be same as solvent_ions

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_mdp = os.path.join(tmpdirname, "genion.mdp")
        tmp_tpr = os.path.join(tmpdirname, "genion.tpr")
        tmp_mdout = os.path.join(tmpdirname, "mdout.mdp")
        with open(tmp_mdp, "w") as f:
            f.write(genion_mdp)
        arguments = f"grompp -f {tmp_mdp} -c {system_gro} -p {system_top} -po {tmp_mdout} -o {tmp_tpr} -maxwarn 1"
        systools.gmx_process(arguments)

        cation_type, anion_type = solvent_ions
        cmd = f"genion -s {tmp_tpr} -o {system_gro} -p {system_top} -pname {cation_type} -nname {anion_type} -neutral "
        if conc > 0.0:
            cmd += f"-conc {conc}"
        systools.gmx_process(cmd, user_input="echo SOL |")

    # step 2.4 cleanup top format
    tfs = topparse.TopoFullSystem.from_file(system_top)
    tfs.write_top(system_top)

    return (system_gro, system_top)


@dataclass
class SystemBuilderConfig:
    # Protein-related inputs
    protein_structure: str = None  # Path to .gro file; can include cofactors
    protein_top: str = None  # Path to .top file for the protein

    # Ligand-cofactor-related inputs
    ligand_structures: List[str] = None  # List of ligand & cofactor structure files (.xyz, .sdf)
    ligand_itps: List[str] = None  # List of ligand & cofactor topology files (.itp)
    ligand_or_cofactor_identities: List[
        str] = None  # identity of ligand vs cofactor in list ligand_structures and ligand_itps ("LIG" or "COF")

    # Output directory
    output_dir: str = "."  # Path to the directory where outputs will be stored

    # Solvation/gas settings
    solvent_or_gas: str = None  # "solvent" or "gas"
    solvent_ions: Tuple[str, str] = ("NA", "CL")  # Ion types to use in solvent; (cation, anion)
    solvent_concentration: float = 0.0  # Molar concentration of ions; M; mol/L
    move_solute_to_box_center: bool = False

    # Box creation settings
    box_creation_mode: str = None  # "padding" or "dimension"
    box_padding: float = None  # unit: nm. Padding distance if using "padding" mode
    box_dimension: Tuple[float, float, float] = (0, 0, 0)  # unit: nm. Dimensions if using "dimension" mode

    @classmethod
    def from_json(cls, json_file: str):
        assert os.path.exists(json_file)
        with open(json_file, 'r') as f:
            config = json.load(f)
        return cls(**config)

    def __post_init__(self):
        # Normalize strings
        self.solvent_or_gas = self.solvent_or_gas.lower()
        self.box_creation_mode = self.box_creation_mode.lower()

        # Validate input
        assert self.protein_structure is not None or self.ligand_structures is not None
        if self.protein_structure is not None:
            assert self.protein_top is not None
            check_legal_path(self.protein_structure)
            check_legal_path(self.protein_top)

        if self.ligand_structures is not None:
            assert self.ligand_itps is not None
            assert len(self.ligand_structures) == len(self.ligand_itps)
            check_legal_path(self.ligand_structures)
            check_legal_path(self.ligand_itps)

        assert self.solvent_or_gas in ["solvent", "gas"]
        assert self.box_creation_mode in ["padding", "dimension"]
        if not self.move_solute_to_box_center:
            assert self.box_creation_mode == "dimension"

        if self.box_creation_mode == "padding":
            assert self.box_padding and isinstance(self.box_padding, float) or isinstance(self.box_padding, int)
            assert self.box_padding >= 0.9  # nm # padding distance should be larger than 0.9 nm set by gen_ion.mdp
        else:
            assert self.box_dimension and len(self.box_dimension) == 3
            assert not any(d <= 0 for d in self.box_dimension)

        if self.solvent_or_gas == "gas" and self.box_creation_mode == "padding":
            assert self.box_padding >= 5  # nm
        elif self.solvent_or_gas == "gas" and self.box_creation_mode == "dimension":
            box_x_len, box_y_len, box_z_len = self.box_dimension
            assert box_x_len > 10  # nm
            assert box_y_len > 10  # nm
            assert box_z_len > 10  # nm

        if self.solvent_or_gas == "solvent":
            assert isinstance(self.solvent_concentration, float) or isinstance(self.solvent_concentration, int)
            assert self.solvent_concentration >= 0.0
            assert self.solvent_ions[0] in ["LI", "NA", "K", "RB", "CS", "TL", "CU", "AG", "F"]
            assert self.solvent_ions[1] in ["CL", "BR", "I"]

        os.makedirs(self.output_dir, exist_ok=True)


def prepare_system(config: SystemBuilderConfig):
    """
    system assembly sequence: 
        always start from ligands, then protein, cofactor, ...
    """

    # Step 0. record input config
    logger.info(f"system builder config: {asdict(config)}")

    # Step 1.1 prepare solute (available protein and cofactor and ligands) gro file
    gro_list = []
    protein_molecule_names = []  # has amino acid residue name inside molecule
    cofactor_molecule_names = []  # appears in protein.top and not protein
    ligand_molecule_names = []
    protein_top = None
    small_mol_itps = None
    if config.protein_structure is not None:
        protein_gro, protein_top = prepare_protein(config.protein_structure, config.protein_top, config.output_dir)
        gro_list.append(protein_gro)
        protein_molecule_names, cofactor_molecule_names = systools.get_prot_cofactor_molnames(protein_top)

    if config.ligand_structures:
        if config.ligand_or_cofactor_identities is None:
            config.ligand_or_cofactor_identities = len(config.ligand_structures) * ["LIG"]

        small_mol_gros, small_mol_itps = prepare_small_molecules(config.ligand_structures, config.ligand_itps,
                                                                 config.output_dir,
                                                                 config.ligand_or_cofactor_identities)
        lig_mol_gros = [gro for gro, id in zip(small_mol_gros, config.ligand_or_cofactor_identities) if id == "LIG"]
        # ligand prior to proteins in gro_list
        gro_list = lig_mol_gros + gro_list
        cof_mol_gros = [gro for gro, id in zip(small_mol_gros, config.ligand_or_cofactor_identities) if id == "COF"]
        gro_list.extend(cof_mol_gros)
        logger.info(f"ligand gro files: {lig_mol_gros}")
        logger.info(f"cofactor gro files: {cof_mol_gros}")
        logger.info(f"gro_list: {gro_list}")

        ligand_molecule_names, cofactor_molecule_names_2add = systools.get_ligand_cofactor_molnames(
            small_mol_itps, config.ligand_or_cofactor_identities)
        cofactor_molecule_names = set(cofactor_molecule_names + cofactor_molecule_names_2add)
        cofactor_molecule_names = list(cofactor_molecule_names)

    assert len(gro_list) > 0, "current design does NOT support pure bulk water system"

    solute_gro = os.path.join(config.output_dir, "solute.gro")
    # concatentate gro files, preserve residue name in original gro files
    systools.combine_gro_lst(gro_list, solute_gro)

    # Step 1.2 prepare solute (available protein and cofactor and ligands) top file
    solute_top = os.path.join(config.output_dir, "solute.top")
    systools.create_combined_top(solute_top,
                                 prot_top=protein_top,
                                 itp_list=small_mol_itps,
                                 itp_identities=config.ligand_or_cofactor_identities)

    systools.format_combined_gro(solute_gro, solute_top)

    # Step 1.3 set up box for solute
    if config.box_creation_mode == "padding":
        padding = config.box_padding
        arguments = f"editconf -f {solute_gro} -o {solute_gro} -d {padding} -bt triclinic"
    else:
        box_x_len, box_y_len, box_z_len = config.box_dimension
        arguments = f"editconf -f {solute_gro} -o {solute_gro} -box {box_x_len} {box_y_len} {box_z_len} -bt triclinic"

    if not config.move_solute_to_box_center:
        arguments += " -noc"

    systools.gmx_process(arguments)

    # Step 2.1 for different use cases, prepare system_gro and system_top accordingly
    if config.solvent_or_gas == "gas":
        system_gro = os.path.join(config.output_dir, "system.gro")
        system_top = os.path.join(config.output_dir, "system.top")
        shutil.copy(solute_gro, system_gro)
        shutil.copy(solute_top, system_top)

    else:
        system_gro, system_top = add_water_ions_counterions(solute_gro, solute_top, config.solvent_concentration,
                                                            config.solvent_ions, config.output_dir)

    # Step 3. prepare atom id json file
    system_aid = os.path.join(config.output_dir, "atom_ids.json")
    molnames = {
        "ligands": ligand_molecule_names,
        "protein": protein_molecule_names,
        "cofactors": cofactor_molecule_names
    }
    systools.get_index_summary_file(system_top, system_aid, molnames)
