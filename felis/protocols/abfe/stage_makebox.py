# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import shutil

from bytemol.toolkit.system_builder.build_system import prepare_system as prepare_system_bytemol
from bytemol.toolkit.system_builder.build_system import SystemBuilderConfig
from felis.configs import dump_config
from felis.configs import load_config
from felis.protocols.abfe.config_types import ABFEInputConfig
from felis.protocols.abfe.config_types import ABStage
from felis.protocols.abfe.config_types import ABStageExecutionBase
from felis.protocols.abfe.config_types import ABStageExecutionRegister


def prepare_system(outdir: str,
                   progro: str,
                   protop: str,
                   sdf: str,
                   itp: str,
                   cofactor_sdfs: list[str] = None,
                   cofactor_itps: list[str] = None,
                   pro_ionic_strength: float = 0.0):
    assert os.path.isfile(sdf)
    assert os.path.isfile(itp)
    assert Path(sdf).stem == Path(itp).stem

    if cofactor_sdfs is None:
        cofactor_sdfs = []
    if cofactor_itps is None:
        cofactor_itps = []
    assert len(cofactor_sdfs) == len(cofactor_itps)

    def sysX_impl(sdf, itp, cof_sdf_list, cof_itp_list, x, dict_template):
        stem = Path(sdf).stem
        prepare_dir = f"{outdir}/{stem}/prepare"
        outX_dir = f"{prepare_dir}/out{x}"
        sysX_atom_ids = f"{prepare_dir}/sys{x}_atom_ids.json"
        sysX_gro = f"{prepare_dir}/sys{x}.gro"
        sysX_top = f"{prepare_dir}/sys{x}.top"
        sysX_abjson = f"{prepare_dir}/sys{x}_ab_ligatoms.json"
        lig_structs, lig_itps, lig_identities = [sdf], [itp], ["LIG"]
        for cof_sdf, cof_itp in zip(cof_sdf_list, cof_itp_list):
            lig_structs.append(cof_sdf)
            lig_itps.append(cof_itp)
            lig_identities.append("COF")
        bconfig = SystemBuilderConfig(ligand_structures=lig_structs,
                                      ligand_itps=lig_itps,
                                      ligand_or_cofactor_identities=lig_identities,
                                      output_dir=outX_dir,
                                      **dict_template)
        prepare_system_bytemol(bconfig)
        shutil.copy(f"{outX_dir}/atom_ids.json", sysX_atom_ids)
        shutil.copy(f"{outX_dir}/system.gro", sysX_gro)
        shutil.copy(f"{outX_dir}/system.top", sysX_top)
        with open(sysX_atom_ids) as f_sysX_atom_ids:
            atom_ids = load_config(f_sysX_atom_ids)
        ligands_dict = atom_ids["ligands"]
        ligands_keys = list(ligands_dict.keys())
        ligand_atoms = ligands_dict[ligands_keys[0]][0]
        with open(sysX_abjson, "w") as fw:
            dump_config({"ab": {"ligatoms": ligand_atoms}}, fw)
        posres_dict = {"atoms": [], "k_kcal": 100.0, "tol_angstrom": 0.5}
        protein_heavy_dict = atom_ids["protein_heavy"]
        protein_heavy_keys_list = list(protein_heavy_dict)
        if len(protein_heavy_keys_list):
            k0 = protein_heavy_keys_list[0]
            for l0 in protein_heavy_dict[k0]:
                posres_dict["atoms"].extend(l0)
        else:
            ligands_dict = atom_ids["ligands"]
            for l0 in ligands_dict["M00"]:
                posres_dict["atoms"].extend(l0)
        sysX_posres_json = f"{prepare_dir}/sys{x}_posres.json"
        with open(sysX_posres_json, "w") as fw:
            dump_config({"posres": posres_dict}, fw)

    prolig_dict_template = {
        "protein_structure": progro,
        "protein_top": protop,
        # "ligand_structures": [],
        # "ligand_itps": [],
        # "output_dir": "",
        "solvent_or_gas": "solvent",
        "solvent_ions": ["NA", "CL"],
        "solvent_concentration": pro_ionic_strength,
        "move_solute_to_box_center": True,
        "box_creation_mode": "padding",
        "box_padding": 1.0
    }
    ligand_dict_template = {
        # "ligand_structures": [],
        # "ligand_itps": [],
        # "output_dir": "",
        "solvent_or_gas": "solvent",
        "solvent_ions": ["NA", "CL"],
        "solvent_concentration": 0.0,
        "move_solute_to_box_center": True,
        "box_creation_mode": "padding",
        "box_padding": 1.2
    }
    os.makedirs(outdir, exist_ok=True)
    sysX_impl(sdf, itp, [], [], "A", ligand_dict_template)
    sysX_impl(sdf, itp, cofactor_sdfs, cofactor_itps, "B", prolig_dict_template)


@ABStageExecutionRegister(ABStage.makebox)
class ABStageExecution_makebox(ABStageExecutionBase):

    def __init__(self, acfg: ABFEInputConfig, sdf_abs: str, **kwargs):
        super().__init__(**kwargs)
        self._acfg = acfg
        self._sdf_abs = sdf_abs

    def exec(self):
        stage = self.stage
        planned_stages = self._planned_stages
        job_context = self._job_context
        acfg = self._acfg
        sdf_abs = self._sdf_abs

        if stage not in planned_stages:
            return

        prepare_system(
            outdir=acfg.tmpdir,
            progro=acfg.progro,
            protop=acfg.protop,
            sdf=sdf_abs,
            itp=acfg.itpfile,
            cofactor_sdfs=acfg.cofsdfs,
            cofactor_itps=acfg.cofitps,
            pro_ionic_strength=acfg.pro_ionic_strength,
        )
        shutil.copy(sdf_abs, job_context.prepare_dir)
        shutil.copy(acfg.itpfile, job_context.prepare_dir)

        job_context.write_stage_done(stage)
