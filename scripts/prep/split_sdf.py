# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re

from rdkit import Chem


def split_ligand_sdf(sdf_file, output_dir):
    """
    Split an SDF file containing multiple ligands into individual SDF files.

    Parameters:
    - sdf_file (str): Path to the input SDF file.
    - output_dir (str): Path to the directory where individual SDF files will be saved.
    """
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)
    output_dir = os.path.join(output_dir, "ligands")
    os.makedirs(output_dir, exist_ok=True)

    # split SDF file
    for mol in suppl:
        if mol is None:
            continue
        name = re.sub(r'[ (),]', '_', mol.GetProp('_Name'))
        writer = Chem.SDWriter(f"{output_dir}/{name}.sdf")
        writer.write(mol)
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split an SDF file containing multiple ligands into individual SDF files.")
    parser.add_argument("--sdf_file", type=str, help="Path to the input SDF file.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=".",
                        help="Path to the directory where individual SDF files will be saved.")
    args = parser.parse_args()

    split_ligand_sdf(args.sdf_file, args.output_dir)
