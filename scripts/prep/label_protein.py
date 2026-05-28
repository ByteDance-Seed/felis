# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import shutil

from bytemol.toolkit.protein import label_one_pdb
from bytemol.toolkit.protein.parse_pdb import PDBParser
from bytemol.toolkit.protein.pdb_output import PureProteinOutput

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def label_protein(pdb, label_ff):
    """
    ff14SBonlysc labeling: extract pure protein from pdb and label with given FF
    ff14sb labeling: extract protein with ions from pdb and label given given FF
    output dir name f"protein_{label_ff}" under same directory as input pdb file
    """
    assert label_ff in ["ff14SBonlysc", "ff14sb"]
    output_dir = os.path.join(os.path.dirname(pdb), f"protein_{label_ff}")
    shutil.rmtree(output_dir, ignore_errors=True)
    logger.info(f"start {label_ff} labeling for {pdb}.\n\n")

    # step 1. solve alternative structure issue for pdb parsing
    pdb_parser = PDBParser.parse_pdb(pdb, check_names=False)

    # step 2. extract pure protein from pdb file for ff14SBonlysc labeling service
    if label_ff == "ff14SBonlysc":
        pure_protein_fname = "extracted"
        pdb_to_label = PureProteinOutput(pdb_parser.pdb_data, pure_protein_fname).write_pdb(os.path.dirname(pdb))
        water_ion_model = None
    else:
        pdb_to_label = pdb
        water_ion_model = "tip3p"

    # step 3. label protein
    os.makedirs(output_dir)
    gro_file, top_file = label_one_pdb(pdb_to_label, output_dir, label_ff, water_ion_model=water_ion_model)
    assert gro_file is not None, f"Labeling failed for {pdb_to_label}. Check logs.\n\n"
    logger.info(f"finish {output_dir} protein labeling with FF {label_ff}, generate {gro_file} and {top_file}.\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract cofactors from protein pdb file.")
    parser.add_argument("--protein_pdb",
                        type=str,
                        default="protein_w_cofactors.pdb",
                        help="Path that contains protein pdb file.")
    parser.add_argument("--label_ff", type=str, required=True, help="[ff14sb|ff14SBonlysc]")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    assert os.path.exists(args.protein_pdb), f"{args.protein_pdb} not exist."

    label_protein(args.protein_pdb, args.label_ff)

    logger.info(f"finish {args.label_ff} labeling for {args.protein_pdb}.\n\n")
