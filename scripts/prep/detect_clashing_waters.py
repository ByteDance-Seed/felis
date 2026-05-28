# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import logging
import os

import MDAnalysis as mda
import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist

# Set up simple logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_all_ligand_coords(ligand_dir: str) -> np.ndarray:
    """
    Reads all .sdf files in the ligand directory and returns a single numpy array
    of all their atomic coordinates (in Angstroms).
    """
    sdf_files = glob.glob(os.path.join(ligand_dir, "*.sdf"))
    if not sdf_files:
        return np.array([])

    all_coords = []
    for sdf_file in sdf_files:
        suppl = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)
        for mol in suppl:
            if mol is None:
                continue
            conf = mol.GetConformer()
            coords = conf.GetPositions()  # n_atoms x 3 array in Angstroms
            all_coords.append(coords)

    if not all_coords:
        return np.array([])

    return np.vstack(all_coords)


def detect_clashes_for_target(target_dir: str, threshold: float = 1.0) -> None:
    """
    Detects clashing crystal waters in a target directory and writes them to a new .gro file.
    """
    gro_file = os.path.join(target_dir, "protein_ff14sb", "protein_w_cofactors_crystal_waters.gro")
    ligand_dir = os.path.join(target_dir, "ligands")
    out_file = os.path.join(target_dir, "protein_ff14sb", "clashed_water.gro")

    if not os.path.exists(gro_file):
        logging.warning(f"Water file not found: {gro_file}")
        return

    if not os.path.exists(ligand_dir):
        logging.warning(f"Ligand directory not found: {ligand_dir}")
        return

    # Load all ligand coordinates
    ligand_coords = get_all_ligand_coords(ligand_dir)
    if len(ligand_coords) == 0:
        logging.warning(f"No valid ligands found in {ligand_dir}")
        return

    # Load the crystal waters
    try:
        u = mda.Universe(gro_file)
    except Exception as e:
        logging.error(f"Failed to load {gro_file} with MDAnalysis: {e}")
        return

    # Assuming all residues in this .gro file are waters.
    # If not, you might need to select them: u.select_atoms("resname CRW or resname SOL or resname HOH")
    # For now, we iterate over all residues.
    clashing_water_indices = []

    for res in u.residues:
        water_atoms = res.atoms
        water_coords = water_atoms.positions  # in Angstroms

        # Calculate pairwise distances between water atoms and all ligand atoms
        distances = cdist(water_coords, ligand_coords)

        # If the minimum distance is below the threshold, it's a clash
        if np.min(distances) < threshold:
            # We store the 0-based index of the residue or the atoms directly
            clashing_water_indices.extend(atom.index for atom in water_atoms)

    if clashing_water_indices:
        out_file_exclude = os.path.join(target_dir, "protein_ff14sb",
                                        "protein_w_cofactors_crystal_waters_exclude_clash.gro")
        with open(gro_file, 'r') as f:
            lines = f.readlines()

        # Lines to exclude:
        # - Header (first 2 lines) and Footer (last 1 line) are treated separately
        # - Atom lines are 2 to N+2 (0-based index)
        # - Atom index i corresponds to line i + 2
        lines_to_exclude = {i + 2 for i in clashing_water_indices}

        # Write clashing waters
        out_lines = []
        out_lines.append(f"clashed crystal waters from {target_dir}\n")
        out_lines.append(f"{len(clashing_water_indices):>5}\n")

        for i, line in enumerate(lines[2:-1]):
            line_idx = i + 2
            if line_idx in lines_to_exclude:
                out_lines.append(line)

        # Append box vector (last line)
        if len(lines) > 2:
            out_lines.append(lines[-1])

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, 'w') as f:
            f.writelines(out_lines)

        logging.info(
            f"Target {target_dir}: Found {len(clashing_water_indices) // 3} clashing water molecules. Wrote to {out_file}"
        )

        new_lines = []
        # Keep title
        new_lines.append(f'{lines[0].strip()} (exclude clash < {threshold:.3f} A)\n')

        # Calculate new atom count
        try:
            old_atom_count = int(lines[1].strip())
            new_atom_count = old_atom_count - len(clashing_water_indices)
            # Preserve approximate spacing if possible, standard gro is %5d
            new_lines.append(f"{new_atom_count:>5}\n")
        except ValueError:
            logging.warning(f"Could not parse atom count from {gro_file}, using calculated value.")
            new_lines.append(f"{len(u.atoms) - len(clashing_water_indices):>5}\n")

        # Filter atom lines
        # Total lines = 2 (header) + Natoms + 1 (box) = Natoms + 3
        # We iterate from line 2 to end-1
        for i, line in enumerate(lines[2:-1]):
            line_idx = i + 2
            if line_idx not in lines_to_exclude:
                new_lines.append(line)

        # Append box vector (last line)
        if len(lines) > 2:
            new_lines.append(lines[-1])

        with open(out_file_exclude, 'w') as f:
            f.writelines(new_lines)

        logging.info(
            f"Target {target_dir}: Wrote {new_atom_count // 3} non-clashing water molecules to {out_file_exclude}")

    else:
        # If no clashes, do not write anything for clashed_water.gro
        # But should we write the full file as excluded?
        # The request implies we want the excluded set. If nothing clashes, the excluded set is the full set.
        logging.info(f"Target {target_dir}: No clashing waters found.")

        # Copy the original file to the exclude file since nothing clashed
        out_file_exclude = os.path.join(target_dir, "protein_ff14sb",
                                        "protein_w_cofactors_crystal_waters_exclude_clash.gro")
        # Read and write to ensure same behavior (or just copy)
        # Simple copy is safest for "text based" if we don't need to filter
        import shutil
        shutil.copy(gro_file, out_file_exclude)
        logging.info(f"Target {target_dir}: Copied full water set to {out_file_exclude}")


def main():
    parser = argparse.ArgumentParser(description="Detect crystal waters that clash with any ligand.")
    parser.add_argument("target_dirs",
                        nargs="+",
                        help="Path(s) to target folder(s) (e.g., pl_bfe_dataset/Schrodinger/waterset/hsp90_kung)")
    parser.add_argument("--threshold",
                        type=float,
                        default=1.0,
                        help="Clash distance threshold in Angstroms (default: 1.0)")

    args = parser.parse_args()

    for target_dir in args.target_dirs:
        detect_clashes_for_target(target_dir, args.threshold)


if __name__ == "__main__":
    main()
