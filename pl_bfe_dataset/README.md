# Protein-Ligand Binding Free Energy Benchmark Dataset

This repository contains a collection of protein-ligand binding free energy benchmark datasets curated for consistent evaluations of docking scores, relaxed docking (relaxed du), and absolute binding free energy perturbations (ABFEP).

## Data Description

The dataset consolidates high-quality binding affinity data from the following public sources:

- **Schrödinger FEP+ Data Source:** [https://github.com/schrodinger/public_binding_free_energy_benchmark](https://github.com/schrodinger/public_binding_free_energy_benchmark)

## Dataset Filter Criteria

To ensure data quality and consistency, the following filters were applied during curation:

1. **Unique Ligand Count:** Each target must have more than 6 unique ligands. (Note: Multiple protonation states or conformations of the same molecule count as a single unique ligand).
2. **Clean Binding Sites:** No cofactors or metal ions are present near the ligand binding site.
3. **Residue Compatibility:** Only non-standard residues covered by `PDBParser` are permitted.

## File Structure

The data is organized hierarchically by protein target (e.g., `Schrodinger/merck/pfkfb3`). Below is the file structure for a representative target:

```
merck/pfkfb3
├── open_source_benchmarks.csv
├── correction_config.json
├── group_id.csv
├── cofactors/
│   ├── cofactor_F6P.sdf
│   ├── ...
│   └── cofactor_POP.sdf
├── cofactors_itps_joint-25/
│   ├── cofactor_F6P.itp
│   ├── ...
│   └── cofactor_POP.itp
├── ligands/
│   ├── 19_flip.sdf
│   ├── ...
│   ├── 70_flip.sdf
│   └── 70.sdf
├── ligands_itps_joint-25/
│   ├── 19_flip.itp
│   ├── ...
│   ├── 70_flip.itp
│   └── 70.itp
├── ligands_itps_abcg2/
│   ├── 19_flip.itp
│   ├── ...
│   ├── 70_flip.itp
│   └── 70.itp
├── protein_ff14sb/
│   ├── protein_w_cofactors_crystal_waters.gro
│   ├── protein_w_cofactors.gro
│   ├── protein_w_cofactors.pdb
│   └── protein_w_cofactors.top
└── protein_ff14SBonlysc/
    ├── extracted_pure_protein.gro
    ├── extracted_pure_protein.pdb
    └── extracted_pure_protein.top
```

## File Descriptions

* **`correction_config.json`**
  Configuration file containing corrections and reference values used for benchmarking:
    * `exp_ref`: Experimentally measured absolute binding free energy ($\Delta G$), unit: kcal/mol.
    * `rotamer`: List of initial poses for the saved ligand; each pose corresponds to a unique SDF filename.
    * `pKa_tautomer`: Lists multiple pKa/tautomer state corrections for each ligand.
    * `symmetry`: Rigid Body (RB) symmetry corrections for each ligand.

* **`group_id.csv`**
  Maps each `ligand.sdf` to a `group_id`.
    * Allows for quick verification if multiple ligand SDF files (e.g., different tautomers or conformers) correspond to the same experimental measurement datapoint.

* **`open_source_benchmarks.csv`**
  Aggregated benchmark results from published open-source methods, including FEP+ RBFEP and Uni-FEP RBFEP data.

## Directory Descriptions

* **`cofactors/`**
  Contains SDF structure files for any cofactor molecules associated with the protein target.

* **`cofactors_itps_*/`** (e.g., `cofactors_itps_joint-25`)
  Contains topology files (`.itp`) for cofactors using specific force field parameter sets.

* **`ligands/`**
  Contains SDF structure files for all ligands in the dataset. Enumerated possible protonation states and poses for ligands are included.

* **`ligands_itps_*/`** (e.g., `ligands_itps_joint-25`, `ligands_itps_abcg2`)
  Contains topology files (`.itp`) for ligands using specific force field parameter sets.

* **`protein_ff14sb/`**
  Contains protein structure and topology files prepared using the AMBER ff14SB force field. Crystal waters are separately listed in `protein_w_cofactors_crystal_waters.gro`.

* **`protein_ff14SBonlysc/`**
  Contains protein structure and topology files using AMBER ff14SBonlysc force field for implicit solvent model compatibility, representing the pure protein component without any solvent molecules or ions.