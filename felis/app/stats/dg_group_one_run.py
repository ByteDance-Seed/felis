# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os
from pathlib import Path

from pandas import DataFrame, concat, read_csv

FILE_OUT = "_abfe.tsv"

def read_analysis_results(analysis_dir: str, ligand_name: str):
    sysA_results_file = os.path.join(analysis_dir, "sysA_free_energy.json")
    sysB_results_file = os.path.join(analysis_dir, "sysB_free_energy.json")
    R_fe_file = os.path.join(analysis_dir, "R_fe_table.tsv")

    if not Path(sysA_results_file).is_file() or not Path(sysB_results_file).is_file() or not Path(R_fe_file).is_file():
        print(f"Warning: {sysA_results_file}, {sysB_results_file}, {R_fe_file} do not exist in {analysis_dir}")
        return None

    import pandas as pd
    import json
    # Read sysA_results_file
    with open(sysA_results_file, "r") as f:
        sysA_results = json.load(f)
    # Read sysB_results_file
    with open(sysB_results_file, "r") as f:
        sysB_results = json.load(f)
    # Read R_fe_file
    with open(R_fe_file, "r") as f:
        R_fe = pd.read_csv(f, sep="\t")

    dG = {"ligand": ligand_name, "solv": None, "protein": None, "restraint": None}
    dG["solv"] = sysA_results["Delta_f"] / 4.184 # kJ/mol to kcal/mol
    dG["protein"] = sysB_results["Delta_f"] / 4.184 # kJ/mol to kcal/mol
    dG["restraint"] = R_fe.iloc[-1, -1] # kcal/mol
    
    dG["dG(kcal/mol)"] =  - (dG["protein"] - dG["solv"] - dG["restraint"])
    std = (sysB_results["dDelta_f"] / 4.184)**2 + (sysA_results["dDelta_f"] / 4.184)**2
    std = std**0.5
    dG["dG_std"] = std

    return dG

ap = argparse.ArgumentParser()
ap.add_argument("--dir", "--rundir", type=str, required=True)
ap.add_argument("--outdir", type=str, default=None)
args = ap.parse_args()

one_run_dir = args.dir
subdirs = [str(d.relative_to(one_run_dir)) for d in Path(one_run_dir).iterdir() if d.is_dir()]

df = DataFrame()
for subd in subdirs:
    abfe_tsv = f"{one_run_dir}/{subd}/analysis/sys_abfe.tsv"
    if Path(abfe_tsv).is_file():
        df1 = read_csv(abfe_tsv, sep="\t")
        df = concat([df, df1], ignore_index=True)
    else:
        # Read the felis-v2 result for analysis.
        dG = read_analysis_results(f"{one_run_dir}/{subd}/analysis", subd)
        if dG is None:
            continue
        df1 = DataFrame([dG])
        df = concat([df, df1], ignore_index=True)
    
out_dir = args.outdir if args.outdir else one_run_dir
Path(out_dir).mkdir(parents=True, exist_ok=True)
csv_out = f"{out_dir}/{FILE_OUT}"
df.to_csv(csv_out, sep="\t", index=False)
