# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from pandas import DataFrame, concat, read_csv

FILE_OUT = "_abfe.tsv"

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
out_dir = args.outdir if args.outdir else one_run_dir
Path(out_dir).mkdir(parents=True, exist_ok=True)
csv_out = f"{out_dir}/{FILE_OUT}"
df.to_csv(csv_out, sep="\t", index=False)
