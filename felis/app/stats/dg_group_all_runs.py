# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pro", type=str, required=True)
ap.add_argument("--benchmark", type=str, required=True)
ap.add_argument("--correction_config", type=str, default=None, required=False)
ap.add_argument("--outdir", type=str, required=True)
ap.add_argument("--runs", type=str, nargs="*", required=True)
ap.add_argument("--select-method", type=str, default="precise")
args = ap.parse_args()

# from felis.protocols.stats.main_dg_group_all_runs import _compare_filtered_dGsv1 as compare_filtered_dGs
# from felis.protocols.stats.main_dg_group_all_runs import compare_filtered_dGs_v2 as compare_filtered_dGs
from felis.protocols.stats.main_dg_group_all_runs import compare_filtered_dGs_v3 as compare_filtered_dGs
from felis.protocols.stats.main_dg_group_all_runs import (compare_full_dGs, dump_metrics, plot_all_metrics,
                                                          read_benchmark_file, read_result_one_run)

proname = args.pro
outdir = args.outdir
benchmark_file = args.benchmark
list_of_runs = args.runs
select_method = args.select_method

Path(outdir).mkdir(parents=True, exist_ok=True)

# read input files
bm_full, bm_exp, group_id_dict = read_benchmark_file(benchmark_file)
df_runs = [read_result_one_run(irun, group_id_dict, select_method) for irun in list_of_runs]

# compare dGs
df_comparedG, df_comparesol, df_comparepro, df_compareres = compare_full_dGs(bm_full,
                                                                             [df_full for df_full, _ in df_runs])
df_comparedG.to_csv(Path(outdir) / "compare-dG.tsv", sep="\t", float_format="%.4f", index=False)
df_comparesol.to_csv(Path(outdir) / "compare-sol.tsv", sep="\t", float_format="%.4f", index=False)
df_comparepro.to_csv(Path(outdir) / "compare-pro.tsv", sep="\t", float_format="%.4f", index=False)
df_compareres.to_csv(Path(outdir) / "compare-res.tsv", sep="\t", float_format="%.4f", index=False)

# filtered dGs: plot and metrics
# df_compare_filtered, np_dG_exp, np_dG_mean, np_dG_runs, mean_metrics, list_of_metrics = compare_filtered_dGs(
#     bm_exp, [df_filtered for _, df_filtered in df_runs])
# df_compare_filtered.to_csv(Path(outdir) / "compare-dG-filtered.tsv", sep="\t", float_format="%.4f", index=False)

if args.correction_config is None:
    args.correction_config = Path(args.benchmark).parent / "correction_config.json"
assert Path(args.correction_config).exists()
df_compare_filtered, np_dG_exp, np_dG_mean, np_dG_std, np_dG_runs, mean_metrics, list_of_metrics = compare_filtered_dGs(
    bm_full, bm_exp, df_comparedG, select_method, [df_full for df_full, _ in df_runs], outdir, args.correction_config)
df_compare_filtered.to_csv(Path(outdir) / "compare-dG-filtered-v3.tsv", sep="\t", float_format="%.4f", index=False)
plot_all_metrics(np_dG_exp, np_dG_mean, np_dG_runs, mean_metrics, proname,
                 Path(outdir) / "stats_pred_label.png",
                 Path(outdir) / "stats_shiftedpred_label.png", np_dG_std,
                 Path(outdir) / "stats_pred_label_stdev.png",
                 Path(outdir) / "stats_shiftedpred_label_stdev.png")
dump_metrics(mean_metrics, Path(outdir) / "stats_avg.tsv")
for idx, metrics in enumerate(list_of_metrics):
    dump_metrics(metrics, Path(outdir) / f"stats_r{idx+1}.tsv")
