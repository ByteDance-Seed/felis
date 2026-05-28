# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

from felis.protocols.correction.ccc_rbfe_network import read_rbfe_edges, run_ccc_from_edge_df


def _infer_sep(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".tsv") or lower.endswith(".tab"):
        return "\t"
    return ","


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=str, required=True, help="Input RBFE edges CSV/TSV")
    parser.add_argument("--out-edges", type=str, required=True, help="Output corrected edges CSV/TSV")
    parser.add_argument("--out-nodes", type=str, default=None, help="Optional output node potentials CSV/TSV")
    parser.add_argument("--lig1-col", type=str, default="Lig 1")
    parser.add_argument("--lig2-col", type=str, default="Lig 2")
    parser.add_argument("--ddg-col", type=str, default="Bennett ddG (kcal/mol)")
    parser.add_argument("--sigma-col", type=str, default="Bennett std. error (kcal/mol)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    edges_path = args.edges
    out_edges_path = args.out_edges
    out_nodes_path = args.out_nodes

    df_edges = read_rbfe_edges(edges_path)
    res = run_ccc_from_edge_df(
        df_edges,
        lig1_col=args.lig1_col,
        lig2_col=args.lig2_col,
        ddg_col=args.ddg_col,
        sigma_col=args.sigma_col,
        out_lig1_col=args.lig1_col,
        out_lig2_col=args.lig2_col,
        out_ddg_col=args.ddg_col,
        out_sigma_col=args.sigma_col,
    )

    Path(out_edges_path).parent.mkdir(parents=True, exist_ok=True)
    res.edge_df.to_csv(out_edges_path, sep=_infer_sep(out_edges_path), float_format="%.6f", index=False)

    if out_nodes_path is not None:
        Path(out_nodes_path).parent.mkdir(parents=True, exist_ok=True)
        res.node_df.to_csv(out_nodes_path, sep=_infer_sep(out_nodes_path), float_format="%.6f", index=False)

    # Emit basic diagnostics to stdout (JSON) for pipelines.
    print(json.dumps(res.diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
