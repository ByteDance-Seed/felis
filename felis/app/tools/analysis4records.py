# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os

from felis.protocols.abfe.main_fe_restraints import calc_restraints
from felis.protocols.abfe_v2.lambda_config import read_lambda_schedule
from felis.protocols.abfe_v2.protein_lambda_rules import validate_protein_lambda_schedule
from felis.protocols.abfe_v2.utilities import generate_post_analysis_figures
from felis.protocols.abfe_v2.utilities import read_records
from felis.protocols.boresch.main_boresch_restraints import BORESCH_CFG_JSON
from felis.utils.setup_logger import setup_logger


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Standalone result analysis for an HREMD records.sqlite file."
    )
    ap.add_argument(
        "--energy_matrix",
        type=str,
        required=True,
        help="Path to the HREMD records.sqlite records file.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for analysis figures and results.",
    )
    ap.add_argument(
        "--prepare_dir",
        type=str,
        required=False,
        default=None,
        help="Path to the prepare directory containing the BORESCH configuration and sysB_lam.json files.",
    )
    ap.add_argument(
        "--simulated_time_length",
        type=float,
        default=None,
        help=(
            "Deprecated and ignored. The free-energy x-axis is derived from "
            "stored MD steps and optional records.sqlite timing metadata."
        ),
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional prefix for output file names.",
    )
    return ap


def main(argv: list[str] = None) -> None:
    args = _build_parser().parse_args(argv)

    setup_logger()

    prepare_dir = args.prepare_dir
    if prepare_dir is not None:
        sysB_lam_file = os.path.join(prepare_dir, "sysB_lam.json")
        if not os.path.exists(sysB_lam_file):
            raise FileNotFoundError(f"Missing required input: {sysB_lam_file}")
        sysB_lam = read_lambda_schedule(sysB_lam_file)
        validate_protein_lambda_schedule(
            sysB_lam,
            source=f"production protein-leg lambda schedule {sysB_lam_file}",
        )

    os.makedirs(args.outdir, exist_ok=True)

    records, metadata = read_records(args.energy_matrix, include_metadata=True)
    if len(records) == 0:
        raise ValueError(f"No records found in: {args.energy_matrix}")

    generate_post_analysis_figures(
        energy_records=records,
        metadata=metadata,
        analysis_dir=args.outdir,
        prefix=args.prefix,
        enable_free_energy=True,
        enable_overlap_matrix=True,
        enable_stacked_bars=True,
        enable_replica_jumping=True,
        enable_dG_vs_replica_pairs=True,
        enable_flow_profile=True,
        enable_ladder_span=True,
    )

    if prepare_dir is not None:
        boresch_cfg_file = f"{prepare_dir}/{BORESCH_CFG_JSON}"
        if not os.path.exists(boresch_cfg_file):
            raise FileNotFoundError(f"Missing required input: {boresch_cfg_file}")
        calc_restraints(
            boresch_cfg_json=boresch_cfg_file,
            outdir=args.outdir,
        )


if __name__ == "__main__":
    main()
