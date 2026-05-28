# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", type=str, required=True)
ap.add_argument("--xml-system", type=str, required=True)
ap.add_argument("--xml-state", type=str, required=True)
ap.add_argument("--xml-integrator", type=str, required=True)
ap.add_argument("--out-stem", type=str, required=True)
ap.add_argument("--nsteps", type=int, default=30)
ap.add_argument("--ninterval", type=int, default=1)
ap.add_argument("--platform", type=str, default="CUDA")
ap.add_argument("--save-force", type=int, default=0)
args = ap.parse_args()

from felis.utils.setup_logger import setup_logger

_logger = setup_logger()

from felis.protocols.dyn.main_ommtools_xml_debug import debug_short_ommtools_run

debug_short_ommtools_run(args.pdb, args.xml_system, args.xml_state, args.xml_integrator, args.out_stem, args.nsteps,
                         args.ninterval, args.platform, args.save_force)
