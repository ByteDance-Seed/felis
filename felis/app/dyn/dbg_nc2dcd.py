# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

from felis.utils.omm.format_tools import trj_nc2dcd
from felis.utils.setup_logger import setup_logger

logger = setup_logger()

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", type=str, required=True)
ap.add_argument("--basedir", type=str, required=True)
ap.add_argument("--nc-stem", type=str, required=True)
ap.add_argument("--nstep-per-snapshot", type=int, required=True)
ap.add_argument("--state-index", nargs="+", type=int, default=[0])
ap.add_argument("--discard", type=int, default=0)
args = ap.parse_args()

if __name__ == "__main__":
    basedir = args.basedir
    outdir = args.outdir
    nc_stem = args.nc_stem
    state_index = args.state_index
    nc = f"{basedir}/trj/{nc_stem}.nc"
    sys_pdb = f"{basedir}/prepare/sysB_em.pdb"
    sys_top = f"{basedir}/prepare/sysB.top"
    trj_dcd = f"{outdir}/{nc_stem}-{'-'.join(map(str, state_index))}.dcd"

    os.makedirs(outdir, exist_ok=True)
    trj_nc2dcd(nc=nc,
               state_index=state_index,
               discard=args.discard,
               sys_pdb=sys_pdb,
               sys_top=sys_top,
               trj_dcd=trj_dcd,
               nstep_per_snapshot=args.nstep_per_snapshot)
