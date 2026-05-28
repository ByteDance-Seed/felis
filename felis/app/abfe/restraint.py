# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from felis.utils.setup_logger import setup_logger

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", type=str, required=True)
ap.add_argument("--sdffile", type=str, required=True)
ap.add_argument("--k-r-a-dih", type=float, nargs=3, default=[2., 80., 80.])
ap.add_argument("--workdir", type=str, default=None)
ap.add_argument("--systop", type=str, default=None)
ap.add_argument("--trjdcd", type=str, default=None)
ap.add_argument("--atom-ids-json", type=str, default=None)
args = ap.parse_args()

if __name__ == "__main__":
    from felis.protocols.boresch.main_boresch_restraints import generate_boresch_restraints

    outdir = args.outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)

    lig_sdf = str(Path(args.sdffile).absolute())

    kr, ka, kdih = args.k_r_a_dih

    if args.workdir:
        workdir = str(Path(args.workdir).absolute())
    else:
        workdir = None

    if args.systop is None:
        assert workdir
        sys_top = str(Path(workdir) / "prepare/sysB.top")
    else:
        sys_top = str(Path(args.systop).absolute())
    if args.trjdcd is None:
        assert workdir
        trj_dcd = str(Path(workdir) / "trj/sysB_boresch_npt.dcd")
    else:
        trj_dcd = str(Path(args.trjdcd).absolute())
    if args.atom_ids_json is None:
        assert workdir
        atom_ids_json = str(Path(workdir) / "prepare/sysB_atom_ids.json")
    else:
        atom_ids_json = str(Path(args.atom_ids_json).absolute())

    new_log = str(Path(outdir) / "sys_boresch0.log")
    setup_logger(new_log)
    generate_boresch_restraints(kr=kr,
                                ka=ka,
                                kdih=kdih,
                                lig_sdf=lig_sdf,
                                sys_top=sys_top,
                                trj_dcd=trj_dcd,
                                atom_ids_json=atom_ids_json,
                                outdir=outdir)
