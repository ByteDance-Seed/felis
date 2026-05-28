# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy

from felis.configs import GlobalKeys
from felis.utils import mkdir_and_get_filepath
from felis.utils.mpi_tools import mpicomm
from felis.utils.setup_logger import setup_logger


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, nargs="*", default=[])
    ap.add_argument("--tkv", type=str, nargs="*", default=[])
    ap.add_argument("--rextkv", type=str, nargs="*", default=[])
    return ap


def mainfunc(argv: list[str], protocol_mainfunc) -> None:
    args = _build_parser().parse_args(argv)

    gk0 = GlobalKeys()
    for icfg in args.cfg:
        gk0.update_by_cfg(icfg)
    for itkv in args.tkv:
        gk0.update_by_tkv(itkv)

    gk_list: list[GlobalKeys] = []
    for irextkv in args.rextkv:
        igk = copy.deepcopy(gk0)
        igk.update_by_comma_sep_tkv(irextkv)
        igk.check()
        gk_list.append(igk)

    for idx, igk in enumerate(gk_list):
        igk.integrator.randomseed += idx
        igk.check()

    ext = f".rank{mpicomm.mpi_rank:02d}.rexlog"
    gk = gk_list[0]
    newlog = mkdir_and_get_filepath(basedir=None, dirname=gk.dir.trj, filename=gk.filename.stem, ext=ext)
    logger = setup_logger(newlog)

    try:
        protocol_mainfunc(gk_list)
    except Exception as e:
        logger.error(e)
        raise


if __name__ == "__main__":
    from felis.protocols.dyn.main_replica_exchange import mainfunc as _protocol_mainfunc
    mainfunc(None, _protocol_mainfunc)
