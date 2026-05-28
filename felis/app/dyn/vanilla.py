# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse

from felis.configs import GlobalKeys
from felis.utils import mkdir_and_get_filepath
from felis.utils.setup_logger import setup_logger


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, nargs="*", default=[])
    ap.add_argument("--tkv", type=str, nargs="*", default=[])
    return ap


def mainfunc(argv: list[str], protocol_mainfunc) -> None:

    args = _build_parser().parse_args(argv)

    gk0 = GlobalKeys()
    for icfg in args.cfg:
        gk0.update_by_cfg(icfg)
    for itkv in args.tkv:
        gk0.update_by_tkv(itkv)
    gk0.check()

    newlog = mkdir_and_get_filepath(basedir=None, dirname=gk0.dir.trj, filename=gk0.filename.stem, ext=".dynlog")
    logger = setup_logger(newlog)

    try:
        protocol_mainfunc(gk0)
    except Exception as e:
        logger.error(f"{type(e)}:{e}")
        raise


if __name__ == "__main__":
    from felis.protocols.dyn.main_vanilla import mainfunc as _protocol_mainfunc
    mainfunc(None, _protocol_mainfunc)
