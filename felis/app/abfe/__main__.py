# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
from typing import get_args, get_origin

from felis.protocols.abfe.config_types import ABFEInputConfig
from felis.utils import copy_tree
from felis.utils.setup_logger import setup_logger

logger = setup_logger()

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--abfecfg", type=str, default=None, help="common configs; will be overwritten by other args")


def setup_cfg():
    for n, t, _, h in ABFEInputConfig.get_list_of_field_type_default_help_tuples():
        if get_origin(t) == list:
            list_args = get_args(t)
            assert len(list_args) == 1, f"list type {t} should have args of length 1"
            ap.add_argument(f"--{n}", type=list_args[0], nargs="*", default=None, help=h)
        else:
            ap.add_argument(f"--{n}", type=t, default=None, help=h)
    args = ap.parse_args()

    abipt_cfg = ABFEInputConfig.from_file(args.abfecfg)
    abipt_cfg.update_by_dict(vars(args))
    abipt_cfg.check()
    return abipt_cfg


def main(abipt_cfg: ABFEInputConfig):

    sdf_stem = Path(abipt_cfg.sdffile).stem
    if abipt_cfg.tmpdir is not None:
        _tmpdir = str(Path(abipt_cfg.tmpdir) / sdf_stem)
        _outdir = str(Path(abipt_cfg.outdir) / sdf_stem)
        if Path(_outdir).exists():
            copy_tree(_outdir, _tmpdir)

    try:
        from felis.protocols.abfe.main_abfe4 import mainfunc
        mainfunc(abipt_cfg)
    except Exception as e:
        logger.error(e)
        raise
    finally:
        if abipt_cfg.tmpdir is not None:
            copy_tree(_tmpdir, _outdir)

    return


if __name__ == "__main__":
    cfg = setup_cfg()
    main(cfg)
