# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
from pathlib import Path
from typing import get_args, get_origin

from felis.protocols.abfe.config_types import ABFEInputConfig
from felis.utils import copy_tree
from felis.utils.setup_logger import setup_logger

logger = setup_logger()


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value (true/false, 1/0, yes/no, on/off), got {value!r}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--abfecfg", type=str, default=None, help="common configs; will be overwritten by other args")
    for n, t, _, h in ABFEInputConfig.get_list_of_field_type_default_help_tuples():
        if get_origin(t) == list:
            list_args = get_args(t)
            assert len(list_args) == 1, f"list type {t} should have args of length 1"
            parser.add_argument(f"--{n}", type=list_args[0], nargs="*", default=None, help=h)
        else:
            parser.add_argument(f"--{n}", type=_parse_bool if t is bool else t, default=None, help=h)
    return parser


def setup_cfg(argv=None):
    args = _build_parser().parse_args(argv)

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
        from felis.protocols.abfe_v2.abfe_workflow import mainfunc

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
