# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional, TextIO, Union

from felis.utils.mpi_tools import mpicomm


def get_formatter(with_lineno: bool) -> logging.Formatter:
    if not with_lineno:
        fmt = "[%(asctime)s PID %(process)d] %(levelname)s %(message)s"
    else:
        fmt = "[%(asctime)s PID %(process)d] %(levelname)s %(message)s (%(name)s:%(lineno)d)"
    return logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def get_root_logger(logging_level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging_level)
    return logger


def setup_logger(file_path: Optional[Union[str, TextIO]] = None) -> logging.Logger:

    # disable numexpr pymbar warnings
    logging.getLogger("numexpr.utils").setLevel(logging.ERROR)
    logging.getLogger("pymbar.timeseries").setLevel(logging.ERROR)
    logging.getLogger("pymbar.mbar_solvers").setLevel(logging.ERROR)

    logger = get_root_logger()
    mpi_rank = mpicomm.mpi_rank
    if mpi_rank == 0:
        import sys
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setFormatter(get_formatter(False))
        logger.addHandler(stdout_handler)
    if file_path:
        if isinstance(file_path, str):
            logger_handler = logging.FileHandler(file_path, mode="a")
        else:
            logger_handler = logging.StreamHandler(stream=file_path)
        logger_handler.setFormatter(get_formatter(True))
        logger.addHandler(logger_handler)
    return logger
