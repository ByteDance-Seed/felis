# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Provide small NumPy helpers used across the FELIS codebase.
"""

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def argsort_2d_array(a: NDArray) -> NDArray:
    """Return indices that sort a 2D array by value.

    This function sorts `a` in flattened order (row-major) and returns the
    corresponding `(row, col)` index pairs.

    Args:
        a (NDArray): Input array. Shape `(m, n)`.

    Returns:
        NDArray: Integer indices into `a`. Shape `(m*n, 2)` where each row is
        `[row, col]`.

    Raises:
        ValueError: If `a` is not a 2D array.
    """
    _m, n = a.shape
    flat_idx = np.argsort(a, axis=None)
    rows = flat_idx // n
    cols = flat_idx % n
    return np.column_stack((rows, cols))
