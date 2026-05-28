# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from felis.utils.numpy_tools import argsort_2d_array


def test_argsort_2d_array_returns_row_col_indices():
    a = np.array([[3, 1], [4, 2]])
    idx = argsort_2d_array(a)

    assert idx.shape == (4, 2)
    assert idx.dtype.kind in ("i", "u")

    expected = np.array([
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
    ])
    assert np.array_equal(idx, expected)


def test_argsort_2d_array_rejects_non_2d():
    with pytest.raises(ValueError):
        argsort_2d_array(np.array([1, 2, 3]))
