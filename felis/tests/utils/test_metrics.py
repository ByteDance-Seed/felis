# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from felis.utils.metrics import Metrics, calculate_percentage_correctly_ordered, get_metrics


def test_get_metrics_perfect_correlation():
    pred = np.array([1, 2, 3, 4, 5])
    label = np.array([1, 2, 3, 4, 5])
    metrics = get_metrics(pred, label)
    assert isinstance(metrics, Metrics)
    np.testing.assert_almost_equal(metrics.R2, 1.0)
    np.testing.assert_almost_equal(metrics.PearsonR, 1.0)
    np.testing.assert_almost_equal(metrics.Spearman, 1.0)
    np.testing.assert_almost_equal(metrics.KendallTau, 1.0)
    np.testing.assert_almost_equal(metrics.Shift, 0.0)
    np.testing.assert_almost_equal(metrics.RMSE, 0.0)
    np.testing.assert_almost_equal(metrics.MAE, 0.0)


def test_get_metrics_perfect_correlation_with_shift():
    pred = np.array([2, 3, 4, 5, 6])
    label = np.array([1, 2, 3, 4, 5])
    metrics = get_metrics(pred, label)
    assert isinstance(metrics, Metrics)
    np.testing.assert_almost_equal(metrics.R2, 1.0)
    np.testing.assert_almost_equal(metrics.PearsonR, 1.0)
    np.testing.assert_almost_equal(metrics.Spearman, 1.0)
    np.testing.assert_almost_equal(metrics.KendallTau, 1.0)
    np.testing.assert_almost_equal(metrics.Shift, -1.0)
    np.testing.assert_almost_equal(metrics.RMSE, 0.0)
    np.testing.assert_almost_equal(metrics.MAE, 0.0)


def test_get_metrics_imperfect_correlation():
    pred = np.array([1.1, 2.2, 2.8, 4.3, 5.0])
    label = np.array([1, 2, 3, 4, 5])
    metrics = get_metrics(pred, label)
    assert isinstance(metrics, Metrics)
    np.testing.assert_almost_equal(metrics.R2, 0.9852, decimal=4)
    np.testing.assert_almost_equal(metrics.PearsonR, 0.99258, decimal=4)
    np.testing.assert_almost_equal(metrics.Spearman, 1.0, decimal=4)
    np.testing.assert_almost_equal(metrics.KendallTau, 1.0, decimal=4)
    np.testing.assert_almost_equal(metrics.Shift, -0.08, decimal=4)
    np.testing.assert_almost_equal(metrics.RMSE, 0.1720, decimal=4)
    np.testing.assert_almost_equal(metrics.MAE, 0.144, decimal=4)
    np.testing.assert_almost_equal(metrics.CorrectOrderRatio, 1.0, decimal=4)


def test_calculate_percentage_correctly_ordered_returns_zero_when_no_relevant_pairs():
    # With thresh=2.0, label diffs never exceed 2.0, so there are no relevant pairs.
    pred = np.array([0.0, 1.0, 2.0])
    label = np.array([0.0, 1.0, 2.0])

    assert calculate_percentage_correctly_ordered(pred, label, thresh=2.0) == 0.0


def test_calculate_percentage_correctly_ordered_rejects_non_1d_or_shape_mismatch():
    with pytest.raises(ValueError, match="1D"):
        calculate_percentage_correctly_ordered(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]), thresh=0.0)

    with pytest.raises(ValueError, match="same shape"):
        calculate_percentage_correctly_ordered(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), thresh=0.0)
