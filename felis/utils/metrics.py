# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    R2: float
    PearsonR: float
    Spearman: float
    KendallTau: float
    Shift: float
    RMSE: float
    MAE: float
    CorrectOrderRatio: float


def calculate_percentage_correctly_ordered(pred: np.ndarray, label: np.ndarray, thresh: float):
    """
    Calculates the percentage of correctly ordered pairs in 'pred' among all
    pairs where the label difference is greater than thresh.

    Args:
        pred (np.ndarray): 1D array of predicted values.
        label (np.ndarray): 1D array of true label values.

    Returns:
        float: The percentage of correctly ordered pairs meeting the criterion.
    """
    # Ensure inputs are NumPy arrays
    pred = np.asarray(pred)
    label = np.asarray(label)

    if pred.shape != label.shape or pred.ndim != 1:
        raise ValueError("Input arrays must be 1D and have the same shape.")

    n = len(label)
    total_relevant_pairs = 0
    correctly_ordered_pairs = 0

    # Iterate through all unique pairs of indices (i, j)
    for i in range(n):
        for j in range(i + 1, n):
            # 1. Check if the absolute difference in labels is greater than 2
            if abs(label[i] - label[j]) > thresh:
                total_relevant_pairs += 1

                # 2. Check if the pair is correctly ordered
                # This condition is true if both preds and labels are in the
                # same order (both increasing or both decreasing).
                label_order = label[i] - label[j]
                pred_order = pred[i] - pred[j]

                if (label_order > 0 and pred_order > 0) or \
                   (label_order < 0 and pred_order < 0):
                    correctly_ordered_pairs += 1

    # Avoid division by zero if no pairs meet the criterion
    if total_relevant_pairs == 0:
        return 0.0

    percentage = (correctly_ordered_pairs * 1.0 / total_relevant_pairs)
    return percentage


def get_metrics(
        pred: np.ndarray,
        label: np.ndarray,
        thresh: float = 2.0,  # default to 2.0 kcal/mol
) -> Metrics:
    # x is label, y is pred
    _, _, r_value, _, _ = stats.linregress(label, pred)
    r2 = r_value**2
    pearson_r = stats.pearsonr(label, pred).statistic
    spearman = stats.spearmanr(label, pred).statistic
    kendall_tau = stats.kendalltau(label, pred).statistic
    shift = np.mean(label - pred)
    rmse = np.sqrt(np.mean((label - (pred + shift))**2))
    mae = np.mean(np.abs(label - (pred + shift)))
    correct_order_ratio = calculate_percentage_correctly_ordered(pred, label, thresh)

    return Metrics(
        R2=r2,
        PearsonR=pearson_r,
        Spearman=spearman,
        KendallTau=kendall_tau,
        Shift=shift,
        RMSE=rmse,
        MAE=mae,
        CorrectOrderRatio=correct_order_ratio,
    )
