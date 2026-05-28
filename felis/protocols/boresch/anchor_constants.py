# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

FREQ_THRESHOLD = 0.4
VALID_ANGLE_RANGE = (45.0, 135.0)
EXCLUDED_SMARTS = {
    "[#16](=O)=O": [0, 1, 2],
    "C(=O)[O!R]": [1, 2],
    "C(=O)[O-]": [1, 2],
}  # S, O in SO2; O in COO-/COOH
SORT_KEY_RESOLUTION_THRESHOLD = 0.02
FREQ_THRESHOLD_DIST = 0.15
PROLIF_JSON = "sys_prolif.json"
