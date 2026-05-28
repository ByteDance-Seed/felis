# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

logger = logging.getLogger(__name__)

electrostatic_lambda_recipes = {
    "e05": [0.00, 0.25, 0.50, 0.75, 1.00],  # 5 states
    "e11": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 11 states
    "e23": [
        0.000, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.325, 0.350, 0.375, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650,
        0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 1.000
    ],  # 23 states
    "e29": [
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76,
        0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00
    ],  # 29 states
    "e31": [
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.36, 0.38, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68,
        0.72, 0.76, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00
    ],  # 31 states
}

vdw_lambda_recipes = {
    "v05": [0.00, 0.25, 0.50, 0.75, 1.00],  # 5 states
    "v18": [
        0.000, 0.100, 0.150, 0.200, 0.220, 0.240, 0.270, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.700, 0.800,
        0.900, 1.000
    ],  # 18 states
    "v20": [
        0.000, 0.100, 0.150, 0.200, 0.220, 0.240, 0.270, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700,
        0.750, 0.800, 0.900, 1.000
    ],  # 20 states
    "v24": [
        0.000, 0.050, 0.100, 0.150, 0.180, 0.200, 0.220, 0.240, 0.270, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600,
        0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 1.000
    ],  # 24 states
    "v33": [
        0.000, 0.050, 0.100, 0.150, 0.180, 0.210, 0.240, 0.270, 0.300, 0.330, 0.360, 0.390, 0.420, 0.450, 0.480, 0.510,
        0.540, 0.570, 0.600, 0.630, 0.660, 0.690, 0.720, 0.750, 0.780, 0.810, 0.840, 0.870, 0.900, 0.930, 0.960, 0.980,
        1.000
    ],  # 33 states
    "v45": [
        0.000, 0.040, 0.080, 0.115, 0.150, 0.180, 0.210, 0.240, 0.270, 0.300, 0.330, 0.360, 0.390, 0.420, 0.450, 0.480,
        0.510, 0.540, 0.568, 0.596, 0.624, 0.650, 0.676, 0.702, 0.728, 0.752, 0.776, 0.798, 0.820, 0.840, 0.860, 0.878,
        0.894, 0.908, 0.921, 0.933, 0.945, 0.956, 0.966, 0.975, 0.983, 0.990, 0.996, 0.999, 1.000
    ],  # 45 states
    "v46": [
        0.000, 0.040, 0.080, 0.110, 0.130, 0.150, 0.180, 0.210, 0.240, 0.270, 0.300, 0.330, 0.360, 0.390, 0.420, 0.450,
        0.480, 0.510, 0.540, 0.568, 0.596, 0.624, 0.650, 0.676, 0.702, 0.728, 0.752, 0.776, 0.798, 0.820, 0.840, 0.860,
        0.878, 0.894, 0.908, 0.921, 0.933, 0.945, 0.956, 0.966, 0.975, 0.983, 0.990, 0.996, 0.999, 1.000
    ],  # 46 states
}


def get_ve_lambdas_with_res(recipe: str, e_or_v: str, ascend: bool, reslam: float, suppl_lams: list[float]):
    if e_or_v == "e":
        if recipe in electrostatic_lambda_recipes.keys():
            lambdas = [[1.0, rr, reslam] for rr in electrostatic_lambda_recipes[recipe]]
        else:
            assert suppl_lams, f"suppl_lams should be provided when recipe {recipe} is not in electrostatic_lambda_recipes"
            assert recipe == "esuppl", "elamrecipe must be esuppl to use supplementary_elec_lambda_list"
            lambdas = [[1.0, rr, reslam] for rr in suppl_lams]
    else:
        assert e_or_v == "v"
        if recipe in vdw_lambda_recipes.keys():
            lambdas = [[rr, 0.0, reslam] for rr in vdw_lambda_recipes[recipe]]
        else:
            assert suppl_lams, f"suppl_lams should be provided when recipe {recipe} is not in vdw_lambda_recipes"
            assert recipe == "vsuppl", "vlamrecipe must be vsuppl to use supplementary_vdw_lambda_list"
            lambdas = [[rr, 0.0, reslam] for rr in suppl_lams]

    if not ascend:
        lambdas = lambdas[::-1]
    return lambdas


restraint_lambda_recipes = {
    "r01": [0.00, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 1.00],
    "r02": [0.00, 0.01, 0.03, 0.10, 0.30, 0.50, 0.75, 1.00],
}


def get_r_lambdas_dim3(recipe: str, ascend: bool, vlam: float, elam: float, suppl_lams: list[float]):
    if recipe in restraint_lambda_recipes.keys():
        lambdas = [[vlam, elam, rr] for rr in restraint_lambda_recipes[recipe]]
    else:
        assert suppl_lams, f"suppl_lams should be provided when recipe {recipe} is not in restraint_lambda_recipes"
        assert recipe == "rsuppl", "reslamrecipe must be rsuppl to use supplementary_restraint_lambda_list"
        lambdas = [[vlam, elam, rr] for rr in suppl_lams]
    if not ascend:
        lambdas = lambdas[::-1]
    return lambdas


def split_replica_exchange_jobs(ngpus: int, jobs: list):
    njobs = len(jobs)
    logger.info(f"Grouping {njobs} jobs into {ngpus} GPUs")
    if ngpus == 0:
        return []
    if njobs == 1:
        return [jobs]
    ninterval = njobs - 1

    scratch1 = [0] * ngpus
    count, nremain = 0, ninterval
    while nremain > 0:
        pos = count % ngpus
        if count < ngpus and nremain > 1:
            scratch1[pos] += 2
            nremain -= 2
        else:
            scratch1[pos] += 1
            nremain -= 1
        count += 1

    scratch2 = []
    for n1 in scratch1:
        if n1 > 0:
            scratch2.append([i for i in range(n1 + 1)])
    prev = 0
    for i in range(len(scratch2)):
        for j in range(len(scratch2[i])):
            scratch2[i][j] += prev
        prev = scratch2[i][-1]

    ret = [[jobs[i2] for i2 in s2] for s2 in scratch2]
    return ret
