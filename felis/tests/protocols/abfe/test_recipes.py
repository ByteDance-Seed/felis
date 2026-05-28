# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from felis.protocols.abfe import recipes


def test_get_ve_lambdas_with_res_e_known_recipe_ascend_and_descend():
    reslam = 0.2

    asc = recipes.get_ve_lambdas_with_res("e05", "e", ascend=True, reslam=reslam, suppl_lams=[])
    desc = recipes.get_ve_lambdas_with_res("e05", "e", ascend=False, reslam=reslam, suppl_lams=[])

    assert len(asc) == 5
    assert asc[0] == [1.0, 0.0, reslam]
    assert asc[-1] == [1.0, 1.0, reslam]
    assert desc == list(reversed(asc))


def test_get_ve_lambdas_with_res_v_known_recipe_ascend_and_descend():
    reslam = 0.7

    asc = recipes.get_ve_lambdas_with_res("v05", "v", ascend=True, reslam=reslam, suppl_lams=[])
    desc = recipes.get_ve_lambdas_with_res("v05", "v", ascend=False, reslam=reslam, suppl_lams=[])

    assert len(asc) == 5
    assert asc[0] == [0.0, 0.0, reslam]
    assert asc[-1] == [1.0, 0.0, reslam]
    assert desc == list(reversed(asc))


def test_get_ve_lambdas_with_res_e_supplementary_requires_esuppl_and_nonempty_list():
    reslam = 0.0

    with pytest.raises(AssertionError):
        recipes.get_ve_lambdas_with_res("not_a_recipe", "e", ascend=True, reslam=reslam, suppl_lams=[])

    with pytest.raises(AssertionError):
        recipes.get_ve_lambdas_with_res("wrong_suppl_name", "e", ascend=True, reslam=reslam, suppl_lams=[0.0, 1.0])

    suppl = [0.0, 0.4, 1.0]
    out = recipes.get_ve_lambdas_with_res("esuppl", "e", ascend=True, reslam=reslam, suppl_lams=suppl)
    assert out == [[1.0, 0.0, reslam], [1.0, 0.4, reslam], [1.0, 1.0, reslam]]


def test_get_ve_lambdas_with_res_v_supplementary_requires_vsuppl_and_nonempty_list():
    reslam = 0.5

    with pytest.raises(AssertionError):
        recipes.get_ve_lambdas_with_res("not_a_recipe", "v", ascend=True, reslam=reslam, suppl_lams=[])

    with pytest.raises(AssertionError):
        recipes.get_ve_lambdas_with_res("wrong_suppl_name", "v", ascend=True, reslam=reslam, suppl_lams=[0.0, 1.0])

    suppl = [0.0, 0.2, 1.0]
    out = recipes.get_ve_lambdas_with_res("vsuppl", "v", ascend=True, reslam=reslam, suppl_lams=suppl)
    assert out == [[0.0, 0.0, reslam], [0.2, 0.0, reslam], [1.0, 0.0, reslam]]


def test_get_r_lambdas_dim3_known_recipe_ascend_and_descend():
    vlam = 0.8
    elam = 0.3
    asc = recipes.get_r_lambdas_dim3("r02", ascend=True, vlam=vlam, elam=elam, suppl_lams=[])
    desc = recipes.get_r_lambdas_dim3("r02", ascend=False, vlam=vlam, elam=elam, suppl_lams=[])

    assert len(asc) == len(recipes.restraint_lambda_recipes["r02"])
    assert asc[0] == [vlam, elam, 0.0]
    assert asc[-1] == [vlam, elam, 1.0]
    assert desc == list(reversed(asc))


def test_get_r_lambdas_dim3_supplementary_requires_rsuppl_and_nonempty_list():
    vlam = 1.0
    elam = 0.0

    with pytest.raises(AssertionError):
        recipes.get_r_lambdas_dim3("not_a_recipe", ascend=True, vlam=vlam, elam=elam, suppl_lams=[])

    with pytest.raises(AssertionError):
        recipes.get_r_lambdas_dim3("wrong_suppl_name", ascend=True, vlam=vlam, elam=elam, suppl_lams=[0.0, 1.0])

    suppl = [0.0, 0.6, 1.0]
    out = recipes.get_r_lambdas_dim3("rsuppl", ascend=True, vlam=vlam, elam=elam, suppl_lams=suppl)
    assert out == [[vlam, elam, 0.0], [vlam, elam, 0.6], [vlam, elam, 1.0]]


def test_get_ve_lambdas_with_res_invalid_selector_raises():
    with pytest.raises(AssertionError):
        recipes.get_ve_lambdas_with_res("v05", "x", ascend=True, reslam=0.0, suppl_lams=[])


@pytest.mark.parametrize(
    "ngpus,jobs,expected",
    [
        (0, [0, 1, 2], []),
        (2, ["only"], [["only"]]),
        (2, [0, 1, 2, 3, 4], [[0, 1, 2], [2, 3, 4]]),
        (3, [0, 1, 2, 3], [[0, 1, 2], [2, 3]]),
    ],
)
def test_split_replica_exchange_jobs_examples(ngpus, jobs, expected):
    assert recipes.split_replica_exchange_jobs(ngpus=ngpus, jobs=jobs) == expected


@pytest.mark.parametrize(
    "ngpus,njobs",
    [
        (1, 2),
        (1, 6),
        (2, 3),
        (2, 8),
        (4, 5),
        (8, 9),
    ],
)
def test_split_replica_exchange_jobs_connectivity_and_coverage(ngpus, njobs):
    jobs = list(range(njobs))
    groups = recipes.split_replica_exchange_jobs(ngpus=ngpus, jobs=jobs)

    assert groups
    assert groups[0][0] == jobs[0]
    assert groups[-1][-1] == jobs[-1]

    for g in groups:
        assert len(g) >= 2
        idx = [jobs.index(x) for x in g]
        assert idx == list(range(min(idx), max(idx) + 1))

    for left, right in zip(groups, groups[1:]):
        assert left[-1] == right[0]

    flattened = [x for g in groups for x in g]
    assert set(flattened) == set(jobs)
