# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import pytest

from felis.app.abfe.__main__ import _build_parser


@pytest.mark.parametrize(
    "value, expected",
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_boolean_cli_values(value, expected):
    args = _build_parser().parse_args(["--rmc_enable", value])

    assert args.rmc_enable is expected


@pytest.mark.parametrize(
    "option",
    [
        "--rmc_enable",
        "--rmc_orientations_enabled",
        "--rmc_use_cavity_bias",
        "--rmc_exclude_alchemical_water",
    ],
)
def test_all_boolean_cli_options_parse_false(option):
    args = _build_parser().parse_args([option, "false"])

    assert getattr(args, option.removeprefix("--")) is False


def test_boolean_cli_rejects_unknown_value():
    with pytest.raises(SystemExit) as exc_info:
        _build_parser().parse_args(["--rmc_enable", "sometimes"])

    assert exc_info.value.code == 2
