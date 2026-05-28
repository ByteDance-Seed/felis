# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import io
import json

from felis.configs import dump_config
from felis.configs import load_config


def test_load_config_valid_yaml():
    """Test loading a valid YAML configuration."""
    yaml_content = """
key1: value1
key2: 42
nested:
  subkey: subvalue
list_key:
  - item1
  - item2
"""
    file_handle = io.StringIO(yaml_content)
    result = load_config(file_handle)

    assert isinstance(result, dict)
    assert result["key1"] == "value1"
    assert result["key2"] == 42
    assert result["nested"]["subkey"] == "subvalue"
    assert result["list_key"] == ["item1", "item2"]


def test_load_config_empty_yaml():
    """Test loading an empty YAML file."""
    file_handle = io.StringIO("")
    result = load_config(file_handle)

    # yaml.safe_load returns None for empty files
    assert result is None


def test_load_config_simple_types():
    """Test loading various simple YAML types."""
    yaml_content = """
string_value: "hello"
integer: 123
float_val: 3.14
boolean_true: true
boolean_false: false
null_value: null
"""
    file_handle = io.StringIO(yaml_content)
    result = load_config(file_handle)

    assert result["string_value"] == "hello"
    assert result["integer"] == 123
    assert result["float_val"] == 3.14
    assert result["boolean_true"] is True
    assert result["boolean_false"] is False
    assert result["null_value"] is None


def test_dump_config_default_indent():
    """Test dumping config with default indent (-1, no pretty print)."""
    data = {"key1": "value1", "key2": 42}
    file_handle = io.StringIO()

    dump_config(data, file_handle)

    # Reset file handle to read back
    file_handle.seek(0)
    result = file_handle.read()

    # Should be compact JSON (no newlines or spaces after colons/commas)
    expected = json.dumps(data)
    assert result == expected


def test_dump_config_with_indent():
    """Test dumping config with explicit indent for pretty printing."""
    data = {"key1": "value1", "nested": {"subkey": "subvalue"}}
    file_handle = io.StringIO()

    dump_config(data, file_handle, indent=2)

    file_handle.seek(0)
    result = file_handle.read()

    # Should be pretty-printed with 2-space indentation
    expected = json.dumps(data, indent=2)
    assert result == expected


def test_dump_config_indent_zero():
    """Test dumping config with indent=0 (newline after each item)."""
    data = {"key1": "value1", "key2": "value2"}
    file_handle = io.StringIO()

    dump_config(data, file_handle, indent=0)

    file_handle.seek(0)
    result = file_handle.read()

    # With indent=0, each item should be on its own line with 0-space indent
    expected = json.dumps(data, indent=0)
    assert result == expected


def test_dump_config_nested_data():
    """Test dumping nested dictionary data."""
    data = {"level1": {"level2": {"level3": "deep_value"}, "list_key": [1, 2, 3]}}
    file_handle = io.StringIO()

    dump_config(data, file_handle, indent=2)

    file_handle.seek(0)
    result = file_handle.read()

    # Verify it's valid JSON and structure is preserved
    parsed = json.loads(result)
    assert parsed["level1"]["level2"]["level3"] == "deep_value"
    assert parsed["level1"]["list_key"] == [1, 2, 3]


def test_dump_config_special_values():
    """Test dumping config with special values (None, bool, etc.)."""
    data = {"none_value": None, "true_value": True, "false_value": False, "int_value": 42, "float_value": 3.14159}
    file_handle = io.StringIO()

    dump_config(data, file_handle)

    file_handle.seek(0)
    result = file_handle.read()

    # Verify JSON is valid and values are preserved
    parsed = json.loads(result)
    assert parsed["none_value"] is None
    assert parsed["true_value"] is True
    assert parsed["false_value"] is False
    assert parsed["int_value"] == 42
    assert parsed["float_value"] == 3.14159


def test_dump_config_empty_dict():
    """Test dumping an empty dictionary."""
    data = {}
    file_handle = io.StringIO()

    dump_config(data, file_handle)

    file_handle.seek(0)
    result = file_handle.read()

    assert result == "{}"


def test_roundtrip_load_dump():
    """Test roundtrip: load YAML, dump to JSON, verify data integrity."""
    yaml_content = """
config_name: test_config
version: 1.0
settings:
  debug: true
  timeout: 30
items:
  - id: 1
    name: first
  - id: 2
    name: second
"""
    # Load from YAML
    input_handle = io.StringIO(yaml_content)
    data = load_config(input_handle)

    # Dump to JSON
    output_handle = io.StringIO()
    dump_config(data, output_handle, indent=2)

    # Verify JSON output
    output_handle.seek(0)
    json_result = output_handle.read()
    parsed = json.loads(json_result)

    assert parsed["config_name"] == "test_config"
    assert parsed["version"] == 1.0
    assert parsed["settings"]["debug"] is True
    assert parsed["settings"]["timeout"] == 30
    assert len(parsed["items"]) == 2
    assert parsed["items"][0]["name"] == "first"
