# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import os

import openmm

from RelocationMCPlugin import ExchangeResult, RelocationMC


default_plugin_directory = openmm.Platform.getDefaultPluginsDirectory()
plugin_directory = os.environ.get(
    "RELOCATIONMC_PLUGIN_DIR", os.environ.get("OPENMM_PLUGIN_DIR", default_plugin_directory)
)
load_cuda_plugin = os.environ.get("RELOCATIONMC_SMOKE_LOAD_CUDA", "1") != "0"
if load_cuda_plugin:
    openmm.Platform.loadPluginsFromDirectory(default_plugin_directory)
    if os.path.abspath(plugin_directory) != os.path.abspath(default_plugin_directory):
        openmm.Platform.loadPluginsFromDirectory(plugin_directory)
    load_failures = openmm.Platform.getPluginLoadFailures()
    failures = [entry for entry in load_failures if "RelocationMC" in entry]
    if failures:
        raise RuntimeError("RelocationMC native plugin failed to load: " + "; ".join(failures))

force = RelocationMC(300.0, 0.5, 0, 1)
if force.getTemperature() != 300.0 or force.getSphereRadius() != 0.5:
    raise RuntimeError("RelocationMC Python wrapper returned unexpected parameters")
if force.getProtocolVersion() != RelocationMC.MTM:
    raise RuntimeError("RelocationMC Python wrapper did not default to MTM")
if force.getUseCavityBias():
    raise RuntimeError("RelocationMC MTM default unexpectedly enabled cavity bias")
if force.getProtocolVersion() != RelocationMC.MTM:
    raise RuntimeError("RelocationMC Python wrapper did not retain MTM")
result = ExchangeResult()
if result.protocolVersion != 0 or result.acceptanceProbability != 0.0:
    raise RuntimeError("ExchangeResult MTM diagnostics did not default to zero")

print("RelocationMCPlugin installation test passed.")
