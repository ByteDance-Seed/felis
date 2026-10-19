# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: LGPL-3.0-or-later

import math
import os

import openmm
from openmm import unit

import PerturbNonbondedPlugin
from PerturbNonbondedPlugin import PerturbNonbondedForce


default_plugin_directory = openmm.Platform.getDefaultPluginsDirectory()
plugin_directory = os.environ.get(
    "PERTURB_NONBONDED_PLUGIN_DIR", os.environ.get("OPENMM_PLUGIN_DIR", default_plugin_directory)
)
openmm.Platform.loadPluginsFromDirectory(default_plugin_directory)
if os.path.abspath(plugin_directory) != os.path.abspath(default_plugin_directory):
    openmm.Platform.loadPluginsFromDirectory(plugin_directory)
load_cuda_plugin = os.environ.get("PERTURB_NONBONDED_SMOKE_LOAD_CUDA", "1") != "0"
failures = [entry for entry in openmm.Platform.getPluginLoadFailures() if "PerturbNonbonded" in entry]
if not load_cuda_plugin:
    failures = [entry for entry in failures if "CUDA" not in entry]
if failures:
    raise RuntimeError("PerturbNonbonded native plugin failed to load: " + "; ".join(failures))

force = PerturbNonbondedForce()
force.addGlobalParameter("lambdaA_charge", 1.0)
force.addGlobalParameter("lambdaB_charge", 0.0)
force.addGlobalParameter("lambdaC_charge", 0.0)
force.addGlobalParameter("rest2_scale", 1.0)
force.addGlobalParameter("lambdaA_vdw", 1.0)
force.addGlobalParameter("lambdaB_vdw", 0.0)
force.addParticle(0.25, 0.30, 0.10, 0.25, 0.30, 0.10, False, False)
force.addParticle(-0.25, 0.30, 0.10, -0.25, 0.30, 0.10, False, False)
force.setNonbondedMethod(PerturbNonbondedForce.NoCutoff)

system = openmm.System()
system.addParticle(12.0)
system.addParticle(12.0)
system.addForce(force)
integrator = openmm.VerletIntegrator(0.001)
platform_name = "CUDA" if load_cuda_plugin else "Reference"
context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName(platform_name))
context.setPositions([openmm.Vec3(0.0, 0.0, 0.0), openmm.Vec3(0.5, 0.0, 0.0)]*unit.nanometer)
energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
if not math.isfinite(energy):
    raise RuntimeError("PerturbNonbonded installation smoke test produced nonfinite energy")

print("PerturbNonbondedPlugin", PerturbNonbondedPlugin.__version__, "installation test passed on", platform_name)
