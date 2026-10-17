# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: LGPL-3.0-or-later

from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import PerturbNonbondedPlugin
from PerturbNonbondedPlugin import PerturbNonbondedForce
import copy
import os
import time

total_steps = 10000

perturb_plugin_version = PerturbNonbondedPlugin.__version__
print("PerturbNonbondedPlugin Git Version:", perturb_plugin_version)

print("PerturbNonbondedPlugin Energy Consistency Test\n")

pdb_file_path = os.path.join(os.path.dirname(__file__), 'input.pdb')
pdb = PDBFile(pdb_file_path)
forcefield = ForceField('amber19-all.xml', 'amber19/tip3pfb.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)

nonbonded_force = None
nonbonded_force_index = None
force_group_name = dict()
for ii in range(system.getNumForces()):
    force = system.getForce(ii)
    force.setForceGroup(ii)
    force_group_name[ii] = force.getName()
    if isinstance(force, NonbondedForce):
        nonbonded_force = force
        nonbonded_force_index = ii
        force.addGlobalParameter("lambdaA_charge", 1.0)
        force.addGlobalParameter("lambdaB_charge", 0.0)
        force.addGlobalParameter("lambdaC_charge", 0.0)
        force.addGlobalParameter("rest2_scale", 0.0)
        force.addGlobalParameter("lambdaA_vdw", 1.0)
        force.addGlobalParameter("lambdaB_vdw", 0.0)
        break

perturb_nonbonded_force = PerturbNonbondedForce()
# Build a custom Bonded force for 14-interaction pairs in LJ potential
custom14_lj_BondForce = CustomBondForce('4 * epsilon * ((sigmaR)^6 - 1 ) * (sigmaR)^6;sigmaR = (sigma/r);')
custom14_lj_BondForce.addPerBondParameter("sigma")
custom14_lj_BondForce.addPerBondParameter("epsilon")

# Build a custom Nonbonded force for 14-interaction pairs in Coulomb potential
custom14_charge_BondForce = CustomBondForce('138.935456 * chargeProd / r')
custom14_charge_BondForce.addPerBondParameter("chargeProd")

num_particles = nonbonded_force.getNumParticles()
for ii in range(num_particles):
    charge, sigma, epsilon = nonbonded_force.getParticleParameters(ii)
    perturb_nonbonded_force.addParticle(charge, sigma, epsilon, 0.0, sigma, epsilon, False, False) ## isDummy and isREST2
perturb_nonbonded_force.setSoftcoreAlpha(1.0)
perturb_nonbonded_force.addGlobalParameter("lambdaA_charge", 1.0)
perturb_nonbonded_force.addGlobalParameter("lambdaB_charge", 0.0)
perturb_nonbonded_force.addGlobalParameter("lambdaC_charge", 0.0)
perturb_nonbonded_force.addGlobalParameter("rest2_scale", 1.0)
perturb_nonbonded_force.addGlobalParameter("lambdaA_vdw", 1.0)
perturb_nonbonded_force.addGlobalParameter("lambdaB_vdw", 0.0)
perturb_nonbonded_force.setNonbondedMethod(PerturbNonbondedForce.PME)
perturb_nonbonded_force.setCutoffDistance(1*nanometer)

perturb_nonbonded_force.enableEnergyConstantCache(True)
lambdaA_charge_schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
lambdaB_charge_schedule = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
lambdaC_charge_schedule = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
lambdaA_vdw_schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
lambdaB_vdw_schedule = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
rest2_scale_schedule = [1.0, 0.9, 0.8, 0.8, 0.9, 1.0]
perturb_nonbonded_force.setLambdaSchedules(
    lambdaA_charge_schedule, 
    lambdaB_charge_schedule, 
    lambdaC_charge_schedule, 
    lambdaA_vdw_schedule, 
    lambdaB_vdw_schedule, 
    rest2_scale_schedule)

perturb_nonbonded_force.addGlobalParameter("replica_id", 0)



num_exceptions = nonbonded_force.getNumExceptions()
for ii in range(num_exceptions):
    atom1, atom2, chargeProd, sigma, epsilon  = nonbonded_force.getExceptionParameters(ii)
    custom14_lj_BondForce.addBond(atom1, atom2, [sigma, epsilon])
    custom14_charge_BondForce.addBond(atom1, atom2, [chargeProd])

    #if chargeProd._value != 0 or epsilon._value != 0:
    #    print(atom1, atom2, chargeProd, sigma, epsilon) 

    # Add exception to perturb_nonbonded_force
    perturb_nonbonded_force.addException(atom1, atom2, 0.0, sigma, 0.0)

custom14_lj_BondForce.setName("custom14_lj_BondForce")
custom14_charge_BondForce.setName("custom14_charge_BondForce")


normal_system = copy.copy(system)
perturb_system = copy.copy(system)


perturb_system.removeForce(nonbonded_force_index)
perturb_system.addForce(custom14_lj_BondForce)
perturb_system.addForce(custom14_charge_BondForce)
perturb_system.addForce(perturb_nonbonded_force)

normal_force_group_name = dict()
for ii, force in enumerate(normal_system.getForces()):
    force.setForceGroup(ii)
    normal_force_group_name[ii] = force.getName()

perturb_force_group_name = dict()
for ii, force in enumerate(perturb_system.getForces()):
    force.setForceGroup(ii)
    perturb_force_group_name[ii] = force.getName()


normal_integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
normal_simulation = Simulation(pdb.topology, normal_system, normal_integrator)
normal_simulation.context.setPositions(pdb.positions)
normal_simulation.minimizeEnergy()

state_reporter = StateDataReporter(stdout, 1000, totalSteps = total_steps, temperature=True, step=True, potentialEnergy=True, speed=True, time=True, progress=True)
normal_simulation.reporters.append(state_reporter)

start_time = time.time()
normal_simulation.step(total_steps)
end_time = time.time()
print("Normal system simulation time:", end_time - start_time)

print("Energy decomposition of Normal system:")
normal_energy = normal_simulation.context.getState(getEnergy=True).getPotentialEnergy()
normal_energies = dict()
for ii in range(normal_system.getNumForces()):
    normal_energies[normal_force_group_name[ii]] = normal_simulation.context.getState(getEnergy=True, groups=1<<ii).getPotentialEnergy()
    print(normal_force_group_name[ii], normal_energies[normal_force_group_name[ii]])

normal_positions = normal_simulation.context.getState(getPositions=True).getPositions()
normal_state = normal_simulation.context.getState(getEnergy=True, getPositions=True)


print("normal_energy:", normal_energy)

print("----------")

print("Energy decomposition of Perturbed system:")

perturb_integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
#cuda_platform = Platform.getPlatform("CUDA")
#platform_properties = {"CudaPrecision": "mixed", "DeviceIndex": "0"}
perturb_simulation = Simulation(pdb.topology, perturb_system, perturb_integrator)
perturb_simulation.context.setPositions(normal_positions)

perturb_energy = perturb_simulation.context.getState(getEnergy=True).getPotentialEnergy()


perturb_energies = dict()
for ii in range(perturb_system.getNumForces()):
    perturb_energies[perturb_force_group_name[ii]] = perturb_simulation.context.getState(getEnergy=True, groups=1<<ii).getPotentialEnergy()
    print(perturb_force_group_name[ii], perturb_energies[perturb_force_group_name[ii]])

#for name, energy in perturb_energies.items():
#    print(name, energy)

print("perturb_energy:", perturb_energy)
print("----------")

#assert abs(normal_energy._value - perturb_energy._value) < 1e-1, "Energy difference between Normal and Perturbed system is too large"
#print("PerturbNonbondedPlugin Installation Test Passed.")


perturb_state_reporter = StateDataReporter(stdout, 1000, totalSteps = total_steps, temperature=True, step=True, potentialEnergy=True, speed=True, time=True, progress=True)
perturb_simulation.reporters.append(perturb_state_reporter)

start_time = time.time()
perturb_simulation.step(total_steps)
end_time = time.time()
print("Perturbed system simulation time:", end_time - start_time)


start_time = time.time()
normal_simulation.context.setPositions(normal_positions)
energy1 = normal_simulation.context.getState(getEnergy=True).getPotentialEnergy()
end_time = time.time()
print("Normal system energy time:", end_time - start_time)

start_time = time.time()
perturb_simulation.context.setParameter("replica_id", 5)
energy2 = perturb_simulation.context.getState(getEnergy=True).getPotentialEnergy()
end_time = time.time()
print("Perturbed system energy time:", end_time - start_time)

