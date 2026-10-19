#!/usr/bin/env python3
# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Run an OpenMM MD simulation with automatic RelocationMC moves."""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run MD with RelocationMC automatically attempting one move every "
            "exchange-frequency integration steps (1000 by default)."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdb", type=Path, help="PDB used with --forcefield to create a System")
    source.add_argument("--system-xml", type=Path, help="Serialized OpenMM System XML")
    parser.add_argument(
        "--state-xml", type=Path,
        help="Serialized State XML; required with --system-xml and supplies positions/box",
    )
    parser.add_argument(
        "--topology-pdb", type=Path,
        help="PDB providing topology for --system-xml (must match the serialized System)",
    )
    parser.add_argument(
        "--forcefield", action="append", default=[],
        help="ForceField XML for --pdb; repeat as needed (for example amber14-all.xml)",
    )
    parser.add_argument("--ligand-center-indices", help="Comma-separated zero-based ligand center atom indices")
    parser.add_argument("--ligand-resname", help="Select the unique ligand residue by name when indices are omitted")
    parser.add_argument(
        "--ligand-center-atom-names",
        help="Optional comma-separated atom names within --ligand-resname; default uses all atoms",
    )
    parser.add_argument("--labile-resname", default="HOH", help="Residue name for homogeneous labile molecules")
    parser.add_argument(
        "--labile-atom-names", default="O,H1,H2",
        help="Ordered comma-separated site names shared by every labile molecule",
    )
    parser.add_argument("--reference-atom-name", default="O", help="Labile site used for region classification")
    parser.add_argument(
        "--site-parameters",
        help=(
            "Optional semicolon-separated charge,sigma_nm,epsilon_kJmol triplets. "
            "By default they are read from the first labile molecule."
        ),
    )
    parser.add_argument(
        "--reference-positions",
        help=(
            "Optional semicolon-separated x,y,z offsets in nm. By default rigid "
            "offsets are inferred from the first labile molecule."
        ),
    )
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in kelvin")
    parser.add_argument("--sphere-radius", type=float, default=0.5, help="Inner sphere radius in nm")
    parser.add_argument("--exchange-frequency", type=int, default=1000, help="MD steps between automatic MC attempts")
    parser.add_argument("--candidate-count", type=int, default=300,
                        help="Candidate poses per MTM attempt, or anchor sites for the Legacy protocol")
    parser.add_argument("--protocol", choices=("mtm", "legacy"), default="mtm",
                        help="Relocation protocol; MTM is the statistically balanced default")
    parser.add_argument("--seed", type=int, default=20260829, help="RelocationMC random seed")
    parser.add_argument("--disable-orientations", action="store_true", help="Use only the identity orientation")
    parser.add_argument("--disable-cavity-bias", action="store_true", help="Disable the cavity volume term")
    parser.add_argument("--legacy-asymmetric-energies", action="store_true",
                        help="Use the historical RF-delete/PME-insert estimator instead of PME environment corrections")
    parser.add_argument("--steps", type=int, default=10000, help="Total MD integration steps")
    parser.add_argument("--report-interval", type=int, default=1000, help="Progress/energy reporting interval")
    parser.add_argument("--timestep-fs", type=float, default=2.0, help="MD time step in femtoseconds")
    parser.add_argument("--friction-ps", type=float, default=1.0, help="Langevin friction in 1/ps")
    parser.add_argument("--cutoff-nm", type=float, default=0.9, help="PME real-space cutoff for PDB-created Systems")
    parser.add_argument("--precision", choices=("single", "mixed", "double"), default="mixed")
    parser.add_argument("--device-index", help="Optional CUDA device index")
    args = parser.parse_args()

    if args.system_xml and (args.state_xml is None or args.topology_pdb is None):
        parser.error("--system-xml requires both --state-xml and --topology-pdb")
    if args.pdb and not args.forcefield:
        parser.error("--pdb requires at least one --forcefield")
    for value, name in ((args.steps, "--steps"), (args.report_interval, "--report-interval"),
                        (args.exchange_frequency, "--exchange-frequency"),
                        (args.candidate_count, "--candidate-count")):
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.protocol == "mtm" and args.disable_orientations:
        parser.error("--disable-orientations is only valid with --protocol legacy")
    if args.protocol == "mtm" and not args.disable_cavity_bias:
        args.disable_cavity_bias = True
    if args.protocol == "mtm" and args.legacy_asymmetric_energies:
        parser.error("--legacy-asymmetric-energies is only valid with --protocol legacy")
    return args


def csv_names(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_triplets(value, label):
    try:
        rows = [[float(component) for component in row.split(",")] for row in value.split(";")]
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError(f"{label} must contain semicolon-separated x,y,z triplets")
    return rows


def find_ligand_center_atoms(topology, explicit, residue_name, atom_names):
    if explicit:
        indices = [int(value) for value in csv_names(explicit)]
        if not indices or min(indices) < 0 or max(indices) >= topology.getNumAtoms():
            raise ValueError("--ligand-center-indices contains an out-of-range atom")
        return indices
    if not residue_name:
        raise ValueError("provide --ligand-center-indices or --ligand-resname")
    residues = [residue for residue in topology.residues() if residue.name == residue_name]
    if len(residues) != 1:
        raise ValueError(f"expected exactly one {residue_name!r} ligand residue, found {len(residues)}")
    selected_names = set(csv_names(atom_names)) if atom_names else None
    indices = [atom.index for atom in residues[0].atoms() if selected_names is None or atom.name in selected_names]
    if selected_names is not None and len(indices) != len(selected_names):
        raise ValueError("not every --ligand-center-atom-names entry was found in the ligand")
    if not indices:
        raise ValueError("ligand center atom selection is empty")
    return indices


def find_labile_molecules(topology, residue_name, ordered_names, reference_name):
    molecules = []
    for residue in topology.residues():
        if residue.name != residue_name:
            continue
        atoms = {atom.name: atom.index for atom in residue.atoms()}
        missing = [name for name in ordered_names if name not in atoms]
        if missing:
            raise ValueError(f"residue {residue.index} is missing labile sites {missing}")
        if reference_name not in atoms or reference_name not in ordered_names:
            raise ValueError("--reference-atom-name must be one of --labile-atom-names")
        molecules.append(([atoms[name] for name in ordered_names], atoms[reference_name]))
    if not molecules:
        raise ValueError(f"no {residue_name!r} labile residues were found")
    return molecules


def convert_standard_nonbonded(system, plugin_module, openmm, unit, cutoff_nm):
    existing = [force for force in system.getForces() if plugin_module.PerturbNonbondedForce.isinstance(force)]
    if existing:
        if len(existing) != 1:
            raise ValueError("System must contain exactly one PerturbNonbondedForce")
        return plugin_module.PerturbNonbondedForce.cast(existing[0])

    standard = [(index, force) for index, force in enumerate(system.getForces())
                if isinstance(force, openmm.NonbondedForce)]
    if len(standard) != 1:
        raise ValueError("System must contain one NonbondedForce or one PerturbNonbondedForce")
    index, source = standard[0]
    if source.getNumParticleParameterOffsets() or source.getNumExceptionParameterOffsets():
        raise ValueError("automatic NonbondedForce conversion does not support parameter offsets")

    target = plugin_module.PerturbNonbondedForce()
    target.setName("perturb_nonbonded")
    target.setForceGroup(source.getForceGroup())
    target.setNonbondedMethod(plugin_module.PerturbNonbondedForce.PME)
    target.setCutoffDistance(cutoff_nm)
    target.setReactionFieldDielectric(source.getReactionFieldDielectric())
    target.setEwaldErrorTolerance(source.getEwaldErrorTolerance())
    target.setUseSwitchingFunction(source.getUseSwitchingFunction())
    if source.getUseSwitchingFunction():
        target.setSwitchingDistance(source.getSwitchingDistance().value_in_unit(unit.nanometer))
    target.setUseDispersionCorrection(source.getUseDispersionCorrection())
    target.setExceptionsUsePeriodicBoundaryConditions(source.getExceptionsUsePeriodicBoundaryConditions())
    target.setReciprocalSpaceForceGroup(source.getReciprocalSpaceForceGroup())

    for particle in range(source.getNumParticles()):
        charge, sigma, epsilon = source.getParticleParameters(particle)
        q = charge.value_in_unit(unit.elementary_charge)
        sig = sigma.value_in_unit(unit.nanometer)
        eps = epsilon.value_in_unit(unit.kilojoule_per_mole)
        target.addParticle(q, sig, eps, q, sig, eps, False, False)
    for exception in range(source.getNumExceptions()):
        atom1, atom2, charge_product, sigma, epsilon = source.getExceptionParameters(exception)
        target.addException(
            atom1, atom2,
            charge_product.value_in_unit(unit.elementary_charge**2),
            sigma.value_in_unit(unit.nanometer),
            epsilon.value_in_unit(unit.kilojoule_per_mole),
        )
    for name, default in (("lambdaA_charge", 1.0), ("lambdaB_charge", 0.0),
                          ("lambdaC_charge", 0.0), ("rest2_scale", 1.0),
                          ("lambdaA_vdw", 1.0), ("lambdaB_vdw", 0.0)):
        target.addGlobalParameter(name, default)
    system.removeForce(index)
    system.addForce(target)
    return target


def infer_site_parameters(force, molecule):
    values = []
    for atom in molecule:
        charge, sigma, epsilon, *_ = force.getParticleParameters(atom)
        values.append((float(charge), float(sigma), float(epsilon)))
    return values


def infer_reference_positions(positions_nm, box_nm, molecule, reference_atom):
    reference = positions_nm[reference_atom]
    lengths = (box_nm[0][0], box_nm[1][1], box_nm[2][2])
    offsets = []
    for atom in molecule:
        delta = [positions_nm[atom][axis]-reference[axis] for axis in range(3)]
        for axis in range(3):
            delta[axis] -= round(delta[axis]/lengths[axis])*lengths[axis]
        offsets.append(delta)
    return offsets


def run(args):
    import openmm
    from openmm import XmlSerializer, unit
    from openmm import app
    import PerturbNonbondedPlugin
    from RelocationMCPlugin import LabileMolSiteParameters, RelocationMC

    topology_pdb = app.PDBFile(str(args.topology_pdb or args.pdb))
    topology = topology_pdb.topology
    if args.system_xml:
        system = XmlSerializer.deserialize(args.system_xml.read_text())
        state = XmlSerializer.deserialize(args.state_xml.read_text())
        positions = state.getPositions()
    else:
        forcefield = app.ForceField(*args.forcefield)
        system = forcefield.createSystem(
            topology, nonbondedMethod=app.PME,
            nonbondedCutoff=args.cutoff_nm*unit.nanometer,
            constraints=app.HBonds,
        )
        state = None
        positions = topology_pdb.positions

    perturb_force = convert_standard_nonbonded(
        system, PerturbNonbondedPlugin, openmm, unit, args.cutoff_nm
    )
    center_atoms = find_ligand_center_atoms(
        topology, args.ligand_center_indices, args.ligand_resname,
        args.ligand_center_atom_names,
    )
    ordered_names = csv_names(args.labile_atom_names)
    molecules = find_labile_molecules(
        topology, args.labile_resname, ordered_names, args.reference_atom_name
    )

    positions_nm = positions.value_in_unit(unit.nanometer)
    box = state.getPeriodicBoxVectors() if state is not None else topology.getPeriodicBoxVectors()
    if box is None:
        raise ValueError("RelocationMC requires periodic box vectors")
    box_nm = box.value_in_unit(unit.nanometer)

    site_values = (parse_triplets(args.site_parameters, "--site-parameters")
                   if args.site_parameters else infer_site_parameters(perturb_force, molecules[0][0]))
    reference_values = (parse_triplets(args.reference_positions, "--reference-positions")
                        if args.reference_positions else
                        infer_reference_positions(positions_nm, box_nm, molecules[0][0], molecules[0][1]))
    if len(site_values) != len(ordered_names) or len(reference_values) != len(ordered_names):
        raise ValueError("site parameters and reference positions must match --labile-atom-names")

    relocation = RelocationMC(
        args.temperature, args.sphere_radius,
        args.exchange_frequency, args.candidate_count,
    )
    if args.protocol == "legacy":
        relocation.setProtocolVersion(RelocationMC.Legacy)
    relocation.setRandomSeed(args.seed)
    relocation.setLigandAtoms(center_atoms)
    relocation.setLabileMolSiteParameters([
        LabileMolSiteParameters(charge, sigma, epsilon)
        for charge, sigma, epsilon in site_values
    ])
    # Exact PME mode requires every intramolecular labile-atom pair to be a
    # zeroed PerturbNonbonded exception; nonzero 1-4 terms are unsupported.
    relocation.setLabileMolReferencePositions([
        openmm.Vec3(x, y, z) for x, y, z in reference_values
    ])
    relocation.setOrientationsEnabled(not args.disable_orientations)
    relocation.setUseCavityBias(not args.disable_cavity_bias)
    relocation.setUseLegacyAsymmetricEnergies(args.legacy_asymmetric_energies)
    for atom_indices, reference_atom in molecules:
        relocation.addLabileMol(atom_indices, reference_atom)
    system.addForce(relocation)

    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature*unit.kelvin,
        args.friction_ps/unit.picosecond,
        args.timestep_fs*unit.femtosecond,
    )
    properties = {"Precision": args.precision, "UseCpuPme": "false"}
    if args.device_index is not None:
        properties["DeviceIndex"] = args.device_index
    simulation = app.Simulation(
        topology, system, integrator,
        openmm.Platform.getPlatformByName("CUDA"), properties,
    )
    if state is not None:
        simulation.context.setState(state)
    else:
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(args.temperature*unit.kelvin, args.seed)

    # Generate a fresh PME grid before integration. RelocationMC then runs through
    # updateContextState every exchange_frequency steps; do not call attemptExchange().
    initial = simulation.context.getState(getEnergy=True)
    print(f"Initial potential energy: {initial.getPotentialEnergy()}")
    print(
        f"Registered {len(molecules)} {args.labile_resname} molecules; automatic MC "
        f"frequency={args.exchange_frequency} MD steps, candidates={args.candidate_count}"
    )
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout, args.report_interval, step=True, time=True,
        potentialEnergy=True, temperature=True, speed=True,
    ))
    remaining = args.steps
    while remaining:
        block = min(remaining, args.report_interval)
        simulation.step(block)
        remaining -= block


def main():
    args = parse_args()
    try:
        run(args)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
