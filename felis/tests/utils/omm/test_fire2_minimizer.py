# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import io
import logging
import sys

import numpy as np
import pytest

logger = logging.getLogger(__name__)


def _require_real_openmm() -> None:
    mm = pytest.importorskip("openmm")
    # In artifact mode, `sitecustomize.py` replaces `openmm` with shims.
    if not hasattr(mm, "CustomIntegrator") or not hasattr(mm, "Platform"):
        pytest.skip("This test requires real OpenMM (not the FELIS testkit shims).")


def _load_water_forcefield(app):
    # Prefer the simplest built-in water FF; fall back to Amber XMLs if needed.
    try:
        return app.ForceField("tip3p.xml")
    except Exception:  # pragma: no cover
        return app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")


def _build_ten_water_pdb(app):
    # Ten TIP3P waters packed closely to create an initial high repulsion energy.
    # Coordinates are in Angstrom.
    # NOTE: This is not a periodic box; it is just a small cluster for a fast CPU test.
    lines = []
    atom_id = 1

    # TIP3P-like geometry in Angstrom, relative to the oxygen.
    rel = {
        "O": (0.000, 0.000, 0.000),
        "H1": (0.957, 0.000, 0.000),
        "H2": (-0.240, 0.927, 0.000),
    }

    # Place oxygens along x, spaced slightly below typical O-O distances.
    # This ensures a deterministic high starting energy that should decrease quickly.
    oo_spacing = 2.4
    for res_id in range(1, 11):
        ox = (res_id - 1) * oo_spacing
        for atom_name, elem in (("O", "O"), ("H1", "H"), ("H2", "H")):
            rx, ry, rz = rel[atom_name]
            x, y, z = ox + rx, ry, rz
            atom_field = f" {atom_name:<3s}"  # PDB atom name field (4 chars)
            lines.append(
                f"ATOM  {atom_id:5d} {atom_field} HOH A{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:>2s}"
            )
            atom_id += 1
        lines.append("TER")

    lines.append("END")
    return app.PDBFile(io.StringIO("\n".join(lines) + "\n"))


def _get_platform(mm):
    for name in ("CPU", "Reference"):
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            pass
    raise RuntimeError("No suitable OpenMM platform found (CPU/Reference).")


def _attach_state_data_reporter(app, simulation):
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            1,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
        ))


def _get_residue_atom_indices(topology):
    residues = []
    for res in topology.residues():
        name_to_idx = {a.name: a.index for a in res.atoms()}
        residues.append((name_to_idx["O"], name_to_idx["H1"], name_to_idx["H2"]))
    return residues


def _pair_dist_nm(pos_nm: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(pos_nm[i] - pos_nm[j]))


def _get_global(integrator, name: str) -> float:
    for i in range(integrator.getNumGlobalVariables()):
        if integrator.getGlobalVariableName(i) == name:
            return float(integrator.getGlobalVariable(i))
    raise KeyError(f"Global variable not found: {name}")


def _build_single_particle_harmonic_simulation(mm, app, unit, integrator, velocity_sign: float):
    topology = app.Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue("MOL", chain)
    topology.addAtom("X", app.Element.getByAtomicNumber(1), residue)

    system = mm.System()
    system.addParticle(1.0 * unit.amu)
    force = mm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
    force.addGlobalParameter("k", 1000.0)
    force.addParticle(0, [])
    system.addForce(force)

    simulation = app.Simulation(topology, system, integrator, _get_platform(mm))
    _attach_state_data_reporter(app, simulation)
    simulation.context.setPositions([[1.0, 0.0, 0.0]] * unit.nanometer)
    simulation.context.setVelocities([[velocity_sign, 0.0, 0.0]] * unit.nanometer / unit.picosecond)
    return simulation


def test_fire2_integrator_smoke():
    _require_real_openmm()
    import openmm as mm
    import openmm.unit as unit

    from felis.utils.omm.fire2_minimizer import FIRE2Integrator

    integrator = FIRE2Integrator(dt_start=1.0 * unit.femtoseconds, dt_max=10.0 * unit.femtoseconds, tolerance=1e-6)
    assert isinstance(integrator, mm.CustomIntegrator)
    assert abs(integrator.getConstraintTolerance() - 1e-6) < 1e-12

    names = {integrator.getGlobalVariableName(i) for i in range(integrator.getNumGlobalVariables())}
    assert {"dt_var", "alpha", "n_pos", "P", "fnorm", "vnorm", "sum_f2", "sum_v2"}.issubset(names)


def test_fire2_integrator_logs_configurable_constants(caplog):
    _require_real_openmm()
    import openmm.unit as unit

    from felis.utils.omm.fire2_minimizer import FIRE2Integrator

    caplog.set_level(logging.INFO, logger="felis.utils.omm.fire2_minimizer")
    FIRE2Integrator(
        dt_start=2.0 * unit.femtoseconds,
        dt_max=100.0 * unit.femtoseconds,
        tolerance=1e-6,
        N_min=7,
        f_inc=1.25,
        f_dec=0.4,
        alpha_start=0.3,
        f_alpha=0.5,
    )
    assert any("FIRE2Integrator config" in rec.getMessage() for rec in caplog.records)


def test_fire2_energy_decreases_on_rigid_waters():
    _require_real_openmm()
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    from felis.utils.omm.fire2_minimizer import FIRE2Integrator

    pdb = _build_ten_water_pdb(app)
    ff = _load_water_forcefield(app)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        rigidWater=True,
    )
    integrator = FIRE2Integrator(dt_start=1.0 * unit.femtoseconds, dt_max=10.0 * unit.femtoseconds, tolerance=1e-6)

    simulation = app.Simulation(pdb.topology, system, integrator, _get_platform(mm))
    _attach_state_data_reporter(app, simulation)
    simulation.context.setPositions(pdb.positions)

    # Deterministic start.
    n = system.getNumParticles()
    simulation.context.setVelocities(np.zeros((n, 3)) * unit.nanometer / unit.picosecond)

    e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    simulation.step(20)
    e1 = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    logger.info(f"e0: {e0:.6f}, e1: {e1:.6f}")
    assert e1 < e0


def test_fire2_preserves_rigid_water_geometry():
    _require_real_openmm()
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    from felis.utils.omm.fire2_minimizer import FIRE2Integrator

    pdb = _build_ten_water_pdb(app)
    ff = _load_water_forcefield(app)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        rigidWater=True,
    )
    integrator = FIRE2Integrator(dt_start=1.0 * unit.femtoseconds, dt_max=10.0 * unit.femtoseconds, tolerance=1e-6)
    simulation = app.Simulation(pdb.topology, system, integrator, _get_platform(mm))
    _attach_state_data_reporter(app, simulation)
    simulation.context.setPositions(pdb.positions)

    residues = _get_residue_atom_indices(pdb.topology)
    pos0_nm = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    d0 = [(
        _pair_dist_nm(pos0_nm, o, h1),
        _pair_dist_nm(pos0_nm, o, h2),
        _pair_dist_nm(pos0_nm, h1, h2),
    ) for (o, h1, h2) in residues]

    simulation.step(20)

    pos1_nm = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    d1 = [(
        _pair_dist_nm(pos1_nm, o, h1),
        _pair_dist_nm(pos1_nm, o, h2),
        _pair_dist_nm(pos1_nm, h1, h2),
    ) for (o, h1, h2) in residues]

    tol_nm = 2e-3
    for (a0, b0, c0), (a1, b1, c1) in zip(d0, d1):
        assert abs(a1 - a0) < tol_nm
        assert abs(b1 - b0) < tol_nm
        assert abs(c1 - c0) < tol_nm


def test_fire2_control_values_are_user_settable():
    _require_real_openmm()
    import openmm.unit as unit

    from felis.utils.omm.fire2_minimizer import FIRE2Integrator

    integrator = FIRE2Integrator(
        dt_start=2.0 * unit.femtoseconds,
        dt_max=100.0 * unit.femtoseconds,
        tolerance=1e-6,
        N_min=7,
        f_inc=1.25,
        f_dec=0.4,
        alpha_start=0.3,
        f_alpha=0.5,
    )
    assert integrator.N_min == 7
    assert integrator.f_inc == 1.25
    assert integrator.f_dec == 0.4
    assert integrator.alpha_start == 0.3
    assert integrator.f_alpha == 0.5
    assert abs(_get_global(integrator, "alpha") - 0.3) < 1e-12


def test_fire2_control_values_affect_dt_and_alpha_when_P_positive():
    _require_real_openmm()
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    from felis.utils.omm.fire2_minimizer import FIRE2Integrator

    dt_start = 1.0 * unit.femtoseconds
    integrator = FIRE2Integrator(
        dt_start=dt_start,
        dt_max=100.0 * unit.femtoseconds,
        tolerance=1e-6,
        N_min=0,
        f_inc=1.2,
        alpha_start=0.3,
        f_alpha=0.5,
    )
    simulation = _build_single_particle_harmonic_simulation(mm, app, unit, integrator, velocity_sign=-1.0)
    simulation.step(1)

    assert abs(_get_global(integrator, "dt_var") - (dt_start.value_in_unit(unit.picoseconds) * 1.2)) < 1e-12
    assert abs(_get_global(integrator, "alpha") - (0.3 * 0.5)) < 1e-12


def test_fire2_control_values_affect_dt_and_alpha_when_P_nonpositive():
    _require_real_openmm()
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    from felis.utils.omm.fire2_minimizer import FIRE2Integrator

    dt_start = 1.0 * unit.femtoseconds
    integrator = FIRE2Integrator(
        dt_start=dt_start,
        dt_max=100.0 * unit.femtoseconds,
        tolerance=1e-6,
        f_dec=0.4,
        alpha_start=0.3,
    )
    simulation = _build_single_particle_harmonic_simulation(mm, app, unit, integrator, velocity_sign=1.0)
    simulation.step(1)

    assert abs(_get_global(integrator, "dt_var") - (dt_start.value_in_unit(unit.picoseconds) * 0.4)) < 1e-12
    assert abs(_get_global(integrator, "alpha") - 0.3) < 1e-12
