# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import sys
import types
from pathlib import Path

import pytest


def _import_omm_system_or_skip():
    pytest.importorskip("openmm")
    pytest.importorskip("ase")
    import felis.utils.omm.omm_system as omm_system

    return omm_system


def test_get_system_topology_passes_hydrogen_mass(monkeypatch, tmp_path):
    omm_system = _import_omm_system_or_skip()
    from felis.configs import GlobalKeys

    calls = {}

    class FakePDB:

        def __init__(self):
            self.positions = []
            self.topology = self

        def getPeriodicBoxVectors(self):
            return [[], [], []]

    class FakeTopFile:

        def __init__(self, filename, periodicBoxVectors=None):
            self.filename = filename
            self.periodicBoxVectors = periodicBoxVectors
            self.topology = object()

        def createSystem(self, **kwargs):
            calls["createSystem"] = kwargs
            return omm_system.System()

    monkeypatch.setattr(omm_system, "_get_pdb", lambda _path: FakePDB())
    monkeypatch.setattr(omm_system, "_patch_openmm_gromacstopfile_hmr_water_resnames", lambda: None)
    monkeypatch.setattr(omm_system, "GromacsTopFile", FakeTopFile)
    monkeypatch.setattr(omm_system, "get_force_from_system_by_name", lambda _system, _name: None)

    top_path = tmp_path / "sys.top"
    crd_path = tmp_path / "sys.pdb"
    top_path.touch()
    crd_path.touch()

    gk = GlobalKeys()
    gk.filename.sys = str(top_path)
    gk.filename.crd = str(crd_path)
    gk.integrator.hmr = 3.0

    system, topology = omm_system.get_system_topology(gk)

    assert system is not None
    assert topology is not None
    hydrogen_mass = calls["createSystem"]["hydrogenMass"]
    assert hydrogen_mass.value_in_unit(omm_system.unit.amu) == 3.0


def test_abfe_input_config_hmr_tkv(tmp_path):
    from felis.protocols.abfe.config_types import ABFEInputConfig

    acfg = ABFEInputConfig(
        sdffile=str(tmp_path / "ligand.sdf"),
        itpfile=str(tmp_path / "ligand.itp"),
        outdir=str(tmp_path),
        md_integrator_dt_ps=0.004,
        md_integrator_hmr=3.0,
    )
    acfg.check()
    assert acfg.hmr_tkv() == "f:integrator.hmr:3.0"
    assert acfg.dt_tkv() == "f:integrator.dt_ps:0.004"

    acfg.md_integrator_hmr = None
    assert acfg.hmr_tkv() == "f:integrator.hmr:"


def test_patch_openmm_gromacstopfile_hmr_water_resnames_rewrites_real_openmm(monkeypatch):
    pytest.importorskip("openmm.app.gromacstopfile")
    import openmm.app.gromacstopfile as gtf

    omm_system = _import_omm_system_or_skip()

    original_create_system = gtf.GromacsTopFile.createSystem
    try:
        omm_system._patch_openmm_gromacstopfile_hmr_water_resnames()

        patched = gtf.GromacsTopFile.createSystem
        assert ("HOH", "SOL", "CRW") in patched.__code__.co_consts
        assert getattr(patched, "_apply_hmr_water_resname_patch", False) is True
    finally:
        gtf.GromacsTopFile.createSystem = original_create_system


def test_patch_openmm_gromacstopfile_hmr_water_resnames_errors_when_no_hoh_checks(monkeypatch):
    import inspect
    import types

    omm_system = _import_omm_system_or_skip()

    def fake_create_system(self, rigidWater=True, hydrogenMass=None):
        return hydrogenMass

    fake_gtf_module = types.SimpleNamespace(
        GromacsTopFile=types.SimpleNamespace(createSystem=fake_create_system)
    )
    fake_source = "\n".join(
        [
            "def createSystem(self, rigidWater=True, hydrogenMass=None):",
            "    if rigidWater and atom2.residue.name == 'WAT':",
            "        return None",
            "    return hydrogenMass",
        ]
    )
    import openmm.app

    monkeypatch.setattr(inspect, "getsource", lambda _obj: fake_source)
    monkeypatch.setattr(inspect, "getsourcefile", lambda _obj: "/tmp/fake_gromacstopfile.py")
    monkeypatch.setattr(openmm.app, "gromacstopfile", fake_gtf_module)

    try:
        omm_system._patch_openmm_gromacstopfile_hmr_water_resnames()
    except ImportError as exc:
        assert "no rigidWater HOH residue-name checks were found" in str(exc)
    else:
        raise AssertionError("Expected ImportError when no HOH rigidWater checks are patchable")


@pytest.mark.parametrize("ligand_charge", [1.0, -1.0])
def test_zero_charge_change_skips_alchemical_water_selection(
    monkeypatch, ligand_charge
):
    omm_system = _import_omm_system_or_skip()

    class FakePerturbNonbondedForce:
        PME = 1

        def __init__(self):
            self.particles = []

        def addParticle(self, *parameters):
            self.particles.append(parameters)

        def setSoftcoreAlpha(self, _value):
            pass

        def setNonbondedMethod(self, _method):
            pass

        def setCutoffDistance(self, _distance):
            pass

        def enableEnergyConstantCache(self, _enabled):
            pass

        def setLambdaSchedules(self, *_schedules):
            pass

        def addGlobalParameter(self, _name, _value):
            pass

        def addException(self, *_parameters):
            pass

    fake_plugin = types.SimpleNamespace(
        PerturbNonbondedForce=FakePerturbNonbondedForce
    )
    monkeypatch.setitem(sys.modules, "PerturbNonbondedPlugin", fake_plugin)

    def fail_if_water_count_is_requested(_charge):
        raise AssertionError("zero compensation must not select alchemical waters")

    monkeypatch.setattr(
        omm_system, "get_num_alchemwat", fail_if_water_count_is_requested
    )

    nbforce = omm_system.NonbondedForce()
    charges = [ligand_charge, -0.8, 0.4, 0.4]
    for charge in charges:
        nbforce.addParticle(
            charge * omm_system.unit.elementary_charge,
            0.3 * omm_system.unit.nanometer,
            0.1 * omm_system.unit.kilojoule_per_mole,
        )

    schedules = {
        "lambdaA_charge": [1.0],
        "lambdaB_charge": [0.0],
        "lambdaC_charge": [0.0],
        "lambdaA_vdw": [1.0],
        "lambdaB_vdw": [0.0],
        "rest2_scale": [1.0],
    }
    perturb_force, _, _ = omm_system._make_perturb_nonbonded_force(
        nbforce=nbforce,
        natoms=4,
        alchem_atoms=[0],
        lambda_schedules=schedules,
        lambda_index=0,
        OHH_list=[(1, 2, 3)],
        enable_alchem_water=True,
    )

    assert len(perturb_force.particles) == 4
    for atom_index in (1, 2, 3):
        parameters = perturb_force.particles[atom_index]
        charge_a = parameters[0].value_in_unit(omm_system.unit.elementary_charge)
        charge_b = parameters[3].value_in_unit(omm_system.unit.elementary_charge)
        assert charge_a == pytest.approx(charges[atom_index])
        assert charge_b == pytest.approx(charges[atom_index])


def test_relocation_mc_disabled_does_not_add_force():
    from types import SimpleNamespace

    omm_system = _import_omm_system_or_skip()

    class NoForceSystem:

        def addForce(self, _force):
            raise AssertionError("disabled RelocationMC must not add a Force")

    gk = SimpleNamespace(relocation_mc=SimpleNamespace(enable=False))
    omm_system._add_relocation_mc_force(NoForceSystem(), gk)


def test_relocation_mc_uses_fixed_mtm_protocol(monkeypatch):
    import sys
    from types import ModuleType
    from types import SimpleNamespace

    omm_system = _import_omm_system_or_skip()
    calls = []

    class FakeRelocationMC:
        MTM = 2

        def __init__(self, temperature, radius, frequency, num_candidate_sites):
            calls.append(("construct", temperature, radius, frequency, num_candidate_sites))

        def setRandomSeed(self, seed):
            calls.append(("setRandomSeed", seed))

        def setLigandAtoms(self, atoms):
            calls.append(("setLigandAtoms", atoms))

        def setUseCavityBias(self, enabled):
            calls.append(("setUseCavityBias", enabled))

        def setOrientationsEnabled(self, enabled):
            calls.append(("setOrientationsEnabled", enabled))

        def setUseLegacyAsymmetricEnergies(self, enabled):
            calls.append(("setUseLegacyAsymmetricEnergies", enabled))

        def setProtocolVersion(self, protocol):
            calls.append(("setProtocolVersion", protocol))

        def setLabileMolSiteParameters(self, parameters):
            calls.append(("setLabileMolSiteParameters", parameters))

        def setLabileMolReferencePositions(self, positions):
            calls.append(("setLabileMolReferencePositions", positions))

        def addLabileMol(self, atoms, reference_atom):
            calls.append(("addLabileMol", atoms, reference_atom))

    class FakeLabileMolSiteParameters:

        def __init__(self, charge, sigma, epsilon):
            self.values = (charge, sigma, epsilon)

    relocation_module = ModuleType("RelocationMCPlugin")
    relocation_module.RelocationMC = FakeRelocationMC
    relocation_module.LabileMolSiteParameters = FakeLabileMolSiteParameters
    monkeypatch.setitem(sys.modules, "RelocationMCPlugin", relocation_module)

    class FakePerturbForce:

        def getParticleParameters(self, _atom):
            return 0.0, 0.3, 0.1

    perturb_force = FakePerturbForce()

    class FakePerturbNonbondedForce:

        @staticmethod
        def isinstance(force):
            return force is perturb_force

        @staticmethod
        def cast(force):
            return force

    perturb_module = ModuleType("PerturbNonbondedPlugin")
    perturb_module.PerturbNonbondedForce = FakePerturbNonbondedForce
    monkeypatch.setitem(sys.modules, "PerturbNonbondedPlugin", perturb_module)

    class FakeQuantity:

        def __init__(self, value):
            self.value = value

        def value_in_unit(self, _unit):
            return self.value

    class FakeSystem:

        def __init__(self):
            self.added_forces = []

        def getForces(self):
            return [perturb_force]

        def addForce(self, force):
            self.added_forces.append(force)

    gk = SimpleNamespace(
        relocation_mc=SimpleNamespace(
            enable=True,
            exchange_frequency=123,
            num_candidate_sites=8192,
            sphere_radius_nm=0.5,
            temperature_K=None,
            random_seed=456,
            orientations_enabled=False,
            use_cavity_bias=True,
            exclude_alchemical_water=True,
            verbose_log=None,
        ),
        ab=SimpleNamespace(ligatoms=[0]),
        filename=SimpleNamespace(getListOfOHH=lambda: [[1, 2, 3]]),
        integrator=SimpleNamespace(targetT_K=300.0),
    )
    positions = FakeQuantity([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.1, 0.1],
        [0.1, 0.2, 0.1],
    ])
    box = FakeQuantity([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    monkeypatch.setattr(omm_system, "get_initial_pos_pbc", lambda _gk: (positions, box))
    monkeypatch.setattr(omm_system, "_get_relocationmc_ligand_heavy_atoms", lambda _system, _atoms: [0])
    monkeypatch.setattr(
        omm_system,
        "_get_relocationmc_alchemical_water_atoms",
        lambda _force, _ligand_atoms, _waters: set(),
    )

    system = FakeSystem()
    omm_system._add_relocation_mc_force(system, gk)

    assert calls[0] == ("construct", 300.0, 0.5, 123, 8192)
    protocol_calls = [
        call for call in calls
        if call[0] in {
            "setUseCavityBias",
            "setOrientationsEnabled",
            "setUseLegacyAsymmetricEnergies",
            "setProtocolVersion",
        }
    ]
    assert protocol_calls == [
        ("setUseCavityBias", False),
        ("setOrientationsEnabled", True),
        ("setUseLegacyAsymmetricEnergies", False),
        ("setProtocolVersion", FakeRelocationMC.MTM),
    ]
    assert len(system.added_forces) == 1
    assert isinstance(system.added_forces[0], FakeRelocationMC)
