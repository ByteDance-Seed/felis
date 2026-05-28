# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Global configuration keys for FELIS workflows.

This module defines configuration dataclasses used throughout the codebase
for managing simulation parameters, file paths, and runtime settings.

Dataclasses:
    GKDir: Directory path configuration.
    GKFilename: Filename pattern configuration.
    GKIntegrator: Molecular dynamics integrator settings.
    GKPosres: Position restraint configuration.
    GKBoresch: Boresch restraint configuration.
    GKAB: Absolute free energy settings.
    GKOpenMM: OpenMM and openmmtools configuration.
    GlobalKeys: Aggregates all configuration sections.
"""

from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import is_dataclass
from pathlib import Path
import random
from typing import Any, Type

from ._config_tools import load_config
from .global_keys_option_enums import IntegratorNameOption
from .global_keys_option_enums import MinimizeRelaxOption


def _resolve_path(path) -> str:
    """Resolve a filesystem path to an absolute path string.

    Unlike :meth:`pathlib.Path.resolve`, this helper uses ``strict=False`` so it
    can resolve paths that do not exist yet (common for output paths).
    """
    return str(Path(path).resolve(strict=False))


@dataclass
class GKDir:
    """Directory path configuration.

    Attributes:
        outbase: Base output directory. Defaults to current directory.
        trj: Trajectory output directory. Defaults to outbase if not set.
    """

    outbase: str = None
    trj: str = None

    def __post_init__(self):
        if self.outbase is not None:
            self.outbase = _resolve_path(self.outbase)
        else:
            self.outbase = _resolve_path(".")

        if self.trj is None:
            self.trj = self.outbase
        elif Path(self.trj).is_absolute():
            pass
        else:
            self.trj = Path(self.outbase) / Path(self.trj)
            self.trj = str(self.trj)


@dataclass
class GKFilename:
    """Filename pattern configuration.

    Attributes:
        stem: Base filename stem for output files.
        sys: System topology filename.
        crd: Coordinate file name.
        monomer: Monomer structure filename.
        atom_ids: JSON file with atom IDs.
    """

    stem: str = None
    sys: str = None
    crd: str = None
    monomer: str = None
    atom_ids: str = None

    def getListOfOHH(self):
        if self.atom_ids and Path(self.atom_ids).is_file():
            with open(self.atom_ids) as f:
                d = load_config(f)
                if "others" in d.keys():
                    if "SOL" in d["others"].keys():
                        return d["others"]["SOL"]
        return []


@dataclass
class GKIntegrator:
    """Molecular dynamics integrator settings.

    Attributes:
        name: Integrator class name. Default is "LangevinMiddleIntegrator".
        dt_ps: Time step in picoseconds. Default is 0.002 (2 fs).
        constraint_tol: Constraint tolerance. Default is 1e-8.
        friction_1_ps: Friction coefficient in 1/ps. Default is 0.5.
        nstep_per_snapshot: Steps between trajectory snapshots. Default is 500.
        nsnapshots: Number of snapshots to collect. Default is 1.
        targetT_K: Target temperature in Kelvin. Default is 298.15.
        targetP_bar: Target pressure in bar. Default is 1.01325.
        minimize: Energy minimization protocol. Default is 0. See MinimizeRelaxOption.
        npt: NPT ensemble flag (0=NVT, 1=NPT). Default is 0.
        npt_mc_freq: Monte Carlo barostat frequency. Default is 25.
        randomseed: Random seed for reproducibility. Default is randomly generated.
    """

    name: str = "LangevinMiddleIntegrator"
    dt_ps: float = 0.002
    constraint_tol: float = 1.e-8
    friction_1_ps: float = 0.5
    nstep_per_snapshot: int = 500
    nsnapshots: int = 1
    targetT_K: float = 298.15
    targetP_bar: float = 1.01325
    minimize: int = 0
    npt: int = 0
    npt_mc_freq: int = 25
    randomseed: int = field(default_factory=lambda: random.randint(0, 2147483647))

    def __post_init__(self):
        self.name = IntegratorNameOption(self.name).value
        self.minimize = MinimizeRelaxOption(self.minimize).value
        assert self.npt in (0, 1)


@dataclass
class GKPosres:
    """Position restraints configuration.

    Attributes:
        atoms: List of atom indices to restrain. None to disable restraints.
        k_kcal: Restraint force constant in kcal/mol/A^2. Default is 25.0.
        tol_angstrom: Restraint tolerance in Angstroms. Default is 0.5.
    """

    atoms: list = None  # set None to disable
    k_kcal: float = 25.
    tol_angstrom: float = 0.5


@dataclass
class GKBoresch:
    """Boresch restraint configuration for binding free energy calculations.

    Boresch restraints are used to restrain the ligand relative to the protein
    during alchemical calculations.

    Attributes:
        ligatoms: List of 3 ligand atom indices for restraint definition.
                  Set None to disable Boresch restraints.
        proatoms: List of 3 protein atom indices for restraint definition.
        r_theta_phi: List of equilibrium values for distance (P1L1), theta (P2P1L1), phi (P3P2P1L1).
        alpha_beta_gamma: List of equilibrium values P1L1L2, P2P1L1L2, and P1L1L2L3.
        k_r_a_dih_kcal: List of force constants for restraints.
    """

    ligatoms: list = None  # set None to disable
    proatoms: list = None
    r_theta_phi: list = None
    alpha_beta_gamma: list = None
    k_r_a_dih_kcal: list = None

    def __post_init__(self):
        if self.ligatoms:
            assert len(self.ligatoms) == len(self.proatoms) == 3
        if self.proatoms:
            assert len(self.ligatoms) == len(self.proatoms) == 3


@dataclass
class GKAB:
    """Absolute free energy calculation settings.

    Controls lambda values for van der Waals, electrostatic, and restraint
    interactions during alchemical transformations.

    Attributes:
        ligatoms: List of ligand atom indices. Set None to disable alchemical binding.
        vlam: van der Waals lambda value. Default is 1.0.
        elam: Electrostatic lambda value. Default is 1.0.
        reslam: Restraint lambda value. Default is 0.0.
        lam_list: List of lambda value tuples for multi-step transformations.
        ilam: Index of current lambda in lam_list. Default is 0.
    """

    ligatoms: list = None  # set None to disable
    vlam: float = 1.0
    elam: float = 1.0
    reslam: float = 0.0

    lam_list: list = None
    ilam: int = 0

    def __post_init__(self):
        if self.lam_list is not None:
            entry = self.lam_list[self.ilam]
            if len(entry) == 2:
                self.vlam, self.elam = entry
            elif len(entry) == 3:
                self.vlam, self.elam, self.reslam = entry


@dataclass
class GKOpenMM:
    """OpenMM and openmmtools configuration.

    Attributes:
        platform: Computation platform ("CUDA", "CPU", or "Reference"). Default is "CUDA".
        precision: Numerical precision for CUDA platform ("mixed" or "double"). Default is "mixed".
        params_ecosystem: Parameter file format (currently only "gromacs"). Default is "gromacs".
        checkpoint_interval: Steps between checkpoint writes. Default is 50.
    """

    platform: str = "CUDA"
    precision: str = "mixed"
    params_ecosystem: str = "gromacs"
    checkpoint_interval: int = 50

    def __post_init__(self):
        assert self.platform in ("CUDA", "CPU", "Reference")
        if self.platform == "CUDA":
            assert self.precision in ("mixed", "double")
        assert self.params_ecosystem in ("gromacs",)


def _update_by_dict(cls: Type[Any], obj: Any, data: dict) -> None:
    """Update a dataclass object from a dictionary.

    Args:
        cls: The dataclass type.
        obj: The dataclass instance to update.
        data: Dictionary of field names and values to set.
    """
    if not is_dataclass(cls):
        raise TypeError(f"should be called on a dataclass, not {cls}")
    if cls != type(obj):
        raise TypeError(f"object {obj} is not an object of class {cls}")
    # lower_case_field: Original_Case_Field
    valid_fields = {f.name.lower(): f.name for f in fields(cls)}
    _validfields = set(valid_fields.keys())
    for k, v in data.items():
        klow = k.lower()
        assert klow in _validfields, f"{k} is not a valid field name of {cls}"
        kori = valid_fields[klow]
        setattr(obj, kori, v)


@dataclass
class GlobalKeys:
    """Aggregate configuration for simulations.

    This class holds all configuration sub-sections for directory paths,
    filenames, integrator settings, restraints, and OpenMM parameters.
    It provides methods to update configuration from various sources.

    Attributes:
        dir: Directory path configuration (GKDir).
        filename: Filename pattern configuration (GKFilename).
        integrator: MD integrator settings (GKIntegrator).
        posres: Position restraints configuration (GKPosres).
        boresch: Boresch restraints configuration (GKBoresch).
        ab: Alchemical binding settings (GKAB).
        openmm: OpenMM platform configuration (GKOpenMM).
    """

    dir: GKDir = field(default_factory=lambda: GKDir())
    filename: GKFilename = field(default_factory=lambda: GKFilename())
    integrator: GKIntegrator = field(default_factory=lambda: GKIntegrator())
    posres: GKPosres = field(default_factory=lambda: GKPosres())
    boresch: GKBoresch = field(default_factory=lambda: GKBoresch())
    ab: GKAB = field(default_factory=lambda: GKAB())
    openmm: GKOpenMM = field(default_factory=lambda: GKOpenMM())

    def __str__(self) -> str:
        s = ""
        for f in fields(GlobalKeys):
            s = s + "\n" + str(getattr(self, f.name))
        return s

    def check(self):
        """Validate all configuration sections by calling their __post_init__ methods."""
        for f in fields(GlobalKeys):
            obj = getattr(self, f.name)
            if hasattr(obj, "__post_init__"):
                obj.__post_init__()

    def _update_by_nested_dict(self, nested_data: dict) -> None:
        """Update configuration from a nested dictionary.

        Args:
            nested_data: Dictionary with section names as keys and
                         section-specific parameter dictionaries as values.
                         Section and parameter names are case-insensitive.
        """
        # lower_case_field: Original_Case_Field
        valid_fields = {f.name.lower(): f.name for f in fields(GlobalKeys)}
        _validfields = set(valid_fields.keys())
        valid_names_types = {f.name: f.type for f in fields(GlobalKeys)}
        for field_name, field_data in nested_data.items():
            fnlow = field_name.lower()
            assert fnlow in _validfields, f"{field_name} is not a valid field name of GlobalKeys"
            fnori = valid_fields[fnlow]
            _update_by_dict(valid_names_types[fnori], getattr(self, fnori), field_data)

    def update_by_cfg(self, cfg: str) -> None:
        """Update configuration from a YAML or JSON file.

        Args:
            cfg: Path to configuration file (YAML or JSON format).
        """
        with open(cfg) as f:
            self._update_by_nested_dict(load_config(f))

    def update_by_tkv(self, tkv: str) -> None:
        """Update a single parameter using type:key:value format.

        Args:
            tkv: String of form "type:key1.key2:value" or "type:key1.key2:"
                 for None value. Type can be "i" (int), "f" (float), or "s" (string).

        Example:
            >>> gk.update_by_tkv("f:integrator.targetT_K:300.0")
            >>> gk.update_by_tkv("s:openmm.platform:CPU")
        """
        vs = tkv.split(":")
        t, k, v = None, None, None
        if len(vs) == 3:
            t, k, v = vs
        elif len(vs) == 2:
            t, k = vs
        else:
            raise ValueError(f"invalid tkv: {tkv}")
        if v == "":
            v = None
        assert t in ("i", "f", "s")
        if v is not None:
            if t == "i":
                v = int(v)
            elif t == "f":
                v = float(v)
        ks = k.split(".")
        assert len(ks) == 2
        k1, k2 = ks
        self._update_by_nested_dict({k1: {k2: v}})

    def update_by_comma_sep_tkv(self, comma_sep_tkv: str) -> None:
        """Update multiple parameters from comma-separated tkv strings.

        Args:
            comma_sep_tkv: Comma-separated list of tkv strings. Empty entries are ignored.

        Example:
            >>> gk.update_by_comma_sep_tkv("f:integrator.targetT_K:300.0,s:openmm.platform:CPU")
        """
        extra_tkv = [w0 for w0 in comma_sep_tkv.split(",") if w0.strip() != ""]
        for t in extra_tkv:
            self.update_by_tkv(t)
