# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

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
    GKOpenMM: OpenMM configuration.
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
        hmr: Hydrogen mass repartitioning target hydrogen mass in amu. Default is None.
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
    hmr: float = None

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
    lambdaA_charge: float = 1.0 # Equilivalent to elam for ligand charge interpolation in ABFE.
    lambdaB_charge: float = 0.0 # Used for alchemical water charge interpolation in ABFE.
    lambdaC_charge: float = 0.0 # Used for RBFE charge interpolation

    lambdaA_vdw: float = 1.0 # Equilivalent to vlam for ligand vdw interpolation in ABFE.
    lambdaB_vdw: float = 0.0 #
    
    torsionLambda: float = 1.0 # Lambda for ligand torsion interpolation in ABFE.
    reslam: float = 0.0

    lam_list: list = None
    ilam: int = 0

    lambda_schedules: dict = field(
        default_factory=lambda: {
            "lambdaA_charge": [],
            "lambdaB_charge": [],
            "lambdaC_charge": [],
            "lambdaA_vdw": [],
            "lambdaB_vdw": [],
            "torsionLambda": [],
            "reslam": [],
            "rest2_scale": [],
        }
    )

    def __post_init__(self):
        if self.lam_list is not None:
            if self.lam_list and len(self.lam_list[0]) == 7:
                # Build the full REST2 lambda schedule across all windows so
                # that downstream code (e.g. HREMD replica setup) can index
                # any replica_id, not just the current ilam.
                for entry in self.lam_list:
                    if len(entry) != 7:
                        raise ValueError(
                            f"lam_list entry length mismatch: expect 7, got {len(entry)}: {entry}"
                        )
                    (
                        lambdaA_charge,
                        lambdaB_charge,
                        lambdaC_charge,
                        lambdaA_vdw,
                        lambdaB_vdw,
                        torsionLambda,
                        reslam,
                    ) = entry
                    self.lambda_schedules["lambdaA_charge"].append(lambdaA_charge)
                    self.lambda_schedules["lambdaB_charge"].append(lambdaB_charge)
                    self.lambda_schedules["lambdaC_charge"].append(lambdaC_charge)
                    self.lambda_schedules["lambdaA_vdw"].append(lambdaA_vdw)
                    self.lambda_schedules["lambdaB_vdw"].append(lambdaB_vdw)
                    self.lambda_schedules["torsionLambda"].append(torsionLambda)
                    self.lambda_schedules["reslam"].append(reslam)
                    self.lambda_schedules["rest2_scale"].append(1.0)
                # Set the current window fields based on ilam.
                (
                    self.lambdaA_charge,
                    self.lambdaB_charge,
                    self.lambdaC_charge,
                    self.lambdaA_vdw,
                    self.lambdaB_vdw,
                    self.torsionLambda,
                    self.reslam,
                ) = self.lam_list[self.ilam]
            else:
                entry = self.lam_list[self.ilam]
                if len(entry) == 2:
                    self.vlam, self.elam = entry
                elif len(entry) == 3:
                    self.vlam, self.elam, self.reslam = entry


@dataclass
class GKOpenMM:
    """OpenMM configuration.

    Attributes:
        platform: Computation platform ("CUDA", "CPU", or "Reference"). Default is "CUDA".
        precision: Numerical precision for CUDA platform ("mixed" or "double"). Default is "mixed".
        params_ecosystem: Parameter file format (currently only "gromacs"). Default is "gromacs".
        checkpoint_interval: Steps between checkpoint writes. Default is 50.
        resume: Resume HREMD from saved checkpoints when present. Default is True.
    """

    platform: str = "CUDA"
    precision: str = "mixed"
    params_ecosystem: str = "gromacs"
    checkpoint_interval: int = 125000
    exchange_interval: int = 250
    tmp_directory: str = "/tmp"
    resume: bool = True

    def __post_init__(self):
        assert self.platform in ("CUDA", "CPU", "Reference")
        if self.platform == "CUDA":
            assert self.precision in ("mixed", "double")
        assert self.params_ecosystem in ("gromacs",)


@dataclass
class GKRelocationMC:
    """Water enhanced sampling (RelocationMC) configuration.

    RelocationMC is a force-free OpenMM Monte Carlo protocol that relocates
    complete rigid water molecules between a ligand-centred spherical region and
    bulk solvent. It only changes coordinates; particle count and force-field
    parameters are unchanged. In FELIS it is supported on the protein leg
    (sysB) together with the PerturbNonbondedForce.

    Attributes:
        enable: Master switch. Default is False (off).
        exchange_frequency: Number of MD integration steps between automatic
            relocation attempts. Default is 1000.
        num_candidate_sites: MTM proposal-pool size K passed directly to the
            plugin as numCandidateSites. The default is K=16384; it is not
            multiplied by the legacy 60-orientation expansion.
        sphere_radius_offset_nm: Additive offset (nm) added to the ligand
            heavy-atom radius when deriving the inner sphere radius.
            Default is 0.3 nm.
        sphere_radius_scale: Multiplier applied to the maximum pairwise
            distance among ligand heavy atoms before adding the offset.
            Default is 0.5 (radius based on the ligand heavy-atom "diameter").
        sphere_radius_nm: Optional explicit sphere radius in nm. When set
            (not None), it overrides the derived radius. Default is None.
        temperature_K: RelocationMC Metropolis temperature in kelvin. When
            None, the integrator target temperature is used. Default is None.
        random_seed: Random seed for the Monte Carlo sampler. Default is 20260829.
        orientations_enabled: Legacy compatibility field. MTM always enables
            orientations, regardless of this value.
        use_cavity_bias: Legacy compatibility field. MTM always disables the
            cavity/proposal volume bias, regardless of this value.
        exclude_alchemical_water: When the ligand is charged, exclude the
            alchemical (charge-compensating) waters from the labile set.
            Default is True.
    """

    enable: bool = False
    exchange_frequency: int = 1000
    num_candidate_sites: int = 16384
    sphere_radius_offset_nm: float = 0.3
    sphere_radius_scale: float = 0.5
    sphere_radius_nm: float = None
    temperature_K: float = None
    random_seed: int = 20260829
    orientations_enabled: bool = True
    use_cavity_bias: bool = True
    exclude_alchemical_water: bool = True
    verbose_log: str = None

    def __post_init__(self):
        if not self.enable:
            return
        assert self.exchange_frequency > 0, "relocation_mc.exchange_frequency must be positive"
        assert self.num_candidate_sites > 0, "relocation_mc.num_candidate_sites must be positive"
        assert self.sphere_radius_offset_nm >= 0.0, "relocation_mc.sphere_radius_offset_nm must be >= 0"
        assert self.sphere_radius_scale > 0.0, "relocation_mc.sphere_radius_scale must be > 0"
        if self.sphere_radius_nm is not None:
            assert self.sphere_radius_nm > 0.0, "relocation_mc.sphere_radius_nm must be positive"


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
        relocation_mc: RelocationMC water enhanced sampling settings (GKRelocationMC).
    """

    dir: GKDir = field(default_factory=lambda: GKDir())
    filename: GKFilename = field(default_factory=lambda: GKFilename())
    integrator: GKIntegrator = field(default_factory=lambda: GKIntegrator())
    posres: GKPosres = field(default_factory=lambda: GKPosres())
    boresch: GKBoresch = field(default_factory=lambda: GKBoresch())
    ab: GKAB = field(default_factory=lambda: GKAB())
    openmm: GKOpenMM = field(default_factory=lambda: GKOpenMM())
    relocation_mc: GKRelocationMC = field(default_factory=lambda: GKRelocationMC())

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
