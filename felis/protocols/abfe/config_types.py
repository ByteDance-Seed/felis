# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from enum import Enum
import importlib
from pathlib import Path
import tempfile

from felis.configs import GlobalKeys
from felis.configs import load_config
from felis.configs import MinimizeRelaxOption


ABFE_V2_BORESCH_SAMPLING_PROTOCOLS = (
    "hremd_same_temperature",
    "long_npt",
)


def _int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value


class ABStage(Enum):
    makebox = "makebox"
    boresch_em_v2 = "boresch_em_v2"
    boresch_npt_v2 = "boresch_npt_v2"
    boresch_post_process_v2 = "boresch_post_process_v2"
    sysA_em_v2 = "sysA_em_v2"
    sysA_opt_lambdas_v2 = "sysA_opt_lambdas_v2"
    sysA_v2 = "sysA_v2"
    sysB_em_v2 = "sysB_em_v2"
    sysB_opt_lambdas_v2 = "sysB_opt_lambdas_v2"
    sysB_v2 = "sysB_v2"
    post_analysis_v2 = "post_analysis_v2"

_ABStageExecutionRegistry = {}

_ABStageExecutionModuleByStage = {
    ABStage.makebox: "felis.protocols.abfe.stage_makebox",
    ABStage.boresch_em_v2: "felis.protocols.abfe_v2.stage_boresch",
    ABStage.boresch_post_process_v2: "felis.protocols.abfe_v2.stage_boresch",
    ABStage.boresch_npt_v2: "felis.protocols.abfe_v2.stage_boresch",
    ABStage.sysA_em_v2: "felis.protocols.abfe_v2.stage_sysA",
    ABStage.sysA_opt_lambdas_v2: "felis.protocols.abfe_v2.stage_sysA",
    ABStage.sysA_v2: "felis.protocols.abfe_v2.stage_sysA",
    ABStage.sysB_em_v2: "felis.protocols.abfe_v2.stage_sysB",
    ABStage.sysB_opt_lambdas_v2: "felis.protocols.abfe_v2.stage_sysB",
    ABStage.sysB_v2: "felis.protocols.abfe_v2.stage_sysB",
    ABStage.post_analysis_v2: "felis.protocols.abfe_v2.stage_post_analysis",
}


def ABStageExecutionRegister(stage: ABStage):

    def deco(cls):
        if stage in _ABStageExecutionRegistry:
            raise ValueError(f"ABStageExecution stage {stage.value} already registered")
        cls._stage = stage
        _ABStageExecutionRegistry[stage] = cls
        return cls

    return deco


class ABStageExecutionBase:
    _stage: ABStage

    @property
    def stage(self) -> ABStage:
        return self.__class__._stage

    @classmethod
    def create(cls, stage: ABStage, *args, **kwargs):
        if stage not in _ABStageExecutionRegistry:
            # Lazy import: registration happens on module import via decorator.
            module_name = _ABStageExecutionModuleByStage.get(stage)
            if module_name is not None:
                importlib.import_module(module_name)
        if stage not in _ABStageExecutionRegistry:
            raise KeyError(f"ABStageExecution stage {stage.value} not registered")
        impl = _ABStageExecutionRegistry[stage]
        return impl(*args, **kwargs)

    def __init__(self, **kwargs):
        self._planned_stages = kwargs.pop("planned_stages", None)
        self._job_context = kwargs.pop("job_context", None)
        if kwargs:
            raise TypeError(f"unexpected kwargs: {kwargs}")
        super().__init__()

    def exec(self):
        pass


@dataclass
class ABFEInputConfig:
    progro: str = None
    protop: str = None
    cofsdfs: list[str] = None
    cofitps: list[str] = None
    sdffile: str = None
    itpfile: str = None
    outdir: str = None
    tmpdir: str = None
    stages: list[str] = None
    wat_lam_json: str = None
    pro_lam_json: str = None
    pro_ionic_strength: float = 0.0
    md_checkpoint_interval: int = 125000
    md_sol_nsnapshots: int = 10
    md_pro_nsnapshots: int = 10
    md_exchange_interval: int = 250
    md_traj_interval: int = 125000
    omm_tmp_directory: str = "/tmp"
    md_boresch_nsnapshots: int = 300
    md_boresch_traj_interval: int = 2500
    md_integrator_dt_ps: float = 0.004
    md_integrator_hmr: float = 1.5
    md_sol_em_version: int = MinimizeRelaxOption.heating.value
    md_pro_em_version: int = MinimizeRelaxOption.fire2.value
    k_r_a_dih: list[float] = None
    env_list: list[str] = None
    boresch_sampling_protocol: str = "hremd_same_temperature"

    # RelocationMC water enhanced sampling (protein leg / sysB only, requires abfe_version="v2").
    rmc_enable: bool = False
    rmc_exchange_frequency: int = 1000
    rmc_num_candidate_sites: int = field(
        default=16384,
        metadata={
            "help": (
                "MTM proposal-pool size K passed directly as numCandidateSites; "
                "the default is K=16384 (no 60x multiplier)"
            )
        },
    )
    rmc_sphere_radius_offset_nm: float = 0.3
    rmc_sphere_radius_scale: float = 0.5
    rmc_sphere_radius_nm: float = None
    rmc_random_seed: int = 20260829
    # Retained to parse existing configs; fixed MTM setup ignores both fields.
    rmc_orientations_enabled: bool = True
    rmc_use_cavity_bias: bool = True
    rmc_exclude_alchemical_water: bool = True


    # e.g., "HDF5_USE_FILE_LOCKING": "FALSE"
    # https://github.com/choderalab/yank/issues/1256

    def update_by_dict(self, d: dict) -> None:
        for f in fields(ABFEInputConfig):
            if f.name in d:
                if d[f.name] is not None:
                    setattr(self, f.name, d[f.name])

    def hmr_tkv(self) -> str:
        if self.md_integrator_hmr is None:
            return "f:integrator.hmr:"
        return f"f:integrator.hmr:{self.md_integrator_hmr}"

    def dt_tkv(self) -> str:
        return f"f:integrator.dt_ps:{self.md_integrator_dt_ps}"

    def relocation_mc_tkv(self) -> list[str]:
        """Return tkv flags configuring RelocationMC for the protein leg (sysB).

        Returns an empty list when RelocationMC is disabled so the sysA leg and
        default runs are unaffected.
        """
        if not self.rmc_enable:
            return []
        flags = [
            "i:relocation_mc.enable:1",
            f"i:relocation_mc.exchange_frequency:{int(self.rmc_exchange_frequency)}",
            f"i:relocation_mc.num_candidate_sites:{int(self.rmc_num_candidate_sites)}",
            f"f:relocation_mc.sphere_radius_offset_nm:{float(self.rmc_sphere_radius_offset_nm)}",
            f"f:relocation_mc.sphere_radius_scale:{float(self.rmc_sphere_radius_scale)}",
            f"i:relocation_mc.random_seed:{int(self.rmc_random_seed)}",
            f"i:relocation_mc.orientations_enabled:{int(bool(self.rmc_orientations_enabled))}",
            f"i:relocation_mc.use_cavity_bias:{int(bool(self.rmc_use_cavity_bias))}",
            f"i:relocation_mc.exclude_alchemical_water:{int(bool(self.rmc_exclude_alchemical_water))}",
        ]
        if self.rmc_sphere_radius_nm is not None:
            flags.append(f"f:relocation_mc.sphere_radius_nm:{float(self.rmc_sphere_radius_nm)}")
        return flags

    def check(self):

        def _make_abs(x):
            if x is None:
                return None
            return str(Path(x).absolute())

        self.progro = _make_abs(self.progro)
        self.protop = _make_abs(self.protop)
        if self.cofsdfs is None:
            self.cofsdfs = []
        if self.cofitps is None:
            self.cofitps = []
        self.cofsdfs = [_make_abs(x) for x in self.cofsdfs]
        self.cofitps = [_make_abs(x) for x in self.cofitps]
        assert len(self.cofsdfs) == len(self.cofitps)
        self.sdffile = _make_abs(self.sdffile)
        self.itpfile = _make_abs(self.itpfile)
        self.wat_lam_json = _make_abs(self.wat_lam_json)
        self.pro_lam_json = _make_abs(self.pro_lam_json)
        self.outdir = _make_abs(self.outdir)
        if self.outdir is None:
            user_home_dir = str(Path.home())
            self.outdir = tempfile.mkdtemp(dir=user_home_dir)
        else:
            Path(self.outdir).mkdir(parents=True, exist_ok=True)
        # if tmpdir is not set, tmpdir is set to outdir; otherwise, use a tmp path
        if self.tmpdir is None:
            self.tmpdir = self.outdir
        else:
            user_home_dir = str(Path.home())
            self.tmpdir = tempfile.mkdtemp(dir=user_home_dir)
        if self.md_checkpoint_interval is None:
            self.md_checkpoint_interval = 125000
        if self.md_exchange_interval <= 0:
            raise ValueError("md_exchange_interval must be positive")
        if self.md_traj_interval % self.md_exchange_interval != 0:
            raise ValueError("md_traj_interval must be a multiple of md_exchange_interval")
        if self.md_checkpoint_interval % self.md_exchange_interval != 0:
            raise ValueError("md_checkpoint_interval must be a multiple of md_exchange_interval")
        if self.md_checkpoint_interval < self.md_traj_interval:
            raise ValueError("md_checkpoint_interval must be >= md_traj_interval")
        if self.boresch_sampling_protocol not in ABFE_V2_BORESCH_SAMPLING_PROTOCOLS:
            raise ValueError(
                "boresch_sampling_protocol must be one of "
                f"{ABFE_V2_BORESCH_SAMPLING_PROTOCOLS}; got {self.boresch_sampling_protocol}"
            )
        if self.md_boresch_traj_interval <= 0:
            raise ValueError("md_boresch_traj_interval must be positive")
        if self.boresch_sampling_protocol == "hremd_same_temperature":
            if self.md_boresch_traj_interval % self.md_exchange_interval != 0:
                raise ValueError(
                    "md_boresch_traj_interval must be a multiple of md_exchange_interval"
                )
            if self.md_checkpoint_interval < self.md_boresch_traj_interval:
                raise ValueError(
                    "md_checkpoint_interval must be >= md_boresch_traj_interval"
                )
        self.md_sol_em_version = MinimizeRelaxOption(self.md_sol_em_version).value
        self.md_pro_em_version = MinimizeRelaxOption(self.md_pro_em_version).value
        if self.k_r_a_dih is None:
            self.k_r_a_dih = [2., 80., 80.]
        if self.env_list is None:
            self.env_list = []
        for envstr in self.env_list:
            assert ":" in envstr, f"envstr {envstr} should be in the format of 'env:value'"

    @classmethod
    def from_file(cls, path: str):
        if path is None:
            return cls()
        else:
            with open(path) as f:
                d = load_config(f)
            known_fields = {f.name for f in fields(cls)}
            for k in d.keys():
                if k not in known_fields:
                    raise ValueError(f"Unknown key {k} in config file {path}")
            d = {k: v for k, v in d.items() if k in known_fields}
            return cls(**d)

    @classmethod
    def get_list_of_field_type_default_help_tuples(cls) -> list:
        for f in fields(cls):
            h = f.metadata.get("help", f.name)
            typ = f.type
            if f.name in ("md_pro_em_version", "md_sol_em_version"):
                typ = _int_or_str
            yield (f.name, typ, f.default, h)
