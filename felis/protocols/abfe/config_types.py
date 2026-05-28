# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from dataclasses import fields
from enum import Enum
import importlib
from pathlib import Path
import tempfile

import numpy as np

from felis.configs import GlobalKeys
from felis.configs import load_config
from felis.configs import MinimizeRelaxOption


def _int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value


class ABStage(Enum):
    makebox = "makebox"

    boresch_em = "boresch_em"
    boresch_npt = "boresch_npt"
    boresch_post_process = "boresch_post_process"

    sysA_em = "sysA_em"
    sysA = "sysA"  # ligand in solvent e: 1->0; v: 1->0
    sysB_em = "sysB_em"
    sysB = "sysB"  # ligand in protein v: 0->1; e: 0->1; res: 1->0

    mbar = "mbar"
    summarize = "summarize"


_ABStageExecutionRegistry = {}

_ABStageExecutionModuleByStage = {
    ABStage.makebox: "felis.protocols.abfe.stage_makebox",
    ABStage.boresch_em: "felis.protocols.abfe.stage_boresch",
    ABStage.boresch_npt: "felis.protocols.abfe.stage_boresch",
    ABStage.boresch_post_process: "felis.protocols.abfe.stage_boresch",
    ABStage.sysA_em: "felis.protocols.abfe.stage_sysA",
    ABStage.sysA: "felis.protocols.abfe.stage_sysA",
    ABStage.sysB_em: "felis.protocols.abfe.stage_sysB",
    ABStage.sysB: "felis.protocols.abfe.stage_sysB",
    ABStage.mbar: "felis.protocols.abfe.stage_post_analysis",
    ABStage.summarize: "felis.protocols.abfe.stage_post_analysis",
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
    elamrecipe: str = "e29"
    vlamrecipe: str = "v45"
    reslamrecipe: str = "r02"
    supplementary_elec_lambda_list: list[float] = None  # elamrecipe = esuppl
    supplementary_vdw_lambda_list: list[float] = None  # vlamrecipe = vsuppl
    supplementary_restraint_lambda_list: list[float] = None  # reslamrecipe = rsuppl
    pro_ionic_strength: float = 0.0
    eq_pose: int = 0  # not used; keep for backward compatibility
    md_checkpoint_interval: int = None
    md_sol_nsnapshots: int = None
    md_pro_nsnapshots: int = None
    md_nsnapshots: int = 2000
    md_eq_nsnapshots: int = 600
    md_sol_em_version: int = MinimizeRelaxOption.heating.value
    md_pro_em_version: int = MinimizeRelaxOption.fire2.value
    k_r_a_dih: list[float] = None
    env_list: list[str] = None

    # e.g., "HDF5_USE_FILE_LOCKING": "FALSE"
    # https://github.com/choderalab/yank/issues/1256

    def update_by_dict(self, d: dict) -> None:
        for f in fields(ABFEInputConfig):
            if f.name in d:
                if d[f.name] is not None:
                    setattr(self, f.name, d[f.name])

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
        suppl_names = [
            "supplementary_elec_lambda_list",
            "supplementary_vdw_lambda_list",
            "supplementary_restraint_lambda_list",
        ]
        for name in suppl_names:
            l = getattr(self, name) or []
            l = sorted(l)
            if l and not (np.isclose(l[0], 0.0, atol=1.e-10) and np.isclose(l[-1], 1.0, atol=1.e-10)):
                raise ValueError(f"{name} should start with 0.0 and end with 1.0")
            if l:
                l[0] = 0.0
                l[-1] = 1.0
            setattr(self, name, l)
        self.md_checkpoint_interval = self.md_checkpoint_interval or GlobalKeys().openmm.checkpoint_interval
        self.md_sol_nsnapshots = self.md_sol_nsnapshots or self.md_nsnapshots
        self.md_pro_nsnapshots = self.md_pro_nsnapshots or self.md_nsnapshots
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
            h = f.name
            typ = f.type
            if f.name in ("md_pro_em_version", "md_sol_em_version"):
                typ = _int_or_str
            yield (f.name, typ, f.default, h)
