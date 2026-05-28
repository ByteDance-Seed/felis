# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np


class MultiStateReporter:

    def __init__(self, storage, **kwargs):
        self.storage = Path(storage)
        self.checkpoint_storage = self.storage.with_name(f"{self.storage.stem}_checkpoint.nc")

    def _touch(self):
        self.storage.parent.mkdir(parents=True, exist_ok=True)
        self.storage.touch()
        self.checkpoint_storage.touch()


class ReplicaExchangeSampler:

    def __init__(self, **kwargs):
        self._storage = None

    def create(self, thermodynamic_states, **kwargs):
        self._storage = kwargs["storage"]

    def run(self):
        self._storage._touch()

    @classmethod
    def from_storage(cls, storage):
        obj = cls()
        obj._storage = MultiStateReporter(storage)
        return obj


class MultiStateSamplerAnalyzer:
    pass


class ReplicaExchangeAnalyzer:

    def __init__(self, *args, **kwargs):
        import felis.tests.testkit.openmm.unit as unit
        self.kT = 2478.957029602388 * unit.kilojoule_per_mole
        self.n_states = 4
        self.n_equilibration_iterations = 1

        self._dummy_n_iterations = 601

    def get_free_energy(self):
        fe = np.identity(self.n_states)
        std_fe = np.zeros_like(fe)
        return fe, std_fe

    def generate_mixing_statistics(self):
        matrix = np.identity(self.n_states)
        eigenval = np.ones(self.n_states)
        ineff = np.ones(self.n_states)
        return matrix, eigenval, ineff

    def read_energies(self):
        ematx = np.zeros((self.n_states, self.n_states, self._dummy_n_iterations))
        _umatx = None
        _nbs = None
        ridx = np.array([[idx for idx in range(self.n_states)] for _ in range(self._dummy_n_iterations)]).transpose()
        return ematx, _umatx, _nbs, ridx
