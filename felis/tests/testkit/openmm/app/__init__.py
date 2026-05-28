# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

from .. import Context, Integrator, System
from .topology import Topology


class Simulation:

    def __init__(self, topology, system, integrator, platform=None, platformProperties=None, state=None):
        self.reporters = []
        self.context = Context()
        self.integrator = integrator
        self.topology = topology
        self.currentStep = 0

    def step(self, nsteps: int):
        self.currentStep += int(nsteps)
        return None

    def minimizeEnergy(self, **kwargs):
        reporter = kwargs.get("reporter", None)
        max_iterations = 3

        if reporter is None:
            return None

        args = {
            "system energy": 0.0,
            "restraint energy": 0.0,
            "restraint strength": 0.0,
            "max constraint error": 0.0,
        }

        iteration = 0
        while iteration < max_iterations:
            stop = reporter.report(iteration, None, None, args)
            if stop:
                break
            iteration += 1

        return None

    def loadCheckpoint(self, fh):
        return None


class CheckpointReporter:

    def __init__(self, file_handle, **_kwargs):
        try:
            file_handle.close()
        except Exception:
            pass


class StateDataReporter:

    def __init__(self, file_handle, **_kwargs):
        try:
            file_handle.close()
        except Exception:
            pass


class DCDReporter:

    def __init__(self, file: str, **_kwargs):
        p = Path(file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


class DCDFile:

    def __init__(self, file_handle, _topology=None, _dt=None):
        self._fh = file_handle

    def writeModel(self, **_kwargs):
        try:
            self._fh.write(b"DCD\n")
        except TypeError:
            self._fh.write("DCD\n")


class PDBFile:

    def __init__(self, file: str, **kwargs):
        self.topology = Topology()
        self.positions = []

    @staticmethod
    def writeFile(topology, positions, file=sys.stdout, keepIds=False, **kwargs):
        return


class GromacsGroFile:

    def __init__(self, file: str, **kwargs):
        self.positions = []

    def getPeriodicBoxVectors(self):
        return [[], [], []]


class GromacsTopFile:

    def __init__(self, file, **kwargs):
        self.topology = Topology()

    def createSystem(self, **kwargs):
        return System()
