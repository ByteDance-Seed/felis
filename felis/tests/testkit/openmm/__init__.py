# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import importlib.metadata

__version__ = importlib.metadata.version("openmm")

from .force import (CMMotionRemover, CustomBondForce, CustomCompoundBondForce, CustomExternalForce,
                    CustomNonbondedForce, CustomTorsionForce, Force, MonteCarloBarostat, NonbondedForce, PeriodicTorsionForce)
from .integrator import BrownianIntegrator, CustomIntegrator, Integrator, LangevinMiddleIntegrator


class Context:

    def __init__(self):
        self._step_count = 0
        self._parameters = {}

    def getState(self, **kwargs):
        return State()

    def setPositions(self, positions):
        return

    def setPeriodicBoxVectors(self, a, b, c):
        return

    def setVelocitiesToTemperature(self, *args):
        return None

    def setParameter(self, name, value):
        self._parameters[name] = value

    def getParameter(self, name):
        return self._parameters.get(name)

    def getStepCount(self):
        return self._step_count

    def setStepCount(self, step_count):
        self._step_count = int(step_count)

    def createCheckpoint(self):
        return int(self._step_count).to_bytes(8, "little", signed=False)

    def loadCheckpoint(self, data):
        self._step_count = int.from_bytes(bytes(data), "little", signed=False)


class System:

    def __init__(self):
        self._forces = []

    def getForces(self):
        return self._forces

    def addForce(self, force):
        self._forces.append(force)


class Platform:

    @classmethod
    def getPlatformByName(cls, name):
        return cls()


class Vec3(tuple):

    def __new__(cls, x, y, z):
        return super().__new__(cls, (x, y, z))


class State:

    def getPositions(self):
        return None

    def getPeriodicBoxVectors(self):
        return (None, None, None)


class MinimizationReporter:

    def __init__(self):
        return None

    def report(self, iteration, x, grad, args):
        raise NotImplementedError
