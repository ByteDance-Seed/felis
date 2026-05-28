# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

__version__ = importlib.metadata.version("openmm")

from .force import (CMMotionRemover, CustomBondForce, CustomCompoundBondForce, CustomExternalForce,
                    CustomNonbondedForce, Force, MonteCarloBarostat, NonbondedForce, PeriodicTorsionForce)
from .integrator import BrownianIntegrator, CustomIntegrator, Integrator, LangevinMiddleIntegrator


class Context:

    def getState(self, **kwargs):
        return State()

    def setPositions(self, positions):
        return

    def setPeriodicBoxVectors(self, a, b, c):
        return

    def setVelocitiesToTemperature(self, *args):
        return None


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
