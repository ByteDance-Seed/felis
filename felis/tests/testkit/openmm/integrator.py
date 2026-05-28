# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

class Integrator:

    def __init__(self):
        return None

    def setConstraintTolerance(self, tol):
        return None


class LangevinMiddleIntegrator(Integrator):

    def __init__(self, *args):
        super().__init__()
        return

    def setRandomNumberSeed(self, seed):
        return

    def setTemperature(self, temp):
        return


class BrownianIntegrator(Integrator):

    def __init__(self, *args):
        super().__init__()
        return


class CustomIntegrator(Integrator):

    def __init__(self, *args):
        super().__init__()
        return

    def addGlobalVariable(self, name, initialValue):
        return

    def addComputeSum(self, variable, expression):
        return

    def addComputeGlobal(self, variable, expression):
        return

    def addComputePerDof(self, variable, expression):
        return

    def beginIfBlock(self, condition):
        return

    def endBlock(self):
        return

    def addConstrainVelocities(self):
        return

    def addConstrainPositions(self):
        return
