# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import numpy as np
import openmm as omm
import openmm.app as app

logger = logging.getLogger(__name__)


class SimpleBinaryReporter:

    def __init__(self, file, reportInterval, enforcePeriodicBox: bool = False):

        self._reportInterval = reportInterval
        self._enforcePeriodicBox = enforcePeriodicBox

        if os.path.exists(file):
            logger.warning(f"File {file} already exists. Overwriting.")
        self._out = open(file, 'wb')

    def describeNextReport(self, simulation: app.Simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return {'steps': steps, 'periodic': self._enforcePeriodicBox, 'include': ['positions']}

    def report(self, simulation: app.Simulation, state: omm.State):
        positions = state.getPositions(asNumpy=True)  # default openmm unit
        positions = np.asarray(positions, dtype=np.float32)  # explicit dtype

        # positions have a fixed shape (natoms, 3)
        positions.tofile(self._out)

        if hasattr(self._out, 'flush') and callable(self._out.flush):
            self._out.flush()

    def __del__(self):
        self._out.close()


class MinimizerReporter(omm.MinimizationReporter):

    def __init__(self, logger_object=None):
        super().__init__()
        self.logger = logger_object

    def report(self, iteration, _x, _grad, args):
        system_energy = args["system energy"]
        restraint_energy = args["restraint energy"]
        restraint_strength = args["restraint strength"]
        max_constraint_error = args["max constraint error"]
        l1 = " iteration   system_energy   restraint_energy   restraint_strength   max_constraint_error"
        l2 = f"{iteration:10d}{system_energy:16.4f}{restraint_energy:16.4f}{restraint_strength:12.2e}{max_constraint_error:10.6f}"

        if iteration == 0:
            if self.logger:
                self.logger.info(l1)
            else:
                print(l1)

        if self.logger:
            self.logger.info(l2)
        else:
            print(l2)
        return False
