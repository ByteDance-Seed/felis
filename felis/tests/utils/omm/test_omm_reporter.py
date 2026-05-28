# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import numpy as np
import openmm as omm
import openmm.app as app
import openmm.unit as ou

from felis.utils.omm.omm_reporter import SimpleBinaryReporter


def test_simple_binary_reporter():
    pdb_path = os.path.join(os.path.dirname(__file__), '..', '..', 'testdata', '5dfr.pdb')

    pdb = app.PDBFile(pdb_path)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    integrator = omm.LangevinMiddleIntegrator(300 * ou.kelvin, 1 / ou.picosecond, 0.001 * ou.picoseconds)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, 'output.bin')
        report_interval = 1
        reporter = SimpleBinaryReporter(out_file, reportInterval=report_interval, enforcePeriodicBox=False)
        simulation.reporters.append(reporter)

        n_steps = 3
        simulation.step(n_steps)

        # Force flush and close
        del reporter

        assert os.path.exists(out_file)

        # Parse binary output
        n_atoms = 2489
        known_dtype = np.float32
        known_shape = (n_atoms, 3)

        all_data_1d = np.fromfile(out_file, dtype=known_dtype)
        all_frames_stacked = all_data_1d.reshape(-1, known_shape[0], known_shape[1])

        expected_frames = n_steps // report_interval
        assert all_frames_stacked.shape[0] == expected_frames
        assert all_frames_stacked.shape[1] == n_atoms
        assert all_frames_stacked.shape[2] == 3
