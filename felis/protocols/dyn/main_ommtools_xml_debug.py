# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import traceback
from collections import defaultdict
from pathlib import Path

import MDAnalysis
import numpy as np
from openmm import Integrator, Platform, State, System, XmlSerializer
from openmm.app import DCDReporter, PDBFile, Simulation, StateDataReporter, Topology
from pandas import DataFrame

from felis.utils.omm.energy_tools import get_energies_in_kcal_mol_by_names
from felis.utils.omm.format_tools import convert_lattice6_to_vec3x3
from felis.utils.omm.omm_system import write_pdb

logger = logging.getLogger(__name__)


def debug_short_ommtools_run(pdb_file: str, xml_system: str, xml_state: str, xml_integrator: str, out_stem: str,
                             nsteps: int, ninterval: int, plfname: str, save_force: int):

    pdb = PDBFile(pdb_file)
    otop: Topology = pdb.topology

    with open(xml_system) as f:
        xml_sys = f.read()
        osys: System = XmlSerializer.deserialize(xml_sys)
    with open(xml_state) as f:
        xml_state = f.read()
        ostate: State = XmlSerializer.deserialize(xml_state)
    with open(xml_integrator) as f:
        xml_itg = f.read()
        oitg: Integrator = XmlSerializer.deserialize(xml_itg)

    oplf: Platform = Platform.getPlatformByName(plfname)
    osim: Simulation = Simulation(otop, osys, oitg, oplf)

    newcsv = f"{out_stem}.csv"
    newdcd = f"{out_stem}.dcd"
    try:
        newpdb0 = f"{out_stem}-0.pdb"
        write_pdb(otop, ostate.getPositions(), ostate.getPeriodicBoxVectors(), newpdb0)
        if nsteps > 0:
            with open(newcsv, "w") as newcsv_handle:
                osim.reporters.append(
                    StateDataReporter(newcsv_handle,
                                      reportInterval=ninterval,
                                      step=True,
                                      time=True,
                                      potentialEnergy=True,
                                      kineticEnergy=True,
                                      totalEnergy=True,
                                      temperature=True,
                                      volume=True,
                                      elapsedTime=True,
                                      separator=","))

                osim.reporters.append(
                    DCDReporter(newdcd, reportInterval=ninterval, append=False, enforcePeriodicBox=False))
                osim.context.setState(ostate)
                vels = ostate.getVelocities(asNumpy=True)
                if np.isnan(vels).any():
                    logger.warning("NaN velocities found in restart file. Reset to random velocities at 298.15 K.")
                    osim.context.setVelocitiesToTemperature(298.15)
                osim.step(nsteps)
    except Exception as e:
        traceback.print_exc()
        raise e
    finally:
        kcal_to_kJ = 4.184
        if Path(newdcd).exists():
            potential_energy_names = []
            for f in osys.getForces():
                if f.getForceGroup() > 0:
                    potential_energy_names.append(f.getName())
            if len(potential_energy_names) == 0:
                for f in osys.getForces():
                    fn = f.getName()
                    if fn.endswith("Force"):
                        potential_energy_names.append(fn)
            u = MDAnalysis.Universe(pdb_file, newdcd)

            get_forces = bool(save_force)
            d2 = defaultdict(list)
            d = defaultdict(list)
            for idx, t in enumerate(u.trajectory):
                pos_nm = t.positions * 0.1
                a, b, c, al, be, ga = t.dimensions
                a, b, c = a * 0.1, b * 0.1, c * 0.1
                vec3x3 = convert_lattice6_to_vec3x3(a, b, c, al, be, ga)
                d1 = get_energies_in_kcal_mol_by_names(potential_energy_names,
                                                       osim,
                                                       pos_nm,
                                                       vec3x3,
                                                       get_forces=get_forces)

                if get_forces:
                    natoms, _ = pos_nm.shape
                    d2["frame"].extend([idx] * natoms)
                    d2["rank"].extend(list(range(natoms)))

                tot = 0.0
                for k, v in d1.items():
                    if k in potential_energy_names:
                        v2 = v * kcal_to_kJ
                        d[k].append(v2)
                        tot = tot + v2
                    if k.endswith("_f") and k[:-2] in potential_energy_names:
                        kx, ky, kz = f"{k}x", f"{k}y", f"{k}z"
                        v2 = v * kcal_to_kJ
                        d2[kx].extend(v2[:, 0].tolist())
                        d2[ky].extend(v2[:, 1].tolist())
                        d2[kz].extend(v2[:, 2].tolist())
                d["Total"].append(tot)

            df = DataFrame(d)
            engtsv = f"{out_stem}-pot.tsv"
            df.to_csv(engtsv, sep="\t", index=False)

            if get_forces:
                df2 = DataFrame(d2)
                frctsv = f"{out_stem}-force.tsv"
                df2.to_csv(frctsv, sep="\t", index=False)
