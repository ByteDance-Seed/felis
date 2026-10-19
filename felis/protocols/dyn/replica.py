# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import os
from pathlib import Path

import openmm as mm
from openmm import unit
from openmm import Context
from openmm.app import DCDFile, Topology

from felis.configs.global_keys import GlobalKeys
from felis.utils.omm.omm_system import get_initial_pos_pbc, get_integrator, get_system_topology


class Replica:
    """A single Hamiltonian-replica worker state for HREMD.

    The replica owns its OpenMM ``Context`` and (for standalone single-replica
    use) an optional DCD trajectory handle. In HREMD the coordinator process
    owns trajectory output and each worker only updates its ``replica_id`` and
    Hamiltonian parameters on an accepted exchange; the per-replica logger and
    checkpoint targets are pointed at the current Hamiltonian's files by the
    HREMD driver.
    """

    def __init__(
        self,
        system: mm.System,
        topology: Topology,
        is_boresch_restraint: bool = False, # Complex = Protein + Ligand
        replica_id: int = -1,
        pressure: float = None,
        temperature: float = None,
        lambda_schedules: dict = None,
        trajectory_flush_interval: int = 10,
        initial_positions=None,
        initial_box_vectors=None,
        velocity_temperature_K: float = None,
        velocity_seed: int = None,
    ):
        # Contexts are created after fork, so sharing the parent System lets
        # workers benefit from copy-on-write without sharing runtime state.
        self.system = system
        self.context: Context = None
        self.replica_id = replica_id
        self.lambda_schedules = lambda_schedules
        self.integrator: mm.Integrator = None
        self.platform: mm.Platform = None
        self.platform_properties: None
        self.num_replicas: int = None
        self.pressure = pressure
        self.temperature = temperature
        self.topology = topology
        self.is_boresch_restraint = is_boresch_restraint

        # Initial state (applied in init_context() inside each subprocess)
        self.initial_positions = initial_positions
        self.initial_box_vectors = initial_box_vectors
        self.velocity_temperature_K = velocity_temperature_K
        self.velocity_seed = velocity_seed

        # Trajectory state: a DCDFile writer bound to an open file handle.
        # Both are set together by open_trajectory() and cleared together
        # by close_trajectory(). swap_trajectory() moves them to another
        # Replica atomically.
        self._traj_path = None
        self._traj_file = None
        self._traj_writer: DCDFile = None
        self._traj_dt_ps: float = 0.0
        # Flush the underlying file every N frames instead of every frame
        # to avoid fsync overhead dominating on networked filesystems.
        self._traj_flush_interval: int = max(1, int(trajectory_flush_interval))
        self._traj_frames_since_flush: int = 0

        # Per-replica logger state. The logger handle is opened in the worker
        # process and follows replica_id, just like the trajectory handle.
        self.replica_logger: logging.Logger = None
        self._logger_path = None
        self._logger_handler: logging.FileHandler = None

        # Checkpoint target path. Unlike trajectory/logger, checkpoint writes
        # are atomic one-shot writes to the current target file, so no file
        # handle stays open between writes.
        self._checkpoint_path = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def set_num_replicas(self, num_replicas: int):
        self.num_replicas = num_replicas

    def set_integrator(self, integrator: mm.Integrator):
        self.integrator = integrator

    def set_platform(self, platform: mm.Platform, properties: dict):
        self.platform = platform
        self.platform_properties = properties

    def init_context(self):
        if self.system is None:
            raise ValueError("System in Replica is None")
        if self.platform is None:
            raise ValueError("Platform is not specified in Replica.")
        
        if self.platform_properties is not None:
            self.context = Context(self.system, self.integrator, self.platform, self.platform_properties)
        else:
            self.context = Context(self.system, self.integrator, self.platform)
        if self.replica_id < 0:
            raise ValueError("replica_id must be >= 0")
        exist_parameters = self.context.getParameters()
        if "replica_id" not in exist_parameters.keys():
            raise ValueError("replica_id parameter is not found in the system.")
        else:
            self.context.setParameter("replica_id", self.replica_id)
        if "torsionLambda" not in exist_parameters.keys():
            pass # Skip if torsionLambda is not found in the system
        else:
            self.context.setParameter("torsionLambda", self.lambda_schedules["torsionLambda"][self.replica_id])
        if self.is_boresch_restraint:
            self.context.setParameter("reslam", self.lambda_schedules["reslam"][self.replica_id])

        if self.initial_box_vectors is not None:
            self.context.setPeriodicBoxVectors(*self.initial_box_vectors)
        else:
            raise ValueError("initial_box_vectors must be set to initialize the replica.")
        
        if self.initial_positions is not None:
            self.context.setPositions(self.initial_positions)
        else:
            raise ValueError("initial_positions must be set to initialize the replica.")
        
        if self.velocity_temperature_K is not None:
            seed = self.velocity_seed if self.velocity_seed is not None else 0
            self.context.setVelocitiesToTemperature(
                self.velocity_temperature_K * unit.kelvin, seed
            )

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------
    def evaluate_energy(self, with_pv: bool = True):
        state = self.context.getState(getEnergy=True)
        if with_pv:
            if self.pressure is None:
                raise ValueError("pressure must be set to evaluate PV term")
            # Compute PV term using proper OpenMM unit arithmetic
            box = state.getPeriodicBoxVectors()
            volume = box[0][0] * box[1][1] * box[2][2]
            pressure = self.pressure * unit.bar
            # Calculate PV in bar·nm³, convert to kJ/mol using unit system
            pv_term = pressure * volume * unit.AVOGADRO_CONSTANT_NA
            pv_term = pv_term.value_in_unit(unit.kilojoule_per_mole) * unit.kilojoule_per_mole
        else:
            pv_term = 0.0 * unit.kilojoule_per_mole
        return state.getPotentialEnergy() + pv_term

    def evaluate_neighbor_energy(
        self,
        neighbor_replica_id: int,
        with_pv: bool = True,
    ):
        """Evaluate this worker's current configuration under a neighbor Hamiltonian."""
        if neighbor_replica_id < 0 or neighbor_replica_id >= self.num_replicas:
            raise ValueError("neighbor_replica_id out of range")
        exist_parameters = self.context.getParameters()
        if "replica_id" not in exist_parameters.keys():
            raise ValueError("replica_id parameter is not found in the system.")
        else:
            self.context.setParameter("replica_id", neighbor_replica_id)
        if "torsionLambda" in exist_parameters.keys():
            self.context.setParameter(
                "torsionLambda",
                self.lambda_schedules["torsionLambda"][neighbor_replica_id],
            )

        if self.is_boresch_restraint:
            self.context.setParameter(
                "reslam", self.lambda_schedules["reslam"][neighbor_replica_id]
            )
        try:
            energy = self.evaluate_energy(with_pv)
        finally:
            # Restore original parameters
            if "replica_id" not in exist_parameters.keys():
                raise ValueError("replica_id parameter is not found in the system.")
            else:
                self.context.setParameter("replica_id", self.replica_id)
            if "torsionLambda" in exist_parameters.keys():
                self.context.setParameter(
                    "torsionLambda", self.lambda_schedules["torsionLambda"][self.replica_id]
                )
            if self.is_boresch_restraint:
                self.context.setParameter(
                    "reslam", self.lambda_schedules["reslam"][self.replica_id]
                )
        return energy

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def update_replica_id(self, replica_id: int):
        self.replica_id = replica_id
        exist_parameters = self.context.getParameters()
        if "replica_id" not in exist_parameters.keys():
            raise ValueError("replica_id parameter is not found in the system.")
        else:
            self.context.setParameter("replica_id", replica_id)
        if "torsionLambda" in exist_parameters.keys():
            self.context.setParameter("torsionLambda", self.lambda_schedules["torsionLambda"][replica_id])
        
        if self.is_boresch_restraint:
            self.context.setParameter("reslam", self.lambda_schedules["reslam"][replica_id])

    def step(self, num_steps: int):
        try:
            self.integrator.step(num_steps)
        except Exception as e:
            self.replica_logger.error(f"Step failed in replica {self.replica_id}: {e}")
            raise

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------
    def close_checkpoint(self):
        """Clear the currently configured checkpoint target."""
        self._checkpoint_path = None

    def write_checkpoint(self, chunk_id: int, step_count: int):
        """Atomically write one generation-specific OpenMM checkpoint.

        ``records.sqlite`` is the sole commit authority. The coordinator only
        publishes a generation there after every worker has durably replaced
        its checkpoint binary and crossed the checkpoint barrier.
        """
        if self._checkpoint_path is None:
            raise RuntimeError(
                f"Replica {self.replica_id} has no checkpoint target configured"
            )
        if self.context is None:
            raise RuntimeError(
                f"Replica {self.replica_id} has no OpenMM Context for checkpointing"
            )
        if chunk_id < 0:
            raise ValueError("chunk_id must be non-negative")
        context_step = int(self.context.getStepCount())
        if int(step_count) != context_step:
            raise RuntimeError(
                f"Checkpoint step_count {step_count} does not match Context "
                f"step {context_step}"
            )

        checkpoint_path = Path(self._checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
        checkpoint_data = self.context.createCheckpoint()
        with open(tmp_path, "wb") as fh:
            fh.write(checkpoint_data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, checkpoint_path)
        directory_fd = os.open(checkpoint_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def get_step_count(self) -> int:
        """Return the integrator step count of the current OpenMM Context."""
        if self.context is None:
            raise RuntimeError(
                f"Replica {self.replica_id} has no OpenMM Context"
            )
        return int(self.context.getStepCount())

    def load_checkpoint(self, path) -> int:
        """Load an OpenMM checkpoint into the existing context.

        Restores positions/velocities/box and global parameters from the
        checkpoint, then re-asserts this replica's Hamiltonian parameters for
        safety. Returns the integrator step count stored in the checkpoint.
        """
        if self.context is None:
            raise RuntimeError(
                f"Replica {self.replica_id} has no OpenMM Context to load into"
            )
        with open(path, "rb") as fh:
            data = fh.read()
        self.context.loadCheckpoint(data)
        # Re-assert Hamiltonian parameters in case a future OpenMM version
        # does not restore global parameters from the checkpoint.
        exist_parameters = self.context.getParameters()
        if "replica_id" not in exist_parameters.keys():
            raise ValueError("replica_id parameter is not found in the system.")
        else:
            self.context.setParameter("replica_id", self.replica_id)
        if "torsionLambda" in exist_parameters.keys():
            self.context.setParameter(
                "torsionLambda", self.lambda_schedules["torsionLambda"][self.replica_id]
            )
        if self.is_boresch_restraint:
            self.context.setParameter(
                "reslam", self.lambda_schedules["reslam"][self.replica_id]
            )
        return int(self.context.getStepCount())

    # ------------------------------------------------------------------
    # Trajectory management
    # ------------------------------------------------------------------
    def open_trajectory(self, path, dt_ps: float = 0.0):
        """Open (or append to) a DCD trajectory bound to this replica.

        Args:
            path: Path to the DCD file.
            dt_ps: Timestep in picoseconds written into the DCD header.
        """
        if self._traj_writer is not None:
            raise RuntimeError(
                f"Replica {self.replica_id} already has an open trajectory"
            )
        self._traj_path = str(path)
        self._traj_dt_ps = float(dt_ps)
        append = os.path.exists(self._traj_path) and os.path.getsize(self._traj_path) > 0
        mode = "r+b" if append else "wb"
        self._traj_file = open(self._traj_path, mode)
        topology = self.topology
        if topology is None:
            raise ValueError(
                "Replica.topology must be provided to write DCD trajectories"
            )
        self._traj_writer = DCDFile(
            self._traj_file,
            topology,
            self._traj_dt_ps * unit.picoseconds,
            append=append,
        )

    def set_trajectory_flush_interval(self, n: int):
        """Flush the trajectory file every ``n`` frames (>= 1)."""
        self._traj_flush_interval = max(1, int(n))

    def write_trajectory(self):
        """Write one frame (positions + box) to the current trajectory.

        The underlying file is flushed every ``trajectory_flush_interval``
        frames (configurable via :meth:`set_trajectory_flush_interval`),
        not on every frame, to keep I/O cost low.
        """
        if self._traj_writer is None:
            raise RuntimeError(
                f"Replica {self.replica_id} has no open trajectory handle"
            )
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self._traj_writer.writeModel(
            positions=state.getPositions(),
            periodicBoxVectors=state.getPeriodicBoxVectors(),
        )
        self._traj_frames_since_flush += 1
        if self._traj_frames_since_flush >= self._traj_flush_interval:
            self._traj_file.flush()
            self._traj_frames_since_flush = 0

    def close_trajectory(self):
        """Close the currently open trajectory handle (if any)."""
        if self._traj_file is not None:
            self._traj_file.flush()
            self._traj_file.close()
        self._traj_path = None
        self._traj_file = None
        self._traj_writer = None
        self._traj_dt_ps = 0.0
        self._traj_frames_since_flush = 0

def get_list_of_replicas(gks: list[GlobalKeys]) -> list[Replica]:
    """Build a list of :class:`~felis.protocols.dyn.replica.Replica` for HREMD.

    Each entry in ``gks`` is treated as one Hamiltonian (e.g. one lambda
    state for FEP). The resulting replicas are configured so that:

    - ``replica.replica_id`` is the Hamiltonian id (0..n-1).
    - the OpenMM system/topology/integrator/platform are set on the Replica
      object, but the actual OpenMM Context is created later in each
      subprocess (see :meth:`Replica.init_context`).
    - initial positions/box and velocities are attached to each Replica
      so the child process can initialize the Context deterministically.
    """
    if not gks:
        return []

    n = len(gks)

    # Build ONE shared system/topology/integrator from the first GK.
    # HREMD assumes the Hamiltonian is selected by Context parameter
    # "replica_id" (handled inside the System/forces), so all workers
    # can share the same System object.
    gk0 = gks[0]
    osys, otop = get_system_topology(gk0, enable_perturb_nonbonded=True)
    
    pos0, box0 = get_initial_pos_pbc(gk0)

    is_boresch_restraint = False
    if gk0.boresch.proatoms is not None:
        is_boresch_restraint = True
    else:
        is_boresch_restraint = False

    replicas: list[Replica] = []
    for replica_id in range(n):
        gk = gks[replica_id]
        rep = Replica(
            system=osys,
            topology=otop,
            is_boresch_restraint=is_boresch_restraint,
            replica_id=replica_id,
            lambda_schedules = gk0.ab.lambda_schedules,
            pressure=gk0.integrator.targetP_bar,
            temperature=gk0.integrator.targetT_K,
            initial_positions=pos0,
            initial_box_vectors=box0,
            velocity_temperature_K=gk0.integrator.targetT_K,
            velocity_seed=gk.integrator.randomseed,
        )
        itg = get_integrator(gk)
        rep.set_integrator(itg)
        rep.set_num_replicas(n)
        replicas.append(rep)

    return replicas
