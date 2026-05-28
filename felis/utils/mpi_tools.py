# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Provide a minimal MPI abstraction with an `mpiplus` backend.

This module imports `mpiplus` and fetches its communicator. If the communicator
is not available (i.e., `mpiplus.get_mpicomm()` returns `None`), it falls back
to a single-process stub.

The public entry point is `mpicomm`, which exposes `mpi_rank`, `mpi_nproc`,
`barrier()`, and `bcast()`.
"""

import logging
import os
from typing import TypeVar

try:
    import mpiplus
    _mpicomm = mpiplus.get_mpicomm()
    _mpi_nproc = _mpicomm.size
    _mpi_rank = _mpicomm.rank
except (ImportError, AttributeError) as exc:
    _test_mpi_np = max((
        int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0)),  # Open MPI
        int(os.environ.get("PMI_SIZE", 0)),  # MPICH / Intel MPI / Slurm PMI
        int(os.environ.get("MV2_COMM_WORLD_SIZE", 0)),  # MVAPICH2
    ))

    if _test_mpi_np > 0:
        raise RuntimeError("Must install mpiplus to run with MPI.") from exc

    _mpicomm = None
    _mpi_nproc = 1
    _mpi_rank = 0

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _MPISimpleInterface:
    """Expose a tiny MPI-like API used by the rest of the codebase.

    This wrapper presents a consistent interface regardless of whether an MPI
    communicator is available.
    """

    def __init__(self):
        """Initialize the interface from the optional backend communicator."""
        self._comm = _mpicomm
        self.mpi_nproc = _mpi_nproc
        self.mpi_rank = _mpi_rank

    def barrier(self):
        """Synchronize ranks.

        If no MPI communicator is available, this is a no-op.
        """
        if self._comm is not None:
            self._comm.barrier()

    def bcast(self, data: T, root: int = 0) -> T:
        """Broadcast data from `root` to all ranks.

        If no MPI communicator is available, return `data` unchanged.

        Args:
            data (Any): Python object to broadcast.
            root (int): Rank that provides the data.

        Returns:
            Any: Broadcasted data.
        """
        if self._comm is not None:
            return self._comm.bcast(data, root=root)
        else:
            return data


mpicomm = _MPISimpleInterface()
