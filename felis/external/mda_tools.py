# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import os
from pathlib import Path

import numpy as np

from felis.configs import GlobalKeys

logger = logging.getLogger(__name__)

try:
    from MDAnalysis import Universe
    from MDAnalysis import Writer
except ImportError as exc:  # pragma: no cover - exercised only when optional dependency is absent
    _MDA_IMPORT_ERROR = exc

    def Universe(*_args, **_kwargs):
        raise _MDA_IMPORT_ERROR

    def Writer(*_args, **_kwargs):
        raise _MDA_IMPORT_ERROR

try:
    from felis.utils.topfile_tools import get_top_system_symbols
except ImportError as exc:  # pragma: no cover - exercised only when optional dependency is absent
    _TOPFILE_IMPORT_ERROR = exc

    def get_top_system_symbols(*_args, **_kwargs):
        raise _TOPFILE_IMPORT_ERROR


def get_mda_universe(gk: GlobalKeys, trj: str) -> Universe:
    if gk.openmm.params_ecosystem == "gromacs":
        gmxtop = gk.filename.sys
        u = Universe(gmxtop, trj, topology_format="ITP")
        sys_symbols = get_top_system_symbols(gmxtop)
        np_symbols = np.array([elem.upper() for elem in sys_symbols], dtype=object)  # must be upper case!
        u.add_TopologyAttr("elements", np_symbols)
        return u
    else:
        raise NotImplementedError(gk.openmm.params_ecosystem)


def get_trj_nsnapshots(gk: GlobalKeys, trj: str) -> int:
    u = get_mda_universe(gk, trj)
    return len(u.trajectory)


def discard_initial_trajectory_frames(gk: GlobalKeys, trj: str, discard: int) -> None:
    if discard <= 0:
        return

    u = get_mda_universe(gk, trj)
    n_frames = len(u.trajectory)
    if n_frames <= discard:
        raise ValueError(f"Cannot discard {discard} frames from trajectory with {n_frames} frames: {trj}")

    trj_path = Path(trj)
    tmp_path = trj_path.with_name(f"{trj_path.stem}.discard.tmp{trj_path.suffix}")
    try:
        with Writer(str(tmp_path), n_atoms=u.atoms.n_atoms) as writer:
            for frame_id, _ts in enumerate(u.trajectory):
                if frame_id >= discard:
                    writer.write(u.atoms)
        close = getattr(u.trajectory, "close", None)
        if close is not None:
            close()
        os.replace(tmp_path, trj_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def concatenate_trajectory_files(gk: GlobalKeys, trj_list: list[str], out_trj: str, discard: int = 0) -> None:
    if not trj_list:
        raise ValueError("trj_list must contain at least one trajectory")
    if discard < 0:
        raise ValueError("discard must be non-negative")

    out_path = Path(out_trj)
    tmp_path = out_path.with_name(f"{out_path.stem}.concat.tmp{out_path.suffix}")
    n_written = 0
    try:
        first_universe = get_mda_universe(gk, trj_list[0])
        n_atoms = first_universe.atoms.n_atoms
        close = getattr(first_universe.trajectory, "close", None)
        if close is not None:
            close()

        with Writer(str(tmp_path), n_atoms=n_atoms) as writer:
            for trj in trj_list:
                u = get_mda_universe(gk, trj)
                if u.atoms.n_atoms != n_atoms:
                    raise ValueError(
                        f"Trajectory atom count mismatch for {trj}: "
                        f"{u.atoms.n_atoms} != {n_atoms}"
                    )
                for frame_id, _ts in enumerate(u.trajectory):
                    if frame_id >= discard:
                        writer.write(u.atoms)
                        n_written += 1
                close = getattr(u.trajectory, "close", None)
                if close is not None:
                    close()
        if n_written == 0:
            raise ValueError(
                f"Cannot concatenate trajectories after discarding {discard} frames from each input"
            )
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
