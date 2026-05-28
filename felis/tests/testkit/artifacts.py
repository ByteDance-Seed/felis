# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for ABFE artifact stage tests.

These helpers consolidate the repeated arrange-stage-prerequisites code that
appears across ``felis/tests/artifacts/protocols/abfe/test_stage_*.py`` so the
tests can focus on the behavior under verification.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional


@dataclass
class ArtifactStageContext:
    """Bundle of paths and the JobContext used by ABFE artifact stage tests.

    Attributes
    ----------
    tmp_path: working directory root used by the test.
    lig_stem: ligand stem used to derive the job root.
    job_root: relative job root directory (``Path(lig_stem)``).
    job_context: ``felis.protocols.abfe.job_context.JobContext`` for the job.
    """

    tmp_path: Path
    lig_stem: str
    job_root: Path
    job_context: object


def make_artifact_stage_context(monkeypatch, tmp_path: Path, lig_stem: str = "ligand0") -> ArtifactStageContext:
    """Create the standard ABFE stage test context.

    The current working directory is changed to ``tmp_path`` via ``monkeypatch.chdir``
    so the chdir is automatically reverted on test teardown (including failures).
    """
    from felis.protocols.abfe.job_context import JobContext

    monkeypatch.chdir(tmp_path)
    job_root = Path(lig_stem)
    return ArtifactStageContext(
        tmp_path=tmp_path,
        lig_stem=lig_stem,
        job_root=job_root,
        job_context=JobContext(job_root),
    )


def seed_text_files(mapping: Mapping[Path, str]) -> None:
    """Create each parent directory and write the given text content."""
    for path, content in mapping.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def seed_empty_files(paths: Iterable[Path]) -> None:
    """Create each parent directory and ``touch`` an empty file."""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def replica_exchange_manifest(prefixes: Iterable[str], ndev: int, np: int) -> list[str]:
    """Return the canonical replica-exchange output manifest under ``trj/``.

    For each ``prefix`` and replica index ``i`` in ``range(ndev)``, the per-replica
    files ``{prefix}{i}.create_done``, ``{prefix}{i}.nc``, ``{prefix}{i}_checkpoint.nc``
    and the per-rank ``{prefix}{i}.rank{j:02d}.rexlog`` for ``j`` in ``range(np)`` are
    emitted, in that order.
    """
    files: list[str] = []
    for prefix in prefixes:
        for idx in range(ndev):
            stem = f"{prefix}{idx}"
            files.append(f"trj/{stem}.create_done")
            files.append(f"trj/{stem}.nc")
            files.append(f"trj/{stem}_checkpoint.nc")
            for j in range(np):
                files.append(f"trj/{stem}.rank{j:02d}.rexlog")
    return files


def boresch_npt_manifest(stem: str, np: int) -> list[str]:
    """Return the canonical replica-exchange output manifest for the boresch NPT stage."""
    files = [
        f"trj/{stem}.create_done",
        f"trj/{stem}.nc",
        f"trj/{stem}_checkpoint.nc",
        f"trj/{stem}.dcd",
    ]
    for j in range(np):
        files.append(f"trj/{stem}.rank{j:02d}.rexlog")
    return files


def make_abfe_input_config(
    outdir: str,
    sdffile: Optional[str] = None,
    itpfile: Optional[str] = None,
    elamrecipe: Optional[str] = None,
    vlamrecipe: Optional[str] = None,
    reslamrecipe: Optional[str] = None,
):
    """Build a checked ``ABFEInputConfig`` with optional lambda recipe overrides."""
    from felis.protocols.abfe.config_types import ABFEInputConfig

    kwargs: dict = {"outdir": outdir}
    if sdffile is not None:
        kwargs["sdffile"] = sdffile
    if itpfile is not None:
        kwargs["itpfile"] = itpfile
    if elamrecipe is not None:
        kwargs["elamrecipe"] = elamrecipe
    if vlamrecipe is not None:
        kwargs["vlamrecipe"] = vlamrecipe
    if reslamrecipe is not None:
        kwargs["reslamrecipe"] = reslamrecipe
    acfg = ABFEInputConfig(**kwargs)
    acfg.check()
    return acfg
