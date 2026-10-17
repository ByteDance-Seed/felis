# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Test helpers for ABFE artifact stage tests.

These helpers consolidate the repeated arrange-stage-prerequisites code that
appears across ``felis/tests/artifacts/protocols/abfe/test_stage_*.py`` so the
tests can focus on the behavior under verification.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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


def make_abfe_input_config(
    outdir: str,
    sdffile: Optional[str] = None,
    itpfile: Optional[str] = None,
):
    """Build a checked ``ABFEInputConfig`` for artifact tests."""
    from felis.protocols.abfe.config_types import ABFEInputConfig

    kwargs: dict = {"outdir": outdir}
    if sdffile is not None:
        kwargs["sdffile"] = sdffile
    if itpfile is not None:
        kwargs["itpfile"] = itpfile
    acfg = ABFEInputConfig(**kwargs)
    acfg.check()
    return acfg
