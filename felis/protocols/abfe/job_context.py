# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
import shlex
import subprocess
from typing import Union

from felis.configs import dump_config
from felis.protocols.abfe.config_types import ABStage
from felis.utils import resolve_path

logger = logging.getLogger(__name__)


class JobContext:

    @classmethod
    def build_cuda_mps_mpirun_command(cls, common_cmd: list[str], gpu_id: int, np: int) -> list[str]:
        pipe_dir = f"/tmp/mps_{gpu_id}"
        log_dir = f"/tmp/mps_log_{gpu_id}"
        bash_str = rf"""
    export CUDA_MPS_PIPE_DIRECTORY={pipe_dir}
    export CUDA_MPS_LOG_DIRECTORY={log_dir}

    if [ -d {pipe_dir} ]; then
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    fi
    sleep 5
    rm -rf {pipe_dir} {log_dir}
    mkdir -p {pipe_dir} {log_dir}

    unset CUDA_VISIBLE_DEVICES
    nvidia-cuda-mps-control -d 2>/dev/null || true

    export CUDA_VISIBLE_DEVICES={gpu_id}
    mpirun --allow-run-as-root -np {np} -x NP_VALUE={np} {shlex.join(common_cmd)}
    """
        return ["bash", "-c", bash_str]

    def __init__(self, working_dir: str = ".", extra_envs: dict[str, str] = None, use_mps: bool = False):
        if extra_envs is None:
            extra_envs = dict()
        self.envs = os.environ.copy()
        for k, v in extra_envs.items():
            self.envs[k] = v

        self.working_dir = resolve_path(working_dir)
        self.prepare_dir = str(Path(self.working_dir) / Path("prepare"))
        self.trj_dir = str(Path(self.working_dir) / Path("trj"))
        self.progress_dir = str(Path(self.working_dir) / Path("progress"))
        self.analysis_dir = str(Path(self.working_dir) / Path("analysis"))

        Path(self.prepare_dir).mkdir(parents=True, exist_ok=True)
        Path(self.trj_dir).mkdir(parents=True, exist_ok=True)
        Path(self.progress_dir).mkdir(parents=True, exist_ok=True)

        try:
            if use_mps:
                self._run_one_subprocess(["nvidia-cuda-mps-control", "-d"], self.working_dir)
            else:
                self._run_one_subprocess("echo quit | nvidia-cuda-mps-control", self.working_dir)
        except OSError:
            pass

    def _run_one_subprocess(self,
                            args: Union[list[str], str],
                            working_dir: str,
                            extra_envs: dict[str, str] = None) -> int:
        if extra_envs is None:
            extra_envs = dict()
        env = dict()
        for k, v in self.envs.items():
            env[k] = v
        for k, v in extra_envs.items():
            env[k] = v
        shell = False
        if isinstance(args, str):
            shell = True
        p = subprocess.Popen(args, cwd=working_dir, env=env, shell=shell)
        errcode = p.wait()
        return errcode

    def _run_one_python_function(self, callable_obj, args: list) -> int:
        callable_obj(*args)
        return 0

    def run_one_call(self, args: list, working_dir: str = None, extra_envs: dict[str, str] = None) -> int:
        if extra_envs is None:
            extra_envs = dict()
        call_type = "python_function"
        if isinstance(args[0], str):
            call_type = "subprocess"
        logger.info(f"Calling {call_type} {args}")
        if call_type == "subprocess":
            wdir = working_dir
            if working_dir is None:
                wdir = self.working_dir
            return self._run_one_subprocess(args, wdir, extra_envs)
        elif call_type == "python_function":
            return self._run_one_python_function(args[0], args[1:])
        else:
            assert False  # pragma: no cover

    def run_one_subprocess_nowait(self, args: list[str], working_dir: str, extra_envs: dict[str, str] = None):
        if extra_envs is None:
            extra_envs = dict()
        env = dict()
        for k, v in self.envs.items():
            env[k] = v
        for k, v in extra_envs.items():
            env[k] = v
        wdir = self.working_dir if working_dir is None else working_dir
        logger.info(f"Calling subprocess {args} with {extra_envs}")
        p = subprocess.Popen(args, cwd=wdir, env=env)
        return p

    def check_stage_done(self, stage: ABStage) -> bool:
        done_file = Path(self.progress_dir) / Path(f"{stage.value}.done")
        if done_file.is_file():
            logger.info(f"Skip stage {stage.value}")
            return True
        else:
            logger.info(f"Stage {stage.value}")
            return False

    def write_stage_done(self, stage: ABStage, extra_info_dict: dict = None):
        done_file = Path(self.progress_dir) / Path(f"{stage.value}.done")
        done_file = str(done_file)
        with open(done_file, "w") as _:
            if extra_info_dict is not None:
                dump_config(extra_info_dict, _)
