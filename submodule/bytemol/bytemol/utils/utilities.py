# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# some functions in this file are adapted from openff-toolkit
# https://github.com/openforcefield/openff-toolkit/blob/main/openff/toolkit/utils/utils.py
# MIT License
# Copyright (c) 2016-2019 Open Force Field Initiative

import dataclasses
import errno
import json
import logging
import math
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import PosixPath
from tempfile import TemporaryDirectory
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@contextmanager
def temporary_cd(directory_path: Optional[str] = None) -> Generator[None, None, None]:
    """Context manager to temporarily change the working directory.

    Temporarily changes the current working directory to the specified path. 
    If no path is provided, a temporary directory will be created and used.
    The original working directory is restored automatically when exiting the context.

    Parameters
    ----------
    directory_path : Optional[str], default: None
        Path to the target directory. If None, a temporary directory is created.
        Accepts both string paths and Path objects (automatically converted to strings).

    Yields
    ------
    Generator[None, None, None]
        Context manager that yields control after changing directories.

    Examples
    --------
    >>> with temporary_cd("/tmp/new_dir"):
    ...     # Working directory is now "/tmp/new_dir"
    ...     pass
    >>> # Working directory restored to original

    >>> with temporary_cd():
    ...     # Working directory is a temporary directory
    ...     temp_path = os.getcwd()
    >>> # Temporary directory is automatically cleaned up
    """
    if isinstance(directory_path, PosixPath):
        directory_path = directory_path.as_posix()

    if directory_path is not None and len(directory_path) == 0:
        yield
        return

    old_directory = os.getcwd()

    try:

        if directory_path is None:

            with TemporaryDirectory() as new_directory:
                os.chdir(new_directory)
                yield

        else:

            os.makedirs(directory_path, exist_ok=True)
            os.chdir(directory_path)
            yield

    finally:
        os.chdir(old_directory)

    return


def is_file_and_not_empty(file_path):
    """Checks that a file both exists at the specified ``path`` and is not empty.

    Parameters
    ----------
    file_path: str
        The file path to check.

    Returns
    -------
    bool
        That a file both exists at the specified ``path`` and is not empty.
    """
    return os.path.isfile(file_path) and (os.path.getsize(file_path) != 0)


def get_data_file_path(relative_path: str, package_name: str) -> str:
    """Get the full path to one of the files in the data directory.

    If no file is found at `relative_path`, a second attempt will be made
    with `data/` preprended. If no files exist at either path, a FileNotFoundError
    is raised.

    Parameters
    ----------
    relative_path : str
        The relative path of the file to load.
    package_name : str
        The name of the package in which a file is to be loaded, i.e.

    Returns
    -------
        The absolute path to the file.

    Raises
    ------
    FileNotFoundError
    """
    from importlib.resources import files

    file_path = files(package_name) / relative_path

    if not file_path.is_file():
        try_path = files(package_name) / f"data/{relative_path}"
        if try_path.is_file():
            file_path = try_path
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

    return file_path.as_posix()  # type: ignore


def run_command_and_check(cmd: str,
                          *,
                          allow_error: bool = False,
                          separate_stderr: bool = True,
                          env: Dict = None,
                          redirect_stdout=True,
                          user_input: str = None,
                          timeout: float = None) -> Tuple[int, str, str]:
    """Executes a shell command and returns its output.

    Args:
        cmd: The command to execute.
        allow_error: If True, do not raise an exception for non-zero exit codes.
        separate_stderr: If True, capture stderr separately. Otherwise, it's merged with stdout.
        env: A dictionary of environment variables to set for the command.
        redirect_stdout: If True, capture stdout.
        user_input: A string to pass to the command's stdin.
        timeout: The maximum time in seconds to wait for the command to complete.

    Returns:
        A tuple containing the return code, stdout, and stderr.

    Raises:
        RuntimeError: If the command returns a non-zero exit code and `allow_error` is False.
    """

    cmdenv = dict()
    cmdenv.update(**os.environ)
    if env is not None:
        cmdenv.update(**env)

    logger.debug(f'running cmd {cmd} with extra env {env}')
    stdout = subprocess.PIPE if redirect_stdout else None
    stderr = subprocess.PIPE if separate_stderr else subprocess.STDOUT
    stderr = stderr if redirect_stdout else None
    user_input = user_input.encode() if isinstance(user_input, str) else None
    result = subprocess.run(cmd,
                            stdout=stdout,
                            stderr=stderr,
                            env=cmdenv,
                            shell=True,
                            check=False,
                            input=user_input,
                            timeout=timeout)

    stdout = result.stdout.decode('utf-8') if result.stdout is not None else None
    stderr = result.stderr.decode('utf-8') if result.stderr is not None else None

    if result.returncode != 0 and not allow_error:
        logger.error(f'cmd {cmd}')
        logger.error(f'return code {result.returncode}')
        logger.error(f'stdout {stdout}')
        logger.error(f'stderr {stderr}')
        raise RuntimeError(f'fail to run "{cmd}". ')

    return result.returncode, stdout, stderr


def split_array_evenly(array: List, num_parts: int) -> List[List]:
    """
    Split an array into num_parts subarrays of roughly equal size.

    Args:
        array: An array to be split.
        num_parts: The number of subarrays to split the array into.

    Returns:
        A list of lists, where each sublist contains a roughly equal portion of the original array.
        The number of sublists equals num_parts, unless the length of the array is not evenly divisible by num_parts.
    """
    num_elements_per_part = math.ceil(len(array) / num_parts)
    vacancy = num_parts * num_elements_per_part - len(array)
    subarrays = []
    start_index = 0

    for i in range(num_parts):
        end_index = start_index + num_elements_per_part - 1 if i >= num_parts - vacancy else start_index + num_elements_per_part
        subarrays.append(array[start_index:end_index])
        start_index = end_index

    return subarrays


def get_current_time_str():
    return datetime.now().strftime("%y_%m_%d_%H_%M_%S")


def convert_keys_to_string(obj):
    if dataclasses.is_dataclass(obj):
        return convert_keys_to_string(dataclasses.asdict(obj))
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [convert_keys_to_string(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): convert_keys_to_string(value) for key, value in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_dict_to_json(data, file_name, **kwargs):
    string_key_data = convert_keys_to_string(data)

    with open(file_name, 'w') as json_file:
        json.dump(string_key_data, json_file, **kwargs)