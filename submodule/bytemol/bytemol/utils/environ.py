# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

from matplotlib.font_manager import FontProperties, findfont


def set_omp_num_threads(nt: int = 1):
    os.environ['OMP_NUM_THREADS'] = str(nt)


def get_omp_num_threads():
    return int(os.environ['OMP_NUM_THREADS'])


def launch_mps_server():
    while True:
        if subprocess.call('echo quit | nvidia-cuda-mps-control', shell=True) != 0:
            break
    print('launching mps server')
    subprocess.call('nvidia-cuda-mps-control -d', shell=True)


def find_default_font(style: str = 'sans_serif', fonttext: str = 'ttf') -> str:
    font = findfont(FontProperties(family=[style]), fontext=fonttext)
    if isinstance(font, list):
        font = font[0]
    return font
