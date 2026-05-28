# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from PIL import Image

logger = logging.getLogger(__name__)


def merge_images(files, per_row, output_file, clean: bool = False):
    images = [Image.open(f) for f in files]
    total_images = len(images)
    num_rows = (total_images + per_row - 1) // per_row

    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)

    merged_width = per_row * max_width
    merged_height = num_rows * max_height
    merged_image = Image.new("RGB", (merged_width, merged_height), color=(255, 255, 255))

    for index, img in enumerate(images):
        row = index // per_row
        col = index % per_row
        x = col * max_width
        y = row * max_height
        merged_image.paste(img, (x, y))
    merged_image.save(output_file)

    if clean:
        for f in files:
            os.remove(f)
