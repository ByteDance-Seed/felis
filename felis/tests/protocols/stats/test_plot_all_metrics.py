# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Unit test for plot_all_metrics function.

This module tests the plotting functionality of the dg_group_all_runs module,
including generation of scatter plots comparing computed vs experimental dG values.
"""

from pathlib import Path
import struct
import zlib

import numpy as np
import pytest

from felis.protocols.stats.main_dg_group_all_runs import plot_all_metrics
from felis.utils.metrics import get_metrics

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _read_png_chunks(fileobj):
    """Yield (chunk_type, chunk_data) tuples from a PNG file positioned after the signature."""
    while True:
        header = fileobj.read(8)
        if len(header) < 8:
            break
        chunk_len = struct.unpack(">I", header[:4])[0]
        chunk_type = header[4:8]
        chunk_data = fileobj.read(chunk_len) if chunk_len > 0 else b""
        fileobj.read(4)  # CRC
        yield chunk_type, chunk_data
        if chunk_type == b"IEND":
            break


def _png_raw_pixels(path):
    """Parse *path* as PNG and return ``(width, height, pixels_flat, samples_per_pixel)``.

    ``pixels_flat`` is a :class:`bytes` object containing unfiltered 8-bit
    pixel data in row-major order.  Only 8-bit RGB / RGBA / grayscale /
    grayscale+alpha colour types are supported (the typical output of
    matplotlib).
    """
    with open(path, "rb") as fh:
        sig = fh.read(8)
        if sig != _PNG_SIGNATURE:
            raise AssertionError(f"{path} is not a valid PNG file")

        width = height = None
        bit_depth = color_type = None
        idat_chunks = []

        for chunk_type, chunk_data in _read_png_chunks(fh):
            if chunk_type == b"IHDR":
                width, height = struct.unpack(">II", chunk_data[:8])
                bit_depth = chunk_data[8]
                color_type = chunk_data[9]
            elif chunk_type == b"IDAT":
                idat_chunks.append(chunk_data)

    if width is None or height is None:
        raise AssertionError(f"{path} is missing IHDR chunk")

    if bit_depth != 8:
        raise AssertionError(f"{path} has unsupported bit depth {bit_depth}")

    # samples per pixel
    if color_type == 0:  # Grayscale
        samples = 1
    elif color_type == 2:  # RGB
        samples = 3
    elif color_type == 4:  # Grayscale + Alpha
        samples = 2
    elif color_type == 6:  # RGBA
        samples = 4
    else:
        raise AssertionError(f"{path} has unsupported colour type {color_type}")

    raw = zlib.decompress(b"".join(idat_chunks))
    bpp = samples  # bytes per pixel (bit_depth == 8)
    stride = width * bpp

    prev = bytearray(stride)
    out = bytearray()

    pos = 0
    for _ in range(height):
        ft = raw[pos]
        pos += 1
        row = raw[pos:pos + stride]
        pos += stride

        if ft == 0:  # None
            cur = bytearray(row)
        elif ft == 1:  # Sub
            cur = bytearray(row)
            for i in range(bpp, stride):
                cur[i] = (cur[i] + cur[i - bpp]) & 0xFF
        elif ft == 2:  # Up
            cur = bytearray(row)
            for i in range(stride):
                cur[i] = (cur[i] + prev[i]) & 0xFF
        elif ft == 3:  # Average
            cur = bytearray(row)
            for i in range(stride):
                left = cur[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + (left + prev[i]) // 2) & 0xFF
        elif ft == 4:  # Paeth
            cur = bytearray(row)
            for i in range(stride):
                left = cur[i - bpp] if i >= bpp else 0
                up = prev[i]
                up_left = prev[i - bpp] if i >= bpp else 0
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                pr = left if pa <= pb and pa <= pc else (up if pb <= pc else up_left)
                cur[i] = (cur[i] + pr) & 0xFF
        else:
            raise AssertionError(f"{path} has unknown filter type {ft}")

        out.extend(cur)
        prev = cur

    return width, height, bytes(out), samples


def assert_valid_nonblank_png(path: Path, *, min_width: int = 100, min_height: int = 100) -> None:
    """Assert that ``path`` is a non-blank PNG image with plausible dimensions.

    Parses the PNG with the standard library (:mod:`struct`, :mod:`zlib`) so
    that :mod:`PIL` is not required.  Verifies the file is a real PNG, has at
    least the requested width/height, and contains non-zero variance once
    converted to grayscale (catching blank/uniform-colour images that would
    still pass a file-size check).
    """

    assert path.exists(), f"PNG file {path} was not created"
    width, height, pixels, samples = _png_raw_pixels(path)

    assert width >= min_width, f"{path} width {width} < {min_width}"
    assert height >= min_height, f"{path} height {height} < {min_height}"

    arr = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, samples)
    if samples >= 3:
        # RGB / RGBA  ->  luminance
        gray = (arr[:, :, 0].astype(np.float64) * 0.299 + arr[:, :, 1].astype(np.float64) * 0.587 +
                arr[:, :, 2].astype(np.float64) * 0.114)
    else:
        gray = arr[:, :, 0].astype(np.float64)

    assert gray.size > 0, f"{path} contains no pixel data"
    assert float(gray.var()) > 0.0, f"{path} appears blank (zero grayscale variance)"


def create_mock_plot_data(  # pylint: disable=too-many-positional-arguments
    n_ligands=8,
    n_runs=3,
    seed=42,
    systematic_shift=3.0,
    noise_std=0.5,
    run_noise_std=0.3,
):
    """
    Create mock data for plot_all_metrics testing.

    This function generates synthetic data that mimics the output of ABFE calculations
    with experimental and predicted dG values. All parameters are tweakable to allow
    testing different plot styles and scenarios.

    Args:
        n_ligands: Number of ligands/data points to generate
        n_runs: Number of independent runs (affects legend entries)
        seed: Random seed for reproducibility
        systematic_shift: Systematic shift between predicted and experimental values
        noise_std: Standard deviation of noise in mean prediction
        run_noise_std: Standard deviation of noise between individual runs

    Returns:
        Dictionary containing all necessary inputs for plot_all_metrics:
        - np_dG_exp: Experimental dG values (n_ligands,)
        - np_dG_mean: Mean predicted dG values (n_ligands,)
        - np_dG_runs: Individual run predictions (n_runs, n_ligands)
        - mean_metrics: Metrics object with calculated statistics
        - proname: Protein name for plot title
        - np_dG_stdev: Standard deviation across runs (n_ligands,)
    """
    rng = np.random.default_rng(seed)

    # Generate experimental dG values - spaced across typical binding affinity range
    # Using a range from -12 to -3 kcal/mol (typical for drug-like molecules)
    base_exp = np.linspace(-11.5, -3.5, n_ligands)
    # Add small random variation to make it more realistic
    np_dG_exp = base_exp + rng.normal(0, 0.2, n_ligands)

    # Generate mean predicted dG values with systematic shift and noise
    # This simulates a typical ABFE calculation with some systematic error
    np_dG_mean = np_dG_exp + rng.normal(systematic_shift, noise_std, n_ligands)

    # Generate individual runs with additional variance
    np_dG_runs = np.zeros((n_runs, n_ligands))
    for i in range(n_runs):
        # Each run has variance around the mean
        np_dG_runs[i] = np_dG_mean + rng.normal(0, run_noise_std, n_ligands)

    # Calculate standard deviation across runs
    np_dG_stdev = np.std(np_dG_runs, axis=0, ddof=1)

    # Calculate metrics using get_metrics
    mean_metrics = get_metrics(pred=np_dG_mean, label=np_dG_exp)

    return {
        "np_dG_exp": np_dG_exp,
        "np_dG_mean": np_dG_mean,
        "np_dG_runs": np_dG_runs,
        "mean_metrics": mean_metrics,
        "proname": "TEST_PROTEIN",
        "np_dG_stdev": np_dG_stdev,
    }


@pytest.fixture
def mock_plot_data():
    """Fixture providing default mock data for plot testing."""
    return create_mock_plot_data()


class TestPlotAllMetrics:
    """Test suite for plot_all_metrics function."""

    # pylint: disable=redefined-outer-name
    def test_basic_plot_generation(self, mock_plot_data, tmp_path):
        """
        Test that all 4 PNG plots are generated successfully.

        This test verifies the core functionality of plot_all_metrics by:
        1. Creating output paths for all 4 plot types
        2. Calling plot_all_metrics with mock data
        3. Verifying all 4 PNG files exist and are non-empty
        """
        # Define output paths
        pngname = tmp_path / "stats_pred_label.png"
        pngname_shifted = tmp_path / "stats_shiftedpred_label.png"
        pngname_stdev = tmp_path / "stats_pred_label_stdev.png"
        pngname_shifted_stdev = tmp_path / "stats_shiftedpred_stdev.png"

        # Call the function
        plot_all_metrics(
            np_dG_exp=mock_plot_data["np_dG_exp"],
            np_dG_mean=mock_plot_data["np_dG_mean"],
            np_dG_runs=mock_plot_data["np_dG_runs"],
            mean_metrics=mock_plot_data["mean_metrics"],
            proname=mock_plot_data["proname"],
            pngname=str(pngname),
            pngname_shifted=str(pngname_shifted),
            np_dG_stdev=mock_plot_data["np_dG_stdev"],
            pngname_stdev=str(pngname_stdev),
            pngname_shifted_stdev=str(pngname_shifted_stdev),
        )

        # Verify all 4 PNG files exist, are real PNGs, and not blank.
        for png_file in [pngname, pngname_shifted, pngname_stdev, pngname_shifted_stdev]:
            assert_valid_nonblank_png(png_file)

    def test_plot_with_different_styles(self, tmp_path):
        """
        Test that plots can be customized by tweaking mock data parameters.

        This test verifies the flexibility of the plotting function by:
        1. Creating mock data with different parameters (more ligands, more runs)
        2. Verifying plots are generated successfully with the modified data
        3. Confirming the plot files are valid PNGs

        This allows users to conveniently tweak plot styles by adjusting:
        - n_ligands: Number of data points
        - n_runs: Number of runs (affects legend)
        - systematic_shift: Changes the offset in shifted plots
        - noise parameters: Affects data spread
        """
        # Create custom mock data with different parameters
        custom_data = create_mock_plot_data(
            n_ligands=12,  # More ligands
            n_runs=5,  # More runs
            seed=123,  # Different seed
            systematic_shift=3.5,  # Larger systematic shift
            noise_std=0.8,  # More noise
            run_noise_std=0.5,  # More run variance
        )

        # Define output paths
        pngname = tmp_path / "custom_stats_pred_label.png"
        pngname_shifted = tmp_path / "custom_stats_shiftedpred_label.png"
        pngname_stdev = tmp_path / "custom_stats_pred_stdev.png"
        pngname_shifted_stdev = tmp_path / "custom_stats_shiftedpred_stdev.png"

        # Call the function with custom data
        plot_all_metrics(
            np_dG_exp=custom_data["np_dG_exp"],
            np_dG_mean=custom_data["np_dG_mean"],
            np_dG_runs=custom_data["np_dG_runs"],
            mean_metrics=custom_data["mean_metrics"],
            proname="CUSTOM_PROTEIN",
            pngname=str(pngname),
            pngname_shifted=str(pngname_shifted),
            np_dG_stdev=custom_data["np_dG_stdev"],
            pngname_stdev=str(pngname_stdev),
            pngname_shifted_stdev=str(pngname_shifted_stdev),
        )

        # Verify all 4 PNG files exist, are real PNGs, and not blank.
        for png_file in [pngname, pngname_shifted, pngname_stdev, pngname_shifted_stdev]:
            assert_valid_nonblank_png(png_file)
