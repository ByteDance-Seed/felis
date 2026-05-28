# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

from felis.utils.metrics import get_metrics


def parse_args():
    """Parse command line arguments for the Schrodinger benchmark analysis tool.

    Returns:
        argparse.Namespace: Parsed command line arguments with the following attributes:
            root_dir (str): Root directory containing Schrodinger benchmark subfolders.
                Default: ../../felis/pl_bfe_dataset/cleaned_datasets/Schrodinger
            output_file (str): Output file path to save the calculated statistics.
                Default: schrodinger_benchmark_stats.csv
            threshold (float): Threshold for correct order ratio calculation in kcal/mol.
                Default: 2.0
    """
    parser = argparse.ArgumentParser(description="Analyze Schrodinger benchmarks and calculate statistics")
    parser.add_argument("--root-dir",
                        type=str,
                        default="../../pl_bfe_dataset/Schrodinger",
                        help="Root directory containing Schrodinger benchmark subfolders")
    parser.add_argument("--output-file",
                        type=str,
                        default="schrodinger_benchmark_stats.csv",
                        help="Output file to save the statistics")
    parser.add_argument("--threshold",
                        type=float,
                        default=2.0,
                        help="Threshold for correct order ratio calculation (kcal/mol)")
    return parser.parse_args()


def analyze_single_benchmark(benchmark_file: str, threshold: float = 2.0):
    """Analyze a single Schrodinger benchmark CSV file and calculate free energy metrics.

    Reads and validates the benchmark data, removes invalid entries, deduplicates by group ID,
    and computes statistical metrics comparing experimental and FEP-calculated free energies.

    Args:
        benchmark_file (str): Path to the Schrodinger benchmark CSV file.
        threshold (float): Threshold for correct order ratio calculation in kcal/mol.
            Default: 2.0

    Returns:
        dict or None: Dictionary containing calculated statistics with the following keys:
            R2 (float): R-squared correlation coefficient
            PearsonR (float): Pearson correlation coefficient
            Spearman (float): Spearman rank correlation coefficient
            KendallTau (float): Kendall's Tau correlation coefficient
            Shift (float): Mean unsigned shift between FEP and experimental values
            RMSE (float): Root mean squared error
            MAE (float): Mean absolute error
            CorrectOrderRatio (float): Ratio of compound pairs with FEP order matching experimental
            NumValidPoints (int): Number of valid data points used for analysis
            NumPoints (int): Total number of points in the input file
        Returns None if analysis fails or insufficient valid data points.

    Raises:
        IOError: If required columns are missing from the input file.
        AssertionError: If 'group_id' column is missing from the input file.
        Exception: For general errors during file reading or analysis.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(benchmark_file)

        # Check if required columns exist
        if 'exp_dG_kcal_mol' not in df.columns or 'FEP+_21-4_RBFEP_dG_kcal_mol' not in df.columns:
            raise IOError(f"Missing required columns in {benchmark_file}")

        assert 'group_id' in df.columns, f"Missing 'group_id' column in {benchmark_file}"

        # First remove rows with NaN in either exp_dG or fep_dG
        df_valid = df.dropna(subset=['exp_dG_kcal_mol', 'FEP+_21-4_RBFEP_dG_kcal_mol'])

        # Deduplicate by group_id, keeping the first occurrence
        df_dedup = df_valid.drop_duplicates(subset=['group_id'], keep='first')

        print(f"Deduplicated: {len(df_valid)} -> {len(df_dedup)} rows (by group_id)")

        # Extract the relevant columns from deduplicated data
        exp_dG = df_dedup['exp_dG_kcal_mol'].values
        fep_dG = df_dedup['FEP+_21-4_RBFEP_dG_kcal_mol'].values

        if len(exp_dG) < 2:
            print(f"Warning: Insufficient valid data points in {benchmark_file}")
            return None

        # Calculate metrics
        metrics = get_metrics(fep_dG, exp_dG, threshold)

        return {
            'R2': metrics.R2,
            'PearsonR': metrics.PearsonR,
            'Spearman': metrics.Spearman,
            'KendallTau': metrics.KendallTau,
            'Shift': metrics.Shift,
            'RMSE': metrics.RMSE,
            'MAE': metrics.MAE,
            'CorrectOrderRatio': metrics.CorrectOrderRatio,
            'NumValidPoints': len(exp_dG),
            'NumPoints': len(df),
        }

    except Exception as e:
        print(f"Error analyzing {benchmark_file}: {str(e)}")
        return None


def main():
    """Main entry point for the Schrodinger benchmark analysis tool.

    Orchestrates the analysis workflow:
    1. Parses command line arguments
    2. Creates output directory if it doesn't exist
    3. Finds all benchmark CSV files in the specified root directory
    4. Analyzes each benchmark file
    5. Collects and saves results to a CSV file
    6. Prints summary statistics of the results

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    root_dir = args.root_dir
    output_file = args.output_file
    threshold = args.threshold

    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Find all open_source_benchmarks.csv files in subfolders
    benchmark_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'open_source_benchmarks.csv':
                benchmark_files.append(os.path.join(root, file))

    print(f"Found {len(benchmark_files)} benchmark files")

    # Analyze each benchmark file
    results = []
    for benchmark_file in benchmark_files:
        # Get the relative path from root_dir
        relative_path = os.path.relpath(benchmark_file, root_dir)
        print(f"Analyzing: {relative_path}")

        stats = analyze_single_benchmark(benchmark_file, threshold)
        if stats:
            stats['folder'] = relative_path
            results.append(stats)

    # Write results to CSV
    if results:
        # Define the column order
        columns = [
            'folder', 'NumValidPoints', 'R2', 'PearsonR', 'Spearman', 'KendallTau', 'Shift', 'RMSE', 'MAE',
            'CorrectOrderRatio'
        ]

        # Create a DataFrame from results
        df_results = pd.DataFrame(results)
        df_results = df_results[columns]  # Reorder columns

        # Save to CSV
        df_results.to_csv(output_file, index=False, float_format="%.4f")
        print(f"Results saved to {output_file}")

        # Print summary statistics
        print("\nSummary Statistics:")
        print(df_results.describe())
    else:
        print("No valid results to save.")


if __name__ == "__main__":
    main()
