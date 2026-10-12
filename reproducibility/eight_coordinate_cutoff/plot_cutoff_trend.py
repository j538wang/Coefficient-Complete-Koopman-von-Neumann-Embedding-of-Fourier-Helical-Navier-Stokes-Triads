#!/usr/bin/env python3
"""Create a compact reviewer-response figure from the completed cutoff scan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-stem", type=Path, default=Path(__file__).resolve().parent / "eight_coordinate_cutoff_trend")
    args = parser.parse_args()
    root = args.results_dir
    summary = read_csv(root / "cutoff_summary.csv")
    convergence = read_csv(root / "cutoff_convergence.csv")
    n = np.asarray([int(row["total_number_cutoff"]) for row in summary])
    boundary = np.asarray([float(row["maximum_boundary_probability_last_two_shells"]) for row in summary])
    classical_error = np.asarray([float(row["final_relative_mean_error_vs_classical"]) for row in summary])
    mean_reference = np.asarray(
        [float(row["relative_mean_trajectory_difference_vs_reference"]) for row in convergence]
    )
    covariance_reference = np.asarray(
        [float(row["final_relative_covariance_frobenius_difference_vs_reference"]) for row in convergence]
    )

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    panels = (
        (axes[0, 0], boundary, r"max. last-two-shell probability", "(a) Boundary occupation"),
        (axes[0, 1], classical_error, r"final mean error vs classical", "(b) Mean-field error"),
        (axes[1, 0], mean_reference, r"mean trajectory difference vs $N=16$", "(c) Cutoff convergence of means"),
        (axes[1, 1], covariance_reference, r"final covariance difference vs $N=16$", "(d) Cutoff convergence of covariance"),
    )
    for axis, values, ylabel, title in panels:
        positive = values > 0
        axis.semilogy(n[positive], values[positive], "o-", color="#1f5a94", lw=1.6, ms=5)
        axis.set_xticks(n)
        axis.set_xlabel(r"total-number cutoff $N$")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=9.5)
        axis.grid(True, which="both", alpha=0.22)
    fig.suptitle("Full eight-coordinate two-triad KvN cutoff scan", fontsize=11)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(args.output_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
