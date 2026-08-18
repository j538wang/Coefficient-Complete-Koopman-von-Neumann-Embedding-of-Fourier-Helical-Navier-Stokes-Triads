#!/usr/bin/env python3
"""Plot the dual-cutoff Gaussian compilation and strength audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-cutoff", type=Path, required=True)
    parser.add_argument("--global-cutoff", type=Path, required=True)
    parser.add_argument("--global-strang", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    local = pd.read_csv(args.local_cutoff)
    global_cutoff = pd.read_csv(args.global_cutoff)
    strang = pd.read_csv(args.global_strang)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25))

    ax = axes[0]
    ax.semilogy(
        local["cutoff"],
        local["maximum_single_factor_state_l2_residual"],
        "o-",
        label="single factor",
    )
    ax.semilogy(
        local["cutoff"],
        local["physical_one_step_state_l2_residual"],
        "s--",
        label="one Strang step",
    )
    ax.semilogy(
        local["cutoff"],
        local["physical_one_step_maximum_intermediate_edge_probability_last_two_levels"],
        "^:",
        label="intermediate edge",
    )
    ax.set_xlabel("local cutoff $d$")
    ax.set_ylabel("residual / probability")
    ax.set_title("Local tensor cutoff")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    ax.semilogy(
        global_cutoff["total_number_cutoff"],
        global_cutoff["maximum_gaussian_map_action_relative_residual"],
        "o-",
        label="map action",
    )
    ax.semilogy(
        global_cutoff["total_number_cutoff"],
        global_cutoff["physical_to_collective_one_step_state_l2_residual"],
        "s--",
        label="physical/collective",
    )
    ax.semilogy(
        global_cutoff["total_number_cutoff"],
        global_cutoff["maximum_intermediate_total_boundary_probability"],
        "^:",
        label="total boundary",
    )
    ax.set_xlabel("total-number cutoff $N$")
    ax.set_title("Passive-invariant cutoff")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    ax.semilogy(
        strang["strang_steps"],
        strang["physical_to_collective_state_l2_residual"],
        "o-",
        label="physical/collective",
    )
    ax.semilogy(
        strang["strang_steps"],
        strang["collective_to_monomial_state_l2_residual"],
        "s--",
        label="collective/monomial",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel("Strang steps $n_{\\rm step}$")
    ax.set_title("Global cutoff $N=16$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(args.out / "gaussian_compilation_validation.pdf", bbox_inches="tight")
    fig.savefig(
        args.out / "gaussian_compilation_validation.png", dpi=240, bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
