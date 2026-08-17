#!/usr/bin/env python3
"""
One-step nonlinear update consistency test.

This script loads a sparse Fourier velocity field and compares an explicit Euler
nonlinear update using the Cartesian projected-pair RHS and the helical-triad
reconstructed RHS:

    u_ref^{n+1}   = u^n + dt N_ref(u^n),
    u_triad^{n+1} = u^n + dt N_triad(u^n).

It is not intended as a full time-integrated solver.  It is a minimal diagnostic
showing that the RHS-level agreement translates into a one-step update agreement.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fluid_helical_utils import (
    load_modes_npz, cartesian_pair_rhs, helical_pair_rhs, relative_error,
    dict_l2_norm, dict_difference_norm, modes_energy, transversality_error,
    conjugate_symmetry_error, save_json, v_add, v_neg,
)


def add_scaled(a, b, scale):
    out = {k: v.copy() for k, v in a.items()}
    for k, v in b.items():
        out[k] = out.get(k, np.zeros(3, dtype=complex)) + scale * v
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modes", type=str, required=True, help="Sparse modes npz.")
    p.add_argument("--dt-list", type=str, default="1e-3,1e-2,5e-2", help="Comma-separated dt values.")
    p.add_argument("--out", type=str, default="results_one_step_update")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes, metadata = load_modes_npz(args.modes)
    dt_values = [float(x) for x in args.dt_list.split(",") if x.strip()]
    print(f"Loaded modes: {len(modes)}")

    rhs_ref, n_pairs = cartesian_pair_rhs(modes)
    rhs_tri, n_hel, _ = helical_pair_rhs(modes, store_contributions=False)
    rhs_err = relative_error(rhs_tri, rhs_ref)
    rows = []
    for dt in dt_values:
        u_ref = add_scaled(modes, rhs_ref, dt)
        u_tri = add_scaled(modes, rhs_tri, dt)
        step_diff = dict_difference_norm(u_tri, u_ref)
        step_rel = step_diff / (dict_l2_norm(u_ref) + 1e-300)
        rows.append({
            "dt": dt,
            "rhs_relative_error": rhs_err,
            "step_relative_error": step_rel,
            "step_absolute_difference": step_diff,
            "energy_before": modes_energy(modes),
            "energy_after_ref": modes_energy(u_ref),
            "energy_after_triad": modes_energy(u_tri),
            "energy_difference_ref_triad": modes_energy(u_tri) - modes_energy(u_ref),
            "transversality_after_ref": transversality_error(u_ref),
            "transversality_after_triad": transversality_error(u_tri),
            "conjugate_symmetry_after_ref": conjugate_symmetry_error(u_ref),
            "conjugate_symmetry_after_triad": conjugate_symmetry_error(u_tri),
            "cartesian_pair_contributions": n_pairs,
            "helical_amplitude_contributions": n_hel,
        })
        print(f"dt={dt:.3e}: step_rel={step_rel:.3e}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "one_step_update_summary.csv", index=False)
    save_json(out / "one_step_update_metadata.json", {**metadata, "input_modes": len(modes), "rhs_relative_error": rhs_err})

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.loglog(df["dt"], df["step_relative_error"], marker="o", linestyle="-", label="one-step update error")
    ax.axhline(rhs_err, linestyle="--", linewidth=1.0, label="RHS relative error")
    ax.set_xlabel(r"nonlinear update step $\Delta t$")
    ax.set_ylabel("relative difference")
    ax.set_title("One-step update consistency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out / "one_step_error_vs_dt.png", dpi=230)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(df["dt"], df["energy_after_ref"] - df["energy_before"], marker="o", linestyle="-", label="Cartesian reference")
    ax.plot(df["dt"], df["energy_after_triad"] - df["energy_before"], marker="s", linestyle="--", label="Helical triad")
    ax.set_xlabel(r"nonlinear update step $\Delta t$")
    ax.set_ylabel(r"one-step energy change")
    ax.set_title("Energy change under one explicit nonlinear update")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out / "one_step_energy_change.png", dpi=230)
    plt.close(fig)
    print(f"Saved outputs to: {out}")


if __name__ == "__main__":
    main()
