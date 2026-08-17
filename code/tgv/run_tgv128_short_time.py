#!/usr/bin/env python3
"""
Short-time evolved Taylor--Green vortex benchmark generator.

This script runs a classical pseudo-spectral Navier--Stokes short-time evolution
from the Taylor--Green vortex on an N^3 grid, then exports a thresholded sparse
Fourier field for the extended helical-RHS diagnostics.

The exported sparse field can be used by:
  * run_shell_helicity_transfer.py
  * run_one_step_update.py
  * run_active_triad_truncation.py

Recommended first Alliance run:
  python run_tgv128_short_time.py --N 128 --dt 0.001 --t-final 0.05 \
      --nu 0.000625 --mode-rel-tol 1e-9 --max-modes 1200 \
      --out results_tgv128

Notes
-----
* nu=0.000625 corresponds to Re=1600 if U=L=1 are used.
* The sparse threshold controls the explicit triad cost in later scripts.
* Increasing max-modes makes the later explicit triad diagnostics more expensive.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fluid_helical_utils import (
    tgv_initial_uhat, rk4_step, kinetic_energy, pseudospectral_rhs,
    transversality_error, conjugate_symmetry_error, uhat_to_modes,
    modes_energy, save_modes_npz, save_json, energy_spectrum_by_shell,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=128, help="Grid size per direction.")
    p.add_argument("--dt", type=float, default=0.001, help="RK4 timestep.")
    p.add_argument("--t-final", type=float, default=0.05, help="Final short-time evolution time.")
    p.add_argument("--nu", type=float, default=0.000625, help="Kinematic viscosity.")
    p.add_argument("--amplitude", type=float, default=1.0, help="Initial TGV amplitude.")
    p.add_argument("--mode-rel-tol", type=float, default=1e-9, help="Relative threshold for exporting sparse modes.")
    p.add_argument("--max-modes", type=int, default=1200, help="Maximum sparse modes to export; use <=0 for no cap.")
    p.add_argument("--out", type=str, default="results_tgv128_short_time", help="Output directory.")
    p.add_argument("--no-dealias", action="store_true", help="Disable 2/3 dealiasing.")
    p.add_argument("--save-dense", action="store_true", help="Also save dense final uhat; can be large.")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dealias = not args.no_dealias
    steps = int(round(args.t_final / args.dt))
    if abs(steps * args.dt - args.t_final) > 1e-12:
        raise ValueError("t-final must be an integer multiple of dt for this script.")

    print(f"Generating TGV: N={args.N}, dt={args.dt}, t_final={args.t_final}, steps={steps}, nu={args.nu}")
    uhat = tgv_initial_uhat(args.N, amplitude=args.amplitude)
    E0 = kinetic_energy(uhat)
    rows = [{"step": 0, "time": 0.0, "energy": E0}]

    tic = time.time()
    for n in range(1, steps + 1):
        uhat = rk4_step(uhat, args.dt, nu=args.nu, dealias=dealias)
        if n == steps or n % max(1, steps // 10) == 0:
            E = kinetic_energy(uhat)
            rows.append({"step": n, "time": n * args.dt, "energy": E})
            print(f"  step {n:5d}/{steps}, t={n*args.dt:.5f}, E={E:.12e}")
    runtime = time.time() - tic

    dense_path = None
    if args.save_dense:
        dense_path = str(out / "tgv_final_dense_uhat.npz")
        np.savez_compressed(dense_path, uhat=uhat, N=args.N, dt=args.dt, t_final=args.t_final, nu=args.nu)

    max_modes = args.max_modes if args.max_modes and args.max_modes > 0 else None
    modes = uhat_to_modes(uhat, rel_tol=args.mode_rel_tol, max_modes=max_modes, enforce_conjugate=True)
    sparse_path = out / "tgv_short_time_sparse_modes.npz"
    metadata = {
        "case": "short_time_evolved_tgv",
        "N": args.N,
        "dt": args.dt,
        "t_final": args.t_final,
        "nu": args.nu,
        "dealias": dealias,
        "mode_rel_tol": args.mode_rel_tol,
        "max_modes_requested": args.max_modes,
        "retained_modes": len(modes),
        "dense_path": dense_path,
        "runtime_seconds": runtime,
    }
    save_modes_npz(sparse_path, modes, metadata)

    # Diagnostics for exported sparse field.
    spec = energy_spectrum_by_shell(modes)
    pd.DataFrame(rows).to_csv(out / "tgv_short_time_energy_history.csv", index=False)
    pd.DataFrame([{"shell": k, "energy": v} for k, v in sorted(spec.items())]).to_csv(out / "tgv_sparse_energy_spectrum.csv", index=False)

    diag = dict(metadata)
    diag.update({
        "initial_energy_dense": E0,
        "final_energy_dense": kinetic_energy(uhat),
        "retained_sparse_energy": modes_energy(modes),
        "retained_sparse_transversality_error": transversality_error(modes),
        "retained_sparse_conjugate_symmetry_error": conjugate_symmetry_error(modes),
    })
    save_json(out / "tgv_short_time_summary.json", diag)

    # Energy history figure.
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(df["time"], df["energy"], marker="o", linestyle="-")
    ax.set_xlabel("time")
    ax.set_ylabel("kinetic energy")
    ax.set_title("Short-time evolved Taylor--Green vortex")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tgv_short_time_energy_history.png", dpi=220)
    plt.close(fig)

    # Sparse spectrum figure.
    srows = sorted(spec.items())
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy([x[0] for x in srows], [x[1] for x in srows], marker="o", linestyle="-")
    ax.set_xlabel("wavenumber shell")
    ax.set_ylabel("retained modal energy")
    ax.set_title("Sparse exported TGV spectrum")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tgv_sparse_energy_spectrum.png", dpi=220)
    plt.close(fig)

    print(f"Saved sparse modes to: {sparse_path}")
    print(f"Retained modes: {len(modes)}")
    print(f"Runtime: {runtime:.2f} s")


if __name__ == "__main__":
    main()
