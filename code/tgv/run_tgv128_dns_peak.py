#!/usr/bin/env python3
"""
Run a dealiased pseudo-spectral Taylor--Green vortex evolution to the
DNS peak-dissipation time and compare global quantities with a public DNS
reference.

The script is designed for the strengthened fluid benchmark in the spectral
triadic quantum transfer manuscript.  It serves two purposes:

1. Generate a dynamically evolved TGV Fourier field, preferably at t=8.97,
   near the peak-dissipation time of the public 512^3 DNS reference.
2. Save a thresholded sparse Fourier snapshot that can be passed to the
   helical RHS reconstruction, shell/helicity transfer, one-step update, and
   active-triad truncation diagnostics.

Important interpretation
------------------------
The public DNS data are used as an external global reference for energy,
dissipation, and enstrophy histories.  The exact nonlinear-RHS reference for
operator reconstruction remains the Cartesian projected convolution evaluated
on the same saved Fourier snapshot.

Recommended Alliance run
------------------------
python run_tgv128_dns_peak.py \
  --N 128 --dt 0.002 --t-final 8.97 --nu 0.000625 \
  --save-times 0,1,2,4,6,8,8.97 \
  --mode-rel-tol 1e-10 --max-modes 2000 \
  --dns-csv dns_reference/hiocfd_ucl_512_global.csv \
  --out results_tgv128_t897
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fluid_helical_utils import (
    tgv_initial_uhat, rk4_step, kinetic_energy, enstrophy_dense,
    dissipation_dense, transversality_error, conjugate_symmetry_error,
    uhat_to_modes, modes_energy, save_modes_npz, save_json,
    energy_spectrum_by_shell, dense_energy_spectrum_by_shell,
    dense_active_mode_count,
)


def parse_list_floats(text: str) -> List[float]:
    if text is None or str(text).strip() == "":
        return []
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def time_tag(t: float) -> str:
    return f"t{t:08.3f}".replace(".", "p")


def find_dns_csv(path_text: str | None) -> Path | None:
    if path_text and path_text.lower() != "auto":
        p = Path(path_text)
        return p if p.exists() else None
    candidates = [
        Path("dns_reference/hiocfd_ucl_512_global.csv"),
        Path("public_TGV_DNS_reference_csv/hiocfd_ucl_512_global.csv"),
        Path("hiocfd_ucl_512_global.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_dns_reference(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=rename)
    needed = {"t", "energy", "dissipation", "enstrophy"}
    if not needed.issubset(df.columns):
        raise ValueError(f"DNS CSV must contain columns {needed}; got {df.columns.tolist()}")
    return df[["t", "energy", "dissipation", "enstrophy"]].copy()


def interpolate_dns(dns: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"time": times})
    for col in ["energy", "dissipation", "enstrophy"]:
        out[f"dns_{col}"] = np.interp(times, dns["t"].to_numpy(), dns[col].to_numpy())
    return out


def save_spectrum_plot(spec: Dict[int, float], path: Path, title: str) -> None:
    rows = sorted(spec.items())
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    if rows:
        ax.semilogy([r[0] for r in rows], [r[1] for r in rows], marker="o", linestyle="-")
    ax.set_xlabel("wavenumber shell")
    ax.set_ylabel("modal kinetic energy")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.30)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def save_snapshot(uhat: np.ndarray, t: float, args, out: Path, final_alias: bool = False) -> Path:
    max_modes = args.max_modes if args.max_modes and args.max_modes > 0 else None
    modes = uhat_to_modes(
        uhat,
        rel_tol=args.mode_rel_tol,
        max_modes=max_modes,
        enforce_conjugate=True,
    )
    tag = time_tag(t)
    path = out / f"tgv_sparse_modes_{tag}.npz"
    metadata = {
        "case": "tgv128_peak_dns_context",
        "time": float(t),
        "N": int(args.N),
        "dt": float(args.dt),
        "t_final": float(args.t_final),
        "nu": float(args.nu),
        "Re": float(1.0 / args.nu) if args.nu != 0 else None,
        "mode_rel_tol": float(args.mode_rel_tol),
        "max_modes_requested": int(args.max_modes),
        "retained_modes": int(len(modes)),
        "retained_sparse_energy": modes_energy(modes),
        "retained_sparse_transversality_error": transversality_error(modes),
        "retained_sparse_conjugate_symmetry_error": conjugate_symmetry_error(modes),
    }
    save_modes_npz(path, modes, metadata)

    spec = energy_spectrum_by_shell(modes)
    pd.DataFrame([{"shell": k, "energy": v} for k, v in sorted(spec.items())]).to_csv(
        out / f"tgv_sparse_energy_spectrum_{tag}.csv", index=False
    )
    save_spectrum_plot(spec, out / f"tgv_sparse_energy_spectrum_{tag}.png", f"Sparse TGV spectrum at t={t:g}")

    if final_alias:
        alias = out / "tgv_peak_sparse_modes.npz"
        shutil.copyfile(path, alias)
        # Also save aliases with names expected by previous scripts.
        shutil.copyfile(path, out / "tgv_short_time_sparse_modes.npz")
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=128, help="Grid size per direction.")
    p.add_argument("--dt", type=float, default=0.002, help="RK4 time step. dt=0.002 exactly reaches t=8.97.")
    p.add_argument("--t-final", type=float, default=8.97, help="Final time, default DNS peak-dissipation time.")
    p.add_argument("--nu", type=float, default=0.000625, help="Kinematic viscosity; nu=1/1600.")
    p.add_argument("--amplitude", type=float, default=1.0, help="Initial TGV amplitude.")
    p.add_argument("--save-times", type=str, default="0,1,2,4,6,8,8.97", help="Comma-separated sparse snapshot times.")
    p.add_argument("--history-every", type=int, default=10, help="Write global diagnostics every this many RK4 steps.")
    p.add_argument("--mode-rel-tol", type=float, default=1e-10, help="Relative threshold for sparse Fourier export.")
    p.add_argument("--max-modes", type=int, default=2000, help="Maximum sparse modes to export; <=0 means no cap.")
    p.add_argument("--dense-active-rel-tol", type=float, default=1e-12, help="Threshold for dense active-mode count diagnostic.")
    p.add_argument("--dns-csv", type=str, default="auto", help="DNS CSV path or auto.")
    p.add_argument("--out", type=str, default="results_tgv128_t897", help="Output directory.")
    p.add_argument("--no-dealias", action="store_true", help="Disable 2/3 dealiasing.")
    p.add_argument("--save-dense-final", action="store_true", help="Save dense final uhat; large file.")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dealias = not args.no_dealias

    steps_float = args.t_final / args.dt
    steps = int(round(steps_float))
    if abs(steps * args.dt - args.t_final) > 1e-10:
        raise ValueError(f"t-final={args.t_final} is not an integer multiple of dt={args.dt}; got {steps_float}")

    save_times = sorted(set([t for t in parse_list_floats(args.save_times) if -1e-12 <= t <= args.t_final + 1e-12]))
    save_steps: Dict[int, float] = {}
    for t in save_times:
        n = int(round(t / args.dt))
        if abs(n * args.dt - t) > max(1e-10, 0.25 * args.dt):
            raise ValueError(f"save time {t} is not compatible with dt={args.dt}")
        save_steps[n] = float(n * args.dt)
    save_steps[steps] = float(args.t_final)

    print("TGV peak benchmark")
    print(f"  N={args.N}, dt={args.dt}, t_final={args.t_final}, steps={steps}, nu={args.nu}")
    print(f"  dealias={dealias}, sparse rel_tol={args.mode_rel_tol}, max_modes={args.max_modes}")
    print(f"  save steps: {save_steps}")

    uhat = tgv_initial_uhat(args.N, amplitude=args.amplitude)
    rows = []
    snapshot_paths = []
    start_time = time.time()

    def record(step: int, t: float, runtime_so_far: float) -> None:
        E = kinetic_energy(uhat)
        Om = enstrophy_dense(uhat)
        eps = 2.0 * args.nu * Om
        active_count = dense_active_mode_count(uhat, rel_tol=args.dense_active_rel_tol)
        rows.append({
            "step": step,
            "time": t,
            "energy": E,
            "dissipation": eps,
            "enstrophy": Om,
            "dense_active_modes": active_count,
            "runtime_seconds": runtime_so_far,
        })

    record(0, 0.0, 0.0)
    if 0 in save_steps:
        snapshot_paths.append(save_snapshot(uhat, 0.0, args, out, final_alias=(steps == 0)))

    for n in range(1, steps + 1):
        uhat = rk4_step(uhat, args.dt, nu=args.nu, dealias=dealias)
        t = n * args.dt
        if n % args.history_every == 0 or n == steps or n in save_steps:
            record(n, t, time.time() - start_time)
        if n in save_steps:
            print(f"  saving sparse snapshot at t={t:.5f}, step={n}")
            snapshot_paths.append(save_snapshot(uhat, t, args, out, final_alias=(n == steps)))
        if n % max(1, steps // 20) == 0 or n == steps:
            E = kinetic_energy(uhat)
            eps = dissipation_dense(uhat, args.nu)
            print(f"  step {n:6d}/{steps}, t={t:8.4f}, E={E:.10e}, eps={eps:.10e}")

    runtime = time.time() - start_time
    history = pd.DataFrame(rows).drop_duplicates(subset=["step"], keep="last").sort_values("step")
    history.to_csv(out / "tgv128_global_history.csv", index=False)

    # Final dense/sparse spectra.
    final_dense_spec = dense_energy_spectrum_by_shell(uhat)
    pd.DataFrame([{"shell": k, "energy": v} for k, v in sorted(final_dense_spec.items())]).to_csv(
        out / "tgv128_dense_energy_spectrum_final.csv", index=False
    )
    save_spectrum_plot(final_dense_spec, out / "tgv128_dense_energy_spectrum_final.png", f"Dense TGV spectrum at t={args.t_final:g}")

    if args.save_dense_final:
        np.savez_compressed(out / "tgv128_final_dense_uhat.npz", uhat=uhat, N=args.N, dt=args.dt, t_final=args.t_final, nu=args.nu)

    # Compare with DNS reference if available.
    dns_path = find_dns_csv(args.dns_csv)
    dns_summary = {}
    if dns_path is not None:
        print(f"  loading DNS reference: {dns_path}")
        dns = read_dns_reference(dns_path)
        dns_peak_idx = int(np.argmax(dns["dissipation"].to_numpy()))
        dns_peak_time = float(dns.iloc[dns_peak_idx]["t"])
        dns_peak_eps = float(dns.iloc[dns_peak_idx]["dissipation"])
        dns_interp = interpolate_dns(dns, history["time"].to_numpy())
        merged = pd.concat([history.reset_index(drop=True), dns_interp.drop(columns=["time"])], axis=1)
        for col in ["energy", "dissipation", "enstrophy"]:
            merged[f"relerr_{col}"] = np.abs(merged[col] - merged[f"dns_{col}"]) / (np.abs(merged[f"dns_{col}"]) + 1e-300)
        merged.to_csv(out / "tgv128_vs_dns_global.csv", index=False)

        # Overlay figures.
        for col, ylabel in [("energy", "kinetic energy"), ("dissipation", "dissipation"), ("enstrophy", "enstrophy")]:
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            ax.plot(dns["t"], dns[col], linestyle="-", linewidth=1.6, label="public $512^3$ DNS")
            ax.plot(history["time"], history[col], linestyle="--", linewidth=1.8, label=f"present ${args.N}^3$ run")
            ax.axvline(dns_peak_time, color="0.3", linestyle=":", linewidth=1.2, label=f"DNS peak $t={dns_peak_time:.2f}$" if col == "dissipation" else None)
            ax.set_xlabel("time")
            ax.set_ylabel(ylabel)
            ax.set_title(f"TGV {ylabel}: present run vs DNS reference")
            ax.grid(True, alpha=0.30)
            ax.legend(frameon=True)
            fig.tight_layout()
            fig.savefig(out / f"tgv128_vs_dns_{col}.png", dpi=240)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for col in ["energy", "dissipation", "enstrophy"]:
            ax.semilogy(merged["time"], merged[f"relerr_{col}"], marker="o", markersize=2.0, linestyle="-", label=col)
        ax.axvline(dns_peak_time, color="0.3", linestyle=":", linewidth=1.2, label=f"DNS peak $t={dns_peak_time:.2f}$")
        ax.set_xlabel("time")
        ax.set_ylabel("relative difference from DNS")
        ax.set_title("Global TGV relative differences against DNS reference")
        ax.grid(True, which="both", alpha=0.30)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(out / "tgv128_vs_dns_relative_errors.png", dpi=240)
        plt.close(fig)

        final_row = merged.iloc[-1]
        dns_summary = {
            "dns_csv": str(dns_path),
            "dns_peak_time": dns_peak_time,
            "dns_peak_dissipation": dns_peak_eps,
            "final_relerr_energy": float(final_row["relerr_energy"]),
            "final_relerr_dissipation": float(final_row["relerr_dissipation"]),
            "final_relerr_enstrophy": float(final_row["relerr_enstrophy"]),
            "final_dns_energy": float(final_row["dns_energy"]),
            "final_dns_dissipation": float(final_row["dns_dissipation"]),
            "final_dns_enstrophy": float(final_row["dns_enstrophy"]),
        }
    else:
        print("  DNS reference not found; skipping DNS comparison.")

    # History figure even without DNS.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(history["time"], history["energy"], label="energy")
    ax.set_xlabel("time")
    ax.set_ylabel("kinetic energy")
    ax.set_title(f"Present TGV run, N={args.N}")
    ax.grid(True, alpha=0.30)
    fig.tight_layout()
    fig.savefig(out / "tgv128_energy_history.png", dpi=240)
    plt.close(fig)

    summary = {
        "case": "tgv128_peak_dns_benchmark",
        "N": int(args.N),
        "dt": float(args.dt),
        "t_final": float(args.t_final),
        "nu": float(args.nu),
        "Re": float(1.0 / args.nu) if args.nu != 0 else None,
        "dealias": bool(dealias),
        "steps": int(steps),
        "runtime_seconds": float(runtime),
        "final_energy": float(kinetic_energy(uhat)),
        "final_dissipation": float(dissipation_dense(uhat, args.nu)),
        "final_enstrophy": float(enstrophy_dense(uhat)),
        "final_dense_active_modes": int(dense_active_mode_count(uhat, rel_tol=args.dense_active_rel_tol)),
        "sparse_snapshot_paths": [str(p) for p in snapshot_paths],
        **dns_summary,
    }
    save_json(out / "tgv128_peak_summary.json", summary)
    print(f"Saved results to {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
