#!/usr/bin/env python3
"""
Compute spectra, shell energy/helicity transfer, and cumulative fluxes from a
full-domain periodic velocity cube saved as `.npz`.

Important correction relative to earlier versions:
  1. Conservation diagnostics are computed over ALL Fourier shells in the
     cubic spectral grid, not only k <= N/2.  In 3-D, many valid modes have
     radial magnitude |k| > N/2 even though each component lies in [-N/2,N/2].
  2. If `--mode galerkin23` is used, the 2/3 mask is applied to the INPUT
     velocity before forming the nonlinear RHS.  This creates a closed
     Galerkin-truncated system and gives meaningful conservative transfer
     diagnostics on the retained modes.

Modes:
  --mode dns_full    : use all modes present on the coarse periodic cube.
                       Best for checking the nonlinear operator on the cube.
  --mode galerkin23  : apply a component-wise 2/3 Galerkin filter to uhat
                       before computing the nonlinearity. Best for a clean
                       conservative pseudo-spectral reference.

For JHTDB stride-downsampled data, this is still an instantaneous transfer
reconstruction test, not a fully helical cascade validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fftfreq_wavenumbers(N: int, L: float):
    k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    return np.meshgrid(k1, k1, k1, indexing="ij")


def curl_hat(vhat, kx, ky, kz):
    vx, vy, vz = vhat
    return np.array([
        1j * (ky * vz - kz * vy),
        1j * (kz * vx - kx * vz),
        1j * (kx * vy - ky * vx),
    ])


def project_incompressible(vhat, kx, ky, kz):
    vx, vy, vz = vhat
    k2 = kx*kx + ky*ky + kz*kz
    kdotv = kx*vx + ky*vy + kz*vz
    out = np.array([vx.copy(), vy.copy(), vz.copy()])
    mask = k2 > 0
    out[0][mask] -= kx[mask] * kdotv[mask] / k2[mask]
    out[1][mask] -= ky[mask] * kdotv[mask] / k2[mask]
    out[2][mask] -= kz[mask] * kdotv[mask] / k2[mask]
    out[:, ~mask] = 0.0
    return out


def mask_23(kx, ky, kz, N):
    kcut = N // 3
    return (np.abs(kx) <= kcut) & (np.abs(ky) <= kcut) & (np.abs(kz) <= kcut)


def pseudo_spectral_rhs_rotational(uhat, kx, ky, kz):
    omega_hat = curl_hat(uhat, kx, ky, kz)
    u = np.fft.ifftn(uhat, axes=(1, 2, 3))
    omega = np.fft.ifftn(omega_hat, axes=(1, 2, 3))

    # Rotational form: -(u·grad)u = u × omega - grad(|u|^2/2).
    # The pressure/gradient part is removed by the incompressible projection.
    nx = u[1]*omega[2] - u[2]*omega[1]
    ny = u[2]*omega[0] - u[0]*omega[2]
    nz = u[0]*omega[1] - u[1]*omega[0]
    nhat = np.fft.fftn(np.array([nx, ny, nz]), axes=(1, 2, 3))
    nhat = project_incompressible(nhat, kx, ky, kz)
    nhat[:, 0, 0, 0] = 0.0
    return nhat


def shell_statistics(uhat, nhat, kx, ky, kz, active_mask=None):
    omega_hat = curl_hat(uhat, kx, ky, kz)
    kabs = np.sqrt(kx*kx + ky*ky + kz*kz)
    shell = np.rint(kabs).astype(np.int32)
    kmax = int(shell.max())

    if active_mask is None:
        active_mask = np.ones_like(kabs, dtype=bool)
    active_mask = active_mask & (kabs > 0)

    E_point = 0.5 * np.real(np.sum(np.conjugate(uhat) * uhat, axis=0))
    H_point = 0.5 * np.real(np.sum(np.conjugate(uhat) * omega_hat, axis=0))
    TE_point = np.real(np.sum(np.conjugate(uhat) * nhat, axis=0))
    TH_point = np.real(np.sum(np.conjugate(omega_hat) * nhat, axis=0))

    rows = []
    for ell in range(kmax + 1):
        m = (shell == ell) & active_mask
        rows.append({
            "k": ell,
            "E": float(E_point[m].sum()),
            "H": float(H_point[m].sum()),
            "T_E": float(TE_point[m].sum()),
            "T_H": float(TH_point[m].sum()),
        })
    df = pd.DataFrame(rows)
    df["Pi_E"] = -df["T_E"].cumsum()
    df["Pi_H"] = -df["T_H"].cumsum()
    return df


def diagnostics(df):
    TE = df["T_E"].to_numpy()
    TH = df["T_H"].to_numpy()
    PiE = df["Pi_E"].to_numpy()
    PiH = df["Pi_H"].to_numpy()
    return {
        "sum_T_E": float(TE.sum()),
        "sum_abs_T_E": float(np.sum(np.abs(TE))),
        "energy_transfer_residual": float(abs(TE.sum()) / (np.sum(np.abs(TE)) + 1e-300)),
        "sum_T_H": float(TH.sum()),
        "sum_abs_T_H": float(np.sum(np.abs(TH))),
        "helicity_transfer_residual": float(abs(TH.sum()) / (np.sum(np.abs(TH)) + 1e-300)),
        "Pi_E_end": float(PiE[-1]),
        "Pi_E_max_abs": float(np.max(np.abs(PiE))),
        "energy_flux_endpoint_residual": float(abs(PiE[-1]) / (np.max(np.abs(PiE)) + 1e-300)),
        "Pi_H_end": float(PiH[-1]),
        "Pi_H_max_abs": float(np.max(np.abs(PiH))),
        "helicity_flux_endpoint_residual": float(abs(PiH[-1]) / (np.max(np.abs(PiH)) + 1e-300)),
    }


def make_plots(df, outdir: Path, stem: str, kplot_max=None):
    outdir.mkdir(parents=True, exist_ok=True)
    d = df if kplot_max is None else df[df["k"] <= kplot_max]

    plt.figure(figsize=(6.5, 4.2))
    mask = (d["k"] > 0) & (d["E"] > 0)
    plt.loglog(d.loc[mask, "k"], d.loc[mask, "E"], marker="o", linewidth=1.0, markersize=3)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k)$")
    plt.title("Energy spectrum")
    plt.tight_layout()
    plt.savefig(outdir / f"{stem}_energy_spectrum.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(d["k"], d["H"], marker="o", linewidth=1.0, markersize=3)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$H(k)$")
    plt.title("Helicity spectrum")
    plt.tight_layout()
    plt.savefig(outdir / f"{stem}_helicity_spectrum.png", dpi=220)
    plt.close()

    for col, lab, title, suffix in [
        ("T_E", r"$T_E(k)$", "Shell energy transfer", "shell_energy_transfer"),
        ("T_H", r"$T_H(k)$", "Shell helicity transfer", "shell_helicity_transfer"),
        ("Pi_E", r"$\Pi_E(k)$", "Cumulative energy flux", "energy_flux"),
        ("Pi_H", r"$\Pi_H(k)$", "Cumulative helicity flux", "helicity_flux"),
    ]:
        plt.figure(figsize=(6.5, 4.2))
        plt.plot(d["k"], d[col], marker="o", linewidth=1.0, markersize=3)
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel(r"$k$")
        plt.ylabel(lab)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outdir / f"{stem}_{suffix}.png", dpi=220)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compute spectral helicity transfer from a velocity cube.")
    parser.add_argument("--input", required=True, help="Input .npz file from 01_fetch_jhtdb_full_domain_coarse.py")
    parser.add_argument("--outdir", default="results_jhtdb")
    parser.add_argument("--mode", choices=["dns_full", "galerkin23"], default="galerkin23",
                        help="dns_full: no spectral filtering; galerkin23: closed 2/3 Galerkin-truncated reference")
    parser.add_argument("--kplot-max", type=int, default=None, help="Only limit plotted range; CSV still contains all shells.")
    args = parser.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(inp, allow_pickle=True)
    u = np.asarray(data["u"], dtype=np.float64)
    L = float(data.get("domain_length", 2*np.pi))
    if u.ndim != 4 or u.shape[0] != 3:
        raise ValueError(f"Expected u with shape (3,N,N,N), got {u.shape}")
    N = u.shape[1]

    uhat = np.fft.fftn(u, axes=(1, 2, 3))
    kx, ky, kz = fftfreq_wavenumbers(N, L)
    uhat = project_incompressible(uhat, kx, ky, kz)
    uhat[:, 0, 0, 0] = 0.0

    active_mask = None
    if args.mode == "galerkin23":
        active_mask = mask_23(kx, ky, kz, N)
        uhat *= active_mask[None, :, :, :]

    nhat = pseudo_spectral_rhs_rotational(uhat, kx, ky, kz)
    if args.mode == "galerkin23":
        nhat *= active_mask[None, :, :, :]

    df = shell_statistics(uhat, nhat, kx, ky, kz, active_mask=active_mask)

    stem = f"{inp.stem}_{args.mode}"
    csv = outdir / f"{stem}_spectral_transfers.csv"
    df.to_csv(csv, index=False)
    make_plots(df, outdir, stem, kplot_max=args.kplot_max)

    diag = diagnostics(df)
    diag_path = outdir / f"{stem}_diagnostics.txt"
    with open(diag_path, "w") as f:
        for key, val in diag.items():
            f.write(f"{key}: {val:.16e}\n")
    print(f"wrote {csv}")
    print(f"wrote {diag_path}")
    print(f"N={N}, mode={args.mode}")
    print(f"energy_transfer_residual={diag['energy_transfer_residual']:.6e}")
    print(f"helicity_transfer_residual={diag['helicity_transfer_residual']:.6e}")
    print(f"energy_flux_endpoint_residual={diag['energy_flux_endpoint_residual']:.6e}")
    print(f"helicity_flux_endpoint_residual={diag['helicity_flux_endpoint_residual']:.6e}")


if __name__ == "__main__":
    main()
