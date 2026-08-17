#!/usr/bin/env python3
"""
Helical-channel reconstruction benchmark for a JHTDB velocity cube.

This script uses the same Galerkin-truncated pseudo-spectral reference as
02_compute_helicity_flux_from_npz.py, then decomposes the velocity field into
helical components u^+ and u^- and reconstructs the nonlinear RHS as a sum of
8 helical channels:

    N^{r;pq}(k) = P_r(k) P(k) FFT[ u^p(x) x omega^q(x) ],

where r,p,q in {+,-}, P is the incompressible projection, and P_r is the
helical output projection. The sum over r,p,q should reproduce the reference
projected nonlinear RHS to roundoff.

Outputs:
  *_helical_total_compare.csv
  *_helical_channel_transfers.csv
  *_helical_reconstruction_diagnostics.txt
  comparison plots for T_E, T_H, Pi_E, Pi_H
  channel plots for helicity flux by channel.

Interpretation:
  This is a helicity-resolved triadic/RHS reconstruction test on a standard
  JHTDB forced-HIT snapshot. It is not a fully helical cascade validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import itertools
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

    # Rotational form: -(u·grad)u = u x omega - grad(|u|^2/2).
    # The pressure/gradient part is removed by the incompressible projection.
    nx = u[1]*omega[2] - u[2]*omega[1]
    ny = u[2]*omega[0] - u[0]*omega[2]
    nz = u[0]*omega[1] - u[1]*omega[0]
    nhat = np.fft.fftn(np.array([nx, ny, nz]), axes=(1, 2, 3))
    nhat = project_incompressible(nhat, kx, ky, kz)
    nhat[:, 0, 0, 0] = 0.0
    return nhat


def helical_basis(kx, ky, kz):
    """Return h_plus, h_minus with i k x h_s = s |k| h_s.

    The returned arrays have shape (3,N,N,N). At k=0 they are set to zero.
    """
    kabs = np.sqrt(kx*kx + ky*ky + kz*kz)
    mask = kabs > 0

    khat = np.zeros((3,) + kx.shape, dtype=np.float64)
    khat[0][mask] = kx[mask] / kabs[mask]
    khat[1][mask] = ky[mask] / kabs[mask]
    khat[2][mask] = kz[mask] / kabs[mask]

    # Choose a reference direction not parallel to khat.
    ref = np.zeros_like(khat)
    # Default ref = z-hat.
    ref[2] = 1.0
    # If |khat dot z| is too large, use y-hat.
    near_z = np.abs(khat[2]) > 0.9
    ref[:, near_z] = 0.0
    ref[1][near_z] = 1.0

    # e1 = normalize(ref x khat), e2 = khat x e1.
    e1 = np.empty_like(khat)
    e1[0] = ref[1]*khat[2] - ref[2]*khat[1]
    e1[1] = ref[2]*khat[0] - ref[0]*khat[2]
    e1[2] = ref[0]*khat[1] - ref[1]*khat[0]
    e1n = np.sqrt(np.sum(e1*e1, axis=0))
    good = mask & (e1n > 0)
    for j in range(3):
        e1[j][good] /= e1n[good]
        e1[j][~good] = 0.0

    e2 = np.empty_like(khat)
    e2[0] = khat[1]*e1[2] - khat[2]*e1[1]
    e2[1] = khat[2]*e1[0] - khat[0]*e1[2]
    e2[2] = khat[0]*e1[1] - khat[1]*e1[0]

    hplus = (e1 + 1j*e2) / np.sqrt(2.0)
    hminus = (e1 - 1j*e2) / np.sqrt(2.0)
    hplus[:, ~mask] = 0.0
    hminus[:, ~mask] = 0.0
    return hplus.astype(np.complex128), hminus.astype(np.complex128)


def helical_project(vhat, h):
    """Project a vector Fourier field onto one helical basis h."""
    coeff = np.sum(np.conjugate(h) * vhat, axis=0)
    return h * coeff[None, :, :, :]


def shell_df_for_rhs(uhat, nhat, kx, ky, kz, active_mask):
    omega_hat = curl_hat(uhat, kx, ky, kz)
    kabs = np.sqrt(kx*kx + ky*ky + kz*kz)
    shell = np.rint(kabs).astype(np.int32)
    kmax = int(shell.max())
    active = active_mask & (kabs > 0)

    E_point = 0.5 * np.real(np.sum(np.conjugate(uhat) * uhat, axis=0))
    H_point = 0.5 * np.real(np.sum(np.conjugate(uhat) * omega_hat, axis=0))
    TE_point = np.real(np.sum(np.conjugate(uhat) * nhat, axis=0))
    TH_point = np.real(np.sum(np.conjugate(omega_hat) * nhat, axis=0))

    rows = []
    for ell in range(kmax + 1):
        m = (shell == ell) & active
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


def shell_transfer_for_channel(uhat_r, omega_hat_r, nhat_channel, kx, ky, kz, active_mask, r_label, p_label, q_label):
    kabs = np.sqrt(kx*kx + ky*ky + kz*kz)
    shell = np.rint(kabs).astype(np.int32)
    kmax = int(shell.max())
    active = active_mask & (kabs > 0)
    TE_point = np.real(np.sum(np.conjugate(uhat_r) * nhat_channel, axis=0))
    TH_point = np.real(np.sum(np.conjugate(omega_hat_r) * nhat_channel, axis=0))
    rows = []
    for ell in range(kmax + 1):
        m = (shell == ell) & active
        rows.append({
            "k": ell,
            "r": r_label,
            "p": p_label,
            "q": q_label,
            "channel": f"{r_label};{p_label}{q_label}",
            "T_E": float(TE_point[m].sum()),
            "T_H": float(TH_point[m].sum()),
        })
    df = pd.DataFrame(rows)
    df["Pi_E"] = -df["T_E"].cumsum()
    df["Pi_H"] = -df["T_H"].cumsum()
    return df


def rel_l2(a, b):
    return float(np.linalg.norm((a-b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-300))


def make_compare_plots(df, outdir: Path, stem: str, kplot_max=None):
    d = df if kplot_max is None else df[df["k"] <= kplot_max]
    for base, ylabel, title, suffix in [
        ("T_E", r"$T_E(k)$", "Shell energy transfer", "helical_compare_T_E"),
        ("T_H", r"$T_H(k)$", "Shell helicity transfer", "helical_compare_T_H"),
        ("Pi_E", r"$\Pi_E(k)$", "Cumulative energy flux", "helical_compare_Pi_E"),
        ("Pi_H", r"$\Pi_H(k)$", "Cumulative helicity flux", "helical_compare_Pi_H"),
    ]:
        plt.figure(figsize=(6.5, 4.2))
        plt.plot(d["k"], d[f"{base}_ref"], marker="o", linewidth=1.0, markersize=3, label="reference")
        plt.plot(d["k"], d[f"{base}_hel"], linestyle="--", linewidth=1.0, label="sum of helical channels")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel(r"$k$")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{stem}_{suffix}.png", dpi=220)
        plt.close()


def make_channel_piH_plot(ch_df, outdir: Path, stem: str, kplot_max=None):
    d = ch_df if kplot_max is None else ch_df[ch_df["k"] <= kplot_max]
    plt.figure(figsize=(7.2, 4.8))
    for ch, g in d.groupby("channel"):
        plt.plot(g["k"], g["Pi_H"], linewidth=1.0, label=ch)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\Pi_H^{r;pq}(k)$")
    plt.title("Helicity flux by helical channel")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"{stem}_helical_channel_Pi_H.png", dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Helical-channel RHS reconstruction benchmark.")
    parser.add_argument("--input", required=True, help="Input velocity .npz file")
    parser.add_argument("--outdir", default="results_jhtdb_helical")
    parser.add_argument("--mode", choices=["galerkin23"], default="galerkin23")
    parser.add_argument("--kplot-max", type=int, default=32)
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

    kx, ky, kz = fftfreq_wavenumbers(N, L)
    active_mask = mask_23(kx, ky, kz, N)

    uhat = np.fft.fftn(u, axes=(1, 2, 3))
    uhat = project_incompressible(uhat, kx, ky, kz)
    uhat[:, 0, 0, 0] = 0.0
    uhat *= active_mask[None, :, :, :]

    nhat_ref = pseudo_spectral_rhs_rotational(uhat, kx, ky, kz)
    nhat_ref *= active_mask[None, :, :, :]

    hplus, hminus = helical_basis(kx, ky, kz)
    hdict = {"+": hplus, "-": hminus}

    uhat_s = {s: helical_project(uhat, hdict[s]) for s in ["+", "-"]}
    omega_hat_s = {s: curl_hat(uhat_s[s], kx, ky, kz) for s in ["+", "-"]}

    recon_error = rel_l2(uhat_s["+"] + uhat_s["-"], uhat)

    nhat_hel_total = np.zeros_like(nhat_ref)
    channel_frames = []
    for r, p, q in itertools.product(["+", "-"], repeat=3):
        u_p = np.fft.ifftn(uhat_s[p], axes=(1, 2, 3))
        om_q = np.fft.ifftn(omega_hat_s[q], axes=(1, 2, 3))
        n_x = u_p[1]*om_q[2] - u_p[2]*om_q[1]
        n_y = u_p[2]*om_q[0] - u_p[0]*om_q[2]
        n_z = u_p[0]*om_q[1] - u_p[1]*om_q[0]
        nhat_pq = np.fft.fftn(np.array([n_x, n_y, n_z]), axes=(1, 2, 3))
        nhat_pq = project_incompressible(nhat_pq, kx, ky, kz)
        nhat_pq *= active_mask[None, :, :, :]
        nhat_rpq = helical_project(nhat_pq, hdict[r])
        nhat_rpq *= active_mask[None, :, :, :]
        nhat_hel_total += nhat_rpq
        channel_frames.append(shell_transfer_for_channel(
            uhat_s[r], omega_hat_s[r], nhat_rpq, kx, ky, kz, active_mask, r, p, q
        ))

    rhs_rel = rel_l2(nhat_hel_total, nhat_ref)

    df_ref = shell_df_for_rhs(uhat, nhat_ref, kx, ky, kz, active_mask)
    df_hel = shell_df_for_rhs(uhat, nhat_hel_total, kx, ky, kz, active_mask)
    df_compare = pd.DataFrame({"k": df_ref["k"]})
    for col in ["E", "H", "T_E", "T_H", "Pi_E", "Pi_H"]:
        df_compare[f"{col}_ref"] = df_ref[col]
        df_compare[f"{col}_hel"] = df_hel[col]
        df_compare[f"{col}_abs_err"] = df_hel[col] - df_ref[col]

    ch_df = pd.concat(channel_frames, ignore_index=True)

    stem = f"{inp.stem}_helical_{args.mode}"
    compare_csv = outdir / f"{stem}_total_compare.csv"
    channel_csv = outdir / f"{stem}_channel_transfers.csv"
    diag_txt = outdir / f"{stem}_diagnostics.txt"
    df_compare.to_csv(compare_csv, index=False)
    ch_df.to_csv(channel_csv, index=False)

    # Compare transfer errors.
    eps_TE = float(np.linalg.norm((df_compare["T_E_hel"] - df_compare["T_E_ref"]).to_numpy()) /
                   (np.linalg.norm(df_compare["T_E_ref"].to_numpy()) + 1e-300))
    eps_TH = float(np.linalg.norm((df_compare["T_H_hel"] - df_compare["T_H_ref"]).to_numpy()) /
                   (np.linalg.norm(df_compare["T_H_ref"].to_numpy()) + 1e-300))
    eps_PiE = float(np.linalg.norm((df_compare["Pi_E_hel"] - df_compare["Pi_E_ref"]).to_numpy()) /
                    (np.linalg.norm(df_compare["Pi_E_ref"].to_numpy()) + 1e-300))
    eps_PiH = float(np.linalg.norm((df_compare["Pi_H_hel"] - df_compare["Pi_H_ref"]).to_numpy()) /
                    (np.linalg.norm(df_compare["Pi_H_ref"].to_numpy()) + 1e-300))

    # Channel closure: sum channel transfers vs total helical transfer.
    ch_sum = ch_df.groupby("k")[["T_E", "T_H"]].sum().reset_index()
    ch_sum["Pi_E"] = -ch_sum["T_E"].cumsum()
    ch_sum["Pi_H"] = -ch_sum["T_H"].cumsum()
    ch_TE_err = float(np.linalg.norm((ch_sum["T_E"].to_numpy() - df_ref["T_E"].to_numpy())) /
                      (np.linalg.norm(df_ref["T_E"].to_numpy()) + 1e-300))
    ch_TH_err = float(np.linalg.norm((ch_sum["T_H"].to_numpy() - df_ref["T_H"].to_numpy())) /
                      (np.linalg.norm(df_ref["T_H"].to_numpy()) + 1e-300))

    with open(diag_txt, "w") as f:
        f.write(f"input: {inp}\n")
        f.write(f"N: {N}\n")
        f.write("mode: galerkin23\n")
        f.write(f"helical_velocity_reconstruction_rel_l2: {recon_error:.16e}\n")
        f.write(f"rhs_helical_sum_rel_l2: {rhs_rel:.16e}\n")
        f.write(f"T_E_rel_l2: {eps_TE:.16e}\n")
        f.write(f"T_H_rel_l2: {eps_TH:.16e}\n")
        f.write(f"Pi_E_rel_l2: {eps_PiE:.16e}\n")
        f.write(f"Pi_H_rel_l2: {eps_PiH:.16e}\n")
        f.write(f"channel_sum_T_E_rel_l2: {ch_TE_err:.16e}\n")
        f.write(f"channel_sum_T_H_rel_l2: {ch_TH_err:.16e}\n")

    make_compare_plots(df_compare, outdir, stem, kplot_max=args.kplot_max)
    make_channel_piH_plot(ch_df, outdir, stem, kplot_max=args.kplot_max)

    print(f"wrote {compare_csv}")
    print(f"wrote {channel_csv}")
    print(f"wrote {diag_txt}")
    print(f"helical_velocity_reconstruction_rel_l2={recon_error:.6e}")
    print(f"rhs_helical_sum_rel_l2={rhs_rel:.6e}")
    print(f"T_E_rel_l2={eps_TE:.6e}")
    print(f"T_H_rel_l2={eps_TH:.6e}")
    print(f"Pi_E_rel_l2={eps_PiE:.6e}")
    print(f"Pi_H_rel_l2={eps_PiH:.6e}")


if __name__ == "__main__":
    main()
