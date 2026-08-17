#!/usr/bin/env python3
"""
Active helical triad contribution and truncation diagnostic.

This script quantifies how the reconstructed nonlinear RHS is distributed among
individual helical amplitude contributions.  It answers questions such as:
  * What fraction of helical contributions accounts for most of the raw
    contribution magnitude?
  * How accurately can the RHS be reconstructed if only the largest helical
    contributions are retained?

The truncation order is by |dot{u}_s(k)| for each helical contribution
(k,p,q,s_k,s_p,s_q).  Since nonlinear RHS contributions can cancel by phase, the
raw cumulative contribution curve and the actual truncated-RHS error curve are
both reported.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fluid_helical_utils import (
    load_modes_npz, save_json, v_neg, helical_basis, relative_error,
    cartesian_pair_rhs, helical_pair_rhs, dict_l2_norm, modes_energy,
)


def subset_modes(modes, max_modes: int):
    if max_modes is None or max_modes <= 0 or len(modes) <= max_modes:
        return modes
    entries = sorted(modes.items(), key=lambda kv: np.linalg.norm(kv[1]), reverse=True)
    out = {}
    for k,u in entries:
        if len(out) >= max_modes:
            break
        out[k] = u
        nk = v_neg(k)
        if nk in modes and len(out) < max_modes:
            out[nk] = modes[nk]
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modes", type=str, required=True, help="Sparse modes npz.")
    p.add_argument("--out", type=str, default="results_active_triad_truncation")
    p.add_argument("--max-input-modes", type=int, default=700, help="Subset to this many largest modes for tractability; <=0 means no subset.")
    p.add_argument("--contrib-rel-tol", type=float, default=1e-14, help="Contribution tolerance relative to max |u|^2, used to skip tiny records.")
    p.add_argument("--fractions", type=str, default="0.001,0.005,0.01,0.02,0.05,0.1,0.25,0.5,1.0")
    p.add_argument("--top-csv", type=int, default=50000, help="Save this many largest helical contributions to CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes, metadata = load_modes_npz(args.modes)
    modes = subset_modes(modes, args.max_input_modes)
    print(f"Using modes: {len(modes)}")
    max_u = max(np.linalg.norm(u) for u in modes.values()) if modes else 0.0
    contrib_tol = args.contrib_rel_tol * max(max_u * max_u, 1e-300)

    rhs_ref, n_pairs = cartesian_pair_rhs(modes)
    rhs_full, n_hel, records = helical_pair_rhs(modes, store_contributions=True, contrib_tol=contrib_tol)
    full_err = relative_error(rhs_full, rhs_ref)
    print(f"Stored helical contribution records: {len(records)}")
    print(f"Full helical vs Cartesian RHS error: {full_err:.3e}")

    if not records:
        raise RuntimeError("No helical contributions stored. Lower --contrib-rel-tol.")

    # Sort records by contribution magnitude descending.
    records.sort(key=lambda r: r[7], reverse=True)
    mags = np.asarray([r[7] for r in records], dtype=float)
    mag2 = mags * mags
    total_mag2 = float(np.sum(mag2))
    cum_raw = np.cumsum(mag2) / (total_mag2 + 1e-300)
    Ntot = len(records)

    # Raw contribution concentration metrics.
    targets = [0.90, 0.95, 0.99]
    target_rows = []
    for targ in targets:
        idx = int(np.searchsorted(cum_raw, targ, side="left")) + 1
        target_rows.append({"target_raw_fraction": targ, "M_required": idx, "retained_fraction": idx / Ntot})
    pd.DataFrame(target_rows).to_csv(out / "raw_contribution_targets.csv", index=False)

    fractions = sorted(set(float(x) for x in args.fractions.split(",") if x.strip()))
    trunc_rows = []
    zvec = np.zeros(3, dtype=complex)
    norm_ref = dict_l2_norm(rhs_ref)
    for frac in fractions:
        M = max(1, int(math.ceil(frac * Ntot)))
        rhsM = {}
        for rec in records[:M]:
            k, p, q, sk, sp, sq, dotus, mag = rec
            vec = dotus * helical_basis(k, sk)
            rhsM[k] = rhsM.get(k, zvec.copy()) + vec
        err_vs_ref = relative_error(rhsM, rhs_ref)
        err_vs_full = relative_error(rhsM, rhs_full)
        trunc_rows.append({
            "retained_fraction": frac,
            "M_retained": M,
            "raw_contribution_fraction": float(cum_raw[M-1]),
            "truncated_rhs_error_vs_reference": err_vs_ref,
            "truncated_rhs_error_vs_full_helical": err_vs_full,
        })
        print(f"frac={frac:.3g}, M={M}, raw={cum_raw[M-1]:.4f}, err={err_vs_ref:.3e}")
    df_trunc = pd.DataFrame(trunc_rows)
    df_trunc.to_csv(out / "truncation_error_curve.csv", index=False)

    # Cumulative curve downsampled for CSV/plot.
    sample_idx = np.unique(np.concatenate([
        np.linspace(0, Ntot-1, min(2000, Ntot), dtype=int),
        np.asarray([max(0, int(f*Ntot)-1) for f in fractions], dtype=int)
    ]))
    df_cum = pd.DataFrame({
        "M": sample_idx + 1,
        "retained_fraction": (sample_idx + 1) / Ntot,
        "raw_contribution_fraction": cum_raw[sample_idx],
    })
    df_cum.to_csv(out / "cumulative_contribution_curve.csv", index=False)

    # Save top contribution records only.
    top = records[:min(args.top_csv, Ntot)]
    top_rows = []
    for rank, rec in enumerate(top, start=1):
        k,p,q,sk,sp,sq,dotus,mag = rec
        top_rows.append({
            "rank": rank, "k": str(k), "p": str(p), "q": str(q),
            "s_k": sk, "s_p": sp, "s_q": sq,
            "dotus_real": np.real(dotus), "dotus_imag": np.imag(dotus),
            "contribution_abs": mag,
            "contribution_abs2_fraction": (mag*mag)/(total_mag2+1e-300),
        })
    pd.DataFrame(top_rows).to_csv(out / "top_helical_contributions.csv", index=False)

    summary = {
        **metadata,
        "input_modes_used": len(modes),
        "input_energy_used": modes_energy(modes),
        "cartesian_pair_contributions": n_pairs,
        "helical_amplitude_contributions_computed": n_hel,
        "helical_contributions_stored_after_tol": Ntot,
        "contrib_rel_tol": args.contrib_rel_tol,
        "full_helical_rhs_error_vs_reference": full_err,
        "raw_top_1_percent_fraction": float(cum_raw[max(0, int(0.01*Ntot)-1)]),
        "raw_top_5_percent_fraction": float(cum_raw[max(0, int(0.05*Ntot)-1)]),
        "raw_top_10_percent_fraction": float(cum_raw[max(0, int(0.10*Ntot)-1)]),
    }
    for row in target_rows:
        summary[f"M_for_{int(row['target_raw_fraction']*100)}pct_raw"] = int(row["M_required"])
        summary[f"fraction_for_{int(row['target_raw_fraction']*100)}pct_raw"] = float(row["retained_fraction"])
    save_json(out / "active_triad_truncation_summary.json", summary)

    # Figures.
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.semilogx(df_cum["retained_fraction"], df_cum["raw_contribution_fraction"], linestyle="-", linewidth=2)
    ax.set_xlabel("retained helical contribution fraction")
    ax.set_ylabel("cumulative raw contribution fraction")
    ax.set_title("Concentration of helical triad contributions")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "triad_cumulative_contribution.png", dpi=230)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.loglog(df_trunc["retained_fraction"], df_trunc["truncated_rhs_error_vs_reference"], marker="o", linestyle="-", label="error vs Cartesian reference")
    ax.loglog(df_trunc["retained_fraction"], df_trunc["truncated_rhs_error_vs_full_helical"], marker="s", linestyle="--", label="error vs full helical RHS")
    ax.set_xlabel("retained helical contribution fraction")
    ax.set_ylabel("relative truncated-RHS error")
    ax.set_title("RHS error under helical-triad truncation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out / "truncated_rhs_error_vs_retained_triads.png", dpi=230)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.hist(mags, bins=80, log=True, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(r"helical contribution magnitude $|\dot u_s^{(T)}(k)|$")
    ax.set_ylabel("count")
    ax.set_title("Distribution of individual helical triad contributions")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "triad_contribution_histogram.png", dpi=230)
    plt.close(fig)

    print(f"Saved outputs to: {out}")


if __name__ == "__main__":
    main()
