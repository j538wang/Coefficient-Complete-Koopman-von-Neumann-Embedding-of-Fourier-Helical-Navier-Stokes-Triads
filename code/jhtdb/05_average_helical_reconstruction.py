#!/usr/bin/env python3
"""Average helical reconstruction total-compare CSV files."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="*_total_compare.csv files")
    parser.add_argument("--outdir", default="results_jhtdb_helical_average")
    parser.add_argument("--kplot-max", type=int, default=32)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for p in args.inputs:
        df = pd.read_csv(p)
        df["source"] = Path(p).stem
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    numeric_cols = [c for c in all_df.columns if c not in {"source"} and c != "k"]
    avg = all_df.groupby("k")[numeric_cols].agg(["mean", "std"])
    avg.columns = [f"{c}_{stat}" for c, stat in avg.columns]
    avg = avg.reset_index()
    out_csv = outdir / "time_averaged_helical_total_compare.csv"
    avg.to_csv(out_csv, index=False)

    # Diagnostics on averaged curves.
    for base in ["T_E", "T_H", "Pi_E", "Pi_H"]:
        ref = avg[f"{base}_ref_mean"].to_numpy()
        hel = avg[f"{base}_hel_mean"].to_numpy()
        rel = np.linalg.norm(hel - ref) / (np.linalg.norm(ref) + 1e-300)
        print(f"{base}_mean_rel_l2 = {rel:.6e}")

    d = avg if args.kplot_max is None else avg[avg["k"] <= args.kplot_max]
    plot_specs = [
        ("T_E", r"$\langle T_E(k)\rangle$", "Time-averaged shell energy transfer", "avg_compare_T_E"),
        ("T_H", r"$\langle T_H(k)\rangle$", "Time-averaged shell helicity transfer", "avg_compare_T_H"),
        ("Pi_E", r"$\langle \Pi_E(k)\rangle$", "Time-averaged cumulative energy flux", "avg_compare_Pi_E"),
        ("Pi_H", r"$\langle \Pi_H(k)\rangle$", "Time-averaged cumulative helicity flux", "avg_compare_Pi_H"),
    ]
    for base, ylabel, title, suffix in plot_specs:
        plt.figure(figsize=(6.5, 4.2))
        k = d["k"]
        ref = d[f"{base}_ref_mean"]
        hel = d[f"{base}_hel_mean"]
        ref_std = d.get(f"{base}_ref_std", pd.Series(np.zeros_like(ref), index=d.index)).fillna(0.0)
        plt.plot(k, ref, marker="o", linewidth=1.0, markersize=3, label="reference")
        plt.fill_between(k, ref-ref_std, ref+ref_std, alpha=0.20)
        plt.plot(k, hel, linestyle="--", linewidth=1.0, label="helical-channel sum")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel(r"$k$")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{suffix}.png", dpi=220)
        plt.close()

    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
