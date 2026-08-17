#!/usr/bin/env python3
"""Average per-snapshot spectral transfer CSV files."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--outdir", default="results_jhtdb_average")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for p in args.inputs:
        df = pd.read_csv(p)
        df["source"] = Path(p).stem
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    cols = ["E", "H", "T_E", "T_H", "Pi_E", "Pi_H"]
    avg = all_df.groupby("k")[cols].agg(["mean", "std"])
    avg.columns = [f"{c}_{stat}" for c, stat in avg.columns]
    avg = avg.reset_index()
    avg.to_csv(outdir / "time_averaged_spectral_transfers.csv", index=False)

    for quantity, ylabel in [("Pi_H", r"$\langle \Pi_H(k_c)\rangle_T$"), ("T_H", r"$\langle T_H(k)\rangle_T$")]:
        plt.figure(figsize=(6.5,4.2))
        k = avg["k"]
        mean = avg[f"{quantity}_mean"]
        std = avg[f"{quantity}_std"].fillna(0.0)
        plt.plot(k, mean, marker="o", linewidth=1.0, markersize=3)
        plt.fill_between(k, mean-std, mean+std, alpha=0.25)
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel(r"$k$")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(outdir / f"time_averaged_{quantity}.png", dpi=220)
        plt.close()

    print(f"wrote {outdir / 'time_averaged_spectral_transfers.csv'}")

if __name__ == "__main__":
    main()
