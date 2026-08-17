from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the deterministic-ensemble figure and compact summary.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    active = df.loc[df["status"] == "active"].copy()
    helicity_order = [
        "(+1,+1,+1)", "(+1,+1,-1)", "(+1,-1,+1)", "(+1,-1,-1)",
        "(-1,+1,+1)", "(-1,+1,-1)", "(-1,-1,+1)", "(-1,-1,-1)",
    ]

    eps = active["single_g_res_rel"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)

    axes[0].hist(eps, bins=np.linspace(0.0, 1.0, 31), color="#3B6EA8", edgecolor="white")
    axes[0].axvline(np.median(eps), color="#B33A3A", linestyle="--", linewidth=1.7,
                    label=rf"median $={np.median(eps):.3f}$")
    axes[0].set_xlabel(r"best-fit single-$g$ residual $\varepsilon_g$")
    axes[0].set_ylabel("active triads")
    axes[0].set_title("(a) Unified active ensemble")
    axes[0].legend(frameon=False)

    values = [
        active.loc[active["helicity_tuple"] == label, "single_g_res_rel"].to_numpy(float)
        for label in helicity_order
    ]
    box = axes[1].boxplot(values, showfliers=False, patch_artist=True, widths=0.62)
    for patch in box["boxes"]:
        patch.set_facecolor("#A8C5E5")
        patch.set_edgecolor("#2F4F6F")
    for median in box["medians"]:
        median.set_color("#B33A3A")
        median.set_linewidth(1.5)
    labels = [label.replace("+1", "+").replace("-1", "-") for label in helicity_order]
    axes[1].set_xticks(range(1, len(labels) + 1), labels, rotation=38, ha="right")
    axes[1].set_ylabel(r"$\varepsilon_g$")
    axes[1].set_title("(b) Geometry spread within helicity classes")
    axes[1].set_ylim(0.0, 1.05)

    fig.savefig(args.out, dpi=320, bbox_inches="tight")
    plt.close(fig)

    signature = active["gauge_metric_class"].value_counts().to_dict()
    rescaling = active["rescale_status"].value_counts().to_dict()
    rescaled = active.loc[active["rescale_status"] == "pass"].copy()
    created_modes = rescaled["rescale_created_mode"].value_counts().to_dict()
    lookup = {
        (int(row.geom_id), int(row.sK), int(row.sP), int(row.sQ)): row
        for row in active.itertuples(index=False)
    }
    seen = set()
    reversal_pairs = 0
    reversal_status_mismatches = 0
    reversal_created_mode_mismatches = 0
    max_reversal_scale_relative_difference = 0.0
    for key, row in lookup.items():
        opposite = (key[0], -key[1], -key[2], -key[3])
        pair_key = tuple(sorted((key, opposite)))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        other = lookup[opposite]
        reversal_pairs += 1
        reversal_status_mismatches += int(row.rescale_status != other.rescale_status)
        reversal_created_mode_mismatches += int(
            str(row.rescale_created_mode) != str(other.rescale_created_mode)
        )
        if row.rescale_status == "pass":
            r1 = np.array([row.rescale_r0, row.rescale_r1, row.rescale_r2], dtype=float)
            r2 = np.array([other.rescale_r0, other.rescale_r1, other.rescale_r2], dtype=float)
            rel = float(np.linalg.norm(r1 - r2) / (np.linalg.norm(r1) + 1e-300))
            max_reversal_scale_relative_difference = max(
                max_reversal_scale_relative_difference, rel
            )
    by_helicity = pd.crosstab(active["helicity_tuple"], active["gauge_metric_class"])
    summary = {
        "total_cases": int(len(df)),
        "active_cases": int(len(active)),
        "single_g_min": float(np.min(eps)),
        "single_g_median": float(np.median(eps)),
        "single_g_mean": float(np.mean(eps)),
        "single_g_max": float(np.max(eps)),
        "signature_counts": {str(k): int(v) for k, v in signature.items()},
        "positive_rescaling_counts": {str(k): int(v) for k, v in rescaling.items()},
        "rescaled_created_mode_counts": {str(k): int(v) for k, v in created_modes.items()},
        "max_rescaled_coefficient_residual_relative": float(rescaled["rescale_coefficient_rel"].max()),
        "max_rescaled_energy_weight_residual_relative": float(rescaled["rescale_energy_weight_rel"].max()),
        "max_rescaled_helicity_weight_residual_relative": float(rescaled["rescale_helicity_weight_rel"].max()),
        "rescaling_condition_median": float(rescaled["rescale_condition"].median()),
        "rescaling_condition_q99": float(rescaled["rescale_condition"].quantile(0.99)),
        "rescaling_condition_max": float(rescaled["rescale_condition"].max()),
        "global_helicity_reversal_pairs": int(reversal_pairs),
        "global_helicity_reversal_status_mismatches": int(reversal_status_mismatches),
        "global_helicity_reversal_created_mode_mismatches": int(reversal_created_mode_mismatches),
        "max_global_helicity_reversal_scale_relative_difference": float(
            max_reversal_scale_relative_difference
        ),
        "max_energy_residual_relative": float(active["energy_res_rel"].max()),
        "max_helicity_residual_relative": float(active["helicity_res_rel"].max()),
        "max_gauge_realness_residual": float(active["gauge_eta_im_rel"].max()),
        "by_helicity": {
            str(index): {str(k): int(v) for k, v in row.items()}
            for index, row in by_helicity.iterrows()
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
