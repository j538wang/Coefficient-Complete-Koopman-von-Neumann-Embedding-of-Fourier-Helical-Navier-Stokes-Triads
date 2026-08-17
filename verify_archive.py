from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def close(actual: float, expected: float, *, rel: float = 5e-10, abs_: float = 1e-18) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=abs_):
        raise AssertionError(f"value mismatch: {actual!r} != {expected!r}")


def parse_diagnostics(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def verify_manifest() -> int:
    manifest = ROOT / "MANIFEST.csv"
    if not manifest.exists():
        return 0
    checked = 0
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            path = ROOT / row["path"]
            if path.stat().st_size != int(row["bytes"]):
                raise AssertionError(f"size mismatch for {row['path']}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                raise AssertionError(f"SHA-256 mismatch for {row['path']}")
            checked += 1
    return checked


def main() -> None:
    ensemble = load_json("processed/unified_ensemble/compact_summary.json")
    assert ensemble["total_cases"] == 6400
    assert ensemble["active_cases"] == 6388
    assert ensemble["signature_counts"] == {"indefinite": 6028, "singular": 360}
    close(ensemble["single_g_median"], 0.6071198957111068)

    scaling = load_json("processed/global_scaling_audit/global_compatibility_summary.json")
    assert scaling["global_nodes"] == 252
    assert scaling["global_constraints"] == 12056
    assert scaling["least_squares"]["rank"] == 251
    close(scaling["least_squares"]["residual_relative_l2"], 0.9124147864499432)
    assert scaling["parallel_pair_audit"]["inconsistent_parallel_node_pairs"] == 2144

    kvn = load_json("processed/kvn_global_graph_audit/global_kvn_summary.json")
    assert kvn["real_kvn_coordinate_count"] == 504
    assert kvn["sparse_quadratic_terms_after_combination"] == 74112
    close(kvn["verification"]["maximum_global_reconstruction_relative_error"], 9.352193173421239e-16)
    close(kvn["verification"]["energy_commutator_polynomial"]["max_abs"], 3.552713678800501e-15)
    close(kvn["verification"]["helicity_commutator_polynomial"]["max_abs"], 1.7763568394002505e-14)

    gate = load_json("processed/kvn_two_triad_gate_validation/gate_validation_summary.json")
    assert gate["representation"]["tested_real_invariant_coordinates"] == 4
    assert gate["gate_compilation"]["non_gaussian_cubic_phase_primitives_in_one_strang_step"] == 44
    close(gate["trotter_observed_order_from_last_three_points"], 2.0043358056518863)
    assert gate["overall_gate_level_validation_success"] is True

    resource = load_json("processed/full_spectral_resource_scaling/resource_scaling_summary.json")
    rows = {row["spectral_cutoff_K"]: row for row in resource["exact_rows"]}
    assert rows[1]["exact_noncollinear_unordered_triad_geometries"] == 44
    assert rows[3]["exact_noncollinear_unordered_triad_geometries"] == 8232
    assert rows[6]["exact_noncollinear_unordered_triad_geometries"] == 339900

    tgv = load_json("processed/tgv/shell_transfer/shell_helicity_summary.json")
    assert tgv["N"] == 128 and tgv["retained_modes"] == 5000
    close(tgv["time"], 8.97)
    close(tgv["global_rhs_relative_error"], 7.194479321355066e-15)
    close(tgv["max_modewise_rhs_relative_error"], 6.051588712243716e-13)

    truncation = load_json("processed/tgv/active_triad_truncation/active_triad_truncation_summary.json")
    assert truncation["input_modes_used"] == 1200
    assert truncation["helical_amplitude_contributions_computed"] == 11494656
    close(truncation["raw_top_10_percent_fraction"], 0.8662073406499353)

    diagnostics = sorted((ROOT / "processed/jhtdb/per_snapshot").glob("*diagnostics.txt"))
    assert len(diagnostics) == 8
    rhs_mean = sum(float(parse_diagnostics(path)["rhs_helical_sum_rel_l2"]) for path in diagnostics) / 8
    close(rhs_mean, 8.39379695125166e-16)

    count = verify_manifest()
    print(f"Archive verification passed. Manifest files checked: {count}.")


if __name__ == "__main__":
    main()
