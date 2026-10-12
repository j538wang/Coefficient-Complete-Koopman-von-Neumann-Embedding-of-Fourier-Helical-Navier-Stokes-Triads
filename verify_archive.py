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


def load_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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

    dedup = load_json("processed/full_kmax3_unique/deduplication_summary.json")
    assert dedup["source_created_mode_geometries"] == 12348
    assert dedup["created_mode_orientations_per_reality_reduced_class"] == 3
    assert dedup["unique_reality_reduced_closed_triad_geometries"] == 4116
    assert dedup["retained_helicity_resolved_rows"] == 32928
    assert dedup["status_counts"] == {"active": 32880, "inactive": 48}
    assert dedup["positive_rescaling_status_counts"] == {
        "pass": 30000,
        "singular": 2880,
        "inactive": 48,
    }

    full_scaling = load_json("processed/full_kmax3_global_scaling_audit/global_compatibility_summary.json")
    assert full_scaling["total_input_rows"] == 32928
    assert full_scaling["local_pass_rows"] == 30000
    assert full_scaling["global_nodes"] == 342
    assert full_scaling["global_constraints"] == 60000
    assert full_scaling["least_squares"]["rank"] == 341
    close(full_scaling["least_squares"]["residual_relative_l2"], 0.9613814168073568)
    assert full_scaling["spanning_cycle_audit"]["cycle_closures_above_tolerance"] == 59402
    assert full_scaling["parallel_pair_audit"]["inconsistent_parallel_node_pairs"] == 18168
    assert full_scaling["globally_compatible"] is False

    full_kvn = load_json("processed/full_kmax3_kvn_graph_audit/global_kvn_summary.json")
    assert full_kvn["source_total_rows"] == 32928
    assert full_kvn["source_active_triads"] == 32880
    assert full_kvn["shared_complex_fourier_helicity_nodes"] == 342
    assert full_kvn["real_kvn_coordinate_count"] == 684
    assert full_kvn["sparse_quadratic_terms_after_combination"] == 734176
    assert full_kvn["full_lattice_equivalent_helicity_triads"] == 65856
    close(full_kvn["verification"]["maximum_global_reconstruction_relative_error"], 1.2920964810000398e-15)
    close(full_kvn["verification"]["energy_commutator_polynomial"]["max_abs"], 1.1587952819525071e-15)
    close(full_kvn["verification"]["helicity_commutator_polynomial"]["max_abs"], 5.329070518200751e-15)

    local_bplus = load_json("processed/kvn_two_triad_gate_validation_bplus/gate_validation_summary.json")
    assert local_bplus["representation"]["gaussian_conjugations_instantiated"] is True
    assert local_bplus["overall_local_product_cutoff_stress_audit_success"] is True
    assert local_bplus["diagnostic_thresholds"]["selected_cutoff_10_eight_step_compilation_residual_below_1e-3"] is False

    global_bplus = load_json("processed/kvn_two_triad_gate_validation_bplus/global_fock_gaussian_summary.json")
    assert global_bplus["basis"]["selected_total_number_cutoff"] == 16
    assert global_bplus["basis"]["selected_hilbert_dimension"] == 4845
    assert global_bplus["basis"]["passive_maps_preserve_truncated_space"] is True
    assert global_bplus["overall_restricted_gaussian_compilation_success"] is True

    cutoff_rows = {int(row["total_number_cutoff"]): row for row in load_csv(
        "processed/kvn_two_triad_gate_validation_bplus/global_fock_gaussian_cutoff_scan.csv"
    )}
    close(cutoff_rows[16]["maximum_gaussian_map_action_relative_residual"], 3.175706191044993e-15)
    close(cutoff_rows[16]["physical_to_collective_one_step_state_l2_residual"], 2.8279095998277067e-14)
    close(cutoff_rows[16]["maximum_intermediate_total_boundary_probability"], 6.834695132964088e-05)

    strang_rows = {int(row["strang_steps"]): row for row in load_csv(
        "processed/kvn_two_triad_gate_validation_bplus/global_fock_gaussian_strang_scan.csv"
    )}
    assert int(strang_rows[1]["single_mode_cubic_phase_primitive_count"]) == 44
    assert int(strang_rows[8]["single_mode_cubic_phase_primitive_count"]) == 324
    assert int(strang_rows[8]["gaussian_phase_rotation_count"]) == 648
    assert int(strang_rows[8]["gaussian_beam_splitter_rotation_count"]) == 1296
    close(strang_rows[8]["physical_to_collective_state_l2_residual"], 2.010669495086239e-13)
    close(strang_rows[8]["maximum_abs_single_mode_cubic_strength"], 0.0031751043335554855)
    close(strang_rows[8]["total_abs_single_mode_cubic_strength"], 0.40641335469510215)

    full8_manifest = load_json("reproducibility/eight_coordinate_cutoff/run_manifest.json")
    assert full8_manifest["completed_cutoffs"] == [6, 8, 10, 12, 14, 16]
    assert full8_manifest["monotonic_maximum_boundary_probability"] is True
    assert full8_manifest["monotonic_final_mean_error_vs_classical"] is True
    assert full8_manifest["reference_cutoff_for_direct_trajectory_comparison"] == 16
    assert "same untruncated eight-mode product coherent state" in full8_manifest["initial_state_construction"]

    full8_n14 = load_json("reproducibility/eight_coordinate_cutoff/N014_summary.json")
    full8_n16 = load_json("reproducibility/eight_coordinate_cutoff/N016_summary.json")
    assert full8_n14["hilbert_dimension"] == 319770
    assert full8_n16["hilbert_dimension"] == 735471
    assert full8_n16["initial_state_projected_and_renormalized_separately_for_each_cutoff"] is True
    close(full8_n14["maximum_boundary_probability_last_two_shells"], 1.5119194786027343e-4)
    close(full8_n16["maximum_boundary_probability_last_two_shells"], 1.0934690865451348e-5)
    close(full8_n16["initial_projection_retained_probability"], 0.9999997158320072)

    full8_convergence = {int(row["total_number_cutoff"]): row for row in load_csv(
        "reproducibility/eight_coordinate_cutoff/cutoff_convergence.csv"
    )}
    close(full8_convergence[14]["relative_mean_trajectory_difference_vs_reference"], 2.2051809763838902e-5)
    close(full8_convergence[14]["final_relative_covariance_frobenius_difference_vs_reference"], 2.5170266494757776e-4)

    initial_diagnostics = {int(row["total_number_cutoff"]): row for row in load_csv(
        "reproducibility/eight_coordinate_cutoff/initial_projection_diagnostics.csv"
    )}
    close(initial_diagnostics[6]["untruncated_initial_mean_total_occupation"], 3.609375)
    close(initial_diagnostics[6]["initial_relative_first_moment_projection_bias"], 8.977191540395599e-2)
    close(initial_diagnostics[16]["initial_relative_first_moment_projection_bias"], 1.0733984863836187e-6)
    close(
        initial_diagnostics[6]["initial_normalized_covariance_trace_convention_absolute_difference"],
        1.005445452524307e-1,
    )
    close(
        initial_diagnostics[16]["initial_normalized_covariance_trace_convention_absolute_difference"],
        2.060925093856548e-6,
    )
    initial_diagnostics_json = load_json(
        "reproducibility/eight_coordinate_cutoff/initial_projection_diagnostics.json"
    )
    assert initial_diagnostics_json["status"] == "PASS"
    assert initial_diagnostics_json["covariance_comparison_scope"] == "initial normalized covariance trace only"

    projection = load_json("reproducibility/eight_coordinate_cutoff/projection_verification.json")
    assert projection["status"] == "PASS"
    assert projection["tested_monomials"] == 24
    close(projection["maximum_complete_product_relative_difference"], 0.0)
    close(projection["maximum_factor_by_factor_projected_relative_difference"], 0.5773502691896258)

    resource = load_json("processed/full_spectral_resource_scaling/resource_scaling_summary.json")
    rows = {row["spectral_cutoff_K"]: row for row in resource["exact_rows"]}
    assert rows[1]["exact_noncollinear_unordered_triad_geometries"] == 44
    assert rows[3]["exact_noncollinear_unordered_triad_geometries"] == 8232
    assert rows[6]["exact_noncollinear_unordered_triad_geometries"] == 339900

    resource_bplus = load_json("processed/full_spectral_resource_scaling_bplus/resource_scaling_summary.json")
    close(resource_bplus["global_audit_calibration"]["observed_terms_per_full_lattice_helical_triad"], 11.148202137998057)
    resource_bplus_rows = {row["spectral_cutoff_K"]: row for row in resource_bplus["exact_rows"]}
    assert resource_bplus_rows[3]["direct_first_order_non_gaussian_cubic_phase_primitives_empirical_estimate"] == 2936704
    close(resource_bplus_rows[3]["empirical_direct_to_structured_cubic_primitive_ratio"], 122.36266666666667)

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
