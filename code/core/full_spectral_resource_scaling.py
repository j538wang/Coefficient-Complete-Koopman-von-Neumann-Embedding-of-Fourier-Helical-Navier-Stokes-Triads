#!/usr/bin/env python3
"""Resource audit for coefficient-complete cubic KvN Navier--Stokes circuits.

The audit separates two constructions.

1. Direct spectral hypergraph: enumerate every non-collinear unordered wavevector
   triad in the cube [-K,K]^3, attach all eight helicity assignments, and bound
   the number of real cubic KvN monomials.
2. Structured pseudospectral construction: use invertible Fourier/helicity
   Gaussian transforms plus reversible work modes for curl and projection, then
   evaluate the rotational-form nonlinearity locally as six signed products per
   physical grid point.

The second construction is a resource proposal, not a completed full-solver
gate validation. Its local non-Gaussian cubic-phase primitive count is exact
under the stated ideal work-register architecture. Gaussian conjugations,
finite squeezing, work-mode preparation/reset, transform precision,
error-correction, viscosity, state preparation, and readout remain explicit
qualifications.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Vector = Tuple[int, int, int]


def vector_modes(cutoff: int) -> List[Vector]:
    return [
        (i, j, k)
        for i in range(-cutoff, cutoff + 1)
        for j in range(-cutoff, cutoff + 1)
        for k in range(-cutoff, cutoff + 1)
        if (i, j, k) != (0, 0, 0)
    ]


def noncollinear(a: Vector, b: Vector) -> bool:
    return (
        a[1] * b[2] - a[2] * b[1] != 0
        or a[2] * b[0] - a[0] * b[2] != 0
        or a[0] * b[1] - a[1] * b[0] != 0
    )


def count_unordered_triad_geometries(cutoff: int) -> int:
    """Count {k,p,q}, k+p+q=0, once, excluding collinear triples."""
    modes = vector_modes(cutoff)
    mode_set = set(modes)
    geometries = set()
    for left_index, left in enumerate(modes):
        for right in modes[left_index:]:
            third = (
                -left[0] - right[0],
                -left[1] - right[1],
                -left[2] - right[2],
            )
            if third == (0, 0, 0) or third not in mode_set:
                continue
            if not noncollinear(left, right):
                continue
            geometries.add(tuple(sorted((left, right, third))))
    return len(geometries)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cutoff", type=int, default=6)
    parser.add_argument(
        "--kvn-summary",
        type=Path,
        default=Path("kvn_global_graph_audit/global_kvn_summary.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("full_spectral_resource_scaling"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with args.kvn_summary.open(encoding="utf-8") as handle:
        sampled_audit = json.load(handle)
    terms_per_active_helical_triad = (
        sampled_audit["sparse_quadratic_terms_after_combination"]
        / sampled_audit["source_active_triads"]
    )

    rows: List[Dict[str, object]] = []
    for cutoff in range(1, args.max_cutoff + 1):
        retained_side = 2 * cutoff + 1
        grid_points = retained_side**3
        nonzero_wavevectors = grid_points - 1
        dealiased_side = 3 * cutoff + 1
        dealiased_grid_points = dealiased_side**3
        triad_geometries = count_unordered_triad_geometries(cutoff)
        helical_triads = 8 * triad_geometries
        estimated_direct_monomials = terms_per_active_helical_triad * helical_triads
        upper_direct_monomials = 24 * helical_triads
        structured_monomials = 6 * dealiased_grid_points
        rows.append(
            {
                "spectral_cutoff_K": cutoff,
                "grid_points_G": grid_points,
                "dealiased_grid_side_P": dealiased_side,
                "dealiased_grid_points_Gpad": dealiased_grid_points,
                "nonzero_wavevectors_W": nonzero_wavevectors,
                "independent_complex_helicity_amplitudes": nonzero_wavevectors,
                "real_kvn_coordinates_and_native_CV_data_modes": 2 * nonzero_wavevectors,
                "exact_noncollinear_unordered_triad_geometries": triad_geometries,
                "helicity_resolved_triads": helical_triads,
                "direct_real_cubic_monomials_empirical_estimate": int(
                    round(estimated_direct_monomials)
                ),
                "direct_real_cubic_monomials_rigorous_upper_bound": upper_direct_monomials,
                "direct_first_order_non_gaussian_cubic_phase_primitives_empirical_estimate": int(
                    round(4 * estimated_direct_monomials)
                ),
                "direct_first_order_non_gaussian_cubic_phase_primitives_upper_bound": 4
                * upper_direct_monomials,
                "structured_local_real_cubic_monomials": structured_monomials,
                "structured_first_order_non_gaussian_cubic_phase_primitives": 4 * structured_monomials,
                "structured_asymptotic_strang_non_gaussian_cubic_phase_primitives_per_step": 8
                * structured_monomials,
                "empirical_direct_to_structured_cubic_primitive_ratio": float(
                    4 * estimated_direct_monomials / (4 * structured_monomials)
                ),
            }
        )
        print(
            f"K={cutoff}: G={grid_points}, T={triad_geometries}, "
            f"direct/structured={rows[-1]['empirical_direct_to_structured_cubic_primitive_ratio']:.2f}",
            flush=True,
        )

    write_csv(args.out / "exact_cube_scaling.csv", rows)
    wavevector_counts = np.asarray([row["nonzero_wavevectors_W"] for row in rows], dtype=float)
    triad_counts = np.asarray(
        [row["exact_noncollinear_unordered_triad_geometries"] for row in rows], dtype=float
    )
    fit_slice = slice(max(0, len(rows) - 4), None)
    triad_power, triad_log_prefactor = np.polyfit(
        np.log(wavevector_counts[fit_slice]), np.log(triad_counts[fit_slice]), 1
    )

    extrapolated_rows: List[Dict[str, object]] = []
    for side in (16, 32, 64, 128):
        grid_points = side**3
        wavevectors = grid_points - 1
        dealiased_side = math.ceil(1.5 * side)
        dealiased_grid_points = dealiased_side**3
        fitted_triads = math.exp(triad_log_prefactor) * wavevectors**triad_power
        fitted_helical = 8 * fitted_triads
        direct_gate_estimate = 4 * terms_per_active_helical_triad * fitted_helical
        structured_primitive_count = 24 * dealiased_grid_points
        extrapolated_rows.append(
            {
                "nominal_grid_side": side,
                "grid_points_G": grid_points,
                "dealiased_grid_side": dealiased_side,
                "dealiased_grid_points_Gpad": dealiased_grid_points,
                "native_CV_data_modes_approximately": 2 * wavevectors,
                "fitted_unordered_triad_geometries": int(round(fitted_triads)),
                "fitted_direct_first_order_non_gaussian_cubic_phase_primitives": int(
                    round(direct_gate_estimate)
                ),
                "structured_first_order_non_gaussian_cubic_phase_primitives": structured_primitive_count,
                "fitted_direct_to_structured_ratio": float(
                    direct_gate_estimate / structured_primitive_count
                ),
                "logical_qubits_if_binary_cutoff_d16": 2 * wavevectors * 4,
            }
        )
    write_csv(args.out / "nominal_grid_extrapolation.csv", extrapolated_rows)

    summary = {
        "counting_definition": {
            "spectral_domain": "all nonzero integer wavevectors in [-K,K]^3",
            "reality_and_helicity": (
                "Reality halves the wavevector set and two helicities restore W "
                "independent complex amplitudes; splitting real and imaginary parts gives 2W real KvN coordinates."
            ),
            "triads": (
                "unordered noncollinear wavevector triples summing to zero, with all eight helicity assignments"
            ),
            "dealiasing": (
                "The structured real-space product uses P=3K+1 points per dimension, which is sufficient to "
                "prevent a quadratic product from aliasing back into the retained |k_i|<=K cube."
            ),
        },
        "sampled_global_audit_calibration": {
            "active_helical_triads": sampled_audit["source_active_triads"],
            "combined_real_quadratic_terms": sampled_audit[
                "sparse_quadratic_terms_after_combination"
            ],
            "observed_terms_per_active_helical_triad": terms_per_active_helical_triad,
            "rigorous_real_monomial_upper_bound_per_helical_triad": 24,
        },
        "fitted_triad_scaling": {
            "power_of_nonzero_wavevector_count": float(triad_power),
            "prefactor": float(math.exp(triad_log_prefactor)),
            "interpretation": "finite-cutoff fit confirming the expected quadratic spectral-hypergraph growth",
        },
        "direct_spectral_compilation": {
            "cubic_phase_primitives_per_real_kvn_monomial": 4,
            "first_order_count": "4 M_spec",
            "n_step_strang_count_after_adjacent_merging": "4 [2 n (M_spec-1)+1]",
            "asymptotic_non_gaussian_primitive_count_per_strang_step": "8 M_spec = O(G^2)",
            "parallel_depth_qualification": (
                "Disjoint hyperedges can be colored and executed together, but shared Fourier modes make the "
                "depth grow with hypergraph degree; the audit does not claim constant depth."
            ),
        },
        "structured_pseudospectral_proposal": {
            "nonlinearity": "projected rotational form P[u cross curl(u)]",
            "local_signed_products_per_grid_point": 6,
            "real_cubic_kvn_monomials": "6 G_pad on the dealiased grid",
            "first_order_non_gaussian_cubic_phase_primitives": "24 G_pad",
            "asymptotic_strang_non_gaussian_cubic_phase_primitives_per_step": "48 G_pad",
            "gaussian_transform_gate_count": "O(G_pad log G_pad) for padded Fourier/helicity transforms",
            "ideal_all_to_all_gaussian_depth": "O(log G), plus constant-depth local color classes",
            "work_modes": "O(G_pad) reversible work modes for physical velocity, curl, and uncomputation",
            "ideal_ancilla_assumption": (
                "Work modes are treated as ideal known quadrature-reference states and Gaussian shears as exact; "
                "finite squeezing and preparation/reset errors are excluded."
            ),
            "canonical_implementation": (
                "Invertible transforms act symplectically on q and p; curl and other non-invertible linear maps "
                "must be computed into work modes by reversible Gaussian shears; transverse projection is handled "
                "in the divergence-free Fourier/helicity basis."
            ),
            "status": (
                "resource architecture only; not yet validated as a complete noisy, viscous, fault-tolerant solver"
            ),
        },
        "efficiency_boundary": {
            "per_step_comparison": (
                "Structured quantum and classical pseudospectral updates are both O(G_pad log G_pad) once linear transforms "
                "are included; sparsity alone does not establish asymptotic quantum advantage."
            ),
            "missing_costs": [
                "product-formula steps versus physical time, cutoff K, amplitude, and target error",
                "finite-Fock or fault-tolerant synthesis precision",
                "viscous dissipation via dilation, reservoirs, or non-unitary algorithms",
                "state preparation and constraint-preserving initialization",
                "measurement cost and reconstruction of a G-component flow field",
                "hardware connectivity, loss, squeezing, and error correction",
            ],
            "readout_lower_bound": (
                "Reporting the complete resolved velocity field requires at least Omega(G) output quantities, so an "
                "exponential end-to-end speedup cannot follow from compact quantum state storage alone."
            ),
            "credible_targets": (
                "global observables, correlation functions, rare-event amplitudes, or embedded subroutines with "
                "limited readout, provided a full precision and fault-tolerance analysis later closes the gap"
            ),
        },
        "exact_rows": rows,
        "nominal_grid_extrapolations_are_fits_not_exact_counts": extrapolated_rows,
    }
    with (args.out / "resource_scaling_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    try:
        import matplotlib.pyplot as plt
    except (ImportError, PermissionError):
        plt = None
    if plt is not None:
        fig, axis = plt.subplots(figsize=(5.4, 3.8))
        axis.loglog(
            [row["grid_points_G"] for row in rows],
            [row["direct_first_order_non_gaussian_cubic_phase_primitives_empirical_estimate"] for row in rows],
            "o-",
            label="direct spectral estimate",
        )
        axis.loglog(
            [row["grid_points_G"] for row in rows],
            [row["structured_first_order_non_gaussian_cubic_phase_primitives"] for row in rows],
            "s-",
            label="structured local count",
        )
        axis.set_xlabel(r"grid points $G=(2K+1)^3$")
        axis.set_ylabel("cubic-phase primitives / first-order layer")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.out / "resource_scaling.pdf", bbox_inches="tight")
        fig.savefig(args.out / "resource_scaling.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(json.dumps({"summary": str(args.out), "triad_power": triad_power}, indent=2))


if __name__ == "__main__":
    main()
