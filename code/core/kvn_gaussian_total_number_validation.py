#!/usr/bin/env python3
"""Global-total-number Fock audit of the passive Gaussian cubic compilation.

The tensor-product local cutoff used for the wave-packet benchmark is not
invariant under passive mode mixing.  This independent audit uses the basis of
four-mode Fock states with total occupation not exceeding N.  Passive phase
and beam-splitter transformations preserve this space exactly, so it cleanly
separates Gaussian-map correctness from boundary effects caused by the cubic
gate and polarization sequence.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import expm_multiply

import kvn_two_triad_gate_validation as base


Occupation = Tuple[int, int, int, int]


def total_number_operators(
    total_cutoff: int,
) -> Tuple[
    List[Occupation],
    np.ndarray,
    List[csc_matrix],
    List[csc_matrix],
    List[csc_matrix],
    List[csc_matrix],
]:
    basis = [
        occupation
        for occupation in itertools.product(range(total_cutoff + 1), repeat=4)
        if sum(occupation) <= total_cutoff
    ]
    index = {occupation: position for position, occupation in enumerate(basis)}
    dimension = len(basis)
    a_ops: List[csc_matrix] = []
    for mode in range(4):
        rows: List[int] = []
        columns: List[int] = []
        values: List[float] = []
        for column, occupation in enumerate(basis):
            if occupation[mode] == 0:
                continue
            lowered = list(occupation)
            lowered[mode] -= 1
            rows.append(index[tuple(lowered)])
            columns.append(column)
            values.append(math.sqrt(occupation[mode]))
        a_ops.append(
            coo_matrix((values, (rows, columns)), shape=(dimension, dimension)).tocsc()
        )
    q_ops = [(operator + operator.getH()) / math.sqrt(2.0) for operator in a_ops]
    p_ops = [(operator - operator.getH()) / (1j * math.sqrt(2.0)) for operator in a_ops]
    number_ops = [(operator.getH() @ operator).tocsc() for operator in a_ops]
    totals = np.asarray([sum(occupation) for occupation in basis], dtype=int)
    return basis, totals, a_ops, q_ops, p_ops, number_ops


def truncated_product_coherent(
    q_means: np.ndarray, basis: Sequence[Occupation]
) -> np.ndarray:
    alphas = np.asarray(q_means, dtype=float) / math.sqrt(2.0)
    state = np.empty(len(basis), dtype=complex)
    prefactor = math.exp(-0.5 * float(np.dot(alphas, alphas)))
    for index, occupation in enumerate(basis):
        value = prefactor
        for alpha, number in zip(alphas, occupation):
            value *= alpha**number / math.sqrt(math.factorial(number))
        state[index] = value
    return state / np.linalg.norm(state)


def total_boundary_probability(
    state: np.ndarray, totals: np.ndarray, total_cutoff: int
) -> float:
    return float(np.sum(np.abs(state[totals >= max(0, total_cutoff - 1)]) ** 2))


def state_metrics(
    state: np.ndarray, totals: np.ndarray, total_cutoff: int
) -> Tuple[float, float]:
    return (
        total_boundary_probability(state, totals, total_cutoff),
        abs(float(np.linalg.norm(state)) - 1.0),
    )


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    records = list(rows)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def apply_gaussian_factor(
    state: np.ndarray,
    term: base.CubicTerm,
    signs: Sequence[int],
    theta: float,
    q_ops: Sequence[csc_matrix],
    a_ops: Sequence[csc_matrix],
    number_ops: Sequence[csc_matrix],
    beam_cache: MutableMapping[Tuple[int, int], csc_matrix],
    cube_cache: MutableMapping[int, csc_matrix],
    trace: List[Tuple[float, float]],
    totals: np.ndarray,
    total_cutoff: int,
) -> np.ndarray:
    def tracker(_label: str, intermediate: np.ndarray) -> None:
        trace.append(state_metrics(intermediate, totals, total_cutoff))

    state = base.apply_gaussian_map(
        state,
        term,
        signs,
        a_ops,
        number_ops,
        beam_cache,
        inverse=False,
        tracker=tracker,
    )
    if term.target not in cube_cache:
        q_target = q_ops[term.target]
        cube_cache[term.target] = (q_target @ q_target @ q_target).tocsc()
    strength = theta * term.coefficient * math.sqrt(3.0) ** 3 / 24.0
    state = expm_multiply(-1j * strength * cube_cache[term.target], state)
    tracker("single_mode_cubic", state)
    return base.apply_gaussian_map(
        state,
        term,
        signs,
        a_ops,
        number_ops,
        beam_cache,
        inverse=True,
        tracker=tracker,
    )


def factor_audit(
    state: np.ndarray,
    term: base.CubicTerm,
    signs: Sequence[int],
    theta: float,
    q_ops: Sequence[csc_matrix],
    p_ops: Sequence[csc_matrix],
    a_ops: Sequence[csc_matrix],
    number_ops: Sequence[csc_matrix],
    totals: np.ndarray,
    total_cutoff: int,
    beam_cache: MutableMapping[Tuple[int, int], csc_matrix],
    cube_cache: MutableMapping[int, csc_matrix],
) -> Dict[str, object]:
    linear = (
        signs[0] * p_ops[term.target]
        + signs[1] * q_ops[term.input_1]
        + signs[2] * q_ops[term.input_2]
    )
    norm_l = math.sqrt(3.0)
    mapped = base.apply_gaussian_map(
        state, term, signs, a_ops, number_ops, beam_cache, inverse=False
    )
    mapped = q_ops[term.target] @ mapped
    mapped = base.apply_gaussian_map(
        mapped, term, signs, a_ops, number_ops, beam_cache, inverse=True
    )
    reference_action = (linear @ state) / norm_l
    map_residual = float(
        np.linalg.norm(mapped - reference_action)
        / (np.linalg.norm(reference_action) + 1e-300)
    )

    collective_cube = (linear @ linear @ linear).tocsc()
    direct = expm_multiply(
        -1j * theta * term.coefficient * collective_cube / 24.0, state
    )
    trace = [state_metrics(state, totals, total_cutoff)]
    compiled = apply_gaussian_factor(
        state,
        term,
        signs,
        theta,
        q_ops,
        a_ops,
        number_ops,
        beam_cache,
        cube_cache,
        trace,
        totals,
        total_cutoff,
    )
    return {
        "total_number_cutoff": total_cutoff,
        "term": term.label,
        "sign_x": int(signs[0]),
        "sign_y": int(signs[1]),
        "sign_z": int(signs[2]),
        "gaussian_map_action_relative_residual": map_residual,
        "single_factor_state_l2_residual": base.phase_aligned_state_residual(
            direct, compiled
        ),
        "single_factor_state_infidelity": float(
            max(0.0, 1.0 - abs(np.vdot(direct, compiled)) ** 2)
        ),
        "maximum_intermediate_total_boundary_probability": max(item[0] for item in trace),
        "maximum_intermediate_norm_error": max(item[1] for item in trace),
    }


def collective_evolution(
    initial_state: np.ndarray,
    terms: Sequence[base.CubicTerm],
    q_ops: Sequence[csc_matrix],
    p_ops: Sequence[csc_matrix],
    final_time: float,
    steps: int,
) -> np.ndarray:
    state = initial_state.copy()
    for index, theta in base.strang_schedule(len(terms), final_time, steps):
        state = base.apply_compiled_monomial(
            state, terms[index], q_ops, p_ops, theta
        )
    return state


def monomial_evolution(
    initial_state: np.ndarray,
    h_terms: Sequence[csc_matrix],
    final_time: float,
    steps: int,
) -> np.ndarray:
    state = initial_state.copy()
    for index, theta in base.strang_schedule(len(h_terms), final_time, steps):
        state = expm_multiply(-1j * theta * h_terms[index], state)
    return state


def physical_evolution(
    initial_state: np.ndarray,
    terms: Sequence[base.CubicTerm],
    q_ops: Sequence[csc_matrix],
    a_ops: Sequence[csc_matrix],
    number_ops: Sequence[csc_matrix],
    totals: np.ndarray,
    total_cutoff: int,
    final_time: float,
    steps: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    state = initial_state.copy()
    trace = [state_metrics(state, totals, total_cutoff)]
    beam_cache: Dict[Tuple[int, int], csc_matrix] = {}
    cube_cache: Dict[int, csc_matrix] = {}
    schedule = base.strang_schedule(len(terms), final_time, steps)
    strengths: List[float] = []
    for index, theta in schedule:
        term = terms[index]
        for signs in base.POLARIZATION_SIGNS:
            strengths.append(
                abs(theta * term.coefficient * math.sqrt(3.0) ** 3 / 24.0)
            )
            state = apply_gaussian_factor(
                state,
                term,
                signs,
                theta,
                q_ops,
                a_ops,
                number_ops,
                beam_cache,
                cube_cache,
                trace,
                totals,
                total_cutoff,
            )
    primitives = 4 * len(schedule)
    return state, {
        "single_mode_cubic_phase_primitive_count": primitives,
        "gaussian_phase_rotation_count": 2 * primitives,
        "gaussian_beam_splitter_rotation_count": 4 * primitives,
        "ideal_map_squeezing_operation_count": 0,
        "maximum_abs_single_mode_cubic_strength": float(max(strengths)),
        "total_abs_single_mode_cubic_strength": float(sum(strengths)),
        "maximum_intermediate_total_boundary_probability": max(item[0] for item in trace),
        "maximum_intermediate_norm_error": max(item[1] for item in trace),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total-cutoffs", default="8,10,12,14,16")
    parser.add_argument("--selected-cutoff", type=int, default=14)
    parser.add_argument("--amplitude", type=float, default=2.5)
    parser.add_argument("--tau-end", type=float, default=0.18)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cutoffs = [int(value) for value in args.total_cutoffs.split(",")]
    if args.selected_cutoff not in cutoffs:
        raise ValueError("selected cutoff must be included in --total-cutoffs")
    terms, metadata = base.load_network(args.csv)
    q0 = args.amplitude * np.array([0.80, 0.55, 0.35, -0.30])
    final_time = args.tau_end / args.amplitude
    cutoff_rows: List[Dict[str, object]] = []
    selected_factor_rows: List[Dict[str, object]] = []
    selected_payload = None

    for cutoff in cutoffs:
        basis, totals, a_ops, q_ops, p_ops, number_ops = total_number_operators(cutoff)
        state = truncated_product_coherent(q0, basis)
        beam_cache: Dict[Tuple[int, int], csc_matrix] = {}
        cube_cache: Dict[int, csc_matrix] = {}
        factor_rows = [
            factor_audit(
                state,
                term,
                signs,
                final_time / 4.0,
                q_ops,
                p_ops,
                a_ops,
                number_ops,
                totals,
                cutoff,
                beam_cache,
                cube_cache,
            )
            for term in terms
            for signs in base.POLARIZATION_SIGNS
        ]
        collective = collective_evolution(state, terms, q_ops, p_ops, final_time, 1)
        physical, diagnostics = physical_evolution(
            state,
            terms,
            q_ops,
            a_ops,
            number_ops,
            totals,
            cutoff,
            final_time,
            1,
        )
        h_terms = base.hamiltonian_terms(terms, q_ops, p_ops)
        monomial = monomial_evolution(state, h_terms, final_time, 1)
        row: Dict[str, object] = {
            "total_number_cutoff": cutoff,
            "hilbert_dimension": len(basis),
            "maximum_gaussian_map_action_relative_residual": max(
                item["gaussian_map_action_relative_residual"] for item in factor_rows
            ),
            "maximum_single_factor_state_l2_residual": max(
                item["single_factor_state_l2_residual"] for item in factor_rows
            ),
            "physical_to_collective_one_step_state_l2_residual": base.phase_aligned_state_residual(
                collective, physical
            ),
            "physical_to_collective_one_step_state_infidelity": float(
                max(0.0, 1.0 - abs(np.vdot(collective, physical)) ** 2)
            ),
            "collective_to_monomial_one_step_state_l2_residual": base.phase_aligned_state_residual(
                monomial, collective
            ),
            "maximum_intermediate_total_boundary_probability": diagnostics[
                "maximum_intermediate_total_boundary_probability"
            ],
            "maximum_intermediate_norm_error": diagnostics[
                "maximum_intermediate_norm_error"
            ],
        }
        cutoff_rows.append(row)
        print(
            f"N={cutoff} dim={len(basis)} map={row['maximum_gaussian_map_action_relative_residual']:.3e} "
            f"physical/collective={row['physical_to_collective_one_step_state_l2_residual']:.3e} "
            f"edge={row['maximum_intermediate_total_boundary_probability']:.3e}",
            flush=True,
        )
        if cutoff == args.selected_cutoff:
            selected_factor_rows = factor_rows
            selected_payload = (
                basis,
                totals,
                a_ops,
                q_ops,
                p_ops,
                number_ops,
                state,
                h_terms,
            )

    if selected_payload is None:
        raise RuntimeError("selected total-number cutoff was not generated")
    basis, totals, a_ops, q_ops, p_ops, number_ops, state, h_terms = selected_payload
    strang_rows: List[Dict[str, object]] = []
    for steps in (1, 2, 4, 8):
        monomial = monomial_evolution(state, h_terms, final_time, steps)
        collective = collective_evolution(state, terms, q_ops, p_ops, final_time, steps)
        physical, diagnostics = physical_evolution(
            state,
            terms,
            q_ops,
            a_ops,
            number_ops,
            totals,
            args.selected_cutoff,
            final_time,
            steps,
        )
        row = {
            "strang_steps": steps,
            "physical_to_collective_state_l2_residual": base.phase_aligned_state_residual(
                collective, physical
            ),
            "physical_to_collective_state_infidelity": float(
                max(0.0, 1.0 - abs(np.vdot(collective, physical)) ** 2)
            ),
            "collective_to_monomial_state_l2_residual": base.phase_aligned_state_residual(
                monomial, collective
            ),
            **diagnostics,
        }
        strang_rows.append(row)
        print(
            f"steps={steps} physical/collective={row['physical_to_collective_state_l2_residual']:.3e} "
            f"collective/monomial={row['collective_to_monomial_state_l2_residual']:.3e}",
            flush=True,
        )

    selected_cutoff_row = next(
        row for row in cutoff_rows if row["total_number_cutoff"] == args.selected_cutoff
    )
    success_criteria = {
        "all_24_passive_gaussian_maps_below_1e-10": selected_cutoff_row[
            "maximum_gaussian_map_action_relative_residual"
        ]
        < 1e-10,
        "all_24_single_factor_compilations_below_1e-10": selected_cutoff_row[
            "maximum_single_factor_state_l2_residual"
        ]
        < 1e-10,
        "all_strang_sequences_physical_to_collective_below_1e-10": max(
            row["physical_to_collective_state_l2_residual"] for row in strang_rows
        )
        < 1e-10,
        "intermediate_norm_error_below_1e-10": max(
            row["maximum_intermediate_norm_error"] for row in strang_rows
        )
        < 1e-10,
        "selected_total_boundary_probability_below_1e-4": max(
            row["maximum_intermediate_total_boundary_probability"] for row in strang_rows
        )
        < 1e-4,
    }
    overall_success = bool(all(success_criteria.values()))
    summary = {
        "network": metadata,
        "basis": {
            "type": "four-mode global total-number Fock truncation",
            "selected_total_number_cutoff": args.selected_cutoff,
            "selected_hilbert_dimension": len(basis),
            "passive_maps_preserve_truncated_space": True,
        },
        "compilation": {
            "gaussian_map": "one local phase rotation and two passive Givens beam-splitter rotations",
            "ideal_map_squeezing_operations_per_factor": 0,
            "polarization_factors_per_monomial": 4,
        },
        "success_criteria": success_criteria,
        "overall_restricted_gaussian_compilation_success": overall_success,
        "claim_boundary": (
            "The audit validates explicit passive Gaussian map--single-mode-cubic--unmap "
            "compilation on the four-coordinate real invariant restriction. It separates "
            "Gaussian-map correctness from cubic and polarization boundary effects. The "
            "complete eight-coordinate network and physical gate imperfections remain open."
        ),
    }
    write_csv(args.out / "global_fock_gaussian_cutoff_scan.csv", cutoff_rows)
    write_csv(args.out / "global_fock_gaussian_by_factor.csv", selected_factor_rows)
    write_csv(args.out / "global_fock_gaussian_strang_scan.csv", strang_rows)
    with (args.out / "global_fock_gaussian_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"overall_success": overall_success, "out": str(args.out)}, indent=2))
    if not overall_success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
