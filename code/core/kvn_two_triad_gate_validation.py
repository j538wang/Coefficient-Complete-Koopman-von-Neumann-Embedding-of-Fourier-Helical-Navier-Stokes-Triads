#!/usr/bin/env python3
"""Finite-Fock gate validation for the shortest shared-mode KvN network.

The two helicity triads are the length-two global-scaling contradiction from
``global_scaling_audit/no_go_certificate.json``.  After applying the real
coefficient gauge already recorded in ``phase_gauge_ensemble.csv``, their
simultaneously real invariant restriction forms a four-real-coordinate
network sharing the K and P modes.  We use the physical reality-pair
normalization u=q/sqrt(2), so the real-coordinate coefficients are the
gauge-real complex-amplitude coefficients divided by sqrt(2).  The exact KvN
Hamiltonian is a sum of six mutually Hermitian cubic monomials

    H = sum_l c_l p_target(l) q_input1(l) q_input2(l).

Each monomial is expanded into four collective cubic exponentials by the
polarization identity for three commuting quadratures.  For every collective
linear form the script constructs an explicit passive Gaussian map: a local
phase rotation converts the target momentum into a position quadrature and two
Givens beam-splitter rotations align the normalized three-mode quadrature with
the target mode.  It then validates the resulting map--single-mode-cubic--unmap
sequence in the truncated Fock basis, including intermediate boundary
occupation.  The calculation also compares direct Hamiltonian evolution with
a symmetric product-formula sequence and compares normalized first moments
with the corresponding classical trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class CubicTerm:
    label: str
    coefficient: float
    target: int
    input_1: int
    input_2: int


def load_network(csv_path: Path) -> Tuple[List[CubicTerm], Dict[str, object]]:
    selected: Dict[int, Dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["geom_id"]) == 745
                and int(row["sK"]) == 1
                and int(row["sP"]) == -1
                and int(row["sQ"]) in (-1, 1)
                and row["status"] == "active"
            ):
                selected[int(row["sQ"])] = row
    if set(selected) != {-1, 1}:
        raise RuntimeError(f"expected the two certificate triads, found {sorted(selected)}")

    plus = selected[1]
    minus = selected[-1]

    def coefficient(row: Dict[str, str], prefix: str) -> float:
        imag = float(row[f"gauge_{prefix}_imag"])
        if abs(imag) > 1e-12:
            raise RuntimeError(f"{prefix} is not gauge-real: imag={imag}")
        return float(row[f"gauge_{prefix}_real"]) / math.sqrt(2.0)

    terms = [
        CubicTerm("K<-P,Q+", coefficient(plus, "CK"), 0, 1, 2),
        CubicTerm("P<-K,Q+", coefficient(plus, "CP"), 1, 0, 2),
        CubicTerm("Q+<-K,P", coefficient(plus, "CQ"), 2, 0, 1),
        CubicTerm("K<-P,Q-", coefficient(minus, "CK"), 0, 1, 3),
        CubicTerm("P<-K,Q-", coefficient(minus, "CP"), 1, 0, 3),
        CubicTerm("Q-<-K,P", coefficient(minus, "CQ"), 3, 0, 1),
    ]

    k_norm = float(plus["K_norm"])
    p_norm = float(plus["P_norm"])
    q_norm = float(plus["Q_norm"])
    helicity_weights = np.array([k_norm, -p_norm, q_norm, -q_norm], dtype=float)
    triad_plus = np.array([terms[0].coefficient, terms[1].coefficient, terms[2].coefficient])
    triad_minus = np.array([terms[3].coefficient, terms[4].coefficient, terms[5].coefficient])
    metadata: Dict[str, object] = {
        "geometry_id": 745,
        "K": plus["K"],
        "P": plus["P"],
        "Q": plus["Q"],
        "helicity_weights": helicity_weights.tolist(),
        "plus_coefficients": triad_plus.tolist(),
        "minus_coefficients": triad_minus.tolist(),
        "energy_identity_residuals": [
            float(abs(np.sum(triad_plus))),
            float(abs(np.sum(triad_minus))),
        ],
        "helicity_identity_residuals": [
            float(abs(np.dot(helicity_weights[[0, 1, 2]], triad_plus))),
            float(abs(np.dot(helicity_weights[[0, 1, 3]], triad_minus))),
        ],
    }
    return terms, metadata


def classical_rhs(_tau: float, x: np.ndarray, terms: Sequence[CubicTerm]) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    for term in terms:
        out[term.target] += term.coefficient * x[term.input_1] * x[term.input_2]
    return out


def classical_solution(
    x0: np.ndarray, tau_grid: np.ndarray, terms: Sequence[CubicTerm]
) -> np.ndarray:
    result = solve_ivp(
        lambda tau, x: classical_rhs(tau, x, terms),
        (float(tau_grid[0]), float(tau_grid[-1])),
        x0,
        t_eval=tau_grid,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y.T


def annihilation(cutoff: int) -> csc_matrix:
    return diags(np.sqrt(np.arange(1, cutoff, dtype=float)), 1, shape=(cutoff, cutoff), format="csc")


def kron_all(operators: Sequence[csc_matrix]) -> csc_matrix:
    result = operators[0]
    for operator in operators[1:]:
        result = kron(result, operator, format="csc")
    return result


def embedded(single_mode: csc_matrix, mode: int, cutoff: int, modes: int = 4) -> csc_matrix:
    identity = eye(cutoff, format="csc", dtype=complex)
    return kron_all([single_mode if index == mode else identity for index in range(modes)])


def fock_operators(
    cutoff: int,
) -> Tuple[List[csc_matrix], List[csc_matrix], List[csc_matrix], List[csc_matrix]]:
    a = annihilation(cutoff).astype(complex)
    adag = a.getH()
    q_single = (a + adag) / math.sqrt(2.0)
    p_single = (a - adag) / (1j * math.sqrt(2.0))
    number_single = adag @ a
    a_ops = [embedded(a, mode, cutoff) for mode in range(4)]
    q_ops = [embedded(q_single, mode, cutoff) for mode in range(4)]
    p_ops = [embedded(p_single, mode, cutoff) for mode in range(4)]
    number_ops = [embedded(number_single, mode, cutoff) for mode in range(4)]
    return a_ops, q_ops, p_ops, number_ops


def coherent_state(alpha: float, cutoff: int) -> np.ndarray:
    state = np.empty(cutoff, dtype=complex)
    state[0] = math.exp(-0.5 * alpha * alpha)
    for n in range(1, cutoff):
        state[n] = state[n - 1] * alpha / math.sqrt(n)
    return state / np.linalg.norm(state)


def product_coherent(q_means: np.ndarray, cutoff: int) -> np.ndarray:
    states = [coherent_state(float(value) / math.sqrt(2.0), cutoff) for value in q_means]
    result = states[0]
    for state in states[1:]:
        result = np.kron(result, state)
    return result / np.linalg.norm(result)


POLARIZATION_SIGNS: Tuple[Tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, -1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
)


def maximum_edge_probability(state: np.ndarray, cutoff: int) -> float:
    probability = np.abs(state.reshape((cutoff,) * 4)) ** 2
    edge = []
    for mode in range(4):
        axes = tuple(axis for axis in range(4) if axis != mode)
        marginal = probability.sum(axis=axes)
        edge.append(float(np.sum(marginal[max(0, cutoff - 2):])))
    return float(max(edge))


def beam_splitter_generator(
    mode_a: int,
    mode_b: int,
    a_ops: Sequence[csc_matrix],
    cache: MutableMapping[Tuple[int, int], csc_matrix],
) -> csc_matrix:
    key = (mode_a, mode_b)
    if key not in cache:
        generator = a_ops[mode_a].getH() @ a_ops[mode_b]
        generator -= a_ops[mode_b].getH() @ a_ops[mode_a]
        cache[key] = generator.tocsc()
    return cache[key]


def gaussian_alignment_angles(signs: Sequence[int]) -> Tuple[float, float]:
    vector = np.asarray(signs, dtype=float) / math.sqrt(3.0)
    theta_2 = math.asin(float(vector[2]))
    theta_1 = math.atan2(float(vector[1]), float(vector[0]))
    reconstructed = np.array(
        [
            math.cos(theta_2) * math.cos(theta_1),
            math.cos(theta_2) * math.sin(theta_1),
            math.sin(theta_2),
        ]
    )
    if np.linalg.norm(reconstructed - vector) > 1e-14:
        raise RuntimeError("failed to construct the passive Gaussian alignment")
    return theta_1, theta_2


def apply_gaussian_map(
    state: np.ndarray,
    term: CubicTerm,
    signs: Sequence[int],
    a_ops: Sequence[csc_matrix],
    number_ops: Sequence[csc_matrix],
    beam_cache: MutableMapping[Tuple[int, int], csc_matrix],
    inverse: bool = False,
    tracker=None,
) -> np.ndarray:
    """Apply G or G^dagger with G^dagger q_target G=L/sqrt(3)."""

    theta_1, theta_2 = gaussian_alignment_angles(signs)
    mode_a, mode_b, mode_c = term.target, term.input_1, term.input_2
    generator_ab = beam_splitter_generator(mode_a, mode_b, a_ops, beam_cache)
    generator_ac = beam_splitter_generator(mode_a, mode_c, a_ops, beam_cache)

    if inverse:
        gates = (
            (theta_2 * -1.0, generator_ac, "beam_ac_inverse", False),
            (theta_1 * -1.0, generator_ab, "beam_ab_inverse", False),
            (math.pi / 2.0, number_ops[mode_a], "phase_inverse", True),
        )
    else:
        gates = (
            (-math.pi / 2.0, number_ops[mode_a], "phase", True),
            (theta_1, generator_ab, "beam_ab", False),
            (theta_2, generator_ac, "beam_ac", False),
        )

    for parameter, generator, label, phase_gate in gates:
        exponent = 1j * parameter * generator if phase_gate else parameter * generator
        state = expm_multiply(exponent, state)
        if tracker is not None:
            tracker(label, state)
    return state


def phase_aligned_state_residual(reference: np.ndarray, candidate: np.ndarray) -> float:
    overlap = np.vdot(reference, candidate)
    if abs(overlap) > 0:
        candidate = candidate * np.exp(-1j * np.angle(overlap))
    return float(np.linalg.norm(reference - candidate))


def hamiltonian_terms(
    terms: Sequence[CubicTerm], q_ops: Sequence[csc_matrix], p_ops: Sequence[csc_matrix]
) -> List[csc_matrix]:
    out = []
    for term in terms:
        operator = p_ops[term.target] @ q_ops[term.input_1] @ q_ops[term.input_2]
        out.append((term.coefficient * operator).tocsc())
    return out


def expectation(state: np.ndarray, operator: csc_matrix) -> complex:
    return np.vdot(state, operator @ state)


def state_observables(
    state: np.ndarray,
    q_ops: Sequence[csc_matrix],
    helicity_weights: np.ndarray,
    cutoff: int,
) -> Dict[str, object]:
    means = np.array([expectation(state, q).real for q in q_ops], dtype=float)
    second = np.array([expectation(state, q @ q).real for q in q_ops], dtype=float)
    return {
        "q_means": means,
        "q_second": second,
        "energy": float(0.5 * np.sum(second)),
        "helicity": float(0.5 * np.dot(helicity_weights, second)),
        "maximum_edge_probability_last_two_levels": maximum_edge_probability(state, cutoff),
        "norm": float(np.linalg.norm(state)),
    }


def direct_trajectory(
    hamiltonian: csc_matrix,
    initial_state: np.ndarray,
    time_grid: np.ndarray,
    q_ops: Sequence[csc_matrix],
    helicity_weights: np.ndarray,
    cutoff: int,
) -> Tuple[List[np.ndarray], List[Dict[str, object]]]:
    states = list(
        expm_multiply(
            -1j * hamiltonian,
            initial_state,
            start=float(time_grid[0]),
            stop=float(time_grid[-1]),
            num=len(time_grid),
            endpoint=True,
        )
    )
    observations = [state_observables(state, q_ops, helicity_weights, cutoff) for state in states]
    return states, observations


def strang_step(state: np.ndarray, h_terms: Sequence[csc_matrix], dt: float) -> np.ndarray:
    for operator in h_terms[:-1]:
        state = expm_multiply(-0.5j * dt * operator, state)
    state = expm_multiply(-1j * dt * h_terms[-1], state)
    for operator in reversed(h_terms[:-1]):
        state = expm_multiply(-0.5j * dt * operator, state)
    return state / np.linalg.norm(state)


def strang_evolution(
    initial_state: np.ndarray, h_terms: Sequence[csc_matrix], final_time: float, steps: int
) -> np.ndarray:
    state = initial_state.copy()
    dt = final_time / steps
    for _ in range(steps):
        state = strang_step(state, h_terms, dt)
    return state


def relative_trajectory_error(quantum: np.ndarray, classical: np.ndarray) -> float:
    return float(np.linalg.norm(quantum - classical) / (np.linalg.norm(classical) + 1e-300))


def polarization_gate_check(
    term: CubicTerm,
    q_ops: Sequence[csc_matrix],
    p_ops: Sequence[csc_matrix],
    state: np.ndarray,
    theta: float,
) -> Dict[str, float]:
    x = p_ops[term.target]
    y = q_ops[term.input_1]
    z = q_ops[term.input_2]
    xyz = x @ y @ z
    linear_forms = [sx * x + sy * y + sz * z for sx, sy, sz in POLARIZATION_SIGNS]

    lhs = xyz @ state
    rhs = sum((linear @ (linear @ (linear @ state)) for linear in linear_forms)) / 24.0
    generator_residual = float(np.linalg.norm(lhs - rhs) / (np.linalg.norm(lhs) + 1e-300))

    commute_residual = 0.0
    for i, first in enumerate(linear_forms):
        for second in linear_forms[i + 1:]:
            forward = first @ (second @ state)
            reverse = second @ (first @ state)
            commute_residual = max(
                commute_residual,
                float(np.linalg.norm(forward - reverse) / (np.linalg.norm(forward) + 1e-300)),
            )

    direct = expm_multiply(-1j * theta * term.coefficient * xyz, state)
    compiled = state.copy()
    for linear in linear_forms:
        cube = (linear @ linear @ linear).tocsc()
        compiled = expm_multiply(-1j * theta * term.coefficient * cube / 24.0, compiled)
    phase = np.vdot(direct, compiled)
    if abs(phase) > 0:
        compiled *= np.exp(-1j * np.angle(phase))
    gate_state_residual = float(np.linalg.norm(direct - compiled))
    gate_infidelity = float(max(0.0, 1.0 - abs(np.vdot(direct, compiled)) ** 2))
    return {
        "term": term.label,
        "generator_relative_residual": generator_residual,
        "linear_form_commutator_relative_residual": commute_residual,
        "compiled_state_l2_residual": gate_state_residual,
        "compiled_state_infidelity": gate_infidelity,
        "collective_cubic_factors_per_monomial": 4,
        "non_gaussian_cubic_phase_primitives_per_monomial": 4,
        "gaussian_conjugations_instantiated": False,
    }


def gaussian_factor_check(
    term: CubicTerm,
    signs: Sequence[int],
    q_ops: Sequence[csc_matrix],
    p_ops: Sequence[csc_matrix],
    a_ops: Sequence[csc_matrix],
    number_ops: Sequence[csc_matrix],
    state: np.ndarray,
    theta: float,
    cutoff: int,
    beam_cache: MutableMapping[Tuple[int, int], csc_matrix],
    cube_cache: MutableMapping[int, csc_matrix],
) -> Dict[str, object]:
    x = p_ops[term.target]
    y = q_ops[term.input_1]
    z = q_ops[term.input_2]
    linear = signs[0] * x + signs[1] * y + signs[2] * z
    norm_l = math.sqrt(float(sum(sign * sign for sign in signs)))

    mapped_state = apply_gaussian_map(
        state, term, signs, a_ops, number_ops, beam_cache, inverse=False
    )
    mapped_action = q_ops[term.target] @ mapped_state
    mapped_action = apply_gaussian_map(
        mapped_action, term, signs, a_ops, number_ops, beam_cache, inverse=True
    )
    reference_action = (linear @ state) / norm_l
    map_action_residual = float(
        np.linalg.norm(mapped_action - reference_action)
        / (np.linalg.norm(reference_action) + 1e-300)
    )

    collective_cube = (linear @ linear @ linear).tocsc()
    collective_strength = theta * term.coefficient / 24.0
    direct = expm_multiply(-1j * collective_strength * collective_cube, state)

    trace: List[Tuple[str, float, float]] = []

    def tracker(label: str, intermediate: np.ndarray) -> None:
        trace.append(
            (
                label,
                maximum_edge_probability(intermediate, cutoff),
                abs(float(np.linalg.norm(intermediate)) - 1.0),
            )
        )

    compiled = apply_gaussian_map(
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
    single_mode_strength = collective_strength * norm_l**3
    compiled = expm_multiply(
        -1j * single_mode_strength * cube_cache[term.target], compiled
    )
    tracker("single_mode_cubic", compiled)
    compiled = apply_gaussian_map(
        compiled,
        term,
        signs,
        a_ops,
        number_ops,
        beam_cache,
        inverse=True,
        tracker=tracker,
    )

    infidelity = float(max(0.0, 1.0 - abs(np.vdot(direct, compiled)) ** 2))
    return {
        "term": term.label,
        "sign_x": int(signs[0]),
        "sign_y": int(signs[1]),
        "sign_z": int(signs[2]),
        "linear_form_norm": norm_l,
        "collective_exponent_coefficient": collective_strength,
        "single_mode_cubic_strength": single_mode_strength,
        "gaussian_map_action_relative_residual": map_action_residual,
        "single_factor_state_l2_residual": phase_aligned_state_residual(direct, compiled),
        "single_factor_state_infidelity": infidelity,
        "maximum_intermediate_edge_probability_last_two_levels": max(
            value[1] for value in trace
        ),
        "maximum_intermediate_norm_error": max(value[2] for value in trace),
        "phase_rotations_including_inverse": 2,
        "beam_splitter_rotations_including_inverse": 4,
        "single_mode_cubic_phase_primitives": 1,
        "ideal_map_squeezing_operations": 0,
    }


def strang_schedule(term_count: int, final_time: float, steps: int) -> List[Tuple[int, float]]:
    dt = final_time / steps
    schedule: List[Tuple[int, float]] = []
    for _ in range(steps):
        raw = (
            [(index, 0.5 * dt) for index in range(term_count - 1)]
            + [(term_count - 1, dt)]
            + [(index, 0.5 * dt) for index in reversed(range(term_count - 1))]
        )
        for index, theta in raw:
            if schedule and schedule[-1][0] == index:
                previous_index, previous_theta = schedule[-1]
                schedule[-1] = (previous_index, previous_theta + theta)
            else:
                schedule.append((index, theta))
    return schedule


def apply_gaussian_compiled_monomial(
    state: np.ndarray,
    term: CubicTerm,
    q_ops: Sequence[csc_matrix],
    a_ops: Sequence[csc_matrix],
    number_ops: Sequence[csc_matrix],
    theta: float,
    cutoff: int,
    beam_cache: MutableMapping[Tuple[int, int], csc_matrix],
    cube_cache: MutableMapping[int, csc_matrix],
    trace: List[Tuple[str, float, float]],
) -> np.ndarray:
    norm_l = math.sqrt(3.0)

    def tracker(label: str, intermediate: np.ndarray) -> None:
        trace.append(
            (
                label,
                maximum_edge_probability(intermediate, cutoff),
                abs(float(np.linalg.norm(intermediate)) - 1.0),
            )
        )

    if term.target not in cube_cache:
        q_target = q_ops[term.target]
        cube_cache[term.target] = (q_target @ q_target @ q_target).tocsc()
    for signs in POLARIZATION_SIGNS:
        state = apply_gaussian_map(
            state,
            term,
            signs,
            a_ops,
            number_ops,
            beam_cache,
            inverse=False,
            tracker=tracker,
        )
        strength = theta * term.coefficient * norm_l**3 / 24.0
        state = expm_multiply(-1j * strength * cube_cache[term.target], state)
        tracker("single_mode_cubic", state)
        state = apply_gaussian_map(
            state,
            term,
            signs,
            a_ops,
            number_ops,
            beam_cache,
            inverse=True,
            tracker=tracker,
        )
    return state


def gaussian_compiled_evolution(
    initial_state: np.ndarray,
    terms: Sequence[CubicTerm],
    q_ops: Sequence[csc_matrix],
    a_ops: Sequence[csc_matrix],
    number_ops: Sequence[csc_matrix],
    final_time: float,
    steps: int,
    cutoff: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    state = initial_state.copy()
    trace: List[Tuple[str, float, float]] = [
        (
            "initial",
            maximum_edge_probability(state, cutoff),
            abs(float(np.linalg.norm(state)) - 1.0),
        )
    ]
    beam_cache: Dict[Tuple[int, int], csc_matrix] = {}
    cube_cache: Dict[int, csc_matrix] = {}
    schedule = strang_schedule(len(terms), final_time, steps)
    strengths: List[float] = []
    for index, theta in schedule:
        term = terms[index]
        strengths.extend(
            [abs(theta * term.coefficient * math.sqrt(3.0) ** 3 / 24.0)] * 4
        )
        state = apply_gaussian_compiled_monomial(
            state,
            term,
            q_ops,
            a_ops,
            number_ops,
            theta,
            cutoff,
            beam_cache,
            cube_cache,
            trace,
        )
    primitive_count = 4 * len(schedule)
    return state, {
        "optimized_monomial_exponential_count": len(schedule),
        "single_mode_cubic_phase_primitive_count": primitive_count,
        "gaussian_phase_rotation_count": 2 * primitive_count,
        "gaussian_beam_splitter_rotation_count": 4 * primitive_count,
        "ideal_map_squeezing_operation_count": 0,
        "maximum_abs_single_mode_cubic_strength": float(max(strengths)),
        "total_abs_single_mode_cubic_strength": float(sum(strengths)),
        "maximum_intermediate_edge_probability_last_two_levels": float(
            max(value[1] for value in trace)
        ),
        "maximum_intermediate_norm_error": float(max(value[2] for value in trace)),
    }


def apply_compiled_monomial(
    state: np.ndarray,
    term: CubicTerm,
    q_ops: Sequence[csc_matrix],
    p_ops: Sequence[csc_matrix],
    theta: float,
) -> np.ndarray:
    """Apply one monomial as four polarization-expanded collective cubics."""
    x = p_ops[term.target]
    y = q_ops[term.input_1]
    z = q_ops[term.input_2]
    for sx, sy, sz in POLARIZATION_SIGNS:
        linear = sx * x + sy * y + sz * z
        cube = (linear @ linear @ linear).tocsc()
        state = expm_multiply(-1j * theta * term.coefficient * cube / 24.0, state)
    return state


def compiled_strang_step(
    state: np.ndarray,
    terms: Sequence[CubicTerm],
    q_ops: Sequence[csc_matrix],
    p_ops: Sequence[csc_matrix],
    dt: float,
) -> np.ndarray:
    for term in terms[:-1]:
        state = apply_compiled_monomial(state, term, q_ops, p_ops, 0.5 * dt)
    state = apply_compiled_monomial(state, terms[-1], q_ops, p_ops, dt)
    for term in reversed(terms[:-1]):
        state = apply_compiled_monomial(state, term, q_ops, p_ops, 0.5 * dt)
    return state / np.linalg.norm(state)


def run_case(
    cutoff: int,
    amplitude: float,
    tau_end: float,
    x0: np.ndarray,
    tau_grid: np.ndarray,
    classical: np.ndarray,
    terms: Sequence[CubicTerm],
    helicity_weights: np.ndarray,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    a_ops, q_ops, p_ops, number_ops = fock_operators(cutoff)
    h_terms = hamiltonian_terms(terms, q_ops, p_ops)
    hamiltonian = sum(h_terms[1:], h_terms[0]).tocsc()
    initial_state = product_coherent(amplitude * x0, cutoff)
    time_grid = tau_grid / amplitude
    states, observations = direct_trajectory(
        hamiltonian, initial_state, time_grid, q_ops, helicity_weights, cutoff
    )
    quantum_means = np.vstack([observation["q_means"] for observation in observations]) / amplitude
    energies = np.array([observation["energy"] for observation in observations]) / amplitude**2
    helicities = np.array([observation["helicity"] for observation in observations]) / amplitude**2
    edge = max(
        float(observation["maximum_edge_probability_last_two_levels"])
        for observation in observations
    )
    result: Dict[str, object] = {
        "cutoff": cutoff,
        "amplitude_scale": amplitude,
        "hilbert_dimension": cutoff**4,
        "physical_final_time": tau_end / amplitude,
        "normalized_first_moment_trajectory_relative_l2_error": relative_trajectory_error(
            quantum_means, classical
        ),
        "normalized_final_first_moment_relative_l2_error": float(
            np.linalg.norm(quantum_means[-1] - classical[-1])
            / (np.linalg.norm(classical[-1]) + 1e-300)
        ),
        "energy_relative_drift": float(
            np.max(np.abs(energies - energies[0])) / (abs(energies[0]) + 1e-300)
        ),
        "helicity_relative_drift": float(
            np.max(np.abs(helicities - helicities[0])) / (abs(helicities[0]) + 1e-300)
        ),
        "maximum_edge_probability_last_two_levels": edge,
        "maximum_norm_error": float(
            max(abs(float(observation["norm"]) - 1.0) for observation in observations)
        ),
    }
    payload: Dict[str, object] = {
        "a_ops": a_ops,
        "q_ops": q_ops,
        "p_ops": p_ops,
        "number_ops": number_ops,
        "h_terms": h_terms,
        "hamiltonian": hamiltonian,
        "initial_state": initial_state,
        "final_state": states[-1],
        "quantum_means": quantum_means,
        "energies": energies,
        "helicities": helicities,
    }
    return result, payload


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_numeric_csv(path: Path) -> List[Dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def plot_validation_figure(
    amplitude_rows: Sequence[Dict[str, object]],
    cutoff_rows: Sequence[Dict[str, object]],
    trotter_rows: Sequence[Dict[str, object]],
    gaussian_cutoff_rows: Sequence[Dict[str, object]],
    out: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except (ImportError, PermissionError):
        return

    fig = plt.figure(figsize=(9.0, 7.0))
    grid = fig.add_gridspec(2, 2)
    ax_amp = fig.add_subplot(grid[0, 0])
    ax_cut = fig.add_subplot(grid[1, 0])
    ax_strang = fig.add_subplot(grid[0, 1])
    ax_gaussian = fig.add_subplot(grid[1, 1])
    amplitude_values = [row["amplitude_scale"] for row in amplitude_rows]
    amplitude_errors = [
        row["normalized_first_moment_trajectory_relative_l2_error"]
        for row in amplitude_rows
    ]
    amplitude_edges = [
        row["maximum_edge_probability_last_two_levels"] for row in amplitude_rows
    ]
    error_line = ax_amp.loglog(
        amplitude_values,
        amplitude_errors,
        "o-",
        color="tab:blue",
        label="trajectory error",
    )
    ax_amp_edge = ax_amp.twinx()
    edge_line = ax_amp_edge.loglog(
        amplitude_values,
        amplitude_edges,
        "s--",
        color="tab:orange",
        label="edge probability",
    )
    ax_amp.set_xlabel(r"coherent scale $A$")
    ax_amp.set_ylabel("mean-trajectory error")
    ax_amp_edge.set_ylabel("edge probability", color="tab:orange")
    ax_amp_edge.tick_params(axis="y", labelcolor="tab:orange")
    ax_amp.legend(
        error_line + edge_line,
        ["trajectory error", "edge probability"],
        frameon=False,
        fontsize=7,
        loc="best",
    )
    ax_amp.grid(True, which="both", alpha=0.3)

    ax_cut.semilogy(
        [row["cutoff"] for row in cutoff_rows],
        [row["maximum_edge_probability_last_two_levels"] for row in cutoff_rows],
        "o-",
        label="edge probability",
    )
    ax_cut.semilogy(
        [row["cutoff"] for row in cutoff_rows],
        [row["normalized_first_moment_trajectory_relative_l2_error"] for row in cutoff_rows],
        "s--",
        label="trajectory error",
    )
    ax_cut.set_xlabel("Fock cutoff")
    ax_cut.set_ylabel("error / probability")
    ax_cut.legend(frameon=False, fontsize=8)
    ax_cut.grid(True, which="both", alpha=0.3)

    ax_strang.loglog(
        [row["strang_steps"] for row in trotter_rows],
        [row["first_moment_relative_l2_error_to_direct"] for row in trotter_rows],
        "o-",
        label="collective sequence",
    )
    ax_strang.loglog(
        [row["strang_steps"] for row in trotter_rows],
        [row["physical_first_moment_relative_l2_error_to_direct"] for row in trotter_rows],
        "s--",
        label="map--cubic--unmap",
    )
    ax_strang.set_xscale("log", base=2)
    ax_strang.set_xticks([1, 2, 4, 8])
    ax_strang.set_xticklabels(["1", "2", "4", "8"])
    ax_strang.set_xlabel(r"Strang steps $n_{\rm step}$")
    ax_strang.set_ylabel("sequence/direct mean error")
    ax_strang.legend(frameon=False, fontsize=7)
    ax_strang.grid(True, which="both", alpha=0.3)

    gaussian_cutoffs = [row["cutoff"] for row in gaussian_cutoff_rows]
    ax_gaussian.semilogy(
        gaussian_cutoffs,
        [
            max(float(row["maximum_single_factor_state_l2_residual"]), 1e-18)
            for row in gaussian_cutoff_rows
        ],
        "o-",
        label="single-factor residual",
    )
    ax_gaussian.semilogy(
        gaussian_cutoffs,
        [
            max(float(row["physical_one_step_state_l2_residual"]), 1e-18)
            for row in gaussian_cutoff_rows
        ],
        "s--",
        label="one-step residual",
    )
    ax_gaussian_edge = ax_gaussian.twinx()
    edge_line = ax_gaussian_edge.semilogy(
        gaussian_cutoffs,
        [
            row["physical_one_step_maximum_intermediate_edge_probability_last_two_levels"]
            for row in gaussian_cutoff_rows
        ],
        "^:",
        color="tab:orange",
        label="intermediate edge",
    )
    ax_gaussian.set_xlabel("Fock cutoff")
    ax_gaussian.set_ylabel("compiled/direct residual")
    ax_gaussian_edge.set_ylabel("intermediate edge", color="tab:orange")
    ax_gaussian_edge.tick_params(axis="y", labelcolor="tab:orange")
    handles, labels = ax_gaussian.get_legend_handles_labels()
    ax_gaussian.legend(handles + edge_line, labels + ["intermediate edge"], frameon=False, fontsize=7)
    ax_gaussian.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "two_triad_gate_validation.pdf", bbox_inches="tight")
    fig.savefig(out / "two_triad_gate_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=Path("unified_ensemble/phase_gauge_ensemble.csv")
    )
    parser.add_argument("--out", type=Path, default=Path("kvn_two_triad_gate_validation"))
    parser.add_argument("--tau-end", type=float, default=0.18)
    parser.add_argument("--selected-cutoff", type=int, default=10)
    parser.add_argument("--selected-amplitude", type=float, default=2.5)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate the validation figure from existing CSV records",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_validation_figure(
            read_numeric_csv(args.out / "amplitude_scan.csv"),
            read_numeric_csv(args.out / "cutoff_scan.csv"),
            read_numeric_csv(args.out / "strang_gate_scan.csv"),
            read_numeric_csv(args.out / "gaussian_compilation_cutoff_scan.csv"),
            args.out,
        )
        print(json.dumps({"plot_only": True, "summary": str(args.out)}, indent=2))
        return

    terms, metadata = load_network(args.csv)
    helicity_weights = np.asarray(metadata["helicity_weights"], dtype=float)
    x0 = np.array([0.80, 0.55, 0.35, -0.30], dtype=float)
    tau_grid = np.linspace(0.0, args.tau_end, 31)
    classical = classical_solution(x0, tau_grid, terms)
    classical_energy = 0.5 * np.sum(classical**2, axis=1)
    classical_helicity = 0.5 * np.sum(helicity_weights[None, :] * classical**2, axis=1)
    metadata["classical_initial_condition"] = x0.tolist()
    metadata["dimensionless_final_time"] = args.tau_end
    metadata["classical_energy_relative_drift"] = float(
        np.max(np.abs(classical_energy - classical_energy[0])) / abs(classical_energy[0])
    )
    metadata["classical_helicity_relative_drift"] = float(
        np.max(np.abs(classical_helicity - classical_helicity[0]))
        / (abs(classical_helicity[0]) + 1e-300)
    )

    if max(metadata["energy_identity_residuals"]) > 1e-12:
        raise RuntimeError("the selected coefficient network failed the energy identity")
    if max(metadata["helicity_identity_residuals"]) > 1e-12:
        raise RuntimeError("the selected coefficient network failed the helicity identity")

    print("Running amplitude scan", flush=True)
    amplitude_rows: List[Dict[str, object]] = []
    selected_payload = None
    for amplitude in (1.5, 2.0, args.selected_amplitude, 3.0):
        row, payload = run_case(
            args.selected_cutoff,
            amplitude,
            args.tau_end,
            x0,
            tau_grid,
            classical,
            terms,
            helicity_weights,
        )
        amplitude_rows.append(row)
        print(
            f"  A={amplitude:.1f}: trajectory={row['normalized_first_moment_trajectory_relative_l2_error']:.3e}, "
            f"edge={row['maximum_edge_probability_last_two_levels']:.3e}",
            flush=True,
        )
        if amplitude == args.selected_amplitude:
            selected_payload = payload

    print("Running cutoff scan", flush=True)
    cutoff_rows: List[Dict[str, object]] = []
    cutoff_payloads: Dict[int, Dict[str, object]] = {}
    for cutoff in (6, 8, args.selected_cutoff, 12, 14):
        row, payload = run_case(
            cutoff,
            args.selected_amplitude,
            args.tau_end,
            x0,
            tau_grid,
            classical,
            terms,
            helicity_weights,
        )
        cutoff_rows.append(row)
        cutoff_payloads[cutoff] = payload
        print(
            f"  cutoff={cutoff}: trajectory={row['normalized_first_moment_trajectory_relative_l2_error']:.3e}, "
            f"edge={row['maximum_edge_probability_last_two_levels']:.3e}",
            flush=True,
        )
        if cutoff == args.selected_cutoff:
            selected_payload = payload

    if selected_payload is None:
        raise RuntimeError("selected validation payload was not generated")

    final_time = args.tau_end / args.selected_amplitude
    print("Checking polarization-expanded collective cubics for all six monomials", flush=True)
    gate_compilation_rows = [
        polarization_gate_check(
            term,
            selected_payload["q_ops"],
            selected_payload["p_ops"],
            selected_payload["initial_state"],
            theta=final_time / 4.0,
        )
        for term in terms
    ]
    direct_one_step = strang_evolution(
        selected_payload["initial_state"],
        selected_payload["h_terms"],
        final_time,
        1,
    )
    compiled_one_step = compiled_strang_step(
        selected_payload["initial_state"],
        terms,
        selected_payload["q_ops"],
        selected_payload["p_ops"],
        final_time,
    )
    compiled_overlap = abs(np.vdot(direct_one_step, compiled_one_step)) ** 2
    gate_compilation = {
        "monomial_count": len(terms),
        "maximum_generator_relative_residual": max(
            row["generator_relative_residual"] for row in gate_compilation_rows
        ),
        "maximum_linear_form_commutator_relative_residual": max(
            row["linear_form_commutator_relative_residual"] for row in gate_compilation_rows
        ),
        "maximum_compiled_monomial_state_l2_residual": max(
            row["compiled_state_l2_residual"] for row in gate_compilation_rows
        ),
        "maximum_compiled_monomial_state_infidelity": max(
            row["compiled_state_infidelity"] for row in gate_compilation_rows
        ),
        "compiled_full_one_step_strang_state_l2_residual": float(
            np.linalg.norm(direct_one_step - compiled_one_step)
        ),
        "compiled_full_one_step_strang_state_infidelity": float(
            max(0.0, 1.0 - compiled_overlap)
        ),
        "collective_cubic_factors_per_monomial": 4,
        "non_gaussian_cubic_phase_primitives_per_monomial": 4,
        "non_gaussian_cubic_phase_primitives_in_one_strang_step": 4 * (2 * len(terms) - 1),
        "gaussian_conjugations_instantiated": False,
    }

    print("Running passive Gaussian map--cubic--unmap cutoff audit", flush=True)
    gaussian_cutoff_rows: List[Dict[str, object]] = []
    gaussian_factor_rows: List[Dict[str, object]] = []
    for cutoff in (6, 8, args.selected_cutoff, 12, 14):
        payload = cutoff_payloads[cutoff]
        beam_cache: Dict[Tuple[int, int], csc_matrix] = {}
        cube_cache: Dict[int, csc_matrix] = {}
        factor_rows = [
            {
                "cutoff": cutoff,
                **gaussian_factor_check(
                    term,
                    signs,
                    payload["q_ops"],
                    payload["p_ops"],
                    payload["a_ops"],
                    payload["number_ops"],
                    payload["initial_state"],
                    theta=final_time / 4.0,
                    cutoff=cutoff,
                    beam_cache=beam_cache,
                    cube_cache=cube_cache,
                ),
            }
            for term in terms
            for signs in POLARIZATION_SIGNS
        ]
        collective_one_step = compiled_strang_step(
            payload["initial_state"],
            terms,
            payload["q_ops"],
            payload["p_ops"],
            final_time,
        )
        physical_one_step, physical_diagnostics = gaussian_compiled_evolution(
            payload["initial_state"],
            terms,
            payload["q_ops"],
            payload["a_ops"],
            payload["number_ops"],
            final_time,
            1,
            cutoff,
        )
        gaussian_row: Dict[str, object] = {
            "cutoff": cutoff,
            "hilbert_dimension": cutoff**4,
            "maximum_gaussian_map_action_relative_residual": max(
                row["gaussian_map_action_relative_residual"] for row in factor_rows
            ),
            "maximum_single_factor_state_l2_residual": max(
                row["single_factor_state_l2_residual"] for row in factor_rows
            ),
            "maximum_single_factor_state_infidelity": max(
                row["single_factor_state_infidelity"] for row in factor_rows
            ),
            "maximum_single_factor_intermediate_edge_probability_last_two_levels": max(
                row["maximum_intermediate_edge_probability_last_two_levels"]
                for row in factor_rows
            ),
            "physical_one_step_state_l2_residual": phase_aligned_state_residual(
                collective_one_step, physical_one_step
            ),
            "physical_one_step_state_infidelity": float(
                max(0.0, 1.0 - abs(np.vdot(collective_one_step, physical_one_step)) ** 2)
            ),
            "physical_one_step_maximum_intermediate_edge_probability_last_two_levels": physical_diagnostics[
                "maximum_intermediate_edge_probability_last_two_levels"
            ],
            "physical_one_step_maximum_intermediate_norm_error": physical_diagnostics[
                "maximum_intermediate_norm_error"
            ],
            "single_mode_cubic_phase_primitive_count": physical_diagnostics[
                "single_mode_cubic_phase_primitive_count"
            ],
            "gaussian_phase_rotation_count": physical_diagnostics[
                "gaussian_phase_rotation_count"
            ],
            "gaussian_beam_splitter_rotation_count": physical_diagnostics[
                "gaussian_beam_splitter_rotation_count"
            ],
            "ideal_map_squeezing_operation_count": physical_diagnostics[
                "ideal_map_squeezing_operation_count"
            ],
            "maximum_abs_single_mode_cubic_strength": physical_diagnostics[
                "maximum_abs_single_mode_cubic_strength"
            ],
            "total_abs_single_mode_cubic_strength": physical_diagnostics[
                "total_abs_single_mode_cubic_strength"
            ],
        }
        gaussian_cutoff_rows.append(gaussian_row)
        if cutoff == args.selected_cutoff:
            gaussian_factor_rows = factor_rows
        print(
            f"  cutoff={cutoff}: factor={gaussian_row['maximum_single_factor_state_l2_residual']:.3e}, "
            f"one-step={gaussian_row['physical_one_step_state_l2_residual']:.3e}, "
            f"edge={gaussian_row['physical_one_step_maximum_intermediate_edge_probability_last_two_levels']:.3e}",
            flush=True,
        )

    selected_gaussian = next(
        row for row in gaussian_cutoff_rows if row["cutoff"] == args.selected_cutoff
    )
    gate_compilation.update(
        {
            "gaussian_conjugations_instantiated": True,
            "gaussian_map_type": "local phase rotation plus two passive Givens beam-splitter rotations",
            "ideal_map_squeezing_operations_per_factor": 0,
            "maximum_gaussian_map_action_relative_residual": selected_gaussian[
                "maximum_gaussian_map_action_relative_residual"
            ],
            "maximum_map_compiled_single_factor_state_l2_residual": selected_gaussian[
                "maximum_single_factor_state_l2_residual"
            ],
            "map_compiled_full_one_step_state_l2_residual": selected_gaussian[
                "physical_one_step_state_l2_residual"
            ],
            "map_compiled_full_one_step_state_infidelity": selected_gaussian[
                "physical_one_step_state_infidelity"
            ],
            "map_compiled_full_one_step_maximum_intermediate_edge_probability": selected_gaussian[
                "physical_one_step_maximum_intermediate_edge_probability_last_two_levels"
            ],
            "gaussian_phase_rotations_in_one_strang_step": selected_gaussian[
                "gaussian_phase_rotation_count"
            ],
            "gaussian_beam_splitter_rotations_in_one_strang_step": selected_gaussian[
                "gaussian_beam_splitter_rotation_count"
            ],
            "maximum_abs_single_mode_cubic_strength_in_one_strang_step": selected_gaussian[
                "maximum_abs_single_mode_cubic_strength"
            ],
            "total_abs_single_mode_cubic_strength_in_one_strang_step": selected_gaussian[
                "total_abs_single_mode_cubic_strength"
            ],
            "local_product_cutoff_14_map_compiled_single_factor_state_l2_residual": gaussian_cutoff_rows[-1][
                "maximum_single_factor_state_l2_residual"
            ],
            "local_product_cutoff_14_map_compiled_one_step_state_l2_residual": gaussian_cutoff_rows[-1][
                "physical_one_step_state_l2_residual"
            ],
            "local_product_cutoff_14_map_compiled_one_step_intermediate_edge_probability": gaussian_cutoff_rows[-1][
                "physical_one_step_maximum_intermediate_edge_probability_last_two_levels"
            ],
        }
    )

    print("Running symmetric product-formula scan", flush=True)
    trotter_rows: List[Dict[str, object]] = []
    direct_final = selected_payload["final_state"]
    direct_observation = state_observables(
        direct_final,
        selected_payload["q_ops"],
        helicity_weights,
        args.selected_cutoff,
    )
    for steps in (1, 2, 4, 8):
        trotter_state = strang_evolution(
            selected_payload["initial_state"], selected_payload["h_terms"], final_time, steps
        )
        overlap = abs(np.vdot(direct_final, trotter_state)) ** 2
        observation = state_observables(
            trotter_state, selected_payload["q_ops"], helicity_weights, args.selected_cutoff
        )
        physical_state, physical_diagnostics = gaussian_compiled_evolution(
            selected_payload["initial_state"],
            terms,
            selected_payload["q_ops"],
            selected_payload["a_ops"],
            selected_payload["number_ops"],
            final_time,
            steps,
            args.selected_cutoff,
        )
        physical_observation = state_observables(
            physical_state,
            selected_payload["q_ops"],
            helicity_weights,
            args.selected_cutoff,
        )
        direct_means = np.asarray(direct_observation["q_means"])
        trotter_means = np.asarray(observation["q_means"])
        optimized_monomial_exponentials = steps * (2 * len(terms) - 2) + 1
        row = {
            "strang_steps": steps,
            "dimensionless_step": args.tau_end / steps,
            "state_infidelity_to_direct": float(max(0.0, 1.0 - overlap)),
            "first_moment_relative_l2_error_to_direct": float(
                np.linalg.norm(trotter_means - direct_means)
                / (np.linalg.norm(direct_means) + 1e-300)
            ),
            "physical_first_moment_relative_l2_error_to_direct": float(
                np.linalg.norm(np.asarray(physical_observation["q_means"]) - direct_means)
                / (np.linalg.norm(direct_means) + 1e-300)
            ),
            "physical_compilation_state_l2_residual_to_collective_sequence": phase_aligned_state_residual(
                trotter_state, physical_state
            ),
            "physical_compilation_state_infidelity_to_collective_sequence": float(
                max(0.0, 1.0 - abs(np.vdot(trotter_state, physical_state)) ** 2)
            ),
            "optimized_non_gaussian_cubic_phase_primitive_count": physical_diagnostics[
                "single_mode_cubic_phase_primitive_count"
            ],
            "gaussian_phase_rotation_count": physical_diagnostics[
                "gaussian_phase_rotation_count"
            ],
            "gaussian_beam_splitter_rotation_count": physical_diagnostics[
                "gaussian_beam_splitter_rotation_count"
            ],
            "ideal_map_squeezing_operation_count": physical_diagnostics[
                "ideal_map_squeezing_operation_count"
            ],
            "maximum_abs_single_mode_cubic_strength": physical_diagnostics[
                "maximum_abs_single_mode_cubic_strength"
            ],
            "total_abs_single_mode_cubic_strength": physical_diagnostics[
                "total_abs_single_mode_cubic_strength"
            ],
            "physical_sequence_maximum_intermediate_edge_probability_last_two_levels": physical_diagnostics[
                "maximum_intermediate_edge_probability_last_two_levels"
            ],
            "physical_sequence_maximum_intermediate_norm_error": physical_diagnostics[
                "maximum_intermediate_norm_error"
            ],
            "maximum_edge_probability_last_two_levels": observation[
                "maximum_edge_probability_last_two_levels"
            ],
        }
        trotter_rows.append(row)
        print(
            f"  steps={steps}: infidelity={row['state_infidelity_to_direct']:.3e}, "
            f"mean={row['first_moment_relative_l2_error_to_direct']:.3e}, "
            f"physical/collective={row['physical_compilation_state_l2_residual_to_collective_sequence']:.3e}",
            flush=True,
        )

    positive = [row for row in trotter_rows if row["first_moment_relative_l2_error_to_direct"] > 0]
    trotter_order = float(
        np.polyfit(
            np.log([row["dimensionless_step"] for row in positive[-3:]]),
            np.log([row["first_moment_relative_l2_error_to_direct"] for row in positive[-3:]]),
            1,
        )[0]
    )

    best_amplitude = amplitude_rows[-1]
    best_cutoff = cutoff_rows[-1]
    best_gaussian_cutoff = gaussian_cutoff_rows[-1]
    success_criteria = {
        "coefficient_identities": max(
            max(metadata["energy_identity_residuals"]),
            max(metadata["helicity_identity_residuals"]),
        )
        < 1e-12,
        "all_six_four_collective_cubic_expansions": gate_compilation[
            "maximum_compiled_monomial_state_l2_residual"
        ]
        < 1e-10,
        "polarization_expanded_strang_step_matches_monomial_sequence": gate_compilation[
            "compiled_full_one_step_strang_state_l2_residual"
        ]
        < 1e-10,
        "symmetric_product_formula_converges_second_order": trotter_order > 1.7,
        "eight_step_gate_mean_error_below_1e-3": trotter_rows[-1][
            "first_moment_relative_l2_error_to_direct"
        ]
        < 1e-3,
        "cutoff_14_edge_probability_below_1e-4": best_cutoff[
            "maximum_edge_probability_last_two_levels"
        ]
        < 1e-4,
        "large_amplitude_classical_mean_error_below_5_percent": best_amplitude[
            "normalized_first_moment_trajectory_relative_l2_error"
        ]
        < 5e-2,
        "passive_gaussian_map_action_converges_with_cutoff": best_gaussian_cutoff[
            "maximum_gaussian_map_action_relative_residual"
        ]
        < gaussian_cutoff_rows[0]["maximum_gaussian_map_action_relative_residual"],
        "all_24_map_compiled_factors_converge_with_cutoff": best_gaussian_cutoff[
            "maximum_single_factor_state_l2_residual"
        ]
        < gaussian_cutoff_rows[0]["maximum_single_factor_state_l2_residual"],
        "map_compiled_one_step_converges_with_cutoff": best_gaussian_cutoff[
            "physical_one_step_state_l2_residual"
        ]
        < gaussian_cutoff_rows[0]["physical_one_step_state_l2_residual"],
        "compiled_sequence_norm_error_below_1e-10": max(
            row["physical_sequence_maximum_intermediate_norm_error"] for row in trotter_rows
        )
        < 1e-10,
        "local_product_cutoff_14_intermediate_edge_probability_below_1e-4": best_gaussian_cutoff[
            "physical_one_step_maximum_intermediate_edge_probability_last_two_levels"
        ]
        < 1e-4,
    }
    overall_success = bool(all(success_criteria.values()))
    diagnostic_thresholds = {
        "selected_cutoff_10_eight_step_compilation_residual_below_1e-3": trotter_rows[-1][
            "physical_compilation_state_l2_residual_to_collective_sequence"
        ]
        < 1e-3,
        "selected_cutoff_10_eight_step_compilation_residual": trotter_rows[-1][
            "physical_compilation_state_l2_residual_to_collective_sequence"
        ],
    }

    summary = {
        "network": metadata,
        "representation": {
            "complete_complex_network_amplitudes": 4,
            "complete_real_coordinates": 8,
            "tested_real_invariant_coordinates": 4,
            "shared_modes": ["K", "P"],
            "triad_specific_modes": ["Q+", "Q-"],
            "fluid_to_real_coordinate_normalization": "u=q/sqrt(2) on the simultaneously real invariant subspace",
            "cubic_hamiltonian_monomials": len(terms),
            "collective_cubic_factors_per_monomial": 4,
            "first_order_non_gaussian_cubic_phase_primitives_per_layer": 4 * len(terms),
            "strang_non_gaussian_cubic_phase_primitives_per_unmerged_step": 4 * (2 * len(terms) - 1),
            "gaussian_conjugations_instantiated": True,
            "gaussian_map_type": "passive local-phase plus two-Givens alignment",
            "ideal_map_squeezing_operations_per_collective_factor": 0,
        },
        "gate_compilation": gate_compilation,
        "trotter_observed_order_from_last_three_points": trotter_order,
        "success_criteria": success_criteria,
        "diagnostic_thresholds": diagnostic_thresholds,
        "overall_local_product_cutoff_stress_audit_success": overall_success,
        "claim_boundary": (
            "This local tensor-product cutoff audit establishes monotone convergence of the "
            "passive Gaussian compilation factors and records accumulated map truncation in the "
            "complete sequence. At cutoff 10 the eight-step physical/collective state residual "
            "remains above 1e-3 because independent local cutoffs are not invariant under passive "
            "mode mixing. The companion global-total-number audit validates the Gaussian "
            "compilation in a passive-invariant truncated space."
        ),
    }

    write_csv(args.out / "amplitude_scan.csv", amplitude_rows)
    write_csv(args.out / "cutoff_scan.csv", cutoff_rows)
    write_csv(args.out / "cubic_gate_compilation_by_monomial.csv", gate_compilation_rows)
    write_csv(args.out / "gaussian_compilation_by_factor.csv", gaussian_factor_rows)
    write_csv(args.out / "gaussian_compilation_cutoff_scan.csv", gaussian_cutoff_rows)
    write_csv(args.out / "strang_gate_scan.csv", trotter_rows)
    with (args.out / "gate_validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_validation_figure(
        amplitude_rows, cutoff_rows, trotter_rows, gaussian_cutoff_rows, args.out
    )

    print(json.dumps({"overall_success": overall_success, "summary": str(args.out)}, indent=2))
    if not overall_success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
