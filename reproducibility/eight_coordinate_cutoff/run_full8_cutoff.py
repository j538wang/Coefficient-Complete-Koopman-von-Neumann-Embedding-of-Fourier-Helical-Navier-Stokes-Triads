#!/usr/bin/env python3
"""Exact total-number cutoff scan for the full eight-real-coordinate KvN model.

The model is the shared-(K,P) two-triad network used in the QST manuscript,
without imposing the invariant restriction that all four imaginary amplitude
components vanish.  Each complex amplitude is written as

    u_j = (x_j + i y_j) / sqrt(2),

and the gauge-real complex equations are expanded into 24 real quadratic
terms.  The corresponding divergence-free KvN generator is projected as a
whole onto the eight-mode total-number Fock space sum(n_j) <= N.  Projecting
the complete cubic monomial, rather than multiplying individually projected
quadratures, preserves Hermiticity at the total-number boundary.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import expm_multiply, norm as sparse_norm


MODE_NAMES = ("K_re", "K_im", "P_re", "P_im", "Qplus_re", "Qplus_im", "Qminus_re", "Qminus_im")
HELICITY_WEIGHTS_COMPLEX = np.array(
    [3.605551275463989, -3.605551275463989, 3.7416573867739413, -3.7416573867739413],
    dtype=float,
)
HELICITY_WEIGHTS_REAL = np.repeat(HELICITY_WEIGHTS_COMPLEX, 2)

# Gauge-real complex-amplitude coefficients divided by sqrt(2), matching the
# manuscript's real-coordinate convention u=(x+i y)/sqrt(2).
PLUS_COEFFICIENTS = (1.6294643037185843, -0.030185620157678034, -1.5992786835609065)
MINUS_COEFFICIENTS = (0.030185620157678034, -1.6294643037185843, 1.599278683560906)

# The magnitudes reproduce the four-coordinate benchmark.  Fixed, non-special
# phases move the initial condition off the all-real invariant subspace.
BASE_AMPLITUDES = np.array([0.80, 0.55, 0.35, -0.30], dtype=float)
INITIAL_PHASES = np.array([0.20, -0.35, 0.45, -0.25], dtype=float)

SOURCE_PROVENANCE = {
    "geometry_id": 745,
    "K": [3, 0, -2],
    "P": [0, -2, -3],
    "Q": [3, 2, 1],
    "source_csv_sha256": "68C38C50CF34986CE598178B0CEB994DA287E5A47C3E86CBFBAF64A129D8519C",
    "gauge_real_complex_coefficients_plus": [
        2.3044105177216543,
        -0.04268891341563096,
        -2.2617216043060235,
    ],
    "gauge_real_complex_coefficients_minus": [
        0.04268891341563096,
        -2.3044105177216543,
        2.261721604306023,
    ],
}


@dataclass(frozen=True)
class RealTerm:
    label: str
    coefficient: float
    target: int
    input_1: int
    input_2: int


def append_complex_triad_terms(
    terms: List[RealTerm], q_re: int, q_im: int, coefficients: Sequence[float], branch: str
) -> None:
    """Expand K'=CK P Q, P'=CP K Q*, Q'=CQ K P* into real terms."""

    ck, cp, cq = coefficients
    specs = (
        ("K_re<-P_re,Q_re", ck, 0, 2, q_re),
        ("K_re<-P_im,Q_im", -ck, 0, 3, q_im),
        ("K_im<-P_re,Q_im", ck, 1, 2, q_im),
        ("K_im<-P_im,Q_re", ck, 1, 3, q_re),
        ("P_re<-K_re,Q_re", cp, 2, 0, q_re),
        ("P_re<-K_im,Q_im", cp, 2, 1, q_im),
        ("P_im<-K_im,Q_re", cp, 3, 1, q_re),
        ("P_im<-K_re,Q_im", -cp, 3, 0, q_im),
        (f"{branch}_re<-K_re,P_re", cq, q_re, 0, 2),
        (f"{branch}_re<-K_im,P_im", cq, q_re, 1, 3),
        (f"{branch}_im<-K_im,P_re", cq, q_im, 1, 2),
        (f"{branch}_im<-K_re,P_im", -cq, q_im, 0, 3),
    )
    terms.extend(RealTerm(f"{branch}:{label}", float(c), target, i, j) for label, c, target, i, j in specs)


def real_terms() -> List[RealTerm]:
    terms: List[RealTerm] = []
    append_complex_triad_terms(terms, 4, 5, PLUS_COEFFICIENTS, "Qplus")
    append_complex_triad_terms(terms, 6, 7, MINUS_COEFFICIENTS, "Qminus")
    return terms


def initial_direction() -> np.ndarray:
    result = np.empty(8, dtype=float)
    result[0::2] = BASE_AMPLITUDES * np.cos(INITIAL_PHASES)
    result[1::2] = BASE_AMPLITUDES * np.sin(INITIAL_PHASES)
    return result


def classical_rhs(_tau: float, state: np.ndarray, terms: Sequence[RealTerm]) -> np.ndarray:
    out = np.zeros(8, dtype=float)
    for term in terms:
        out[term.target] += term.coefficient * state[term.input_1] * state[term.input_2]
    return out


def complex_rhs(state: np.ndarray) -> np.ndarray:
    amplitudes = (state[0::2] + 1j * state[1::2]) / math.sqrt(2.0)
    k, p, qp, qm = amplitudes
    out = np.empty(4, dtype=complex)
    out[0] = math.sqrt(2.0) * (PLUS_COEFFICIENTS[0] * p * qp + MINUS_COEFFICIENTS[0] * p * qm)
    out[1] = math.sqrt(2.0) * (PLUS_COEFFICIENTS[1] * k * np.conjugate(qp) + MINUS_COEFFICIENTS[1] * k * np.conjugate(qm))
    out[2] = math.sqrt(2.0) * PLUS_COEFFICIENTS[2] * k * np.conjugate(p)
    out[3] = math.sqrt(2.0) * MINUS_COEFFICIENTS[2] * k * np.conjugate(p)
    result = np.empty(8, dtype=float)
    result[0::2] = math.sqrt(2.0) * out.real
    result[1::2] = math.sqrt(2.0) * out.imag
    return result


def validate_real_expansion(terms: Sequence[RealTerm]) -> Dict[str, float]:
    rng = np.random.default_rng(20261012)
    expansion_residual = 0.0
    energy_residual = 0.0
    helicity_residual = 0.0
    for _ in range(16):
        state = rng.normal(size=8)
        real = classical_rhs(0.0, state, terms)
        expansion_residual = max(expansion_residual, float(np.max(np.abs(real - complex_rhs(state)))))
        energy_residual = max(energy_residual, abs(float(np.dot(state, real))))
        helicity_residual = max(
            helicity_residual,
            abs(float(np.dot(HELICITY_WEIGHTS_REAL * state, real))),
        )
    return {
        "maximum_complex_to_real_rhs_absolute_residual": expansion_residual,
        "maximum_random_state_energy_derivative_residual": energy_residual,
        "maximum_random_state_helicity_derivative_residual": helicity_residual,
    }


def occupations(total_cutoff: int, modes: int = 8) -> np.ndarray:
    rows: List[Tuple[int, ...]] = []

    def append_compositions(prefix: Tuple[int, ...], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append(prefix + (remaining,))
            return
        for value in range(remaining + 1):
            append_compositions(prefix + (value,), remaining - value, slots - 1)

    for total in range(total_cutoff + 1):
        append_compositions((), total, modes)
    return np.asarray(rows, dtype=np.int16)


def encoded_index(basis: np.ndarray, total_cutoff: int) -> Tuple[np.ndarray, Dict[int, int], np.ndarray]:
    radix = total_cutoff + 1
    strides = np.asarray([radix**mode for mode in range(basis.shape[1])], dtype=np.int64)
    codes = basis.astype(np.int64) @ strides
    return codes, {int(code): i for i, code in enumerate(codes)}, strides


def projected_cubic_term(
    term: RealTerm,
    basis: np.ndarray,
    totals: np.ndarray,
    codes: np.ndarray,
    index: Dict[int, int],
    strides: np.ndarray,
    total_cutoff: int,
) -> csc_matrix:
    dimension = len(basis)
    all_rows: List[np.ndarray] = []
    all_columns: List[np.ndarray] = []
    all_values: List[np.ndarray] = []
    for delta_target, delta_1, delta_2 in itertools.product((-1, 1), repeat=3):
        deltas = np.zeros(8, dtype=np.int16)
        deltas[term.target] = delta_target
        deltas[term.input_1] = delta_1
        deltas[term.input_2] = delta_2
        mask = np.all(basis + deltas >= 0, axis=1)
        mask &= totals + int(np.sum(deltas)) <= total_cutoff
        columns = np.flatnonzero(mask)
        if not len(columns):
            continue
        row_codes = codes[columns] + int(np.dot(deltas.astype(np.int64), strides))
        rows = np.fromiter((index[int(code)] for code in row_codes), dtype=np.int64, count=len(row_codes))
        coefficient = np.ones(len(columns), dtype=complex)
        for mode, delta, is_momentum in (
            (term.target, delta_target, True),
            (term.input_1, delta_1, False),
            (term.input_2, delta_2, False),
        ):
            number = basis[columns, mode].astype(float)
            ladder = np.sqrt(number if delta == -1 else number + 1.0) / math.sqrt(2.0)
            if is_momentum:
                ladder = ladder * (-1j if delta == -1 else 1j)
            coefficient *= ladder
        all_rows.append(rows)
        all_columns.append(columns.astype(np.int64))
        all_values.append(term.coefficient * coefficient)
    return coo_matrix(
        (np.concatenate(all_values), (np.concatenate(all_rows), np.concatenate(all_columns))),
        shape=(dimension, dimension),
        dtype=complex,
    ).tocsc()


def build_hamiltonian(
    terms: Sequence[RealTerm], basis: np.ndarray, total_cutoff: int
) -> Tuple[csc_matrix, np.ndarray, np.ndarray, Dict[int, int], np.ndarray]:
    totals = np.sum(basis, axis=1).astype(np.int16)
    codes, index, strides = encoded_index(basis, total_cutoff)
    hamiltonian = csc_matrix((len(basis), len(basis)), dtype=complex)
    for number, term in enumerate(terms, start=1):
        hamiltonian += projected_cubic_term(term, basis, totals, codes, index, strides, total_cutoff)
        if number % 4 == 0:
            print(f"  assembled {number:02d}/{len(terms)} real cubic monomials; nnz={hamiltonian.nnz}", flush=True)
    hamiltonian.sum_duplicates()
    hamiltonian.eliminate_zeros()
    return hamiltonian, totals, codes, index, strides


def projected_q_operator(
    mode: int,
    basis: np.ndarray,
    totals: np.ndarray,
    codes: np.ndarray,
    index: Dict[int, int],
    strides: np.ndarray,
    total_cutoff: int,
) -> csc_matrix:
    row_parts: List[np.ndarray] = []
    column_parts: List[np.ndarray] = []
    value_parts: List[np.ndarray] = []
    for delta in (-1, 1):
        mask = basis[:, mode] + delta >= 0
        mask &= totals + delta <= total_cutoff
        columns = np.flatnonzero(mask)
        row_codes = codes[columns] + delta * strides[mode]
        rows = np.fromiter((index[int(code)] for code in row_codes), dtype=np.int64, count=len(row_codes))
        number = basis[columns, mode].astype(float)
        values = np.sqrt(number if delta == -1 else number + 1.0) / math.sqrt(2.0)
        row_parts.append(rows)
        column_parts.append(columns.astype(np.int64))
        value_parts.append(values)
    return coo_matrix(
        (np.concatenate(value_parts), (np.concatenate(row_parts), np.concatenate(column_parts))),
        shape=(len(basis), len(basis)),
    ).tocsc()


def projected_coherent_state(q_means: np.ndarray, basis: np.ndarray) -> Tuple[np.ndarray, float]:
    """Project one common untruncated product coherent state and renormalize it."""
    alphas = np.asarray(q_means, dtype=float) / math.sqrt(2.0)
    state = np.full(len(basis), math.exp(-0.5 * float(np.dot(alphas, alphas))), dtype=complex)
    factorials = [math.factorial(n) for n in range(int(np.max(basis)) + 1)]
    for mode, alpha in enumerate(alphas):
        numbers = basis[:, mode]
        state *= np.asarray([alpha**int(n) / math.sqrt(factorials[int(n)]) for n in numbers])
    retained_probability = float(np.vdot(state, state).real)
    return state / math.sqrt(retained_probability), retained_probability


def expectation(state: np.ndarray, operator: csc_matrix) -> complex:
    return np.vdot(state, operator @ state)


def boundary_probability(state: np.ndarray, totals: np.ndarray, total_cutoff: int) -> float:
    return float(np.sum(np.abs(state[totals >= max(0, total_cutoff - 1)]) ** 2))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    records = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def initial_projection_diagnostic(summary: Dict[str, object]) -> Dict[str, object]:
    """Return analytic diagnostics for the projected coherent initial state.

    The total occupation of the untruncated product coherent state is Poisson
    with mean lambda.  Conditioning on N_tot <= N multiplies every position
    first moment by Pr(X <= N-1) / Pr(X <= N).  For the covariance trace, the
    difference between the complete quadratic projection P_N q_a^2 P_N and
    the truncated-coordinate product (P_N q_a P_N)^2 is an exact boundary
    contribution.  Only this trace diagnostic is compared here; the reported
    dynamical covariance continues to use the truncated-coordinate operators.
    """
    cutoff = int(summary["total_number_cutoff"])
    amplitude = float(summary["amplitude"])
    direction = np.asarray(summary["normalized_initial_condition"], dtype=float)
    mode_count = int(summary["mode_count"])
    mean_total_occupation = 0.5 * amplitude**2 * float(np.dot(direction, direction))
    poisson_probabilities = [
        math.exp(-mean_total_occupation)
        * mean_total_occupation**number
        / math.factorial(number)
        for number in range(cutoff + 1)
    ]
    retained_probability = float(sum(poisson_probabilities))
    recorded_retained_probability = float(summary["initial_projection_retained_probability"])
    if abs(retained_probability - recorded_retained_probability) > 5e-13:
        raise RuntimeError(
            "analytic and recorded initial projection probabilities disagree: "
            f"{retained_probability} versus {recorded_retained_probability}"
        )
    exact_boundary_probability = float(poisson_probabilities[-1] / retained_probability)
    first_moment_scale = 1.0 - exact_boundary_probability
    projected_coordinate_trace = float(summary["initial_normalized_covariance_trace"])
    trace_difference = (
        (cutoff + mode_count)
        * exact_boundary_probability
        / (2.0 * amplitude**2)
    )
    return {
        "total_number_cutoff": cutoff,
        "untruncated_initial_mean_total_occupation": mean_total_occupation,
        "initial_projection_retained_probability_analytic": retained_probability,
        "initial_exact_cutoff_boundary_probability": exact_boundary_probability,
        "initial_first_moment_scale_factor": first_moment_scale,
        "initial_relative_first_moment_projection_bias": exact_boundary_probability,
        "initial_normalized_covariance_trace_projected_coordinate_product": projected_coordinate_trace,
        "initial_normalized_covariance_trace_complete_quadratic_projection": projected_coordinate_trace
        + trace_difference,
        "initial_normalized_covariance_trace_convention_absolute_difference": trace_difference,
    }


def run_cutoff(
    total_cutoff: int,
    amplitude: float,
    tau_end: float,
    output_points: int,
    out: Path,
    terms: Sequence[RealTerm],
    validation: Dict[str, float],
) -> Dict[str, object]:
    start = time.perf_counter()
    basis = occupations(total_cutoff)
    expected_dimension = math.comb(total_cutoff + 8, 8)
    if len(basis) != expected_dimension:
        raise RuntimeError(f"basis dimension {len(basis)} != C(N+8,8)={expected_dimension}")
    print(f"N={total_cutoff}: dimension={len(basis)}", flush=True)
    hamiltonian, totals, codes, index, strides = build_hamiltonian(terms, basis, total_cutoff)
    build_seconds = time.perf_counter() - start
    hermiticity = float(sparse_norm(hamiltonian - hamiltonian.getH()) / (sparse_norm(hamiltonian) + 1e-300))
    if hermiticity > 2e-13:
        raise RuntimeError(f"N={total_cutoff} Hamiltonian is not Hermitian: {hermiticity}")
    q_ops = [
        projected_q_operator(mode, basis, totals, codes, index, strides, total_cutoff)
        for mode in range(8)
    ]
    direction = initial_direction()
    initial, initial_projection_probability = projected_coherent_state(amplitude * direction, basis)
    tau_grid = np.linspace(0.0, tau_end, output_points)
    time_grid = tau_grid / amplitude
    classical_result = solve_ivp(
        lambda tau, state: classical_rhs(tau, state, terms),
        (0.0, tau_end),
        direction,
        t_eval=tau_grid,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    if not classical_result.success:
        raise RuntimeError(classical_result.message)
    classical = classical_result.y.T
    evolution_start = time.perf_counter()
    states = expm_multiply(
        -1j * hamiltonian,
        initial,
        start=0.0,
        stop=float(time_grid[-1]),
        num=output_points,
        endpoint=True,
        traceA=0.0,
    )
    evolution_seconds = time.perf_counter() - evolution_start
    trajectory_rows: List[Dict[str, object]] = []
    quantum_means: List[np.ndarray] = []
    normalized_covariances: List[np.ndarray] = []
    boundary_values: List[float] = []
    norm_errors: List[float] = []
    mean_total_occupations: List[float] = []
    covariance_traces: List[float] = []
    covariance_largest_eigenvalues: List[float] = []
    for tau, state, classical_state in zip(tau_grid, states, classical):
        q_states = [operator @ state for operator in q_ops]
        raw_means = np.asarray([np.vdot(state, q_state).real for q_state in q_states])
        means = raw_means / amplitude
        second = np.asarray(
            [[np.vdot(q_i_state, q_j_state).real for q_j_state in q_states] for q_i_state in q_states],
            dtype=float,
        )
        covariance = (second - np.outer(raw_means, raw_means)) / amplitude**2
        covariance = 0.5 * (covariance + covariance.T)
        covariance_trace = float(np.trace(covariance))
        covariance_largest_eigenvalue = float(np.linalg.eigvalsh(covariance)[-1])
        mean_total_occupation = float(np.dot(np.abs(state) ** 2, totals))
        quantum_means.append(means)
        normalized_covariances.append(covariance)
        boundary = boundary_probability(state, totals, total_cutoff)
        norm_error = abs(float(np.linalg.norm(state)) - 1.0)
        boundary_values.append(boundary)
        norm_errors.append(norm_error)
        mean_total_occupations.append(mean_total_occupation)
        covariance_traces.append(covariance_trace)
        covariance_largest_eigenvalues.append(covariance_largest_eigenvalue)
        row: Dict[str, object] = {
            "total_number_cutoff": total_cutoff,
            "tau": float(tau),
            "boundary_probability_last_two_shells": boundary,
            "norm_error": norm_error,
            "mean_total_occupation": mean_total_occupation,
            "normalized_covariance_trace": covariance_trace,
            "normalized_covariance_largest_eigenvalue": covariance_largest_eigenvalue,
            "instantaneous_relative_mean_error_vs_classical": float(
                np.linalg.norm(means - classical_state) / (np.linalg.norm(classical_state) + 1e-300)
            ),
        }
        for name, value in zip(MODE_NAMES, means):
            row[f"quantum_{name}"] = float(value)
        for name, value in zip(MODE_NAMES, classical_state):
            row[f"classical_{name}"] = float(value)
        trajectory_rows.append(row)
    quantum_array = np.asarray(quantum_means)
    classical_energy = 0.5 * np.sum(classical**2, axis=1)
    classical_helicity = 0.5 * np.sum(HELICITY_WEIGHTS_REAL[None, :] * classical**2, axis=1)
    total_seconds = time.perf_counter() - start
    summary: Dict[str, object] = {
        "status": "PASS",
        "scope": "exact projected full-eight-real-coordinate KvN evolution; no Gaussian gate compilation in this scan",
        "total_number_cutoff": total_cutoff,
        "mode_count": 8,
        "hilbert_dimension": len(basis),
        "hamiltonian_monomial_count": len(terms),
        "hamiltonian_nnz": int(hamiltonian.nnz),
        "hamiltonian_hermiticity_relative_residual": hermiticity,
        "amplitude": amplitude,
        "dimensionless_final_time": tau_end,
        "output_points": output_points,
        "normalized_initial_condition": direction.tolist(),
        "complex_mode_base_amplitudes": BASE_AMPLITUDES.tolist(),
        "complex_mode_initial_phases_radians": INITIAL_PHASES.tolist(),
        "initial_quantum_state": "normalized P_N projection of a common untruncated eight-mode product coherent state",
        "initial_coherent_displacements": (amplitude * direction / math.sqrt(2.0)).tolist(),
        "initial_conjugate_momentum_means": [0.0] * 8,
        "initial_state_projected_and_renormalized_separately_for_each_cutoff": True,
        "initial_projection_retained_probability": initial_projection_probability,
        "initial_boundary_probability_last_two_shells": boundary_values[0],
        "maximum_boundary_probability_last_two_shells": max(boundary_values),
        "final_boundary_probability_last_two_shells": boundary_values[-1],
        "maximum_norm_error": max(norm_errors),
        "initial_mean_total_occupation": mean_total_occupations[0],
        "maximum_mean_total_occupation": max(mean_total_occupations),
        "final_mean_total_occupation": mean_total_occupations[-1],
        "initial_normalized_covariance_trace": covariance_traces[0],
        "maximum_normalized_covariance_trace": max(covariance_traces),
        "final_normalized_covariance_trace": covariance_traces[-1],
        "initial_normalized_covariance_largest_eigenvalue": covariance_largest_eigenvalues[0],
        "maximum_normalized_covariance_largest_eigenvalue": max(covariance_largest_eigenvalues),
        "final_normalized_covariance_largest_eigenvalue": covariance_largest_eigenvalues[-1],
        "final_normalized_covariance_matrix": normalized_covariances[-1].tolist(),
        "relative_mean_trajectory_error_vs_classical": float(
            np.linalg.norm(quantum_array - classical) / (np.linalg.norm(classical) + 1e-300)
        ),
        "final_relative_mean_error_vs_classical": float(
            np.linalg.norm(quantum_array[-1] - classical[-1]) / (np.linalg.norm(classical[-1]) + 1e-300)
        ),
        "final_normalized_quantum_means": quantum_array[-1].tolist(),
        "final_classical_state": classical[-1].tolist(),
        "classical_energy_relative_drift": float(
            np.max(np.abs(classical_energy - classical_energy[0])) / (abs(classical_energy[0]) + 1e-300)
        ),
        "classical_helicity_relative_drift": float(
            np.max(np.abs(classical_helicity - classical_helicity[0])) / (abs(classical_helicity[0]) + 1e-300)
        ),
        "build_seconds": build_seconds,
        "evolution_seconds": evolution_seconds,
        "total_seconds": total_seconds,
        "analytic_validation": validation,
        "source_provenance": SOURCE_PROVENANCE,
    }
    write_csv(out / f"N{total_cutoff:03d}_trajectory.csv", trajectory_rows)
    (out / f"N{total_cutoff:03d}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "total_number_cutoff", "hilbert_dimension", "hamiltonian_nnz",
        "maximum_boundary_probability_last_two_shells", "final_relative_mean_error_vs_classical",
        "build_seconds", "evolution_seconds", "total_seconds")}, indent=2), flush=True)
    return summary


def aggregate_existing(out: Path) -> None:
    summaries = []
    for path in sorted(out.glob("N*_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        return
    fields = (
        "total_number_cutoff",
        "hilbert_dimension",
        "hamiltonian_nnz",
        "initial_boundary_probability_last_two_shells",
        "maximum_boundary_probability_last_two_shells",
        "final_boundary_probability_last_two_shells",
        "maximum_norm_error",
        "initial_mean_total_occupation",
        "maximum_mean_total_occupation",
        "final_mean_total_occupation",
        "initial_normalized_covariance_trace",
        "maximum_normalized_covariance_trace",
        "final_normalized_covariance_trace",
        "initial_normalized_covariance_largest_eigenvalue",
        "maximum_normalized_covariance_largest_eigenvalue",
        "final_normalized_covariance_largest_eigenvalue",
        "relative_mean_trajectory_error_vs_classical",
        "final_relative_mean_error_vs_classical",
        "build_seconds",
        "evolution_seconds",
        "total_seconds",
    )
    write_csv(out / "cutoff_summary.csv", ({field: item[field] for field in fields} for item in summaries))
    initial_diagnostics = [initial_projection_diagnostic(item) for item in summaries]
    write_csv(out / "initial_projection_diagnostics.csv", initial_diagnostics)
    (out / "initial_projection_diagnostics.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "scope": "analytic initial-state diagnostic; no time evolution is recomputed",
                "first_moment_formula": "Pr_lambda(X<=N-1)/Pr_lambda(X<=N)",
                "covariance_convention_used_in_scan": "Q_a^(N)=P_N q_a P_N and Gamma_ab=A^-2[<{Q_a^(N),Q_b^(N)}>/2-<Q_a^(N)><Q_b^(N)>]",
                "comparison_convention": "complete quadratic projection P_N {q_a,q_b} P_N/2",
                "covariance_comparison_scope": "initial normalized covariance trace only",
                "rows": initial_diagnostics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    trajectories: Dict[int, np.ndarray] = {}
    for item in summaries:
        cutoff = int(item["total_number_cutoff"])
        with (out / f"N{cutoff:03d}_trajectory.csv").open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        trajectories[cutoff] = np.asarray(
            [[float(record[f"quantum_{name}"]) for name in MODE_NAMES] for record in records],
            dtype=float,
        )
    reference_cutoff = max(trajectories)
    reference = trajectories[reference_cutoff]
    final_covariances = {
        int(item["total_number_cutoff"]): np.asarray(item["final_normalized_covariance_matrix"], dtype=float)
        for item in summaries
    }
    reference_covariance = final_covariances[reference_cutoff]
    convergence_rows: List[Dict[str, object]] = []
    previous_cutoff = None
    for item in summaries:
        cutoff = int(item["total_number_cutoff"])
        trajectory = trajectories[cutoff]
        if trajectory.shape != reference.shape:
            raise RuntimeError("trajectory grids differ across cutoff cases")
        if previous_cutoff is None:
            successive = float("nan")
            successive_final = float("nan")
            successive_covariance = float("nan")
        else:
            previous = trajectories[previous_cutoff]
            successive = float(np.linalg.norm(trajectory - previous) / (np.linalg.norm(trajectory) + 1e-300))
            successive_final = float(
                np.linalg.norm(trajectory[-1] - previous[-1]) / (np.linalg.norm(trajectory[-1]) + 1e-300)
            )
            successive_covariance = float(
                np.linalg.norm(final_covariances[cutoff] - final_covariances[previous_cutoff])
                / (np.linalg.norm(final_covariances[cutoff]) + 1e-300)
            )
        convergence_rows.append(
            {
                "total_number_cutoff": cutoff,
                "reference_cutoff": reference_cutoff,
                "relative_mean_trajectory_difference_vs_reference": float(
                    np.linalg.norm(trajectory - reference) / (np.linalg.norm(reference) + 1e-300)
                ),
                "final_relative_mean_difference_vs_reference": float(
                    np.linalg.norm(trajectory[-1] - reference[-1]) / (np.linalg.norm(reference[-1]) + 1e-300)
                ),
                "previous_cutoff": "" if previous_cutoff is None else previous_cutoff,
                "successive_relative_mean_trajectory_difference": successive,
                "successive_final_relative_mean_difference": successive_final,
                "final_relative_covariance_frobenius_difference_vs_reference": float(
                    np.linalg.norm(final_covariances[cutoff] - reference_covariance)
                    / (np.linalg.norm(reference_covariance) + 1e-300)
                ),
                "successive_relative_covariance_frobenius_difference": successive_covariance,
            }
        )
        previous_cutoff = cutoff
    write_csv(out / "cutoff_convergence.csv", convergence_rows)
    manifest = {
        "status": "COMPLETED_CUTOFFS_ONLY",
        "completed_cutoffs": [item["total_number_cutoff"] for item in summaries],
        "boundary_maximum_scope": "maximum over the nine sampled output times for each cutoff",
        "reference_interpretation": "the largest completed cutoff is a finite numerical reference, not an infinite-cutoff limit",
        "initial_state_construction": "each cutoff is the normalized P_N projection of the same untruncated eight-mode product coherent state",
        "projection_convention": "P_N H P_N with each complete cubic monomial projected only after forming the full product",
        "covariance_convention": "Q_a^(N)=P_N q_a P_N; covariance is formed from anticommutators of these truncated coordinate operators",
        "initial_projection_diagnostics": [
            "initial_projection_diagnostics.csv",
            "initial_projection_diagnostics.json",
        ],
        "projection_verification": "projection_verification.json" if (out / "projection_verification.json").exists() else "not_run",
        "monotonic_maximum_boundary_probability": all(
            later["maximum_boundary_probability_last_two_shells"] < earlier["maximum_boundary_probability_last_two_shells"]
            for earlier, later in zip(summaries, summaries[1:])
        ),
        "monotonic_final_mean_error_vs_classical": all(
            later["final_relative_mean_error_vs_classical"] < earlier["final_relative_mean_error_vs_classical"]
            for earlier, later in zip(summaries, summaries[1:])
        ),
        "reference_cutoff_for_direct_trajectory_comparison": reference_cutoff,
        "summaries": [f"N{item['total_number_cutoff']:03d}_summary.json" for item in summaries],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoffs", default="6,8,10,12")
    parser.add_argument("--amplitude", type=float, default=2.5)
    parser.add_argument("--tau-end", type=float, default=0.18)
    parser.add_argument("--output-points", type=int, default=9)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="refresh aggregate tables and initial-state diagnostics without rerunning time evolution",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    terms = real_terms()
    validation = validate_real_expansion(terms)
    if max(validation.values()) > 2e-12:
        raise RuntimeError(f"analytic validation failed: {validation}")
    metadata = {
        "coordinate_order": list(MODE_NAMES),
        "initial_state_construction": {
            "untruncated_state": "eight-mode product coherent state",
            "coherent_displacements": "q_a(0)/sqrt(2), real",
            "conjugate_momentum_means": 0.0,
            "cutoff_state": "P_N |Psi_infinity(0)> / ||P_N |Psi_infinity(0)>||",
            "same_untruncated_state_for_every_cutoff": True,
        },
        "covariance_convention": {
            "truncated_coordinate_operator": "Q_a^(N)=P_N q_a P_N",
            "reported_covariance": "A^-2[<{Q_a^(N),Q_b^(N)}>/2-<Q_a^(N)><Q_b^(N)>]",
            "not_identical_to": "A^-2[<P_N {q_a,q_b} P_N>/2-<Q_a^(N)><Q_b^(N)>]",
        },
        "real_terms": [asdict(term) for term in terms],
        "analytic_validation": validation,
        "source_provenance": SOURCE_PROVENANCE,
    }
    (args.out / "model_definition.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not args.aggregate_only:
        for cutoff in (int(value) for value in args.cutoffs.split(",") if value.strip()):
            run_cutoff(cutoff, args.amplitude, args.tau_end, args.output_points, args.out, terms, validation)
    aggregate_existing(args.out)


if __name__ == "__main__":
    main()
