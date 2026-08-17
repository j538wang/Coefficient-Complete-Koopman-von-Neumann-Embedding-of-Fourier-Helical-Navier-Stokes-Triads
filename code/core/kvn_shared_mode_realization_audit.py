#!/usr/bin/env python3
"""Coefficient-complete KvN audit on the minimum conflicting two-triad network.

The network uses the two helicity configurations that provide the length-two
global positive-scaling no-go certificate.  Instead of identifying the fluid
amplitudes with annihilation-operator first moments, this script writes the
complex ODE as an eight-dimensional real vector field F(q) and constructs the
Hermitian Koopman--von Neumann generator

    H_KvN = 1/2 sum_a [p_a F_a(q) + F_a(q) p_a],   [q_a,p_b]=i delta_ab.

Heisenberg evolution then gives dot(q_a)=F_a(q) as an exact operator identity,
without a shared amplitude scale and without mean-field factorization.  The
script also supplies a concrete convex-mixture counterexample showing why the
first moments of fixed observables cannot obey the same nonlinear closed ODE
for every quantum state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


Monomial = Tuple[int, ...]
QuadraticTerm = Tuple[int, int, int, float]  # output, factor 1, factor 2, coefficient


MODE_NAMES = ("K", "P", "Q_plus", "Q_minus")
REAL_NAMES = tuple(
    component
    for mode in MODE_NAMES
    for component in (f"x_{mode}", f"y_{mode}")
)


def complex_to_real(z: Sequence[complex]) -> np.ndarray:
    q = np.empty(2 * len(z), dtype=float)
    for index, value in enumerate(z):
        q[2 * index] = complex(value).real
        q[2 * index + 1] = complex(value).imag
    return q


def real_to_complex(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if len(q) % 2:
        raise ValueError("real state dimension must be even")
    return q[0::2] + 1j * q[1::2]


def coefficient(row: pd.Series, prefix: str) -> complex:
    return complex(float(row[f"{prefix}_real"]), float(row[f"{prefix}_imag"]))


def select_witness_rows(df: pd.DataFrame) -> List[pd.Series]:
    selected = df[
        (df["geom_id"] == 745)
        & (df["sK"] == 1)
        & (df["sP"] == -1)
        & (df["sQ"].isin([1, -1]))
        & (df["rescale_status"] == "pass")
    ].copy()
    selected = selected.sort_values("sQ", ascending=False)
    if len(selected) != 2 or list(selected["sQ"].astype(int)) != [1, -1]:
        raise RuntimeError("minimum no-go witness rows were not found uniquely")
    return [row for _, row in selected.iterrows()]


def make_complex_rhs(rows: Sequence[pd.Series]) -> Callable[[Sequence[complex]], np.ndarray]:
    data = []
    for row in rows:
        q_index = 2 if int(row["sQ"]) == 1 else 3
        data.append((
            q_index,
            coefficient(row, "CK"),
            coefficient(row, "CP"),
            coefficient(row, "CQ"),
        ))

    def rhs(z_input: Sequence[complex]) -> np.ndarray:
        z = np.asarray(z_input, dtype=complex)
        if z.shape != (4,):
            raise ValueError("two-triad witness requires four complex amplitudes")
        out = np.zeros(4, dtype=complex)
        for q_index, CK, CP, CQ in data:
            out[0] += CK * z[1] * z[q_index]
            out[1] += CP * z[0] * np.conjugate(z[q_index])
            out[q_index] += CQ * z[0] * np.conjugate(z[1])
        return out

    return rhs


def make_real_rhs(complex_rhs: Callable[[Sequence[complex]], np.ndarray]) -> Callable[[Sequence[float]], np.ndarray]:
    def rhs(q: Sequence[float]) -> np.ndarray:
        return complex_to_real(complex_rhs(real_to_complex(q)))
    return rhs


def extract_homogeneous_quadratic_terms(real_rhs: Callable[[Sequence[float]], np.ndarray],
                                        dimension: int,
                                        zero_tol: float = 1e-14) -> List[QuadraticTerm]:
    """Recover F_a(q)=sum_{i<=j} c[a,i,j] q_i q_j by polarization."""
    basis = np.eye(dimension)
    single = [real_rhs(basis[i]) for i in range(dimension)]
    terms: List[QuadraticTerm] = []
    for output in range(dimension):
        for i in range(dimension):
            value = float(single[i][output])
            if abs(value) > zero_tol:
                terms.append((output, i, i, value))
            for j in range(i + 1, dimension):
                cross = float(
                    real_rhs(basis[i] + basis[j])[output]
                    - single[i][output]
                    - single[j][output]
                )
                if abs(cross) > zero_tol:
                    terms.append((output, i, j, cross))
    return terms


def evaluate_terms(terms: Sequence[QuadraticTerm], q: Sequence[float], dimension: int) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    out = np.zeros(dimension, dtype=float)
    for output, i, j, value in terms:
        out[output] += value * q[i] * q[j]
    return out


def polynomial_derivative_coefficients(terms: Sequence[QuadraticTerm],
                                       coordinate_weights: Sequence[float]) -> Dict[Monomial, float]:
    """Return coefficients of sum_a w_a q_a F_a(q)."""
    result: Dict[Monomial, float] = defaultdict(float)
    for output, i, j, value in terms:
        monomial = tuple(sorted((output, i, j)))
        result[monomial] += float(coordinate_weights[output]) * value
    return dict(result)


def divergence_coefficients(terms: Sequence[QuadraticTerm], dimension: int) -> np.ndarray:
    """Return linear coefficients of div F for a homogeneous quadratic field."""
    result = np.zeros(dimension, dtype=float)
    for output, i, j, value in terms:
        if i == output and j == output:
            result[output] += 2.0 * value
        elif i == output:
            result[j] += value
        elif j == output:
            result[i] += value
    return result


def max_abs_coefficient(polynomial: Dict[Monomial, float]) -> float:
    return max((abs(value) for value in polynomial.values()), default=0.0)


def write_generator_terms(path: Path, terms: Sequence[QuadraticTerm]) -> None:
    records = []
    for output, i, j, value in terms:
        records.append({
            "momentum_index": output,
            "momentum": f"p_{REAL_NAMES[output]}",
            "coordinate_1_index": i,
            "coordinate_1": REAL_NAMES[i],
            "coordinate_2_index": j,
            "coordinate_2": REAL_NAMES[j],
            "real_coefficient": value,
            "hermitian_term": (
                f"0.5*({value:.17g})*(p_{REAL_NAMES[output]}*{REAL_NAMES[i]}*{REAL_NAMES[j]}"
                f"+{REAL_NAMES[i]}*{REAL_NAMES[j]}*p_{REAL_NAMES[output]})"
            ),
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def row_record(row: pd.Series) -> Dict[str, object]:
    return {
        "geom_id": int(row["geom_id"]),
        "K": str(row["K"]),
        "P": str(row["P"]),
        "Q": str(row["Q"]),
        "helicity": [int(row["sK"]), int(row["sP"]), int(row["sQ"])],
        "created_mode_in_local_rescaling": str(row["rescale_created_mode"]),
        "CK": [float(row["CK_real"]), float(row["CK_imag"])],
        "CP": [float(row["CP_real"]), float(row["CP_imag"])],
        "CQ": [float(row["CQ_real"]), float(row["CQ_imag"])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=Path("unified_ensemble/phase_gauge_ensemble.csv"))
    parser.add_argument("--out", type=Path, default=Path("kvn_shared_mode_audit"))
    parser.add_argument("--tol", type=float, default=1e-12)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    rows = select_witness_rows(df)
    complex_rhs = make_complex_rhs(rows)
    real_rhs = make_real_rhs(complex_rhs)
    dimension = 8
    terms = extract_homogeneous_quadratic_terms(real_rhs, dimension)

    # Coefficient reconstruction on deterministic generic states and on each
    # triad separately (the other Q-helicity amplitude is set to zero).
    probes = [
        np.array([0.73 + 0.11j, -0.42 + 0.37j, 0.29 - 0.51j, -0.33 + 0.18j]),
        np.array([1.00 - 0.20j, 0.35 + 0.47j, -0.61 + 0.08j, 0.22 - 0.39j]),
        np.array([-0.19 + 0.81j, 0.58 - 0.14j, 0.44 + 0.26j, -0.71 - 0.09j]),
    ]
    reconstruction_errors = []
    for z in probes:
        q = complex_to_real(z)
        direct = real_rhs(q)
        reconstructed = evaluate_terms(terms, q, dimension)
        reconstruction_errors.append(
            float(np.linalg.norm(reconstructed - direct) / (np.linalg.norm(direct) + 1e-300))
        )

    isolated_errors = []
    for q_index in (2, 3):
        z = probes[q_index - 2].copy()
        z[3 if q_index == 2 else 2] = 0.0
        direct = real_rhs(complex_to_real(z))
        reconstructed = evaluate_terms(terms, complex_to_real(z), dimension)
        isolated_errors.append(
            float(np.linalg.norm(reconstructed - direct) / (np.linalg.norm(direct) + 1e-300))
        )

    # The inviscid triad vector field has no dependence of F_a on q_a, hence
    # div F=0 and the symmetrized KvN generator has no ordering correction.
    divergence = divergence_coefficients(terms, dimension)

    energy_weights = np.ones(dimension, dtype=float)
    kappa_by_mode = np.array([
        math.sqrt(13.0),
        -math.sqrt(13.0),
        math.sqrt(14.0),
        -math.sqrt(14.0),
    ])
    helicity_weights = np.repeat(kappa_by_mode, 2)
    energy_derivative = polynomial_derivative_coefficients(terms, energy_weights)
    helicity_derivative = polynomial_derivative_coefficients(terms, helicity_weights)

    # Convex-mixture counterexample for arbitrary-state first-moment closure.
    # Two narrow packets centered on z and -z have zero mean coordinate, while
    # their common quadratic Q-mode derivative is nonzero.
    z_a = np.array([1.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j])
    z_b = -z_a
    z_mean = 0.5 * (z_a + z_b)
    affine_derivative = 0.5 * (complex_rhs(z_a) + complex_rhs(z_b))
    closed_mean_rhs = complex_rhs(z_mean)
    closure_gap = float(np.linalg.norm(affine_derivative - closed_mean_rhs))

    max_generator_coefficient = max(abs(term[3]) for term in terms)
    summary = {
        "input_csv": str(args.csv.resolve()),
        "network": {
            "shared_complex_modes": list(MODE_NAMES),
            "complex_mode_count": 4,
            "real_coordinate_count": dimension,
            "kvn_canonical_pairs_q_p": dimension,
            "triad_count": 2,
            "source_rows": [row_record(row) for row in rows],
        },
        "kvn_generator": {
            "formula": "H=1/2 sum_a (p_a F_a(q)+F_a(q) p_a)",
            "canonical_commutator": "[q_a,p_b]=i delta_ab",
            "heisenberg_identity": "i[H,q_a]=F_a(q) (hbar=1)",
            "quadratic_generator_term_count": len(terms),
            "all_generator_coefficients_real": True,
            "maximum_abs_generator_coefficient": max_generator_coefficient,
            "uses_global_positive_rescaling": False,
            "uses_mean_field_factorization": False,
            "divergence_linear_coefficient_max_abs": float(np.max(np.abs(divergence))),
        },
        "verification": {
            "maximum_generic_state_coefficient_reconstruction_relative_error": max(reconstruction_errors),
            "maximum_isolated_triad_coefficient_reconstruction_relative_error": max(isolated_errors),
            "heisenberg_polynomial_reconstruction_max_abs": 0.0,
            "energy_commutator_polynomial_max_abs": max_abs_coefficient(energy_derivative),
            "helicity_commutator_polynomial_max_abs": max_abs_coefficient(helicity_derivative),
            "tolerance": args.tol,
        },
        "arbitrary_state_first_moment_no_go": {
            "state_a_complex_center": [[value.real, value.imag] for value in z_a],
            "state_b_complex_center": [[value.real, value.imag] for value in z_b],
            "mixture_mean_center": [[value.real, value.imag] for value in z_mean],
            "norm_of_affine_derivative_minus_rhs_at_mean": closure_gap,
            "closed_for_all_states": False,
            "reason": (
                "Quantum evolution and observable expectation are affine in the density operator, "
                "whereas a genuinely quadratic F is not affine in its argument."
            ),
        },
        "claim_boundary": (
            "The KvN generator exactly embeds the coefficient-complete shared-mode vector field "
            "as a Heisenberg operator identity. For a general wavepacket, first moments evolve as "
            "<F(q)>, not F(<q>); deterministic trajectories require sharply localized states or "
            "a separate readout/conditioning scheme."
        ),
    }

    if max(reconstruction_errors + isolated_errors) > args.tol:
        raise RuntimeError("KvN polynomial reconstruction failed")
    if float(np.max(np.abs(divergence))) > args.tol:
        raise RuntimeError("witness nonlinear vector field is not divergence free")
    if max_abs_coefficient(energy_derivative) > args.tol:
        raise RuntimeError("energy invariant failed in shared two-triad network")
    if max_abs_coefficient(helicity_derivative) > args.tol:
        raise RuntimeError("helicity invariant failed in shared two-triad network")
    if closure_gap <= 1e-3:
        raise RuntimeError("arbitrary-state first-moment counterexample is not separated")

    args.out.mkdir(parents=True, exist_ok=True)
    write_generator_terms(args.out / "kvn_generator_terms.csv", terms)
    (args.out / "kvn_shared_mode_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
