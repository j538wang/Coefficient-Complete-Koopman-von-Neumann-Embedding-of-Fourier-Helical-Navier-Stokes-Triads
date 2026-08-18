#!/usr/bin/env python3
"""Build and audit a global coefficient-complete KvN generator.

All active triads in phase_gauge_ensemble.csv, including cases singular under
the positive-rescaling construction, are assembled on a
single graph of shared Fourier--helicity amplitudes.  The real-field convention

    u_s(-k) = -u_s(k)^*

identifies k and -k with one complex graph node.  The resulting complex
quadratic vector field is converted exactly into a sparse real polynomial
F(q) using the physical reality-pair normalization

    u_s(k) = (q_R + i q_I) / sqrt(2).

With this convention the full-field energy is 1/2 sum_a q_a^2.  Its
Hermitian Koopman--von Neumann generator is

    H = 1/2 sum_a (p_a F_a(q) + F_a(q) p_a),

which gives dot(q_a)=i[H,q_a]=F_a(q) as an operator identity (hbar=1).  This
construction uses neither a global positive amplitude rescaling nor a
coherent-state mean-field factorization.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd


Vector = Tuple[int, int, int]
Mode = Tuple[Vector, int]
TermKey = Tuple[int, int, int]  # real output coordinate, ordered factors
CubicKey = Tuple[int, int, int]
SQRT2 = math.sqrt(2.0)


def parse_vector(text: str) -> Vector:
    value = ast.literal_eval(text)
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"invalid wavevector {text!r}")
    return tuple(int(component) for component in value)  # type: ignore[return-value]


def canonical_vector(k: Vector) -> Tuple[Vector, bool]:
    """Return canonical representative and whether the input was reversed."""
    for component in k:
        if component > 0:
            return k, False
        if component < 0:
            return tuple(-entry for entry in k), True  # type: ignore[return-value]
    raise ValueError("zero wavevector is not a Fourier--helicity mode")


def mode_node(k: Vector, helicity: int) -> Mode:
    canonical, _reversed = canonical_vector(k)
    return canonical, int(helicity)


def mode_text(mode: Mode) -> str:
    k, s = mode
    return f"({k[0]},{k[1]},{k[2]});s={s:+d}"


def complex_coefficient(row: pd.Series, prefix: str) -> complex:
    return complex(float(row[f"{prefix}_real"]), float(row[f"{prefix}_imag"]))


def amplitude_linear_form(k: Vector,
                          helicity: int,
                          conjugated: bool,
                          node_index: Mapping[Mode, int]) -> List[Tuple[int, complex]]:
    """Return the global real-coordinate linear form for u_s(k) or its conjugate."""
    canonical, reversed_wavevector = canonical_vector(k)
    node = (canonical, int(helicity))
    base = node_index[node]
    sign = -1.0 if reversed_wavevector else 1.0
    effective_conjugation = bool(conjugated) ^ bool(reversed_wavevector)
    y_coefficient = -1j if effective_conjugation else 1j
    return [
        (2 * base, complex(sign / SQRT2, 0.0)),
        (2 * base + 1, sign * y_coefficient / SQRT2),
    ]


def amplitude_value(k: Vector,
                    helicity: int,
                    conjugated: bool,
                    z: Sequence[complex],
                    node_index: Mapping[Mode, int]) -> complex:
    canonical, reversed_wavevector = canonical_vector(k)
    value = complex(z[node_index[(canonical, int(helicity))]])
    if reversed_wavevector:
        value = -np.conjugate(value)
    return np.conjugate(value) if conjugated else value


def route_target_coefficient(k: Vector, coefficient: complex) -> complex:
    """Map dot u_s(k) coefficient to dot z of the canonical k/-k node."""
    _canonical, reversed_wavevector = canonical_vector(k)
    return -np.conjugate(coefficient) if reversed_wavevector else coefficient


def add_directed_equation(terms: MutableMapping[TermKey, float],
                          target_k: Vector,
                          target_s: int,
                          coefficient: complex,
                          factor_1: Tuple[Vector, int, bool],
                          factor_2: Tuple[Vector, int, bool],
                          node_index: Mapping[Mode, int]) -> None:
    target_node = node_index[mode_node(target_k, target_s)]
    _canonical_target, reversed_target = canonical_vector(target_k)
    left = amplitude_linear_form(*factor_1, node_index=node_index)
    right = amplitude_linear_form(*factor_2, node_index=node_index)
    for i, ci in left:
        for j, cj in right:
            a, b = sorted((i, j))
            complex_value = coefficient * ci * cj
            # If u_s(k)=-z_s(-k)^*, then dot(z_s(-k))=-dot(u_s(k))^*.
            # Conjugation acts on the complete monomial, not only on C.
            if reversed_target:
                complex_value = -np.conjugate(complex_value)
            # qdot_R+i qdot_I = sqrt(2) zdot for
            # z=(q_R+i q_I)/sqrt(2).
            complex_value *= SQRT2
            terms[(2 * target_node, a, b)] += float(complex_value.real)
            terms[(2 * target_node + 1, a, b)] += float(complex_value.imag)


def assemble_terms(passed: pd.DataFrame,
                   node_index: Mapping[Mode, int],
                   drop_tol: float) -> Dict[TermKey, float]:
    terms: Dict[TermKey, float] = defaultdict(float)
    for _, row in passed.iterrows():
        K, P, Q = (parse_vector(str(row[name])) for name in ("K", "P", "Q"))
        sK, sP, sQ = (int(row[name]) for name in ("sK", "sP", "sQ"))
        CK, CP, CQ = (complex_coefficient(row, name) for name in ("CK", "CP", "CQ"))
        add_directed_equation(
            terms, K, sK, CK,
            (P, sP, False), (Q, sQ, False), node_index,
        )
        add_directed_equation(
            terms, P, sP, CP,
            (K, sK, False), (Q, sQ, True), node_index,
        )
        add_directed_equation(
            terms, Q, sQ, CQ,
            (K, sK, False), (P, sP, True), node_index,
        )
    return {key: value for key, value in terms.items() if abs(value) > drop_tol}


def direct_complex_rhs(passed: pd.DataFrame,
                       z: Sequence[complex],
                       node_index: Mapping[Mode, int]) -> np.ndarray:
    out = np.zeros(len(node_index), dtype=complex)
    for _, row in passed.iterrows():
        K, P, Q = (parse_vector(str(row[name])) for name in ("K", "P", "Q"))
        sK, sP, sQ = (int(row[name]) for name in ("sK", "sP", "sQ"))
        CK, CP, CQ = (complex_coefficient(row, name) for name in ("CK", "CP", "CQ"))
        uK = amplitude_value(K, sK, False, z, node_index)
        uP = amplitude_value(P, sP, False, z, node_index)
        uQ = amplitude_value(Q, sQ, False, z, node_index)
        equations = [
            (K, sK, CK * uP * uQ),
            (P, sP, CP * uK * np.conjugate(uQ)),
            (Q, sQ, CQ * uK * np.conjugate(uP)),
        ]
        for target_k, target_s, rhs in equations:
            target = node_index[mode_node(target_k, target_s)]
            out[target] += route_target_coefficient(target_k, rhs)
    return out


def complex_to_real(z: Sequence[complex]) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    q = np.empty(2 * len(z), dtype=float)
    q[0::2] = SQRT2 * z.real
    q[1::2] = SQRT2 * z.imag
    return q


def evaluate_terms(terms: Mapping[TermKey, float], q: Sequence[float], dimension: int) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    out = np.zeros(dimension, dtype=float)
    for (output, i, j), value in terms.items():
        out[output] += value * q[i] * q[j]
    return out


def divergence_coefficients(terms: Mapping[TermKey, float], dimension: int) -> np.ndarray:
    result = np.zeros(dimension, dtype=float)
    for (output, i, j), value in terms.items():
        if i == output and j == output:
            result[output] += 2.0 * value
        elif i == output:
            result[j] += value
        elif j == output:
            result[i] += value
    return result


def invariant_derivative(terms: Mapping[TermKey, float],
                         coordinate_weights: Sequence[float]) -> Dict[CubicKey, float]:
    result: Dict[CubicKey, float] = defaultdict(float)
    for (output, i, j), value in terms.items():
        key = tuple(sorted((output, i, j)))
        result[key] += float(coordinate_weights[output]) * value
    return dict(result)


def polynomial_stats(polynomial: Mapping[Tuple[int, ...], float], tol: float) -> Dict[str, object]:
    values = np.array(list(polynomial.values()), dtype=float)
    if len(values) == 0:
        return {"term_count": 0, "max_abs": 0.0, "l2": 0.0, "above_tolerance": 0}
    return {
        "term_count": int(len(values)),
        "max_abs": float(np.max(np.abs(values))),
        "l2": float(np.linalg.norm(values)),
        "above_tolerance": int(np.sum(np.abs(values) > tol)),
    }


def write_terms(path: Path,
                terms: Mapping[TermKey, float],
                nodes: Sequence[Mode]) -> None:
    def coordinate_name(index: int) -> str:
        mode = mode_text(nodes[index // 2])
        component = "x" if index % 2 == 0 else "y"
        return f"{component}[{mode}]"

    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "momentum_coordinate", "factor_1", "factor_2", "real_coefficient",
            "momentum_index", "factor_1_index", "factor_2_index",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (output, i, j), value in sorted(terms.items()):
            writer.writerow({
                "momentum_coordinate": f"p[{coordinate_name(output)}]",
                "factor_1": coordinate_name(i),
                "factor_2": coordinate_name(j),
                "real_coefficient": value,
                "momentum_index": output,
                "factor_1_index": i,
                "factor_2_index": j,
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=Path("unified_ensemble/phase_gauge_ensemble.csv"))
    parser.add_argument("--out", type=Path, default=Path("kvn_global_graph_audit"))
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--drop-tol", type=float, default=1e-14)
    parser.add_argument("--expected-active", type=int, default=6388)
    parser.add_argument("--expected-nodes", type=int, default=252)
    parser.add_argument(
        "--reality-reduced-full-cube",
        action="store_true",
        help="mark the input as one representative of each full-cube reality pair",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    active = df[df["status"] == "active"].copy()
    if len(active) != args.expected_active:
        raise RuntimeError(f"expected {args.expected_active} active rows, found {len(active)}")

    node_set = set()
    for _, row in active.iterrows():
        for vector_name, helicity_name in (("K", "sK"), ("P", "sP"), ("Q", "sQ")):
            node_set.add(mode_node(parse_vector(str(row[vector_name])), int(row[helicity_name])))
    nodes = sorted(node_set)
    node_index = {node: index for index, node in enumerate(nodes)}
    if len(nodes) != args.expected_nodes:
        raise RuntimeError(
            f"expected {args.expected_nodes} shared complex nodes, found {len(nodes)}"
        )

    terms = assemble_terms(active, node_index, args.drop_tol)
    real_dimension = 2 * len(nodes)
    divergence = divergence_coefficients(terms, real_dimension)
    energy_weights = np.ones(real_dimension, dtype=float)
    helicity_weights = np.empty(real_dimension, dtype=float)
    for index, (k, s) in enumerate(nodes):
        kappa = s * math.sqrt(sum(component * component for component in k))
        helicity_weights[2 * index:2 * index + 2] = kappa
    energy_polynomial = invariant_derivative(terms, energy_weights)
    helicity_polynomial = invariant_derivative(terms, helicity_weights)
    energy_stats = polynomial_stats(energy_polynomial, args.tol)
    helicity_stats = polynomial_stats(helicity_polynomial, args.tol)

    rng = np.random.default_rng(20260803)
    reconstruction_errors = []
    for _ in range(3):
        z = 0.15 * (rng.standard_normal(len(nodes)) + 1j * rng.standard_normal(len(nodes)))
        direct = complex_to_real(direct_complex_rhs(active, z, node_index))
        reconstructed = evaluate_terms(terms, complex_to_real(z), real_dimension)
        reconstruction_errors.append(
            float(np.linalg.norm(reconstructed - direct) / (np.linalg.norm(direct) + 1e-300))
        )

    summary = {
        "input_csv": str(args.csv.resolve()),
        "source_total_rows": int(len(df)),
        "source_active_triads": int(len(active)),
        "source_nonsingular_triads": int(np.sum(active["rescale_status"] == "pass")),
        "source_singular_triads": int(np.sum(active["rescale_status"] == "singular")),
        "shared_complex_fourier_helicity_nodes": int(len(nodes)),
        "real_kvn_coordinate_count": int(real_dimension),
        "kvn_canonical_q_p_pairs": int(real_dimension),
        "complex_to_real_normalization": "u_s(k)=(q_R+i q_I)/sqrt(2) with one representative of each {k,-k} pair",
        "hermitian_generator": "H=1/2 sum_a (p_a F_a(q)+F_a(q) p_a)",
        "sparse_quadratic_terms_after_combination": int(len(terms)),
        "maximum_abs_generator_coefficient": float(max(abs(value) for value in terms.values())),
        "uses_global_positive_rescaling": False,
        "uses_coherent_state_mean_field_factorization": False,
        "source_is_reality_reduced_full_cube": bool(args.reality_reduced_full_cube),
        "full_lattice_equivalent_helicity_triads": (
            int(2 * len(df)) if args.reality_reduced_full_cube else None
        ),
        "verification": {
            "generic_global_state_reconstruction_relative_errors": reconstruction_errors,
            "maximum_global_reconstruction_relative_error": max(reconstruction_errors),
            "divergence_linear_coefficient_max_abs": float(np.max(np.abs(divergence))),
            "energy_commutator_polynomial": energy_stats,
            "helicity_commutator_polynomial": helicity_stats,
            "tolerance": args.tol,
        },
        "operator_statement": (
            "With [q_a,p_b]=i delta_ab, the Heisenberg identity i[H,q_a]=F_a(q) "
            f"holds term by term for the assembled {len(nodes)}-node coefficient graph."
        ),
        "first_moment_boundary": (
            "For a general state, d<q_a>/dt=<F_a(q)> rather than F_a(<q>). "
            "Thus the embedding is exact at operator/distribution level but does not "
            "make nonlinear first moments close for arbitrary density operators."
        ),
    }

    if max(reconstruction_errors) > args.tol:
        raise RuntimeError("global sparse polynomial does not reconstruct the coefficient graph")
    if float(np.max(np.abs(divergence))) > args.tol:
        raise RuntimeError("assembled inviscid vector field is not divergence free")
    if energy_stats["max_abs"] > args.tol:
        raise RuntimeError("global energy commutator polynomial failed")
    if helicity_stats["max_abs"] > args.tol:
        raise RuntimeError("global helicity commutator polynomial failed")

    args.out.mkdir(parents=True, exist_ok=True)
    write_terms(args.out / "global_kvn_generator_terms.csv", terms, nodes)
    (args.out / "global_kvn_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.out / "global_kvn_nodes.csv").write_text(
        "node_index,wavevector,helicity\n" + "\n".join(
            f'{index},"{node[0]}",{node[1]}' for index, node in enumerate(nodes)
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
