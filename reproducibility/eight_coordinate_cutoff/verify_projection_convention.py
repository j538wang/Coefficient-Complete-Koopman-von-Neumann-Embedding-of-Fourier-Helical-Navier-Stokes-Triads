#!/usr/bin/env python3
"""Independently verify the total-number projection convention.

For a small cutoff, construct each cubic monomial in a larger tensor-product
oscillator space, restrict the complete product as P_N (p q q) P_N, and compare
it with the production assembler.  Also record the difference from multiplying
three separately projected factors, which inserts intermediate projectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, eye, kron
from scipy.sparse.linalg import norm as sparse_norm

from run_full8_cutoff import (
    build_hamiltonian,
    encoded_index,
    occupations,
    projected_cubic_term,
    real_terms,
)


def local_quadratures(dimension: int) -> tuple[csc_matrix, csc_matrix]:
    annihilation = csc_matrix(
        (
            np.sqrt(np.arange(1, dimension, dtype=float)),
            (np.arange(dimension - 1), np.arange(1, dimension)),
        ),
        shape=(dimension, dimension),
        dtype=complex,
    )
    creation = annihilation.getH()
    q = (annihilation + creation) / np.sqrt(2.0)
    p = 1j * (creation - annihilation) / np.sqrt(2.0)
    return q.tocsc(), p.tocsc()


def tensor_operator(factors: list[csc_matrix]) -> csc_matrix:
    result = factors[0]
    for factor in factors[1:]:
        result = kron(result, factor, format="csc")
    return result


def embedded_factor(mode: int, operator: csc_matrix, identity: csc_matrix) -> csc_matrix:
    return tensor_operator([operator if index == mode else identity for index in range(8)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("projection_verification.json"))
    args = parser.parse_args()

    cutoff = args.cutoff
    local_dimension = cutoff + 2
    basis = occupations(cutoff)
    totals = np.sum(basis, axis=1).astype(np.int16)
    codes, index, strides = encoded_index(basis, cutoff)
    selected = np.ravel_multi_index(tuple(basis.T), (local_dimension,) * 8)

    q_local, p_local = local_quadratures(local_dimension)
    identity = eye(local_dimension, format="csc", dtype=complex)
    embedded_q = [embedded_factor(mode, q_local, identity) for mode in range(8)]
    embedded_p = [embedded_factor(mode, p_local, identity) for mode in range(8)]

    max_complete_absolute = 0.0
    max_complete_relative = 0.0
    max_factor_projected_relative = 0.0
    production_sum = csc_matrix((len(basis), len(basis)), dtype=complex)
    independent_sum = csc_matrix((len(basis), len(basis)), dtype=complex)
    factor_projected_sum = csc_matrix((len(basis), len(basis)), dtype=complex)
    per_term = []

    for term in real_terms():
        production = projected_cubic_term(term, basis, totals, codes, index, strides, cutoff)
        full_product = term.coefficient * (
            embedded_p[term.target] @ embedded_q[term.input_1] @ embedded_q[term.input_2]
        )
        independent = full_product[selected, :][:, selected].tocsc()

        p_projected = embedded_p[term.target][selected, :][:, selected].tocsc()
        q1_projected = embedded_q[term.input_1][selected, :][:, selected].tocsc()
        q2_projected = embedded_q[term.input_2][selected, :][:, selected].tocsc()
        factor_projected = term.coefficient * (p_projected @ q1_projected @ q2_projected)

        complete_absolute = float(sparse_norm(production - independent))
        independent_norm = float(sparse_norm(independent))
        complete_relative = complete_absolute / (independent_norm + 1e-300)
        factor_relative = float(sparse_norm(factor_projected - independent)) / (independent_norm + 1e-300)
        max_complete_absolute = max(max_complete_absolute, complete_absolute)
        max_complete_relative = max(max_complete_relative, complete_relative)
        max_factor_projected_relative = max(max_factor_projected_relative, factor_relative)
        production_sum += production
        independent_sum += independent
        factor_projected_sum += factor_projected
        per_term.append(
            {
                "label": term.label,
                "complete_product_absolute_difference": complete_absolute,
                "complete_product_relative_difference": complete_relative,
                "factor_by_factor_projected_relative_difference": factor_relative,
            }
        )

    production_hamiltonian, *_ = build_hamiltonian(real_terms(), basis, cutoff)
    summed_complete_relative = float(sparse_norm(production_sum - independent_sum)) / (
        float(sparse_norm(independent_sum)) + 1e-300
    )
    build_sum_relative = float(sparse_norm(production_hamiltonian - independent_sum)) / (
        float(sparse_norm(independent_sum)) + 1e-300
    )
    factor_sum_relative = float(sparse_norm(factor_projected_sum - independent_sum)) / (
        float(sparse_norm(independent_sum)) + 1e-300
    )

    result = {
        "status": "PASS" if max_complete_relative < 1e-13 and build_sum_relative < 1e-13 else "FAIL",
        "tested_cutoff": cutoff,
        "total_number_basis_dimension": len(basis),
        "embedding_local_dimension": local_dimension,
        "embedding_tensor_dimension": local_dimension**8,
        "tested_monomials": len(per_term),
        "maximum_complete_product_absolute_difference": max_complete_absolute,
        "maximum_complete_product_relative_difference": max_complete_relative,
        "summed_complete_product_relative_difference": summed_complete_relative,
        "production_build_relative_difference": build_sum_relative,
        "maximum_factor_by_factor_projected_relative_difference": max_factor_projected_relative,
        "summed_factor_by_factor_projected_relative_difference": factor_sum_relative,
        "interpretation": (
            "The production matrices equal P_N(p_a q_b q_c)P_N. "
            "Multiplying P_N p_a P_N, P_N q_b P_N, and P_N q_c P_N is detectably different."
        ),
        "per_term": per_term,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_term"}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
