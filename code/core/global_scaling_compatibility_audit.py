#!/usr/bin/env python3
"""Audit global compatibility of the local positive-rescaling construction.

Each nonsingular closed triad has an oriented real coefficient triple

    (A, -B, -D),  A,B,D > 0.

If one positive scale r_m is assigned to every shared Fourier--helicity mode,
the local construction imposes, with x_m = log(r_m),

    x_1 - x_0 = 1/2 log(A/B),
    x_2 - x_0 = 1/2 log(A/D).

This script assembles all such difference constraints into L x = b, performs a
full least-squares and graph-cycle audit, and emits a minimum-length no-go
certificate if the system is inconsistent.  Wavevectors k and -k are the same
scale node because the real-field convention relates their amplitudes by
complex conjugation and a phase only; helicity labels remain distinct.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


Vector = Tuple[int, int, int]
Mode = Tuple[Vector, int]


def parse_vector(text: str) -> Vector:
    value = ast.literal_eval(text)
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"invalid wavevector: {text!r}")
    return tuple(int(component) for component in value)  # type: ignore[return-value]


def canonical_vector(k: Vector) -> Vector:
    """Identify k and -k with the representative whose first nonzero entry is positive."""
    for component in k:
        if component > 0:
            return k
        if component < 0:
            return tuple(-entry for entry in k)  # type: ignore[return-value]
    raise ValueError("zero wavevector has no Fourier--helicity scale node")


def mode_node(k: Vector, helicity: int) -> Mode:
    if helicity not in (-1, 1):
        raise ValueError(f"invalid helicity: {helicity}")
    return canonical_vector(k), helicity


def mode_text(mode: Mode) -> str:
    k, s = mode
    return f"({k[0]},{k[1]},{k[2]});s={s:+d}"


def squared_norm(k: Vector) -> int:
    return sum(component * component for component in k)


@dataclass(frozen=True)
class Constraint:
    constraint_id: int
    geom_id: int
    K: Vector
    P: Vector
    Q: Vector
    sK: int
    sP: int
    sQ: int
    created_mode: str
    leg: int
    u: Mode
    v: Mode
    delta: float
    A: float
    B: float
    D: float

    @property
    def canonical_pair(self) -> Tuple[Mode, Mode]:
        return tuple(sorted((self.u, self.v)))  # type: ignore[return-value]

    @property
    def canonical_delta(self) -> float:
        """Return x_pair[1] - x_pair[0] in canonical endpoint order."""
        a, b = self.canonical_pair
        return self.delta if (self.u, self.v) == (a, b) else -self.delta

    def to_record(self) -> Dict[str, object]:
        a, b = self.canonical_pair
        return {
            "constraint_id": self.constraint_id,
            "geom_id": self.geom_id,
            "K": str(self.K),
            "P": str(self.P),
            "Q": str(self.Q),
            "sK": self.sK,
            "sP": self.sP,
            "sQ": self.sQ,
            "created_mode": self.created_mode,
            "leg": self.leg,
            "u": mode_text(self.u),
            "v": mode_text(self.v),
            "delta_x_v_minus_x_u": self.delta,
            "canonical_node_a": mode_text(a),
            "canonical_node_b": mode_text(b),
            "canonical_delta_x_b_minus_x_a": self.canonical_delta,
            "A": self.A,
            "B": self.B,
            "D": self.D,
        }


class UnionFind:
    def __init__(self, nodes: Iterable[Mode]):
        self.parent = {node: node for node in nodes}
        self.size = {node: 1 for node in nodes}

    def find(self, node: Mode) -> Mode:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: Mode, right: Mode) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def oriented_nodes(row: pd.Series) -> Tuple[Mode, Mode, Mode]:
    K = mode_node(parse_vector(str(row["K"])), int(row["sK"]))
    P = mode_node(parse_vector(str(row["P"])), int(row["sP"]))
    Q = mode_node(parse_vector(str(row["Q"])), int(row["sQ"]))
    created = str(row["rescale_created_mode"])
    if created == "K":
        return K, P, Q
    if created == "P":
        return P, K, Q
    if created == "Q":
        return Q, K, P
    raise ValueError(f"invalid created-mode label: {created!r}")


def build_constraints(passed: pd.DataFrame) -> List[Constraint]:
    constraints: List[Constraint] = []
    next_id = 0
    for _, row in passed.iterrows():
        n0, n1, n2 = oriented_nodes(row)
        A = float(row["rescale_A"])
        B = float(row["rescale_B"])
        D = float(row["rescale_D"])
        if min(A, B, D) <= 0.0:
            raise RuntimeError(f"nonpositive local coefficient at geom_id={row['geom_id']}")
        common = {
            "geom_id": int(row["geom_id"]),
            "K": parse_vector(str(row["K"])),
            "P": parse_vector(str(row["P"])),
            "Q": parse_vector(str(row["Q"])),
            "sK": int(row["sK"]),
            "sP": int(row["sP"]),
            "sQ": int(row["sQ"]),
            "created_mode": str(row["rescale_created_mode"]),
            "A": A,
            "B": B,
            "D": D,
        }
        constraints.append(Constraint(
            constraint_id=next_id,
            leg=1,
            u=n0,
            v=n1,
            delta=0.5 * math.log(A / B),
            **common,
        ))
        next_id += 1
        constraints.append(Constraint(
            constraint_id=next_id,
            leg=2,
            u=n0,
            v=n2,
            delta=0.5 * math.log(A / D),
            **common,
        ))
        next_id += 1
    return constraints


def connected_components(nodes: Sequence[Mode], constraints: Sequence[Constraint]) -> List[List[Mode]]:
    uf = UnionFind(nodes)
    for edge in constraints:
        uf.union(edge.u, edge.v)
    groups: Dict[Mode, List[Mode]] = defaultdict(list)
    for node in nodes:
        groups[uf.find(node)].append(node)
    return [sorted(group) for group in groups.values()]


def least_squares_audit(nodes: Sequence[Mode],
                        constraints: Sequence[Constraint],
                        components: Sequence[Sequence[Mode]],
                        tol: float) -> Dict[str, object]:
    anchors = {component[0] for component in components}
    free_nodes = [node for node in nodes if node not in anchors]
    column = {node: index for index, node in enumerate(free_nodes)}
    L = np.zeros((len(constraints), len(free_nodes)), dtype=float)
    b = np.empty(len(constraints), dtype=float)
    for row_index, edge in enumerate(constraints):
        if edge.u in column:
            L[row_index, column[edge.u]] -= 1.0
        if edge.v in column:
            L[row_index, column[edge.v]] += 1.0
        b[row_index] = edge.delta
    solution, _reported_residuals, rank, singular_values = np.linalg.lstsq(L, b, rcond=None)
    residual = L @ solution - b
    abs_residual = np.abs(residual)
    expected_rank = len(nodes) - len(components)
    return {
        "rank": int(rank),
        "expected_rank": int(expected_rank),
        "free_unknowns": int(len(free_nodes)),
        "full_column_rank_after_gauge_fixing": bool(rank == expected_rank),
        "b_l2": float(np.linalg.norm(b)),
        "residual_l2": float(np.linalg.norm(residual)),
        "residual_relative_l2": float(np.linalg.norm(residual) / (np.linalg.norm(b) + 1e-300)),
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "residual_max_abs": float(np.max(abs_residual)),
        "constraints_above_tolerance": int(np.sum(abs_residual > tol)),
        "smallest_singular_value": float(singular_values[-1]),
        "largest_singular_value": float(singular_values[0]),
        "matrix_condition_after_gauge_fixing": float(singular_values[0] / singular_values[-1]),
    }


def spanning_cycle_audit(nodes: Sequence[Mode],
                         constraints: Sequence[Constraint],
                         tol: float) -> Dict[str, object]:
    adjacency: Dict[Mode, List[Tuple[Mode, float, int]]] = defaultdict(list)
    for edge in constraints:
        adjacency[edge.u].append((edge.v, edge.delta, edge.constraint_id))
        adjacency[edge.v].append((edge.u, -edge.delta, edge.constraint_id))

    potential: Dict[Mode, float] = {}
    tree_edges = set()
    for root in nodes:
        if root in potential:
            continue
        potential[root] = 0.0
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor, increment, edge_id in adjacency[current]:
                if neighbor in potential:
                    continue
                potential[neighbor] = potential[current] + increment
                tree_edges.add(edge_id)
                queue.append(neighbor)

    closures = np.array([
        potential[edge.v] - potential[edge.u] - edge.delta
        for edge in constraints
    ], dtype=float)
    non_tree = np.array([
        closures[edge.constraint_id]
        for edge in constraints
        if edge.constraint_id not in tree_edges
    ], dtype=float)
    return {
        "spanning_tree_edges": int(len(tree_edges)),
        "non_tree_edges": int(len(constraints) - len(tree_edges)),
        "cycle_closure_max_abs": float(np.max(np.abs(non_tree))) if len(non_tree) else 0.0,
        "cycle_closures_above_tolerance": int(np.sum(np.abs(non_tree) > tol)),
    }


def parallel_pair_audit(constraints: Sequence[Constraint],
                        tol: float) -> Tuple[Dict[str, object], Tuple[Constraint, Constraint] | None]:
    groups: Dict[Tuple[Mode, Mode], List[Constraint]] = defaultdict(list)
    for edge in constraints:
        groups[edge.canonical_pair].append(edge)

    conflicts: List[Tuple[float, Tuple[Mode, Mode], Constraint, Constraint]] = []
    duplicate_pairs = 0
    for pair, edges in groups.items():
        if len(edges) < 2:
            continue
        duplicate_pairs += 1
        ordered = sorted(edges, key=lambda edge: edge.canonical_delta)
        spread = ordered[-1].canonical_delta - ordered[0].canonical_delta
        if spread > tol:
            conflicts.append((spread, pair, ordered[0], ordered[-1]))
    conflicts.sort(key=lambda item: item[0], reverse=True)
    certificate = (conflicts[0][2], conflicts[0][3]) if conflicts else None
    result = {
        "unique_node_pairs": int(len(groups)),
        "node_pairs_with_multiple_constraints": int(duplicate_pairs),
        "inconsistent_parallel_node_pairs": int(len(conflicts)),
        "maximum_parallel_mismatch": float(conflicts[0][0]) if conflicts else 0.0,
        "minimum_inconsistent_cycle_length": 2 if conflicts else None,
    }
    return result, certificate


def certificate_record(first: Constraint,
                       second: Constraint,
                       tol: float) -> Dict[str, object]:
    if first.canonical_pair != second.canonical_pair:
        raise RuntimeError("certificate edges do not share the same endpoint pair")
    mismatch = second.canonical_delta - first.canonical_delta
    if abs(mismatch) <= tol:
        raise RuntimeError("certificate mismatch does not exceed tolerance")
    a, b = first.canonical_pair
    first_scale_ratio = math.exp(first.canonical_delta)
    second_scale_ratio = math.exp(second.canonical_delta)
    record: Dict[str, object] = {
        "cycle_length": 2,
        "minimality_reason": "No nonzero self-loop exists; a two-edge parallel contradiction is therefore a shortest possible inconsistent cycle.",
        "canonical_node_a": mode_text(a),
        "canonical_node_b": mode_text(b),
        "first_constraint_id": first.constraint_id,
        "second_constraint_id": second.constraint_id,
        "first_required_log_ratio_x_b_minus_x_a": first.canonical_delta,
        "second_required_log_ratio_x_b_minus_x_a": second.canonical_delta,
        "log_cycle_closure": mismatch,
        "first_required_ratio_r_b_over_r_a": first_scale_ratio,
        "second_required_ratio_r_b_over_r_a": second_scale_ratio,
        "multiplicative_disagreement_factor": second_scale_ratio / first_scale_ratio,
        "first_edge": first.to_record(),
        "second_edge": second.to_record(),
    }

    # The most robust certificate in the present ensemble uses two equal-norm
    # modes (squared norm n) and a third mode of squared norm q.  In that case
    # the invariant identities yield the exact factor sqrt(q)+sqrt(n).
    if first.geom_id == second.geom_id:
        norm_squares = {
            "K": squared_norm(first.K),
            "P": squared_norm(first.P),
            "Q": squared_norm(first.Q),
        }
        endpoint_norms = {squared_norm(a[0]), squared_norm(b[0])}
        if len(endpoint_norms) == 1:
            n = next(iter(endpoint_norms))
            third_candidates = [value for value in norm_squares.values() if value != n]
            if len(third_candidates) == 1:
                q = third_candidates[0]
                radical = math.sqrt(q) + math.sqrt(n)
                observed = max(first_scale_ratio, second_scale_ratio)
                if math.isclose(observed, radical, rel_tol=5e-13, abs_tol=5e-13):
                    record["analytic_certificate"] = {
                        "equal_endpoint_squared_norm": n,
                        "third_mode_squared_norm": q,
                        "required_factor_symbolic": f"sqrt({q})+sqrt({n})",
                        "required_factor_numeric": radical,
                        "squared_factor_symbolic": f"{q+n}+2*sqrt({q*n})",
                        "contradiction": (
                            f"The two triads require both r_b/r_a=sqrt({q})+sqrt({n}) "
                            f"and r_a/r_b=sqrt({q})+sqrt({n}); multiplying gives "
                            f"1={q+n}+2*sqrt({q*n}), which is false."
                        ),
                    }
    return record


def write_constraints(path: Path, constraints: Sequence[Constraint]) -> None:
    records = [edge.to_record() for edge in constraints]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=Path("unified_ensemble/phase_gauge_ensemble.csv"))
    parser.add_argument("--out", type=Path, default=Path("global_scaling_audit"))
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--expected-pass", type=int, default=6028)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    passed = df[df["rescale_status"] == "pass"].copy()
    singular = df[df["rescale_status"] == "singular"].copy()
    inactive = df[df["rescale_status"] == "inactive"].copy()
    if len(passed) != args.expected_pass:
        raise RuntimeError(f"expected {args.expected_pass} passing local cases, found {len(passed)}")
    if not np.all(passed["rescale_candidate_count"].to_numpy(dtype=int) == 1):
        raise RuntimeError("at least one local case lacks a unique created-mode orientation")

    constraints = build_constraints(passed)
    nodes = sorted({edge.u for edge in constraints} | {edge.v for edge in constraints})
    components = connected_components(nodes, constraints)
    least_squares = least_squares_audit(nodes, constraints, components, args.tol)
    spanning = spanning_cycle_audit(nodes, constraints, args.tol)
    parallel, certificate_edges = parallel_pair_audit(constraints, args.tol)

    compatible = (
        least_squares["residual_max_abs"] <= args.tol
        and spanning["cycle_closure_max_abs"] <= args.tol
        and parallel["inconsistent_parallel_node_pairs"] == 0
    )
    certificate = None
    if not compatible:
        if certificate_edges is None:
            raise RuntimeError(
                "global system is inconsistent, but this implementation did not find "
                "a minimum two-edge certificate"
            )
        certificate = certificate_record(*certificate_edges, tol=args.tol)

    summary = {
        "input_csv": str(args.csv.resolve()),
        "tolerance": args.tol,
        "total_input_rows": int(len(df)),
        "local_pass_rows": int(len(passed)),
        "local_singular_rows": int(len(singular)),
        "local_inactive_rows": int(len(inactive)),
        "global_nodes": int(len(nodes)),
        "global_constraints": int(len(constraints)),
        "connected_components": int(len(components)),
        "component_sizes": sorted((len(component) for component in components), reverse=True),
        "least_squares": least_squares,
        "spanning_cycle_audit": spanning,
        "parallel_pair_audit": parallel,
        "globally_compatible": bool(compatible),
        "decision": (
            "construct_shared_mode_hamiltonian" if compatible
            else "no_go_for_one_positive_scale_per_shared_mode"
        ),
        "claim_boundary": (
            "Incompatibility rules out a global positive diagonal encoding with one "
            "bosonic amplitude scale per shared Fourier--helicity mode. It does not "
            "rule out auxiliary modes, non-diagonal encodings, channels, or KvN dilations."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_constraints(args.out / "global_constraints.csv", constraints)
    (args.out / "global_compatibility_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if certificate is not None:
        (args.out / "no_go_certificate.json").write_text(
            json.dumps(certificate, indent=2), encoding="utf-8"
        )
        pd.DataFrame([
            certificate_edges[0].to_record(),
            certificate_edges[1].to_record(),
        ]).to_csv(args.out / "minimal_inconsistent_cycle.csv", index=False)

    print(json.dumps(summary, indent=2))
    if certificate is not None:
        print("NO-GO CERTIFICATE")
        print(json.dumps(certificate, indent=2))


if __name__ == "__main__":
    main()
