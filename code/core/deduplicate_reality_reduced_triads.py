#!/usr/bin/env python3
"""Select one coefficient triple per reality-reduced closed Fourier triad.

The coefficient scanner represents a closed triad through K=P+Q and therefore
emits the three possible choices of created/output mode.  Each retained row
already contains the complete directed coefficient triple, so global graph
assembly must retain one output orientation per unordered triad and identify
the simultaneous sign reversal imposed by Fourier reality.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Vector = Tuple[int, int, int]
TriadKey = Tuple[Vector, Vector, Vector]


def parse_vector(value: str) -> Vector:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple) or len(parsed) != 3:
        raise ValueError(f"invalid vector: {value}")
    return tuple(int(component) for component in parsed)


def negate(vector: Sequence[int]) -> Vector:
    return tuple(-int(component) for component in vector)


def reality_reduced_key(k_created: Vector, p_input: Vector, q_input: Vector) -> TriadKey:
    closed = tuple(sorted((negate(k_created), p_input, q_input)))
    reversed_closed = tuple(sorted(negate(vector) for vector in closed))
    return min(closed, reversed_closed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    classes: Dict[TriadKey, List[int]] = defaultdict(list)
    seen_geometry: Dict[int, TriadKey] = {}
    with args.csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            geometry_id = int(row["geom_id"])
            if geometry_id in seen_geometry:
                continue
            key = reality_reduced_key(
                parse_vector(row["K"]), parse_vector(row["P"]), parse_vector(row["Q"])
            )
            seen_geometry[geometry_id] = key
            classes[key].append(geometry_id)

    multiplicities = Counter(len(geometry_ids) for geometry_ids in classes.values())
    if multiplicities != Counter({3: len(classes)}):
        raise RuntimeError(f"expected exactly three output orientations per class: {multiplicities}")
    retained = {min(geometry_ids) for geometry_ids in classes.values()}

    output_csv = args.out / "phase_gauge_ensemble.csv"
    status_counts: Counter[str] = Counter()
    rescale_counts: Counter[str] = Counter()
    output_rows = 0
    with args.csv.open(newline="", encoding="utf-8") as source, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise RuntimeError("input CSV has no header")
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row["geom_id"]) not in retained:
                continue
            writer.writerow(row)
            output_rows += 1
            status_counts[row["status"]] += 1
            rescale_counts[row["rescale_status"]] += 1

    expected_rows = 8 * len(classes)
    if output_rows != expected_rows:
        raise RuntimeError(f"expected {expected_rows} retained rows, found {output_rows}")
    summary = {
        "source_csv": "processed/full_kmax3_ensemble/phase_gauge_ensemble.csv",
        "source_created_mode_geometries": len(seen_geometry),
        "created_mode_orientations_per_reality_reduced_class": 3,
        "unique_reality_reduced_closed_triad_geometries": len(classes),
        "retained_helicity_resolved_rows": output_rows,
        "status_counts": dict(status_counts),
        "positive_rescaling_status_counts": dict(rescale_counts),
        "selection_rule": (
            "For each unordered closed triad {-K,P,Q}, identify its simultaneous sign reversal "
            "and retain the smallest source geometry_id among the three created-mode choices."
        ),
    }
    (args.out / "deduplication_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
