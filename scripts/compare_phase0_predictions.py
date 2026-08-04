from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired semantic comparison of two Phase-0 prediction artifacts."
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    return parser.parse_args()


def load(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row["example_id"])
            if example_id in rows:
                raise ValueError(f"Duplicate example_id in {path}:{line_number}: {example_id}")
            rows[example_id] = row
    return rows


def summarize(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    if not pairs:
        return {"examples": 0}
    same = [left["prediction"] == right["prediction"] for left, right in pairs]
    gaps = [
        abs(float(left["unsafe_probability"]) - float(right["unsafe_probability"]))
        for left, right in pairs
    ]
    left_correct = [left["prediction"] == left["target"] for left, _ in pairs]
    right_correct = [right["prediction"] == right["target"] for _, right in pairs]
    return {
        "examples": len(pairs),
        "same_semantic_decision": int(sum(same)),
        "same_semantic_decision_rate": float(np.mean(same)),
        "mean_absolute_probability_gap": float(np.mean(gaps)),
        "p95_absolute_probability_gap": float(np.quantile(gaps, 0.95)),
        "max_absolute_probability_gap": float(max(gaps)),
        "both_correct": int(sum(a and b for a, b in zip(left_correct, right_correct))),
        "both_wrong": int(sum(not a and not b for a, b in zip(left_correct, right_correct))),
        "left_correct_right_wrong": int(
            sum(a and not b for a, b in zip(left_correct, right_correct))
        ),
        "left_wrong_right_correct": int(
            sum(not a and b for a, b in zip(left_correct, right_correct))
        ),
    }


def main() -> None:
    args = parse_args()
    left = load(args.left)
    right = load(args.right)
    if set(left) != set(right):
        raise ValueError(
            f"Prediction ID sets differ: left_only={len(set(left) - set(right))}, "
            f"right_only={len(set(right) - set(left))}"
        )
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for example_id in sorted(left):
        a, b = left[example_id], right[example_id]
        for key in ("target", "target_label", "record_uid", "view", "language"):
            if a.get(key) != b.get(key):
                raise ValueError(
                    f"Artifact mismatch for {example_id} field {key}: {a.get(key)} != {b.get(key)}"
                )
        pairs.append((a, b))

    slices: dict[str, Any] = {}
    for dimension in ("view", "language", "tag", "length_bucket"):
        grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for pair in pairs:
            grouped[str(pair[0].get(dimension, "unknown"))].append(pair)
        slices[dimension] = {
            key: summarize(value) for key, value in sorted(grouped.items())
        }

    report = {
        "left": {"name": args.left_name, "path": str(args.left)},
        "right": {"name": args.right_name, "path": str(args.right)},
        "overall": summarize(pairs),
        "slices": slices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
