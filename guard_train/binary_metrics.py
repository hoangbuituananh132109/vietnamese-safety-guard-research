from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def length_bucket(length: int) -> str:
    if length <= 128:
        return "0001-0128"
    if length <= 256:
        return "0129-0256"
    if length <= 384:
        return "0257-0384"
    if length <= 512:
        return "0385-0512"
    if length <= 1024:
        return "0513-1024"
    if length <= 2048:
        return "1025-2048"
    if length <= 4096:
        return "2049-4096"
    if length <= 8192:
        return "4097-8192"
    return "8193+"


def expected_calibration_error(
    targets: Sequence[int], unsafe_probabilities: Sequence[float], bins: int = 10
) -> float | None:
    if not targets:
        return None
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(unsafe_probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        mask = (
            (p >= edges[index]) & (p <= edges[index + 1])
            if index == bins - 1
            else (p >= edges[index]) & (p < edges[index + 1])
        )
        count = int(mask.sum())
        if count:
            value += count / len(y) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(value)


def binary_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    targets = [int(row["target"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    probabilities = [float(row["unsafe_probability"]) for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=[0, 1], average=None, zero_division=0
    )
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    has_both = len(set(targets)) == 2
    positives = sum(targets)
    return {
        "examples": len(rows),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, labels=[0, 1], average="macro", zero_division=0)
        ),
        "safe_precision": float(precision[0]),
        "safe_recall": float(recall[0]),
        "safe_f1": float(f1[0]),
        "safe_support": int(support[0]),
        "unsafe_precision": float(precision[1]),
        "unsafe_recall": float(recall[1]),
        "unsafe_f1": float(f1[1]),
        "unsafe_support": int(support[1]),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc_auc": float(roc_auc_score(targets, probabilities)) if has_both else None,
        "average_precision": (
            float(average_precision_score(targets, probabilities)) if positives else None
        ),
        "brier_score": float(
            np.mean((np.asarray(probabilities) - np.asarray(targets)) ** 2)
        ),
        "ece_10": expected_calibration_error(targets, probabilities, bins=10),
    }


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"overall": binary_metrics(rows), "slices": {}}
    for dimension in ("scope", "view", "language", "tag", "length_bucket"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[dimension])].append(row)
        report["slices"][dimension] = {
            key: binary_metrics(value) for key, value in sorted(groups.items())
        }

    paired: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(row["record_uid"], row["view"])][row["language"]] = row
    complete = [value for value in paired.values() if set(value) == {"en", "vi"}]
    report["paired_en_vi"] = {
        "pairs": len(complete),
        "same_decision_rate": (
            float(
                np.mean(
                    [pair["en"]["prediction"] == pair["vi"]["prediction"] for pair in complete]
                )
            )
            if complete
            else None
        ),
        "mean_probability_gap": (
            float(
                np.mean(
                    [
                        abs(
                            pair["en"]["unsafe_probability"]
                            - pair["vi"]["unsafe_probability"]
                        )
                        for pair in complete
                    ]
                )
            )
            if complete
            else None
        ),
    }
    return report
