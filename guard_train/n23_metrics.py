from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support

from guard_smoke.constants import N23_CATEGORIES


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _metric_block(rows: list[dict[str, Any]], *, include_per_label: bool) -> dict[str, Any]:
    truth = np.zeros((len(rows), len(N23_CATEGORIES)), dtype=np.int64)
    predicted = np.zeros_like(truth)
    has_probabilities = all(isinstance(row.get("category_probabilities"), dict) for row in rows)
    probabilities = (
        np.zeros((len(rows), len(N23_CATEGORIES)), dtype=np.float64)
        if has_probabilities
        else None
    )

    for row_index, row in enumerate(rows):
        gold = set(row.get("category_gold") or ())
        guessed = set(row.get("category_predictions") or ())
        scores = row.get("category_probabilities") or {}
        for label_index, label in enumerate(N23_CATEGORIES):
            truth[row_index, label_index] = int(label in gold)
            predicted[row_index, label_index] = int(label in guessed)
            if probabilities is not None:
                probabilities[row_index, label_index] = float(scores.get(label, 0.0))

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        truth, predicted, average="micro", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        truth, predicted, average="macro", zero_division=0
    )
    samples_f1 = f1_score(truth, predicted, average="samples", zero_division=0)
    support = truth.sum(axis=0)
    supported = support > 0
    label_precision, label_recall, label_f1, _ = precision_recall_fscore_support(
        truth, predicted, average=None, zero_division=0
    )
    label_auprc: list[float | None] = []
    for label_index in range(len(N23_CATEGORIES)):
        label_auprc.append(
            _finite(average_precision_score(truth[:, label_index], probabilities[:, label_index]))
            if probabilities is not None and support[label_index] > 0
            else None
        )

    result: dict[str, Any] = {
        "supervised_examples": int(len(rows)),
        "labels": int(len(N23_CATEGORIES)),
        "labels_with_support": int(supported.sum()),
        "threshold": 0.5,
        "exact_match_accuracy": float(np.mean(np.all(truth == predicted, axis=1))),
        "hamming_error_rate": float(np.mean(truth != predicted)),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "macro_f1_supported": float(np.mean(label_f1[supported])) if bool(supported.any()) else None,
        "samples_f1": float(samples_f1),
        "gold_positive_labels": int(truth.sum()),
        "predicted_positive_labels": int(predicted.sum()),
    }
    if probabilities is not None:
        result["micro_auprc"] = _finite(
            average_precision_score(truth.ravel(), probabilities.ravel())
        )
        supported_auprc = [value for value in label_auprc if value is not None]
        result["macro_auprc_supported"] = (
            _finite(np.mean(supported_auprc)) if supported_auprc else None
        )
    if include_per_label:
        per_label: dict[str, dict[str, Any]] = {}
        for label_index, label in enumerate(N23_CATEGORIES):
            column_truth = truth[:, label_index]
            column_pred = predicted[:, label_index]
            per_label[label] = {
                "support": int(column_truth.sum()),
                "predicted_positive": int(column_pred.sum()),
                "tp": int(np.logical_and(column_truth == 1, column_pred == 1).sum()),
                "fp": int(np.logical_and(column_truth == 0, column_pred == 1).sum()),
                "fn": int(np.logical_and(column_truth == 1, column_pred == 0).sum()),
                "precision": float(label_precision[label_index]),
                "recall": float(label_recall[label_index]),
                "f1": float(label_f1[label_index]),
                "auprc": label_auprc[label_index],
            }
        result["per_label"] = per_label
    return result


def build_n23_report(predictions: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    supervised = [
        row
        for row in predictions
        if isinstance(row, dict) and isinstance(row.get("category_predictions"), list)
    ]
    if not supervised:
        return None
    report = _metric_block(supervised, include_per_label=True)
    languages = sorted({str(row.get("language") or "unknown") for row in supervised})
    report["slices"] = {
        "language": {
            language: _metric_block(
                [row for row in supervised if str(row.get("language") or "unknown") == language],
                include_per_label=True,
            )
            for language in languages
        }
    }
    return report
