"""Audit and evaluate base GLiGuard on a paired SEA Safeguard sample.

The run uses only the binary safety schema (safe/unsafe), performs no text
truncation, and only scores EN-VI pairs for which both language instances fit
within the configured total encoder length. All length/eligibility decisions are
written at prediction level for later reproduction on a rental GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from guard_smoke.gliguard import _parse_binary_result
from guard_smoke.constants import TEXT_SAFETY_TASK


LABEL_TO_ID = {"safe": 0, "unsafe": 1}
SAFETY_LABELS = ["safe", "unsafe"]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0}
    array = np.asarray(values)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
        "over_512": int((array > 512).sum()),
        "over_1024": int((array > 1024).sum()),
        "over_2048": int((array > 2048).sum()),
    }


def slice_key(row: dict[str, Any], dimensions: tuple[str, ...]) -> str:
    return "|".join(str(row[name]) for name in dimensions)


def summarize_lengths(rows: list[dict[str, Any]], max_total_tokens: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "examples": len(rows),
        "eligible_examples": sum(row["eligible"] for row in rows),
        "coverage": float(np.mean([row["eligible"] for row in rows])) if rows else math.nan,
        "total_encoder_tokens": distribution([row["total_encoder_tokens"] for row in rows]),
        "schema_items": distribution([row["schema_items"] for row in rows]),
        "text_words_untruncated": distribution([row["text_words_untruncated"] for row in rows]),
        "max_total_tokens": max_total_tokens,
        "truncation": False,
        "slices": {},
    }
    for dimensions in (
        ("language",),
        ("view",),
        ("subset",),
        ("language", "view"),
        ("subset", "language", "view"),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[slice_key(row, dimensions)].append(row)
        summary["slices"]["|".join(dimensions)] = {
            key: {
                "examples": len(items),
                "eligible_examples": sum(item["eligible"] for item in items),
                "coverage": float(np.mean([item["eligible"] for item in items])),
                "total_encoder_tokens": distribution(
                    [item["total_encoder_tokens"] for item in items]
                ),
            }
            for key, items in sorted(grouped.items())
        }
    return summary


def audit_lengths(
    model: Any,
    examples: list[dict[str, Any]],
    max_total_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schema = (
        model.create_schema()
        .classification(TEXT_SAFETY_TASK, SAFETY_LABELS)
        .build()
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, example in enumerate(examples, 1):
        scope = str(example["safety_scope"])
        batch = model.processor.collate_fn_inference(
            [(str(example["text"]), schema)],
            max_len=None,
        )
        total = int(batch.original_lengths[0])
        rows.append(
            {
                "example_id": example["example_id"],
                "pair_uid": example["pair_uid"],
                "subset": example["subset"],
                "language": example["language"],
                "view": example["view"],
                "safety_scope": scope,
                "safety_label": example["safety_label"],
                "official_registered": example["official_registered"],
                "text_words_untruncated": int(batch.text_word_counts[0]),
                # GLiNER2 exposes the number of schema items here, not the
                # number of schema subwords. Eligibility always uses the exact
                # total encoder length in original_lengths.
                "schema_items": int(batch.schema_counts[0]),
                "total_encoder_tokens": total,
                "eligible": total <= max_total_tokens,
                "truncated": False,
            }
        )
        if index % 500 == 0:
            print(f"length_audit={index}/{len(examples)}", flush=True)
    elapsed = time.perf_counter() - started
    summary = summarize_lengths(rows, max_total_tokens)
    summary["audit_seconds"] = elapsed
    summary["examples_per_second"] = len(rows) / elapsed
    return rows, summary


def choose_paired_sample(
    examples: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    pairs_per_cell: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {row["example_id"]: row for row in examples}
    audit_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        audit_by_pair[str(row["pair_uid"])].append(row)

    eligible_pairs: dict[str, list[dict[str, Any]]] = {}
    pair_rejections = Counter()
    for pair_uid, pair in audit_by_pair.items():
        languages = {row["language"] for row in pair}
        if len(pair) != 2 or languages != {"en", "vi"}:
            pair_rejections["not_complete_en_vi_pair"] += 1
            continue
        if not all(row["eligible"] for row in pair):
            pair_rejections["one_or_both_languages_over_limit"] += 1
            continue
        eligible_pairs[pair_uid] = pair

    cells: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for pair_uid, pair in eligible_pairs.items():
        representative = pair[0]
        labels = {row["safety_label"] for row in pair}
        if len(labels) != 1:
            raise ValueError(f"Target mismatch in pair {pair_uid}")
        cells[
            (
                str(representative["subset"]),
                str(representative["view"]),
                str(representative["safety_label"]),
            )
        ].append(pair_uid)

    selected_pair_ids: list[str] = []
    cell_summary: dict[str, Any] = {}
    for cell, pair_ids in sorted(cells.items()):
        ranked = sorted(
            pair_ids,
            key=lambda pair_uid: hashlib.sha256(
                f"{seed}:{pair_uid}".encode("utf-8")
            ).hexdigest(),
        )
        selected = ranked[:pairs_per_cell]
        selected_pair_ids.extend(selected)
        cell_summary["|".join(cell)] = {
            "eligible_pairs": len(pair_ids),
            "selected_pairs": len(selected),
        }

    selected_examples: list[dict[str, Any]] = []
    for pair_uid in sorted(set(selected_pair_ids)):
        pair_audit = sorted(eligible_pairs[pair_uid], key=lambda row: row["language"])
        for audit in pair_audit:
            source = dict(by_id[audit["example_id"]])
            source.update(
                {
                    "total_encoder_tokens": audit["total_encoder_tokens"],
                    "schema_items": audit["schema_items"],
                    "text_words_untruncated": audit["text_words_untruncated"],
                    "eligible": True,
                    "truncated": False,
                }
            )
            selected_examples.append(source)
    return selected_examples, {
        "pairs_per_cell_requested": pairs_per_cell,
        "eligible_pair_units": len(eligible_pairs),
        "selected_pair_units": len(set(selected_pair_ids)),
        "selected_language_instances": len(selected_examples),
        "pair_rejections": dict(sorted(pair_rejections.items())),
        "cells": cell_summary,
        "seed": seed,
    }


def binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = np.asarray([row["target"] for row in rows], dtype=int)
    predictions = np.asarray([row["prediction"] for row in rows], dtype=int)
    probabilities = np.asarray([row["unsafe_probability"] for row in rows], dtype=float)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "examples": len(rows),
        "safe_support": int(support[0]),
        "unsafe_support": int(support[1]),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "safe_precision": float(precision[0]),
        "safe_recall": float(recall[0]),
        "safe_f1": float(f1[0]),
        "unsafe_precision": float(precision[1]),
        "unsafe_recall": float(recall[1]),
        "unsafe_f1": float(f1[1]),
        "confusion_matrix_safe_unsafe": confusion_matrix(
            targets, predictions, labels=[0, 1]
        ).tolist(),
        "brier_unsafe": float(brier_score_loss(targets, probabilities)),
    }
    result["auprc_unsafe"] = (
        float(average_precision_score(targets, probabilities))
        if len(set(targets.tolist())) == 2
        else math.nan
    )
    result["auroc"] = (
        float(roc_auc_score(targets, probabilities))
        if len(set(targets.tolist())) == 2
        else math.nan
    )
    return result


@torch.inference_mode()
def evaluate(
    model: Any,
    examples: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    all_started = time.perf_counter()
    started = time.perf_counter()
    results = model.batch_classify_text(
        [str(row["text"]) for row in examples],
        {TEXT_SAFETY_TASK: SAFETY_LABELS},
        batch_size=batch_size,
        include_confidence=True,
        max_len=None,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if len(results) != len(examples):
        raise RuntimeError(f"Returned {len(results)} results for {len(examples)} rows")
    amortized_latency_ms = 1000.0 * elapsed / len(examples)
    for example, result in zip(examples, results):
        prediction, unsafe_probability, label, confidence = _parse_binary_result(
            result, TEXT_SAFETY_TASK
        )
        predictions.append(
            {
                "example_id": example["example_id"],
                "pair_uid": example["pair_uid"],
                "subset": example["subset"],
                "language": example["language"],
                "view": example["view"],
                "safety_scope": example["safety_scope"],
                "official_registered": example["official_registered"],
                "gold_label": example["safety_label"],
                "target": LABEL_TO_ID[example["safety_label"]],
                "prediction": prediction,
                "predicted_label": label,
                "predicted_confidence": confidence,
                "unsafe_probability": unsafe_probability,
                "total_encoder_tokens": example["total_encoder_tokens"],
                "schema_items": example["schema_items"],
                "text_words_untruncated": example["text_words_untruncated"],
                "eligible": True,
                "truncated": False,
                "latency_ms_per_example_amortized": amortized_latency_ms,
            }
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - all_started
    predictions.sort(key=lambda row: row["example_id"])

    metrics: dict[str, Any] = {
        "overall": binary_metrics(predictions),
        "slices": {},
        "inference": {
            "seconds": total_seconds,
            "examples_per_second": len(predictions) / total_seconds,
            "batch_size": batch_size,
            "peak_vram_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            ),
        },
    }
    for dimensions in (
        ("language",),
        ("view",),
        ("subset",),
        ("language", "view"),
        ("subset", "language", "view"),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            grouped[slice_key(row, dimensions)].append(row)
        metrics["slices"]["|".join(dimensions)] = {
            key: binary_metrics(items) for key, items in sorted(grouped.items())
        }

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        pairs[str(row["pair_uid"])].append(row)
    complete = [
        pair
        for pair in pairs.values()
        if len(pair) == 2 and {row["language"] for row in pair} == {"en", "vi"}
    ]
    metrics["paired_en_vi"] = {
        "pairs": len(complete),
        "same_decision_rate": float(
            np.mean([pair[0]["prediction"] == pair[1]["prediction"] for pair in complete])
        ),
        "mean_absolute_probability_gap": float(
            np.mean(
                [
                    abs(pair[0]["unsafe_probability"] - pair[1]["unsafe_probability"])
                    for pair in complete
                ]
            )
        ),
    }
    return predictions, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/fastino_gliguard_300m")
    )
    parser.add_argument(
        "--paired-manifest",
        type=Path,
        default=Path("data/benchmarks/sea_safeguard/paired_en_vi.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/guard_sea/gliguard_base_paired_sample"),
    )
    parser.add_argument("--max-total-tokens", type=int, default=512)
    parser.add_argument("--pairs-per-cell", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    from gliner2 import GLiNER2

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = list(iter_jsonl(args.paired_manifest))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(
        f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else None}",
        flush=True,
    )
    print(f"loading_model={args.model_path}", flush=True)
    load_started = time.perf_counter()
    model = GLiNER2.from_pretrained(str(args.model_path)).to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_started
    print(f"model_loaded_seconds={load_seconds:.3f}", flush=True)

    audit_rows, length_summary = audit_lengths(
        model, examples, args.max_total_tokens
    )
    write_jsonl(args.output_dir / "length_audit.jsonl", audit_rows)
    (args.output_dir / "length_summary.json").write_text(
        json.dumps(length_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "coverage="
        f"{length_summary['eligible_examples']}/{length_summary['examples']} "
        f"({100.0 * length_summary['coverage']:.2f}%)",
        flush=True,
    )

    sample, sample_summary = choose_paired_sample(
        examples,
        audit_rows,
        pairs_per_cell=args.pairs_per_cell,
        seed=args.seed,
    )
    write_jsonl(args.output_dir / "selected_sample.jsonl", sample)
    print(
        f"selected_pairs={sample_summary['selected_pair_units']} "
        f"instances={sample_summary['selected_language_instances']}",
        flush=True,
    )
    predictions, evaluation = evaluate(model, sample, args.batch_size, device)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)

    archive = ROOT / "SEA-HELM-main.zip"
    model_file = args.model_path / "model.safetensors"
    report = {
        "run_id": "B0-GLI-ZS-SEA-PAIRED-SAMPLE",
        "model": {
            "path": str(args.model_path),
            "model_safetensors_sha256": sha256_file(model_file),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "load_seconds": load_seconds,
        },
        "data": {
            "paired_manifest": str(args.paired_manifest),
            "paired_manifest_sha256": sha256_file(args.paired_manifest),
            "sea_archive_sha256": sha256_file(archive) if archive.exists() else None,
        },
        "schema": {
            "task": TEXT_SAFETY_TASK,
            "task_type": "single_label",
            "labels": SAFETY_LABELS,
            "separate_prompt_response_heads": False,
            "category_labels": [],
        },
        "eligibility": {
            "max_total_encoder_tokens": args.max_total_tokens,
            "truncation": False,
            "length_summary": length_summary,
        },
        "sampling": sample_summary,
        "evaluation": evaluation,
        "artifacts": {
            "length_audit": str(args.output_dir / "length_audit.jsonl"),
            "selected_sample": str(args.output_dir / "selected_sample.jsonl"),
            "predictions": str(args.output_dir / "predictions.jsonl"),
        },
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    concise = {
        "coverage": {
            "eligible": length_summary["eligible_examples"],
            "total": length_summary["examples"],
            "ratio": length_summary["coverage"],
        },
        "sampling": sample_summary,
        "overall": evaluation["overall"],
        "paired_en_vi": evaluation["paired_en_vi"],
        "inference": evaluation["inference"],
        "metrics": str(metrics_path),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
