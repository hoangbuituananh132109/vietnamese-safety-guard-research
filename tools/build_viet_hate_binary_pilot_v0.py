#!/usr/bin/env python3
"""Build the source manifests for the first Vietnamese binary-safety pilot.

Scope:
  - One domain only: hate / harassment / toxic speech.
  - One model-facing label: safe or unsafe.
  - ViHSD and ViCTSD training splits are seed/training sources.
  - Their original validation/test splits and SEAHateCheck Vietnamese Gold
    annotations are evaluation-only.

This script does not call an LLM and does not claim that source toxicity is
identical to general LLM-prompt safety. Generated instruction-style data will
be stored separately and will carry its own transformation/function label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "external" / "viet_hate_sources"
DEFAULT_OUTPUT = ROOT / "data" / "viet_hate_binary_pilot_v0"


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value or "")
    return re.sub(r"\s+", " ", value).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def make_record(
    *,
    dataset: str,
    split: str,
    row_index: int,
    source_label: str,
    binary_label: str,
    text: str,
    intended_use: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    text = normalize_text(text)
    if not text:
        return None
    digest = sha256_text(text)
    return {
        "id": f"{dataset}:{split}:{row_index}:{digest[:12]}",
        "source_dataset": dataset,
        "source_split": split,
        "source_row_index": row_index,
        "source_label": source_label,
        "binary_label": binary_label,
        "text": text,
        "text_sha256": digest,
        "language": "vi",
        "domain": "hate_harassment_toxicity",
        "human_annotated_source": True,
        "intended_use": intended_use,
        "metadata": metadata or {},
    }


def load_vihsd(split: str, intended_use: str) -> list[dict[str, Any]]:
    path = SOURCE_ROOT / "ViHSD" / "data" / "extracted" / "vihsd" / f"{split}.csv"
    rows = read_csv(path)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source_label = str(row["label_id"]).strip()
        record = make_record(
            dataset="ViHSD",
            split=split,
            row_index=index,
            source_label=source_label,
            binary_label="safe" if source_label == "0" else "unsafe",
            text=row["free_text"],
            intended_use=intended_use,
            metadata={
                "binary_mapping": "label_id=0 -> safe; label_id in {1,2} -> unsafe",
                "original_task": "CLEAN/OFFENSIVE/HATE social-media moderation",
            },
        )
        if record:
            output.append(record)
    return output


def load_victsd(split: str, intended_use: str) -> list[dict[str, Any]]:
    path = SOURCE_ROOT / "ViCTSD" / f"ViCTSD_{split}.csv"
    rows = read_csv(path)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source_label = str(row["Toxicity"]).strip()
        record = make_record(
            dataset="ViCTSD",
            split=split,
            row_index=index,
            source_label=source_label,
            binary_label="unsafe" if source_label == "1" else "safe",
            text=row["Comment"],
            intended_use=intended_use,
            metadata={
                "binary_mapping": "Toxicity=0 -> safe; Toxicity=1 -> unsafe",
                "constructiveness": str(row.get("Constructiveness", "")).strip(),
                "topic": normalize_text(row.get("Topic", "")),
                "title": normalize_text(row.get("Title", "")),
                "original_task": "constructive/toxic social-media moderation",
            },
        )
        if record:
            output.append(record)
    return output


def materialize_sea_template(template: str, target_value: str) -> str:
    """Instantiate identity placeholders in the annotated Gold template."""
    target_value = normalize_text(target_value)
    return normalize_text(re.sub(r"\{[^{}]+\}", target_value, template))


def numeric_zero(value: str) -> bool:
    value = str(value or "").strip()
    if not value:
        return True
    try:
        return float(value) == 0.0
    except ValueError:
        return False


def numeric_at_least(value: str, threshold: float) -> bool:
    try:
        return float(str(value or "").strip()) >= threshold
    except ValueError:
        return False


def load_seahatecheck_gold_annotated() -> tuple[list[dict[str, Any]], Counter[str]]:
    path = (
        SOURCE_ROOT
        / "SEAHateCheck"
        / "dataset"
        / "gold_label_annotated"
        / "vn.csv"
    )
    rows = read_csv(path)
    output: list[dict[str, Any]] = []
    audit: Counter[str] = Counter()

    for index, row in enumerate(rows):
        sentiment = normalize_text(row.get("sentiment_majority", "")).lower()
        audit[f"sentiment:{sentiment or 'missing'}"] += 1

        # Keep only binary, context-independent, natural cases with all three
        # annotations present. The intended label is still preserved separately.
        if sentiment not in {"hateful", "non-hateful"}:
            audit["excluded:non_binary_sentiment"] += 1
            continue
        if not numeric_at_least(row.get("gold_count", ""), 3):
            audit["excluded:fewer_than_3_annotations"] += 1
            continue
        if not numeric_zero(row.get("unnatural_count", "")):
            audit["excluded:unnatural"] += 1
            continue
        if not numeric_zero(row.get("context_count", "")):
            audit["excluded:context_dependent"] += 1
            continue

        text = materialize_sea_template(
            row.get("t_case_local", ""),
            row.get("p_value", ""),
        )
        record = make_record(
            dataset="SEAHateCheck",
            split="gold_annotated_vi",
            row_index=index,
            source_label=sentiment,
            binary_label="unsafe" if sentiment == "hateful" else "safe",
            text=text,
            intended_use="external_evaluation_only",
            metadata={
                "target_category": normalize_text(row.get("p_target", "")),
                "target_value_vi": normalize_text(row.get("p_value", "")),
                "functional_test": normalize_text(row.get("t_function", "")),
                "template_id": normalize_text(row.get("t_id_full", "")),
                "annotation_count": row.get("gold_count", ""),
                "original_task": "binary functional hate-speech testing",
            },
        )
        if record:
            output.append(record)
            audit["accepted"] += 1

    return output, audit


def deduplicate(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    seen: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for record in records:
        digest = record["text_sha256"]
        if digest not in seen:
            seen[digest] = record
            continue
        duplicate_count += 1
        first = seen[digest]
        first.setdefault("duplicate_source_ids", []).append(record["id"])
        if first["binary_label"] != record["binary_label"]:
            first.setdefault("quality_flags", []).append(
                "cross_source_binary_label_conflict"
            )
    return list(seen.values()), duplicate_count


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def stable_sample(
    records: list[dict[str, Any]],
    quotas: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["source_dataset"], record["binary_label"])].append(record)

    selected: list[dict[str, Any]] = []
    for group, quota in quotas.items():
        candidates = sorted(
            grouped[group],
            key=lambda item: hashlib.sha256(
                ("viet-hate-pilot-v0|" + item["id"]).encode("utf-8")
            ).hexdigest(),
        )
        if len(candidates) < quota:
            raise RuntimeError(f"Not enough records for {group}: {len(candidates)} < {quota}")
        selected.extend(candidates[:quota])
    return sorted(selected, key=lambda item: item["id"])


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    return {
        "records": len(records),
        "by_dataset": dict(Counter(r["source_dataset"] for r in records)),
        "by_split": dict(Counter(r["source_split"] for r in records)),
        "by_binary_label": dict(Counter(r["binary_label"] for r in records)),
        "by_dataset_and_label": {
            f"{dataset}:{label}": count
            for (dataset, label), count in sorted(
                Counter(
                    (r["source_dataset"], r["binary_label"]) for r in records
                ).items()
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    # Training/generation pool: never reads source validation/test data.
    train_raw = load_vihsd("train", "generation_seed_or_training") + load_victsd(
        "train", "generation_seed_or_training"
    )
    train_pool, train_duplicates = deduplicate(train_raw)

    # Evaluation pool: source holdouts plus high-confidence SEAHateCheck Gold.
    sea_gold, sea_audit = load_seahatecheck_gold_annotated()
    eval_raw = (
        load_vihsd("dev", "evaluation_only")
        + load_vihsd("test", "evaluation_only")
        + load_victsd("valid", "evaluation_only")
        + load_victsd("test", "evaluation_only")
        + sea_gold
    )
    eval_pool, eval_duplicates = deduplicate(eval_raw)

    # Conflicting source labels are useful review cases but must not silently
    # enter either a training seed or an evaluation score.
    conflict_review = [
        record
        for record in train_pool + eval_pool
        if "cross_source_binary_label_conflict" in record.get("quality_flags", [])
    ]
    train_conflicts_removed = sum(
        "cross_source_binary_label_conflict" in record.get("quality_flags", [])
        for record in train_pool
    )
    eval_conflicts_removed = sum(
        "cross_source_binary_label_conflict" in record.get("quality_flags", [])
        for record in eval_pool
    )
    train_pool = [
        record
        for record in train_pool
        if "cross_source_binary_label_conflict" not in record.get("quality_flags", [])
    ]
    eval_pool = [
        record
        for record in eval_pool
        if "cross_source_binary_label_conflict" not in record.get("quality_flags", [])
    ]

    # Evaluation wins over training when identical normalized text appears in
    # both pools. This prevents exact leakage while preserving source holdouts.
    eval_hashes = {record["text_sha256"] for record in eval_pool}
    cross_split_leakage_removed = sum(
        record["text_sha256"] in eval_hashes for record in train_pool
    )
    train_pool = [
        record for record in train_pool if record["text_sha256"] not in eval_hashes
    ]

    # Small, balanced and deterministic seed set for the first generation run.
    seed_sample = stable_sample(
        train_pool,
        {
            ("ViHSD", "safe"): 100,
            ("ViHSD", "unsafe"): 100,
            ("ViCTSD", "safe"): 50,
            ("ViCTSD", "unsafe"): 50,
        },
    )

    write_jsonl(output_dir / "source_train_pool.jsonl", train_pool)
    write_jsonl(output_dir / "source_eval_pool.jsonl", eval_pool)
    write_jsonl(output_dir / "generation_seed_300.jsonl", seed_sample)
    write_jsonl(output_dir / "label_conflicts_review.jsonl", conflict_review)

    summary = {
        "scope": "Vietnamese binary hate/harassment/toxicity input moderation",
        "target_labels": ["safe", "unsafe"],
        "uncertainty_policy": (
            "Disagreement is stored as a review/QC state, not a third target class."
        ),
        "train_pool": summarize(train_pool),
        "eval_pool": summarize(eval_pool),
        "generation_seed": summarize(seed_sample),
        "deduplication": {
            "train_duplicates_removed": train_duplicates,
            "eval_duplicates_removed": eval_duplicates,
            "train_label_conflicts_removed": train_conflicts_removed,
            "eval_label_conflicts_removed": eval_conflicts_removed,
            "cross_split_exact_leakage_removed_from_train": (
                cross_split_leakage_removed
            ),
        },
        "seahatecheck_gold_filter_audit": dict(sea_audit),
        "leakage_policy": {
            "ViHSD_train_and_ViCTSD_train": "seed/training only",
            "ViHSD_dev_test_and_ViCTSD_valid_test": "evaluation only",
            "SEAHateCheck_gold_annotated_vi": "external evaluation only",
        },
    }
    (output_dir / "source_inventory.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
