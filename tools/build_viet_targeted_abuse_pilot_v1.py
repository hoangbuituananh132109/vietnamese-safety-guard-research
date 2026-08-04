#!/usr/bin/env python3
"""Split the exploratory v0 pool into one coherent binary ontology.

Primary task:
  safe   = no targeted abuse/hate
  unsafe = abuse/hate directed at an individual or group

ViHSD CLEAN/HATE and SEAHateCheck non-hateful/hateful align with this task
closely enough for a pilot. ViHSD OFFENSIVE and all ViCTSD toxicity examples
remain available as an auxiliary general-toxicity task and are not silently
merged into the primary binary labels.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
V0_DIR = ROOT / "data" / "viet_hate_binary_pilot_v0"
OUT_DIR = ROOT / "data" / "viet_targeted_abuse_pilot_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def with_ontology(
    record: dict[str, Any],
    *,
    label: str,
    role: str,
    mapping: str,
) -> dict[str, Any]:
    output = dict(record)
    output["binary_label"] = label
    output["ontology"] = "targeted_abuse_hate_v1"
    output["ontology_role"] = role
    output["ontology_mapping"] = mapping
    return output


def stable_sample(
    records: list[dict[str, Any]],
    label: str,
    count: int,
    salt: str,
) -> list[dict[str, Any]]:
    candidates = [record for record in records if record["binary_label"] == label]
    candidates.sort(
        key=lambda record: hashlib.sha256(
            f"{salt}|{record['id']}".encode("utf-8")
        ).hexdigest()
    )
    if len(candidates) < count:
        raise RuntimeError(f"Not enough {label} records: {len(candidates)} < {count}")
    return candidates[:count]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "labels": dict(Counter(record["binary_label"] for record in records)),
        "datasets": dict(Counter(record["source_dataset"] for record in records)),
        "source_labels": dict(
            Counter(
                f"{record['source_dataset']}:{record['source_label']}"
                for record in records
            )
        ),
    }


def main() -> None:
    train_v0 = read_jsonl(V0_DIR / "source_train_pool.jsonl")
    eval_v0 = read_jsonl(V0_DIR / "source_eval_pool.jsonl")

    primary_train: list[dict[str, Any]] = []
    auxiliary_train: list[dict[str, Any]] = []
    for record in train_v0:
        dataset = record["source_dataset"]
        source_label = str(record["source_label"])
        if dataset == "ViHSD" and source_label == "0":
            primary_train.append(
                with_ontology(
                    record,
                    label="safe",
                    role="primary_train",
                    mapping="ViHSD CLEAN(0) -> safe",
                )
            )
        elif dataset == "ViHSD" and source_label == "2":
            primary_train.append(
                with_ontology(
                    record,
                    label="unsafe",
                    role="primary_train",
                    mapping="ViHSD HATE(2) -> unsafe",
                )
            )
        else:
            auxiliary_train.append(
                with_ontology(
                    record,
                    label=record["binary_label"],
                    role="auxiliary_general_toxicity_train",
                    mapping=(
                        "ViHSD OFFENSIVE or ViCTSD Toxicity; out of scope for "
                        "targeted-abuse primary task"
                    ),
                )
            )

    primary_eval: list[dict[str, Any]] = []
    auxiliary_eval: list[dict[str, Any]] = []
    for record in eval_v0:
        dataset = record["source_dataset"]
        source_label = str(record["source_label"])
        if dataset == "ViHSD" and source_label == "0":
            primary_eval.append(
                with_ontology(
                    record,
                    label="safe",
                    role="primary_evaluation",
                    mapping="ViHSD CLEAN(0) -> safe",
                )
            )
        elif dataset == "ViHSD" and source_label == "2":
            primary_eval.append(
                with_ontology(
                    record,
                    label="unsafe",
                    role="primary_evaluation",
                    mapping="ViHSD HATE(2) -> unsafe",
                )
            )
        elif dataset == "SEAHateCheck" and source_label == "non-hateful":
            primary_eval.append(
                with_ontology(
                    record,
                    label="safe",
                    role="external_evaluation",
                    mapping="SEAHateCheck non-hateful -> safe",
                )
            )
        elif dataset == "SEAHateCheck" and source_label == "hateful":
            primary_eval.append(
                with_ontology(
                    record,
                    label="unsafe",
                    role="external_evaluation",
                    mapping="SEAHateCheck hateful -> unsafe",
                )
            )
        else:
            auxiliary_eval.append(
                with_ontology(
                    record,
                    label=record["binary_label"],
                    role="auxiliary_general_toxicity_evaluation",
                    mapping=(
                        "ViHSD OFFENSIVE or ViCTSD Toxicity; reported separately"
                    ),
                )
            )

    generation_seed_300 = sorted(
        stable_sample(primary_train, "safe", 150, "targeted-v1-seed-300")
        + stable_sample(primary_train, "unsafe", 150, "targeted-v1-seed-300"),
        key=lambda record: record["id"],
    )
    smoke_seed_20 = sorted(
        stable_sample(primary_train, "safe", 10, "targeted-v1-smoke-20")
        + stable_sample(primary_train, "unsafe", 10, "targeted-v1-smoke-20"),
        key=lambda record: record["id"],
    )

    write_jsonl(OUT_DIR / "primary_train_pool.jsonl", primary_train)
    write_jsonl(OUT_DIR / "primary_eval_pool.jsonl", primary_eval)
    write_jsonl(OUT_DIR / "auxiliary_general_toxicity_train.jsonl", auxiliary_train)
    write_jsonl(OUT_DIR / "auxiliary_general_toxicity_eval.jsonl", auxiliary_eval)
    write_jsonl(OUT_DIR / "generation_seed_300.jsonl", generation_seed_300)
    write_jsonl(OUT_DIR / "smoke_seed_20.jsonl", smoke_seed_20)

    summary = {
        "ontology": {
            "name": "targeted_abuse_hate_v1",
            "safe": "no targeted abuse/hate",
            "unsafe": "abuse/hate directed at an individual or group",
            "not_a_target_class": (
                "untargeted profanity/general toxicity; retained as auxiliary"
            ),
        },
        "primary_train": summarize(primary_train),
        "primary_eval": summarize(primary_eval),
        "auxiliary_train": summarize(auxiliary_train),
        "auxiliary_eval": summarize(auxiliary_eval),
        "generation_seed_300": summarize(generation_seed_300),
        "smoke_seed_20": summarize(smoke_seed_20),
        "policy": {
            "SEAHateCheck": "external evaluation only",
            "ViCTSD": "auxiliary general-toxicity task only",
            "ViHSD_OFFENSIVE": "auxiliary general-toxicity task only",
            "generated_data": "Silver until human validation",
        },
    }
    (OUT_DIR / "inventory.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
