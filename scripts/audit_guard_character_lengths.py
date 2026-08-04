"""Audit source-record and materialized guard character lengths."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guard_smoke.data import FINAL_FILENAMES, iter_full_examples, iter_jsonl


THRESHOLDS = (512, 1000, 2000, 3000, 5000, 10000, 25000)


def distribution(values: Iterable[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    def quantile(proportion: float) -> float:
        position = (len(ordered) - 1) * proportion
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    result: dict[str, int | float] = {
        "count": len(ordered),
        "min": ordered[0],
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": ordered[-1],
    }
    for threshold in THRESHOLDS:
        result[f"over_{threshold}"] = sum(value > threshold for value in ordered)
    return result


def append(groups: dict[str, list[int]], keys: Iterable[str], value: int) -> None:
    for key in keys:
        groups[key].append(value)


def source_record_lengths(final_dir: Path) -> dict[str, dict[str, int | float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for split, filename in FINAL_FILENAMES.items():
        for row in iter_jsonl(final_dir / filename):
            tag = str(row.get("tag") or "unknown")
            for language in ("en", "vi"):
                prompt = str(row.get(f"prompt_{language}") or "")
                response = row.get(f"response_{language}")
                value = len(prompt) + len(str(response or ""))
                append(
                    groups,
                    (
                        "overall",
                        f"split:{split}",
                        f"language:{language}",
                        f"tag:{tag}",
                        f"split_language:{split}|{language}",
                        f"split_tag:{split}|{tag}",
                        f"cell:{split}|{language}|{tag}",
                    ),
                    value,
                )
    return {key: distribution(values) for key, values in sorted(groups.items())}


def materialized_lengths(final_dir: Path) -> dict[str, dict[str, int | float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for split in FINAL_FILENAMES:
        for example in iter_full_examples(final_dir, split):
            value = len(example.text)
            append(
                groups,
                (
                    "overall",
                    f"split:{split}",
                    f"language:{example.language}",
                    f"view:{example.view}",
                    f"tag:{example.tag}",
                    f"split_language:{split}|{example.language}",
                    f"split_view:{split}|{example.view}",
                    f"split_tag:{split}|{example.tag}",
                    f"split_language_view:{split}|{example.language}|{example.view}",
                    f"cell:{split}|{example.view}|{example.language}|{example.tag}",
                ),
                value,
            )
    return {key: distribution(values) for key, values in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", type=Path, default=Path("data/final"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/guard_character_length_audit.json")
    )
    args = parser.parse_args()
    report = {
        "definition": {
            "source_record": "len(prompt) + len(response or '') for each language",
            "materialized_instance": "len(serialized GuardExample.text) for P/R/PR",
            "unit": "Unicode code points, not tokenizer tokens",
            "thresholds": list(THRESHOLDS),
        },
        "source_records": source_record_lengths(args.final_dir),
        "materialized_instances": materialized_lengths(args.final_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    concise = {
        "source_train_en": report["source_records"]["split_language:train|en"],
        "source_train_vi": report["source_records"]["split_language:train|vi"],
        "materialized_train": report["materialized_instances"]["split:train"],
        "output": str(args.output),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
