"""Audit SEA encoder manifests with the exact local mmBERT tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: list[int]) -> dict[str, int | float]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
        "over_512": int((array > 512).sum()),
        "over_1024": int((array > 1024).sum()),
        "over_2048": int((array > 2048).sum()),
        "over_4096": int((array > 4096).sum()),
        "over_8192": int((array > 8192).sum()),
    }


def audit_manifest(tokenizer, path: Path, batch_size: int) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    batch: list[dict[str, Any]] = []

    def flush() -> None:
        encoded = tokenizer(
            [str(row["text"]) for row in batch],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        for row, raw_length in zip(batch, encoded["length"]):
            value = int(raw_length)
            keys = (
                "overall",
                f"language:{row['language']}",
                f"view:{row['view']}",
                f"subset:{row['subset']}",
                f"label:{row['safety_label']}",
                f"cell:{row['subset']}|{row['language']}|{row['view']}|{row['safety_label']}",
            )
            for key in keys:
                groups[key].append(value)

    for row in iter_jsonl(path):
        batch.append(row)
        if len(batch) >= batch_size:
            flush()
            batch.clear()
    if batch:
        flush()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "distributions": {
            key: distribution(values) for key, values in sorted(groups.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/mmbert_small_base_smoke")
    )
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("data/benchmarks/sea_safeguard/official_en_vi.jsonl"),
    )
    parser.add_argument(
        "--paired",
        type=Path,
        default=Path("data/benchmarks/sea_safeguard/paired_en_vi.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/sea_mmbert_length_audit.json")
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), local_files_only=True, fix_mistral_regex=False
    )
    tokenizer_file = args.model_path / "tokenizer.json"
    report = {
        "model_path": str(args.model_path),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "note": "Exact mmBERT tokens including special tokens; no truncation.",
        "official_en_vi": audit_manifest(tokenizer, args.official, args.batch_size),
        "paired_en_vi": audit_manifest(tokenizer, args.paired, args.batch_size),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    concise = {
        "official": report["official_en_vi"]["distributions"]["overall"],
        "paired": report["paired_en_vi"]["distributions"]["overall"],
        "paired_en": report["paired_en_vi"]["distributions"]["language:en"],
        "paired_vi": report["paired_en_vi"]["distributions"]["language:vi"],
        "output": str(args.output),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
