from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guard_smoke.data import (
    build_balanced_smoke_examples,
    write_gliner_jsonl,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", type=Path, default=Path("data/final"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/guard_smoke"))
    parser.add_argument("--train-per-cell", type=int, default=4)
    parser.add_argument("--valid-per-cell", type=int, default=2)
    parser.add_argument("--test-per-cell", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"seed": args.seed, "splits": {}}
    per_split = {
        "train": args.train_per_cell,
        "valid": args.valid_per_cell,
        "test": args.test_per_cell,
    }
    for split, per_cell in per_split.items():
        examples = build_balanced_smoke_examples(
            args.final_dir, split, per_cell, seed=args.seed
        )
        manifest_path = args.output_dir / f"{split}.jsonl"
        gliner_path = args.output_dir / f"{split}_gliner.jsonl"
        write_manifest(examples, manifest_path)
        write_gliner_jsonl(examples, gliner_path)
        cells = Counter(
            (item.view, item.safety_scope, item.safety_label, item.tag, item.language)
            for item in examples
        )
        summary["splits"][split] = {
            "examples": len(examples),
            "paired_semantic_examples": len(examples) // 2,
            "manifest": str(manifest_path),
            "gliner_manifest": str(gliner_path),
            "cells": {"|".join(key): value for key, value in sorted(cells.items())},
        }
    summary_path = args.output_dir / "manifest_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
