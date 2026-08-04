from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guard_smoke.data import GuardExample, iter_full_examples


def distribution(values: list[int]) -> dict[str, float | int]:
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


def flush_batch(tokenizer, batch: list[GuardExample], lengths: dict[str, list[int]]) -> None:
    encoded = tokenizer(
        [example.text for example in batch],
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_length=True,
    )
    for example, token_length in zip(batch, encoded["length"]):
        value = int(token_length)
        lengths["overall"].append(value)
        lengths[f"split:{example.source_split}"].append(value)
        lengths[f"view:{example.view}"].append(value)
        lengths[f"language:{example.language}"].append(value)
        lengths[f"tag:{example.tag}"].append(value)
        lengths[f"scope:{example.safety_scope}"].append(value)
        lengths[
            f"cell:{example.source_split}|{example.view}|{example.language}|{example.tag}"
        ].append(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/mmbert_small_base_smoke"))
    parser.add_argument("--final-dir", type=Path, default=Path("data/final"))
    parser.add_argument("--output", type=Path, default=Path("reports/guard_mmbert_length_audit.json"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        fix_mistral_regex=False,
    )
    lengths: dict[str, list[int]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        batch: list[GuardExample] = []
        for example in iter_full_examples(args.final_dir, split):
            batch.append(example)
            if len(batch) >= args.batch_size:
                flush_batch(tokenizer, batch, lengths)
                batch.clear()
        if batch:
            flush_batch(tokenizer, batch, lengths)
        print(f"tokenized split={split} total={len(lengths['overall']):,}", flush=True)

    report = {
        "model_path": str(args.model_path),
        "tokenizer_class": tokenizer.__class__.__name__,
        "note": "Serialized P/R/PR text, including tokenizer special tokens; no truncation.",
        "distributions": {
            key: distribution(values) for key, values in sorted(lengths.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
