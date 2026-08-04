from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from evaluate_nemotron_decoder_guard import (
    conversation_fields,
    read_jsonl,
    render_chat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile exact Nemotron Guard chat-prompt lengths for SFT planning."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser.parse_args()


def summarize(values: list[int]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.int64)
    result: dict[str, Any] = {
        "count": int(data.size),
        "min": int(data.min()),
        "mean": float(data.mean()),
        "max": int(data.max()),
    }
    for percentile in (50, 75, 90, 95, 99, 99.5, 99.9):
        result[f"p{str(percentile).replace('.', '_')}"] = float(
            np.percentile(data, percentile)
        )
    for limit in (512, 1024, 2048, 3072, 4096, 6144, 8064, 8192):
        count = int((data <= limit).sum())
        result[f"le_{limit}"] = count
        result[f"le_{limit}_fraction"] = count / int(data.size)
    return result


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = read_jsonl(args.manifest)
    groups: dict[str, list[int]] = defaultdict(list)

    completed = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts: list[str] = []
        for row in batch:
            query, response, _ = conversation_fields(row)
            prompts.append(render_chat(tokenizer, query, response))
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=False,
            return_length=True,
            padding=False,
        )
        for row, length in zip(batch, encoded["length"], strict=True):
            value = int(length)
            language = str(row.get("language", "unknown"))
            view = str(row.get("view", "unknown"))
            groups["all"].append(value)
            groups[f"language:{language}"].append(value)
            groups[f"view:{view}"].append(value)
            groups[f"language_view:{language}:{view}"].append(value)
        completed += len(batch)
        if args.progress_every and (
            completed == len(rows) or completed // args.progress_every != start // args.progress_every
        ):
            print(f"profiled={completed:,}/{len(rows):,}", flush=True)

    payload = {
        "model": str(args.model),
        "manifest": str(args.manifest),
        "prompt_renderer": "scripts/evaluate_nemotron_decoder_guard.py::render_chat",
        "records": len(rows),
        "groups": {name: summarize(values) for name, values in sorted(groups.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["groups"]["all"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
