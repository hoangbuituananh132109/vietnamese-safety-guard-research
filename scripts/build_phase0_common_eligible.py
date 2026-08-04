from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from guard_smoke.data import GuardExample
from guard_smoke.gliguard import _schema_for_example
from guard_smoke.mmbert_schema import BinarySchemaCodec, load_mmbert_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact no-truncation Phase-0 manifests eligible for both "
            "GLiGuard and mmBERT at the same total encoder context length."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/guard_full"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/guard_phase0_common_512")
    )
    parser.add_argument(
        "--gliguard-model", type=Path, default=Path("models/fastino_gliguard_300m")
    )
    parser.add_argument(
        "--mmbert-model", type=Path, default=Path("models/mmbert_small_base_smoke")
    )
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--verification-samples", type=int, default=24)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser.parse_args()


def distribution(values: list[int], context_length: int) -> dict[str, int | float]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.int64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
        f"over_{context_length}": int((array > context_length).sum()),
        "over_1024": int((array > 1024).sum()),
        "over_2048": int((array > 2048).sum()),
        "over_4096": int((array > 4096).sum()),
        "over_8192": int((array > 8192).sum()),
    }


def example_from_row(row: dict[str, Any]) -> GuardExample:
    allowed = {field.name for field in fields(GuardExample)}
    payload = {key: row[key] for key in allowed}
    payload["categories"] = tuple(payload.get("categories") or ())
    example = GuardExample(**payload)
    example.validate()
    return example


def _token_subword_length(processor: Any, token: str) -> int:
    if token in processor._token_cache:
        return len(processor._token_cache[token])
    return len(processor._tokenize_cached(token))


def exact_fast_gliguard_length(
    model: Any,
    example: GuardExample,
    fixed_schema_cache: dict[str, int],
) -> tuple[int, dict[str, Any]]:
    """Match GLiNER2's final untruncated encoder length without tensor collation."""

    processor = model.processor
    schema = _schema_for_example(model, example)
    cache_key = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    if cache_key not in fixed_schema_cache:
        transformed = processor._transform_record(
            {"text": "", "schema": copy.deepcopy(schema)}, max_len=None
        )
        fixed_schema_cache[cache_key] = len(transformed.input_ids)

    text = example.text
    if text and not text.endswith((".", "!", "?")):
        text += "."
    elif not text:
        text = "."
    text_tokens = [token for token, _, _ in processor.word_splitter(text, lower=True)]
    length = fixed_schema_cache[cache_key] + sum(
        _token_subword_length(processor, token) for token in text_tokens
    )
    return int(length), schema


def update_slices(counter: Counter[str], row: dict[str, Any], prefix: str) -> None:
    counter[f"{prefix}:all"] += 1
    for key in ("view", "language", "tag", "safety_label", "safety_scope"):
        counter[f"{prefix}:{key}:{row[key]}"] += 1
    counter[
        f"{prefix}:cell:{row['view']}|{row['language']}|{row['tag']}|{row['safety_label']}"
    ] += 1


def main() -> None:
    args = parse_args()
    if args.context_length <= 0:
        raise ValueError("context-length must be positive")

    from gliner2 import GLiNER2

    print(f"loading GLiGuard processor/model from {args.gliguard_model}", flush=True)
    gliguard = GLiNER2.from_pretrained(str(args.gliguard_model))
    print(f"loading mmBERT tokenizer from {args.mmbert_model}", flush=True)
    tokenizer = load_mmbert_tokenizer(str(args.mmbert_model))
    mmbert_codec = BinarySchemaCodec(tokenizer, max_length=args.context_length)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    report: dict[str, Any] = {
        "contract": {
            "task": "text safety classification",
            "task_type": "single_label",
            "labels": ["safe", "unsafe"],
            "context_length": args.context_length,
            "truncation": False,
            "eligibility": (
                "gliguard_total_encoder_tokens <= context_length AND "
                "mmbert_total_serialized_tokens <= context_length"
            ),
            "mmbert_schema": mmbert_codec.canonical_schema_text,
            "gliguard_length_semantics": (
                "Final schema+text subwords produced by GLiNER2 processor with max_len=None"
            ),
        },
        "models": {
            "gliguard": str(args.gliguard_model),
            "mmbert": str(args.mmbert_model),
        },
        "splits": {},
    }
    fixed_schema_cache: dict[str, int] = {}

    for split in splits:
        source = args.input_dir / f"{split}.jsonl"
        if not source.exists():
            raise FileNotFoundError(source)
        eligible_path = args.output_dir / f"{split}.jsonl"
        excluded_path = args.output_dir / f"{split}_excluded.jsonl"
        lengths_path = args.output_dir / f"{split}_lengths.jsonl"

        counters: Counter[str] = Counter()
        for key in (
            "total", "eligible", "excluded", "status:both_ok",
            "status:both_over", "status:gliguard_only_over", "status:mmbert_only_over",
        ):
            counters[key] = 0
        gli_lengths: list[int] = []
        mm_lengths: list[int] = []
        max_lengths: list[int] = []
        longest: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        verified = 0
        boundary_verified = 0

        with (
            source.open("r", encoding="utf-8") as input_handle,
            eligible_path.open("w", encoding="utf-8", newline="\n") as eligible_handle,
            excluded_path.open("w", encoding="utf-8", newline="\n") as excluded_handle,
            lengths_path.open("w", encoding="utf-8", newline="\n") as lengths_handle,
        ):
            for line_number, line in enumerate(input_handle, 1):
                if not line.strip():
                    continue
                if args.limit_per_split is not None and counters["total"] >= args.limit_per_split:
                    break
                row = json.loads(line)
                example = example_from_row(row)
                if example.example_id in seen_ids:
                    raise ValueError(f"Duplicate example_id in {split}: {example.example_id}")
                seen_ids.add(example.example_id)

                gli_length, schema = exact_fast_gliguard_length(
                    gliguard, example, fixed_schema_cache
                )
                mm_length = mmbert_codec.length(example)

                standard_check = verified < args.verification_samples
                boundary_check = (
                    not standard_check
                    and abs(gli_length - args.context_length) <= 4
                    and boundary_verified < 8
                )
                if standard_check or boundary_check:
                    exact = int(
                        gliguard.processor.collate_fn_inference(
                            [(example.text, copy.deepcopy(schema))], max_len=None
                        ).original_lengths[0]
                    )
                    if exact != gli_length:
                        raise AssertionError(
                            f"GLiGuard fast/exact mismatch {example.example_id}: "
                            f"fast={gli_length}, exact={exact}"
                        )
                    if standard_check:
                        verified += 1
                    else:
                        boundary_verified += 1

                gli_ok = gli_length <= args.context_length
                mm_ok = mm_length <= args.context_length
                eligible = gli_ok and mm_ok
                reasons: list[str] = []
                if not gli_ok:
                    reasons.append("gliguard_over_context")
                if not mm_ok:
                    reasons.append("mmbert_over_context")

                length_row = {
                    "example_id": example.example_id,
                    "record_uid": example.record_uid,
                    "split": split,
                    "view": example.view,
                    "language": example.language,
                    "tag": example.tag,
                    "safety_scope": example.safety_scope,
                    "safety_label": example.safety_label,
                    "gliguard_tokens": gli_length,
                    "mmbert_tokens": mm_length,
                    "max_model_tokens": max(gli_length, mm_length),
                    "eligible": eligible,
                    "reasons": reasons,
                }
                lengths_handle.write(json.dumps(length_row, ensure_ascii=False) + "\n")
                if eligible:
                    eligible_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    update_slices(counters, row, "eligible")
                else:
                    excluded_handle.write(json.dumps(length_row, ensure_ascii=False) + "\n")
                    update_slices(counters, row, "excluded")

                counters["total"] += 1
                counters["eligible" if eligible else "excluded"] += 1
                counters[
                    "status:both_ok" if eligible else (
                        "status:both_over" if not gli_ok and not mm_ok else
                        "status:gliguard_only_over" if not gli_ok else
                        "status:mmbert_only_over"
                    )
                ] += 1
                gli_lengths.append(gli_length)
                mm_lengths.append(mm_length)
                max_lengths.append(max(gli_length, mm_length))
                longest.append(length_row)
                if len(longest) > 200:
                    longest = sorted(
                        longest,
                        key=lambda item: item["max_model_tokens"],
                        reverse=True,
                    )[:40]

                if counters["total"] % args.progress_every == 0:
                    print(
                        f"{split}: processed={counters['total']:,} "
                        f"eligible={counters['eligible']:,} "
                        f"excluded={counters['excluded']:,}",
                        flush=True,
                    )

        if verified < min(args.verification_samples, counters["total"]):
            raise AssertionError(
                f"Only verified {verified} GLiGuard lengths in {split}; "
                f"requested {args.verification_samples}"
            )
        report["splits"][split] = {
            "counts": dict(sorted(counters.items())),
            "coverage": (
                float(counters["eligible"] / counters["total"])
                if counters["total"]
                else 0.0
            ),
            "gliguard_tokens": distribution(gli_lengths, args.context_length),
            "mmbert_tokens": distribution(mm_lengths, args.context_length),
            "max_model_tokens": distribution(max_lengths, args.context_length),
            "exact_gliguard_length_checks": verified + boundary_verified,
            "standard_exact_checks": verified,
            "boundary_exact_checks": boundary_verified,
            "longest": sorted(
                longest,
                key=lambda item: item["max_model_tokens"],
                reverse=True,
            )[:20],
            "artifacts": {
                "eligible": str(eligible_path),
                "excluded": str(excluded_path),
                "lengths": str(lengths_path),
            },
        }
        print(
            f"{split}: total={counters['total']:,} eligible={counters['eligible']:,} "
            f"coverage={report['splits'][split]['coverage']:.4%}",
            flush=True,
        )

    report["schema_variants_seen"] = len(fixed_schema_cache)
    report_path = args.output_dir / "eligibility_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
