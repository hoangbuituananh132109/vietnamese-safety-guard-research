from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from guard_smoke.data import GuardExample, iter_full_examples
from guard_smoke.gliguard import _schema_for_example


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


def append_lengths(
    lengths: dict[str, list[int]],
    mode: str,
    examples: list[GuardExample],
    values: list[int],
    include_categories: bool,
) -> None:
    for example, value in zip(examples, values):
        keys = (
            "overall",
            f"split:{example.source_split}",
            f"view:{example.view}",
            f"language:{example.language}",
            f"tag:{example.tag}",
            f"scope:{example.safety_scope}",
            "schema_count:1",
            f"cell:{example.source_split}|{example.view}|{example.language}|{example.tag}",
        )
        for key in keys:
            lengths[f"{mode}|{key}"].append(int(value))


def _token_subword_length(processor, token: str) -> int:
    if token in processor._token_cache:
        return len(processor._token_cache[token])
    return len(processor._tokenize_cached(token))


def _fast_lengths(
    model,
    example: GuardExample,
    cap_words: int,
    fixed_cache: dict,
    compute_untruncated: bool,
    include_categories: bool,
) -> tuple[int | None, int, int]:
    """Exact length without span mappings/tensor padding used only for auditing."""

    processor = model.processor
    schema = _schema_for_example(model, example)
    key = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    if key not in fixed_cache:
        # Empty text still includes any classification prefix. This is the
        # exact schema+prefix+SEP_TEXT cost; it is reused for every matching
        # task schema.
        transformed = processor._transform_record(
            {"text": "", "schema": copy.deepcopy(schema)}, max_len=None
        )
        fixed_cache[key] = len(transformed.input_ids)
    fixed_subwords = fixed_cache[key]

    text = example.text
    if text and not text.endswith((".", "!", "?")):
        text += "."
    elif not text:
        text = "."
    text_tokens = [token for token, _, _ in processor.word_splitter(text, lower=True)]
    capped_text_tokens = text_tokens[:cap_words]
    capped = fixed_subwords + sum(
        _token_subword_length(processor, token) for token in capped_text_tokens
    )
    untruncated = None
    if compute_untruncated:
        untruncated = capped + sum(
            _token_subword_length(processor, token)
            for token in text_tokens[cap_words:]
        )
    return untruncated, capped, min(len(text_tokens), cap_words)


def flush_batch(
    model,
    examples: list[GuardExample],
    cap_words: int,
    lengths,
    longest,
    fixed_cache: dict,
    verification: dict,
    cap_only: bool,
    include_categories: bool,
) -> None:
    mode_values: dict[str, list[int]] = defaultdict(list)
    capped_word_counts: list[int] = []
    for example in examples:
        untruncated, capped, capped_words = _fast_lengths(
            model,
            example,
            cap_words,
            fixed_cache,
            compute_untruncated=not cap_only,
            include_categories=include_categories,
        )
        if untruncated is not None:
            mode_values["untruncated"].append(untruncated)
        mode_values[f"cap_{cap_words}_words"].append(capped)
        capped_word_counts.append(capped_words)

        if verification["remaining"] > 0:
            schema = _schema_for_example(
                model, example, include_categories=include_categories
            )
            exact_cap = model.processor.collate_fn_inference(
                [(example.text, schema)], max_len=cap_words
            ).original_lengths[0]
            exact_full = None
            if not cap_only:
                exact_full = model.processor.collate_fn_inference(
                    [(example.text, schema)], max_len=None
                ).original_lengths[0]
            if capped != int(exact_cap) or (
                not cap_only and untruncated != int(exact_full)
            ):
                raise AssertionError(
                    f"Fast length mismatch for {example.example_id}: "
                    f"fast={(untruncated, capped)} exact={(exact_full, exact_cap)}"
                )
            verification["remaining"] -= 1
            verification["passed"] += 1

    for mode, values in mode_values.items():
        append_lengths(
            lengths, mode, examples, values, include_categories=include_categories
        )
        words_for_mode = (
            capped_word_counts
            if mode.startswith("cap_")
            else [None] * len(examples)
        )
        for example, subwords, words in zip(examples, values, words_for_mode):
            longest[mode].append(
                {
                    "example_id": example.example_id,
                    "split": example.source_split,
                    "view": example.view,
                    "language": example.language,
                    "tag": example.tag,
                    "schema_count": int(
                        1
                        if (not include_categories or example.category_scope == "unavailable")
                        else 2
                    ),
                    "text_words_after_cap": int(words) if words is not None else None,
                    "encoder_subwords": subwords,
                }
            )
        longest[mode] = sorted(
            longest[mode], key=lambda row: row["encoder_subwords"], reverse=True
        )[:20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/fastino_gliguard_300m"))
    parser.add_argument("--final-dir", type=Path, default=Path("data/final"))
    parser.add_argument("--output", type=Path, default=Path("reports/guard_gliner_length_audit.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cap-words", type=int, default=128)
    parser.add_argument(
        "--sample-modulus",
        type=int,
        default=16,
        help="Keep example_id hashes where hash %% modulus == 0; 1 audits all.",
    )
    parser.add_argument("--verification-samples", type=int, default=8)
    parser.add_argument(
        "--cap-only",
        action="store_true",
        help="Skip untruncated subword tokenization; useful for a fast full-corpus cap audit.",
    )
    parser.add_argument(
        "--binary-only",
        action="store_true",
        help="Deprecated no-op: Phase 0 always uses one binary text-safety schema.",
    )
    args = parser.parse_args()

    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(str(args.model_path))
    lengths: dict[str, list[int]] = defaultdict(list)
    longest: dict[str, list[dict]] = defaultdict(list)
    fixed_cache: dict[str, int] = {}
    verification = {"remaining": args.verification_samples, "passed": 0}
    total = 0
    for split in ("train", "valid", "test"):
        pending: list[GuardExample] = []
        for example in iter_full_examples(args.final_dir, split):
            bucket = int.from_bytes(
                hashlib.sha256(example.example_id.encode("utf-8")).digest()[:8],
                "big",
            )
            if bucket % args.sample_modulus != 0:
                continue
            pending.append(example)
            if len(pending) >= args.batch_size:
                flush_batch(
                    model,
                    pending,
                    args.cap_words,
                    lengths,
                    longest,
                    fixed_cache,
                    verification,
                    args.cap_only,
                    False,
                )
                total += len(pending)
                pending.clear()
                if total % 2_000 < args.batch_size:
                    print(f"processed={total:,}", flush=True)
        if pending:
            flush_batch(
                model,
                pending,
                args.cap_words,
                lengths,
                longest,
                fixed_cache,
                verification,
                args.cap_only,
                False,
            )
            total += len(pending)
        print(f"processed split={split} total={total:,}", flush=True)

    report = {
        "model_path": str(args.model_path),
        "cap_words": args.cap_words,
        "sampling": {
            "method": "sha256(example_id) modulo",
            "modulus": args.sample_modulus,
            "expected_fraction": 1.0 / args.sample_modulus,
            "sampled_examples": total,
        },
        "cap_only": args.cap_only,
        "binary_only": True,
        "task": "text safety classification",
        "critical_note": (
            "GLiNER2 max_len caps text word tokens before schema. All reported "
            "encoder_subwords include schema and tokenizer expansion."
        ),
        "fast_length_verification": {
            "exact_collate_samples_passed": verification["passed"],
            "schema_variants": len(fixed_cache),
        },
        "distributions": {
            key: distribution(values) for key, values in sorted(lengths.items())
        },
        "longest": dict(longest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
