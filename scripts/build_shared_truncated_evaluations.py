from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import fields
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

from transformers import AutoTokenizer

from guard_smoke.constants import SAFETY_LABELS, TEXT_SAFETY_TASK
from guard_smoke.data import GuardExample, gliner_payload
from guard_train.truncation import truncate_text_exact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build E1/E2 native-fit and shared-text truncated Nemotron/SEA evaluations."
        )
    )
    parser.add_argument("--full-dir", type=Path, default=Path("data/guard_full"))
    parser.add_argument(
        "--sea-paired",
        type=Path,
        default=Path("data/benchmarks/sea_safeguard/paired_en_vi.jsonl"),
    )
    parser.add_argument(
        "--gliguard-model", type=Path, default=Path("models/fastino_gliguard_300m")
    )
    parser.add_argument(
        "--mmbert-model", type=Path, default=Path("models/mmbert_small_base_smoke")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/eval_shared_gliguard_512")
    )
    parser.add_argument("--gliguard-limit", type=int, default=512)
    parser.add_argument("--mmbert-limit", type=int, default=8192)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def as_guard_example(row: dict[str, Any]) -> GuardExample:
    allowed = {field.name for field in fields(GuardExample)}
    payload = {key: row[key] for key in allowed}
    payload["categories"] = tuple(payload.get("categories") or ())
    example = GuardExample(**payload)
    example.validate()
    return example


def direct_gliner_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": row["text"],
        "output": {
            "classifications": [
                {
                    "task": TEXT_SAFETY_TASK,
                    "labels": list(SAFETY_LABELS),
                    "true_label": [row["safety_label"]],
                }
            ]
        },
        "metadata": {
            "example_id": row["example_id"],
            "pair_uid": row.get("pair_uid"),
            "view": row["view"],
            "language": row["language"],
        },
    }


def process_dataset(
    *,
    name: str,
    source: Path,
    output_dir: Path,
    processor: Any,
    schema: dict[str, Any],
    mm_tokenizer: Any,
    gl_limit: int,
    mm_limit: int,
    sea: bool,
) -> dict[str, Any]:
    rows = list(iter_jsonl(source))
    audits: list[dict[str, Any]] = []
    native_candidates: list[dict[str, Any]] = []
    truncated_rows: list[dict[str, Any]] = []
    pair_masks: dict[str, int] = defaultdict(int)
    native_pair_masks: dict[str, int] = defaultdict(int)
    counts: Counter[str] = Counter()

    def gl_length(text: str) -> int:
        batch = processor.collate_fn_inference([(text, schema)], max_len=None)
        return int(batch.original_lengths[0])

    def mm_length(text: str) -> int:
        return len(mm_tokenizer.encode(text, add_special_tokens=True))

    for index, row in enumerate(rows, 1):
        language = str(row["language"])
        pair_key = (
            str(row["pair_uid"])
            if sea
            else f"{row['record_uid']}:{row['view']}"
        )
        language_bit = 1 if language == "en" else 2
        pair_masks[pair_key] |= language_bit
        original_text = str(row["text"])
        gl_original = gl_length(original_text)
        mm_original = mm_length(original_text)
        native = gl_original <= gl_limit and mm_original <= mm_limit
        if native:
            native_pair_masks[pair_key] |= language_bit
            native_candidates.append(row)

        fitted, trunc_audit = truncate_text_exact(
            original_text, str(row["view"]), gl_length, gl_limit
        )
        # Extremely tokenizer-pathological strings can fit GLiGuard yet explode
        # beyond mmBERT 8K. If that happens, shorten the same string for both.
        if mm_length(fitted) > mm_limit:
            def joint_scaled_length(value: str) -> int:
                mm_scaled = math.ceil(mm_length(value) * gl_limit / mm_limit)
                return max(gl_length(value), mm_scaled)

            fitted, _ = truncate_text_exact(
                fitted, str(row["view"]), joint_scaled_length, gl_limit
            )
        gl_final = gl_length(fitted)
        mm_final = mm_length(fitted)
        if gl_final > gl_limit or mm_final > mm_limit:
            raise AssertionError(
                f"Final shared text over limit: {row['example_id']} gl={gl_final} mm={mm_final}"
            )
        output_row = dict(row)
        output_row["text"] = fitted
        if "char_count" in output_row:
            output_row["char_count"] = len(fitted)
        truncated_rows.append(output_row)
        was_truncated = fitted != original_text
        counts["instances"] += 1
        counts["native_instances"] += int(native)
        counts["truncated_instances"] += int(was_truncated)
        counts[f"language:{language}"] += 1
        counts[f"view:{row['view']}"] += 1
        audits.append(
            {
                "example_id": row["example_id"],
                "pair_key": pair_key,
                "language": language,
                "view": row["view"],
                "native_fit": native,
                "was_truncated": was_truncated,
                "strategy": trunc_audit.strategy if was_truncated else "none",
                "original_text_sha256": sha256_text(original_text),
                "final_text_sha256": sha256_text(fitted),
                "original_chars": len(original_text),
                "final_chars": len(fitted),
                "gliguard_original_tokens": gl_original,
                "gliguard_final_tokens": gl_final,
                "mmbert_original_tokens": mm_original,
                "mmbert_final_tokens": mm_final,
                "gliguard_limit": gl_limit,
                "mmbert_limit": mm_limit,
            }
        )
        if index % 1000 == 0:
            print(f"{name}: processed={index:,}/{len(rows):,}", flush=True)

    incomplete = [key for key, mask in pair_masks.items() if mask != 3]
    if incomplete:
        raise AssertionError(f"{name}: {len(incomplete)} incomplete source pairs")
    native_keys = {key for key, mask in native_pair_masks.items() if mask == 3}
    native_rows = [
        row
        for row in native_candidates
        if (
            str(row["pair_uid"])
            if sea
            else f"{row['record_uid']}:{row['view']}"
        )
        in native_keys
    ]
    if len(native_rows) != 2 * len(native_keys):
        raise AssertionError(f"{name}: native paired export is not exactly bilingual")

    native_path = output_dir / f"{name}_native.jsonl"
    truncated_path = output_dir / f"{name}_shared_truncated.jsonl"
    audit_path = output_dir / f"{name}_truncation_audit.jsonl"
    native_gliner_path = output_dir / f"{name}_native_gliner.jsonl"
    truncated_gliner_path = output_dir / f"{name}_shared_truncated_gliner.jsonl"
    write_jsonl(native_path, native_rows)
    write_jsonl(truncated_path, truncated_rows)
    write_jsonl(audit_path, audits)
    if sea:
        write_jsonl(native_gliner_path, (direct_gliner_payload(row) for row in native_rows))
        write_jsonl(
            truncated_gliner_path,
            (direct_gliner_payload(row) for row in truncated_rows),
        )
    else:
        write_jsonl(
            native_gliner_path,
            (gliner_payload(as_guard_example(row)) for row in native_rows),
        )
        write_jsonl(
            truncated_gliner_path,
            (gliner_payload(as_guard_example(row)) for row in truncated_rows),
        )
    return {
        "source": str(source),
        "source_instances": len(rows),
        "source_pair_units": len(pair_masks),
        "native_pair_units": len(native_keys),
        "native_instances": len(native_rows),
        "shared_truncated_instances": len(truncated_rows),
        "counts": dict(sorted(counts.items())),
        "artifacts": {
            "native": str(native_path),
            "native_gliner": str(native_gliner_path),
            "shared_truncated": str(truncated_path),
            "shared_truncated_gliner": str(truncated_gliner_path),
            "audit": str(audit_path),
        },
    }


def main() -> None:
    args = parse_args()
    from gliner2 import GLiNER2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading GLiGuard processor from {args.gliguard_model}", flush=True)
    gliguard = GLiNER2.from_pretrained(str(args.gliguard_model))
    schema = (
        gliguard.create_schema()
        .classification(TEXT_SAFETY_TASK, list(SAFETY_LABELS))
        .build()
    )
    mm_tokenizer = AutoTokenizer.from_pretrained(
        str(args.mmbert_model), local_files_only=True, fix_mistral_regex=False
    )
    report = {
        "contract": {
            "primary": "complete EN/VI pairs natively fitting GLiGuard<=512 and mmBERT<=8192",
            "secondary": "all complete EN/VI pairs; text truncated once by GLiGuard limit and sent unchanged to both models",
            "schema_overhead_included": True,
            "PR_priority": "response head+tail retained because PR target is response safety",
        },
        "datasets": {},
    }
    for split in ("valid", "test"):
        name = f"nemotron_{split}"
        report["datasets"][name] = process_dataset(
            name=name,
            source=args.full_dir / f"{split}.jsonl",
            output_dir=args.output_dir,
            processor=gliguard.processor,
            schema=schema,
            mm_tokenizer=mm_tokenizer,
            gl_limit=args.gliguard_limit,
            mm_limit=args.mmbert_limit,
            sea=False,
        )
    report["datasets"]["sea_paired"] = process_dataset(
        name="sea_paired",
        source=args.sea_paired,
        output_dir=args.output_dir,
        processor=gliguard.processor,
        schema=schema,
        mm_tokenizer=mm_tokenizer,
        gl_limit=args.gliguard_limit,
        mm_limit=args.mmbert_limit,
        sea=True,
    )
    report_path = args.output_dir / "manifest_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
