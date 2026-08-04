from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize no-truncation mmBERT fixed/schema Phase-0 manifests to 8K."
    )
    parser.add_argument("--full-dir", type=Path, default=Path("data/guard_full"))
    parser.add_argument(
        "--schema-length-dir", type=Path, default=Path("data/guard_phase0_common_512")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/guard_phase0_mmbert_8k")
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/mmbert_small_base_smoke")
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_schema_lengths(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in iter_jsonl(path):
        example_id = str(row["example_id"])
        if example_id in result:
            raise ValueError(f"Duplicate schema length ID: {example_id}")
        result[example_id] = int(row["mmbert_tokens"])
    return result


def assigned_language(record_uid: str) -> str:
    value = hashlib.sha256(record_uid.encode("utf-8")).digest()[0]
    return "en" if value % 2 == 0 else "vi"


def update_counts(counter: Counter[str], prefix: str, row: dict[str, Any]) -> None:
    counter[prefix] += 1
    for key in ("language", "view", "tag", "safety_label"):
        counter[f"{prefix}:{key}:{row[key]}"] += 1
    counter[
        f"{prefix}:cell:{row['view']}|{row['language']}|{row['tag']}|{row['safety_label']}"
    ] += 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        fix_mistral_regex=False,
    )
    fixed_dir = args.output_dir / "fixed_full"
    english_dir = args.output_dir / "fixed_english"
    matched_dir = args.output_dir / "fixed_ev_matched"
    schema_dir = args.output_dir / "schema_full"
    tail_dir = args.output_dir / "tails"
    for directory in (fixed_dir, english_dir, matched_dir, schema_dir, tail_dir):
        directory.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "contract": {
            "task": "text safety classification",
            "labels": ["safe", "unsafe"],
            "max_length": args.max_length,
            "truncation": False,
            "E3": "fixed-head English-only",
            "E4": "one hash-selected language per paired (record_uid, view)",
            "E5": "fixed-head full EN+VI",
            "E7": "schema-head full EN+VI",
        },
        "model_path": str(args.model_path),
        "splits": {},
    }
    complete_train_pair_keys: set[tuple[str, str]] = set()

    for split in ("train", "valid", "test"):
        source = args.full_dir / f"{split}.jsonl"
        schema_lengths = load_schema_lengths(
            args.schema_length_dir / f"{split}_lengths.jsonl"
        )
        fixed_path = fixed_dir / f"{split}.jsonl"
        english_path = english_dir / f"{split}.jsonl"
        schema_path = schema_dir / f"{split}.jsonl"
        fixed_tail_path = tail_dir / f"{split}_fixed.jsonl"
        schema_tail_path = tail_dir / f"{split}_schema.jsonl"
        counters: Counter[str] = Counter()
        pair_masks: dict[tuple[str, str], int] = defaultdict(int)
        seen_ids: set[str] = set()

        with (
            fixed_path.open("w", encoding="utf-8", newline="\n") as fixed_handle,
            english_path.open("w", encoding="utf-8", newline="\n") as english_handle,
            schema_path.open("w", encoding="utf-8", newline="\n") as schema_handle,
            fixed_tail_path.open("w", encoding="utf-8", newline="\n") as fixed_tail_handle,
            schema_tail_path.open("w", encoding="utf-8", newline="\n") as schema_tail_handle,
        ):
            pending: list[dict[str, Any]] = []

            def flush() -> None:
                if not pending:
                    return
                encoded = tokenizer(
                    [str(row["text"]) for row in pending],
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    return_length=True,
                )
                for row, fixed_length_raw in zip(pending, encoded["length"]):
                    example_id = str(row["example_id"])
                    if example_id in seen_ids:
                        raise ValueError(f"Duplicate example_id in {split}: {example_id}")
                    seen_ids.add(example_id)
                    if example_id not in schema_lengths:
                        raise KeyError(f"Missing schema length for {example_id}")
                    fixed_length = int(fixed_length_raw)
                    schema_length = schema_lengths[example_id]
                    fixed_ok = fixed_length <= args.max_length
                    schema_ok = schema_length <= args.max_length
                    line = json.dumps(row, ensure_ascii=False) + "\n"
                    if fixed_ok:
                        fixed_handle.write(line)
                        update_counts(counters, "fixed_eligible", row)
                        key = (str(row["record_uid"]), str(row["view"]))
                        pair_masks[key] |= 1 if row["language"] == "en" else 2
                        if row["language"] == "en":
                            english_handle.write(line)
                            update_counts(counters, "english_eligible", row)
                    else:
                        fixed_tail_handle.write(
                            json.dumps(
                                {
                                    "example_id": example_id,
                                    "record_uid": row["record_uid"],
                                    "view": row["view"],
                                    "language": row["language"],
                                    "fixed_tokens": fixed_length,
                                    "max_length": args.max_length,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        update_counts(counters, "fixed_tail", row)
                    if schema_ok:
                        schema_handle.write(line)
                        update_counts(counters, "schema_eligible", row)
                    else:
                        schema_tail_handle.write(
                            json.dumps(
                                {
                                    "example_id": example_id,
                                    "record_uid": row["record_uid"],
                                    "view": row["view"],
                                    "language": row["language"],
                                    "schema_tokens": schema_length,
                                    "max_length": args.max_length,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        update_counts(counters, "schema_tail", row)
                    counters["total"] += 1
                pending.clear()

            for row in iter_jsonl(source):
                pending.append(row)
                if len(pending) >= args.batch_size:
                    flush()
            flush()

        if len(seen_ids) != len(schema_lengths):
            raise AssertionError(
                f"Full/schema length ID count mismatch in {split}: "
                f"full={len(seen_ids)}, lengths={len(schema_lengths)}"
            )
        complete = {key for key, mask in pair_masks.items() if mask == 3}
        if split == "train":
            complete_train_pair_keys = complete
        report["splits"][split] = {
            "counts": dict(sorted(counters.items())),
            "complete_fixed_en_vi_pairs": len(complete),
            "artifacts": {
                "fixed_full": str(fixed_path),
                "fixed_english": str(english_path),
                "schema_full": str(schema_path),
                "fixed_tail": str(fixed_tail_path),
                "schema_tail": str(schema_tail_path),
            },
        }
        print(
            f"{split}: total={counters['total']:,} fixed={counters['fixed_eligible']:,} "
            f"schema={counters['schema_eligible']:,} complete_pairs={len(complete):,}",
            flush=True,
        )

    # E4 training set: one stable language per complete paired semantic view.
    matched_path = matched_dir / "train.jsonl"
    matched_counts: Counter[str] = Counter()
    seen_matched: set[tuple[str, str]] = set()
    with matched_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in iter_jsonl(args.full_dir / "train.jsonl"):
            key = (str(row["record_uid"]), str(row["view"]))
            if key not in complete_train_pair_keys:
                continue
            if row["language"] != assigned_language(str(row["record_uid"])):
                continue
            if key in seen_matched:
                raise ValueError(f"Duplicate matched semantic key: {key}")
            seen_matched.add(key)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            update_counts(matched_counts, "matched", row)

    if seen_matched != complete_train_pair_keys:
        missing = complete_train_pair_keys - seen_matched
        raise AssertionError(f"E4 matched export missed {len(missing)} pair keys")
    report["E4_train"] = {
        "counts": dict(sorted(matched_counts.items())),
        "complete_pair_keys": len(complete_train_pair_keys),
        "selected_rows": len(seen_matched),
        "selection": "sha256(record_uid)[0] parity; same record uses same language across views",
        "artifact": str(matched_path),
    }

    artifacts: list[Path] = []
    for directory in (fixed_dir, english_dir, matched_dir, schema_dir, tail_dir):
        artifacts.extend(sorted(directory.glob("*.jsonl")))
    report["artifact_sha256"] = {
        str(path): sha256_file(path) for path in artifacts
    }
    report_path = args.output_dir / "manifest_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"E4 matched train={len(seen_matched):,} "
        f"EN={matched_counts['matched:language:en']:,} "
        f"VI={matched_counts['matched:language:vi']:,}",
        flush=True,
    )
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
