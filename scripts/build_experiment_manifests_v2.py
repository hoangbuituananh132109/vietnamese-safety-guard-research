from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the full-coverage E3/E4 training manifests without dropping long rows."
    )
    parser.add_argument("--full-dir", type=Path, default=Path("data/guard_full"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/guard_experiments_v2")
    )
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


def assigned_language(record_uid: str) -> str:
    """Choose one language stably per source record, preserving the E3/E4 budget."""

    return "en" if hashlib.sha256(record_uid.encode("utf-8")).digest()[0] % 2 == 0 else "vi"


def update_counts(counts: Counter[str], row: dict[str, Any]) -> None:
    counts["instances"] += 1
    for field in ("language", "view", "safety_label", "category_scope"):
        counts[f"{field}:{row[field]}"] += 1
    categories = list(row.get("categories") or ())
    counts["category_supervision_available"] += int(row["category_scope"] != "unavailable")
    counts["category_positive_instances"] += int(bool(categories))
    for category in categories:
        counts[f"category:{category}"] += 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    e3_dir = args.output_dir / "e3_english_full"
    e4_dir = args.output_dir / "e4_ev_matched_full"
    e3_dir.mkdir(parents=True, exist_ok=True)
    e4_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "contract": {
            "E3": "all English instances; binary CE plus masked N23 BCE",
            "E4": "one hash-selected language for every (record_uid, view); binary CE plus masked N23 BCE",
            "E5": "data/guard_full train/valid/test; binary CE plus masked N23 BCE",
            "E7": "data/guard_full train/valid/test; dynamic schema binary CE plus masked N23 BCE",
            "long_sequence_policy": "retain every instance; runtime scope-aware audited truncation",
            "category_mask": "category loss exists only when category_scope is prompt or interaction",
        },
        "splits": {},
    }

    for split in ("train", "valid", "test"):
        source = args.full_dir / f"{split}.jsonl"
        e3_path = e3_dir / f"{split}.jsonl"
        e4_path = e4_dir / f"{split}.jsonl"
        e3_counts: Counter[str] = Counter()
        e4_counts: Counter[str] = Counter()
        all_counts: Counter[str] = Counter()
        seen_ids: set[str] = set()
        seen_pair_languages: dict[tuple[str, str], set[str]] = {}
        e4_keys: set[tuple[str, str]] = set()

        with (
            e3_path.open("w", encoding="utf-8", newline="\n") as e3_handle,
            e4_path.open("w", encoding="utf-8", newline="\n") as e4_handle,
        ):
            for row in iter_jsonl(source):
                example_id = str(row["example_id"])
                if example_id in seen_ids:
                    raise ValueError(f"Duplicate example_id in {source}: {example_id}")
                seen_ids.add(example_id)
                update_counts(all_counts, row)
                key = (str(row["record_uid"]), str(row["view"]))
                seen_pair_languages.setdefault(key, set()).add(str(row["language"]))

                line = json.dumps(row, ensure_ascii=False) + "\n"
                if row["language"] == "en":
                    e3_handle.write(line)
                    update_counts(e3_counts, row)

                if row["language"] == assigned_language(str(row["record_uid"])):
                    if key in e4_keys:
                        raise ValueError(f"E4 duplicate semantic key: {key}")
                    e4_keys.add(key)
                    e4_handle.write(line)
                    update_counts(e4_counts, row)

        incomplete = {key: langs for key, langs in seen_pair_languages.items() if langs != {"en", "vi"}}
        if incomplete:
            raise AssertionError(f"{split} has {len(incomplete)} incomplete EN/VI semantic pairs")
        if e3_counts["instances"] != len(seen_pair_languages):
            raise AssertionError(f"E3 is not one English instance per semantic pair in {split}")
        if e4_counts["instances"] != len(seen_pair_languages):
            raise AssertionError(f"E4 is not one selected language per semantic pair in {split}")
        if e3_counts["instances"] != e4_counts["instances"]:
            raise AssertionError(f"E3/E4 budgets differ in {split}")

        report["splits"][split] = {
            "full": dict(sorted(all_counts.items())),
            "semantic_pair_units": len(seen_pair_languages),
            "E3": dict(sorted(e3_counts.items())),
            "E4": dict(sorted(e4_counts.items())),
            "artifacts": {"E3": str(e3_path), "E4": str(e4_path)},
        }
        print(
            f"{split}: full={all_counts['instances']:,} "
            f"E3={e3_counts['instances']:,} E4={e4_counts['instances']:,}",
            flush=True,
        )

    artifacts = sorted(e3_dir.glob("*.jsonl")) + sorted(e4_dir.glob("*.jsonl"))
    report["artifact_sha256"] = {str(path): sha256_file(path) for path in artifacts}
    report_path = args.output_dir / "manifest_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
