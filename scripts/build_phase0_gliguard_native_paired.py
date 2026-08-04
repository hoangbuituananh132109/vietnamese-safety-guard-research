from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guard_smoke.data import GuardExample, gliner_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dir", type=Path, default=Path("data/guard_full"))
    parser.add_argument(
        "--length-dir", type=Path, default=Path("data/guard_phase0_common_512")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/guard_phase0_gliguard_native_512")
    )
    parser.add_argument("--max-gliguard-tokens", type=int, default=512)
    parser.add_argument(
        "--max-mmbert-tokens",
        type=int,
        default=8192,
        help="8K fairness gate for the paired E1/E2 comparison; not a 512 intersection.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def as_example(row: dict[str, Any]) -> GuardExample:
    allowed = {field.name for field in fields(GuardExample)}
    payload = {key: row[key] for key in allowed}
    payload["categories"] = tuple(payload.get("categories") or ())
    example = GuardExample(**payload)
    example.validate()
    return example


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "contract": {
            "task": "text safety classification",
            "labels": ["safe", "unsafe"],
            "truncation": False,
            "eligibility": (
                "both EN and VI of the same (record_uid, view) have final GLiGuard "
                "schema+text encoder length <= 512 and mmBERT schema+text length <= 8192"
            ),
            "schema_overhead_included": True,
        },
        "splits": {},
    }

    for split in ("train", "valid", "test"):
        length_rows = list(iter_jsonl(args.length_dir / f"{split}_lengths.jsonl"))
        lengths = {str(row["example_id"]): row for row in length_rows}
        pair_masks: dict[tuple[str, str], int] = defaultdict(int)
        for row in length_rows:
            if (
                int(row["gliguard_tokens"]) <= args.max_gliguard_tokens
                and int(row["mmbert_tokens"]) <= args.max_mmbert_tokens
            ):
                key = (str(row["record_uid"]), str(row["view"]))
                pair_masks[key] |= 1 if row["language"] == "en" else 2
        complete_keys = {key for key, mask in pair_masks.items() if mask == 3}

        canonical_path = args.output_dir / f"{split}.jsonl"
        gliner_path = args.output_dir / f"{split}_gliner.jsonl"
        excluded_path = args.output_dir / f"{split}_excluded_pairs.jsonl"
        counters: Counter[str] = Counter()
        included_ids: set[str] = set()
        all_pair_keys: set[tuple[str, str]] = set()
        with (
            canonical_path.open("w", encoding="utf-8", newline="\n") as canonical,
            gliner_path.open("w", encoding="utf-8", newline="\n") as gliner,
        ):
            for row in iter_jsonl(args.full_dir / f"{split}.jsonl"):
                key = (str(row["record_uid"]), str(row["view"]))
                all_pair_keys.add(key)
                if key not in complete_keys:
                    continue
                example_id = str(row["example_id"])
                if example_id in included_ids:
                    raise ValueError(f"Duplicate example_id: {example_id}")
                included_ids.add(example_id)
                canonical.write(json.dumps(row, ensure_ascii=False) + "\n")
                gliner.write(json.dumps(gliner_payload(as_example(row)), ensure_ascii=False) + "\n")
                counters["instances"] += 1
                counters[f"language:{row['language']}"] += 1
                counters[f"view:{row['view']}"] += 1
                counters[f"label:{row['safety_label']}"] += 1
                length = lengths[example_id]
                counters["mmbert_schema_over_512"] += int(int(length["mmbert_tokens"]) > 512)
                counters["mmbert_schema_over_8192"] += int(int(length["mmbert_tokens"]) > 8192)

        with excluded_path.open("w", encoding="utf-8", newline="\n") as excluded:
            for key in sorted(all_pair_keys - complete_keys):
                excluded.write(
                    json.dumps(
                        {
                            "record_uid": key[0],
                            "view": key[1],
                            "reason": "one_or_both_languages_outside_e1_e2_native_limits",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if counters["language:en"] != counters["language:vi"]:
            raise AssertionError(f"EN/VI instance mismatch in {split}: {dict(counters)}")
        if counters["instances"] != 2 * len(complete_keys):
            raise AssertionError(f"Expected exactly two languages per pair in {split}")
        report["splits"][split] = {
            "semantic_pair_units": len(complete_keys),
            "instances": counters["instances"],
            "excluded_pair_units": len(all_pair_keys - complete_keys),
            "counts": dict(sorted(counters.items())),
            "artifacts": {
                "canonical": str(canonical_path),
                "gliner": str(gliner_path),
                "excluded_pairs": str(excluded_path),
            },
        }
        print(
            f"{split}: pairs={len(complete_keys):,} instances={counters['instances']:,} "
            f"mmbert_schema_over_512={counters['mmbert_schema_over_512']:,}",
            flush=True,
        )

    output = args.output_dir / "manifest_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={output}", flush=True)


if __name__ == "__main__":
    main()
