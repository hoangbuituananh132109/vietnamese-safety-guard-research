#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from translator.jsonl_io import read_jsonl, write_jsonl
from translator.validators import quality_warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--repair", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reason", default="manual_qa_repair")
    args = parser.parse_args()
    base = [row for _, row, _ in read_jsonl(args.base)]
    repairs = {row["record_uid"]: row for _, row, _ in read_jsonl(args.repair)}
    base_uids = {row["record_uid"] for row in base}
    unknown = sorted(set(repairs) - base_uids)
    if unknown:
        raise SystemExit(f"repair contains unknown record_uid: {unknown}")
    merged = []
    for row in base:
        replacement = repairs.get(row["record_uid"])
        if replacement is None:
            merged.append(row)
            continue
        replacement["qa_repair_history"] = [{
            "previous_prompt_version": row.get("translation_prompt_version"),
            "replacement_prompt_version": replacement.get("translation_prompt_version"),
            "reason": args.reason,
        }]
        replacement["validation_warnings"] = quality_warnings(replacement, replacement)
        merged.append(replacement)
    write_jsonl(args.output, merged)
    print(f"merged={len(merged)} replaced={len(repairs)} output={args.output}")


if __name__ == "__main__":
    main()
