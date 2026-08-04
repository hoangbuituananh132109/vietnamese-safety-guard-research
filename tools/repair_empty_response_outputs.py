#!/usr/bin/env python3
"""One-time QA repair: enforce empty source response -> empty translation."""
from __future__ import annotations

import argparse
from pathlib import Path

from translator.jsonl_io import read_jsonl, write_jsonl
from translator.pipeline import text_hash
from translator.validators import quality_warnings


def repair(path: Path, source_by_uid: dict[str, dict]) -> int:
    rows = [r for _, r, _ in read_jsonl(path)]
    changed = 0
    for row in rows:
        source = source_by_uid.get(row.get("record_uid"), {})
        source_response = row.get("response_en", row.get("response", source.get("response")))
        if source_response == "" and row.get("response_vi") != "":
            row["response_vi"] = ""
            row["translation_text_sha256"] = text_hash(row.get("prompt_vi"), "")
            row["output_translation_chars"] = len(row.get("prompt_vi") or "")
            changed += 1
        if source:
            row["validation_warnings"] = quality_warnings(source, row)
            if source_response == "":
                row["validation_warnings"].append("empty_response_preserved_post_qa")
    write_jsonl(path, rows)
    return changed


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--source", required=True, type=Path); p.add_argument("paths", nargs="+", type=Path); args = p.parse_args()
    source_by_uid = {r["record_uid"]: r for _, r, _ in read_jsonl(args.source)}
    for path in args.paths:
        print(f"{path}: changed={repair(path, source_by_uid)}")


if __name__ == "__main__": main()
