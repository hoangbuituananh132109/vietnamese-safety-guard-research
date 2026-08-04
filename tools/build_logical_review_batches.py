#!/usr/bin/env python3
"""Build reviewer-friendly logical batches from the unresolved handoff.

Unlike the old transport batches, each logical batch holds up to 20 records.
Very long records remain inside their logical batch but are flagged so a
reviewer can handle them one at a time without losing their place in the
20-record sequence.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator.checkpoint import load_checkpoint
from translator.full_run import paths
from translator.jsonl_io import read_jsonl, write_jsonl


DONE_STATUSES = {"terra_revised", "luna_revised", "gemini_revised"}


def completed_uids(root: Path) -> set[str]:
    done: set[str] = set()
    for group in range(5):
        for split in ("train", "valid", "test"):
            done.update(
                uid for uid, row in load_checkpoint(paths(root, split, group)["checkpoint"]).items()
                if row.get("translation_status") in DONE_STATUSES
            )
    return done


def source_chars(row: dict) -> int:
    source = row.get("source", {})
    return len(source.get("prompt_en") or "") + len(source.get("response_en") or "")


def build(root: Path, reviewer: str, size: int, oversized_chars: int) -> dict:
    handoff = root / "data" / "revision_handoff" / f"revision_{reviewer}.jsonl"
    output = root / "data" / "revision_handoff" / f"logical_{reviewer}_batches"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    done = completed_uids(root)
    rows = [row for _, row, _ in read_jsonl(handoff) if row["record_uid"] not in done]
    manifest = []
    for index in range(0, len(rows), size):
        chunk = rows[index:index + size]
        number = index // size + 1
        file_name = f"batch_{number:03d}.jsonl"
        enriched = []
        for position, row in enumerate(chunk, 1):
            copy = dict(row)
            chars = source_chars(copy)
            copy["logical_review"] = {
                "reviewer": reviewer,
                "batch": number,
                "position": position,
                "source_chars": chars,
                "oversized": chars > oversized_chars,
                "candidate_first": True,
            }
            enriched.append(copy)
        write_jsonl(output / file_name, enriched)
        manifest.append({
            "file": file_name,
            "records": len(enriched),
            "record_uids": [row["record_uid"] for row in enriched],
            "source_chars": sum(source_chars(row) for row in enriched),
            "oversized_records": [
                row["record_uid"] for row in enriched if source_chars(row) > oversized_chars
            ],
        })
    write_jsonl(output / "manifest.jsonl", manifest)
    return {
        "reviewer": reviewer,
        "remaining_records": len(rows),
        "logical_batches": len(manifest),
        "output": str(output),
        "oversized_records": sum(len(row["oversized_records"]) for row in manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--size", default=20, type=int)
    parser.add_argument("--oversized-chars", default=55_000, type=int)
    args = parser.parse_args()
    if args.size < 1:
        raise SystemExit("--size must be positive")
    root = args.root.resolve()
    summary = [build(root, reviewer, args.size, args.oversized_chars) for reviewer in ("terra", "luna")]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
