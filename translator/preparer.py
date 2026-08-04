from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from .batching import length_bucket, source_chars
from .jsonl_io import read_jsonl, write_jsonl


def prepare_rows(source: Path, split: str):
    for line_no, row, physical in read_jsonl(source):
        digest = hashlib.sha256(physical.encode("utf-8")).hexdigest()
        item = dict(row)
        chars = source_chars(row)
        item.update({
            "record_uid": f"en-{split}-{line_no:08d}-{digest[:12]}",
            "source_split": split,
            "source_language": "en",
            "source_line_number": line_no,
            "source_line_sha256": digest,
            "source_id": str(row.get("id") or ""),
            "source_chars": chars,
            "length_bucket": length_bucket(chars),
            "is_above_p95": chars > 2342,
            "oversized_single_record": chars > 25_000,
        })
        yield item


def build(source: Path, output: Path, manifest_path: Path, split: str) -> dict[str, Any]:
    counts = collections.Counter()

    def rows():
        for item in prepare_rows(source, split):
            counts["rows"] += 1
            counts[f"bucket:{item['length_bucket']}"] += 1
            counts[f"tag:{item.get('tag')}"] += 1
            counts["empty_prompt"] += item.get("prompt") == ""
            counts["null_response"] += item.get("response") is None
            counts["empty_response"] += item.get("response") == ""
            yield item

    write_jsonl(output, rows())
    manifest = {
        "source_file": str(source), "prepared_file": str(output), "split": split,
        "translation_prompt_version": "nemotron-en-vi-v10", "counts": dict(counts),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.manifest, args.split), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
