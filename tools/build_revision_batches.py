from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from translator.jsonl_io import read_jsonl, write_jsonl


def estimate(row: dict) -> int:
    source = row.get("source") or {}
    candidate_chars = sum(
        len(str(item)) for candidate in row.get("candidates", [])
        for item in (candidate.get("response", {}).get("items", []))
    )
    return len(source.get("prompt_en") or "") + len(source.get("response_en") or "") + candidate_chars


def build(queue: Path, output: Path, max_records: int, max_chars: int) -> int:
    rows = [row for _, row, _ in read_jsonl(queue)]
    rows.sort(key=lambda row: (row.get("source_split", ""), estimate(row)), reverse=True)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    batches: list[list[dict]] = []
    sizes: list[int] = []
    for row in rows:
        size = estimate(row)
        placed = False
        for index, batch in enumerate(batches):
            if len(batch) < max_records and sizes[index] + size <= max_chars:
                batch.append(row)
                sizes[index] += size
                placed = True
                break
        if not placed:
            batches.append([row])
            sizes.append(size)
    manifest = []
    for index, (batch, size) in enumerate(zip(batches, sizes), 1):
        name = f"batch_{index:03d}.jsonl"
        write_jsonl(output / name, batch)
        manifest.append({"file": name, "records": len(batch), "estimated_chars": size, "record_uids": [x["record_uid"] for x in batch]})
    write_jsonl(output / "manifest.jsonl", manifest)
    return len(batches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    handoff = root / "data" / "revision_handoff"
    terra = build(handoff / "revision_terra.jsonl", handoff / "terra_batches", max_records=4, max_chars=55_000)
    luna = build(handoff / "revision_luna.jsonl", handoff / "luna_batches", max_records=12, max_chars=55_000)
    print(f"terra_batches={terra} luna_batches={luna}")


if __name__ == "__main__":
    main()
