#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from translator.jsonl_io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--indices", required=True, help="One-based comma-separated record indices")
    args = parser.parse_args()
    wanted = {int(value) for value in args.indices.split(",") if value.strip()}
    rows = [row for _, row, _ in read_jsonl(args.source)]
    selected = [row for index, row in enumerate(rows, 1) if index in wanted]
    missing = sorted(wanted - set(range(1, len(rows) + 1)))
    if missing:
        raise SystemExit(f"indices outside source: {missing}")
    write_jsonl(args.output, selected)
    print(f"repair_set_rows={len(selected)} indices={sorted(wanted)}")


if __name__ == "__main__":
    main()
