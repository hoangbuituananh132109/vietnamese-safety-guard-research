#!/usr/bin/env python3
from pathlib import Path

from translator.jsonl_io import read_jsonl, write_jsonl


INDICES = {6, 14, 27, 42, 47, 50}


def main() -> None:
    rows = [r for _, r, _ in read_jsonl("data/pilot/nemotron_en_train_pilot_50_diverse_v2.jsonl")]
    selected = [r for i, r in enumerate(rows, 1) if i in INDICES]
    write_jsonl("data/pilot/nemotron_en_train_pilot_6_repair_v4.jsonl", selected)
    print(f"repair_set_rows={len(selected)}")


if __name__ == "__main__": main()
