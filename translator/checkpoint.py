from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .jsonl_io import append_jsonl, read_jsonl


def load_checkpoint(path: str | Path) -> dict[str, dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for _, row, _ in read_jsonl(target):
        uid = row["record_uid"]
        if uid in completed and json.dumps(completed[uid], sort_keys=True) != json.dumps(row, sort_keys=True):
            raise ValueError(f"Conflicting duplicate record_uid in checkpoint: {uid}")
        completed[uid] = row
    return completed


def save_completed(path: str | Path, row: dict[str, Any]) -> None:
    append_jsonl(path, row, fsync=True)

