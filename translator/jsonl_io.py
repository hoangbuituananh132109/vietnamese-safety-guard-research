from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def _parse_constant(_: str) -> None:
    return None


def read_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any], str]]:
    """Stream physical JSONL lines; never use splitlines()."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            yield line_number, json.loads(line, parse_constant=_parse_constant), line


def safe_json_dumps(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return text.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def append_jsonl(path: str | Path, value: dict[str, Any], fsync: bool = True) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(safe_json_dumps(value) + "\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], atomic: bool = True) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp") if atomic else target
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(safe_json_dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if atomic:
        os.replace(temp, target)

