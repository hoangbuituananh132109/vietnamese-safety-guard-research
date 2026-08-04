from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class BucketRule:
    max_items: int
    max_chars: int | None


RULES = {
    "normal": BucketRule(30, 50_000),
    "near_tail": BucketRule(20, 50_000),
    "tail": BucketRule(14, 50_000),
    "high_tail": BucketRule(2, 50_000),
    "oversized": BucketRule(1, None),
}

BUCKET_ORDER = {name: index for index, name in enumerate(RULES)}


def source_chars(row: dict[str, Any]) -> int:
    return len(row.get("prompt") or "") + len(row.get("response") or "")


def length_bucket(chars: int) -> str:
    if chars <= 1695:
        return "normal"
    if chars <= 2342:
        return "near_tail"
    if chars <= 3772:
        return "tail"
    if chars <= 25_000:
        return "high_tail"
    return "oversized"


def group_for_batching(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group records by length class while keeping stable order inside each class.

    The final dataset order is reconstructed separately from the source rows. This
    ordering only prevents an interleaved source file from prematurely closing a
    batch every time its next record belongs to a different length bucket.
    """
    return sorted(
        rows,
        key=lambda row: BUCKET_ORDER[
            row.get("length_bucket") or length_bucket(source_chars(row))
        ],
    )


def make_batches(rows: Iterable[dict[str, Any]], prefix: str = "pilot") -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    current_bucket: str | None = None
    for row in rows:
        bucket = row.get("length_bucket") or length_bucket(source_chars(row))
        rule = RULES[bucket]
        chars = source_chars(row)
        must_close = bool(current) and (
            bucket != current_bucket
            or len(current) >= rule.max_items
            or (rule.max_chars is not None and current_chars + chars > rule.max_chars)
        )
        if must_close:
            batches.append(current)
            current, current_chars = [], 0
        current_bucket = bucket
        current.append(row)
        current_chars += chars
        if len(current) >= rule.max_items or rule.max_items == 1:
            batches.append(current)
            current, current_chars, current_bucket = [], 0, None
    if current:
        batches.append(current)
    return batches
