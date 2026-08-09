from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "data/luna_remaining_20260803/queue.jsonl"
RUN = ROOT / "data/luna_remaining_20260803/run_light_retry2"
OUT = ROOT / "data/luna_remaining_20260803/recovery"


def tolerant_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    if not path.exists():
        return rows, bad
    for raw in path.read_bytes().splitlines():
        try:
            rows.append(json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            bad += 1
    return rows, bad


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def latest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: str(row.get("recorded_at") or row.get("attempted_at") or ""))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    queue, queue_bad = tolerant_jsonl(QUEUE)
    queue_by_uid = {str(row["record_uid"]): (seq, row) for seq, row in enumerate(queue, 1)}

    passed, passed_bad = tolerant_jsonl(RUN / "passed.jsonl")
    audit, audit_bad = tolerant_jsonl(RUN / "needs_audit.jsonl")
    exhausted, exhausted_bad = tolerant_jsonl(RUN / "exhausted_normal.jsonl")
    tail, tail_bad = tolerant_jsonl(RUN / "tail_escalation.jsonl")
    attempts, attempts_bad = tolerant_jsonl(RUN / "attempts.jsonl")

    accepted = {str(row["record_uid"]) for row in passed + audit}
    terminal_failures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exhausted + tail:
        uid = str(row["record_uid"])
        if uid not in accepted:
            terminal_failures[uid].append(row)

    attempts_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        attempts_by_uid[str(row["record_uid"])].append(row)

    terminal_uids = accepted | set(terminal_failures)
    remaining = [uid for uid in queue_by_uid if uid not in terminal_uids]
    continuation: list[dict[str, Any]] = []
    pre_web: list[dict[str, Any]] = [latest(rows) for rows in terminal_failures.values()]
    forced_after_two = 0
    for uid in remaining:
        attempt_rows = attempts_by_uid.get(uid, [])
        _, source = queue_by_uid[uid]
        if len(attempt_rows) < 2:
            continuation.append(source)
            continue
        last = latest(attempt_rows)
        bucket = str(last.get("length_bucket") or source.get("length_bucket") or "normal")
        pre_web.append({
            "record_uid": uid,
            "seq": queue_by_uid[uid][0],
            "route": last.get("route"),
            "length_bucket": bucket,
            "attempts": 2,
            "actual_attempt_rows_before_repair": len(attempt_rows),
            "batch_id": last.get("batch_id"),
            "model": "gpt-5.6-luna",
            "reasoning_effort": "low",
            "hard_errors": last.get("hard_errors") or ["missing candidate after at least two Luna attempts"],
            "heuristic_warnings": last.get("heuristic_warnings") or [],
            "candidate": last.get("candidate"),
            "translation_status": "exhausted_normal" if bucket in {"normal", "near_tail"} else "tail_escalation",
            "recovered_from_overlapped_run": True,
        })
        forced_after_two += 1

    # One deterministic row per UID. Accepted output is only an inventory; original full
    # candidates remain in passed/audit and are never sent to Sol.
    accepted_inventory = [{"record_uid": uid} for uid in sorted(accepted)]
    pre_web_by_uid = {str(row["record_uid"]): row for row in pre_web if str(row["record_uid"]) not in accepted}
    write_jsonl(OUT / "continuation_queue.jsonl", continuation)
    write_jsonl(OUT / "pre_web_failures.jsonl", list(pre_web_by_uid.values()))
    write_jsonl(OUT / "accepted_unique.jsonl", accepted_inventory)
    manifest = {
        "source_records": len(queue),
        "accepted_unique": len(accepted),
        "terminal_failure_unique": len(terminal_failures),
        "remaining_before_repair": len(remaining),
        "forced_to_web_after_two_or_more_attempt_rows": forced_after_two,
        "continuation_one_attempt_remaining": len(continuation),
        "pre_web_failure_unique": len(pre_web_by_uid),
        "malformed_lines_skipped": {
            "queue": queue_bad, "passed": passed_bad, "audit": audit_bad,
            "exhausted": exhausted_bad, "tail": tail_bad, "attempts": attempts_bad,
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
