from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.luna_overnight_runner import validate_item
from translator.models import TranslationInputItem


BASE = ROOT / "data/sol_missing_25"
RESULTS = BASE / "sol_results"
OUT = RESULTS / "validation"
PREFIX = "nemotron-sol-missing-25-"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    expected: dict[str, dict[str, Any]] = {}
    expected_batch: dict[str, str] = {}
    for path in sorted(BASE.glob("batch-*.jsonl")):
        index = int(path.stem.split("-")[-1])
        batch_id = f"{PREFIX}{index:03d}"
        for row in read_jsonl(path):
            uid = str(row["record_uid"])
            if uid in expected:
                raise RuntimeError(f"duplicate source UID: {uid}")
            expected[uid] = row
            expected_batch[uid] = batch_id

    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    batch_summaries: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob(f"{PREFIX}*-browser-result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = [item for item in payload.get("items") or [] if isinstance(item, dict)]
        batch_summaries.append({
            "batch_id": payload.get("batch_id"),
            "items": len(items),
            "structure_pass": bool((payload.get("validation") or {}).get("structure_pass", True)),
            "file": path.name,
        })
        for item in items:
            by_uid[str(item.get("record_uid") or "")].append(item)

    missing = sorted(set(expected) - set(by_uid))
    extra = sorted(set(by_uid) - set(expected) - {""})
    duplicates = {uid: len(items) for uid, items in by_uid.items() if uid and len(items) > 1}
    structural_issues = (
        [{"record_uid": uid, "issue": "missing"} for uid in missing]
        + [{"record_uid": uid, "issue": "extra"} for uid in extra]
        + [{"record_uid": uid, "issue": "duplicate", "occurrences": count} for uid, count in duplicates.items()]
    )
    if "" in by_uid:
        structural_issues.append({"record_uid": None, "issue": "blank_uid", "occurrences": len(by_uid[""])})

    hard_pass: list[dict[str, Any]] = []
    needs_audit: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    hard_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    aggregate_items: list[dict[str, Any]] = []

    for uid in sorted(set(expected) & set(by_uid), key=lambda value: (str(expected[value]["source_split"]), int(expected[value]["seq"]))):
        if len(by_uid[uid]) != 1:
            continue
        source = expected[uid]
        raw = by_uid[uid][0]
        candidate = {key: raw.get(key) for key in ("seq", "record_uid", "prompt_vi", "response_vi", "warnings")}
        candidate["warnings"] = candidate.get("warnings") or []
        request = TranslationInputItem(
            seq=int(source["seq"]),
            record_uid=uid,
            prompt=str(source.get("prompt_en") or ""),
            response=source.get("response_en"),
        )
        hard, warnings = validate_item(request, candidate, expected_batch[uid])
        row = {
            "seq": int(source["seq"]),
            "record_uid": uid,
            "batch_id": expected_batch[uid],
            "source_split": source.get("source_split"),
            "length_bucket": source.get("length_bucket"),
            "hard_errors": hard,
            "heuristic_warnings": warnings,
            "candidate": candidate,
        }
        hard_counts.update(hard)
        warning_counts.update(warnings)
        (failed if hard else needs_audit if warnings else hard_pass).append(row)
        aggregate_items.append({**candidate, "_sol_batch_id": expected_batch[uid]})

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "hard_pass.jsonl", hard_pass)
    write_jsonl(OUT / "needs_audit.jsonl", needs_audit)
    write_jsonl(OUT / "failed.jsonl", failed)
    write_jsonl(OUT / "structural_issues.jsonl", structural_issues)
    aggregate = {
        "expected_batches": len(batch_summaries),
        "expected_records": len(expected),
        "parsed_items": len(aggregate_items),
        "batches": batch_summaries,
        "items": aggregate_items,
    }
    (RESULTS / "sol-missing-25-all-results.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "expected_records": len(expected),
        "result_batches": len(batch_summaries),
        "unique_output_uids": len({uid for uid in by_uid if uid}),
        "missing_uids": len(missing),
        "extra_uids": len(extra),
        "duplicate_uids": len(duplicates),
        "structural_issues": len(structural_issues),
        "hard_pass_no_warning": len(hard_pass),
        "needs_audit_warning_only": len(needs_audit),
        "hard_failed": len(failed),
        "hard_error_counts": hard_counts.most_common(),
        "warning_counts": warning_counts.most_common(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
