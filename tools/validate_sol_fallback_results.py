from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from luna_overnight_runner import validate_item

from translator.models import TranslationInputItem


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/sol_fallback_after_luna"
RESULT = SOURCE_DIR / "sol_results/sol-fallback-all-results.json"
OUT = SOURCE_DIR / "sol_results/validation"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").split("\n"):
        if raw.strip():
            rows.append(json.loads(raw))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def error_category(error: str) -> str:
    if "strong profanity was softened" in error:
        return "profanity_softened"
    if "mostly untranslated English/leet" in error:
        return "mostly_untranslated_english_or_leet"
    if "leetspeak remains English" in error:
        return "leetspeak_remains_english"
    if "English leetspeak line was copied unchanged" in error:
        return "english_leetspeak_copied_unchanged"
    return "other"


def main() -> None:
    expected: dict[str, dict[str, Any]] = {}
    expected_batch: dict[str, str] = {}
    for path in sorted(SOURCE_DIR.glob("batch-*.jsonl")):
        batch_id = f"nemotron-sol-fallback-after-luna-{int(path.stem.split('-')[-1]):03d}"
        for row in read_jsonl(path):
            uid = str(row["record_uid"])
            if uid in expected:
                raise RuntimeError(f"duplicate UID in source batches: {uid}")
            expected[uid] = row
            expected_batch[uid] = batch_id

    export = json.loads(RESULT.read_text(encoding="utf-8"))
    exported_items = [row for row in export.get("items") or [] if isinstance(row, dict)]
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    blank_uid: list[dict[str, Any]] = []
    for item in exported_items:
        uid = str(item.get("record_uid") or "")
        if uid:
            by_uid[uid].append(item)
        else:
            blank_uid.append(item)

    expected_uids = set(expected)
    output_uids = set(by_uid)
    missing = sorted(expected_uids - output_uids)
    extra = sorted(output_uids - expected_uids)
    duplicates = {uid: len(rows) for uid, rows in by_uid.items() if len(rows) > 1}

    structure_issues: list[dict[str, Any]] = []
    for uid in missing:
        structure_issues.append({"record_uid": uid, "issue": "missing", "expected_batch_id": expected_batch[uid]})
    for uid in extra:
        structure_issues.append({"record_uid": uid, "issue": "extra", "occurrences": len(by_uid[uid])})
    for uid, count in duplicates.items():
        structure_issues.append({"record_uid": uid, "issue": "duplicate", "occurrences": count})
    for item in blank_uid:
        structure_issues.append({"record_uid": None, "issue": "blank_uid", "candidate": item})

    hard_pass: list[dict[str, Any]] = []
    needs_audit: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    hard_error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    hard_category_records: Counter[str] = Counter()

    for uid in sorted(expected_uids & output_uids, key=lambda value: int(expected[value].get("seq") or 0)):
        if len(by_uid[uid]) != 1:
            continue
        source = expected[uid]
        raw_candidate = by_uid[uid][0]
        candidate = {key: raw_candidate.get(key) for key in ("seq", "record_uid", "prompt_vi", "response_vi", "warnings")}
        candidate["warnings"] = candidate.get("warnings") or []
        request_item = TranslationInputItem(
            seq=int(source["seq"]),
            record_uid=uid,
            prompt=str(source.get("prompt_en") or ""),
            response=source.get("response_en"),
        )
        hard, warnings = validate_item(request_item, candidate, expected_batch[uid])
        row = {
            "seq": int(source["seq"]),
            "record_uid": uid,
            "batch_id": expected_batch[uid],
            "source_split": source.get("source_split"),
            "length_bucket": source.get("length_bucket"),
            "route": source.get("route"),
            "hard_errors": hard,
            "heuristic_warnings": warnings,
            "candidate": candidate,
        }
        for error in hard:
            hard_error_counts[error] += 1
        for warning in warnings:
            warning_counts[warning] += 1
        if hard:
            for category in {error_category(error) for error in hard}:
                hard_category_records[category] += 1
            failed.append(row)
        elif warnings:
            needs_audit.append(row)
        else:
            hard_pass.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "hard_pass.jsonl", hard_pass)
    write_jsonl(OUT / "needs_audit.jsonl", needs_audit)
    write_jsonl(OUT / "failed.jsonl", failed)
    write_jsonl(OUT / "structural_issues.jsonl", structure_issues)

    batch_reports = export.get("batches") or []
    browser_parse_errors = [row for row in batch_reports if row.get("parse_error")]
    browser_count_mismatches = [
        row for row in batch_reports
        if int(row.get("parsed") or 0) != int(row.get("expected") or 0)
    ]
    summary = {
        "expected_records": len(expected),
        "exported_items": len(exported_items),
        "unique_output_uids": len(output_uids),
        "missing_uids": len(missing),
        "extra_uids": len(extra),
        "duplicate_uids": len(duplicates),
        "blank_uids": len(blank_uid),
        "browser_parse_error_batches": len(browser_parse_errors),
        "browser_count_mismatch_batches": len(browser_count_mismatches),
        "hard_pass_no_warning": len(hard_pass),
        "needs_audit_warning_only": len(needs_audit),
        "hard_failed": len(failed),
        "unclassified_due_to_structure": len(expected) - len(hard_pass) - len(needs_audit) - len(failed),
        "hard_error_counts": hard_error_counts.most_common(),
        "hard_error_category_records": hard_category_records.most_common(),
        "hard_failed_by_route": Counter(str(row.get("route")) for row in failed).most_common(),
        "hard_failed_by_bucket": Counter(str(row.get("length_bucket")) for row in failed).most_common(),
        "hard_failed_by_split": Counter(str(row.get("source_split")) for row in failed).most_common(),
        "warning_counts": warning_counts.most_common(),
        "browser_parse_errors": browser_parse_errors,
        "browser_count_mismatches": browser_count_mismatches,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
