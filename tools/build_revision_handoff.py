from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from translator.checkpoint import load_checkpoint
from translator.full_run import paths
from translator.jsonl_io import read_jsonl, write_jsonl


HEAVY = (
    "mostly untranslated", "leetspeak", "copied unchanged", "identity slur",
    "missing record_uid", "unknown record_uid", "duplicate record_uid",
    "missing response_vi", "empty prompt_vi",
)
FOCUSED = (
    "strong profanity was softened", "redaction marker", "json/schema keys changed",
    "compromised means", "explicit photos", "graphic detail", "schizophrenia",
    "null response", "empty response", "seq mismatch", "batch_id mismatch",
)


def route(errors: list[str], candidate_count: int) -> tuple[str, str]:
    low = "\n".join(errors).lower()
    if candidate_count == 0:
        return "terra", "no structured candidate; reconstruct translation from source"
    if any(marker in low for marker in HEAVY):
        return "terra", "heavy semantic/coverage/structure failure"
    if any(marker in low for marker in FOCUSED):
        return "luna", "focused constrained correction"
    if len(errors) > 2:
        return "terra", "multiple unresolved validation failures"
    return "luna", "candidate comparison and focused review"


def main() -> None:
    root = Path(".").resolve()
    out_dir = root / "data" / "revision_handoff"
    report_dir = root / "reports" / "full_run"
    checkpoint_status = Counter()
    source_total = 0
    unresolved_rows: list[dict] = []
    split_summary = defaultdict(Counter)

    for group in range(1, 6):
        for split in ("train", "valid", "test"):
            p = paths(root, split, group - 1)
            source = {
                row["record_uid"]: row
                for index, (_, row, _) in enumerate(read_jsonl(p["input"]))
                if index % 5 == group - 1
            }
            source_total += len(source)
            completed = load_checkpoint(p["checkpoint"])
            for uid, row in completed.items():
                if uid in source:
                    status = row.get("translation_status", "unknown")
                    checkpoint_status[status] += 1
                    split_summary[(group, split)][status] += 1

            archives = {}
            candidate_path = p["failed"].with_name(p["failed"].name.replace("_failed.jsonl", "_candidates.jsonl"))
            if candidate_path.exists():
                archives = {row["candidate_archive_id"]: row for _, row, _ in read_jsonl(candidate_path)}

            latest_failed = {}
            if p["failed"].exists():
                for _, row, _ in read_jsonl(p["failed"]):
                    uid = row.get("record_uid")
                    if uid in source and uid not in completed:
                        latest_failed[uid] = row

            for uid, failed in latest_failed.items():
                archive = archives.get(failed.get("candidate_archive_id"), {})
                candidates = archive.get("candidates", [])
                errors = sorted({
                    error
                    for candidate in candidates
                    for error in (candidate.get("validation_errors") or [])
                    if uid in error or not any(prefix in error for prefix in ("en-train-", "en-valid-", "en-test-"))
                })
                if not errors and failed.get("error"):
                    errors = [str(failed["error"])]
                model_route, reason = route(errors, len(candidates))
                unresolved_rows.append({
                    "record_uid": uid,
                    "source_split": split,
                    "group": group,
                    "current_status": failed.get("translation_status", "failed"),
                    "recommended_model": model_route,
                    "routing_reason": reason,
                    "source": {
                        "prompt_en": source[uid].get("prompt"),
                        "response_en": source[uid].get("response"),
                        "violated_categories": source[uid].get("violated_categories"),
                        "prompt_label": source[uid].get("prompt_label"),
                        "response_label": source[uid].get("response_label"),
                    },
                    "candidate_archive_id": failed.get("candidate_archive_id"),
                    "candidate_count": len(candidates),
                    "validation_errors": errors,
                    "candidates": candidates,
                    "required_output": {
                        "record_uid": uid,
                        "prompt_vi": "string",
                        "response_vi": "string|null",
                        "review_status": "revised",
                        "fix_notes": ["string"],
                    },
                })
                split_summary[(group, split)][f"unresolved_{model_route}"] += 1

    luna = [row for row in unresolved_rows if row["recommended_model"] == "luna"]
    terra = [row for row in unresolved_rows if row["recommended_model"] == "terra"]
    write_jsonl(out_dir / "revision_luna.jsonl", luna)
    write_jsonl(out_dir / "revision_terra.jsonl", terra)
    write_jsonl(out_dir / "revision_all.jsonl", unresolved_rows)

    usable = sum(checkpoint_status.values())
    summary = {
        "source_records": source_total,
        "usable_checkpoint_records": usable,
        "usable_rate": usable / source_total if source_total else 0,
        "checkpoint_status": dict(checkpoint_status),
        "unresolved_records": len(unresolved_rows),
        "revision_luna_records": len(luna),
        "revision_terra_records": len(terra),
        "accounting_difference": source_total - usable - len(unresolved_rows),
        "by_group_split": {
            f"g{group}_{split}": dict(values)
            for (group, split), values in sorted(split_summary.items())
        },
        "files": {
            "all": str(out_dir / "revision_all.jsonl"),
            "luna": str(out_dir / "revision_luna.jsonl"),
            "terra": str(out_dir / "revision_terra.jsonl"),
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "revision_handoff_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
