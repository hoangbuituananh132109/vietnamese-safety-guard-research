from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator.dashboard import DashboardData


def best_legacy_candidate(row: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    uid = row["record_uid"]
    choices: list[tuple[int, dict[str, Any], list[str]]] = []
    for candidate in row.get("candidates") or []:
        items = (candidate.get("response") or {}).get("items") or []
        item = next((value for value in items if value.get("record_uid") == uid), None)
        if not isinstance(item, dict):
            continue
        errors = [str(error) for error in candidate.get("validation_errors") or [] if uid in str(error)]
        choices.append((len(errors), item, errors))
    if choices:
        _, item, errors = min(choices, key=lambda choice: choice[0])
        return item, errors or [str(error) for error in row.get("validation_errors") or []]
    return None, [str(error) for error in row.get("validation_errors") or []]


def bootstrap(root: Path, reviewer: str) -> dict[str, int]:
    data = DashboardData(root)
    completed = data.completed_review_uids()
    archived = root / "data" / "revision_handoff" / f"{reviewer}_web_results"
    snapshots = records = skipped = 0
    for batch, _ in enumerate(data.review_manifest(reviewer), 1):
        # A web-result file proves this batch was processed before persistent
        # failure quarantine existed. Never infer attempts for untouched batches.
        if not (archived / f"batch_{batch:03d}.jsonl").exists():
            continue
        target = data.review_failure_path(reviewer, batch)
        if target.exists():
            skipped += 1
            continue
        pending = [row for row in data.review_rows(reviewer, batch) if row["record_uid"] not in completed]
        if not pending:
            continue
        items: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for row in pending:
            candidate, errors = best_legacy_candidate(row)
            if candidate is None:
                results.append({
                    "record_uid": row["record_uid"],
                    "status": "missing",
                    "errors": errors or ["No usable legacy candidate was recoverable."],
                })
            else:
                items.append(candidate)
                results.append({
                    "record_uid": row["record_uid"],
                    "status": "fail",
                    "errors": errors or ["Legacy candidate remained pending after web review."],
                })
        raw = json.dumps({"batch_id": f"recovered-{reviewer}-{batch:03d}", "items": items}, ensure_ascii=False)
        validation = {
            "expected": len(pending),
            "received": len(items),
            "pass_count": 0,
            "fail_count": len(results),
            "results": results,
        }
        data.persist_review_failures(reviewer, batch, raw, "recovered-legacy-handoff-candidate", validation)
        snapshot = json.loads(target.read_text(encoding="utf-8"))
        snapshot["recovered_from"] = "logical_handoff_candidate"
        snapshot["recovery_note"] = (
            "The original rejected web output predated persistent quarantine and was unavailable. "
            "Candidate/source/errors were recovered from the logical handoff for later human review."
        )
        for failure in snapshot.get("failures", []):
            failure["recovered_from"] = "logical_handoff_candidate"
        data._atomic_write_json(target, snapshot)
        snapshots += 1
        records += len(results)
    return {"snapshots": snapshots, "records": records, "skipped_existing": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover pre-quarantine web-review failures from logical handoff data.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--reviewer", choices=("terra", "luna"), default="terra")
    args = parser.parse_args()
    result = bootstrap(Path(args.root).resolve(), args.reviewer)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
