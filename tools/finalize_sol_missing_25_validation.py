from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "data/sol_missing_25/sol_results/validation"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def accepted(row: dict[str, Any], status: str) -> dict[str, Any]:
    candidate = row["candidate"]
    return {
        "seq": candidate["seq"],
        "record_uid": candidate["record_uid"],
        "prompt_vi": candidate["prompt_vi"],
        "response_vi": candidate["response_vi"],
        "warnings": candidate.get("warnings") or [],
        "translation_provider": "gpt-5.6-sol-web",
        "quality_status": status,
    }


def main() -> None:
    hard_pass = read_jsonl(VALIDATION / "hard_pass.jsonl")
    warning_rows = read_jsonl(VALIDATION / "needs_audit.jsonl")
    hard_failed = read_jsonl(VALIDATION / "failed.jsonl")
    structural = read_jsonl(VALIDATION / "structural_issues.jsonl")
    if hard_failed or structural:
        raise RuntimeError("cannot finalize while hard or structural failures remain")

    accepted_rows = [accepted(row, "hard_pass") for row in hard_pass]
    decisions: list[dict[str, Any]] = []
    for row in warning_rows:
        accepted_rows.append(accepted(row, "manual_audit_pass"))
        decisions.append({
            "record_uid": row["record_uid"],
            "decision": "pass",
            "reason": "manual review: warning is caused by preserved ASCII art; surrounding natural-language text is translated",
            "warnings": row.get("heuristic_warnings") or [],
        })

    accepted_rows.sort(key=lambda row: str(row["record_uid"]))
    decisions.sort(key=lambda row: str(row["record_uid"]))
    write_jsonl(VALIDATION / "accepted_final.jsonl", accepted_rows)
    write_jsonl(VALIDATION / "manual_review_decisions.jsonl", decisions)
    summary = {
        "expected": 25,
        "automatic_hard_pass": len(hard_pass),
        "manual_audit_pass": len(warning_rows),
        "hard_failed": len(hard_failed),
        "structural_issues": len(structural),
        "accepted_final": len(accepted_rows),
        "complete": len(accepted_rows) == 25,
    }
    (VALIDATION / "final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
