from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/sol_fallback_after_luna"
VALIDATION = BASE / "sol_results/validation"

REPAIR_REASONS = {
    "en-train-00020107-7cd91d92b3ef": "natural-language jailbreak inside a fenced block remained entirely English",
    "en-train-00012179-fb3f7c715e03": "source URL changed from impact-market to impact_market",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    expected: dict[str, dict[str, Any]] = {}
    for path in sorted(BASE.glob("batch-*.jsonl")):
        for row in read_jsonl(path):
            expected[str(row["record_uid"])] = row

    clean = read_jsonl(VALIDATION / "hard_pass.jsonl")
    audit = read_jsonl(VALIDATION / "needs_audit.jsonl")
    hard_failed = read_jsonl(VALIDATION / "failed.jsonl")
    if hard_failed:
        raise RuntimeError("hard-failed records remain; review before finalization")

    accepted_rows: list[dict[str, Any]] = []
    review_decisions: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []

    for row in clean:
        candidate = row["candidate"]
        accepted_rows.append({
            "seq": candidate["seq"], "record_uid": candidate["record_uid"],
            "prompt_vi": candidate["prompt_vi"], "response_vi": candidate["response_vi"],
            "translation_provider": "gpt-5.6-sol-web",
            "quality_status": "hard_pass",
        })

    for row in audit:
        uid = str(row["record_uid"])
        candidate = row["candidate"]
        if uid in REPAIR_REASONS:
            source = expected[uid]
            decision = {
                "record_uid": uid,
                "decision": "repair",
                "reason": REPAIR_REASONS[uid],
                "warnings": row.get("heuristic_warnings") or [],
            }
            review_decisions.append(decision)
            repair_rows.append({
                "seq": int(source["seq"]),
                "record_uid": uid,
                "source_split": source.get("source_split"),
                "length_bucket": source.get("length_bucket"),
                "route": source.get("route"),
                "prompt_en": source.get("prompt_en"),
                "response_en": source.get("response_en"),
                "previous_prompt_vi": candidate.get("prompt_vi"),
                "previous_response_vi": candidate.get("response_vi"),
                "validator_errors": [REPAIR_REASONS[uid]],
                "validator_warnings": row.get("heuristic_warnings") or [],
            })
        else:
            review_decisions.append({
                "record_uid": uid,
                "decision": "pass",
                "reason": "manual review: preservation-scoped code/data/URL/proper names or meaning-preserving locale rendering",
                "warnings": row.get("heuristic_warnings") or [],
            })
            accepted_rows.append({
                "seq": candidate["seq"], "record_uid": candidate["record_uid"],
                "prompt_vi": candidate["prompt_vi"], "response_vi": candidate["response_vi"],
                "translation_provider": "gpt-5.6-sol-web",
                "quality_status": "manual_audit_pass",
            })

    accepted_rows.sort(key=lambda row: int(row["seq"]))
    repair_rows.sort(key=lambda row: int(row["seq"]))
    review_decisions.sort(key=lambda row: int(expected[row["record_uid"]]["seq"]))
    write_jsonl(VALIDATION / "accepted_final.jsonl", accepted_rows)
    write_jsonl(VALIDATION / "repair_queue.jsonl", repair_rows)
    write_jsonl(VALIDATION / "manual_review_decisions.jsonl", review_decisions)

    summary = {
        "expected": len(expected),
        "automatic_hard_pass": len(clean),
        "manual_audit_total": len(audit),
        "manual_audit_pass": sum(row["decision"] == "pass" for row in review_decisions),
        "repair_required": len(repair_rows),
        "accepted_final": len(accepted_rows),
        "coverage": len(accepted_rows) + len(repair_rows),
        "repair_uids": [row["record_uid"] for row in repair_rows],
    }
    (VALIDATION / "final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
