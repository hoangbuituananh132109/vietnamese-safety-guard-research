from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.luna_overnight_runner import source_fields, validate_item
from translator.models import TranslationInputItem


OUT = ROOT / "data/luna_remaining_20260803"
SPLITS = {
    "train": (ROOT / "data/final/nemotron_train_en_vi_v10_final.jsonl", ROOT / "data/luna_overnight/nemotron_train_20260801"),
    "valid": (ROOT / "data/final/nemotron_valid_en_vi_v10_final.jsonl", ROOT / "data/luna_overnight/nemotron_valid_20260802"),
    "test": (ROOT / "data/final/nemotron_test_en_vi_v10_final.jsonl", ROOT / "data/luna_overnight/nemotron_test_20260802"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    queue: list[dict[str, Any]] = []
    recovered: list[dict[str, Any]] = []
    sol_existing: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}

    for split, (source_path, run_dir) in SPLITS.items():
        source_rows = read_jsonl(source_path)
        terminal: dict[str, tuple[str, dict[str, Any]]] = {}
        for filename, status in (
            ("passed.jsonl", "pass"), ("needs_audit.jsonl", "audit"),
            ("exhausted_normal.jsonl", "exhausted"), ("tail_escalation.jsonl", "tail"),
        ):
            for row in read_jsonl(run_dir / filename):
                terminal[str(row["record_uid"])] = (status, row)
        latest_attempt: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(run_dir / "attempts.jsonl"):
            latest_attempt[str(row["record_uid"])] = row

        split_counts = {"accepted": 0, "recovered": 0, "luna_queue": 0, "sol_existing": 0}
        for original_seq, source in enumerate(source_rows, 1):
            uid = str(source["record_uid"])
            status, terminal_row = terminal.get(uid, ("missing", {}))
            if status in {"pass", "audit"}:
                split_counts["accepted"] += 1
                continue

            attempt = latest_attempt.get(uid) or terminal_row
            candidate = attempt.get("candidate") if isinstance(attempt, dict) else None
            hard_errors: list[str] = []
            warnings: list[str] = []
            if isinstance(candidate, dict):
                prompt, response = source_fields(source)
                expected = TranslationInputItem(
                    seq=original_seq, record_uid=uid, prompt=prompt, response=response
                )
                candidate_for_validation = dict(candidate)
                candidate_for_validation["seq"] = original_seq
                hard_errors, warnings = validate_item(
                    expected, candidate_for_validation, "remaining-queue-revalidation"
                )
                if not hard_errors:
                    recovered.append({
                        "record_uid": uid, "source_split": split, "original_seq": original_seq,
                        "candidate": candidate_for_validation, "warnings": warnings,
                        "recovered_by": "current_validator_revalidation",
                    })
                    split_counts["recovered"] += 1
                    continue

            # Valid records with a real candidate and persistent hard errors already exhausted
            # the previous retry policy; keep them in the prebuilt Sol web queue. Missing/null
            # candidates are not proven model failures and return to Luna.
            if split == "valid" and status in {"exhausted", "tail"} and isinstance(candidate, dict):
                sol_existing.append({
                    "record_uid": uid, "source_split": split, "original_seq": original_seq,
                    "hard_errors": hard_errors, "warnings": warnings,
                })
                split_counts["sol_existing"] += 1
                continue

            prompt, response = source_fields(source)
            queue.append({
                "record_uid": uid,
                "source_split": split,
                "original_seq": original_seq,
                "prompt_en": prompt,
                "response_en": response,
                "length_bucket": source.get("length_bucket"),
                "prompt_label": source.get("prompt_label"),
                "response_label": source.get("response_label"),
                "violated_categories": source.get("violated_categories"),
            })
            split_counts["luna_queue"] += 1
        counts[split] = split_counts

    append_jsonl(OUT / "queue.jsonl", queue)
    append_jsonl(OUT / "recovered_without_model.jsonl", recovered)
    append_jsonl(OUT / "sol_existing.jsonl", sol_existing)
    manifest = {
        "queue_records": len(queue),
        "recovered_without_model": len(recovered),
        "sol_existing": len(sol_existing),
        "splits": counts,
        "retry_policy_for_new_run": {"reasoning_effort": "low", "workers": 10, "max_attempts_per_uid": 2},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
