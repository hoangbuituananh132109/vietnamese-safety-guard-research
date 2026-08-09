from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator.batching import source_chars
from translator.checkpoint import load_checkpoint, save_completed
from translator.dashboard import DashboardData, REVIEW_DONE_STATUSES
from translator.full_run import paths
from translator.jsonl_io import append_jsonl
from translator.models import TranslationInputItem, TranslationRequest, TranslationResponse
from translator.pipeline import text_hash
from translator.validators import validate_response


QUOTED_PROSE_UIDS = {
    "en-train-00003546-b52493fbc49c",
    "en-valid-00000578-0d7832ca2476",
    "en-train-00021164-9ddac8fdd983",
}

STRUCTURED_CHEMISTRY_UIDS = {
    "en-train-00037809-acdef625a0ba",
}

ORDINARY_COMPROMISE_UIDS = {
    "en-valid-00002401-b91cda9b6c81",
    "en-train-00036042-b5f832c877df",
    "en-train-00035139-fa9c9823a34b",
    "en-train-00036489-37e86a484189",
    "en-train-00038724-8f9d20f93fba",
    "en-train-00035725-abd63697c758",
    "en-train-00036970-1fa4b3caaddd",
    "en-train-00039885-c44149ecd87e",
}

EXPECTED_UIDS = QUOTED_PROSE_UIDS | STRUCTURED_CHEMISTRY_UIDS | ORDINARY_COMPROMISE_UIDS


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reason_for(uid: str) -> str:
    if uid in QUOTED_PROSE_UIDS:
        return (
            "False positive: the JSON-key regex matched quoted natural-language labels/examples. "
            "They are prose, not schema keys, so translating them into Vietnamese is correct."
        )
    if uid in STRUCTURED_CHEMISTRY_UIDS:
        return (
            "False positive: the copied digit-heavy line is structured JSON data containing schema keys, "
            "chemical names, and quantities that must remain exact; the surrounding prose and values are translated."
        )
    if uid in ORDINARY_COMPROMISE_UIDS:
        return (
            "False positive: 'compromise' means negotiation/concession in this sentence, so 'thỏa hiệp' is correct. "
            "The cybersecurity heuristic was triggered only because another broad keyword such as communication, "
            "operation, security, network, or logistics appeared elsewhere in the same long field."
        )
    raise ValueError(f"No reviewed rationale for {uid}")


def validate_translation(
    source: dict[str, Any], uid: str, prompt_vi: str, response_vi: str | None
) -> None:
    request = TranslationRequest(
        batch_id="codex-luna-human-review",
        items=[
            TranslationInputItem(
                seq=1,
                record_uid=uid,
                prompt=source.get("prompt") or "",
                response=source.get("response"),
            )
        ],
    )
    response = TranslationResponse.model_validate(
        {
            "batch_id": "codex-luna-human-review",
            "items": [
                {
                    "seq": 1,
                    "record_uid": uid,
                    "prompt_vi": prompt_vi,
                    "response_vi": response_vi,
                    "warnings": [],
                }
            ],
        }
    )
    # The hard-quality errors for this set have already been manually reviewed
    # and documented as false positives. Structural invariants still must pass.
    validate_response(request, response)


def load_failure_records(data: DashboardData) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((data.review_failure_root / "luna").glob("batch_*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        for item in snapshot.get("failures", []):
            uid = item.get("record_uid")
            if isinstance(uid, str):
                records[uid] = {**item, "snapshot_path": path}
    return records


def prune_resolved_failures(data: DashboardData) -> None:
    completed = data.completed_review_uids()
    for path in sorted((data.review_failure_root / "luna").glob("batch_*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        remaining = [
            item
            for item in snapshot.get("failures", [])
            if not isinstance(item.get("record_uid"), str)
            or item["record_uid"] not in completed
        ]
        if remaining:
            snapshot["failures"] = remaining
            snapshot["saved_at"] = now()
            data._atomic_write_json(path, snapshot)
        else:
            path.unlink()


def run(root: Path, apply: bool) -> dict[str, Any]:
    data = DashboardData(root)
    failures = load_failure_records(data)
    current_uids = set(failures)
    unexpected = current_uids - EXPECTED_UIDS
    if unexpected:
        raise ValueError(f"Luna queue changed; refusing unreviewed UIDs: {sorted(unexpected)}")
    missing_expected = EXPECTED_UIDS - current_uids - set(data.completed_review_uids())
    if missing_expected:
        raise ValueError(f"Expected Luna UIDs are neither queued nor completed: {sorted(missing_expected)}")

    source_index = data.source_index()
    completed = data.completed_review_uids()
    decisions: list[dict[str, Any]] = []
    for uid in sorted(current_uids):
        if uid in completed:
            continue
        failure = failures[uid]
        candidate = failure.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError(f"Missing candidate for {uid}")
        prompt_vi = candidate.get("prompt_vi")
        response_vi = candidate.get("response_vi")
        if not isinstance(prompt_vi, str) or not prompt_vi:
            raise ValueError(f"Invalid prompt_vi for {uid}")
        group, split, source = source_index[uid]
        validate_translation(source, uid, prompt_vi, response_vi)
        decisions.append(
            {
                "record_uid": uid,
                "group": group,
                "split": split,
                "batch": failure.get("batch"),
                "prompt_vi": prompt_vi,
                "response_vi": response_vi,
                "decision": "human_approved_override",
                "validator_override": True,
                "reason": reason_for(uid),
                "original_validator_errors": list(failure.get("errors") or []),
                "source_failure_snapshot": str(
                    Path(failure["snapshot_path"]).relative_to(root)
                ),
            }
        )

    if not apply:
        return {
            "mode": "dry-run",
            "queue_records": len(current_uids),
            "decisions": len(decisions),
            "overrides": len(decisions),
            "quoted_prose_false_positives": sum(
                item["record_uid"] in QUOTED_PROSE_UIDS for item in decisions
            ),
            "structured_data_false_positives": sum(
                item["record_uid"] in STRUCTURED_CHEMISTRY_UIDS for item in decisions
            ),
            "ordinary_compromise_false_positives": sum(
                item["record_uid"] in ORDINARY_COMPROMISE_UIDS for item in decisions
            ),
        }

    timestamp = now()
    audit_path = (
        root
        / "data"
        / "revision_handoff"
        / "human_review_decisions"
        / "luna_codex_review.jsonl"
    )
    applied = skipped = 0
    for decision in decisions:
        uid = decision["record_uid"]
        group = int(decision["group"])
        split = str(decision["split"])
        source = source_index[uid][2]
        checkpoint = paths(root, split, group - 1)["checkpoint"]
        existing = load_checkpoint(checkpoint).get(uid)
        if existing and existing.get("translation_status") in REVIEW_DONE_STATUSES:
            skipped += 1
            continue
        translated = {
            "record_uid": uid,
            "prompt_vi": decision["prompt_vi"],
            "response_vi": decision["response_vi"],
            "translation_provider": "codex_human_review",
            "translation_model": "codex-luna-manual-review",
            "translation_prompt_version": "revision-luna-codex-human-v1",
            "translation_batch_id": f"codex-review-luna-{int(decision['batch']):03d}",
            "translation_attempt": 1,
            "translation_status": "luna_revised",
            "translation_api_key_slot": None,
            "translated_at": timestamp,
            "source_text_sha256": text_hash(
                source.get("prompt"), source.get("response")
            ),
            "translation_text_sha256": text_hash(
                decision["prompt_vi"], decision["response_vi"]
            ),
            "input_source_chars": source_chars(source),
            "output_translation_chars": len(decision["prompt_vi"] or "")
            + len(decision["response_vi"] or ""),
            "warnings": [],
            "usage_metadata": None,
            "validation_warnings": [
                "codex_human_review",
                "human_validator_override",
                *decision["original_validator_errors"],
            ],
            "revision_status": decision["decision"],
            "human_review": {
                "reviewer": "codex-luna",
                "reviewed_at": timestamp,
                "decision": decision["decision"],
                "validator_override": True,
                "reason": decision["reason"],
                "original_validator_errors": decision["original_validator_errors"],
                "source_failure_snapshot": decision["source_failure_snapshot"],
            },
        }
        save_completed(checkpoint, translated)
        append_jsonl(
            audit_path,
            {
                "record_uid": uid,
                "split": split,
                "group": group,
                "batch": decision["batch"],
                "reviewed_at": timestamp,
                "decision": decision["decision"],
                "validator_override": True,
                "reason": decision["reason"],
                "original_validator_errors": decision[
                    "original_validator_errors"
                ],
                "translation_text_sha256": translated[
                    "translation_text_sha256"
                ],
            },
        )
        applied += 1

    prune_resolved_failures(data)
    status = data.review_status()
    queue = data.review_failure_queue("luna")
    return {
        "mode": "apply",
        "applied": applied,
        "skipped": skipped,
        "overrides": len(decisions),
        "luna_remaining": status["reviewers"]["luna"]["remaining"],
        "luna_queue": queue["total"],
        "completed_total": status["completed_total"],
        "remaining_total": status["remaining_total"],
        "audit_path": str(audit_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the manually reviewed Luna false-positive overrides."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = run(Path(args.root).resolve(), args.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
