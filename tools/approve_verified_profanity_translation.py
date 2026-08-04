from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator.batching import source_chars
from translator.checkpoint import load_checkpoint
from translator.dashboard import DashboardData
from translator.full_run import paths
from translator.jsonl_io import append_jsonl, read_jsonl, write_jsonl
from translator.models import TranslationInputItem, TranslationRequest, TranslationResponse
from translator.pipeline import text_hash
from translator.validators import validate_hard_quality, validate_response


UID = "en-valid-00001697-8e2511f76608"


def run(root: Path, apply: bool) -> dict:
    data = DashboardData(root)
    group, split, source = data.source_index()[UID]
    checkpoint = paths(root, split, group - 1)["checkpoint"]
    existing = load_checkpoint(checkpoint)[UID]
    if existing.get("translation_status") == "codex_quality_approved":
        return {"mode": "already-applied", "record_uid": UID}
    if existing.get("translation_status") != "provisional_mild_profanity":
        raise ValueError(f"Unexpected prior status: {existing.get('translation_status')}")

    request = TranslationRequest(
        batch_id="verified-profanity-review",
        items=[TranslationInputItem(seq=1, record_uid=UID, prompt=source.get("prompt") or "", response=source.get("response"))],
    )
    response = TranslationResponse.model_validate(
        {
            "batch_id": "verified-profanity-review",
            "items": [{
                "seq": 1,
                "record_uid": UID,
                "prompt_vi": existing.get("prompt_vi"),
                "response_vi": existing.get("response_vi"),
                "warnings": [],
            }],
        }
    )
    validate_response(request, response)
    validate_hard_quality(request, response)
    if not apply:
        return {"mode": "dry-run", "record_uid": UID, "hard_quality": "pass"}

    timestamp = datetime.now(timezone.utc).isoformat()
    updated = {
        **existing,
        "translation_provider": "codex_human_review",
        "translation_model": "codex-final-quality-review",
        "translation_prompt_version": "final-quality-codex-human-v1",
        "translation_batch_id": "codex-final-profanity-verification-001",
        "translation_attempt": int(existing.get("translation_attempt") or 0) + 1,
        "translation_status": "codex_quality_approved",
        "translation_api_key_slot": None,
        "translated_at": timestamp,
        "source_text_sha256": text_hash(source.get("prompt"), source.get("response")),
        "translation_text_sha256": text_hash(existing.get("prompt_vi"), existing.get("response_vi")),
        "input_source_chars": source_chars(source),
        "output_translation_chars": len(existing.get("prompt_vi") or "") + len(existing.get("response_vi") or ""),
        "validation_warnings": ["codex_final_quality_review", "profanity_register_human_approved"],
        "revision_status": "final_quality_human_approved",
        "human_review": {
            "reviewer": "codex-final-quality",
            "reviewed_at": timestamp,
            "decision": "human_approved_false_positive",
            "validator_override": False,
            "reason": "The Vietnamese line retains comparable aggression with 'hỏng mẹ nó' and 'đống nhảm nhí'; the updated hard-quality gate passes it.",
            "prior_translation": {
                "status": existing.get("translation_status"),
                "provider": existing.get("translation_provider"),
                "model": existing.get("translation_model"),
                "translation_text_sha256": existing.get("translation_text_sha256"),
            },
        },
    }
    rows = [row for _, row, _ in read_jsonl(checkpoint)]
    if sum(row.get("record_uid") == UID for row in rows) != 1:
        raise ValueError("Expected exactly one checkpoint row")
    write_jsonl(checkpoint, [updated if row.get("record_uid") == UID else row for row in rows])
    audit_path = root / "data" / "revision_handoff" / "human_review_decisions" / "final_codex_quality_review.jsonl"
    append_jsonl(audit_path, {
        "record_uid": UID,
        "split": split,
        "group": group,
        "reviewed_at": timestamp,
        "decision": "human_approved_false_positive",
        "validator_override": False,
        "reason": updated["human_review"]["reason"],
        "prior_translation_text_sha256": existing.get("translation_text_sha256"),
        "translation_text_sha256": updated["translation_text_sha256"],
    })
    return {"mode": "apply", "record_uid": UID, "status": updated["translation_status"], "audit_path": str(audit_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.root).resolve(), args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
