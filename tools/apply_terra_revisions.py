#!/usr/bin/env python3
"""Validate and merge human/Terra revisions into the original checkpoints.

The input is a JSONL file using the ``required_output`` schema emitted in
``data/revision_handoff/terra_batches``.  Each row is validated against its
original English record, is routed to that record's group checkpoint, and is
only written when both the structural and hard-quality gates pass.

The script is intentionally idempotent: a UID already written with
``translation_status=terra_revised`` is reported as skipped rather than being
silently replaced.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# The tool is intentionally runnable as ``python tools/apply_terra_revisions.py``
# from a normal PowerShell session, without requiring the caller to set
# PYTHONPATH first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator.batching import source_chars
from translator.checkpoint import load_checkpoint, save_completed
from translator.full_run import paths
from translator.jsonl_io import read_jsonl, write_jsonl
from translator.models import TranslationInputItem, TranslationRequest, TranslationResponse
from translator.pipeline import text_hash
from translator.validators import TranslationValidationError, validate_hard_quality, validate_response


def source_index(root: Path) -> dict[str, tuple[int, str, dict]]:
    indexed: dict[str, tuple[int, str, dict]] = {}
    for split in ("train", "valid", "test"):
        input_path = root / "data" / "prepared" / f"nemotron_en_{split}_full_v1.jsonl"
        for seq, row, _ in read_jsonl(input_path):
            # Full-run grouping is round-robin over the original zero-based row index.
            indexed[row["record_uid"]] = ((seq - 1) % 5 + 1, split, row)
    return indexed


def validate(uid: str, source: dict, revision: dict) -> list[str]:
    try:
        request = TranslationRequest(batch_id="terra-revision", items=[TranslationInputItem(
            seq=1, record_uid=uid, prompt=source["prompt"], response=source.get("response"),
        )])
        response = TranslationResponse.model_validate({"batch_id": "terra-revision", "items": [{
            "seq": 1, "record_uid": uid,
            "prompt_vi": revision.get("prompt_vi"),
            "response_vi": revision.get("response_vi"),
            "warnings": [],
        }]})
        validate_response(request, response)
        validate_hard_quality(request, response)
        return []
    except (TranslationValidationError, ValueError) as exc:
        return list(exc.errors) if isinstance(exc, TranslationValidationError) else [str(exc)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--provider", default="manual_terra_revision")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--status", default="terra_revised")
    args = parser.parse_args()
    root = args.root.resolve()
    index = source_index(root)
    report_path = args.report or root / "reports" / "full_run" / "terra_revision_apply_report.jsonl"
    results: list[dict] = []
    counts: Counter[str] = Counter()

    for _, revision, _ in read_jsonl(args.input):
        uid = revision.get("record_uid")
        if uid not in index:
            results.append({"record_uid": uid, "status": "rejected_unknown_uid"})
            counts["rejected_unknown_uid"] += 1
            continue
        group, split, source = index[uid]
        errors = validate(uid, source, revision)
        if errors:
            results.append({"record_uid": uid, "status": "rejected_validation", "errors": errors})
            counts["rejected_validation"] += 1
            continue
        checkpoint = paths(root, split, group - 1)["checkpoint"]
        completed = load_checkpoint(checkpoint)
        if completed.get(uid, {}).get("translation_status") in {"terra_revised", "luna_revised", "gemini_revised"}:
            results.append({"record_uid": uid, "status": "skipped_already_revised"})
            counts["skipped_already_revised"] += 1
            continue
        if args.apply:
            now = datetime.now(timezone.utc).isoformat()
            translated = {
                "record_uid": uid,
                "prompt_vi": revision["prompt_vi"],
                "response_vi": revision.get("response_vi"),
                "translation_provider": args.provider,
                "translation_model": args.model,
                "translation_prompt_version": f"revision-{args.status}-v1",
                "translation_batch_id": revision.get("batch_id", "manual"),
                "translation_attempt": 1,
                "translation_status": args.status,
                "translation_api_key_slot": None,
                "translated_at": now,
                "source_text_sha256": text_hash(source.get("prompt"), source.get("response")),
                "translation_text_sha256": text_hash(revision.get("prompt_vi"), revision.get("response_vi")),
                "input_source_chars": source_chars(source),
                "output_translation_chars": len(revision.get("prompt_vi") or "") + len(revision.get("response_vi") or ""),
                "warnings": [],
                "usage_metadata": None,
                "validation_warnings": ["terra_manual_revision", *revision.get("fix_notes", [])],
                "revision_status": revision.get("review_status", "revised"),
            }
            save_completed(checkpoint, translated)
        results.append({"record_uid": uid, "status": "accepted" if args.apply else "would_accept", "group": group, "split": split})
        counts["accepted" if args.apply else "would_accept"] += 1

    write_jsonl(report_path, results)
    print(json.dumps({"apply": args.apply, **counts, "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
