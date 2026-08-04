#!/usr/bin/env python3
"""Foreground revision runner for the unresolved EN→VI handoff batches.

It deliberately does *not* overwrite any checkpoint.  Validated Gemini
revisions are appended to a standalone results file; a second, local
``apply_terra_revisions.py`` pass is responsible for merging them.  This makes
an interrupted run resumable and keeps a complete audit trail of model output.

The process is foreground by design: it can be monitored through its JSONL
events and terminated safely without losing already completed batches.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator.jsonl_io import append_jsonl
from translator.models import TranslationInputItem, TranslationRequest
from translator.providers import GeminiProvider
from translator.validators import TranslationValidationError, validate_hard_quality, validate_response


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validated(provider: GeminiProvider, request: TranslationRequest, source_errors: list[str]):
    """Give a structurally usable response two focused chances before splitting."""
    errors = list(source_errors)
    candidates = []
    for attempt in range(1, 3):
        result = provider.translate(request, repair_errors=errors if attempt > 1 else None)
        try:
            validate_response(request, result.response)
            validate_hard_quality(request, result.response)
            return result, candidates
        except TranslationValidationError as exc:
            errors = exc.errors
            candidates.append({
                "attempt": attempt,
                "key_slot": result.key_slot,
                "usage_metadata": result.usage_metadata,
                "validation_errors": errors,
                "response": result.response.model_dump(mode="json"),
            })
    raise TranslationValidationError(errors)


def make_request(rows: list[dict], batch_id: str) -> TranslationRequest:
    return TranslationRequest(
        batch_id=batch_id,
        items=[TranslationInputItem(
            seq=index,
            record_uid=row["record_uid"],
            prompt=row["source"]["prompt_en"],
            response=row["source"].get("response_en"),
        ) for index, row in enumerate(rows, 1)],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--key-slots", default="11,12,13,14,15")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--request-interval-seconds", type=float, default=60.0)
    args = parser.parse_args()

    root = args.root.resolve()
    result_dir = root / "data" / "revision_handoff" / "gemini_revision_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    event_path = root / "data" / "run" / "gemini_revision_events.jsonl"
    provider = GeminiProvider(
        args.model,
        root / "API.txt",
        key_slots=[int(x) for x in args.key_slots.split(",") if x.strip()],
        min_request_interval_seconds=args.request_interval_seconds,
        event_log_path=event_path,
    )
    manifest_path = root / "data" / "revision_handoff" / "terra_batches" / "manifest.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    processed = 0
    accepted = 0

    for position, entry in enumerate(manifest, 1):
        if position < args.start_batch:
            continue
        if args.max_batches is not None and processed >= args.max_batches:
            break
        batch_file = root / "data" / "revision_handoff" / "terra_batches" / entry["file"]
        output_file = result_dir / entry["file"]
        existing = {
            json.loads(line)["record_uid"]
            for line in output_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        } if output_file.exists() else set()
        rows = [json.loads(line) for line in batch_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [row for row in rows if row["record_uid"] not in existing]
        if not rows:
            continue
        processed += 1
        batch_id = f"terra-revision-{position:03d}"
        source_errors = [error for row in rows for error in row.get("validation_errors", [])]
        def revise_rows(part: list[dict], suffix: str = "") -> None:
            """Split malformed/oversized original batches before giving up."""
            nonlocal accepted
            part_id = batch_id + suffix
            part_errors = [error for row in part for error in row.get("validation_errors", [])]
            try:
                result, _ = validated(provider, make_request(part, part_id), part_errors)
                by_uid = {item.record_uid: item for item in result.response.items}
                for row in part:
                    item = by_uid[row["record_uid"]]
                    append_jsonl(output_file, {
                        "record_uid": item.record_uid,
                        "prompt_vi": item.prompt_vi,
                        "response_vi": item.response_vi,
                        "review_status": "revised",
                        "fix_notes": [
                            "gemini revision draft passed structural and hard-quality validation",
                            f"source_batch={entry['file']}",
                            f"key_slot={result.key_slot}",
                        ],
                    }, fsync=True)
                    accepted += 1
                append_jsonl(event_path, {"timestamp": now(), "event": "batch_validated", "batch": entry["file"], "part": suffix or "whole", "records": len(part), "key_slot": result.key_slot}, fsync=True)
                return
            except Exception as exc:
                if len(part) > 1:
                    append_jsonl(event_path, {"timestamp": now(), "event": "batch_split", "batch": entry["file"], "part": suffix or "whole", "records": len(part), "error": str(exc)[:1200]}, fsync=True)
                    midpoint = len(part) // 2
                    revise_rows(part[:midpoint], suffix + "-L")
                    revise_rows(part[midpoint:], suffix + "-R")
                    return
                append_jsonl(event_path, {
                    "timestamp": now(), "event": "record_needs_manual_review", "batch": entry["file"], "part": suffix or "whole",
                    "records": [part[0]["record_uid"]], "error": str(exc)[:4000],
                }, fsync=True)

        revise_rows(rows)
        print(json.dumps({"batch": entry["file"], "status": "processed", "records": len(rows), "processed_batches": processed}, ensure_ascii=False), flush=True)

    print(json.dumps({"status": "complete", "processed_batches": processed, "accepted_records": accepted, "provider_calls": provider.calls}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
