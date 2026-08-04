from __future__ import annotations

import argparse
import copy
import re
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from translator.batching import source_chars
from translator.checkpoint import load_checkpoint, save_completed
from translator.full_run import paths
from translator.jsonl_io import read_jsonl, write_jsonl
from translator.pipeline import text_hash
from translator.models import TranslationInputItem, TranslationRequest, TranslationResponse
from translator.validators import TranslationValidationError, validate_hard_quality, validate_response


MILD_MARKERS = ("strong profanity was softened",)


def relevant_errors(uid: str, errors: list[str]) -> list[str]:
    relevant = []
    for error in errors:
        # Per-record validators always include the UID. A structural error with
        # no UID (for example batch_id mismatch) applies to the entire response.
        if uid in error or "en-train-" not in error and "en-valid-" not in error and "en-test-" not in error:
            relevant.append(error)
    return relevant


def invariant_errors(source: dict, item: dict) -> list[str]:
    errors = []
    if item.get("record_uid") != source.get("record_uid"):
        errors.append("record_uid mismatch")
    prompt_vi = item.get("prompt_vi")
    response_vi = item.get("response_vi")
    if source.get("prompt") == "" and prompt_vi != "":
        errors.append("empty prompt not preserved")
    if source.get("prompt") != "" and (not isinstance(prompt_vi, str) or not prompt_vi.strip()):
        errors.append("missing prompt_vi")
    if source.get("response") is None and response_vi is not None:
        errors.append("null response not preserved")
    if source.get("response") == "" and response_vi != "":
        errors.append("empty response not preserved")
    if source.get("response") not in (None, "") and (not isinstance(response_vi, str) or not response_vi.strip()):
        errors.append("missing response_vi")
    return errors


def repair_safe_format(source: dict, uid: str, item: dict, errors: list[str]) -> tuple[dict, list[str], list[str]]:
    fixed = copy.deepcopy(item)
    remaining, repairs = [], []
    for error in errors:
        if "batch_id mismatch" in error:
            repairs.append("batch_id metadata reset to requested batch")
        elif f"seq mismatch for {uid}" in error:
            repairs.append("seq metadata ignored; record_uid is authoritative")
        elif f"null response not preserved for {uid}" in error and source.get("response") is None:
            fixed["response_vi"] = None
            repairs.append("response_vi restored to null")
        elif f"empty response not preserved for {uid}" in error and source.get("response") == "":
            fixed["response_vi"] = ""
            repairs.append("response_vi restored to empty string")
        elif f"empty prompt not preserved for {uid}" in error and source.get("prompt") == "":
            fixed["prompt_vi"] = ""
            repairs.append("prompt_vi restored to empty string")
        else:
            remaining.append(error)
    return fixed, remaining, repairs


def repair_schema_keys(source: dict, item: dict) -> tuple[dict, list[str]]:
    """Restore quoted JSON keys only when both sides have equal key counts/order."""
    fixed = copy.deepcopy(item)
    repairs: list[str] = []
    key_pattern = re.compile(r'"([^"\\]+)"\s*:')
    for source_field, target_field in (("prompt", "prompt_vi"), ("response", "response_vi")):
        source_text = source.get(source_field)
        target_text = fixed.get(target_field)
        if not isinstance(source_text, str) or not isinstance(target_text, str):
            continue
        source_keys = key_pattern.findall(source_text)
        target_keys = key_pattern.findall(target_text)
        if not source_keys or len(source_keys) != len(target_keys) or source_keys == target_keys:
            continue
        index = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal index
            value = f'"{source_keys[index]}":'
            index += 1
            return value

        fixed[target_field] = key_pattern.sub(replace, target_text)
        repairs.append(f"{target_field} JSON/schema keys restored from source order")
    return fixed, repairs


def hard_errors(source: dict, item: dict) -> list[str]:
    """Run the actual per-record gates after deterministic repairs."""
    try:
        request = TranslationRequest(batch_id="revision", items=[TranslationInputItem(
            seq=1, record_uid=source["record_uid"], prompt=source["prompt"], response=source.get("response"),
        )])
        response_item = dict(item)
        response_item["seq"] = 1
        response_item["record_uid"] = source["record_uid"]
        response = TranslationResponse.model_validate({"batch_id": "revision", "items": [response_item]})
        validate_response(request, response)
        validate_hard_quality(request, response)
        return []
    except (TranslationValidationError, ValueError) as exc:
        return list(exc.errors) if isinstance(exc, TranslationValidationError) else [str(exc)]


def candidate_rank(kind: str, errors: list[str], item: dict) -> tuple:
    order = {"provisional_clean": 0, "provisional_format_repaired": 1, "provisional_mild_profanity": 2}
    return (order[kind], len(errors), -len(item.get("prompt_vi", "")) - len(item.get("response_vi") or ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--groups", default="1,2,3,4,5")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    groups = [int(x) for x in args.groups.split(",")]
    report_path = root / "reports" / "full_run" / "candidate_salvage_decisions.jsonl"
    summary = Counter()
    decisions = []

    for group in groups:
        for split in ("train", "valid", "test"):
            p = paths(root, split, group - 1)
            if not p["failed"].exists():
                continue
            candidate_path = p["failed"].with_name(p["failed"].name.replace("_failed.jsonl", "_candidates.jsonl"))
            if not candidate_path.exists():
                continue
            archives = {row["candidate_archive_id"]: row for _, row, _ in read_jsonl(candidate_path)}
            review = {}
            for _, row, _ in read_jsonl(p["failed"]):
                if row.get("translation_status") in {"needs_revision", "quarantined_validation"}:
                    review[row.get("record_uid")] = row
            completed = load_checkpoint(p["checkpoint"])
            sources = {row["record_uid"]: row for i, (_, row, _) in enumerate(read_jsonl(p["input"])) if i % 5 == group - 1}

            for uid, failed in review.items():
                if uid in completed or uid not in sources:
                    continue
                archive = archives.get(failed.get("candidate_archive_id"))
                if not archive:
                    summary["no_archive"] += 1
                    continue
                choices = []
                for candidate in archive.get("candidates", []):
                    item = next((x for x in candidate.get("response", {}).get("items", []) if x.get("record_uid") == uid), None)
                    if not item:
                        continue
                    errors = relevant_errors(uid, list(candidate.get("validation_errors") or []))
                    item, errors, repairs = repair_safe_format(sources[uid], uid, item, errors)
                    item, schema_repairs = repair_schema_keys(sources[uid], item)
                    repairs.extend(schema_repairs)
                    errors.extend(invariant_errors(sources[uid], item))
                    # A full per-record recheck decides whether a schema repair
                    # really resolved the candidate rather than trusting strings.
                    actual_errors = hard_errors(sources[uid], item)
                    if not actual_errors:
                        errors = []
                    elif schema_repairs:
                        errors = actual_errors
                    if not errors:
                        kind = "provisional_format_repaired" if repairs else "provisional_clean"
                    elif all(any(marker in error for marker in MILD_MARKERS) for error in errors):
                        kind = "provisional_mild_profanity"
                    else:
                        continue
                    choices.append((candidate_rank(kind, errors, item), kind, errors, repairs, candidate, item))
                if not choices:
                    summary["still_needs_revision"] += 1
                    continue
                _, kind, errors, repairs, candidate, item = min(choices, key=lambda value: value[0])
                source = sources[uid]
                decision = {
                    "record_uid": uid, "source_split": split, "group": group,
                    "decision": kind, "candidate_archive_id": archive["candidate_archive_id"],
                    "candidate_number": candidate["candidate_number"], "remaining_errors": errors,
                    "repairs": repairs,
                    "decided_at": datetime.now(timezone.utc).isoformat(),
                }
                decisions.append(decision)
                summary[kind] += 1
                if args.apply:
                    translated = {
                        "record_uid": uid, "prompt_vi": item["prompt_vi"], "response_vi": item.get("response_vi"),
                        "translation_provider": candidate.get("provider"), "translation_model": candidate.get("model"),
                        "translation_prompt_version": "nemotron-en-vi-v10",
                        "translation_batch_id": candidate.get("batch_id_requested"),
                        "translation_attempt": candidate.get("candidate_number"),
                        "translation_status": kind,
                        "translation_api_key_slot": candidate.get("api_key_slot"),
                        "translated_at": archive.get("archived_at"),
                        "source_text_sha256": text_hash(source.get("prompt"), source.get("response")),
                        "translation_text_sha256": text_hash(item.get("prompt_vi"), item.get("response_vi")),
                        "input_source_chars": source_chars(source),
                        "output_translation_chars": len(item.get("prompt_vi") or "") + len(item.get("response_vi") or ""),
                        "warnings": list(item.get("warnings") or []),
                        "usage_metadata": candidate.get("usage_metadata"),
                        "validation_warnings": ["provisional_candidate_salvage", *repairs, *errors],
                        "candidate_archive_id": archive["candidate_archive_id"],
                        "candidate_number": candidate["candidate_number"],
                    }
                    save_completed(p["checkpoint"], translated)

    write_jsonl(report_path, decisions)
    print(json.dumps({"apply": args.apply, "decisions": len(decisions), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
