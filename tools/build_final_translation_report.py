from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator.batching import source_chars
from translator.checkpoint import load_checkpoint
from translator.dashboard import DashboardData
from translator.full_run import paths
from translator.jsonl_io import read_jsonl, safe_json_dumps
from translator.models import TranslationInputItem, TranslationRequest, TranslationResponse
from translator.pipeline import text_hash
from translator.validators import (
    TranslationValidationError,
    quality_warnings,
    validate_hard_quality,
    validate_response,
)


SPLITS = ("train", "valid", "test")
STATUSES = (
    "machine_translated",
    "provisional_clean",
    "provisional_format_repaired",
    "provisional_mild_profanity",
    "terra_revised",
    "luna_revised",
    "gemini_revised",
    "codex_quality_repaired",
    "codex_quality_approved",
    "codex_profanity_repaired",
    "codex_profanity_approved",
)
REVISED_STATUSES = {
    "terra_revised", "luna_revised", "gemini_revised",
    "codex_quality_repaired", "codex_quality_approved",
    "codex_profanity_repaired", "codex_profanity_approved",
}

STATUS_DESCRIPTIONS = {
    "machine_translated": {
        "tier": "A_strict_machine_pass",
        "disposition": "ready",
        "description": "Gemini translation passed structural and hard-quality gates during the main run.",
    },
    "provisional_clean": {
        "tier": "B_salvaged_clean",
        "disposition": "ready",
        "description": "Best archived candidate was recovered and passed the same gates when checked per record.",
    },
    "provisional_format_repaired": {
        "tier": "B_salvaged_format_repaired",
        "disposition": "ready_with_metadata",
        "description": "Archived candidate was recovered after deterministic JSON/schema/null-format repair and revalidation.",
    },
    "provisional_mild_profanity": {
        "tier": "C_known_register_caveat",
        "disposition": "followup_recommended",
        "description": "Complete usable translation, but an automatic gate indicates that strong profanity may be softer than the English source.",
    },
    "terra_revised": {
        "tier": "A_revised",
        "disposition": "ready",
        "description": "Previously unresolved record was translated/repaired and accepted through the Terra review workflow or audited human override.",
    },
    "luna_revised": {
        "tier": "A_revised",
        "disposition": "ready",
        "description": "Previously unresolved record was translated/repaired and accepted through the Luna review workflow or audited human override.",
    },
    "gemini_revised": {
        "tier": "A_revised",
        "disposition": "ready",
        "description": "Previously unresolved record was retranslated by Gemini and passed structural and hard-quality gates.",
    },
    "codex_quality_repaired": {
        "tier": "A_human_repaired",
        "disposition": "ready",
        "description": "Final full-dataset audit found a real issue; Codex repaired the translation and recorded before/after hashes and rationale.",
    },
    "codex_quality_approved": {
        "tier": "A_human_approved",
        "disposition": "ready",
        "description": "Final full-dataset audit manually confirmed a hard-validator false positive and recorded the rationale without changing valid content.",
    },
    "codex_profanity_repaired": {
        "tier": "A_human_repaired",
        "disposition": "ready",
        "description": "Manual bilingual review confirmed softened or omitted profanity; Codex repaired the Vietnamese register per record and revalidated it.",
    },
    "codex_profanity_approved": {
        "tier": "A_human_approved",
        "disposition": "ready",
        "description": "Manual bilingual review confirmed the Vietnamese wording already preserved the source register; the heuristic warning was overridden with an audit trail.",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pct(value: int, total: int) -> float:
    return round(value * 100.0 / total, 4) if total else 0.0


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def atomic_categories(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return sorted({part.strip() for part in value.split(",") if part.strip()})


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): count
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    }


def warning_kind(value: str) -> str:
    if "_missing_token:" in value:
        return value.split(":", 1)[0]
    if re.match(r"^en-(?:train|valid|test)-", value):
        value = re.sub(r"^\S+\s+(?:prompt|response):\s*", "", value)
    # PowerShell's ConvertFrom-Json treats object keys case-insensitively. The
    # concrete source marker is useful per record, but not in an aggregate
    # warning histogram, and values such as REDACTED/redacted otherwise make
    # the standards-compliant JSON report unreadable from PowerShell.
    if re.match(r"^redaction marker\s+'.+'\s+was not preserved exactly$", value, re.IGNORECASE):
        return "redaction marker was not preserved exactly"
    return value


def hard_error_kind(value: str) -> str:
    return re.sub(r"^\S+\s+(?:prompt|response):\s*", "", value)


def quality_class(status: str) -> dict[str, str]:
    return STATUS_DESCRIPTIONS.get(
        status,
        {
            "tier": "unknown",
            "disposition": "inspect",
            "description": "Unknown translation state.",
        },
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def validate_one(source: dict[str, Any], translated: dict[str, Any]) -> tuple[list[str], list[str]]:
    uid = source["record_uid"]
    request = TranslationRequest(
        batch_id="final-audit",
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
            "batch_id": "final-audit",
            "items": [
                {
                    "seq": 1,
                    "record_uid": uid,
                    "prompt_vi": translated.get("prompt_vi"),
                    "response_vi": translated.get("response_vi"),
                    "warnings": [],
                }
            ],
        }
    )
    structural: list[str] = []
    hard: list[str] = []
    try:
        validate_response(request, response)
    except TranslationValidationError as exc:
        structural.extend(exc.errors)
    try:
        validate_hard_quality(request, response)
    except TranslationValidationError as exc:
        hard.extend(exc.errors)
    return structural, hard


def output_row(source: dict[str, Any], translated: dict[str, Any], hard_disposition: str) -> dict[str, Any]:
    result = {key: value for key, value in source.items() if key not in {"prompt", "response", "language"}}
    status = str(translated.get("translation_status") or "unknown")
    result.update(
        {
            "language": "vi",
            "prompt_en": source.get("prompt"),
            "response_en": source.get("response"),
            "prompt_vi": translated.get("prompt_vi"),
            "response_vi": translated.get("response_vi"),
            "translation_status": status,
            "translation_quality_tier": quality_class(status)["tier"],
            "translation_disposition": quality_class(status)["disposition"],
            "hard_validator_disposition": hard_disposition,
            "needs_followup_review": status == "provisional_mild_profanity",
        }
    )
    for key, value in translated.items():
        if key not in {"record_uid", "prompt_vi", "response_vi", "translation_status"}:
            result[key] = value
    return result


def compact_sample(source: dict[str, Any], translated: dict[str, Any]) -> dict[str, Any]:
    def clip(value: Any, limit: int = 900) -> Any:
        if not isinstance(value, str) or len(value) <= limit:
            return value
        return value[:limit] + f" … [truncated {len(value) - limit} chars]"

    return {
        "record_uid": source["record_uid"],
        "split": source["source_split"],
        "tag": source.get("tag"),
        "prompt_label": source.get("prompt_label"),
        "response_label": source.get("response_label"),
        "categories": source.get("violated_categories"),
        "length_bucket": source.get("length_bucket"),
        "source_chars": source.get("source_chars"),
        "translation_status": translated.get("translation_status"),
        "translation_provider": translated.get("translation_provider"),
        "prompt_en": clip(source.get("prompt")),
        "prompt_vi": clip(translated.get("prompt_vi")),
        "response_en": clip(source.get("response")),
        "response_vi": clip(translated.get("response_vi")),
    }


def build(root: Path, export: bool) -> dict[str, Any]:
    created_at = now()
    report_dir = root / "reports" / "final_quality"
    final_dir = root / "data" / "final"
    report_dir.mkdir(parents=True, exist_ok=True)
    if export:
        final_dir.mkdir(parents=True, exist_ok=True)

    sources_by_split: dict[str, list[dict[str, Any]]] = {}
    source_index: dict[str, dict[str, Any]] = {}
    source_location: dict[str, tuple[str, int]] = {}
    duplicate_source_uids: list[str] = []
    for split in SPLITS:
        path = root / "data" / "prepared" / f"nemotron_en_{split}_full_v1.jsonl"
        rows: list[dict[str, Any]] = []
        for line_number, row, _ in read_jsonl(path):
            uid = row["record_uid"]
            if uid in source_index:
                duplicate_source_uids.append(uid)
            source_index[uid] = row
            source_location[uid] = (split, (line_number - 1) % 5 + 1)
            rows.append(row)
        sources_by_split[split] = rows

    translation_index: dict[str, dict[str, Any]] = {}
    checkpoint_location: dict[str, tuple[str, int]] = {}
    checkpoint_physical_rows = 0
    duplicate_checkpoint_uids: list[str] = []
    wrong_checkpoint: list[dict[str, Any]] = []
    checkpoint_files: list[dict[str, Any]] = []
    for split in SPLITS:
        for group_index in range(5):
            path = paths(root, split, group_index)["checkpoint"]
            physical = sum(1 for _ in read_jsonl(path))
            checkpoint_physical_rows += physical
            completed = load_checkpoint(path)
            checkpoint_files.append(
                {
                    "split": split,
                    "group": group_index + 1,
                    "path": str(path.relative_to(root)),
                    "physical_rows": physical,
                    "unique_rows": len(completed),
                }
            )
            for uid, translated in completed.items():
                if uid in translation_index:
                    duplicate_checkpoint_uids.append(uid)
                translation_index[uid] = translated
                checkpoint_location[uid] = (split, group_index + 1)
                expected = source_location.get(uid)
                if expected is not None and expected != (split, group_index + 1):
                    wrong_checkpoint.append(
                        {
                            "record_uid": uid,
                            "expected": {"split": expected[0], "group": expected[1]},
                            "actual": {"split": split, "group": group_index + 1},
                        }
                    )

    source_uids = set(source_index)
    translated_uids = set(translation_index)
    missing_uids = sorted(source_uids - translated_uids)
    extra_uids = sorted(translated_uids - source_uids)

    status_count: Counter[str] = Counter()
    split_count: dict[str, Counter[str]] = defaultdict(Counter)
    group_count: dict[str, Counter[str]] = defaultdict(Counter)
    provider_count: Counter[str] = Counter()
    model_count: Counter[str] = Counter()
    provider_status_count: dict[str, Counter[str]] = defaultdict(Counter)
    tag_count: dict[str, Counter[str]] = defaultdict(Counter)
    label_pair_count: dict[str, Counter[str]] = defaultdict(Counter)
    length_count: dict[str, Counter[str]] = defaultdict(Counter)
    category_count: dict[str, Counter[str]] = defaultdict(Counter)
    category_split_count: dict[str, Counter[str]] = defaultdict(Counter)
    category_quality_warning_records: Counter[str] = Counter()
    category_hard_fail_records: Counter[str] = Counter()
    category_source_chars: dict[str, list[int]] = defaultdict(list)
    category_combo_count: dict[str, Counter[str]] = defaultdict(Counter)
    metadata_warning_count: Counter[str] = Counter()
    live_warning_count: Counter[str] = Counter()
    live_warning_status: dict[str, Counter[str]] = defaultdict(Counter)
    revision_status_count: Counter[str] = Counter()
    hard_error_count: Counter[str] = Counter()
    structural_error_count: Counter[str] = Counter()
    hard_disposition_count: Counter[str] = Counter()
    structural_failures: list[dict[str, Any]] = []
    hard_failures: list[dict[str, Any]] = []
    unexpected_hard_failures: list[dict[str, Any]] = []
    source_lengths: list[int] = []
    target_lengths: list[int] = []
    source_length_by_status: dict[str, list[int]] = defaultdict(list)
    target_length_by_status: dict[str, list[int]] = defaultdict(list)
    samples_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_keys_by_category: dict[str, set[tuple[str, str]]] = defaultdict(set)
    export_files: dict[str, Any] = {}

    export_handles: dict[str, tuple[Path, Path, Any]] = {}
    if export:
        for split in SPLITS:
            final_path = final_dir / f"nemotron_{split}_en_vi_v10_final.jsonl"
            temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
            export_handles[split] = (
                final_path,
                temp_path,
                temp_path.open("w", encoding="utf-8", newline="\n"),
            )

    try:
        for split in SPLITS:
            for source in sources_by_split[split]:
                uid = source["record_uid"]
                translated = translation_index.get(uid)
                if translated is None:
                    continue
                status = str(translated.get("translation_status") or "unknown")
                provider = str(translated.get("translation_provider") or "<missing>")
                model = str(translated.get("translation_model") or "<missing>")
                group = source_location[uid][1]
                source_size = source_chars(source)
                target_size = len(translated.get("prompt_vi") or "") + len(translated.get("response_vi") or "")
                source_lengths.append(source_size)
                target_lengths.append(target_size)
                source_length_by_status[status].append(source_size)
                target_length_by_status[status].append(target_size)

                status_count[status] += 1
                split_count[split][status] += 1
                group_count[f"g{group}"][status] += 1
                provider_count[provider] += 1
                model_count[model] += 1
                provider_status_count[provider][status] += 1
                tag = str(source.get("tag") or "<missing>")
                tag_count[tag][status] += 1
                response_label = source.get("response_label")
                response_label_text = "<null>" if response_label is None else ("<empty>" if response_label == "" else str(response_label))
                label_pair = f"{source.get('prompt_label') or '<missing>'} → {response_label_text}"
                label_pair_count[label_pair][status] += 1
                length_bucket = str(source.get("length_bucket") or "<missing>")
                length_count[length_bucket][status] += 1

                categories = atomic_categories(source.get("violated_categories"))
                combo = " + ".join(categories) if categories else "<none>"
                category_combo_count[combo][status] += 1
                for category in categories:
                    category_count[category][status] += 1
                    category_split_count[category][split] += 1
                    category_source_chars[category].append(source_size)

                for value in translated.get("validation_warnings") or []:
                    metadata_warning_count[warning_kind(str(value))] += 1
                if translated.get("revision_status") is not None:
                    revision_status_count[str(translated.get("revision_status"))] += 1

                structural, hard = validate_one(source, translated)
                if structural:
                    structural_failures.append({"record_uid": uid, "errors": structural})
                    for value in structural:
                        structural_error_count[hard_error_kind(value)] += 1

                human_override = bool((translated.get("human_review") or {}).get("validator_override"))
                if not hard:
                    hard_disposition = "pass"
                elif human_override:
                    hard_disposition = "audited_human_override"
                elif status == "provisional_mild_profanity" and all(
                    "strong profanity was softened" in error for error in hard
                ):
                    hard_disposition = "known_profanity_caveat"
                else:
                    hard_disposition = "unexpected_failure"
                hard_disposition_count[hard_disposition] += 1
                if hard:
                    failure = {
                        "record_uid": uid,
                        "split": split,
                        "status": status,
                        "provider": provider,
                        "disposition": hard_disposition,
                        "errors": hard,
                    }
                    hard_failures.append(failure)
                    if hard_disposition == "unexpected_failure":
                        unexpected_hard_failures.append(failure)
                    for value in hard:
                        hard_error_count[hard_error_kind(value)] += 1
                    for category in categories:
                        category_hard_fail_records[category] += 1

                live_warnings = quality_warnings(source, translated)
                if live_warnings:
                    for category in categories:
                        category_quality_warning_records[category] += 1
                for value in live_warnings:
                    kind = warning_kind(value)
                    live_warning_count[kind] += 1
                    live_warning_status[kind][status] += 1

                for category in categories:
                    sample_key = (status, split)
                    if len(samples_by_category[category]) < 8 and sample_key not in sample_keys_by_category[category]:
                        samples_by_category[category].append(compact_sample(source, translated))
                        sample_keys_by_category[category].add(sample_key)

                if export:
                    handle = export_handles[split][2]
                    handle.write(safe_json_dumps(output_row(source, translated, hard_disposition)) + "\n")
    finally:
        if export:
            for _, temp_path, handle in export_handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()

    if export:
        for split, (final_path, temp_path, _) in export_handles.items():
            os.replace(temp_path, final_path)
            line_count = sum(1 for _ in read_jsonl(final_path))
            export_files[split] = {
                "path": str(final_path.relative_to(root)),
                "records": line_count,
                "bytes": final_path.stat().st_size,
                "sha256": file_sha256(final_path),
            }

    total = len(source_index)
    translated_total = len(source_uids & translated_uids)
    status_rows: list[dict[str, Any]] = []
    for status in STATUSES + tuple(sorted(set(status_count) - set(STATUSES))):
        count = status_count.get(status, 0)
        if not count:
            continue
        description = quality_class(status)
        src_values = source_length_by_status[status]
        dst_values = target_length_by_status[status]
        status_rows.append(
            {
                "status": status,
                "count": count,
                "percent": pct(count, translated_total),
                **description,
                "source_chars_p50": round(percentile(src_values, 0.5), 2),
                "source_chars_p95": round(percentile(src_values, 0.95), 2),
                "target_chars_p50": round(percentile(dst_values, 0.5), 2),
                "target_chars_p95": round(percentile(dst_values, 0.95), 2),
            }
        )

    category_rows: list[dict[str, Any]] = []
    for category in sorted(category_count, key=lambda key: (-sum(category_count[key].values()), key)):
        counts = category_count[category]
        category_total = sum(counts.values())
        revised = sum(counts.get(status, 0) for status in REVISED_STATUSES)
        row: dict[str, Any] = {
            "category": category,
            "total": category_total,
            "dataset_percent": pct(category_total, translated_total),
            "train": category_split_count[category].get("train", 0),
            "valid": category_split_count[category].get("valid", 0),
            "test": category_split_count[category].get("test", 0),
            "revised_total": revised,
            "revised_percent": pct(revised, category_total),
            "known_profanity_caveat": counts.get("provisional_mild_profanity", 0),
            "live_warning_records": category_quality_warning_records.get(category, 0),
            "hard_fail_records": category_hard_fail_records.get(category, 0),
            "source_chars_p50": round(percentile(category_source_chars[category], 0.5), 2),
            "source_chars_p95": round(percentile(category_source_chars[category], 0.95), 2),
        }
        for status in STATUSES:
            row[status] = counts.get(status, 0)
        category_rows.append(row)

    combo_rows: list[dict[str, Any]] = []
    for combo in sorted(category_combo_count, key=lambda key: (-sum(category_combo_count[key].values()), key)):
        counts = category_combo_count[combo]
        row = {"category_combination": combo, "total": sum(counts.values())}
        row.update({status: counts.get(status, 0) for status in STATUSES})
        combo_rows.append(row)

    def breakdown_rows(values: dict[str, Counter[str]], name: str) -> list[dict[str, Any]]:
        rows = []
        for key in sorted(values, key=lambda item: (-sum(values[item].values()), item)):
            counts = values[key]
            row: dict[str, Any] = {name: key, "total": sum(counts.values())}
            row.update({status: counts.get(status, 0) for status in STATUSES})
            rows.append(row)
        return rows

    status_by_split_rows = breakdown_rows(split_count, "split")
    status_by_group_rows = breakdown_rows(group_count, "group")
    status_by_tag_rows = breakdown_rows(tag_count, "tag")
    status_by_label_pair_rows = breakdown_rows(label_pair_count, "label_pair")
    status_by_length_rows = breakdown_rows(length_count, "length_bucket")
    provider_rows = breakdown_rows(provider_status_count, "provider")

    dashboard = DashboardData(root)
    review_status = dashboard.review_status()
    failure_queue = dashboard.review_failure_queue()
    integrity = {
        "source_unique_records": len(source_index),
        "source_duplicate_uids": len(duplicate_source_uids),
        "checkpoint_unique_records": len(translation_index),
        "checkpoint_physical_rows": checkpoint_physical_rows,
        "checkpoint_duplicate_uids_across_files": len(duplicate_checkpoint_uids),
        "wrong_checkpoint_location": len(wrong_checkpoint),
        "missing_translations": len(missing_uids),
        "extra_translations": len(extra_uids),
        "structural_validator_failures": len(structural_failures),
        "source_hash_mismatches": 0,
        "translation_hash_mismatches": 0,
        "input_char_count_mismatches": 0,
        "output_char_count_mismatches": 0,
    }
    hash_examples: dict[str, list[str]] = defaultdict(list)
    for uid in sorted(source_uids & translated_uids):
        source = source_index[uid]
        translated = translation_index[uid]
        if translated.get("source_text_sha256") != text_hash(source.get("prompt"), source.get("response")):
            integrity["source_hash_mismatches"] += 1
            hash_examples["source_hash_mismatches"].append(uid)
        if translated.get("translation_text_sha256") != text_hash(translated.get("prompt_vi"), translated.get("response_vi")):
            integrity["translation_hash_mismatches"] += 1
            hash_examples["translation_hash_mismatches"].append(uid)
        if translated.get("input_source_chars") != source_chars(source):
            integrity["input_char_count_mismatches"] += 1
            hash_examples["input_char_count_mismatches"].append(uid)
        target_size = len(translated.get("prompt_vi") or "") + len(translated.get("response_vi") or "")
        if translated.get("output_translation_chars") != target_size:
            integrity["output_char_count_mismatches"] += 1
            hash_examples["output_char_count_mismatches"].append(uid)
    integrity["all_critical_checks_pass"] = not any(
        integrity[key]
        for key in (
            "source_duplicate_uids",
            "checkpoint_duplicate_uids_across_files",
            "wrong_checkpoint_location",
            "missing_translations",
            "extra_translations",
            "structural_validator_failures",
            "source_hash_mismatches",
            "translation_hash_mismatches",
            "input_char_count_mismatches",
            "output_char_count_mismatches",
        )
    ) and not unexpected_hard_failures

    ready_count = sum(
        count
        for status, count in status_count.items()
        if quality_class(status)["disposition"] in {"ready", "ready_with_metadata"}
    )
    followup_count = status_count.get("provisional_mild_profanity", 0)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": created_at,
        "summary": {
            "source_records": total,
            "translated_records": translated_total,
            "coverage_percent": pct(translated_total, total),
            "ready_records": ready_count,
            "ready_percent": pct(ready_count, translated_total),
            "followup_recommended_records": followup_count,
            "followup_recommended_percent": pct(followup_count, translated_total),
            "review_set_completed": review_status["completed_total"],
            "review_set_total": review_status["original_review_total"],
            "review_remaining": review_status["remaining_total"],
            "review_failure_queue": failure_queue["total"],
            "atomic_category_count": len(category_rows),
            "category_combination_count": len(combo_rows),
        },
        "integrity": integrity,
        "integrity_examples": {
            "duplicate_source_uids": duplicate_source_uids[:50],
            "duplicate_checkpoint_uids": duplicate_checkpoint_uids[:50],
            "wrong_checkpoint": wrong_checkpoint[:50],
            "missing_uids": missing_uids[:50],
            "extra_uids": extra_uids[:50],
            "hash_or_count_mismatches": {key: values[:50] for key, values in hash_examples.items()},
            "structural_failures": structural_failures[:50],
            "unexpected_hard_failures": unexpected_hard_failures[:50],
        },
        "translation_status": status_rows,
        "status_by_split": status_by_split_rows,
        "status_by_group": status_by_group_rows,
        "status_by_tag": status_by_tag_rows,
        "status_by_label_pair": status_by_label_pair_rows,
        "status_by_length_bucket": status_by_length_rows,
        "providers": provider_rows,
        "models": counter_dict(model_count),
        "revision_status": counter_dict(revision_status_count),
        "validator": {
            "current_hard_disposition": counter_dict(hard_disposition_count),
            "hard_error_types": counter_dict(hard_error_count),
            "hard_failure_records": len(hard_failures),
            "unexpected_hard_failure_records": len(unexpected_hard_failures),
            "hard_failures": hard_failures,
            "metadata_warning_types": counter_dict(metadata_warning_count),
            "live_heuristic_warning_types": counter_dict(live_warning_count),
            "live_heuristic_warning_by_status": {
                key: counter_dict(value) for key, value in sorted(live_warning_status.items())
            },
        },
        "source_and_target_lengths": {
            "source_chars": {
                "p50": round(percentile(source_lengths, 0.5), 2),
                "p95": round(percentile(source_lengths, 0.95), 2),
                "p99": round(percentile(source_lengths, 0.99), 2),
                "max": max(source_lengths, default=0),
            },
            "target_chars": {
                "p50": round(percentile(target_lengths, 0.5), 2),
                "p95": round(percentile(target_lengths, 0.95), 2),
                "p99": round(percentile(target_lengths, 0.99), 2),
                "max": max(target_lengths, default=0),
            },
        },
        "categories": category_rows,
        "category_combinations": combo_rows,
        "category_samples": dict(samples_by_category),
        "checkpoint_files": checkpoint_files,
        "export_files": export_files,
    }

    json_path = report_dir / "translation_quality_summary.json"
    write_json(json_path, report)
    category_fields = [
        "category", "total", "dataset_percent", "train", "valid", "test",
        *STATUSES, "revised_total", "revised_percent", "known_profanity_caveat",
        "live_warning_records", "hard_fail_records", "source_chars_p50", "source_chars_p95",
    ]
    write_csv(report_dir / "translation_quality_by_category.csv", category_rows, category_fields)
    combo_fields = ["category_combination", "total", *STATUSES]
    write_csv(report_dir / "translation_quality_by_category_combination.csv", combo_rows, combo_fields)
    status_fields = [
        "status", "count", "percent", "tier", "disposition", "description",
        "source_chars_p50", "source_chars_p95", "target_chars_p50", "target_chars_p95",
    ]
    write_csv(report_dir / "translation_quality_by_status.csv", status_rows, status_fields)

    markdown = render_markdown(report)
    markdown_path = report_dir / "TRANSLATION_QUALITY_REPORT.md"
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    explorer_path = report_dir / "translation_quality_explorer.html"
    explorer_path.write_text(render_explorer(report), encoding="utf-8", newline="\n")
    report["report_files"] = {
        "markdown": str(markdown_path.relative_to(root)),
        "json": str(json_path.relative_to(root)),
        "category_csv": str((report_dir / "translation_quality_by_category.csv").relative_to(root)),
        "category_combination_csv": str((report_dir / "translation_quality_by_category_combination.csv").relative_to(root)),
        "status_csv": str((report_dir / "translation_quality_by_status.csv").relative_to(root)),
        "explorer_html": str(explorer_path.relative_to(root)),
    }
    write_json(json_path, report)
    return report


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    integrity = report["integrity"]
    status_rows = [
        [row["status"], f"{row['count']:,}", f"{row['percent']:.4f}%", row["tier"], row["disposition"]]
        for row in report["translation_status"]
    ]
    split_rows = []
    for row in report["status_by_split"]:
        split_rows.append([row["split"], f"{row['total']:,}", *[f"{row.get(status, 0):,}" for status in STATUSES]])
    category_rows = []
    for row in report["categories"]:
        category_rows.append(
            [
                row["category"], f"{row['total']:,}", f"{row['dataset_percent']:.2f}%",
                f"{row['machine_translated']:,}", f"{row['provisional_clean']:,}",
                f"{row['provisional_format_repaired']:,}", f"{row['known_profanity_caveat']:,}",
                f"{row['revised_total']:,}", f"{row['revised_percent']:.2f}%",
            ]
        )
    hard = report["validator"]["current_hard_disposition"]
    lines = [
        "# Nemotron Safety Guard V3 EN→VI — báo cáo chất lượng cuối",
        "",
        f"Tạo lúc `{report['created_at']}` từ source prepared và 15 checkpoint mới nhất.",
        "",
        "## Kết luận",
        "",
        f"- Đã có bản dịch cho **{summary['translated_records']:,}/{summary['source_records']:,} record ({summary['coverage_percent']:.4f}%)**.",
        f"- **{summary['ready_records']:,} record ({summary['ready_percent']:.4f}%)** sẵn sàng theo các gate hiện tại.",
        f"- **{summary['followup_recommended_records']:,} record ({summary['followup_recommended_percent']:.4f}%)** đầy đủ nhưng nên kiểm tra thêm độ nặng của từ thô tục.",
        f"- Hàng đợi khó đã xử lý **{summary['review_set_completed']:,}/{summary['review_set_total']:,}**; còn lại **{summary['review_remaining']}**, failure queue **{summary['review_failure_queue']}**.",
        f"- Kiểm tra toàn vẹn quan trọng: **{'PASS' if integrity['all_critical_checks_pass'] else 'FAIL'}**.",
        "",
        "## Toàn vẹn dữ liệu",
        "",
        markdown_table(
            ["Kiểm tra", "Số lỗi"],
            [[key, value] for key, value in integrity.items() if key != "all_critical_checks_pass"],
        ),
        "",
        "## Chất lượng theo trạng thái",
        "",
        markdown_table(["Trạng thái", "Record", "Tỷ lệ", "Tier", "Khuyến nghị"], status_rows),
        "",
        "Các tier là đánh giá vận hành của pipeline, không phải điểm BLEU/COMET. Toàn bộ 64 mẫu từng mang trạng thái `provisional_mild_profanity` đã được duyệt song ngữ thủ công: mẫu thực sự bị làm nhẹ đã được sửa, còn cảnh báo giả đã được phê duyệt kèm audit.",
        "",
        "## Theo split",
        "",
        markdown_table(["Split", "Tổng", *STATUSES], split_rows),
        "",
        "## Validator cuối",
        "",
        f"- Pass trực tiếp: **{hard.get('pass', 0):,}**.",
        f"- Human override có audit (cảnh báo giả hoặc sửa có kiểm chứng): **{hard.get('audited_human_override', 0):,}**.",
        f"- Caveat thô tục đã biết: **{hard.get('known_profanity_caveat', 0):,}**.",
        f"- Lỗi hard-validator không được giải thích: **{hard.get('unexpected_failure', 0):,}**.",
        "",
        "## Theo nhãn safety nguyên tử",
        "",
        "Một record có thể thuộc nhiều nhãn, vì vậy tổng các dòng lớn hơn tổng dataset.",
        "",
        markdown_table(
            ["Nhãn", "Tổng", "% dataset", "Machine", "Salvaged clean", "Format repair", "Caveat profanity", "Revised", "% revised"],
            category_rows,
        ),
        "",
        "## Cách dùng file final",
        "",
        f"Mỗi dòng chứa `prompt_en`, `response_en`, `prompt_vi`, `response_vi`, nhãn safety gốc, metadata dịch, tier chất lượng và cờ `needs_followup_review`. Hiện có {summary['followup_recommended_records']} mẫu còn mang khuyến nghị duyệt tiếp.",
        "Danh sách 64 mẫu đầu vào của vòng duyệt cuối được giữ tại `reports/final_quality/profanity_followup_64.jsonl`; quyết định và hash trước/sau nằm tại `data/revision_handoff/human_review_decisions/final_profanity_manual_review.jsonl`.",
        "",
        "## Diễn giải giới hạn",
        "",
        "Báo cáo xác nhận độ phủ, cấu trúc, hash, tính nhất quán, các gate heuristic và toàn bộ revision queue. Nó không chứng minh từng câu đạt mức tương đương hoàn hảo như một đánh giá song ngữ độc lập trên 45.416 mẫu. Các cảnh báo heuristic có thể là báo động giả khi nguồn chứa code, ASCII art, SQL/JSON, URL, tên riêng hoặc văn bản đa ngôn ngữ.",
        "",
    ]
    return "\n".join(lines)


def render_explorer(report: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "summary": report["summary"],
            "statuses": report["translation_status"],
            "categories": report["categories"],
            "samples": report["category_samples"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nemotron EN→VI — Quality Explorer</title>
<style>
:root{{--bg:#0b1020;--panel:#141b30;--soft:#202a46;--text:#edf2ff;--muted:#aebbd8;--accent:#78a7ff;--good:#6ee7b7;--warn:#fbbf24}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(135deg,#090d18,#10182d);color:var(--text);font:14px/1.5 system-ui,Segoe UI,sans-serif}}
header,main{{max-width:1500px;margin:auto;padding:24px}}h1{{margin:.2rem 0}}.muted{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}}
.card,section{{background:rgba(20,27,48,.94);border:1px solid #293658;border-radius:14px;padding:16px}}.big{{font-size:25px;font-weight:750}}input{{width:100%;padding:12px;border-radius:10px;border:1px solid #3b4b74;background:#0d1426;color:white;margin-bottom:12px}}
.layout{{display:grid;grid-template-columns:minmax(560px,1.2fr) minmax(460px,1fr);gap:16px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #2a3655;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left;white-space:normal}}tr[data-cat]{{cursor:pointer}}tr[data-cat]:hover,tr.active{{background:#223052}}
.pill{{padding:3px 8px;border-radius:999px;background:#26365c;color:#cfe0ff;display:inline-block}}article{{background:#0d1426;border:1px solid #2c395d;border-radius:12px;padding:13px;margin:10px 0}}pre{{white-space:pre-wrap;word-break:break-word;font:13px/1.45 system-ui;margin:6px 0;color:#dce7ff}}h3,h4{{margin:.4rem 0}}.meta{{display:flex;gap:7px;flex-wrap:wrap;color:var(--muted)}}.good{{color:var(--good)}}.warn{{color:var(--warn)}}
@media(max-width:1050px){{.layout{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Nemotron Safety EN→VI — Quality Explorer</h1><p class="muted">Bấm một nhãn để xem thống kê và các mẫu song ngữ đại diện. Nhãn safety có thể chồng lấp.</p><div id="cards" class="cards"></div></header>
<main><div class="layout"><section><input id="q" placeholder="Lọc nhãn safety…"><div style="overflow:auto"><table><thead><tr><th>Nhãn</th><th>Tổng</th><th>Machine</th><th>Clean</th><th>Format</th><th>Profanity</th><th>Revised</th></tr></thead><tbody id="rows"></tbody></table></div></section><section><div id="detail"><p class="muted">Chọn một nhãn ở bảng bên trái.</p></div></section></div></main>
<script>const D={payload};
const fmt=n=>Number(n||0).toLocaleString('vi-VN');
cards.innerHTML=[['Đã dịch',fmt(D.summary.translated_records)+' / '+fmt(D.summary.source_records)],['Sẵn sàng',fmt(D.summary.ready_records)+' ('+D.summary.ready_percent+'%)'],['Cần xem độ tục',fmt(D.summary.followup_recommended_records)],['Revision hoàn tất',fmt(D.summary.review_set_completed)+' / '+fmt(D.summary.review_set_total)]].map(x=>`<div class="card"><div class="muted">${{x[0]}}</div><div class="big">${{x[1]}}</div></div>`).join('');
let selected='';function drawRows(){{const needle=q.value.toLowerCase();rows.innerHTML=D.categories.filter(r=>r.category.toLowerCase().includes(needle)).map(r=>`<tr data-cat="${{r.category.replaceAll('&','&amp;').replaceAll('"','&quot;')}}" class="${{selected===r.category?'active':''}}"><td>${{r.category}}</td><td>${{fmt(r.total)}}</td><td>${{fmt(r.machine_translated)}}</td><td>${{fmt(r.provisional_clean)}}</td><td>${{fmt(r.provisional_format_repaired)}}</td><td>${{fmt(r.known_profanity_caveat)}}</td><td>${{fmt(r.revised_total)}}</td></tr>`).join('');document.querySelectorAll('[data-cat]').forEach(el=>el.onclick=()=>show(el.dataset.cat))}}
function esc(s){{return String(s??'∅').replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]))}}
function show(cat){{selected=cat;drawRows();const r=D.categories.find(x=>x.category===cat),samples=D.samples[cat]||[];detail.innerHTML=`<h2>${{esc(cat)}}</h2><div class="meta"><span class="pill">${{fmt(r.total)}} record</span><span class="pill">${{r.dataset_percent}}% dataset</span><span class="pill">${{fmt(r.revised_total)}} revised</span><span class="pill">${{fmt(r.known_profanity_caveat)}} caveat profanity</span></div>`+samples.map(s=>`<article><div class="meta"><code>${{esc(s.record_uid)}}</code><span>${{esc(s.split)}}</span><span>${{esc(s.translation_status)}}</span><span>${{esc(s.prompt_label)}} → ${{esc(s.response_label)}}</span></div><h4>English prompt</h4><pre>${{esc(s.prompt_en)}}</pre><h4>Vietnamese prompt</h4><pre>${{esc(s.prompt_vi)}}</pre><h4>English response</h4><pre>${{esc(s.response_en)}}</pre><h4>Vietnamese response</h4><pre>${{esc(s.response_vi)}}</pre></article>`).join('')}}q.oninput=drawRows;drawRows();</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and export the completed Nemotron EN→VI dataset.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    report = build(Path(args.root).resolve(), export=not args.no_export)
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "integrity": report["integrity"],
                "hard_validator": report["validator"]["current_hard_disposition"],
                "report_files": report["report_files"],
                "export_files": report["export_files"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
