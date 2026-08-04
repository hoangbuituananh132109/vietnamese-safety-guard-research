from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse

from translator.batching import source_chars
from translator.checkpoint import load_checkpoint, save_completed
from translator.full_run import paths
from translator.jsonl_io import append_jsonl, read_jsonl
from translator.models import TranslationInputItem, TranslationRequest, TranslationResponse
from translator.pipeline import text_hash
from translator.validators import TranslationValidationError, validate_hard_quality, validate_response


REVIEW_DONE_STATUSES = {"terra_revised", "luna_revised", "gemini_revised"}


def parse_revision_output(raw: str) -> list[dict[str, Any]]:
    """Accept a response object, an array, or plain JSONL (with optional fences)."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|jsonl)?\s*", "", text, count=1, flags=re.I)
        text = re.sub(r"\s*```$", "", text, count=1)
    if not text:
        raise ValueError("Ô kết quả đang trống.")
    try:
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("items"), list):
            rows = value["items"]
        elif isinstance(value, list):
            rows = value
        elif isinstance(value, dict) and "record_uid" in value:
            rows = [value]
        else:
            raise ValueError("JSON phải là object có items, một array, hoặc một record revision.")
    except json.JSONDecodeError:
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON/JSONL lỗi tại dòng {number}: {exc.msg} (cột {exc.colno}).") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Không tìm thấy record JSON hợp lệ trong kết quả.")
    return rows


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


class DashboardData:
    def __init__(self, root: Path):
        self.root = root
        self.run_dir = root / "data" / "run"
        self.config_path = root / "configs" / "key_assignments.json"
        self.page_path = root / "dashboard" / "index.html"
        self.review_page_path = root / "dashboard" / "review.html"
        self.review_failure_root = root / "data" / "revision_handoff" / "review_failures"
        self._review_failure_lock = Lock()
        self._source_cache: dict[str, tuple[int, str, dict[str, Any]]] | None = None

    def source_index(self) -> dict[str, tuple[int, str, dict[str, Any]]]:
        if self._source_cache is not None:
            return self._source_cache
        indexed: dict[str, tuple[int, str, dict[str, Any]]] = {}
        for split in ("train", "valid", "test"):
            input_path = self.root / "data" / "prepared" / f"nemotron_en_{split}_full_v1.jsonl"
            for line_number, row, _ in read_jsonl(input_path):
                indexed[row["record_uid"]] = ((line_number - 1) % 5 + 1, split, row)
        self._source_cache = indexed
        return indexed

    def review_dir(self, reviewer: str) -> Path:
        if reviewer not in {"terra", "luna"}:
            raise ValueError("reviewer phải là terra hoặc luna")
        return self.root / "data" / "revision_handoff" / f"logical_{reviewer}_batches"

    def review_manifest(self, reviewer: str) -> list[dict[str, Any]]:
        path = self.review_dir(reviewer) / "manifest.jsonl"
        if not path.exists():
            raise ValueError(f"Chưa có manifest batch cho {reviewer}.")
        return [row for _, row, _ in read_jsonl(path)]

    def completed_review_uids(self) -> dict[str, dict[str, Any]]:
        completed: dict[str, dict[str, Any]] = {}
        for group in range(5):
            for split in ("train", "valid", "test"):
                for uid, row in load_checkpoint(paths(self.root, split, group)["checkpoint"]).items():
                    if row.get("translation_status") in REVIEW_DONE_STATUSES:
                        completed[uid] = row
        return completed

    def review_rows(self, reviewer: str, batch: int) -> list[dict[str, Any]]:
        manifest = self.review_manifest(reviewer)
        if batch < 1 or batch > len(manifest):
            raise ValueError(f"batch phải nằm trong khoảng 1..{len(manifest)}")
        path = self.review_dir(reviewer) / manifest[batch - 1]["file"]
        return [row for _, row, _ in read_jsonl(path)]

    def review_failure_path(self, reviewer: str, batch: int) -> Path:
        self.review_dir(reviewer)  # Validate the reviewer name.
        return self.review_failure_root / reviewer / f"batch_{batch:03d}.json"

    @staticmethod
    def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def persist_review_failures(
        self,
        reviewer: str,
        batch: int,
        raw: str,
        model: str,
        validation: dict[str, Any],
    ) -> dict[str, int]:
        """Keep only the latest rejected candidate for every pending UID in a batch."""
        target = self.review_failure_path(reviewer, batch)
        source_rows = {row["record_uid"]: row for row in self.review_rows(reviewer, batch)}
        try:
            parsed = parse_revision_output(raw)
        except ValueError:
            parsed = []
        parsed_by_uid: dict[str, dict[str, Any]] = {}
        for item in parsed:
            uid = item.get("record_uid")
            if isinstance(uid, str) and uid not in parsed_by_uid:
                parsed_by_uid[uid] = item

        with self._review_failure_lock:
            previous: dict[str, Any] = {}
            if target.exists():
                try:
                    previous = json.loads(target.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    previous = {}
            previous_by_key = {
                str(item.get("review_key")): item
                for item in previous.get("failures", [])
                if isinstance(item, dict) and item.get("review_key") is not None
            }
            now = iso_now()
            failures: list[dict[str, Any]] = []
            unmatched_number = 0
            for result in validation.get("results", []):
                if result.get("status") not in {"fail", "missing", "invalid", "unknown"}:
                    continue
                uid = result.get("record_uid")
                if isinstance(uid, str) and uid in source_rows:
                    review_key = uid
                    source_row = source_rows[uid]
                else:
                    unmatched_number += 1
                    review_key = f"batch:{batch:03d}:unmatched:{unmatched_number}"
                    source_row = None
                old = previous_by_key.get(review_key, {})
                candidate = parsed_by_uid.get(uid) if isinstance(uid, str) else None
                candidate_reused = False
                if candidate is None and isinstance(old.get("candidate"), dict):
                    # A later response may omit the UID entirely. Keep the last
                    # actual rejected translation instead of replacing it with
                    # an empty/missing placeholder.
                    candidate = old["candidate"]
                    candidate_reused = True
                failures.append({
                    "review_key": review_key,
                    "record_uid": uid if isinstance(uid, str) else None,
                    "reviewer": reviewer,
                    "batch": batch,
                    "status": result.get("status"),
                    "errors": list(result.get("errors") or []),
                    "candidate": candidate,
                    "candidate_reused_from_previous_attempt": candidate_reused,
                    "source_split": source_row.get("source_split") if source_row else None,
                    "group": source_row.get("group") if source_row else None,
                    "source": source_row.get("source") if source_row else None,
                    "model": model or "external-review-model",
                    "first_seen_at": old.get("first_seen_at") or now,
                    "last_seen_at": now,
                    "attempt_count": int(old.get("attempt_count") or 0) + 1,
                })

            previous_count = len(previous.get("failures", [])) if isinstance(previous.get("failures"), list) else 0
            if not failures:
                if target.exists():
                    target.unlink()
                return {"saved_failures": 0, "replaced_failures": 0, "cleared_failures": previous_count}

            snapshot = {
                "schema_version": 1,
                "reviewer": reviewer,
                "batch": batch,
                "model": model or "external-review-model",
                "saved_at": now,
                "attempt_count": int(previous.get("attempt_count") or 0) + 1,
                "raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "raw_output": raw,
                "validation_summary": {
                    "expected": validation.get("expected", 0),
                    "received": validation.get("received", 0),
                    "pass_count": validation.get("pass_count", 0),
                    "fail_count": validation.get("fail_count", len(failures)),
                    "parse_error": validation.get("parse_error"),
                },
                "failures": failures,
            }
            self._atomic_write_json(target, snapshot)
            replaced = sum(item["review_key"] in previous_by_key for item in failures)
            return {
                "saved_failures": len(failures),
                "replaced_failures": replaced,
                "cleared_failures": max(0, previous_count - replaced),
            }

    def review_failure_queue(
        self,
        reviewer: str | None = None,
        completed: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if reviewer is not None:
            self.review_dir(reviewer)
            reviewers = (reviewer,)
        else:
            reviewers = ("terra", "luna")
        completed = completed if completed is not None else self.completed_review_uids()
        records: list[dict[str, Any]] = []
        stale_count = 0
        for name in reviewers:
            directory = self.review_failure_root / name
            if not directory.exists():
                continue
            for path in sorted(directory.glob("batch_*.json")):
                try:
                    snapshot = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for item in snapshot.get("failures", []):
                    if not isinstance(item, dict):
                        continue
                    uid = item.get("record_uid")
                    if isinstance(uid, str) and uid in completed:
                        stale_count += 1
                        continue
                    records.append(item)
        records.sort(key=lambda item: (str(item.get("last_seen_at") or ""), str(item.get("review_key") or "")), reverse=True)
        by_status = Counter(str(item.get("status") or "unknown") for item in records)
        by_reviewer = Counter(str(item.get("reviewer") or "unknown") for item in records)
        return {
            "updated_at": iso_now(),
            "total": len(records),
            "stale_completed": stale_count,
            "by_status": dict(by_status),
            "by_reviewer": dict(by_reviewer),
            "records": records,
        }

    def review_status(self) -> dict[str, Any]:
        completed = self.completed_review_uids()
        reviewers: dict[str, Any] = {}
        for reviewer in ("terra", "luna"):
            batches = []
            total = done = 0
            for number, entry in enumerate(self.review_manifest(reviewer), 1):
                uids = entry["record_uids"]
                count_done = sum(uid in completed for uid in uids)
                total += len(uids)
                done += count_done
                batches.append({
                    "number": number,
                    "file": entry["file"],
                    "records": len(uids),
                    "done": count_done,
                    "remaining": len(uids) - count_done,
                    "source_chars": entry.get("source_chars", 0),
                    "oversized_records": entry.get("oversized_records", []),
                })
            reviewers[reviewer] = {"total": total, "done": done, "remaining": total - done, "batches": batches}
        original_total = 478
        revised = Counter(row.get("translation_status") for row in completed.values())
        failure_queue = self.review_failure_queue(completed=completed)
        return {
            "updated_at": iso_now(),
            "original_review_total": original_total,
            "completed_total": sum(revised.values()),
            "remaining_total": max(0, original_total - sum(revised.values())),
            "completed_by_status": dict(revised),
            "failure_queue": {
                "total": failure_queue["total"],
                "by_status": failure_queue["by_status"],
                "by_reviewer": failure_queue["by_reviewer"],
            },
            "reviewers": reviewers,
        }

    def build_review_prompt(self, reviewer: str, batch: int) -> tuple[str, list[dict[str, Any]]]:
        completed = self.completed_review_uids()
        rows = [row for row in self.review_rows(reviewer, batch) if row["record_uid"] not in completed]
        translation_prompt_path = self.root / "configs" / "revision_fresh_translation_prompt.md"
        translation_prompt = translation_prompt_path.read_text(encoding="utf-8")
        header = f"""# NEMOTRON SAFETY EN→VI — FRESH TRANSLATION BATCH {batch:03d}

{translation_prompt.strip()}

## Mục tiêu của lượt này

- Dịch mới hoàn toàn đúng {len(rows)} record bên dưới từ tiếng Anh sang tiếng Việt.
- Không có candidate hoặc lỗi validator cũ trong prompt này; không phân tích bản dịch trước.
- Tiếng Anh trong `source` là nguồn chân lý duy nhất.
- Dịch đầy đủ prompt lẫn response, kể cả response không liên quan prompt, văn bản lặp, bị cắt hoặc giống một prompt mới.
- Giữ chính xác null so với chuỗi rỗng, URL, PII, số liệu, Markdown, code fence, placeholder, identifier và JSON/schema key.
- Slur, profanity, tình dục, đe dọa, quấy rối và ý định phạm tội phải giữ mức độ tương đương; không làm nhẹ, không thêm refusal/cảnh báo.
- Leetspeak: giải mã nghĩa tiếng Anh, dịch sang tiếng Việt, rồi mới obfuscate nhẹ bằng ký tự/số nếu nguồn có obfuscation.

## Đầu ra bắt buộc

Chỉ trả về đúng một JSON object, không kèm Markdown hay giải thích:

{{
  "batch_id": "review-{reviewer}-{batch:03d}",
  "items": [
    {{
      "record_uid": "giữ nguyên UID",
      "prompt_vi": "bản dịch đầy đủ",
      "response_vi": null,
      "review_status": "revised"
    }}
  ]
}}

Phải trả đúng một item cho mỗi `record_uid` trong batch, không thêm UID khác.

## Record cần review
"""
        blocks = []
        for position, row in enumerate(rows, 1):
            record = {
                "position": position,
                "record_uid": row["record_uid"],
                "labels": {
                    "violated_categories": row.get("source", {}).get("violated_categories"),
                    "prompt_label": row.get("source", {}).get("prompt_label"),
                    "response_label": row.get("source", {}).get("response_label"),
                },
                "source": {
                    "prompt_en": row.get("source", {}).get("prompt_en"),
                    "response_en": row.get("source", {}).get("response_en"),
                },
            }
            blocks.append(f"\n### RECORD {position}/{len(rows)}\n" + json.dumps(record, ensure_ascii=False, indent=2))
        return header + "".join(blocks), rows

    @staticmethod
    def validate_review_item(source_row: dict[str, Any], revision: dict[str, Any]) -> list[str]:
        uid = source_row["record_uid"]
        source = source_row["source"]
        try:
            request = TranslationRequest(batch_id="web-review", items=[TranslationInputItem(
                seq=1,
                record_uid=uid,
                prompt=source.get("prompt_en") or "",
                response=source.get("response_en"),
            )])
            response = TranslationResponse.model_validate({
                "batch_id": "web-review",
                "items": [{
                    "seq": 1,
                    "record_uid": uid,
                    "prompt_vi": revision.get("prompt_vi"),
                    "response_vi": revision.get("response_vi"),
                    "warnings": [],
                }],
            })
            validate_response(request, response)
            validate_hard_quality(request, response)
            return []
        except (TranslationValidationError, ValueError) as exc:
            return list(exc.errors) if isinstance(exc, TranslationValidationError) else [str(exc)]

    def validate_review_output(self, reviewer: str, batch: int, raw: str) -> dict[str, Any]:
        rows = self.review_rows(reviewer, batch)
        completed = self.completed_review_uids()
        pending = {row["record_uid"]: row for row in rows if row["record_uid"] not in completed}
        try:
            parsed = parse_revision_output(raw)
        except ValueError as exc:
            failure = {"record_uid": None, "status": "invalid", "errors": [str(exc)]}
            return {
                "ok": False,
                "reviewer": reviewer,
                "batch": batch,
                "expected": len(pending),
                "received": 0,
                "pass_count": 0,
                "fail_count": 1,
                "results": [failure],
                "normalized": [],
                "parse_error": str(exc),
                "repair_prompt": self.build_repair_prompt(reviewer, batch, raw, [failure]),
            }
        seen: set[str] = set()
        results = []
        normalized = []
        for revision in parsed:
            uid = revision.get("record_uid")
            if not isinstance(uid, str):
                results.append({"record_uid": None, "status": "invalid", "errors": ["Thiếu record_uid dạng chuỗi."]})
                continue
            if uid in seen:
                results.append({"record_uid": uid, "status": "invalid", "errors": ["record_uid bị lặp trong output."]})
                continue
            seen.add(uid)
            if uid not in pending:
                status = "already_done" if uid in completed else "unknown"
                results.append({"record_uid": uid, "status": status, "errors": ["UID không thuộc phần pending của batch này."]})
                continue
            value = {
                "record_uid": uid,
                "prompt_vi": revision.get("prompt_vi"),
                "response_vi": revision.get("response_vi"),
                "review_status": "revised",
                "fix_notes": revision.get("fix_notes") if isinstance(revision.get("fix_notes"), list) else [],
            }
            errors = self.validate_review_item(pending[uid], value)
            status = "pass" if not errors else "fail"
            results.append({"record_uid": uid, "status": status, "errors": errors})
            if not errors:
                normalized.append(value)
        missing = [uid for uid in pending if uid not in seen]
        for uid in missing:
            results.append({"record_uid": uid, "status": "missing", "errors": ["Model chưa trả item cho UID này."]})
        failed = [row for row in results if row["status"] in {"fail", "missing", "invalid", "unknown"}]
        repair_prompt = self.build_repair_prompt(reviewer, batch, raw, failed) if failed else ""
        return {
            "ok": not failed,
            "reviewer": reviewer,
            "batch": batch,
            "expected": len(pending),
            "received": len(parsed),
            "pass_count": len(normalized),
            "fail_count": len(failed),
            "results": results,
            "normalized": normalized,
            "repair_prompt": repair_prompt,
        }

    def build_repair_prompt(self, reviewer: str, batch: int, previous_output: str, failures: list[dict[str, Any]]) -> str:
        failure_text = json.dumps(failures, ensure_ascii=False, indent=2)
        failed_uids = {row.get("record_uid") for row in failures if isinstance(row.get("record_uid"), str)}
        sources = [
            {
                "record_uid": row["record_uid"],
                "source": {
                    "prompt_en": row.get("source", {}).get("prompt_en"),
                    "response_en": row.get("source", {}).get("response_en"),
                },
            }
            for row in self.review_rows(reviewer, batch)
            if row["record_uid"] in failed_uids
        ]
        source_text = json.dumps(sources, ensure_ascii=False, indent=2)
        return f"""Bạn đang sửa output của Nemotron EN→VI review batch {reviewer}-{batch:03d}.

Output trước đã qua validator và còn các lỗi dưới đây:
{failure_text}

Hãy sửa đúng các lỗi nêu trên. Giữ nguyên mọi item đã đúng nếu chúng có trong output. Không bỏ UID, không thêm UID, không giải thích, không dùng Markdown fence. Trả lại đúng JSON object có `batch_id` và `items` theo schema ban đầu.

Các nguyên tắc quan trọng: giữ null/chuỗi rỗng; dịch hết tiếng Anh có nghĩa; giữ identifier/schema key/URL/PII; leetspeak phải thành tiếng Việt rồi mới obfuscate; không làm nhẹ profanity/slur/nội dung độc hại; không thêm refusal.

NGUỒN TIẾNG ANH CHO CÁC UID BỊ LỖI/THIẾU:
{source_text}

OUTPUT CẦN SỬA:
{previous_output.strip()}
"""

    def merge_review_output(self, reviewer: str, batch: int, raw: str, model: str) -> dict[str, Any]:
        validation = self.validate_review_output(reviewer, batch, raw)
        failure_metrics = self.persist_review_failures(reviewer, batch, raw, model, validation)
        index = self.source_index() if validation["normalized"] else {}
        accepted = skipped = 0
        archive_path = self.root / "data" / "revision_handoff" / f"{reviewer}_web_results" / f"batch_{batch:03d}.jsonl"
        for revision in validation["normalized"]:
            uid = revision["record_uid"]
            group, split, source = index[uid]
            checkpoint = paths(self.root, split, group - 1)["checkpoint"]
            existing = load_checkpoint(checkpoint).get(uid)
            if existing and existing.get("translation_status") in REVIEW_DONE_STATUSES:
                skipped += 1
                continue
            timestamp = iso_now()
            status = f"{reviewer}_revised"
            translated = {
                "record_uid": uid,
                "prompt_vi": revision["prompt_vi"],
                "response_vi": revision.get("response_vi"),
                "translation_provider": "manual_web_review",
                "translation_model": model or "external-review-model",
                "translation_prompt_version": f"revision-{status}-web-v1",
                "translation_batch_id": f"review-{reviewer}-{batch:03d}",
                "translation_attempt": 1,
                "translation_status": status,
                "translation_api_key_slot": None,
                "translated_at": timestamp,
                "source_text_sha256": text_hash(source.get("prompt"), source.get("response")),
                "translation_text_sha256": text_hash(revision.get("prompt_vi"), revision.get("response_vi")),
                "input_source_chars": source_chars(source),
                "output_translation_chars": len(revision.get("prompt_vi") or "") + len(revision.get("response_vi") or ""),
                "warnings": [],
                "usage_metadata": None,
                "validation_warnings": ["web_manual_revision", *revision.get("fix_notes", [])],
                "revision_status": "revised",
            }
            save_completed(checkpoint, translated)
            append_jsonl(archive_path, {**revision, "merged_at": timestamp, "model": model}, fsync=True)
            accepted += 1
        return {
            **validation,
            **failure_metrics,
            "merged": accepted,
            "skipped": skipped,
            "status_after_merge": self.review_status(),
        }

    def review_batch_payload(self, reviewer: str, batch: int) -> dict[str, Any]:
        prompt, rows = self.build_review_prompt(reviewer, batch)
        completed = self.completed_review_uids()
        summaries = []
        for row in rows:
            summaries.append({
                "record_uid": row["record_uid"],
                "source_chars": row.get("logical_review", {}).get("source_chars", 0),
                "oversized": row.get("logical_review", {}).get("oversized", False),
                "categories": row.get("source", {}).get("violated_categories"),
                "prompt_label": row.get("source", {}).get("prompt_label"),
                "response_label": row.get("source", {}).get("response_label"),
                "done": row["record_uid"] in completed,
                "source": row.get("source"),
            })
        return {
            "reviewer": reviewer,
            "batch": batch,
            "prompt": prompt,
            "pending_records": len(rows),
            "records": summaries,
        }

    def api_count(self) -> int:
        return sum(1 for line in (self.root / "API.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip())

    def assignments(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        count = self.api_count()
        return {"version": 1, "groups": {str(i): [] for i in range(1, 6)}, "unassigned": list(range(1, count + 1))}

    def save_assignments(self, payload: dict) -> dict:
        count = self.api_count()
        groups = payload.get("groups")
        if not isinstance(groups, dict) or set(groups) != {"1", "2", "3", "4", "5"}:
            raise ValueError("groups must contain exactly 1..5")
        normalized: dict[str, list[int]] = {}
        used: list[int] = []
        for group in map(str, range(1, 6)):
            slots = groups[group]
            if not isinstance(slots, list):
                raise ValueError(f"group {group} must be a list")
            normalized[group] = [int(slot) for slot in slots]
            used.extend(normalized[group])
        if len(used) != len(set(used)):
            raise ValueError("a key can belong to only one group")
        if any(slot < 1 or slot > count for slot in used):
            raise ValueError(f"key slots must be between 1 and {count}")
        value = {
            "version": 1, "updated_at": iso_now(), "groups": normalized,
            "unassigned": [slot for slot in range(1, count + 1) if slot not in set(used)],
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.config_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, self.config_path)
        return value

    def restart_group(self, group: int) -> dict:
        if group not in range(1, 6):
            raise ValueError("group must be between 1 and 5")
        script = self.root / "scripts" / "restart_group_worker.ps1"
        if not script.exists():
            raise ValueError("restart_group_worker.ps1 is missing")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(script), "-Group", str(group),
                "-RequestIntervalSeconds", "60", "-QuotaCooldownSeconds", "60",
            ],
            cwd=self.root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        return {"group": group, "launcher_pid": process.pid}

    def reset_window(self) -> tuple[datetime, datetime]:
        # Gemini resets RPD at midnight Pacific. The current translation run is
        # in July (PDT, UTC-7); using a fixed offset avoids requiring tzdata on
        # minimal Windows Python installations.
        pacific = timezone(timedelta(hours=-7), "PDT")
        now_pt = datetime.now(pacific)
        start_pt = now_pt.replace(hour=0, minute=0, second=0, microsecond=0)
        next_pt = start_pt + timedelta(days=1)
        return start_pt.astimezone(timezone.utc), next_pt.astimezone(timezone.utc)

    def events(self) -> list[dict]:
        rows: list[dict] = []
        for group in range(1, 6):
            path = self.run_dir / f"key_events_g{group}.jsonl"
            if not path.exists():
                continue
            try:
                for _, row, _ in read_jsonl(path):
                    row["group"] = group
                    rows.append(row)
            except (OSError, json.JSONDecodeError):
                continue
        rows.sort(key=lambda row: row.get("timestamp", ""))
        return rows

    def checkpoint_metrics(self) -> tuple[dict[int, dict], dict[int, dict]]:
        by_key = {slot: {"success": 0, "tokens": 0} for slot in range(1, self.api_count() + 1)}
        by_group: dict[int, dict] = {}
        for group in range(1, 6):
            total = 0
            latest = None
            for split in ("train", "valid", "test"):
                path = self.root / "data" / "checkpoints" / f"nemotron_{split}_full_v10_g{group}_completed.jsonl"
                if not path.exists():
                    continue
                try:
                    for _, row, _ in read_jsonl(path):
                        total += 1
                        latest = row
                        slot = row.get("translation_api_key_slot")
                        if isinstance(slot, int) and slot in by_key:
                            by_key[slot]["success"] += 1
                            usage = row.get("usage_metadata") or {}
                            by_key[slot]["tokens"] += int(usage.get("total_token_count") or 0)
                except (OSError, json.JSONDecodeError):
                    pass
            state_path = self.run_dir / f"worker_{group}.json"
            state = {}
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
            by_group[group] = {
                "completed": total, "latest": latest,
                "worker_status": state.get("status", "unknown"),
                "worker_pid": state.get("pid"), "alive": process_alive(state.get("pid")),
                "cooldown_until": state.get("cooldown_until"), "reason": state.get("reason"),
            }
        return by_key, by_group

    def status(self) -> dict:
        reset_start, reset_next = self.reset_window()
        all_events = self.events()
        current_events = [row for row in all_events if row.get("timestamp", "") >= reset_start.isoformat()]
        by_key, groups = self.checkpoint_metrics()
        last_event: dict[int, dict] = {}
        for event in current_events:
            slot = event.get("key_slot")
            if not isinstance(slot, int) or slot not in by_key:
                continue
            metrics = by_key[slot]
            kind = event.get("event")
            if kind == "request_started": metrics["requests_today"] = metrics.get("requests_today", 0) + 1
            elif kind == "success": metrics["api_success_today"] = metrics.get("api_success_today", 0) + 1
            elif kind == "quota_429":
                metrics["errors_429"] = metrics.get("errors_429", 0) + 1
                if event.get("quota_kind") == "rpd_500": metrics["rpd_500"] = True
            elif kind == "key_disabled": metrics["disabled"] = True
            last_event[slot] = event
        keys = []
        assignment = self.assignments()
        owner = {slot: int(group) for group, slots in assignment["groups"].items() for slot in slots}
        for slot, metrics in by_key.items():
            requests = metrics.get("requests_today", 0)
            latest = last_event.get(slot, {})
            if metrics.get("disabled"):
                status = "disabled"
            elif metrics.get("rpd_500"):
                status = "rpd_exhausted"
            elif latest.get("event") == "quota_429":
                status = "overloaded"
            elif latest.get("event") == "request_started":
                status = "running"
            elif latest.get("event") == "success":
                status = "healthy"
            else:
                status = "idle"
            keys.append({
                "slot": slot, "group": owner.get(slot), "status": status,
                "requests_today": requests, "success": metrics.get("success", 0),
                "api_success_today": metrics.get("api_success_today", 0),
                "errors_429": metrics.get("errors_429", 0), "tokens": metrics.get("tokens", 0),
                "last_event": latest.get("event"), "last_seen": latest.get("timestamp"),
            })
        return {
            "timestamp": iso_now(), "quota_reset_at": reset_next.isoformat(),
            "dashboard_pid": os.getpid(),
            "api_count": self.api_count(), "assignments": assignment,
            "keys": keys, "groups": groups, "events": all_events[-120:],
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "NemotronDashboard/1.0"

    @property
    def data(self) -> DashboardData:
        return self.server.data  # type: ignore[attr-defined]

    def send_json(self, value: object, status: int = 200) -> None:
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("Request body phải là JSON object.")
        return value

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/status":
            self.send_json(self.data.status())
            return
        if path == "/api/review/status":
            try:
                self.send_json(self.data.review_status())
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, 500)
            return
        if path == "/api/review/failures":
            try:
                query = parse_qs(parsed.query)
                reviewer = query.get("reviewer", [None])[0]
                self.send_json({"ok": True, **self.data.review_failure_queue(reviewer)})
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
            return
        if path == "/api/review/batch":
            try:
                query = parse_qs(parsed.query)
                reviewer = query.get("reviewer", ["terra"])[0]
                batch = int(query.get("batch", ["1"])[0])
                self.send_json({"ok": True, **self.data.review_batch_payload(reviewer, batch)})
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
            return
        if path in ("/", "/index.html"):
            raw = self.data.page_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return
        if path in ("/review", "/review.html"):
            raw = self.data.review_page_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path in {"/api/review/validate", "/api/review/merge"}:
            try:
                payload = self.read_json_body()
                reviewer = str(payload.get("reviewer", "terra"))
                batch = int(payload.get("batch", 1))
                output = payload.get("output")
                if not isinstance(output, str):
                    raise ValueError("output phải là chuỗi chứa JSON/JSONL.")
                if path.endswith("/validate"):
                    result = self.data.validate_review_output(reviewer, batch, output)
                else:
                    result = self.data.merge_review_output(reviewer, batch, output, str(payload.get("model") or "external-review-model"))
                self.send_json({"ok": True, **result})
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
            return
        match = re.fullmatch(r"/api/groups/([1-5])/restart", path)
        if match:
            try:
                self.send_json({"ok": True, **self.data.restart_group(int(match.group(1)))})
            except (OSError, ValueError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, 500)
            return
        if path != "/api/assignments":
            self.send_error(404)
            return
        try:
            payload = self.read_json_body()
            self.send_json({"ok": True, "assignments": self.data.save_assignments(payload)})
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"ok": False, "error": str(exc)}, 400)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.data = DashboardData(Path(args.root).resolve())  # type: ignore[attr-defined]
    print(f"Nemotron dashboard: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
