from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from translator.batching import group_for_batching, make_batches, source_chars
from translator.checkpoint import load_checkpoint, save_completed
from translator.jsonl_io import append_jsonl, read_jsonl, write_jsonl
from translator.models import TranslationInputItem, TranslationRequest
from translator.prompts import TRANSLATION_PROMPT_VERSION
from translator.providers.base import TranslationProvider
from translator.validators import TranslationValidationError, quality_warnings, validate_hard_quality, validate_response


class RequestBudgetExhausted(RuntimeError):
    """Internal control flow: stop cleanly before exceeding the run budget."""


class TranslationRunDeferred(RuntimeError):
    """Infrastructure/quota failure that should be resumed later, not split."""


class BatchValidationExhausted(RuntimeError):
    """Quarantine one original batch after repeated usable-but-invalid replies."""

    def __init__(self, batch_id: str, errors: list[str], attempts: int, candidates: list[dict[str, Any]]):
        self.batch_id = batch_id
        self.errors = errors
        self.attempts = attempts
        self.candidates = candidates
        super().__init__(f"validation rejected {attempts} API responses: {'; '.join(errors)}")


class CandidateValidationError(ValueError):
    """A structured provider reply exists, but downstream validation rejected it."""

    def __init__(self, request: TranslationRequest, result: Any, cause: Exception):
        self.request = request
        self.result = result
        self.cause = cause
        super().__init__(str(cause))


def text_hash(prompt: str | None, response: str | None) -> str:
    return hashlib.sha256(((prompt or "") + "\n" + (response or "")).encode("utf-8")).hexdigest()


def is_transient(exc: Exception) -> bool:
    # Never interpret digits inside record IDs or validation messages as HTTP
    # status codes. Structured-but-rejected candidates belong in the revision
    # queue; they are not infrastructure failures that should stop a worker.
    if isinstance(exc, (CandidateValidationError, TranslationValidationError, ValidationError)):
        return False
    text = str(exc).lower()
    http_status = bool(re.search(r"(?<!\d)(?:429|500|502|503|504)(?!\d)", text))
    return http_status or any(mark in text for mark in (
        "timeout", "timed out",
        "connection reset", "connection error", "name resolution", "dns",
        "temporarily unavailable", "resource_exhausted", "quota",
        "all api key slots are disabled", "no usable api key slots",
    ))


def retry_sleep_seconds(exc: Exception, attempt: int) -> float:
    """Use a real quota-window cooldown for 429, exponential backoff otherwise."""
    text = str(exc).lower()
    if re.search(r"(?<!\d)429(?!\d)", text) or "resource_exhausted" in text or "quota" in text:
        # A 429 gets a slower fixed retry cadence. If all retries still fail,
        # the worker-level cooldown reactivates this group after one minute.
        return 30
    return min(2 ** (attempt + 1), 16)


class TranslationPipeline:
    def __init__(
        self,
        provider: TranslationProvider,
        checkpoint_path: str | Path,
        failed_path: str | Path,
        max_transient_retries: int = 4,
        max_api_requests: int | None = None,
        max_validation_rejections_per_batch: int = 3,
    ):
        self.provider = provider
        self.checkpoint_path = Path(checkpoint_path)
        self.failed_path = Path(failed_path)
        candidate_name = self.failed_path.name.replace("_failed.jsonl", "_candidates.jsonl")
        if candidate_name == self.failed_path.name:
            candidate_name = f"{self.failed_path.stem}_candidates.jsonl"
        self.candidate_path = self.failed_path.with_name(candidate_name)
        self.max_transient_retries = max_transient_retries
        self.max_api_requests = max_api_requests
        self.max_validation_rejections_per_batch = max_validation_rejections_per_batch
        self.stats = {"api_requests": 0, "retries": 0, "batch_splits": 0, "quarantined_batches": 0, "quarantined_records": 0, "failed": 0, "completed": 0, "skipped_resume": 0, "skipped_quarantined": 0}

    def _request(self, rows: list[dict[str, Any]], batch_id: str, repair_errors: list[str] | None = None):
        req = TranslationRequest(
            batch_id=batch_id,
            items=[TranslationInputItem(seq=int(r["_seq"]), record_uid=r["record_uid"], prompt=r["prompt"], response=r.get("response")) for r in rows],
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_transient_retries + 1):
            if self.max_api_requests is not None and self.stats["api_requests"] >= self.max_api_requests:
                raise RequestBudgetExhausted(f"request budget reached: {self.max_api_requests}")
            try:
                self.stats["api_requests"] += 1
                result = self.provider.translate(req, repair_errors=repair_errors)
                try:
                    validate_response(req, result.response)
                    # Semantic translation gates only make sense for a real provider.
                    if result.provider != "dry-run":
                        validate_hard_quality(req, result.response)
                except (TranslationValidationError, ValidationError, ValueError) as exc:
                    raise CandidateValidationError(req, result, exc) from exc
                return req, result
            except Exception as exc:
                last_exc = exc
                if not is_transient(exc) or attempt >= self.max_transient_retries:
                    raise
                self.stats["retries"] += 1
                time.sleep(retry_sleep_seconds(exc, attempt))
        raise last_exc or RuntimeError("unreachable")

    def _complete_row(self, source: dict[str, Any], item: Any, result: Any, batch_id: str, attempt: int) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        translated = {
            "record_uid": source["record_uid"], "prompt_vi": item.prompt_vi, "response_vi": item.response_vi,
            "translation_provider": result.provider, "translation_model": result.model,
            "translation_prompt_version": TRANSLATION_PROMPT_VERSION, "translation_batch_id": batch_id,
            "translation_attempt": attempt, "translation_status": "machine_translated",
            "translation_api_key_slot": result.key_slot,
            "translated_at": now, "source_text_sha256": text_hash(source.get("prompt"), source.get("response")),
            "translation_text_sha256": text_hash(item.prompt_vi, item.response_vi),
            "input_source_chars": source_chars(source),
            "output_translation_chars": len(item.prompt_vi) + len(item.response_vi or ""),
            "warnings": list(item.warnings), "usage_metadata": result.usage_metadata,
        }
        translated["validation_warnings"] = quality_warnings(source, translated)
        return translated

    def _note_validation_rejection(self, context: dict[str, Any], exc: Exception) -> None:
        context["rejections"] += 1
        cause = exc.cause if isinstance(exc, CandidateValidationError) else exc
        errors = cause.errors if isinstance(cause, TranslationValidationError) else [str(cause)]
        context["last_errors"] = [str(error)[:500] for error in errors]
        if isinstance(exc, CandidateValidationError):
            result = exc.result
            context["candidates"].append({
                "candidate_number": context["rejections"],
                "batch_id_requested": exc.request.batch_id,
                "api_key_slot": result.key_slot,
                "provider": result.provider,
                "model": result.model,
                "usage_metadata": result.usage_metadata,
                "validation_errors": context["last_errors"],
                "response": result.response.model_dump(mode="json"),
            })
        if context["rejections"] >= self.max_validation_rejections_per_batch:
            raise BatchValidationExhausted(
                context["root_batch_id"], context["last_errors"], context["rejections"], context["candidates"],
            ) from exc

    def _translate_recursive(
        self,
        rows: list[dict[str, Any]],
        batch_id: str,
        depth: int = 0,
        validation_context: dict[str, Any] | None = None,
    ) -> None:
        context = validation_context or {
            "root_batch_id": batch_id, "rejections": 0, "last_errors": [], "candidates": [],
        }
        try:
            try:
                req, result = self._request(rows, batch_id)
            except (CandidateValidationError, TranslationValidationError, ValidationError, ValueError) as exc:
                self._note_validation_rejection(context, exc)
                cause = exc.cause if isinstance(exc, CandidateValidationError) else exc
                errors = cause.errors if isinstance(cause, TranslationValidationError) else [str(cause)]
                self.stats["retries"] += 1
                try:
                    req, result = self._request(rows, batch_id, repair_errors=errors)
                except (CandidateValidationError, TranslationValidationError, ValidationError, ValueError) as repair_exc:
                    self._note_validation_rejection(context, repair_exc)
                    raise
            by_uid = {item.record_uid: item for item in result.response.items}
            for row in rows:
                completed = self._complete_row(row, by_uid[row["record_uid"]], result, batch_id, depth + 1)
                save_completed(self.checkpoint_path, completed)
                self.stats["completed"] += 1
            return
        except Exception as exc:
            if isinstance(exc, (RequestBudgetExhausted, TranslationRunDeferred, BatchValidationExhausted)):
                raise
            if is_transient(exc):
                raise TranslationRunDeferred(str(exc)) from exc
            if len(rows) > 1:
                self.stats["batch_splits"] += 1
                mid = len(rows) // 2
                self._translate_recursive(rows[:mid], batch_id + "-L", depth + 1, context)
                self._translate_recursive(rows[mid:], batch_id + "-R", depth + 1, context)
                return
            failed = {
                "record_uid": rows[0]["record_uid"], "translation_status": "failed",
                "error_type": type(exc).__name__, "error": str(exc)[:2000],
                "failed_at": datetime.now(timezone.utc).isoformat(), "translation_batch_id": batch_id,
            }
            append_jsonl(self.failed_path, failed)
            self.stats["failed"] += 1

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        limit: int | None = None,
        resume: bool = True,
        shard_index: int = 0,
        shard_count: int = 1,
        batch_id_prefix: str = "",
    ) -> dict[str, Any]:
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError(f"invalid shard {shard_index}/{shard_count}")
        source_rows = [
            row for source_index, (_, row, _) in enumerate(read_jsonl(input_path))
            if source_index % shard_count == shard_index
        ]
        if limit is not None:
            source_rows = source_rows[:limit]
        for seq, row in enumerate(source_rows, 1):
            row["_seq"] = seq
        completed = load_checkpoint(self.checkpoint_path) if resume else {}
        quarantined: set[str] = set()
        if resume and self.failed_path.exists():
            for _, failed_row, _ in read_jsonl(self.failed_path):
                if failed_row.get("translation_status") in {"quarantined_validation", "needs_revision"}:
                    uid = failed_row.get("record_uid")
                    if isinstance(uid, str):
                        quarantined.add(uid)
        remaining = [
            r for r in source_rows
            if r["record_uid"] not in completed and r["record_uid"] not in quarantined
        ]
        self.stats["skipped_resume"] = len(source_rows) - len(remaining)
        self.stats["skipped_quarantined"] = sum(
            r["record_uid"] in quarantined and r["record_uid"] not in completed for r in source_rows
        )
        batches = make_batches(group_for_batching(remaining))
        deferred_reason = None
        for batch in batches:
            first, last = batch[0]["_seq"], batch[-1]["_seq"]
            try:
                self._translate_recursive(batch, f"{batch_id_prefix}full-{first:06d}-{last:06d}")
            except BatchValidationExhausted as exc:
                self.stats["quarantined_batches"] += 1
                self.stats["quarantined_records"] += len(batch)
                archive_id = f"{exc.batch_id}@{datetime.now(timezone.utc).isoformat()}"
                append_jsonl(self.candidate_path, {
                    "candidate_archive_id": archive_id,
                    "translation_status": "needs_revision",
                    "translation_batch_id": exc.batch_id,
                    "source_record_uids": [row["record_uid"] for row in batch],
                    "candidate_count": len(exc.candidates),
                    "candidates": exc.candidates,
                    "archived_at": datetime.now(timezone.utc).isoformat(),
                })
                for row in batch:
                    append_jsonl(self.failed_path, {
                        "record_uid": row["record_uid"],
                        "translation_status": "needs_revision",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:2000],
                        "validation_rejections": exc.attempts,
                        "candidate_archive_id": archive_id,
                        "candidate_archive_file": str(self.candidate_path),
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "translation_batch_id": exc.batch_id,
                    })
                    self.stats["failed"] += 1
                continue
            except (RequestBudgetExhausted, TranslationRunDeferred) as exc:
                deferred_reason = str(exc)[:1000]
                break
        completed = load_checkpoint(self.checkpoint_path)
        final_rows = []
        for source in source_rows:
            source.pop("_seq", None)
            if source["record_uid"] not in completed:
                continue
            final = dict(source)
            final.update({"prompt_en": source.get("prompt"), "response_en": source.get("response"), "translation_language": "vi"})
            final.update(completed[source["record_uid"]])
            final_rows.append(final)
        write_jsonl(output_path, final_rows)
        self.stats.update({
            "source_records": len(source_rows), "output_records": len(final_rows),
            "batches_initial": len(batches), "deferred": deferred_reason is not None,
            "deferred_reason": deferred_reason,
            "shard_index": shard_index, "shard_count": shard_count,
        })
        return self.stats
