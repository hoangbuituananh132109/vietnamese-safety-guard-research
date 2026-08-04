from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from translator.models import ProviderResult, TranslationRequest, TranslationResponse
from translator.prompts import REPAIR_PROMPT, SYSTEM_PROMPT
from translator.providers.base import TranslationProvider
from translator.jsonl_io import append_jsonl


class ApiKeyPool:
    def __init__(
        self,
        path: str | Path,
        selected_slots: list[int] | None = None,
        reserve_slots: list[int] | None = None,
    ):
        all_keys = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        wanted = selected_slots or list(range(1, len(all_keys) + 1))
        reserves = reserve_slots or []
        all_wanted = wanted + reserves
        if len(all_wanted) != len(set(all_wanted)) or any(slot < 1 or slot > len(all_keys) for slot in all_wanted):
            raise ValueError(f"Invalid API key slots {all_wanted}; file contains {len(all_keys)} non-empty keys")
        self._keys = {slot: all_keys[slot - 1] for slot in all_wanted}
        self._primary_slots = list(wanted)
        self._reserve_slots = list(reserves)
        self._reserve_active = not bool(reserves)
        self._prefer_reserve_once = False
        if not self._keys:
            raise ValueError(f"No API keys found in {path}")
        self._index = 0
        self._lock = threading.Lock()
        self._disabled: set[int] = set()

    @property
    def size(self) -> int:
        return len(self._keys)

    @property
    def reserve_active(self) -> bool:
        return self._reserve_active

    @property
    def reserve_slots(self) -> list[int]:
        return list(self._reserve_slots)

    def activate_reserve(self, prefer_next: bool = True) -> bool:
        with self._lock:
            if not self._reserve_slots:
                return False
            self._reserve_active = True
            self._prefer_reserve_once = prefer_next
            return True

    def next(self) -> tuple[int, str]:
        with self._lock:
            primary = [slot for slot in self._primary_slots if slot not in self._disabled]
            reserve = [slot for slot in self._reserve_slots if slot not in self._disabled]
            if not primary and reserve:
                self._reserve_active = True
                self._prefer_reserve_once = True
            if self._reserve_active and self._prefer_reserve_once and reserve:
                self._prefer_reserve_once = False
                slot = reserve[0]
                return slot, self._keys[slot]
            eligible = primary + (reserve if self._reserve_active else [])
            if eligible:
                slot = eligible[self._index % len(eligible)]
                self._index += 1
                return slot, self._keys[slot]
            raise RuntimeError("All API key slots are disabled")

    def disable(self, one_based_slot: int) -> None:
        with self._lock:
            self._disabled.add(one_based_slot)

    @property
    def active_size(self) -> int:
        eligible = self._primary_slots + (self._reserve_slots if self._reserve_active else [])
        return sum(slot not in self._disabled for slot in eligible)

    @property
    def disabled_size(self) -> int:
        return len(self._disabled)


class GeminiProvider(TranslationProvider):
    name = "gemini"

    def __init__(
        self,
        model: str,
        api_key_file: str | Path,
        key_slots: list[int] | None = None,
        reserve_key_slots: list[int] | None = None,
        min_request_interval_seconds: float = 60.0,
        event_log_path: str | Path | None = None,
    ):
        self.model = model
        self.keys = ApiKeyPool(api_key_file, key_slots, reserve_key_slots)
        self.calls = 0
        self.consecutive_quota_failures = 0
        self.min_request_interval_seconds = min_request_interval_seconds
        self._last_request_started_by_slot: dict[int, float] = {}
        self.event_log_path = Path(event_log_path) if event_log_path else None

    def _emit(self, event: str, slot: int | None, request: TranslationRequest, **extra: Any) -> None:
        if self.event_log_path is None:
            return
        append_jsonl(self.event_log_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event, "key_slot": slot, "batch_id": request.batch_id,
            "model": self.model, **extra,
        }, fsync=False)

    def _throttle(self, slot: int) -> None:
        last_started = self._last_request_started_by_slot.get(slot, 0.0)
        elapsed = time.monotonic() - last_started
        wait = self.min_request_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_started_by_slot[slot] = time.monotonic()

    def translate(self, request: TranslationRequest, repair_errors: list[str] | None = None) -> ProviderResult:
        payload: dict[str, Any] = request.model_dump()
        contents = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if repair_errors:
            contents = REPAIR_PROMPT.format(errors="; ".join(repair_errors)) + "\n\n" + contents
        response = None
        slot = None
        last_error: Exception | None = None
        # A leaked-key 403 is key-specific, not a content/batch failure. Disable
        # that slot for this process and immediately try the next slot.
        for _ in range(self.keys.size):
            slot, key = self.keys.next()
            self._throttle(slot)
            self.calls += 1
            self._emit("request_started", slot, request, call_number=self.calls)
            client = genai.Client(api_key=key)
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_json_schema=TranslationResponse.model_json_schema(),
                        temperature=0.1,
                    ),
                )
                self.consecutive_quota_failures = 0
                usage_obj = getattr(response, "usage_metadata", None)
                usage_preview = usage_obj.model_dump(mode="json") if hasattr(usage_obj, "model_dump") else None
                self._emit("success", slot, request, usage_metadata=usage_preview)
                break
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "reported as leaked" in message or ("403" in message and "permission_denied" in message):
                    self._emit("key_disabled", slot, request, error_type="leaked_or_403", error=str(exc)[:1200])
                    self.keys.disable(slot)
                    continue
                if "429" in message or "resource_exhausted" in message:
                    self.consecutive_quota_failures += 1
                    quota_kind = "rpd_500" if (
                        "generaterequestsperday" in message
                        or "generate_content_free_tier_requests" in message
                        or "limit: 500" in message
                    ) else "transient_429"
                    self._emit("quota_429", slot, request, quota_kind=quota_kind, error=str(exc)[:1600])
                    if self.consecutive_quota_failures >= 3:
                        activated = self.keys.activate_reserve(prefer_next=True)
                        if activated:
                            self._emit("reserve_activated", slot, request, reserve_slots=self.keys.reserve_slots)
                else:
                    self._emit("error", slot, request, error_type=type(exc).__name__, error=str(exc)[:1600])
                raise
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        if response is None:
            raise last_error or RuntimeError("No usable API key slots")
        parsed = response.parsed
        if isinstance(parsed, TranslationResponse):
            validated = parsed
        elif parsed is not None:
            validated = TranslationResponse.model_validate(parsed)
        else:
            validated = TranslationResponse.model_validate_json(response.text)
        usage = None
        if getattr(response, "usage_metadata", None) is not None:
            obj = response.usage_metadata
            usage = obj.model_dump(mode="json") if hasattr(obj, "model_dump") else {"repr": str(obj)}
        return ProviderResult(response=validated, provider=self.name, model=self.model, usage_metadata=usage, key_slot=slot)
