from __future__ import annotations

from translator.models import ProviderResult, TranslationOutputItem, TranslationRequest, TranslationResponse
from translator.providers.base import TranslationProvider


class DryRunProvider(TranslationProvider):
    name = "dry-run"

    def __init__(self, model: str = "dry-run-v1", failure_mode: str | None = None):
        self.model = model
        self.failure_mode = failure_mode
        self.calls = 0

    def translate(self, request: TranslationRequest, repair_errors: list[str] | None = None) -> ProviderResult:
        self.calls += 1
        items = [
            TranslationOutputItem(
                seq=item.seq,
                record_uid=item.record_uid,
                prompt_vi="" if item.prompt == "" else "[BẢN DỊCH THỬ] " + item.prompt,
                response_vi=None if item.response is None else ("" if item.response == "" else "[BẢN DỊCH THỬ] " + item.response),
                warnings=["dry_run_not_a_real_translation"],
            ) for item in request.items
        ]
        if self.failure_mode == "missing" and len(items) > 1 and not repair_errors:
            items.pop()
        if self.failure_mode == "duplicate" and items and not repair_errors:
            items.append(items[0])
        response = TranslationResponse(batch_id=request.batch_id, items=items)
        return ProviderResult(response=response, provider=self.name, model=self.model)
