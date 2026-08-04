from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TranslationInputItem(BaseModel):
    seq: int
    record_uid: str
    prompt: str
    response: str | None


class TranslationRequest(BaseModel):
    batch_id: str
    items: list[TranslationInputItem]


class TranslationOutputItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seq: int
    record_uid: str
    prompt_vi: str
    response_vi: str | None
    warnings: list[str] = Field(default_factory=list)


class TranslationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_id: str
    items: list[TranslationOutputItem]


class ProviderResult(BaseModel):
    response: TranslationResponse
    provider: str
    model: str
    usage_metadata: dict[str, Any] | None = None
    key_slot: int | None = None

