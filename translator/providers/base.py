from __future__ import annotations

from abc import ABC, abstractmethod

from translator.models import ProviderResult, TranslationRequest


class TranslationProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def translate(self, request: TranslationRequest, repair_errors: list[str] | None = None) -> ProviderResult:
        raise NotImplementedError

