from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable


TRUNCATION_MARKER = "[TRUNCATED]"


@dataclass(frozen=True)
class TruncationAudit:
    was_truncated: bool
    strategy: str
    original_chars: int
    final_chars: int
    original_tokens: int
    final_tokens: int
    max_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def _split_prefixed(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def _take_head_tail(value: str, keep: int, head_ratio: float = 0.6) -> str:
    if keep >= len(value):
        return value
    if keep <= 0:
        return ""
    head = min(len(value), max(1, int(round(keep * head_ratio))))
    tail = max(0, keep - head)
    if tail == 0:
        return value[:head]
    return value[:head] + f"\n{TRUNCATION_MARKER}\n" + value[-tail:]


def _candidate(text: str, view: str, keep: int) -> str:
    """Create a scope-aware candidate while retaining the target-bearing region."""

    if view == "P":
        payload = _split_prefixed(text, "Prompt: ")
        return "Prompt: " + _take_head_tail(payload, keep)
    if view == "R":
        payload = _split_prefixed(text, "Response: ")
        return "Response: " + _take_head_tail(payload, keep)
    if view == "PR" and text.startswith("Prompt: ") and "\nResponse: " in text:
        prompt, response = text[len("Prompt: ") :].split("\nResponse: ", 1)
        # The gold target is response safety. Keep prompt context, but devote
        # most of the remaining budget to both ends of the response.
        prompt_keep = min(len(prompt), int(round(keep * 0.25)))
        response_keep = min(len(response), max(0, keep - prompt_keep))
        prompt_part = prompt[:prompt_keep]
        if prompt_keep < len(prompt):
            prompt_part += f"\n{TRUNCATION_MARKER}"
        response_part = _take_head_tail(response, response_keep, head_ratio=0.6)
        return f"Prompt: {prompt_part}\nResponse: {response_part}"
    return _take_head_tail(text, keep, head_ratio=0.5)


def truncate_text_exact(
    text: str,
    view: str,
    token_length: Callable[[str], int],
    max_tokens: int,
) -> tuple[str, TruncationAudit]:
    """Fit text using exact final-sequence token counts and preserve an audit trail.

    ``token_length`` must include every model prefix/special/schema token. The
    returned string is therefore safe to send unchanged to a second comparator
    model. Character boundaries are used only to construct candidates; the
    acceptance decision is always made with exact tokenizer output.
    """

    original_tokens = int(token_length(text))
    if original_tokens <= max_tokens:
        return text, TruncationAudit(
            was_truncated=False,
            strategy="none",
            original_chars=len(text),
            final_chars=len(text),
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            max_tokens=max_tokens,
        )

    low, high = 0, len(text)
    best_text: str | None = None
    best_tokens: int | None = None
    while low <= high:
        keep = (low + high) // 2
        candidate = _candidate(text, view, keep)
        length = int(token_length(candidate))
        if length <= max_tokens:
            best_text, best_tokens = candidate, length
            low = keep + 1
        else:
            high = keep - 1

    if best_text is None or best_tokens is None:
        minimal = _candidate(text, view, 0)
        minimal_tokens = int(token_length(minimal))
        raise ValueError(
            "Schema and structural prefixes alone exceed the context limit: "
            f"view={view} tokens={minimal_tokens} max={max_tokens}"
        )
    if best_tokens > max_tokens:
        raise AssertionError("Exact truncation returned an over-limit candidate")
    return best_text, TruncationAudit(
        was_truncated=True,
        strategy=(
            "prompt_context_plus_response_head_tail"
            if view == "PR"
            else "field_head_tail"
        ),
        original_chars=len(text),
        final_chars=len(best_text),
        original_tokens=original_tokens,
        final_tokens=best_tokens,
        max_tokens=max_tokens,
    )
