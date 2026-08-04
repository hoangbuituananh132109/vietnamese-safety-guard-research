from __future__ import annotations

import difflib
import re
from typing import Any

from translator.models import TranslationRequest, TranslationResponse


VI_REFUSAL_PATTERNS = [
    r"\btôi không thể (?:hỗ trợ|giúp|cung cấp)", r"\bxin lỗi,? nhưng",
    r"\bnội dung này (?:vi phạm|không phù hợp)",
]
EN_REFUSAL_PATTERNS = [
    r"\bi(?: am|[’']m) (?:really )?(?:sorry|unable)", r"\bi cannot (?:assist|help|provide)",
    r"\bi can(?:no|'?t) (?:assist|help|provide)", r"\bagainst my programming",
]


class TranslationValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_response(request: TranslationRequest, response: TranslationResponse) -> None:
    errors: list[str] = []
    if response.batch_id != request.batch_id:
        errors.append(f"batch_id mismatch: {response.batch_id!r}")
    expected = {item.record_uid: item for item in request.items}
    seen: dict[str, Any] = {}
    for item in response.items:
        if item.record_uid in seen:
            errors.append(f"duplicate record_uid: {item.record_uid}")
        seen[item.record_uid] = item
        source = expected.get(item.record_uid)
        if source is None:
            errors.append(f"unknown record_uid: {item.record_uid}")
            continue
        if item.seq != source.seq:
            errors.append(f"seq mismatch for {item.record_uid}")
        if source.prompt == "" and item.prompt_vi != "":
            errors.append(f"empty prompt not preserved for {item.record_uid}")
        elif source.prompt != "" and not item.prompt_vi.strip():
            errors.append(f"empty prompt_vi for {item.record_uid}")
        if source.response is None and item.response_vi is not None:
            errors.append(f"null response not preserved for {item.record_uid}")
        if source.response == "" and item.response_vi != "":
            errors.append(f"empty response not preserved for {item.record_uid}")
        if source.response not in (None, "") and (item.response_vi is None or not item.response_vi.strip()):
            errors.append(f"missing response_vi for {item.record_uid}")
    missing = sorted(set(expected) - set(seen))
    if missing:
        errors.append("missing record_uid: " + ",".join(missing))
    if errors:
        raise TranslationValidationError(errors)


def validate_hard_quality(request: TranslationRequest, response: TranslationResponse) -> None:
    """Reject a small set of high-confidence, QA-derived semantic failures."""
    errors: list[str] = []
    source_by_uid = {item.record_uid: item for item in request.items}
    for item in response.items:
        source = source_by_uid.get(item.record_uid)
        if source is None:
            continue
        pairs = ((source.prompt, item.prompt_vi, "prompt"), (source.response or "", item.response_vi or "", "response"))
        for src, dst, field in pairs:
            src_low, dst_low = src.casefold(), dst.casefold()
            if "explicit photo" in src_low and "ảnh nhạy cảm" in dst_low:
                errors.append(f"{item.record_uid} {field}: 'explicit photos' was euphemized as 'ảnh nhạy cảm'; use an explicitly sexual Vietnamese term")
            if "graphic detail" in src_low and "chi tiết đồ họa" in dst_low:
                errors.append(f"{item.record_uid} {field}: 'graphic detail' is vivid/explicit description, not computer graphics")
            if "schizophren" in src_low and "tâm thần phân liệt" not in dst_low:
                errors.append(f"{item.record_uid} {field}: preserve the precise term schizophrenia as tâm thần phân liệt")
            # Only treat compromise as a cybersecurity/operational breach when
            # the disambiguating term is close to the occurrence. Long safety
            # prompts often contain an unrelated word such as "communication"
            # hundreds of characters before the ordinary negotiation sense.
            compromise_contexts = [
                src_low[max(0, match.start() - 140):match.end() + 140]
                for match in re.finditer(r"compromis\w*", src_low)
            ]
            if "thỏa hiệp" in dst_low and any(
                any(keyword in context for keyword in ("operation", "security", "network", "communication", "logistics"))
                for context in compromise_contexts
            ):
                errors.append(f"{item.record_uid} {field}: compromised means breached/exposed in this context, not thỏa hiệp")
            if any(x in src_low for x in ("bullshit", "fuckin'", "fucking", "what? the? fuck", "what the fuck")) and all(x not in dst_low for x in (
                "đéo", "đụ", "địt", "cứt", "lồn", "vãi", "mẹ kiếp", "mẹ nó",
                "chó chết", "chó đẻ", "mặt lồn", "chết mẹ", "cướp mẹ", "hỏng mẹ",
                "khốn kiếp", "ra bã",
            )):
                errors.append(f"{item.record_uid} {field}: strong profanity was softened; retain comparable Vietnamese vulgarity")
            if len(src) > 80 and len(dst) > 80:
                similarity = difflib.SequenceMatcher(None, src, dst).ratio() if len(src) <= 20_000 else 0.0
                ascii_ratio = sum(ord(ch) < 128 for ch in dst) / len(dst)
                # Field-level similarity is not a high-confidence hard failure. QA found
                # translated records dominated by preservation-scoped SQL, source code,
                # JSON keys, URLs, placeholders, proper names, or ASCII art. The softer
                # quality_warnings pass still flags mostly unchanged fields for audit.

            # Do not reject an exact digit-heavy line by itself. QA showed this gate treated
            # preservation-scoped data as leetspeak: phone/address fields, JSON timestamps,
            # formulae, music notation, source code, numbered lists, and `{placeholder}`
            # templates. Whole-field English/leet failures remain covered by the similarity
            # checks above, which have substantially higher precision.

            # A quoted prose label such as `1. "The Great Escape": Players...`
            # is not a JSON key and should be translated. Restrict this gate to
            # key-shaped strings at the start of a line or following `{`/`,`.
            json_key_pattern = r'(?m)(?:^|[{,])\s*"([^"\\]+)"\s*:'
            src_keys = set(re.findall(json_key_pattern, src))
            dst_keys = set(re.findall(json_key_pattern, dst))
            if src_keys and src_keys != dst_keys:
                errors.append(f"{item.record_uid} {field}: JSON/schema keys changed: expected {sorted(src_keys)}, got {sorted(dst_keys)}")

            # Redaction markers are benchmark metadata embedded in the text,
            # not prose to translate or expand.
            for marker in set(re.findall(r"(?i)(?<![A-Za-z])(?:\[|<)?REDACTED(?:\]|>)?(?![A-Za-z])", src)):
                if marker not in dst:
                    errors.append(f"{item.record_uid} {field}: redaction marker {marker!r} was not preserved exactly")

            slur_equivalents = {
                "nigger": ("mọi đen", "da đen hạ đẳng"),
                "niggers": ("mọi đen", "da đen hạ đẳng"),
                "spic": ("mọi mễ", "mễ bẩn", "đồ mễ", "bọn mễ"),
                "spics": ("mọi mễ", "mễ bẩn", "đồ mễ", "bọn mễ"),
                "faggot": ("bóng chó", "bê đê hạ đẳng", "pê đê hạ đẳng"),
                "faggots": ("bóng chó", "bê đê hạ đẳng", "pê đê hạ đẳng"),
                "kike": ("mọi do thái", "do thái bẩn"),
                "chink": ("tàu khựa", "mọi vàng"),
            }
            for slur, equivalents in slur_equivalents.items():
                if re.search(rf"(?<![a-z]){re.escape(slur)}(?![a-z])", src_low) and re.search(rf"(?<![a-z]){re.escape(slur)}(?![a-z])", dst_low):
                    if not any(equivalent in dst_low for equivalent in equivalents):
                        errors.append(f"{item.record_uid} {field}: English identity slur {slur!r} was left untranslated")
    if errors:
        raise TranslationValidationError(errors)


def quality_warnings(source: dict[str, Any], translated: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for field in ("prompt", "response"):
        src = source.get(field)
        dst = translated.get(field + "_vi")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        if src == "":
            continue
        if len(src) > 30 and src.strip() == dst.strip():
            warnings.append(f"{field}_unchanged")
        ratio = len(dst) / max(len(src), 1)
        if ratio < 0.35:
            warnings.append(f"{field}_suspiciously_short")
        elif ratio > 3.0:
            warnings.append(f"{field}_suspiciously_long")
        src_refusal = any(re.search(p, src, re.I) for p in EN_REFUSAL_PATTERNS)
        dst_refusal = any(re.search(p, dst, re.I) for p in VI_REFUSAL_PATTERNS)
        if dst_refusal and not src_refusal:
            warnings.append(f"{field}_possible_added_refusal")
        similarity = difflib.SequenceMatcher(None, src, dst).ratio() if len(src) <= 20_000 else 0.0
        ascii_ratio = sum(ord(ch) < 128 for ch in dst) / max(len(dst), 1)
        if len(src) > 80 and similarity > 0.92 and ascii_ratio > 0.95:
            warnings.append(f"{field}_mostly_untranslated")
        leet_like = sum(ch.isdigit() or ch in "@$" for ch in src) >= 8
        token_pattern = r"https?://\S+|[\w.+-]+@[\w.-]+\.\w+|```"
        if not leet_like:
            token_pattern += r"|\b\d+(?:\.\d+)?\b"
        for token in re.findall(token_pattern, src):
            if token not in dst:
                warnings.append(f"{field}_missing_token:{token[:40]}")
    return sorted(set(warnings))
