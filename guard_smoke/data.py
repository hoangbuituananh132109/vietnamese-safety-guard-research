"""Manifest construction with strict prompt/response target separation."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .constants import (
    N23_CATEGORIES,
    SAFETY_LABELS,
    TEXT_SAFETY_TASK,
    VIEW_TO_SCOPE,
    parse_categories,
)


FINAL_FILENAMES = {
    "train": "nemotron_train_en_vi_v10_final.jsonl",
    "valid": "nemotron_valid_en_vi_v10_final.jsonl",
    "test": "nemotron_test_en_vi_v10_final.jsonl",
}


@dataclass(frozen=True)
class GuardExample:
    example_id: str
    record_uid: str
    source_split: str
    source_id: str
    tag: str
    language: str
    view: str
    safety_scope: str
    text: str
    safety_label: str
    categories: tuple[str, ...]
    category_scope: str
    normalized_prompt_sha256: str

    def validate(self) -> None:
        expected_scope = VIEW_TO_SCOPE.get(self.view)
        if expected_scope is None:
            raise ValueError(f"Unsupported view: {self.view}")
        if self.safety_scope != expected_scope:
            raise ValueError(
                f"{self.view} must predict {expected_scope}_label, got "
                f"{self.safety_scope}_label for {self.example_id}"
            )
        if self.safety_label not in SAFETY_LABELS:
            raise ValueError(f"Invalid safety label: {self.safety_label}")
        unknown = sorted(set(self.categories).difference(N23_CATEGORIES))
        if unknown:
            raise ValueError(f"Unknown categories in {self.example_id}: {unknown}")
        if not self.text.strip():
            raise ValueError(f"Empty serialized text: {self.example_id}")


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(_normalized_text(text).encode("utf-8")).hexdigest()


def _is_usable_prompt(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text) and text != "REDACTED"


def _is_nonempty(value: object) -> bool:
    return value is not None and bool(str(value).strip())


def _serialize(view: str, prompt: str, response: str) -> str:
    if view == "P":
        return f"Prompt: {prompt}"
    if view == "R":
        return f"Response: {response}"
    if view == "PR":
        return f"Prompt: {prompt}\nResponse: {response}"
    raise ValueError(f"Unsupported view: {view}")


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            yield row


def materialize_row(row: dict, split: str) -> list[GuardExample]:
    """Create P/R/PR views without ever crossing prompt/response labels."""

    prompt_en = str(row.get("prompt_en") or "")
    prompt_vi = str(row.get("prompt_vi") or "")
    response_en = row.get("response_en")
    response_vi = row.get("response_vi")
    prompt_ok = _is_usable_prompt(prompt_en) and _is_usable_prompt(prompt_vi)
    response_ok = _is_nonempty(response_en) and _is_nonempty(response_vi)
    prompt_label = str(row.get("prompt_label") or "").strip()
    response_label = str(row.get("response_label") or "").strip()
    categories = parse_categories(row.get("violated_categories"))
    record_uid = str(row.get("record_uid") or f"{split}-{row['_line_number']:08d}")
    source_id = str(row.get("source_id") or row.get("id") or record_uid)
    tag = str(row.get("tag") or "unknown")
    prompt_hash = _sha256_text(prompt_en) if prompt_ok else ""
    examples: list[GuardExample] = []

    for language, prompt, response in (
        ("en", prompt_en, str(response_en or "")),
        ("vi", prompt_vi, str(response_vi or "")),
    ):
        if prompt_ok and prompt_label in SAFETY_LABELS:
            example = GuardExample(
                example_id=f"{record_uid}:P:{language}",
                record_uid=record_uid,
                source_split=split,
                source_id=source_id,
                tag=tag,
                language=language,
                view="P",
                safety_scope="prompt",
                text=_serialize("P", prompt, ""),
                safety_label=prompt_label,
                # N23 is unambiguously prompt-level only on raw prompt-only rows.
                categories=categories if row.get("response_en") is None else (),
                category_scope="prompt" if row.get("response_en") is None else "unavailable",
                normalized_prompt_sha256=prompt_hash,
            )
            example.validate()
            examples.append(example)

        if response_ok and response_label in SAFETY_LABELS:
            response_only = GuardExample(
                example_id=f"{record_uid}:R:{language}",
                record_uid=record_uid,
                source_split=split,
                source_id=source_id,
                tag=tag,
                language=language,
                view="R",
                safety_scope="response",
                text=_serialize("R", "", response),
                safety_label=response_label,
                # The dataset does not attribute N23 categories to response alone.
                categories=(),
                category_scope="unavailable",
                normalized_prompt_sha256=prompt_hash,
            )
            response_only.validate()
            examples.append(response_only)

            if prompt_ok:
                contextual = GuardExample(
                    example_id=f"{record_uid}:PR:{language}",
                    record_uid=record_uid,
                    source_split=split,
                    source_id=source_id,
                    tag=tag,
                    language=language,
                    view="PR",
                    safety_scope="response",
                    text=_serialize("PR", prompt, response),
                    safety_label=response_label,
                    categories=categories,
                    category_scope="interaction",
                    normalized_prompt_sha256=prompt_hash,
                )
                contextual.validate()
                examples.append(contextual)

    return examples


def _dedupe_prompt_views(examples: Iterable[GuardExample]) -> list[GuardExample]:
    """Keep one P target per normalized prompt and language within a split."""

    seen: set[tuple[str, str, str]] = set()
    output: list[GuardExample] = []
    for example in examples:
        if example.view != "P":
            output.append(example)
            continue
        key = (
            example.source_split,
            example.language,
            example.normalized_prompt_sha256,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(example)
    return output


def iter_full_examples(
    final_dir: Path,
    split: str,
    *,
    languages: set[str] | None = None,
    views: set[str] | None = None,
    tags: set[str] | None = None,
) -> Iterator[GuardExample]:
    """Stream a complete split with deterministic prompt-view deduplication.

    Response views are never deduplicated because two equal response strings can
    belong to different source interactions. Prompt views use the same
    split/language/normalized-prompt key as the smoke manifest.
    """

    if split not in FINAL_FILENAMES:
        raise ValueError(f"Unknown split: {split}")
    source = final_dir / FINAL_FILENAMES[split]
    seen_prompts: set[tuple[str, str, str]] = set()
    for row in iter_jsonl(source):
        for example in materialize_row(row, split):
            if languages is not None and example.language not in languages:
                continue
            if views is not None and example.view not in views:
                continue
            if tags is not None and example.tag not in tags:
                continue
            if example.view == "P":
                key = (
                    example.source_split,
                    example.language,
                    example.normalized_prompt_sha256,
                )
                if key in seen_prompts:
                    continue
                seen_prompts.add(key)
            yield example


def build_balanced_smoke_examples(
    final_dir: Path,
    split: str,
    semantic_rows_per_cell: int,
    seed: int = 3407,
) -> list[GuardExample]:
    """Sample semantic rows by view/label/tag, then retain both EN and VI."""

    if split not in FINAL_FILENAMES:
        raise ValueError(f"Unknown split: {split}")
    source = final_dir / FINAL_FILENAMES[split]
    all_examples: list[GuardExample] = []
    for row in iter_jsonl(source):
        all_examples.extend(materialize_row(row, split))
    all_examples = _dedupe_prompt_views(all_examples)

    # Group EN/VI by semantic record and sample records, not language instances.
    by_cell: dict[tuple[str, str, str], dict[str, list[GuardExample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for example in all_examples:
        cell = (example.view, example.safety_label, example.tag)
        semantic_key = f"{example.record_uid}:{example.view}"
        by_cell[cell][semantic_key].append(example)

    rng = random.Random(seed)
    selected: list[GuardExample] = []
    for cell in sorted(by_cell):
        semantic_groups = list(by_cell[cell].values())
        rng.shuffle(semantic_groups)
        picked = 0
        for group in semantic_groups:
            languages = {item.language for item in group}
            if languages != {"en", "vi"}:
                continue
            selected.extend(sorted(group, key=lambda item: item.language))
            picked += 1
            if picked >= semantic_rows_per_cell:
                break
        if picked < semantic_rows_per_cell:
            raise ValueError(
                f"Cell {cell} only has {picked} paired semantic rows; "
                f"requested {semantic_rows_per_cell}"
            )

    selected.sort(key=lambda item: item.example_id)
    for example in selected:
        example.validate()
    return selected


def write_manifest(examples: Iterable[GuardExample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            example.validate()
            payload = asdict(example)
            payload["categories"] = list(example.categories)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_manifest(path: Path) -> list[GuardExample]:
    examples: list[GuardExample] = []
    for row in iter_jsonl(path):
        row.pop("_line_number", None)
        row["categories"] = tuple(row.get("categories") or ())
        example = GuardExample(**row)
        example.validate()
        examples.append(example)
    return examples


def gliner_payload(example: GuardExample) -> dict:
    """Create one Phase-0 GLiNER2 record for a shared binary text task.

    ``safety_scope`` remains metadata in ``GuardExample`` so P/R/PR target
    provenance can be audited, but it never changes the task name or head.
    N23 is intentionally deferred until the binary pipeline is complete.
    """

    example.validate()
    return {
        "input": example.text,
        "output": {
            "classifications": [
                {
                    "task": TEXT_SAFETY_TASK,
                    "labels": list(SAFETY_LABELS),
                    "true_label": [example.safety_label],
                }
            ]
        },
        "metadata": {
            "example_id": example.example_id,
            "record_uid": example.record_uid,
            "view": example.view,
            "language": example.language,
        },
    }


def write_gliner_jsonl(examples: Iterable[GuardExample], path: Path) -> int:
    """Write official GLiNER2 classification JSONL with independent tasks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(gliner_payload(example), ensure_ascii=False) + "\n")
            count += 1
    return count
