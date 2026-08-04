from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from guard_smoke.data import load_manifest
from guard_smoke.mmbert_schema import (
    BinarySchemaCodec,
    BinarySchemaCollator,
    CANONICAL_LABELS,
    SCHEMA_SPECIAL_TOKENS,
    TASK_NAME,
    load_mmbert_tokenizer,
)


MODEL_PATH = Path("models/mmbert_small_base_smoke")
MANIFEST = Path("data/guard_smoke/train.jsonl")


def test_phase0_schema_and_label_alignment() -> None:
    tokenizer = load_mmbert_tokenizer(str(MODEL_PATH))
    codec = BinarySchemaCodec(tokenizer, max_length=8192)
    example = load_manifest(MANIFEST)[0]

    canonical = codec.encode(example, CANONICAL_LABELS)
    reversed_order = codec.encode(example, tuple(reversed(CANONICAL_LABELS)))

    assert TASK_NAME == "text safety classification"
    assert len(set(codec.marker_token_ids)) == len(SCHEMA_SPECIAL_TOKENS) == 3
    assert canonical.label_order == ("safe", "unsafe")
    assert reversed_order.label_order == ("unsafe", "safe")
    assert canonical.target_index != reversed_order.target_index
    assert [canonical.input_ids[index] for index in canonical.label_positions] == [
        codec.label_marker_id,
        codec.label_marker_id,
    ]


def test_collator_does_not_route_scope_or_view() -> None:
    tokenizer = load_mmbert_tokenizer(str(MODEL_PATH))
    codec = BinarySchemaCodec(tokenizer, max_length=8192)
    examples = load_manifest(MANIFEST)[:2]
    batch = BinarySchemaCollator(codec, shuffle_labels=False, seed=17)(examples)

    assert batch["label_positions"].shape == (2, 2)
    assert batch["targets"].shape == (2,)
    assert "scope_ids" not in batch
    assert "view_ids" not in batch
    assert "category_targets" not in batch
