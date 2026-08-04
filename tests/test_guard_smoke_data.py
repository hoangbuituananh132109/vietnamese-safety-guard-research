from __future__ import annotations

import json

from guard_smoke.constants import TEXT_SAFETY_TASK
from guard_smoke.data import materialize_row, write_gliner_jsonl


def base_row() -> dict:
    return {
        "_line_number": 1,
        "record_uid": "row-1",
        "source_id": "source-1",
        "tag": "generic",
        "prompt_en": "A safe prompt",
        "prompt_vi": "Một yêu cầu an toàn",
        "response_en": "An unsafe response",
        "response_vi": "Một phản hồi không an toàn",
        "prompt_label": "safe",
        "response_label": "unsafe",
        "violated_categories": "Violence",
    }


def test_prompt_and_response_targets_never_cross() -> None:
    examples = materialize_row(base_row(), "train")
    assert len(examples) == 6
    for example in examples:
        if example.view == "P":
            assert example.safety_scope == "prompt"
            assert example.safety_label == "safe"
        else:
            assert example.view in {"R", "PR"}
            assert example.safety_scope == "response"
            assert example.safety_label == "unsafe"


def test_empty_response_does_not_become_text_training_example() -> None:
    row = base_row()
    row["response_en"] = ""
    row["response_vi"] = ""
    row["response_label"] = "safe"
    examples = materialize_row(row, "train")
    assert {example.view for example in examples} == {"P"}


def test_response_only_anomaly_still_produces_r() -> None:
    row = base_row()
    row["prompt_en"] = ""
    row["prompt_vi"] = ""
    examples = materialize_row(row, "valid")
    assert {example.view for example in examples} == {"R"}
    assert all(example.safety_scope == "response" for example in examples)


def test_gliner_jsonl_uses_one_binary_text_task_with_correct_source_target(tmp_path) -> None:
    examples = materialize_row(base_row(), "train")
    output = tmp_path / "guard.jsonl"
    assert write_gliner_jsonl(examples, output) == 6

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    for example, row in zip(examples, rows):
        assert len(row["output"]["classifications"]) == 1
        safety = row["output"]["classifications"][0]
        assert safety["task"] == TEXT_SAFETY_TASK
        assert safety["labels"] == ["safe", "unsafe"]
        if example.view == "P":
            assert safety["true_label"] == ["safe"]
        else:
            assert example.view in {"R", "PR"}
            assert safety["true_label"] == ["unsafe"]
