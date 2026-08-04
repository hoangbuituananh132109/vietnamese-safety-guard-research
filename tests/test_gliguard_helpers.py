from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from guard_smoke.gliguard import _parse_binary_result
from guard_smoke.constants import TEXT_SAFETY_TASK


def test_parse_binary_result_preserves_unsafe_probability() -> None:
    pred, probability, label, confidence = _parse_binary_result(
        {TEXT_SAFETY_TASK: {"label": "unsafe", "confidence": 0.8}},
        TEXT_SAFETY_TASK,
    )
    assert pred == 1
    assert probability == pytest.approx(0.8)
    assert label == "unsafe"
    assert confidence == pytest.approx(0.8)


def test_parse_safe_result_complements_probability() -> None:
    pred, probability, _, _ = _parse_binary_result(
        {TEXT_SAFETY_TASK: {"label": "safe", "confidence": 0.9}},
        TEXT_SAFETY_TASK,
    )
    assert pred == 0
    assert probability == pytest.approx(0.1)


def test_parse_result_rejects_missing_text_safety_task() -> None:
    with pytest.raises(KeyError):
        _parse_binary_result(
            {"prompt_safety": {"label": "safe", "confidence": 0.9}},
            TEXT_SAFETY_TASK,
        )
