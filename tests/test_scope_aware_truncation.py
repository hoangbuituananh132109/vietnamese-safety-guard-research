from guard_train.truncation import TRUNCATION_MARKER, truncate_text_exact


def _characters(value: str) -> int:
    return len(value)


def test_short_text_is_unchanged() -> None:
    text, audit = truncate_text_exact("Prompt: hello", "P", _characters, 100)
    assert text == "Prompt: hello"
    assert not audit.was_truncated


def test_prompt_response_truncation_retains_response_and_limit() -> None:
    original = "Prompt: " + "p" * 100 + "\nResponse: " + "r" * 200
    text, audit = truncate_text_exact(original, "PR", _characters, 100)
    assert text.startswith("Prompt: ")
    assert "\nResponse: " in text
    assert text.endswith("r" * 10)
    assert TRUNCATION_MARKER in text
    assert len(text) <= 100
    assert audit.was_truncated
    assert audit.final_tokens <= 100


def test_response_head_and_tail_survive() -> None:
    original = "Response: BEGIN-" + "x" * 200 + "-END"
    text, audit = truncate_text_exact(original, "R", _characters, 80)
    assert "BEGIN" in text
    assert text.endswith("-END")
    assert audit.strategy == "field_head_tail"
