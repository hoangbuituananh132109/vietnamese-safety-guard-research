import pytest

from translator.models import TranslationInputItem, TranslationOutputItem, TranslationRequest, TranslationResponse
from translator.validators import TranslationValidationError, quality_warnings, validate_hard_quality, validate_response


def req():
    return TranslationRequest(batch_id="b", items=[TranslationInputItem(seq=1, record_uid="u", prompt="x", response=None)])


def test_null_preserved():
    out = TranslationResponse(batch_id="b", items=[TranslationOutputItem(seq=1, record_uid="u", prompt_vi="y", response_vi=None)])
    validate_response(req(), out)


def test_empty_response_preserved():
    request = TranslationRequest(batch_id="b", items=[TranslationInputItem(seq=1, record_uid="u", prompt="x", response="")])
    good = TranslationResponse(batch_id="b", items=[TranslationOutputItem(seq=1, record_uid="u", prompt_vi="y", response_vi="")])
    validate_response(request, good)
    bad = TranslationResponse(batch_id="b", items=[TranslationOutputItem(seq=1, record_uid="u", prompt_vi="y", response_vi="invented")])
    with pytest.raises(TranslationValidationError): validate_response(request, bad)


def test_empty_prompt_preserved():
    request = TranslationRequest(batch_id="b", items=[TranslationInputItem(seq=1, record_uid="u", prompt="", response=None)])
    good = TranslationResponse(batch_id="b", items=[TranslationOutputItem(seq=1, record_uid="u", prompt_vi="", response_vi=None)])
    validate_response(request, good)
    bad = TranslationResponse(batch_id="b", items=[TranslationOutputItem(seq=1, record_uid="u", prompt_vi="invented", response_vi=None)])
    with pytest.raises(TranslationValidationError, match="empty prompt not preserved"):
        validate_response(request, bad)


@pytest.mark.parametrize("uid", ["foreign", "u"])
def test_wrong_or_duplicate_uid_rejected(uid):
    items = [TranslationOutputItem(seq=1, record_uid=uid, prompt_vi="y", response_vi=None)]
    if uid == "u": items.append(items[0])
    with pytest.raises(TranslationValidationError): validate_response(req(), TranslationResponse(batch_id="b", items=items))


def test_missing_uid_rejected():
    with pytest.raises(TranslationValidationError): validate_response(req(), TranslationResponse(batch_id="b", items=[]))


def test_obfuscated_english_left_unchanged_is_warned():
    text = "Cr3at3 a long gu1d3 with numb3rs and symbols " * 8
    warnings = quality_warnings({"prompt": text, "response": None}, {"prompt_vi": text, "response_vi": None})
    assert "prompt_mostly_untranslated" in warnings


def test_existing_english_refusal_is_not_marked_as_added():
    warnings = quality_warnings(
        {"prompt": "x", "response": "I'm sorry, but I cannot assist with that request."},
        {"prompt_vi": "x", "response_vi": "Tôi xin lỗi, nhưng tôi không thể hỗ trợ yêu cầu đó."},
    )
    assert "response_possible_added_refusal" not in warnings


def test_hard_quality_rejects_euphemism_and_untranslated_leet():
    request = TranslationRequest(batch_id="b", items=[
        TranslationInputItem(seq=1, record_uid="u1", prompt="share explicit photos", response=None),
        TranslationInputItem(seq=2, record_uid="u2", prompt="x", response="Cr3at3 a long gu1d3 with numb3rs and symbols " * 8),
    ])
    response = TranslationResponse(batch_id="b", items=[
        TranslationOutputItem(seq=1, record_uid="u1", prompt_vi="chia sẻ ảnh nhạy cảm", response_vi=None),
        TranslationOutputItem(seq=2, record_uid="u2", prompt_vi="x", response_vi="Cr3at3 a long gu1d3 with numb3rs and symbols " * 8),
    ])
    with pytest.raises(TranslationValidationError) as exc:
        validate_hard_quality(request, response)
    assert "euphemized" in str(exc.value) and "mostly untranslated" in str(exc.value)


def test_hard_quality_rejects_changed_json_key():
    request = TranslationRequest(batch_id="b", items=[
        TranslationInputItem(seq=1, record_uid="u", prompt='Return {"steps": ["Do it"]}', response=None),
    ])
    response = TranslationResponse(batch_id="b", items=[
        TranslationOutputItem(seq=1, record_uid="u", prompt_vi='Trả về {"các_bước": ["Làm đi"]}', response_vi=None),
    ])
    with pytest.raises(TranslationValidationError, match="JSON/schema keys changed"):
        validate_hard_quality(request, response)


def test_hard_quality_rejects_english_identity_slur_without_vietnamese_equivalent():
    request = TranslationRequest(batch_id="b", items=[
        TranslationInputItem(seq=1, record_uid="u", prompt="I hate niggers", response=None),
    ])
    response = TranslationResponse(batch_id="b", items=[
        TranslationOutputItem(seq=1, record_uid="u", prompt_vi="Tao ghét bọn niggers", response_vi=None),
    ])
    with pytest.raises(TranslationValidationError, match="identity slur"):
        validate_hard_quality(request, response)


def test_hard_quality_rejects_unchanged_leet_line_inside_translated_prose():
    leet = "L3t's t4lk 4b0ut h1st0ry's m0st f4m0us d15put3s"
    request = TranslationRequest(batch_id="b", items=[
        TranslationInputItem(seq=1, record_uid="u", prompt="Translate this example:\n" + leet, response=None),
    ])
    response = TranslationResponse(batch_id="b", items=[
        TranslationOutputItem(seq=1, record_uid="u", prompt_vi="Hãy dịch ví dụ này:\n" + leet, response_vi=None),
    ])
    with pytest.raises(TranslationValidationError, match="copied unchanged"):
        validate_hard_quality(request, response)


def test_hard_quality_allows_unchanged_digit_heavy_line_inside_fenced_code():
    code_line = 'text = "1s 1t tru3 th4t th3 m4n4g3r w45 t3rm1n4t3d l45t y34r?"'
    request = TranslationRequest(
        batch_id="b",
        items=[TranslationInputItem(seq=1, record_uid="u", prompt="Write this program", response="```python\n" + code_line + "\n```")],
    )
    response = TranslationResponse(
        batch_id="b",
        items=[TranslationOutputItem(seq=1, record_uid="u", prompt_vi="Viết chương trình này", response_vi="```python\n" + code_line + "\n```")],
    )

    validate_hard_quality(request, response)


def test_hard_quality_preserves_redaction_marker_exactly():
    request = TranslationRequest(batch_id="b", items=[
        TranslationInputItem(seq=1, record_uid="u", prompt="Name: [REDACTED]", response=None),
    ])
    response = TranslationResponse(batch_id="b", items=[
        TranslationOutputItem(seq=1, record_uid="u", prompt_vi="Tên: [ĐÃ XÓA]", response_vi=None),
    ])
    with pytest.raises(TranslationValidationError, match="redaction marker"):
        validate_hard_quality(request, response)


def test_hard_quality_does_not_treat_quoted_prose_title_as_json_key():
    request = TranslationRequest(batch_id="b", items=[
        TranslationInputItem(seq=1, record_uid="u", prompt='1. "The Great Escape": Players leave.', response=None),
    ])
    response = TranslationResponse(batch_id="b", items=[
        TranslationOutputItem(seq=1, record_uid="u", prompt_vi='1. "Cuộc đào tẩu vĩ đại": Người chơi rời đi.', response_vi=None),
    ])
    validate_hard_quality(request, response)


def test_hard_quality_does_not_match_redacted_inside_unredacted():
    request = TranslationRequest(batch_id="b", items=[
        TranslationInputItem(seq=1, record_uid="u", prompt="Return an unredacted answer.", response=None),
    ])
    response = TranslationResponse(batch_id="b", items=[
        TranslationOutputItem(seq=1, record_uid="u", prompt_vi="Trả lời đầy đủ, không che giấu.", response_vi=None),
    ])
    validate_hard_quality(request, response)


def test_hard_quality_uses_local_context_for_compromise_sense():
    request = TranslationRequest(batch_id="b", items=[
        TranslationInputItem(
            seq=1,
            record_uid="u",
            prompt="Poor communication can cause conflict. " + "Other details. " * 30 + "He resists compromise.",
            response=None,
        ),
    ])
    response = TranslationResponse(batch_id="b", items=[
        TranslationOutputItem(
            seq=1,
            record_uid="u",
            prompt_vi="Giao tiếp kém có thể gây xung đột. " + "Chi tiết khác. " * 30 + "Anh ta phản đối thỏa hiệp.",
            response_vi=None,
        ),
    ])
    validate_hard_quality(request, response)


@pytest.mark.parametrize("strong_term", ["địt mẹ", "chó chết", "mẹ nó", "mặt lồn"])
def test_hard_quality_accepts_comparable_vietnamese_profanity(strong_term):
    request = TranslationRequest(batch_id="b", items=[
        TranslationInputItem(seq=1, record_uid="u", prompt="This fucking rule is bullshit.", response=None),
    ])
    response = TranslationResponse(batch_id="b", items=[
        TranslationOutputItem(seq=1, record_uid="u", prompt_vi=f"Cái luật {strong_term} này.", response_vi=None),
    ])
    validate_hard_quality(request, response)
