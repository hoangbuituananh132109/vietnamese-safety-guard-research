from __future__ import annotations

import json
from pathlib import Path

import pytest

from translator.checkpoint import load_checkpoint
from translator.dashboard import DashboardData, parse_revision_output
from translator.full_run import paths


def test_parse_revision_output_accepts_object_array_jsonl_and_fence() -> None:
    row = {"record_uid": "u1", "prompt_vi": "xin chào", "response_vi": None}
    assert parse_revision_output(json.dumps({"items": [row]})) == [row]
    assert parse_revision_output(json.dumps([row])) == [row]
    assert parse_revision_output(json.dumps(row) + "\n") == [row]
    assert parse_revision_output("```json\n" + json.dumps({"items": [row]}) + "\n```") == [row]


def test_parse_revision_output_reports_bad_jsonl_line() -> None:
    with pytest.raises(ValueError, match="dòng 2"):
        parse_revision_output('{"record_uid":"u1"}\nnot-json')


def test_validate_review_item_preserves_null_and_rejects_missing_translation() -> None:
    source = {
        "record_uid": "en-test-00000001-test",
        "source": {"prompt_en": "Hello there", "response_en": None},
    }
    assert DashboardData.validate_review_item(source, {
        "record_uid": source["record_uid"], "prompt_vi": "Xin chào", "response_vi": None,
    }) == []
    errors = DashboardData.validate_review_item(source, {
        "record_uid": source["record_uid"], "prompt_vi": "", "response_vi": "không được có",
    })
    assert any("empty prompt_vi" in error for error in errors)
    assert any("null response not preserved" in error for error in errors)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _review_root(tmp_path: Path, rows: list[dict]) -> DashboardData:
    directory = tmp_path / "data" / "revision_handoff" / "logical_terra_batches"
    _write_jsonl(directory / "manifest.jsonl", [{
        "file": "batch_001.jsonl",
        "record_uids": [row["record_uid"] for row in rows],
    }])
    _write_jsonl(directory / "batch_001.jsonl", rows)
    return DashboardData(tmp_path)


def test_persist_review_failures_replaces_latest_candidate_and_clears_on_pass(tmp_path: Path, monkeypatch) -> None:
    uid = "en-train-00000001-failed"
    source_row = {
        "record_uid": uid,
        "source_split": "train",
        "group": 1,
        "source": {"prompt_en": "Hello", "response_en": None},
    }
    data = _review_root(tmp_path, [source_row])
    monkeypatch.setattr(data, "completed_review_uids", lambda: {})

    first_raw = json.dumps({"items": [{"record_uid": uid, "prompt_vi": "cũ", "response_vi": None}]})
    first_validation = {
        "expected": 1,
        "received": 1,
        "pass_count": 0,
        "fail_count": 1,
        "results": [{"record_uid": uid, "status": "fail", "errors": ["old error"]}],
    }
    first = data.persist_review_failures("terra", 1, first_raw, "model-a", first_validation)
    assert first == {"saved_failures": 1, "replaced_failures": 0, "cleared_failures": 0}

    second_raw = json.dumps({"items": [{"record_uid": uid, "prompt_vi": "mới", "response_vi": None}]})
    second_validation = {
        **first_validation,
        "results": [{"record_uid": uid, "status": "fail", "errors": ["new error"]}],
    }
    second = data.persist_review_failures("terra", 1, second_raw, "model-b", second_validation)
    assert second["saved_failures"] == 1
    assert second["replaced_failures"] == 1

    snapshot = json.loads(data.review_failure_path("terra", 1).read_text(encoding="utf-8"))
    assert snapshot["raw_output"] == second_raw
    assert snapshot["attempt_count"] == 2
    assert snapshot["failures"][0]["candidate"]["prompt_vi"] == "mới"
    assert snapshot["failures"][0]["errors"] == ["new error"]
    assert snapshot["failures"][0]["attempt_count"] == 2
    queue = data.review_failure_queue("terra")
    assert queue["total"] == 1
    assert queue["records"][0]["record_uid"] == uid

    missing_raw = json.dumps({"items": [{"record_uid": "some-other-uid", "prompt_vi": "x", "response_vi": None}]})
    data.persist_review_failures("terra", 1, missing_raw, "model-c", {
        "expected": 1,
        "received": 0,
        "pass_count": 0,
        "fail_count": 1,
        "results": [{"record_uid": uid, "status": "missing", "errors": ["missing now"]}],
    })
    snapshot = json.loads(data.review_failure_path("terra", 1).read_text(encoding="utf-8"))
    assert snapshot["failures"][0]["candidate"]["prompt_vi"] == "mới"
    assert snapshot["failures"][0]["candidate_reused_from_previous_attempt"] is True

    cleared = data.persist_review_failures("terra", 1, second_raw, "model-b", {
        "expected": 1,
        "received": 1,
        "pass_count": 1,
        "fail_count": 0,
        "results": [{"record_uid": uid, "status": "pass", "errors": []}],
    })
    assert cleared["saved_failures"] == 0
    assert cleared["cleared_failures"] == 1
    assert not data.review_failure_path("terra", 1).exists()


def test_merge_saves_only_pass_to_checkpoint_and_quarantines_fail(tmp_path: Path, monkeypatch) -> None:
    passed_uid = "en-train-00000001-passed"
    failed_uid = "en-train-00000002-failed"
    batch_rows = [
        {"record_uid": passed_uid, "source_split": "train", "group": 1, "source": {"prompt_en": "Hello", "response_en": None}},
        {"record_uid": failed_uid, "source_split": "train", "group": 1, "source": {"prompt_en": "World", "response_en": None}},
    ]
    data = _review_root(tmp_path, batch_rows)
    for split in ("train", "valid", "test"):
        prepared = [{"record_uid": passed_uid, "prompt": "Hello", "response": None}] if split == "train" else []
        _write_jsonl(tmp_path / "data" / "prepared" / f"nemotron_en_{split}_full_v1.jsonl", prepared)
    monkeypatch.setattr(data, "review_status", lambda: {"completed_total": 1})
    validation = {
        "ok": False,
        "reviewer": "terra",
        "batch": 1,
        "expected": 2,
        "received": 2,
        "pass_count": 1,
        "fail_count": 1,
        "results": [
            {"record_uid": passed_uid, "status": "pass", "errors": []},
            {"record_uid": failed_uid, "status": "fail", "errors": ["still English"]},
        ],
        "normalized": [{"record_uid": passed_uid, "prompt_vi": "Xin chào", "response_vi": None, "fix_notes": []}],
        "repair_prompt": "repair",
    }
    monkeypatch.setattr(data, "validate_review_output", lambda reviewer, batch, raw: validation)
    raw = json.dumps({"items": [
        {"record_uid": passed_uid, "prompt_vi": "Xin chào", "response_vi": None},
        {"record_uid": failed_uid, "prompt_vi": "World", "response_vi": None},
    ]})

    result = data.merge_review_output("terra", 1, raw, "manual-model")
    assert result["merged"] == 1
    assert result["saved_failures"] == 1
    checkpoint = load_checkpoint(paths(tmp_path, "train", 0)["checkpoint"])
    assert set(checkpoint) == {passed_uid}
    assert checkpoint[passed_uid]["translation_status"] == "terra_revised"
    snapshot = json.loads(data.review_failure_path("terra", 1).read_text(encoding="utf-8"))
    assert [item["record_uid"] for item in snapshot["failures"]] == [failed_uid]
    assert snapshot["failures"][0]["candidate"]["prompt_vi"] == "World"


def test_invalid_json_can_be_quarantined_at_batch_level(tmp_path: Path, monkeypatch) -> None:
    uid = "en-train-00000001-invalid-json"
    data = _review_root(tmp_path, [{
        "record_uid": uid,
        "source_split": "train",
        "group": 1,
        "source": {"prompt_en": "Hello", "response_en": None},
    }])
    monkeypatch.setattr(data, "completed_review_uids", lambda: {})
    monkeypatch.setattr(data, "review_status", lambda: {"completed_total": 0})
    raw = '{"items":[{"record_uid":"broken"'

    validation = data.validate_review_output("terra", 1, raw)
    assert validation["pass_count"] == 0
    assert validation["fail_count"] == 1
    assert validation["results"][0]["status"] == "invalid"
    result = data.merge_review_output("terra", 1, raw, "manual-model")
    assert result["merged"] == 0
    assert result["saved_failures"] == 1
    snapshot = json.loads(data.review_failure_path("terra", 1).read_text(encoding="utf-8"))
    assert snapshot["raw_output"] == raw
    assert snapshot["validation_summary"]["parse_error"]
    assert snapshot["failures"][0]["record_uid"] is None
