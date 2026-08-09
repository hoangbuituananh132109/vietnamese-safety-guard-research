from tools.luna_overnight_runner import (
    has_json_shape,
    has_prose_leetspeak,
    make_initial_tasks,
    max_attempts,
    next_resume_task_id,
    route_for,
    usage_credits,
)
from translator.jsonl_io import append_jsonl


def row(uid: str, text: str, bucket: str = "normal") -> dict:
    return {"record_uid": uid, "prompt_en": text, "response_en": None, "length_bucket": bucket, "_seq": 1}


def test_shape_routing_prioritizes_leet_then_json_then_prose():
    assert route_for(row("leet", "Th3 c4t 15 0n th3 t4bl3 and 1t 15 h3r3")) == "leet"
    assert route_for(row("json", 'Return {"Age of Consent":"Adult"}')) == "json"
    assert route_for(row("prose", "Translate this ordinary sentence.")) == "prose"


def test_json_shape_requires_an_actual_quoted_key():
    assert has_json_shape('text {"key": "value"}') is True
    assert has_json_shape("please return JSON someday") is False
    assert has_json_shape("set notation {a, b}") is False


def test_leet_detector_ignores_single_identifier_but_finds_prose():
    assert has_prose_leetspeak("Use ISO27001 and H2") is False
    assert has_prose_leetspeak("pl3453 tr4n5l473 th15 1nt0 l3375p34k f0r m3") is True


def test_batching_uses_route_and_length_profiles():
    rows = [row(f"p{i}", "plain") for i in range(31)]
    rows += [row(f"j{i}", '{"Key":"Value"}', "tail") for i in range(3)]
    tasks = make_initial_tasks(rows)
    assert [(task.route, task.bucket, len(task.rows)) for task in tasks] == [
        ("prose", "normal", 30),
        ("prose", "normal", 1),
        ("json", "tail", 2),
        ("json", "tail", 1),
    ]


def test_tail_gets_one_retry_while_normal_gets_three_total_attempts():
    assert max_attempts("normal") == 3
    assert max_attempts("near_tail") == 3
    assert max_attempts("tail") == 2
    assert max_attempts("high_tail") == 2


def test_resume_batch_ids_continue_after_raw_history(tmp_path):
    raw = tmp_path / "raw_batches.jsonl"
    append_jsonl(raw, {"task_id": 7})
    append_jsonl(raw, {"task_id": 12})
    assert next_resume_task_id(raw) == 13
    tasks = make_initial_tasks([row("p", "plain")], start_task_id=13)
    assert tasks[0].task_id == 13


def test_credit_meter_uses_user_supplied_input_and_output_rates():
    usage = {"input_tokens": 2_000_000, "cached_input_tokens": 1_000_000, "output_tokens": 4_000_000}
    assert usage_credits(usage, 5.0, 0.5, 30.0) == 125.5
