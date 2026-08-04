from translator.jsonl_io import read_jsonl, write_jsonl
from translator.pipeline import TranslationPipeline, is_transient, retry_sleep_seconds
from translator.providers.dry_run import DryRunProvider


class AlwaysWrongBatchIdProvider(DryRunProvider):
    def translate(self, request, repair_errors=None):
        result = super().translate(request, repair_errors=repair_errors)
        result.response.batch_id = "wrong-batch-id"
        return result


def sources(n=3):
    return [{"record_uid": f"u{i}", "id": "same", "prompt": f"p{i}", "response": None, "length_bucket": "normal"} for i in range(n)]


def test_recursive_repair_and_resume(tmp_path):
    src, out, cp, failed = [tmp_path / x for x in ("src.jsonl","out.jsonl","cp.jsonl","failed.jsonl")]
    write_jsonl(src, sources())
    provider = DryRunProvider(failure_mode="missing")
    pipe = TranslationPipeline(provider, cp, failed)
    stats = pipe.run(src, out)
    assert stats["completed"] == 3 and len(list(read_jsonl(out))) == 3
    provider2 = DryRunProvider()
    stats2 = TranslationPipeline(provider2, cp, failed).run(src, out)
    assert stats2["skipped_resume"] == 3 and provider2.calls == 0


def test_duplicate_source_ids_do_not_matter(tmp_path):
    src, out, cp, failed = [tmp_path / x for x in ("src.jsonl","out.jsonl","cp.jsonl","failed.jsonl")]
    write_jsonl(src, sources(2))
    stats = TranslationPipeline(DryRunProvider(), cp, failed).run(src, out)
    assert stats["output_records"] == 2


def test_dry_run_is_not_rejected_by_semantic_translation_gate(tmp_path):
    src, out, cp, failed = [tmp_path / x for x in ("src.jsonl", "out.jsonl", "cp.jsonl", "failed.jsonl")]
    write_jsonl(src, [{
        "record_uid": "u-profane", "id": "x", "prompt": "This fucking text is only a dry-run fixture.",
        "response": None, "length_bucket": "normal",
    }])
    stats = TranslationPipeline(DryRunProvider(), cp, failed).run(src, out)
    assert stats["completed"] == 1
    assert stats["failed"] == 0


def test_request_budget_stops_cleanly_and_resume_finishes(tmp_path):
    src, out, cp, failed = [tmp_path / x for x in ("src.jsonl", "out.jsonl", "cp.jsonl", "failed.jsonl")]
    rows = [{"record_uid": f"u{i}", "id": str(i), "prompt": f"p{i}", "response": None, "length_bucket": "normal"} for i in range(35)]
    write_jsonl(src, rows)
    first = TranslationPipeline(DryRunProvider(), cp, failed, max_api_requests=1).run(src, out)
    assert first["deferred"] is True
    assert first["completed"] == 30
    assert first["failed"] == 0
    second = TranslationPipeline(DryRunProvider(), cp, failed, max_api_requests=1).run(src, out)
    assert second["output_records"] == 35
    assert second["skipped_resume"] == 30


def test_pipeline_groups_length_buckets_but_restores_source_order(tmp_path):
    src, out, cp, failed = [tmp_path / x for x in ("src.jsonl", "out.jsonl", "cp.jsonl", "failed.jsonl")]
    rows = [
        {"record_uid": "normal-1", "prompt": "a", "response": None, "length_bucket": "normal"},
        {"record_uid": "tail-1", "prompt": "b" * 2_500, "response": None, "length_bucket": "tail"},
        {"record_uid": "normal-2", "prompt": "c", "response": None, "length_bucket": "normal"},
        {"record_uid": "tail-2", "prompt": "d" * 2_500, "response": None, "length_bucket": "tail"},
    ]
    write_jsonl(src, rows)
    provider = DryRunProvider()
    stats = TranslationPipeline(provider, cp, failed).run(src, out)
    output_uids = [row["record_uid"] for _, row, _ in read_jsonl(out)]
    assert stats["batches_initial"] == 2
    assert provider.calls == 2
    assert output_uids == [row["record_uid"] for row in rows]


def test_shards_are_disjoint_and_cover_source(tmp_path):
    src = tmp_path / "src.jsonl"
    rows = sources(11)
    write_jsonl(src, rows)
    found = []
    for shard_index in range(5):
        out, cp, failed = [tmp_path / f"{kind}-{shard_index}.jsonl" for kind in ("out", "cp", "failed")]
        stats = TranslationPipeline(DryRunProvider(), cp, failed).run(
            src, out, shard_index=shard_index, shard_count=5,
        )
        assert stats["shard_index"] == shard_index
        found.extend(row["record_uid"] for _, row, _ in read_jsonl(out))
    assert len(found) == len(set(found))
    assert set(found) == {row["record_uid"] for row in rows}


def test_batch_id_prefix_distinguishes_parallel_workers(tmp_path):
    src, out, cp, failed = [tmp_path / x for x in ("src.jsonl", "out.jsonl", "cp.jsonl", "failed.jsonl")]
    write_jsonl(src, sources(2))
    TranslationPipeline(DryRunProvider(), cp, failed).run(src, out, batch_id_prefix="g3-")
    translated = [row for _, row, _ in read_jsonl(out)]
    assert all(row["translation_batch_id"].startswith("g3-full-") for row in translated)


def test_429_uses_quota_window_cooldown():
    assert retry_sleep_seconds(RuntimeError("429 RESOURCE_EXHAUSTED"), 0) == 30
    assert retry_sleep_seconds(RuntimeError("429 RESOURCE_EXHAUSTED"), 4) == 30
    assert retry_sleep_seconds(RuntimeError("503 unavailable"), 0) == 2


def test_http_code_inside_record_uid_is_not_transient():
    error = RuntimeError("en-train-00009577-35033e79d03d prompt: mostly untranslated English")
    assert is_transient(error) is False
    assert is_transient(RuntimeError("HTTP 503 unavailable")) is True


def test_validation_error_with_http_like_record_ids_is_not_transient():
    from translator.validators import TranslationValidationError

    error = TranslationValidationError([
        "missing record_uid: en-train-00000429-abc,en-train-00000504-def"
    ])
    assert is_transient(error) is False


def test_three_invalid_api_replies_quarantine_original_batch_and_continue(tmp_path):
    src, out, cp, failed = [tmp_path / x for x in ("src.jsonl", "out.jsonl", "cp.jsonl", "failed.jsonl")]
    write_jsonl(src, sources(3))
    provider = AlwaysWrongBatchIdProvider()
    stats = TranslationPipeline(provider, cp, failed).run(src, out)
    quarantined = [row for _, row, _ in read_jsonl(failed)]
    candidates_path = failed.with_name("failed_candidates.jsonl")
    candidate_archives = [row for _, row, _ in read_jsonl(candidates_path)]

    assert provider.calls == 3
    assert stats["quarantined_batches"] == 1
    assert stats["quarantined_records"] == 3
    assert stats["failed"] == 3
    assert stats["completed"] == 0
    assert {row["translation_status"] for row in quarantined} == {"needs_revision"}
    assert {row["validation_rejections"] for row in quarantined} == {3}
    assert len(candidate_archives) == 1
    assert candidate_archives[0]["candidate_count"] == 3
    assert len(candidate_archives[0]["candidates"]) == 3
    assert all(candidate["response"] for candidate in candidate_archives[0]["candidates"])

    resumed_provider = AlwaysWrongBatchIdProvider()
    resumed = TranslationPipeline(resumed_provider, cp, failed).run(src, out)
    assert resumed_provider.calls == 0
    assert resumed["skipped_quarantined"] == 3
