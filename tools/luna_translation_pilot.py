from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translator.jsonl_io import append_jsonl, read_jsonl, safe_json_dumps, write_jsonl
from translator.models import TranslationInputItem, TranslationOutputItem, TranslationRequest, TranslationResponse
from translator.validators import TranslationValidationError, quality_warnings, validate_hard_quality, validate_response


PILOT_ROOT = ROOT / "data" / "luna_pilot_200"
FINAL_FILES = [
    ROOT / "data/final/nemotron_train_en_vi_v10_final.jsonl",
    ROOT / "data/final/nemotron_valid_en_vi_v10_final.jsonl",
    ROOT / "data/final/nemotron_test_en_vi_v10_final.jsonl",
]
RAW_GEMINI_FILES = sorted((ROOT / "data/translated").glob("nemotron_*_full_vi_v10_g*.jsonl"))
PROMPT_PATH = ROOT / "configs/luna_fresh_translation_prompt_v1.md"
DEFAULT_CODEX = ROOT / ".runtime/luna-codex-cli/node_modules/.bin/codex.cmd"


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(row for _, row, _ in read_jsonl(path))
    return rows


def load_raw_gemini() -> dict[str, dict[str, Any]]:
    # Keep the highest translation attempt when archived files contain retries.
    result: dict[str, dict[str, Any]] = {}
    for row in load_rows(RAW_GEMINI_FILES):
        uid = str(row["record_uid"])
        current = result.get(uid)
        if current is None or int(row.get("translation_attempt") or 0) >= int(current.get("translation_attempt") or 0):
            result[uid] = row
    return result


def source_text(row: dict[str, Any]) -> str:
    return (row.get("prompt_en") or "") + "\n" + (row.get("response_en") or "")


def injection_score(row: dict[str, Any]) -> int:
    text = source_text(row).casefold()
    markers = (
        "ignore previous", "ignore all", "system prompt", "developer message", "jailbreak",
        "dan mode", "do anything now", "prompt injection", "respond only", "only respond",
        "you are chatgpt", "act as", "roleplay", "role-play", "[prompt]", "targetlanguage",
        "base64", "decode", "leet", "l3t", "1gn0r", "### instruction", "<system>",
    )
    score = sum(2 for marker in markers if marker in text)
    score += min(5, sum(ch.isdigit() or ch in "@$" for ch in text) // 8)
    score += 2 if row.get("tag") == "jailbreaking" else 0
    return score


def stable_pick(rows: list[dict[str, Any]], count: int, seed: int, prefer_jailbreak: bool = False) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for row in rows:
        tie = rng.random()
        rank = (
            -(injection_score(row) if prefer_jailbreak else 0),
            0 if row.get("tag") == "jailbreaking" and prefer_jailbreak else 1,
            0 if row.get("length_bucket") in {"tail", "high_tail", "oversized"} else 1,
            tie,
        )
        ranked.append((rank, row))
    ranked.sort(key=lambda item: item[0])
    chosen: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    for _, row in ranked:
        split = str(row.get("source_split"))
        bucket = str(row.get("length_bucket"))
        # Soft diversity guard; a second pass below fills any remainder.
        if split_counts[split] > count * 0.75 or bucket_counts[bucket] > count * 0.75:
            continue
        chosen.append(row)
        split_counts[split] += 1
        bucket_counts[bucket] += 1
        if len(chosen) == count:
            break
    if len(chosen) < count:
        seen = {row["record_uid"] for row in chosen}
        chosen.extend(row for _, row in ranked if row["record_uid"] not in seen)
        chosen = chosen[:count]
    return chosen


def prepare(seed: int = 560200) -> None:
    final_rows = load_rows(FINAL_FILES)
    raw = load_raw_gemini()
    by_uid = {row["record_uid"]: row for row in final_rows}
    selected: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()

    def add(group: str, candidates: list[dict[str, Any]], count: int, local_seed: int, prefer: bool = False) -> None:
        pool = [row for row in candidates if row["record_uid"] not in used]
        for row in stable_pick(pool, count, local_seed, prefer):
            uid = row["record_uid"]
            if uid in used:
                continue
            used.add(uid)
            selected.append((group, row))

    manual_web = [row for row in final_rows if row.get("translation_provider") == "manual_web_review"]
    audited = [row for row in final_rows if row.get("translation_provider") == "codex_human_review"]
    revised_other = [
        row for row in final_rows
        if row.get("translation_status") in {"gemini_revised", "terra_revised", "luna_revised"}
        and row.get("translation_provider") != "manual_web_review"
        and row.get("translation_provider") != "codex_human_review"
    ]
    machine = [
        row for row in final_rows
        if row.get("translation_status") in {"machine_translated", "provisional_clean", "provisional_format_repaired"}
    ]
    injection_machine = [row for row in machine if injection_score(row) > 0]

    add("sol_web_reference", manual_web, 80, seed + 1, True)
    add("audited_validator_regression", audited, 35, seed + 2, True)
    add("other_revised_reference", revised_other, 15, seed + 3, True)
    add("machine_jailbreak_challenge", injection_machine, 35, seed + 4, True)
    add("machine_control", machine, 35, seed + 5, False)
    if len(selected) != 200:
        raise RuntimeError(f"selection count is {len(selected)}, expected 200")

    output: list[dict[str, Any]] = []
    # First 120 are calibration, but interleave groups deterministically.
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for item in selected:
        grouped[item[0]].append(item)
    ordered: list[tuple[str, dict[str, Any]]] = []
    while any(grouped.values()):
        for group in sorted(grouped):
            if grouped[group]:
                ordered.append(grouped[group].pop(0))
    calibration_uids = {row["record_uid"] for _, row in ordered[:120]}

    for index, (group, row) in enumerate(ordered, 1):
        uid = row["record_uid"]
        gemini = raw.get(uid, row)
        output.append({
            "pilot_seq": index,
            "record_uid": uid,
            "pilot_group": group,
            "pilot_partition": "calibration" if uid in calibration_uids else "holdout",
            "source_split": row.get("source_split"),
            "tag": row.get("tag"),
            "length_bucket": row.get("length_bucket"),
            "violated_categories": row.get("violated_categories"),
            "prompt_label": row.get("prompt_label"),
            "response_label": row.get("response_label"),
            "injection_score": injection_score(row),
            "prompt_en": row.get("prompt_en"),
            "response_en": row.get("response_en"),
            "gemini_prompt_vi": gemini.get("prompt_vi"),
            "gemini_response_vi": gemini.get("response_vi"),
            "gemini_status": gemini.get("translation_status"),
            "reference_prompt_vi": row.get("prompt_vi"),
            "reference_response_vi": row.get("response_vi"),
            "reference_status": row.get("translation_status"),
            "reference_provider": row.get("translation_provider"),
            "reference_model": row.get("translation_model"),
            "reference_provenance": (
                "likely_sol_web_per_project_history_metadata_slug_not_preserved"
                if group == "sol_web_reference" else "final_audited_or_pipeline_reference"
            ),
        })

    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(PILOT_ROOT / "pilot_200.jsonl", output)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "records": len(output),
        "calibration": sum(row["pilot_partition"] == "calibration" for row in output),
        "holdout": sum(row["pilot_partition"] == "holdout" for row in output),
        "groups": dict(Counter(row["pilot_group"] for row in output)),
        "splits": dict(Counter(str(row["source_split"]) for row in output)),
        "tags": dict(Counter(str(row["tag"]) for row in output)),
        "length_buckets": dict(Counter(str(row["length_bucket"]) for row in output)),
        "prompt_sha256": hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(),
        "source_final_records": len(by_uid),
        "raw_gemini_records": len(raw),
    }
    (PILOT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def output_schema() -> dict[str, Any]:
    schema = TranslationResponse.model_json_schema()

    def strictify(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties)
                node["additionalProperties"] = False
            for value in node.values():
                strictify(value)
        elif isinstance(node, list):
            for value in node:
                strictify(value)

    strictify(schema)
    return schema


def make_batches(rows: list[dict[str, Any]], max_items: int, max_chars: int) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    chars = 0
    for row in rows:
        row_chars = len(row.get("prompt_en") or "") + len(row.get("response_en") or "")
        if current and (len(current) >= max_items or chars + row_chars > max_chars):
            batches.append(current)
            current, chars = [], 0
        current.append(row)
        chars += row_chars
    if current:
        batches.append(current)
    return batches


def make_bucketed_batches(rows: list[dict[str, Any]], normal_max_items: int) -> list[list[dict[str, Any]]]:
    rules = {
        "normal": (normal_max_items, 50_000),
        "near_tail": (20, 45_000),
        "tail": (8, 30_000),
        # One high-tail record per ephemeral turn. A paired high-tail turn in the
        # 200-item pilot remained active past the runner's two-hour outer window,
        # so keeping these pathological payloads isolated makes retries lossless.
        "high_tail": (1, 30_000),
        "oversized": (1, 200_000),
    }
    result: list[list[dict[str, Any]]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("length_bucket") or "normal")].append(row)
    for bucket in ("normal", "near_tail", "tail", "high_tail", "oversized"):
        max_items, max_chars = rules[bucket]
        result.extend(make_batches(grouped[bucket], max_items=max_items, max_chars=max_chars))
    return result


def parse_codex_stream(stdout: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            events.append({"type": "unparsed_stdout", "text": line})
            continue
        events.append(event)
        if event.get("type") == "item.completed" and (event.get("item") or {}).get("type") == "agent_message":
            text = event["item"].get("text") or ""
            try:
                final = json.loads(text)
            except json.JSONDecodeError:
                final = {"_unparsed_agent_message": text}
        if event.get("type") == "turn.completed":
            usage = event.get("usage")
    return final, usage, events


def validate_partial(request: TranslationRequest, raw_response: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[str]]:
    saved: list[dict[str, Any]] = []
    errors: list[str] = []
    if raw_response is None:
        return saved, ["missing final response"]
    if "_unparsed_agent_message" in raw_response:
        return saved, ["unparsed agent response"]
    expected = {item.record_uid: item for item in request.items}
    seen: set[str] = set()
    for raw_item in raw_response.get("items") or []:
        uid = str(raw_item.get("record_uid") or "")
        try:
            item = TranslationOutputItem.model_validate(raw_item)
        except Exception as exc:
            errors.append(f"item schema failure uid={uid!r}: {exc}")
            saved.append({"record_uid": uid, "raw_item": raw_item, "structural_errors": [str(exc)]})
            continue
        structural: list[str] = []
        source = expected.get(uid)
        if uid in seen:
            structural.append("duplicate record_uid")
        seen.add(uid)
        if source is None:
            structural.append("unknown record_uid")
        elif item.seq != source.seq:
            structural.append("seq mismatch")
        elif source.response is None and item.response_vi is not None:
            structural.append("null response not preserved")
        elif source.response == "" and item.response_vi != "":
            structural.append("empty response not preserved")
        saved.append({**item.model_dump(), "structural_errors": structural})
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append("missing record_uid: " + ",".join(missing))
    return saved, errors


def run_variant(
    variant: str,
    partition: str,
    max_items: int,
    max_chars: int,
    model: str,
    reasoning_effort: str,
    codex_path: Path,
    resume: bool,
    limit_batches: int | None,
    prompt_path: Path,
    bucket_profile: bool,
    length_bucket: str | None,
    reuse_variants: list[str],
    skip_uids: list[str],
    include_uids: list[str],
) -> None:
    skipped = set(skip_uids)
    included = set(include_uids)
    rows = [
        row for _, row, _ in read_jsonl(PILOT_ROOT / "pilot_200.jsonl")
        if (partition == "all" or row["pilot_partition"] == partition)
        and (length_bucket is None or row.get("length_bucket") == length_bucket)
        and row.get("record_uid") not in skipped
        and (not included or row.get("record_uid") in included)
    ]
    batches = (
        make_bucketed_batches(rows, normal_max_items=max_items)
        if bucket_profile else make_batches(rows, max_items=max_items, max_chars=max_chars)
    )
    out_dir = PILOT_ROOT / "runs" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_path = PILOT_ROOT / "translation_response.schema.json"
    schema_path.write_text(json.dumps(output_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    raw_path = out_dir / "raw_batches.jsonl"
    item_path = out_dir / "items.jsonl"
    event_path = out_dir / "events.jsonl"
    completed = set()
    if resume and item_path.exists():
        completed = {row.get("record_uid") for _, row, _ in read_jsonl(item_path) if row.get("structural_errors") == []}
    for reuse_variant in reuse_variants:
        reuse_path = PILOT_ROOT / "runs" / reuse_variant / "items.jsonl"
        if reuse_path.exists():
            completed.update(
                row.get("record_uid") for _, row, _ in read_jsonl(reuse_path)
                if row.get("record_uid") and row.get("structural_errors") == []
            )

    attempted_batches = 0
    consecutive_infrastructure_failures = 0
    for batch_index, batch in enumerate(batches, 1):
        if all(row["record_uid"] in completed for row in batch):
            continue
        if limit_batches is not None and attempted_batches >= limit_batches:
            break
        attempted_batches += 1
        batch_id = f"{variant}-{partition}-{batch_index:04d}"
        request = TranslationRequest(
            batch_id=batch_id,
            items=[
                TranslationInputItem(
                    seq=int(row["pilot_seq"]),
                    record_uid=row["record_uid"],
                    prompt=row.get("prompt_en") or "",
                    response=row.get("response_en"),
                )
                for row in batch if row["record_uid"] not in completed
            ],
        )
        payload = safe_json_dumps(request.model_dump())
        if "{{SOURCE_RECORDS}}" in prompt:
            stdin = prompt.replace("{{SOURCE_RECORDS}}", payload)
        else:
            stdin = prompt + "\n\n<source_records>\n" + payload + "\n</source_records>\n"
        command = [
            str(codex_path), "exec", "-", "--ephemeral", "--json", "--ignore-user-config", "--ignore-rules",
            "--skip-git-repo-check", "--sandbox", "read-only", "--model", model,
            "--config", f'model_reasoning_effort="{reasoning_effort}"',
            "--output-schema", str(schema_path), "--cd", str(PILOT_ROOT),
        ]
        started = time.time()
        proc = subprocess.run(command, input=stdin, text=True, encoding="utf-8", errors="replace", capture_output=True)
        elapsed = time.time() - started
        final, usage, events = parse_codex_stream(proc.stdout)
        saved, parse_errors = validate_partial(request, final)
        infrastructure_markers = (
            "invalid peer certificate", "unknownissuer", "error sending request",
            "authentication", "rate limit", "too many requests",
        )
        infrastructure_failure = final is None and any(
            marker in proc.stderr.casefold() for marker in infrastructure_markers
        )
        consecutive_infrastructure_failures = (
            consecutive_infrastructure_failures + 1 if infrastructure_failure else 0
        )
        batch_record = {
            "variant": variant, "batch_id": batch_id, "batch_index": batch_index,
            "requested_items": len(request.items), "saved_items": len(saved), "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 3), "usage": usage, "parse_errors": parse_errors,
            "stderr": proc.stderr[-4000:], "raw_final": final,
            "prompt_path": str(prompt_path), "prompt_sha256": prompt_sha256,
        }
        append_jsonl(raw_path, batch_record)
        for event in events:
            append_jsonl(event_path, {"batch_id": batch_id, "event": event}, fsync=False)
        source_by_uid = {item.record_uid: item for item in request.items}
        for item in saved:
            uid = item.get("record_uid")
            source = source_by_uid.get(uid)
            hard_errors: list[str] = []
            heuristic_warnings: list[str] = []
            if source is not None and not item.get("structural_errors"):
                response = TranslationResponse(
                    batch_id=batch_id,
                    items=[TranslationOutputItem.model_validate({
                        key: item.get(key)
                        for key in ("seq", "record_uid", "prompt_vi", "response_vi", "warnings")
                    })],
                )
                one_request = TranslationRequest(batch_id=batch_id, items=[source])
                try:
                    validate_response(one_request, response)
                    validate_hard_quality(one_request, response)
                except TranslationValidationError as exc:
                    hard_errors = exc.errors
                heuristic_warnings = quality_warnings(
                    {"prompt": source.prompt, "response": source.response},
                    {"prompt_vi": item.get("prompt_vi"), "response_vi": item.get("response_vi")},
                )
            append_jsonl(item_path, {
                "variant": variant, "batch_id": batch_id, **item,
                "hard_validator_errors": hard_errors,
                "heuristic_warnings": heuristic_warnings,
                "requires_audit": bool(item.get("structural_errors") or hard_errors or heuristic_warnings),
            })
        print(safe_json_dumps({
            "variant": variant, "batch": batch_index, "batches": len(batches),
            "requested": len(request.items), "saved": len(saved), "rc": proc.returncode,
            "elapsed": round(elapsed, 1), "usage": usage, "parse_errors": parse_errors,
        }), flush=True)
        if consecutive_infrastructure_failures >= 2:
            print(safe_json_dumps({
                "variant": variant,
                "stopped": "infrastructure_circuit_breaker",
                "consecutive_failures": consecutive_infrastructure_failures,
            }), flush=True)
            break


def summarize(variant: str) -> dict[str, Any]:
    out_dir = PILOT_ROOT / "runs" / variant
    raw = load_rows([out_dir / "raw_batches.jsonl"]) if (out_dir / "raw_batches.jsonl").exists() else []
    items = load_rows([out_dir / "items.jsonl"]) if (out_dir / "items.jsonl").exists() else []
    # Last saved item wins only for summary; raw history remains append-only.
    latest = {row.get("record_uid"): row for row in items if row.get("record_uid")}
    usage = Counter()
    successful_raw = [row for row in raw if row.get("returncode") == 0 and row.get("usage")]
    for row in successful_raw:
        for key, value in (row.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] += value
    uncached = max(0, usage.get("input_tokens", 0) - usage.get("cached_input_tokens", 0))
    estimated_credits = (
        uncached * 5 + usage.get("cached_input_tokens", 0) * 0.5 + usage.get("output_tokens", 0) * 30
    ) / 1_000_000
    summary = {
        "variant": variant,
        "attempted_processes": len(raw),
        "successful_processes": len(successful_raw),
        "requested_items_in_successful_processes": sum(int(row.get("requested_items") or 0) for row in successful_raw),
        "unique_saved_items": len(latest),
        "items_with_structural_errors": sum(bool(row.get("structural_errors")) for row in latest.values()),
        "items_with_hard_validator_errors": sum(bool(row.get("hard_validator_errors")) for row in latest.values()),
        "items_with_heuristic_warnings": sum(bool(row.get("heuristic_warnings")) for row in latest.values()),
        "items_requiring_audit": sum(bool(row.get("requires_audit")) for row in latest.values()),
        "usage": dict(usage),
        "estimated_credits": round(estimated_credits, 6),
        "elapsed_seconds_successful": round(sum(float(row.get("elapsed_seconds") or 0) for row in successful_raw), 3),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--seed", type=int, default=560200)
    run = sub.add_parser("run")
    run.add_argument("--variant", required=True)
    run.add_argument("--partition", choices=["calibration", "holdout", "all"], required=True)
    run.add_argument("--max-items", type=int, required=True)
    run.add_argument("--max-chars", type=int, default=50_000)
    run.add_argument("--model", default="gpt-5.6-luna")
    run.add_argument("--reasoning-effort", default="low")
    run.add_argument("--codex-path", type=Path, default=DEFAULT_CODEX)
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--limit-batches", type=int)
    run.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    run.add_argument("--bucket-profile", action="store_true")
    run.add_argument("--length-bucket", choices=["normal", "near_tail", "tail", "high_tail", "oversized"])
    run.add_argument("--reuse-variant", action="append", default=[])
    run.add_argument("--skip-uid", action="append", default=[])
    run.add_argument("--include-uid", action="append", default=[])
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("variants", nargs="+")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.seed)
    elif args.command == "run":
        run_variant(
            args.variant, args.partition, args.max_items, args.max_chars, args.model,
            args.reasoning_effort, args.codex_path, not args.no_resume, args.limit_batches, args.prompt_path,
            args.bucket_profile, args.length_bucket,
            args.reuse_variant,
            args.skip_uid,
            args.include_uid,
        )
    else:
        print(json.dumps([summarize(variant) for variant in args.variants], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
