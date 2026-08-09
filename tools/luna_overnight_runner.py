from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.luna_translation_pilot import output_schema, parse_codex_stream, validate_partial
from translator.jsonl_io import append_jsonl, read_jsonl, safe_json_dumps
from translator.models import TranslationInputItem, TranslationOutputItem, TranslationRequest, TranslationResponse
from translator.validators import TranslationValidationError, quality_warnings, validate_hard_quality, validate_response


DEFAULT_CODEX = ROOT / ".runtime/luna-codex-cli/node_modules/.bin/codex.cmd"
DEFAULT_PROMPTS = {
    "prose": ROOT / "configs/luna_fresh_translation_prompt_v3_sandwich.md",
    "json": ROOT / "configs/luna_fresh_translation_prompt_v6_json_keys.md",
    "leet": ROOT / "configs/luna_fresh_translation_prompt_v5_leet_strict.md",
}
MIXED_TOKEN = re.compile(
    r"(?i)(?<![a-z0-9])(?=[a-z0-9@$']*[a-z])(?=[a-z0-9@$']*[0-9@$])[a-z0-9@$']{2,}(?![a-z0-9])"
)
LEET_TRANS = str.maketrans({"4": "a", "3": "e", "0": "o", "1": "i", "5": "s", "7": "t", "@": "a", "$": "s"})
COMMON_DECODED = {
    "about", "all", "and", "are", "be", "can", "could", "do", "for", "from", "have", "how",
    "in", "into", "is", "it", "make", "me", "not", "of", "on", "or", "our", "out", "please",
    "rules", "should", "someone", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "translate", "use", "was", "we", "what", "when", "where", "which", "who", "will",
    "with", "would", "you", "your",
}


@dataclass(frozen=True)
class BatchTask:
    task_id: int
    rows: tuple[dict[str, Any], ...]
    route: str
    bucket: str


@dataclass
class BatchResult:
    task: BatchTask
    batch_id: str
    request: TranslationRequest
    saved: list[dict[str, Any]]
    parse_errors: list[str]
    returncode: int
    elapsed_seconds: float
    usage: dict[str, Any] | None
    stderr: str
    final: dict[str, Any] | None
    infrastructure_failure: bool
    timed_out: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_fields(row: dict[str, Any]) -> tuple[str, str | None]:
    prompt = row.get("prompt_en") if "prompt_en" in row else row.get("prompt")
    response = row.get("response_en") if "response_en" in row else row.get("response")
    return str(prompt or ""), response if response is None else str(response)


def source_chars(row: dict[str, Any]) -> int:
    prompt, response = source_fields(row)
    return len(prompt) + len(response or "")


def length_bucket(row: dict[str, Any]) -> str:
    existing = row.get("length_bucket")
    if existing in {"normal", "near_tail", "tail", "high_tail", "oversized"}:
        return str(existing)
    chars = source_chars(row)
    if chars <= 1695:
        return "normal"
    if chars <= 2342:
        return "near_tail"
    if chars <= 3772:
        return "tail"
    if chars <= 25_000:
        return "high_tail"
    return "oversized"


def has_prose_leetspeak(text: str) -> bool:
    cleaned = re.sub(r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.\w+|\b[0-9a-f]{16,}\b", " ", text, flags=re.I)
    tokens = [
        token for token in MIXED_TOKEN.findall(cleaned)
        if not re.fullmatch(r"(?i)h[1-6]s?", token)
        and not re.fullmatch(r"(?i)(?:iso)?\d{1,4}[a-z]\d{1,4}", token)
    ]
    decoded = [token.casefold().translate(LEET_TRANS).strip("'") for token in tokens]
    recognized = sum(word in COMMON_DECODED for word in decoded)
    return recognized >= 2 or (len(tokens) >= 10 and recognized >= 1)


def has_json_shape(text: str) -> bool:
    # Require braces plus at least one quoted key followed by a colon. This avoids routing prose that
    # merely says the word JSON without containing a structure whose keys must be protected.
    return "{" in text and "}" in text and bool(re.search(r'"(?:[^"\\]|\\.)+"\s*:', text))


def route_for(row: dict[str, Any]) -> str:
    prompt, response = source_fields(row)
    text = prompt + "\n" + (response or "")
    if has_prose_leetspeak(text):
        return "leet"
    if has_json_shape(text):
        return "json"
    return "prose"


BATCH_RULES: dict[tuple[str, str], tuple[int, int | None]] = {
    ("prose", "normal"): (30, 50_000),
    ("prose", "near_tail"): (20, 45_000),
    ("prose", "tail"): (3, 12_000),
    ("prose", "high_tail"): (1, 25_000),
    ("prose", "oversized"): (1, None),
    ("json", "normal"): (10, 20_000),
    ("json", "near_tail"): (6, 16_000),
    ("json", "tail"): (2, 8_000),
    ("json", "high_tail"): (1, 25_000),
    ("json", "oversized"): (1, None),
    ("leet", "normal"): (10, 16_000),
    ("leet", "near_tail"): (6, 14_000),
    ("leet", "tail"): (2, 8_000),
    ("leet", "high_tail"): (1, 25_000),
    ("leet", "oversized"): (1, None),
}


def make_initial_tasks(rows: Iterable[dict[str, Any]], start_task_id: int = 1) -> list[BatchTask]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (route_for(row), length_bucket(row))
        grouped.setdefault(key, []).append(row)
    tasks: list[BatchTask] = []
    next_id = start_task_id
    bucket_order = {"normal": 0, "near_tail": 1, "tail": 2, "high_tail": 3, "oversized": 4}
    route_order = {"prose": 0, "json": 1, "leet": 2}
    for route, bucket in sorted(grouped, key=lambda key: (bucket_order[key[1]], route_order[key[0]])):
        max_items, max_chars = BATCH_RULES[(route, bucket)]
        current: list[dict[str, Any]] = []
        chars = 0
        for row in grouped[(route, bucket)]:
            row_chars = source_chars(row)
            if current and (len(current) >= max_items or (max_chars is not None and chars + row_chars > max_chars)):
                tasks.append(BatchTask(next_id, tuple(current), route, bucket))
                next_id += 1
                current, chars = [], 0
            current.append(row)
            chars += row_chars
        if current:
            tasks.append(BatchTask(next_id, tuple(current), route, bucket))
            next_id += 1
    return tasks


def max_attempts(bucket: str, override: int | None = None) -> int:
    if override is not None:
        return override
    # Three total attempts for normal/near-tail; one retry (two total attempts) from tail upward.
    return 3 if bucket in {"normal", "near_tail"} else 2


def kill_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    else:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_batch(
    task: BatchTask,
    run_id: str,
    model: str,
    reasoning_effort: str,
    codex_path: Path,
    schema_path: Path,
    prompt_paths: dict[str, Path],
    timeout_seconds: int,
) -> BatchResult:
    batch_id = f"{run_id}-{task.route}-{task.bucket}-{task.task_id:06d}"
    request = TranslationRequest(
        batch_id=batch_id,
        items=[
            TranslationInputItem(
                seq=int(row["_seq"]), record_uid=str(row["record_uid"]),
                prompt=source_fields(row)[0], response=source_fields(row)[1],
            )
            for row in task.rows
        ],
    )
    prompt_path = prompt_paths[task.route]
    prompt = prompt_path.read_text(encoding="utf-8")
    payload = safe_json_dumps(request.model_dump())
    stdin = prompt.replace("{{SOURCE_RECORDS}}", payload) if "{{SOURCE_RECORDS}}" in prompt else (
        prompt + "\n\n<source_records>\n" + payload + "\n</source_records>\n"
    )
    command = [
        str(codex_path), "exec", "-", "--ephemeral", "--json", "--ignore-user-config", "--ignore-rules",
        "--skip-git-repo-check", "--sandbox", "read-only", "--model", model,
        "--config", f'model_reasoning_effort="{reasoning_effort}"',
        "--output-schema", str(schema_path), "--cd", str(schema_path.parent),
    ]
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    started = time.time()
    proc = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", creationflags=creationflags,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(stdin, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_process_tree(proc)
        stdout, stderr = proc.communicate()
        stderr += f"\nrunner timeout after {timeout_seconds}s"
    elapsed = time.time() - started
    final, usage, _events = parse_codex_stream(stdout)
    saved, parse_errors = validate_partial(request, final)
    low = stderr.casefold()
    infrastructure_markers = (
        "invalid peer certificate", "unknownissuer", "error sending request", "authentication",
        "rate limit", "too many requests", "resource_exhausted", "quota", "not logged in",
    )
    infrastructure_failure = final is None and (timed_out or any(marker in low for marker in infrastructure_markers))
    return BatchResult(
        task=task, batch_id=batch_id, request=request, saved=saved, parse_errors=parse_errors,
        returncode=proc.returncode or 0, elapsed_seconds=elapsed, usage=usage, stderr=stderr[-6000:],
        final=final, infrastructure_failure=infrastructure_failure, timed_out=timed_out,
    )


def terminal_uids(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for _, row, _ in read_jsonl(path):
            uid = row.get("record_uid")
            if isinstance(uid, str):
                result.add(uid)
    return result


def next_resume_task_id(raw_path: Path) -> int:
    """Continue append-only batch numbering without reusing IDs after resume."""
    highest = 0
    if raw_path.exists():
        for _, row, _ in read_jsonl(raw_path):
            try:
                highest = max(highest, int(row.get("task_id") or 0))
            except (TypeError, ValueError):
                continue
    return highest + 1


def validate_item(
    request_item: TranslationInputItem,
    saved: dict[str, Any],
    batch_id: str,
) -> tuple[list[str], list[str]]:
    structural = list(saved.get("structural_errors") or [])
    if structural:
        return structural, []
    try:
        item = TranslationOutputItem.model_validate({
            key: saved.get(key) for key in ("seq", "record_uid", "prompt_vi", "response_vi", "warnings")
        })
        one_request = TranslationRequest(batch_id=batch_id, items=[request_item])
        one_response = TranslationResponse(batch_id=batch_id, items=[item])
        validate_response(one_request, one_response)
        validate_hard_quality(one_request, one_response)
        hard: list[str] = []
    except TranslationValidationError as exc:
        hard = list(exc.errors)
    except Exception as exc:
        hard = [f"{type(exc).__name__}: {exc}"]
    warnings = quality_warnings(
        {"prompt": request_item.prompt, "response": request_item.response},
        {"prompt_vi": saved.get("prompt_vi"), "response_vi": saved.get("response_vi")},
    )
    return hard, warnings


def usage_credits(
    usage: dict[str, Any] | None,
    input_credits_per_million: float = 5.0,
    cached_input_credits_per_million: float = 0.5,
    output_credits_per_million: float = 30.0,
) -> float:
    usage = usage or {}
    # cached_input_tokens is a subset of input_tokens, so subtract it before charging ordinary input.
    input_tokens = int(usage.get("input_tokens") or 0)
    cached_input_tokens = int(usage.get("cached_input_tokens") or 0)
    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    return (
        uncached_input_tokens * input_credits_per_million
        + cached_input_tokens * cached_input_credits_per_million
        + int(usage.get("output_tokens") or 0) * output_credits_per_million
    ) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel, resumable Luna translation supervisor.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=datetime.now().strftime("luna-%Y%m%d-%H%M%S"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--codex-path", type=Path, default=DEFAULT_CODEX)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--max-attempts", type=int,
        help="Maximum attempts per UID for every bucket; overrides the default bucket policy.",
    )
    parser.add_argument(
        "--record-uid", action="append", default=[],
        help="Run only the selected record UID; repeat the option to select multiple UIDs.",
    )
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-credits", type=float)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--infra-failure-limit", type=int, default=3)
    parser.add_argument("--input-credits-per-million", type=float, default=5.0)
    parser.add_argument("--cached-input-credits-per-million", type=float, default=0.5)
    parser.add_argument("--output-credits-per-million", type=float, default=30.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--prompt-prose", type=Path, default=DEFAULT_PROMPTS["prose"])
    parser.add_argument("--prompt-json", type=Path, default=DEFAULT_PROMPTS["json"])
    parser.add_argument("--prompt-leet", type=Path, default=DEFAULT_PROMPTS["leet"])
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise SystemExit("--workers must be between 1 and 32")
    if args.max_attempts is not None and args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "passed": out / "passed.jsonl",
        "audit": out / "needs_audit.jsonl",
        "exhausted": out / "exhausted_normal.jsonl",
        "tail": out / "tail_escalation.jsonl",
        "attempts": out / "attempts.jsonl",
        "raw": out / "raw_batches.jsonl",
        "summary": out / "summary.json",
        "manifest": out / "manifest.json",
    }
    schema_path = out / "translation_response.schema.json"
    schema_path.write_text(json.dumps(output_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_paths = {"prose": args.prompt_prose.resolve(), "json": args.prompt_json.resolve(), "leet": args.prompt_leet.resolve()}

    if paths["summary"].exists() and not args.no_resume:
        summary_dir = out / "summaries"
        summary_dir.mkdir(exist_ok=True)
        archived_summary = summary_dir / f"summary-before-resume-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        archived_summary.write_text(paths["summary"].read_text(encoding="utf-8"), encoding="utf-8")

    rows = [dict(row) for _, row, _ in read_jsonl(args.input)]
    if args.max_records is not None:
        rows = rows[: args.max_records]
    seen_source: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for seq, row in enumerate(rows, 1):
        uid = str(row.get("record_uid") or "")
        if not uid or uid in seen_source:
            raise SystemExit(f"missing or duplicate record_uid at source row {seq}: {uid!r}")
        seen_source.add(uid)
        row["_seq"] = seq
        normalized.append(row)
    if args.record_uid:
        requested_uids = set(args.record_uid)
        unknown_uids = requested_uids - seen_source
        if unknown_uids:
            raise SystemExit(f"--record-uid not found in input: {sorted(unknown_uids)!r}")
        # Filter only after assigning source-order sequence numbers so a focused escalation run
        # remains directly comparable with the original ablation and its web result.
        normalized = [row for row in normalized if row["record_uid"] in requested_uids]
    done = set() if args.no_resume else terminal_uids((paths["passed"], paths["audit"], paths["exhausted"], paths["tail"]))
    remaining = [row for row in normalized if row["record_uid"] not in done]
    first_task_id = 1 if args.no_resume else next_resume_task_id(paths["raw"])
    initial_tasks = make_initial_tasks(remaining, start_task_id=first_task_id)
    task_queue: deque[BatchTask] = deque(initial_tasks)
    next_task_id = max((task.task_id for task in initial_tasks), default=0) + 1
    attempts: Counter[str] = Counter()
    if not args.no_resume and paths["attempts"].exists():
        for _, row, _ in read_jsonl(paths["attempts"]):
            uid = row.get("record_uid")
            if isinstance(uid, str) and row.get("counts_toward_retry"):
                attempts[uid] += 1

    manifest = {
        "schema_version": 1, "run_id": args.run_id, "created_at": utc_now(),
        "input": str(args.input.resolve()), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_records": len(normalized), "resume_terminal_records": len(done), "remaining_records": len(remaining),
        "first_task_id": first_task_id,
        "workers": args.workers, "model": args.model, "reasoning_effort": args.reasoning_effort,
        "timeout_seconds": args.timeout_seconds, "max_requests": args.max_requests,
        "max_credits": args.max_credits, "max_hours": args.max_hours,
        "infra_failure_limit": args.infra_failure_limit,
        "credit_meter": {
            "input_credits_per_million": args.input_credits_per_million,
            "cached_input_credits_per_million": args.cached_input_credits_per_million,
            "output_credits_per_million": args.output_credits_per_million,
            "cached_input_policy": "cached_input_tokens is a subset of input_tokens and is charged separately",
        },
        "prompts": {
            route: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for route, path in prompt_paths.items()
        },
        "batch_rules": {f"{route}/{bucket}": {"max_items": rule[0], "max_chars": rule[1]} for (route, bucket), rule in BATCH_RULES.items()},
        "retry_policy": (
            {"all_buckets_total_attempts": args.max_attempts}
            if args.max_attempts is not None
            else {"normal_near_tail_total_attempts": 3, "tail_plus_total_attempts": 2}
        ),
        "leet_policy": "all structurally/hard-valid leet candidates require audit; never auto-pass",
    }
    if paths["manifest"].exists() and not args.no_resume:
        manifest_dir = out / "manifests"
        manifest_dir.mkdir(exist_ok=True)
        invocation_manifest = manifest_dir / f"manifest-resume-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        invocation_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = Counter({"source_records": len(normalized), "resume_terminal_records": len(done), "initial_tasks": len(initial_tasks)})
    credits = 0.0
    start = time.time()
    stop_reason: str | None = None
    active: dict[Future[BatchResult], BatchTask] = {}
    row_by_uid = {str(row["record_uid"]): row for row in normalized}

    def budget_allows_submission() -> bool:
        nonlocal stop_reason
        if stop_reason is not None:
            return False
        if args.max_requests is not None and stats["requests"] >= args.max_requests:
            stop_reason = "max_requests"
            return False
        if args.max_credits is not None and credits >= args.max_credits:
            stop_reason = "max_credits"
            return False
        if args.max_hours is not None and (time.time() - start) / 3600 >= args.max_hours:
            stop_reason = "max_hours"
            return False
        return True

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="luna") as pool:
        while task_queue or active:
            while task_queue and len(active) < args.workers and budget_allows_submission():
                task = task_queue.popleft()
                future = pool.submit(
                    run_batch, task, args.run_id, args.model, args.reasoning_effort, args.codex_path.resolve(),
                    schema_path, prompt_paths, args.timeout_seconds,
                )
                active[future] = task
                stats["requests"] += 1
            if not active:
                break
            completed_futures, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed_futures:
                task = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = BatchResult(
                        task, f"{args.run_id}-{task.task_id:06d}",
                        TranslationRequest(batch_id=f"{args.run_id}-{task.task_id:06d}", items=[]),
                        [], [f"runner exception: {type(exc).__name__}: {exc}"], 1, 0.0, None, str(exc), None, True, False,
                    )
                batch_credits = usage_credits(
                    result.usage, args.input_credits_per_million,
                    args.cached_input_credits_per_million, args.output_credits_per_million,
                )
                credits += batch_credits
                if result.infrastructure_failure:
                    stats["consecutive_infrastructure_failures"] += 1
                    if stats["consecutive_infrastructure_failures"] >= args.infra_failure_limit:
                        stop_reason = "infrastructure_circuit_breaker"
                else:
                    stats["consecutive_infrastructure_failures"] = 0
                append_jsonl(paths["raw"], {
                    "run_id": args.run_id, "batch_id": result.batch_id, "task_id": task.task_id,
                    "route": task.route, "bucket": task.bucket, "record_uids": [r["record_uid"] for r in task.rows],
                    "returncode": result.returncode, "elapsed_seconds": round(result.elapsed_seconds, 3),
                    "usage": result.usage, "estimated_credits": round(batch_credits, 6),
                    "credit_meter": manifest["credit_meter"],
                    "parse_errors": result.parse_errors, "infrastructure_failure": result.infrastructure_failure,
                    "timed_out": result.timed_out, "stderr": result.stderr, "raw_final": result.final,
                    "completed_at": utc_now(),
                })
                expected = {item.record_uid: item for item in result.request.items}
                candidates = {str(item.get("record_uid")): item for item in result.saved if item.get("record_uid")}
                retry_rows: list[dict[str, Any]] = []
                for row in task.rows:
                    uid = str(row["record_uid"])
                    candidate = candidates.get(uid)
                    # A missing/empty item is an output failure for this UID even when the
                    # surrounding request was classified as infrastructure-failed. Count it
                    # toward the per-UID retry budget so a high-tail record cannot rotate
                    # forever without ever reaching retry 3.
                    counts_toward_retry = (not result.infrastructure_failure) or candidate is None
                    if counts_toward_retry:
                        attempts[uid] += 1
                    hard: list[str] = []
                    warnings: list[str] = []
                    if candidate is None:
                        hard = ["missing item in final response"]
                    elif uid not in expected:
                        hard = ["unknown item returned"]
                    else:
                        hard, warnings = validate_item(expected[uid], candidate, result.batch_id)
                    attempt_record = {
                        "run_id": args.run_id, "record_uid": uid, "batch_id": result.batch_id,
                        "route": task.route, "length_bucket": task.bucket, "attempt": attempts[uid],
                        "counts_toward_retry": counts_toward_retry, "infrastructure_failure": result.infrastructure_failure,
                        "hard_errors": hard, "heuristic_warnings": warnings, "candidate": candidate,
                        "attempted_at": utc_now(),
                    }
                    append_jsonl(paths["attempts"], attempt_record)
                    base = {
                        "record_uid": uid, "seq": int(row["_seq"]), "route": task.route,
                        "length_bucket": task.bucket, "attempts": attempts[uid], "batch_id": result.batch_id,
                        "model": args.model, "reasoning_effort": args.reasoning_effort,
                        "prompt_sha256": manifest["prompts"][task.route]["sha256"],
                        "source_text_sha256": hashlib.sha256((source_fields(row)[0] + "\n" + (source_fields(row)[1] or "")).encode("utf-8")).hexdigest(),
                        "hard_errors": hard, "heuristic_warnings": warnings, "candidate": candidate,
                        "recorded_at": utc_now(),
                    }
                    if candidate is not None and not hard:
                        if task.route == "leet" or warnings:
                            base["translation_status"] = "needs_audit"
                            base["audit_reason"] = "leet_requires_manual_audit" if task.route == "leet" else "heuristic_warning"
                            append_jsonl(paths["audit"], base)
                            stats["audit"] += 1
                        else:
                            base["translation_status"] = "passed"
                            append_jsonl(paths["passed"], base)
                            stats["passed"] += 1
                        continue
                    if attempts[uid] < max_attempts(task.bucket, args.max_attempts) and (
                        stop_reason != "infrastructure_circuit_breaker" or candidate is None
                    ):
                        retry_rows.append(row_by_uid[uid])
                        stats["requeued"] += 1
                    elif result.infrastructure_failure:
                        # Keep infrastructure-stopped records pending rather than falsely classifying them as
                        # model failures. They will be selected again by --resume after connectivity recovers.
                        stats["deferred_infrastructure_records"] += 1
                    else:
                        if task.bucket in {"tail", "high_tail", "oversized"}:
                            base["translation_status"] = "tail_escalation"
                            append_jsonl(paths["tail"], base)
                            stats["tail_escalation"] += 1
                        else:
                            base["translation_status"] = "exhausted_normal"
                            append_jsonl(paths["exhausted"], base)
                            stats["exhausted_normal"] += 1
                if retry_rows:
                    # Appending the failed subset implements queue rotation: successful records are already
                    # durable; failures go behind all currently pending normal work instead of immediate retry.
                    task_queue.append(BatchTask(next_task_id, tuple(retry_rows), task.route, task.bucket))
                    next_task_id += 1
                print(safe_json_dumps({
                    "batch_id": result.batch_id, "route": task.route, "bucket": task.bucket,
                    "requested": len(task.rows), "saved": len(result.saved), "requeued": len(retry_rows),
                    "passed_total": stats["passed"], "audit_total": stats["audit"],
                    "credits": round(credits, 4), "pending_tasks": len(task_queue), "active": len(active),
                }), flush=True)
            if stop_reason and not active:
                break

    summary = {
        **dict(stats), "run_id": args.run_id, "estimated_credits": round(credits, 6),
        "elapsed_seconds": round(time.time() - start, 3), "stop_reason": stop_reason,
        "pending_tasks": len(task_queue), "pending_records": sum(len(task.rows) for task in task_queue),
        "completed_at": utc_now(),
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
