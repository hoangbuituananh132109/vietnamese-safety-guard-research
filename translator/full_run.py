from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from translator.checkpoint import load_checkpoint
from translator.jsonl_io import read_jsonl, write_jsonl
from translator.pipeline import TranslationPipeline
from translator.providers import GeminiProvider


SPLITS = ("train", "valid", "test")
MODEL = "gemini-3.1-flash-lite"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def paths(root: Path, split: str, group: int) -> dict[str, Path]:
    return {
        "input": root / "data" / "prepared" / f"nemotron_en_{split}_full_v1.jsonl",
        "checkpoint": root / "data" / "checkpoints" / f"nemotron_{split}_full_v10_g{group + 1}_completed.jsonl",
        "output": root / "data" / "translated" / f"nemotron_{split}_full_vi_v10_g{group + 1}.jsonl",
        "failed": root / "data" / "review_queue" / f"nemotron_{split}_full_v10_g{group + 1}_failed.jsonl",
    }


def expected_shard_size(input_path: Path, group: int, group_count: int) -> int:
    total = sum(1 for _ in read_jsonl(input_path))
    return (total + group_count - 1 - group) // group_count


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def live_progress(root: Path, group_count: int) -> dict[str, Any]:
    progress: dict[str, Any] = {}
    for split in SPLITS:
        input_path = paths(root, split, 0)["input"]
        total = count_nonempty_lines(input_path)
        groups = {
            str(group + 1): count_nonempty_lines(paths(root, split, group)["checkpoint"])
            for group in range(group_count)
        }
        completed = sum(groups.values())
        progress[split] = {
            "completed": completed, "total": total,
            "missing": total - completed, "groups": groups,
        }
    return progress


def worker(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    state_path = root / "data" / "run" / f"worker_{args.group_index + 1}.json"
    slots = [int(value) for value in args.key_slots.split(",")]
    reserve_slots = [int(value) for value in args.reserve_key_slots.split(",")] if args.reserve_key_slots else []
    provider = GeminiProvider(
        args.model, root / "API.txt", key_slots=slots,
        reserve_key_slots=reserve_slots,
        min_request_interval_seconds=args.request_interval_seconds,
        event_log_path=root / "data" / "run" / f"key_events_g{args.group_index + 1}.jsonl",
    )
    state: dict[str, Any] = {
        "updated_at": now(), "pid": os.getpid(), "group": args.group_index + 1,
        "key_slots": slots, "reserve_key_slots": reserve_slots,
        "status": "running", "splits": {},
    }
    write_state(state_path, state)
    exit_code = 0
    try:
        for split in SPLITS:
            p = paths(root, split, args.group_index)
            expected = expected_shard_size(p["input"], args.group_index, args.group_count)
            quota_cooldowns = 0
            while True:
                pipeline = TranslationPipeline(
                    provider, p["checkpoint"], p["failed"],
                    max_transient_retries=args.max_retries,
                    max_api_requests=args.max_api_requests,
                )
                stats = pipeline.run(
                    p["input"], p["output"], resume=True,
                    shard_index=args.group_index, shard_count=args.group_count,
                    batch_id_prefix=f"g{args.group_index + 1}-",
                )
                stats.update({
                    "expected_records": expected,
                    "provider_calls_total": provider.calls,
                    "quota_cooldowns": quota_cooldowns,
                })
                state["splits"][split] = stats
                state["updated_at"] = now()
                write_state(state_path, state)
                if not stats["deferred"]:
                    state.pop("reason", None)
                    break

                reason = stats["deferred_reason"] or ""
                is_quota = "429" in reason or "RESOURCE_EXHAUSTED" in reason.upper()
                can_cooldown = (
                    is_quota
                    and args.max_api_requests is None
                    and quota_cooldowns < args.max_quota_cooldowns
                )
                if not can_cooldown:
                    state["status"] = "deferred"
                    state["reason"] = reason
                    exit_code = 3
                    break

                quota_cooldowns += 1
                cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=args.quota_cooldown_seconds)
                state.update({
                    "status": "quota_cooldown",
                    "reason": reason,
                    "quota_cooldown_number": quota_cooldowns,
                    "quota_cooldown_seconds": args.quota_cooldown_seconds,
                    "cooldown_until": cooldown_until.isoformat(),
                    "updated_at": now(),
                })
                write_state(state_path, state)
                time.sleep(args.quota_cooldown_seconds)
                state["status"] = "running"
                state["updated_at"] = now()
                write_state(state_path, state)
            if exit_code:
                break
        else:
            state["status"] = "complete"
    except Exception as exc:
        state["status"] = "error"
        state["reason"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
        exit_code = 1
    state.update({
        "updated_at": now(), "provider_calls": provider.calls,
        "key_slots_active": provider.keys.active_size,
        "key_slots_disabled": provider.keys.disabled_size,
        "reserve_active": provider.keys.reserve_active,
    })
    write_state(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return exit_code


def merge(root: Path, group_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        completed: dict[str, dict[str, Any]] = {}
        for group in range(group_count):
            checkpoint = paths(root, split, group)["checkpoint"]
            for uid, row in load_checkpoint(checkpoint).items():
                if uid in completed and completed[uid] != row:
                    raise ValueError(f"Conflicting translated record across groups: {uid}")
                completed[uid] = row
        source_path = paths(root, split, 0)["input"]
        merged_rows = []
        total = 0
        for _, source, _ in read_jsonl(source_path):
            total += 1
            translated = completed.get(source["record_uid"])
            if translated is None:
                continue
            final = dict(source)
            final.update({
                "prompt_en": source.get("prompt"), "response_en": source.get("response"),
                "translation_language": "vi",
            })
            final.update(translated)
            merged_rows.append(final)
        output = root / "data" / "translated" / f"nemotron_{split}_full_vi_v10.jsonl"
        write_jsonl(output, merged_rows)
        result[split] = {
            "completed": len(merged_rows), "total": total,
            "missing": total - len(merged_rows), "output": str(output),
        }
    return result


def supervisor(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    run_dir = root / "data" / "run"
    log_dir = root / "reports" / "full_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    key_count = sum(1 for line in (root / "API.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip())
    required = args.group_count * args.keys_per_group
    if key_count < required:
        raise SystemExit(f"Need {required} non-empty API keys, found {key_count}")

    processes: list[tuple[int, subprocess.Popen[Any], Any]] = []
    for group in range(args.group_count):
        first = group * args.keys_per_group + 1
        slots = ",".join(str(slot) for slot in range(first, first + args.keys_per_group))
        log_handle = (log_dir / f"worker_{group + 1}.log").open("a", encoding="utf-8")
        command = [
            sys.executable, "-m", "translator.full_run", "worker",
            "--root", str(root), "--group-index", str(group),
            "--group-count", str(args.group_count), "--key-slots", slots,
            "--model", args.model, "--max-retries", str(args.max_retries),
        ]
        reserve_slot = required + group + 1
        if reserve_slot <= key_count:
            command += ["--reserve-key-slots", str(reserve_slot)]
        if args.max_api_requests is not None:
            command += ["--max-api-requests", str(args.max_api_requests)]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            command, cwd=root, stdout=log_handle, stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        processes.append((group, process, log_handle))
        # Avoid sending five first requests in the same instant. Workers still
        # overlap after this small startup stagger.
        if group + 1 < args.group_count:
            time.sleep(2)

    state_path = run_dir / "runner_state.json"
    write_state(state_path, {
        "updated_at": now(), "pid": os.getpid(), "status": "running",
        "model": args.model, "groups": args.group_count,
        "worker_pids": {str(group + 1): process.pid for group, process, _ in processes},
    })
    exit_codes: dict[str, int] = {}
    while processes:
        remaining = []
        for group, process, handle in processes:
            code = process.poll()
            if code is None:
                remaining.append((group, process, handle))
            else:
                handle.close()
                exit_codes[str(group + 1)] = code
        processes = remaining
        if processes:
            write_state(state_path, {
                "updated_at": now(), "pid": os.getpid(), "status": "running",
                "model": args.model, "groups": args.group_count,
                "active_worker_pids": {
                    str(group + 1): process.pid for group, process, _ in processes
                },
                "worker_exit_codes": exit_codes,
                "progress": live_progress(root, args.group_count),
            })
            time.sleep(10)

    merged = merge(root, args.group_count)
    complete = all(item["missing"] == 0 for item in merged.values())
    status = "complete" if complete else "stopped_incomplete"
    write_state(state_path, {
        "updated_at": now(), "pid": os.getpid(), "status": status,
        "complete": complete, "worker_exit_codes": exit_codes, "progress": merged,
    })
    return 0 if complete else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", required=True)
    common.add_argument("--model", default=MODEL)
    common.add_argument("--group-count", type=int, default=5)
    common.add_argument("--max-retries", type=int, default=4)
    common.add_argument("--max-api-requests", type=int)
    common.add_argument("--quota-cooldown-seconds", type=int, default=60)
    common.add_argument("--max-quota-cooldowns", type=int, default=5)
    common.add_argument("--request-interval-seconds", type=float, default=60.0)
    w = sub.add_parser("worker", parents=[common])
    w.add_argument("--group-index", type=int, required=True)
    w.add_argument("--key-slots", required=True)
    w.add_argument("--reserve-key-slots")
    w.set_defaults(func=worker)
    s = sub.add_parser("supervisor", parents=[common])
    s.add_argument("--keys-per-group", type=int, default=2)
    s.set_defaults(func=supervisor)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
