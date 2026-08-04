from __future__ import annotations

import argparse
import json
from pathlib import Path

from translator.pipeline import TranslationPipeline
from translator.providers import DryRunProvider, GeminiProvider
from translator.reporting import build_review_html, build_summary, write_summary


def translate(args: argparse.Namespace) -> int:
    if args.provider == "gemini":
        if not args.confirm_real_api:
            raise SystemExit("Refusing real API call: add --confirm-real-api")
        if not Path(args.api_key_file).exists():
            raise SystemExit(f"API key file not found: {args.api_key_file}")
        key_slots = [int(value) for value in args.api_key_slots.split(",")] if args.api_key_slots else None
        reserve_slots = [int(value) for value in args.reserve_key_slots.split(",")] if args.reserve_key_slots else None
        provider = GeminiProvider(args.model, args.api_key_file, key_slots=key_slots, reserve_key_slots=reserve_slots)
        print(f"Gemini provider ready: model={args.model}; key_slots={provider.keys.size}")
    else:
        provider = DryRunProvider(failure_mode=args.dry_failure_mode)
    pipeline = TranslationPipeline(
        provider, args.checkpoint, args.failed_output, args.max_retries,
        max_api_requests=args.max_api_requests,
    )
    stats = pipeline.run(
        args.input, args.output, limit=args.limit, resume=args.resume,
        shard_index=args.shard_index, shard_count=args.shard_count,
    )
    if args.provider == "gemini":
        stats.update({
            "provider_calls": provider.calls,
            "key_slots_total": provider.keys.size,
            "key_slots_active": provider.keys.active_size,
            "key_slots_disabled": provider.keys.disabled_size,
            "reserve_active": provider.keys.reserve_active,
        })
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if stats["failed"] == 0 else 2


def report(args: argparse.Namespace) -> int:
    summary = build_summary(args.source, args.translated)
    write_summary(args.summary, summary)
    build_review_html(args.translated, args.html)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def validate(args: argparse.Namespace) -> int:
    summary = build_summary(args.source, args.translated)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["missing_records"] == 0 else 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m translator.cli")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("translate")
    t.add_argument("--input", required=True); t.add_argument("--output", required=True)
    t.add_argument("--checkpoint", required=True); t.add_argument("--failed-output", required=True)
    t.add_argument("--provider", choices=["dry-run", "gemini"], default="dry-run")
    t.add_argument("--model", default="gemini-3.1-flash-lite"); t.add_argument("--api-key-file", default="API.txt")
    t.add_argument("--api-key-slots", help="Comma-separated one-based API.txt slots, e.g. 1,2")
    t.add_argument("--reserve-key-slots", help="Fallback slots activated after repeated 429 errors")
    t.add_argument("--shard-index", type=int, default=0); t.add_argument("--shard-count", type=int, default=1)
    t.add_argument("--limit", type=int); t.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--confirm-real-api", action="store_true"); t.add_argument("--max-retries", type=int, default=4)
    t.add_argument("--max-api-requests", type=int, help="Stop cleanly after this many provider requests; resume later")
    t.add_argument("--dry-failure-mode", choices=["missing", "duplicate"]); t.set_defaults(func=translate)
    for name, func in (("validate", validate), ("report", report)):
        q = sub.add_parser(name); q.add_argument("--source", required=True); q.add_argument("--translated", required=True)
        if name == "report":
            q.add_argument("--summary", required=True); q.add_argument("--html", required=True)
        q.set_defaults(func=func)
    return p


def main() -> None:
    args = parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
