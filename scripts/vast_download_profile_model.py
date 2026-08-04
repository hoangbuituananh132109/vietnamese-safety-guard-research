#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from huggingface_hub import snapshot_download


MODELS = {
    "nemotron": {
        "repo_id": "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3",
        "revision": "8fdc246ba3d56db9c469d534233b9f582d3afafa",
        "local_dir": "/workspace/safety-dataset/models/nemotron-v3",
    },
    "qwen": {
        "repo_id": "Qwen/Qwen3Guard-Gen-4B",
        "revision": "6ec42827da0c1ff11e7a49dc269d2e810d27e108",
        "local_dir": "/workspace/safety-dataset/models/qwen3guard-gen-4b",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(MODELS))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    args = parser.parse_args()

    spec = MODELS[args.name]
    output = Path("/workspace/safety-dataset/results") / f"download_{args.name}.json"
    started = time.monotonic()
    attempts: list[dict[str, object]] = []
    path: str | None = None
    for attempt in range(1, args.retries + 1):
        attempt_started = time.monotonic()
        try:
            path = snapshot_download(
                repo_id=spec["repo_id"],
                revision=spec["revision"],
                local_dir=spec["local_dir"],
                max_workers=args.max_workers,
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "completed",
                    "elapsed_seconds": time.monotonic() - attempt_started,
                }
            )
            break
        except Exception as error:
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "retry",
                    "elapsed_seconds": time.monotonic() - attempt_started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(
                json.dumps(attempts[-1], ensure_ascii=False),
                flush=True,
            )
            traceback.print_exc()
            if attempt == args.retries:
                raise
            time.sleep(args.retry_delay)
    if path is None:
        raise AssertionError("snapshot_download returned no path")
    elapsed = time.monotonic() - started
    files = [item for item in Path(path).rglob("*") if item.is_file()]
    total_bytes = sum(item.stat().st_size for item in files)
    report = {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "local_dir": path,
        "files": len(files),
        "bytes": total_bytes,
        "elapsed_seconds": elapsed,
        "effective_mib_per_second": total_bytes / (1024 * 1024) / elapsed,
        "attempts": attempts,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
