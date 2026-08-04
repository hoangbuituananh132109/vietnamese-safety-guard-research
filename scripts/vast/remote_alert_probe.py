from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit one compact Vast training health probe")
    parser.add_argument("--root", type=Path, default=Path("/workspace/safety-dataset"))
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("reports/overnight_phase0/state.json"),
    )
    parser.add_argument("--tmux", default="phase0_full")
    parser.add_argument("--runner-pattern", default="[r]un_overnight_phase0.py")
    return parser.parse_args()


def process_exists(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    state_path = args.state if args.state.is_absolute() else root / args.state
    state_path = state_path.resolve()
    state_path.relative_to(root)

    now = datetime.now(timezone.utc)
    state: dict[str, Any] = {}
    state_error = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:  # surfaced to the local monitor as a health failure
            state_error = f"{type(exc).__name__}: {exc}"

    updated_at = parse_timestamp(state.get("updated_at"))
    updated_age_seconds = (now - updated_at).total_seconds() if updated_at else None
    tmux_active = subprocess.run(
        ["tmux", "has-session", "-t", args.tmux],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    runner_probe = subprocess.run(
        ["pgrep", "-af", args.runner_pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    runner_lines = [line for line in runner_probe.stdout.splitlines() if line.strip()]
    child_pid = state.get("child_pid")

    payload = {
        "probe_status": "ok",
        "remote_time": now.isoformat(),
        "state_path": str(state_path),
        "state_exists": state_path.exists(),
        "state_read_error": state_error,
        "state_status": state.get("status"),
        "current_stage": state.get("current_stage"),
        "current_run_id": state.get("current_run_id"),
        "stage_index": state.get("stage_index"),
        "stage_count": state.get("stage_count"),
        "stage_elapsed_seconds": state.get("stage_elapsed_seconds"),
        "updated_at": state.get("updated_at"),
        "updated_age_seconds": updated_age_seconds,
        "error": state.get("error"),
        "failed_at": state.get("failed_at"),
        "completed_at": state.get("completed_at"),
        "tmux_session": args.tmux,
        "tmux_active": tmux_active,
        "runner_active": bool(runner_lines),
        "runner_processes": runner_lines,
        "child_pid": child_pid,
        "child_active": process_exists(child_pid),
    }
    print("ALERT_STATE=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
