#!/usr/bin/env python3
"""Package, hand off, verify, and stop the active no-R D3 Vast run.

This process is deliberately separate from the training coordinator.  It can be
installed while training is active without restarting or signalling the trainer.

The training coordinator currently schedules a delayed Vast stop after a terminal
state.  Once the coordinator has exited and recorded that delayed stop, this
guardian changes the pipeline state away from the literal values "completed" and
"failed".  That disarms the old delayed command before it packages the run.

The local Windows watcher downloads the archive and writes a checksum-bearing ACK
back to ACK_PATH.  A verified ACK stops the instance immediately.  If the local
machine is unavailable, the guardian stops the instance after the handoff timeout;
Vast stop preserves the container filesystem and archive, while halting GPU
charges.  It never destroys or recycles the instance.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import time
import traceback
from typing import Any, Iterable


ROOT = Path("/workspace/safety-dataset")
RESULTS = ROOT / "results/no_r_decoder_4080s"
PIPELINE_STATE = RESULTS / "pipeline_state.json"
EXPORTS = ROOT / "exports"
POSTRUN = RESULTS / "postrun"
READY_PATH = EXPORTS / "D3_NEMOTRON_NO_R_4080S_20260724.ready.json"
ACK_PATH = EXPORTS / "D3_NEMOTRON_NO_R_4080S_20260724.download_verified.json"
ARCHIVE_PATH = EXPORTS / "D3_NEMOTRON_NO_R_4080S_20260724.tar.zst"
GUARDIAN_STATE = POSTRUN / "guardian_state.json"
GUARDIAN_LOG = POSTRUN / "guardian.log"
COMPLETED_MARKER = POSTRUN / "guardian_stop_intent.marker.json"
PIPELINE_PROGRAM = "no-r-decoder-pipeline"
POLL_SECONDS = 2
STALE_HEARTBEAT_SECONDS = 240
HANDOFF_TIMEOUT_SECONDS = 30 * 60


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str, **fields: Any) -> None:
    POSTRUN.mkdir(parents=True, exist_ok=True)
    record = {"at": now(), "message": message, **fields}
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    with GUARDIAN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def update_pipeline_state(pipeline_status: str, **postrun: Any) -> None:
    state = read_json(PIPELINE_STATE)
    state["status"] = pipeline_status
    state["current_stage"] = None
    state["updated_at"] = now()
    existing = state.get("postrun")
    if not isinstance(existing, dict):
        existing = {}
    existing.update(postrun)
    existing["updated_at"] = now()
    state["postrun"] = existing
    atomic_json(PIPELINE_STATE, state)


def supervisor_state() -> str:
    result = subprocess.run(
        ["supervisorctl", "status", PIPELINE_PROGRAM],
        capture_output=True,
        text=True,
        check=False,
    )
    words = result.stdout.strip().split()
    return words[1] if len(words) >= 2 else "UNKNOWN"


def cancel_coordinator_delayed_stops() -> list[int]:
    """Terminate only the exact delayed stop sessions created by the coordinator.

    The legacy coordinator uses grep against the entire JSON document, so changing
    only the top-level status is insufficient: completed nested stages also match.
    Its subprocess is a new session containing the fixed pipeline-state path and
    the literal Vast stop command.  Matching both strings keeps this cancellation
    narrowly scoped.
    """

    state_needle = str(PIPELINE_STATE).encode("utf-8")
    # Quote characters are not stable in /proc/<pid>/cmdline after the command
    # crosses an SSH shell boundary, so match the fixed verb phrase plus the exact
    # state path rather than the quoted variable spelling.
    stop_needle = b"vastai stop instance"
    killed: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            command_line = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if state_needle not in command_line or stop_needle not in command_line:
            continue
        try:
            process_group = os.getpgid(pid)
            os.killpg(process_group, signal.SIGTERM)
            killed.append(pid)
            log(
                "legacy_delayed_stop_cancelled",
                pid=pid,
                process_group=process_group,
            )
        except (ProcessLookupError, PermissionError, OSError) as exc:
            log(
                "legacy_delayed_stop_cancel_failed",
                pid=pid,
                error=str(exc),
            )
    return killed


def seconds_since_iso(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - parsed).total_seconds()
    except ValueError:
        return None


def wait_for_terminal_pipeline() -> str:
    log("guardian_waiting_for_terminal_pipeline")
    while True:
        state = read_json(PIPELINE_STATE)
        status = str(state.get("status", "unknown"))
        program_state = supervisor_state()

        # Wait until the coordinator has fully exited so it cannot overwrite the
        # state after the guardian disarms its delayed stop.
        if (
            status in {"completed", "failed"}
            and isinstance(state.get("auto_stop"), dict)
            and program_state != "RUNNING"
        ):
            log(
                "terminal_pipeline_observed",
                terminal_status=status,
                supervisor_state=program_state,
            )
            return status

        # If the process disappeared without reaching its exception handler,
        # preserve its latest checkpoint and logs as an interrupted failure.
        heartbeat_age = seconds_since_iso(state.get("heartbeat_at"))
        if (
            status == "running"
            and program_state not in {"RUNNING", "STARTING"}
            and heartbeat_age is not None
            and heartbeat_age > STALE_HEARTBEAT_SECONDS
        ):
            log(
                "stale_pipeline_detected",
                supervisor_state=program_state,
                heartbeat_age_seconds=heartbeat_age,
            )
            return "interrupted"

        atomic_json(
            GUARDIAN_STATE,
            {
                "status": "waiting_for_pipeline",
                "pipeline_status": status,
                "pipeline_supervisor_state": program_state,
                "heartbeat_age_seconds": heartbeat_age,
                "updated_at": now(),
            },
        )
        time.sleep(POLL_SECONDS)


def run_capture(command: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    output.write_text(
        "$ "
        + shlex.join(command)
        + "\n\nSTDOUT\n"
        + result.stdout
        + "\nSTDERR\n"
        + result.stderr
        + f"\nRETURN_CODE={result.returncode}\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if (
                    child.is_file()
                    and "__pycache__" not in child.parts
                    and child.suffix != ".pyc"
                ):
                    yield child


def capture_reproducibility() -> None:
    POSTRUN.mkdir(parents=True, exist_ok=True)
    commands = [
        (
            ["/venv/main/bin/python", "-m", "pip", "freeze"],
            POSTRUN / "requirements-train-venv.txt",
        ),
        (
            [
                "/workspace/venvs/nemotron-vllm/bin/python",
                "-m",
                "pip",
                "freeze",
            ],
            POSTRUN / "requirements-eval-venv.txt",
        ),
        (["nvidia-smi", "-q"], POSTRUN / "nvidia-smi-q.txt"),
        (
            [
                "/venv/main/bin/python",
                "-c",
                (
                    "import json,platform,torch,transformers,peft;"
                    "print(json.dumps({"
                    "'python':platform.python_version(),"
                    "'torch':torch.__version__,"
                    "'torch_cuda':torch.version.cuda,"
                    "'transformers':transformers.__version__,"
                    "'peft':peft.__version__},indent=2))"
                ),
            ],
            POSTRUN / "core-package-versions.txt",
        ),
        (["git", "rev-parse", "HEAD"], POSTRUN / "git-head.txt"),
        (["git", "status", "--short"], POSTRUN / "git-status-short.txt"),
    ]
    for command, output in commands:
        try:
            run_capture(command, output)
        except Exception:
            output.write_text(traceback.format_exc(), encoding="utf-8")

    dataset_inputs = [
        ROOT / "data/no_r/full/train.jsonl",
        ROOT / "data/no_r/decoder/test_and_sea.jsonl",
    ]
    dataset_inventory: list[dict[str, Any]] = []
    for path in dataset_inputs:
        if path.exists():
            with path.open("rb") as handle:
                rows = sum(1 for _ in handle)
            dataset_inventory.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "rows": rows,
                    "sha256": file_sha256(path),
                }
            )
    atomic_json(
        POSTRUN / "dataset_inventory.json",
        {"generated_at": now(), "files": dataset_inventory},
    )

    selected = [
        RESULTS,
        ROOT / "logs/no_r_decoder_4080s",
        ROOT / "logs/no-r-decoder-pipeline.log",
        ROOT / "logs/no-r-decoder-pipeline.err.log",
        ROOT / "scripts",
        ROOT / "guard_smoke",
        ROOT / "configs",
        ROOT / "results/download_qwen.json",
        ROOT / "results/download_nemotron.json",
    ]
    inventory_lines: list[str] = []
    for path in iter_files(selected):
        # guardian.log remains live while archive creation is being reported, so
        # it is intentionally archived but excluded from the immutable per-file
        # checksum inventory. The outer archive SHA-256 still protects it.
        if path == GUARDIAN_LOG:
            continue
        # The archive checksum protects the transfer.  This per-file inventory is
        # primarily for auditability of model adapters, predictions, and code.
        relative = path.relative_to(ROOT)
        inventory_lines.append(
            f"{file_sha256(path)}  {relative.as_posix()}"
        )
    (POSTRUN / "file_inventory.sha256").write_text(
        "\n".join(inventory_lines) + "\n",
        encoding="utf-8",
    )


def create_archive(terminal_status: str) -> dict[str, Any]:
    update_pipeline_state(
        "postrun_packaging",
        status="packaging",
        terminal_pipeline_status=terminal_status,
        started_at=now(),
        guardian="no_r_d3_postrun_guardian.py",
    )
    atomic_json(
        GUARDIAN_STATE,
        {
            "status": "packaging",
            "terminal_pipeline_status": terminal_status,
            "updated_at": now(),
        },
    )
    capture_reproducibility()
    EXPORTS.mkdir(parents=True, exist_ok=True)

    archive_items = [
        "results/no_r_decoder_4080s",
        "results/download_qwen.json",
        "results/download_nemotron.json",
        "logs/no_r_decoder_4080s",
        "logs/no-r-decoder-pipeline.log",
        "logs/no-r-decoder-pipeline.err.log",
        "scripts",
        "guard_smoke",
        "configs",
    ]
    archive_items = [item for item in archive_items if (ROOT / item).exists()]
    command = [
        "tar",
        "--zstd",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "-cf",
        str(ARCHIVE_PATH),
        *archive_items,
    ]
    log("archive_started", command=shlex.join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    archive_sha256 = file_sha256(ARCHIVE_PATH)
    ready = {
        "schema_version": 1,
        "status": "ready_for_download",
        "terminal_pipeline_status": terminal_status,
        "archive_path": str(ARCHIVE_PATH),
        "archive_name": ARCHIVE_PATH.name,
        "archive_bytes": ARCHIVE_PATH.stat().st_size,
        "archive_sha256": archive_sha256,
        "created_at": now(),
        "handoff_timeout_seconds": HANDOFF_TIMEOUT_SECONDS,
        "stop_policy": (
            "stop after verified local ACK; otherwise stop at timeout; "
            "never destroy or recycle"
        ),
    }
    atomic_json(READY_PATH, ready)
    atomic_json(
        GUARDIAN_STATE,
        {
            **ready,
            "status": "waiting_for_local_download_ack",
            "ready_path": str(READY_PATH),
            "updated_at": now(),
        },
    )
    update_pipeline_state(
        "postrun_ready_for_download",
        status="ready_for_download",
        ready_path=str(READY_PATH),
        archive_path=str(ARCHIVE_PATH),
        archive_bytes=ready["archive_bytes"],
        archive_sha256=archive_sha256,
        ready_at=now(),
    )
    log(
        "archive_ready",
        archive=str(ARCHIVE_PATH),
        bytes=ready["archive_bytes"],
        sha256=archive_sha256,
    )
    return ready


def verified_ack(ready: dict[str, Any]) -> dict[str, Any] | None:
    ack = read_json(ACK_PATH)
    if (
        ack.get("status") == "verified"
        and ack.get("archive_sha256") == ready.get("archive_sha256")
        and int(ack.get("archive_bytes", -1)) == int(ready.get("archive_bytes", -2))
    ):
        return ack
    return None


def stop_instance(reason: str, ready: dict[str, Any]) -> None:
    update_pipeline_state(
        "postrun_stop_requested",
        status="stop_requested",
        stop_reason=reason,
        stop_requested_at=now(),
        archive_path=str(ARCHIVE_PATH),
        archive_sha256=ready["archive_sha256"],
    )
    atomic_json(
        GUARDIAN_STATE,
        {
            "status": "stop_requested",
            "reason": reason,
            "archive_path": str(ARCHIVE_PATH),
            "archive_sha256": ready["archive_sha256"],
            "updated_at": now(),
        },
    )
    container_id = os.environ.get("CONTAINER_ID")
    api_key = os.environ.get("CONTAINER_API_KEY")
    if not container_id or not api_key:
        raise RuntimeError(
            "Vast CONTAINER_ID or CONTAINER_API_KEY is unavailable; "
            "archive is preserved and the instance was not stopped"
        )
    atomic_json(
        COMPLETED_MARKER,
        {
            "status": "stop_intent_recorded",
            "reason": reason,
            "container_id": container_id,
            "archive_path": str(ARCHIVE_PATH),
            "archive_sha256": ready["archive_sha256"],
            "recorded_at": now(),
            "restart_policy": (
                "guardian exits without another stop request when this marker exists"
            ),
        },
    )
    log("vast_stop_requested", reason=reason, container_id=container_id)
    # Never log or persist api_key.
    result = subprocess.run(
        [
            "vastai",
            "stop",
            "instance",
            container_id,
            "--api-key",
            api_key,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    log(
        "vast_stop_command_returned",
        returncode=result.returncode,
        stdout=result.stdout[-1000:],
        stderr=result.stderr[-1000:],
    )
    if result.returncode != 0:
        COMPLETED_MARKER.unlink(missing_ok=True)
        raise RuntimeError(f"vastai stop returned {result.returncode}")


def wait_for_ack_or_timeout(ready: dict[str, Any]) -> str:
    deadline = time.monotonic() + HANDOFF_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        ack = verified_ack(ready)
        if ack is not None:
            update_pipeline_state(
                "postrun_download_verified",
                status="download_verified",
                verified_at=ack.get("verified_at", now()),
                local_path=ack.get("local_path"),
                archive_sha256=ready["archive_sha256"],
            )
            atomic_json(
                GUARDIAN_STATE,
                {
                    "status": "download_verified",
                    "ack": ack,
                    "updated_at": now(),
                },
            )
            log(
                "local_download_verified",
                local_path=ack.get("local_path"),
                sha256=ready["archive_sha256"],
            )
            return "verified_local_download"
        time.sleep(POLL_SECONDS)
    log(
        "local_ack_timeout",
        timeout_seconds=HANDOFF_TIMEOUT_SECONDS,
        archive=str(ARCHIVE_PATH),
    )
    return "local_ack_timeout_archive_preserved_on_stopped_instance"


def main() -> None:
    POSTRUN.mkdir(parents=True, exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)
    if COMPLETED_MARKER.exists():
        log(
            "guardian_already_completed_no_repeated_stop",
            marker=str(COMPLETED_MARKER),
        )
        return
    if READY_PATH.exists() and ARCHIVE_PATH.exists():
        ready = read_json(READY_PATH)
        if ready.get("archive_sha256") == file_sha256(ARCHIVE_PATH):
            log("reusing_existing_verified_archive", archive=str(ARCHIVE_PATH))
        else:
            READY_PATH.unlink(missing_ok=True)
            ready = {}
    else:
        ready = {}

    if not ready:
        terminal_status = wait_for_terminal_pipeline()
        cancelled = cancel_coordinator_delayed_stops()
        # The currently active coordinator predates this guardian and is expected
        # to leave exactly one delayed stop session. Future patched coordinators
        # delegate stopping and therefore legitimately leave none.
        log(
            "legacy_stop_cancellation_complete",
            cancelled_pids=cancelled,
        )
        # Disarm the coordinator's delayed stop before the archive is generated.
        update_pipeline_state(
            "postrun_guardian_claimed",
            status="guardian_claimed",
            terminal_pipeline_status=terminal_status,
            cancelled_legacy_stop_pids=cancelled,
            claimed_at=now(),
        )
        ready = create_archive(terminal_status)

    reason = wait_for_ack_or_timeout(ready)
    stop_instance(reason, ready)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        log(
            "guardian_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        update_pipeline_state(
            "postrun_guardian_failed",
            status="guardian_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            failed_at=now(),
        )
        raise
