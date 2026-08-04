from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback
from typing import Any


ROOT = Path("/workspace/safety-dataset")
PYTHON = Path("/venv/main/bin/python")
EVAL_PYTHON = Path("/workspace/venvs/nemotron-vllm/bin/python")
TRAIN_MANIFEST = ROOT / "data/no_r/full/train.jsonl"
EVAL_MANIFEST = ROOT / "data/no_r/decoder/test_and_sea.jsonl"
QWEN_MODEL = ROOT / "models/qwen3guard-gen-4b"
NEMOTRON_MODEL = ROOT / "models/nemotron-v3"
RESULTS = ROOT / "results/no_r_decoder_4080s"
LOGS = ROOT / "logs/no_r_decoder_4080s"
STATE = RESULTS / "pipeline_state.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_state() -> dict[str, Any]:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "starting",
        "created_at": now(),
        "updated_at": now(),
        "stages": {},
        "contract": {
            "views": ["P", "PR"],
            "response_only_R": "excluded",
            "qwen": {
                "language": "all",
                "expected_rows": 101274,
                "target": "Safety: Safe/Unsafe",
            },
            "nemotron": {
                "language": "vi",
                "expected_rows": 50637,
                "target": "official JSON user/response ratings plus N23",
            },
            "epochs": 1,
            "max_seq_length": 2048,
            "effective_batch": 32,
            "precision": "bf16",
            "quantization": None,
            "lora": {
                "r": 8,
                "alpha": 32,
                "dropout": 0.05,
                "targets": ["q_proj", "v_proj"],
            },
        },
    }


PIPELINE = load_state()


def save_state() -> None:
    PIPELINE["updated_at"] = now()
    atomic_json(STATE, PIPELINE)


def stage_event(name: str, status: str, **extra: Any) -> None:
    value = PIPELINE["stages"].setdefault(name, {})
    value.update(status=status, updated_at=now(), **extra)
    PIPELINE["current_stage"] = name if status == "running" else None
    PIPELINE["status"] = "running"
    save_state()
    print(json.dumps({"stage": name, "status": status, **extra}), flush=True)


def run_stage(
    name: str,
    command: list[str],
    *,
    completed_marker: Path | None = None,
    attempts: int = 1,
    retry_delay: int = 60,
) -> None:
    if completed_marker is not None and completed_marker.exists():
        stage_event(name, "skipped_completed", marker=str(completed_marker))
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(command)
    for attempt in range(1, attempts + 1):
        log_path = LOGS / f"{name}.attempt-{attempt}.log"
        stage_event(
            name,
            "running",
            attempt=attempt,
            command=command_text,
            log=str(log_path),
            started_at=now(),
        )
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n[{now()}] COMMAND {command_text}\n")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env={
                    **os.environ,
                    "HF_HOME": str(ROOT / "hf-cache"),
                    "HUGGINGFACE_HUB_CACHE": str(ROOT / "hf-cache/hub"),
                    "TOKENIZERS_PARALLELISM": "false",
                    "PYTHONUNBUFFERED": "1",
                    # Evaluator scripts run from ROOT/scripts under a dedicated
                    # vLLM interpreter.  Add the repository root explicitly so
                    # shared packages such as guard_smoke remain importable.
                    "PYTHONPATH": os.pathsep.join(
                        filter(
                            None,
                            [str(ROOT), os.environ.get("PYTHONPATH", "")],
                        )
                    ),
                },
            )
            while process.poll() is None:
                PIPELINE["heartbeat_at"] = now()
                PIPELINE["stages"][name]["pid"] = process.pid
                save_state()
                time.sleep(20)
            returncode = process.returncode
        if returncode == 0:
            if completed_marker is not None and not completed_marker.exists():
                raise RuntimeError(
                    f"{name} returned 0 without marker {completed_marker}"
                )
            stage_event(
                name,
                "completed",
                attempt=attempt,
                completed_at=now(),
                returncode=returncode,
            )
            return
        stage_event(
            name,
            "retrying" if attempt < attempts else "failed",
            attempt=attempt,
            returncode=returncode,
            failed_at=now(),
        )
        if attempt < attempts:
            time.sleep(retry_delay)
    raise RuntimeError(f"Stage {name} failed after {attempts} attempt(s)")


def wait_for_download(name: str, model: Path, timeout_hours: float = 4.0) -> None:
    report = ROOT / "results" / f"download_{name}.json"
    deadline = time.monotonic() + timeout_hours * 3600
    stage_name = f"wait_download_{name}"
    stage_event(stage_name, "running", report=str(report))
    while time.monotonic() < deadline:
        if report.exists():
            value = json.loads(report.read_text(encoding="utf-8"))
            if value.get("status") == "completed" and model.exists():
                stage_event(
                    stage_name,
                    "completed",
                    completed_at=now(),
                    bytes=value.get("bytes"),
                    elapsed_seconds=value.get("elapsed_seconds"),
                )
                return
        PIPELINE["heartbeat_at"] = now()
        save_state()
        time.sleep(15)
    raise TimeoutError(f"Timed out waiting for {name} model download")


def supervisor_start(program: str) -> None:
    result = subprocess.run(
        ["supervisorctl", "start", program],
        capture_output=True,
        text=True,
        check=False,
    )
    # "already started" is a successful idempotent state for this pipeline.
    if result.returncode and "already started" not in (result.stdout + result.stderr):
        raise RuntimeError(
            f"Cannot start supervisor program {program}: "
            f"{result.stdout} {result.stderr}"
        )


def completed_download(name: str, model: Path) -> bool:
    report = ROOT / "results" / f"download_{name}.json"
    value = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    return bool(
        isinstance(value, dict)
        and value.get("status") == "completed"
        and model.is_dir()
        and any(model.glob("*.safetensors"))
    )


def profile_microbatch(kind: str, model: Path, language: str, rows: int) -> int:
    for microbatch in (4, 2):
        output = RESULTS / "profiles" / f"{kind}_mb{microbatch}.json"
        if output.exists():
            value = json.loads(output.read_text(encoding="utf-8"))
            if value.get("status") == "completed":
                return microbatch
        try:
            run_stage(
                f"{kind}_stress_mb{microbatch}",
                [
                    str(PYTHON),
                    str(ROOT / "scripts/profile_decoder_lora_training.py"),
                    "--model",
                    str(model),
                    "--model-kind",
                    kind,
                    "--manifest",
                    str(TRAIN_MANIFEST),
                    "--output",
                    str(output),
                    "--language",
                    language,
                    "--sample-limit",
                    "2048",
                    "--projection-rows",
                    str(rows),
                    "--max-seq-length",
                    "2048",
                    "--microbatch",
                    str(microbatch),
                    "--effective-batch",
                    "32",
                    "--optimizer-steps",
                    "3",
                    "--warmup-steps",
                    "1",
                    "--learning-rate",
                    "1e-5",
                    "--stress-longest",
                ],
                completed_marker=output,
            )
            return microbatch
        except RuntimeError:
            if microbatch == 2:
                raise
    raise AssertionError("No safe microbatch found")


def train_model(
    *,
    kind: str,
    model: Path,
    language: str,
    expected_rows: int,
) -> None:
    microbatch = profile_microbatch(kind, model, language, expected_rows)
    PIPELINE["stages"][f"{kind}_batch_selection"] = {
        "status": "completed",
        "microbatch": microbatch,
        "effective_batch": 32,
        "completed_at": now(),
    }
    save_state()
    smoke_dir = RESULTS / "smoke" / kind
    run_stage(
        f"{kind}_trainer_smoke",
        [
            str(PYTHON),
            str(ROOT / "scripts/train_decoder_guard_lora.py"),
            "--model",
            str(model),
            "--model-kind",
            kind,
            "--manifest",
            str(TRAIN_MANIFEST),
            "--output-dir",
            str(smoke_dir),
            "--language",
            language,
            "--max-seq-length",
            "2048",
            "--microbatch",
            str(microbatch),
            "--effective-batch",
            "32",
            "--epochs",
            "1",
            "--sample-limit",
            "512",
            "--max-updates",
            "16",
            "--save-every",
            "8",
            "--stress-longest-first",
            "--resume",
        ],
        completed_marker=smoke_dir / "completed.marker",
        attempts=2,
    )
    full_dir = RESULTS / "train" / kind
    run_stage(
        f"{kind}_full_train",
        [
            str(PYTHON),
            str(ROOT / "scripts/train_decoder_guard_lora.py"),
            "--model",
            str(model),
            "--model-kind",
            kind,
            "--manifest",
            str(TRAIN_MANIFEST),
            "--output-dir",
            str(full_dir),
            "--language",
            language,
            "--max-seq-length",
            "2048",
            "--microbatch",
            str(microbatch),
            "--effective-batch",
            "32",
            "--epochs",
            "1",
            "--save-every",
            "250",
            "--keep-checkpoints",
            "3",
            "--resume",
        ],
        completed_marker=full_dir / "completed.marker",
        attempts=2,
    )


def wait_for_eval_environment(timeout_hours: float = 4.0) -> None:
    marker = ROOT / ".eval_setup_complete"
    deadline = time.monotonic() + timeout_hours * 3600
    stage_event("wait_eval_environment", "running", marker=str(marker))
    while time.monotonic() < deadline:
        if marker.exists() and EVAL_PYTHON.exists():
            stage_event("wait_eval_environment", "completed", completed_at=now())
            return
        status = subprocess.run(
            ["supervisorctl", "status", "decoder-eval-setup"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "FATAL" in status.stdout or "BACKOFF" in status.stdout:
            raise RuntimeError(status.stdout.strip())
        PIPELINE["heartbeat_at"] = now()
        save_state()
        time.sleep(20)
    raise TimeoutError("Timed out waiting for vLLM evaluation environment")


def evaluate_qwen() -> None:
    wait_for_eval_environment()
    qwen_output = RESULTS / "eval" / "qwen"
    run_stage(
        "qwen_eval",
        [
            str(EVAL_PYTHON),
            str(ROOT / "scripts/evaluate_qwen3guard_gen_vllm.py"),
            "--model",
            str(QWEN_MODEL),
            "--lora-adapter",
            str(RESULTS / "train/qwen/final_adapter"),
            "--binary-only-adapter-contract",
            "--manifest",
            str(EVAL_MANIFEST),
            "--output-dir",
            str(qwen_output),
            "--max-input-tokens",
            "8064",
            "--max-model-len",
            "8192",
            "--max-new-tokens",
            "16",
            "--request-chunk-size",
            "512",
            "--max-num-seqs",
            "64",
            "--max-num-batched-tokens",
            "8192",
            "--gpu-memory-utilization",
            "0.90",
            "--resume",
        ],
        completed_marker=qwen_output / "metrics.json",
        attempts=2,
    )


def evaluate_nemotron() -> None:
    wait_for_eval_environment()
    nemotron_output = RESULTS / "eval" / "nemotron"
    run_stage(
        "nemotron_eval",
        [
            str(EVAL_PYTHON),
            str(ROOT / "scripts/evaluate_nemotron_decoder_guard_vllm.py"),
            "--model",
            str(NEMOTRON_MODEL),
            "--lora-adapter",
            str(RESULTS / "train/nemotron/final_adapter"),
            "--manifest",
            str(EVAL_MANIFEST),
            "--output-dir",
            str(nemotron_output),
            "--max-input-tokens",
            "8064",
            "--max-model-len",
            "8192",
            "--max-new-tokens",
            "100",
            "--request-chunk-size",
            "512",
            "--max-num-seqs",
            "64",
            "--max-num-batched-tokens",
            "8192",
            "--gpu-memory-utilization",
            "0.90",
            "--resume",
        ],
        completed_marker=nemotron_output / "metrics.json",
        attempts=2,
    )


def schedule_stop(delay_seconds: int) -> None:
    postrun_guardian = ROOT / "scripts/vast/no_r_d3_postrun_guardian.py"
    if postrun_guardian.exists():
        # The guardian packages the adapter/results, waits for a checksum-bearing
        # local acknowledgement, and then stops the instance.  Spawning the old
        # delayed stop here would race that handoff and can interrupt the download.
        PIPELINE["auto_stop"] = {
            "scheduled_at": now(),
            "action": "delegated_to_postrun_guardian",
            "guardian": str(postrun_guardian),
            "fallback_stop_timeout_seconds": 1800,
        }
        save_state()
        return
    # Vast stop preserves the container filesystem; destroy/recycle is never used.
    # Re-check the durable state after the delay.  Without this guard, a delayed
    # stop left by a failed coordinator can terminate a newly restarted run.
    command = (
        f"sleep {delay_seconds}; "
        f"if grep -Eq '\"status\": \"(failed|completed)\"' {shlex.quote(str(STATE))}; "
        'then vastai stop instance "$CONTAINER_ID" '
        '--api-key "$CONTAINER_API_KEY"; fi'
    )
    subprocess.Popen(
        ["bash", "-lc", command],
        stdout=(LOGS / "auto-stop.log").open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PIPELINE["auto_stop"] = {
        "scheduled_at": now(),
        "delay_seconds": delay_seconds,
        "action": "stop_instance_preserve_filesystem",
    }
    save_state()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-auto-stop",
        action="store_true",
        help="Leave the instance running after success/failure.",
    )
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    # A restarted coordinator supersedes any prior terminal state.  Clear stale
    # dashboard errors/auto-stop metadata before the first resumed stage.
    PIPELINE["status"] = "running"
    PIPELINE["current_stage"] = None
    for stale_key in ("failure", "failed_at", "auto_stop"):
        PIPELINE.pop(stale_key, None)
    save_state()
    try:
        for required in (TRAIN_MANIFEST, EVAL_MANIFEST):
            if not required.exists():
                raise FileNotFoundError(required)
        wait_for_download("qwen", QWEN_MODEL)
        # A resumed pipeline may already have both model snapshots and the vLLM
        # environment. Starting a completed one-shot supervisor program again is
        # not idempotent on every Vast image and can produce a spawn error.
        if not completed_download("nemotron", NEMOTRON_MODEL):
            supervisor_start("decoder-download-nemotron")
        if not (ROOT / ".eval_setup_complete").exists() or not EVAL_PYTHON.exists():
            supervisor_start("decoder-eval-setup")
        train_model(
            kind="qwen",
            model=QWEN_MODEL,
            language="all",
            expected_rows=101274,
        )
        # Evaluate Qwen immediately after its adapter is ready. This keeps the
        # first scientific result available even when the larger Nemotron run
        # takes several additional hours. A completed metrics.json makes this
        # stage idempotent on every later resume.
        evaluate_qwen()
        wait_for_download("nemotron", NEMOTRON_MODEL)
        train_model(
            kind="nemotron",
            model=NEMOTRON_MODEL,
            language="vi",
            expected_rows=50637,
        )
        evaluate_nemotron()
        PIPELINE["status"] = "completed"
        PIPELINE["current_stage"] = None
        PIPELINE["completed_at"] = now()
        save_state()
        (RESULTS / "completed.marker").write_text("completed\n", encoding="utf-8")
        if not args.no_auto_stop:
            schedule_stop(120)
    except BaseException as exc:
        PIPELINE["status"] = "failed"
        PIPELINE["current_stage"] = None
        PIPELINE["failed_at"] = now()
        PIPELINE["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_state()
        (RESULTS / "failed.marker").write_text("failed\n", encoding="utf-8")
        if not args.no_auto_stop:
            schedule_stop(300)
        raise


if __name__ == "__main__":
    main()
