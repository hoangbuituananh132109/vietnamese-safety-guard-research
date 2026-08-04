from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from typing import Any


DEFAULT_ROOT = Path("/workspace/safety-dataset")
DEFAULT_CONFIG = Path("configs/phase0_no_r_experiments.json")
DEFAULT_REPORT_ROOT = Path("reports/no_r_phase0")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the clean P/PR-only E1-E5/E7 encoder matrix, evaluate every "
            "checkpoint, package reproducibility artifacts and stop the Vast "
            "instance without destroying it."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--python", type=Path, default=Path("/venv/main/bin/python"))
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--retry-seconds", type=int, default=60)
    parser.add_argument("--max-wall-hours", type=float, default=11.25)
    parser.add_argument("--ack-wait-seconds", type=int, default=900)
    parser.add_argument(
        "--archive-stem",
        default="PHASE0_NO_R_20260724",
        help="Filename stem for the archive, ready file and download ACK.",
    )
    parser.add_argument("--no-auto-stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class BudgetExpired(RuntimeError):
    pass


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.root.resolve()
        os.chdir(self.root)
        self.config_path = (
            args.config
            if args.config.is_absolute()
            else (self.root / args.config)
        ).resolve()
        self.report_root = (
            args.report_root
            if args.report_root.is_absolute()
            else (self.root / args.report_root)
        ).resolve()
        self.run_root = self.report_root / "experiment_runs"
        self.matrix_root = self.report_root / "evaluation_matrix"
        self.log_root = self.report_root / "logs"
        self.export_root = self.root / "exports"
        self.state_path = self.report_root / "pipeline_state.json"
        self.events_path = self.report_root / "events.jsonl"
        self.archive_stem = str(args.archive_stem)
        if not self.archive_stem or any(char in self.archive_stem for char in "/\\"):
            raise ValueError(f"Invalid archive stem: {self.archive_stem!r}")
        self.ready_path = self.export_root / f"{self.archive_stem}.ready.json"
        self.ack_path = self.export_root / f"{self.archive_stem}.download_verified.json"
        self.archive_path = self.export_root / f"{self.archive_stem}.tar.zst"
        self.inventory_path = self.report_root / "artifact_inventory.json"
        self.started_monotonic = time.monotonic()
        self.started_epoch = time.time()
        self.deadline = (
            self.started_monotonic + float(args.max_wall_hours) * 3600
        )
        self.config = read_json(self.config_path)
        if not isinstance(self.config, dict):
            raise RuntimeError(f"Cannot parse experiment config: {self.config_path}")
        self.runs = list(self.config.get("runs") or [])
        if not self.runs:
            raise RuntimeError("Experiment config contains no runs")
        self.run_by_id = {str(run["id"]): run for run in self.runs}
        if len(self.run_by_id) != len(self.runs):
            raise RuntimeError("Duplicate run IDs in experiment config")
        self.report_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.matrix_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.export_root.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = read_json(self.state_path, {})
        if not isinstance(self.state, dict):
            self.state = {}
        self.state.update(
            {
                "status": "initializing",
                "started_at": self.state.get("started_at") or utc_now(),
                "process_started_at": utc_now(),
                "root": str(self.root),
                "config": str(self.config_path),
                "report_root": str(self.report_root),
                "run_ids": list(self.run_by_id),
                "max_wall_hours": float(args.max_wall_hours),
                "auto_stop": not args.no_auto_stop,
                "dry_run": bool(args.dry_run),
            }
        )
        self.write_state()

    def write_state(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state["updated_at"] = utc_now()
        self.state["process_elapsed_seconds"] = (
            time.monotonic() - self.started_monotonic
        )
        self.state["budget_remaining_seconds"] = max(
            0.0, self.deadline - time.monotonic()
        )
        atomic_json(self.state_path, self.state)

    def event(self, name: str, **payload: Any) -> None:
        row = {"at": utc_now(), "event": name, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)

    def relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))

    def metrics_path(self, run_id: str) -> Path:
        return self.run_root / run_id / "metrics.json"

    def matrix_index_path(self, run_id: str) -> Path:
        return self.matrix_root / run_id / "matrix_index.json"

    def expected_eval_jobs(self, run: dict[str, Any]) -> int:
        count = 0
        label_orders = 2 if run["model_kind"] == "mmbert_schema" else 1
        for suite_name in run.get("eval") or []:
            suite = self.config["benchmarks"][suite_name]
            count += sum(
                1
                for value in suite.values()
                if isinstance(value, str) and value.endswith(".jsonl")
            ) * label_orders
        return count

    def validate_train(self, run: dict[str, Any]) -> dict[str, Any] | None:
        path = self.metrics_path(str(run["id"]))
        data = read_json(path)
        if not isinstance(data, dict) or data.get("status") != "completed":
            return None
        if run["model_kind"] != "gliguard_schema":
            if int(data.get("nonfinite_optimizer_updates", -1)) != 0:
                raise RuntimeError(
                    f"{run['id']} contains non-finite optimizer updates"
                )
            if not bool(data.get("encoder_gradient_verified")):
                raise RuntimeError(f"{run['id']} did not verify encoder gradients")
        else:
            scaler = data.get("scaler_audit") or {}
            if int(scaler.get("skipped_optimizer_updates", 0)) != 0:
                raise RuntimeError(
                    f"{run['id']} skipped optimizer updates: {scaler}"
                )
        return {
            "metrics": self.relative(path),
            "steps": (
                (data.get("train_result") or {}).get("total_steps")
                if run["model_kind"] == "gliguard_schema"
                else data.get("completed_optimizer_steps")
            ),
            "elapsed_seconds": data.get("elapsed_seconds"),
        }

    def validate_eval(self, run: dict[str, Any]) -> dict[str, Any] | None:
        path = self.matrix_index_path(str(run["id"]))
        data = read_json(path)
        if not isinstance(data, dict):
            return None
        jobs = list(data.get("jobs") or [])
        expected = self.expected_eval_jobs(run)
        if len(jobs) != expected:
            return None
        missing = []
        for job in jobs:
            metric = Path(str(job["metrics"]))
            if not metric.is_absolute():
                metric = self.root / metric
            if not metric.exists():
                missing.append(str(metric))
        if missing:
            return None
        return {
            "matrix_index": self.relative(path),
            "jobs": len(jobs),
            "examples": sum(int(job.get("examples") or 0) for job in jobs),
        }

    def latest_checkpoint(self, run_id: str) -> Path | None:
        checkpoint_root = self.run_root / run_id / "checkpoints"
        if not checkpoint_root.exists():
            return None
        candidates = sorted(
            path
            for path in checkpoint_root.glob("step-*")
            if path.is_dir()
        )
        return candidates[-1] if candidates else None

    def archive_partial_gliguard(self, run_id: str) -> None:
        output = self.run_root / run_id
        if not output.exists() or self.metrics_path(run_id).exists():
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive = output.with_name(f"{run_id}-partial-{stamp}")
        shutil.move(str(output), str(archive))
        self.event(
            "gliguard_partial_archived_for_fresh_restart",
            run_id=run_id,
            archive=self.relative(archive),
        )

    def training_command(self, run: dict[str, Any]) -> list[str]:
        run_id = str(run["id"])
        kind = str(run["model_kind"])
        if kind == "gliguard_schema":
            self.archive_partial_gliguard(run_id)
            script = "scripts/train_gliguard_experiment.py"
        elif kind == "mmbert_fixed":
            script = "scripts/train_mmbert_fixed_experiment.py"
        elif kind == "mmbert_schema":
            script = "scripts/train_mmbert_schema_experiment.py"
        else:
            raise ValueError(f"Unsupported model kind: {kind}")
        command = [
            str(self.args.python),
            script,
            "--config",
            self.relative(self.config_path),
            "--run-id",
            run_id,
            "--output-root",
            self.relative(self.run_root),
            "--micro-batch-size",
            str(run.get("micro_batch_size", 8)),
            "--epochs",
            str(run.get("epochs", 2)),
            "--precision",
            "auto",
        ]
        if kind != "gliguard_schema":
            checkpoint = self.latest_checkpoint(run_id)
            if checkpoint is not None:
                command.extend(["--resume-from", self.relative(checkpoint)])
                self.event(
                    "exact_resume_selected",
                    run_id=run_id,
                    checkpoint=self.relative(checkpoint),
                )
        return command

    def evaluation_command(self, run: dict[str, Any]) -> list[str]:
        return [
            str(self.args.python),
            "scripts/evaluate_experiment_matrix.py",
            "--config",
            self.relative(self.config_path),
            "--run-id",
            str(run["id"]),
            "--run-root",
            self.relative(self.run_root),
            "--output-root",
            self.relative(self.matrix_root),
            "--batch-size",
            str(run.get("inference_batch_size", 8)),
            "--precision",
            "auto",
        ]

    def terminate_child(self, process: subprocess.Popen[str]) -> None:
        self.event("budget_deadline_terminating_child", pid=process.pid)
        try:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=45)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=15)

    def run_command(
        self, stage: str, run_id: str, command: list[str]
    ) -> None:
        log_path = self.log_root / f"{stage}.log"
        self.write_state(
            status="running",
            current_stage=stage,
            current_run_id=run_id,
            stage_started_at=utc_now(),
            stage_log=self.relative(log_path),
            command=command,
        )
        self.event(
            "command_started", stage=stage, run_id=run_id, command=command
        )
        if self.args.dry_run:
            self.event("dry_run_command_skipped", stage=stage, run_id=run_id)
            return
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n===== {utc_now()} START {stage} =====\n")
            log.write(json.dumps(command, ensure_ascii=False) + "\n")
            process = subprocess.Popen(
                command,
                cwd=self.root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            stage_started = time.monotonic()
            while process.poll() is None:
                now = time.monotonic()
                self.write_state(
                    child_pid=process.pid,
                    stage_elapsed_seconds=now - stage_started,
                    stage_log_bytes=(
                        log_path.stat().st_size if log_path.exists() else 0
                    ),
                )
                if now >= self.deadline:
                    self.terminate_child(process)
                    raise BudgetExpired(
                        f"Hard wall-time guard reached during {stage}"
                    )
                time.sleep(max(5, int(self.args.poll_seconds)))
            returncode = int(process.returncode)
            log.write(
                f"===== {utc_now()} END {stage} exit={returncode} =====\n"
            )
        elapsed = time.monotonic() - stage_started
        self.event(
            "command_completed",
            stage=stage,
            run_id=run_id,
            returncode=returncode,
            elapsed_seconds=elapsed,
            log=self.relative(log_path),
        )
        if returncode != 0:
            raise RuntimeError(
                f"{stage} failed with exit code {returncode}; log={log_path}"
            )

    def execute_with_retry(
        self,
        stage: str,
        run: dict[str, Any],
        command_factory: Any,
        validator: Any,
    ) -> dict[str, Any]:
        existing = validator(run)
        if existing is not None:
            self.event(
                "completed_stage_reused",
                stage=stage,
                run_id=run["id"],
                result=existing,
            )
            return existing
        for attempt in (1, 2):
            if time.monotonic() >= self.deadline:
                raise BudgetExpired(f"Budget exhausted before {stage}")
            try:
                self.run_command(
                    stage, str(run["id"]), command_factory(run)
                )
                if self.args.dry_run:
                    return {"dry_run": True}
                result = validator(run)
                if result is None:
                    raise RuntimeError(
                        f"{stage} exited successfully but validation failed"
                    )
                return result
            except BudgetExpired:
                raise
            except Exception as exc:
                self.event(
                    "stage_attempt_failed",
                    stage=stage,
                    run_id=run["id"],
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
                if attempt == 2:
                    raise
                self.write_state(
                    status="retry_wait",
                    current_stage=stage,
                    stage_attempt=attempt,
                    retry_seconds=int(self.args.retry_seconds),
                )
                time.sleep(max(1, int(self.args.retry_seconds)))
        raise AssertionError("retry loop exited unexpectedly")

    def preflight(self) -> None:
        output = self.report_root / "preflight.json"
        command = [
            str(self.args.python),
            "scripts/preflight_phase0_experiments.py",
            "--config",
            self.relative(self.config_path),
            "--output",
            self.relative(output),
        ]
        self.run_command("preflight", "all", command)
        if not self.args.dry_run:
            report = read_json(output, {})
            if report.get("status") != "data_preflight_passed":
                raise RuntimeError(f"Data preflight failed: {report}")

    def run_experiments(self) -> None:
        completed = list(self.state.get("completed_stages") or [])
        for index, run in enumerate(self.runs, 1):
            run_id = str(run["id"])
            self.write_state(
                run_index=index,
                run_count=len(self.runs),
                current_run_id=run_id,
            )
            train_stage = f"train_{run_id}"
            train_result = self.execute_with_retry(
                train_stage,
                run,
                self.training_command,
                self.validate_train,
            )
            completed.append(
                {
                    "stage": train_stage,
                    "run_id": run_id,
                    "result": train_result,
                    "at": utc_now(),
                }
            )
            self.write_state(completed_stages=completed)
            eval_stage = f"evaluate_{run_id}"
            eval_result = self.execute_with_retry(
                eval_stage,
                run,
                self.evaluation_command,
                self.validate_eval,
            )
            completed.append(
                {
                    "stage": eval_stage,
                    "run_id": run_id,
                    "result": eval_result,
                    "at": utc_now(),
                }
            )
            self.write_state(completed_stages=completed)

    def referenced_files(self) -> list[Path]:
        values: set[Path] = {self.config_path}
        for suite in self.config.get("benchmarks", {}).values():
            if isinstance(suite, dict):
                for value in suite.values():
                    if isinstance(value, str) and value.endswith(".jsonl"):
                        values.add((self.root / value).resolve())
        for run in self.runs:
            for key in (
                "train_manifest",
                "valid_manifest",
                "canonical_valid",
                "test_manifest",
                "base_model",
            ):
                value = run.get(key)
                if value and key != "base_model":
                    values.add((self.root / str(value)).resolve())
        return sorted(values)

    def build_inventory(self, terminal_status: str) -> dict[str, Any]:
        manifest_files = []
        for path in self.referenced_files():
            manifest_files.append(
                {
                    "path": self.relative(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        code_roots = [
            self.root / "guard_train",
            self.root / "guard_smoke",
            self.root / "scripts",
        ]
        code_files = []
        for code_root in code_roots:
            for path in sorted(code_root.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                code_files.append(
                    {
                        "path": self.relative(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        inventory = {
            "created_at": utc_now(),
            "terminal_status": terminal_status,
            "config_sha256": sha256_file(self.config_path),
            "manifests": manifest_files,
            "code": code_files,
            "environment": {
                "python": sys.version,
                "executable": str(self.args.python),
                "container_id": os.environ.get("CONTAINER_ID"),
            },
        }
        atomic_json(self.inventory_path, inventory)
        return inventory

    def create_archive(self, terminal_status: str) -> dict[str, Any]:
        self.write_state(status="packaging", terminal_status=terminal_status)
        self.build_inventory(terminal_status)
        archive_tmp = self.archive_path.with_suffix(
            self.archive_path.suffix + ".tmp"
        )
        archive_tmp.unlink(missing_ok=True)
        include = [
            self.relative(self.report_root),
            self.relative(self.config_path),
            "guard_train",
            "guard_smoke",
            "scripts",
            "requirements-vast.txt",
        ]
        command = [
            "tar",
            "--zstd",
            "-cf",
            str(archive_tmp),
            *include,
        ]
        self.event("archive_started", archive=str(self.archive_path))
        if not self.args.dry_run:
            subprocess.run(command, cwd=self.root, check=True)
            os.replace(archive_tmp, self.archive_path)
            ready = {
                "status": "ready",
                "terminal_status": terminal_status,
                "created_at": utc_now(),
                "archive_path": str(self.archive_path),
                "archive_bytes": self.archive_path.stat().st_size,
                "archive_sha256": sha256_file(self.archive_path),
                "inventory": self.relative(self.inventory_path),
                "ack_path": str(self.ack_path),
            }
            atomic_json(self.ready_path, ready)
        else:
            ready = {
                "status": "dry_run",
                "terminal_status": terminal_status,
                "archive_path": str(self.archive_path),
            }
        self.event("archive_ready", **ready)
        return ready

    def wait_for_ack(self, ready: dict[str, Any]) -> str:
        if self.args.dry_run:
            return "dry_run"
        deadline = time.monotonic() + max(0, int(self.args.ack_wait_seconds))
        self.write_state(
            status="awaiting_download_ack",
            ready=ready,
            ack_wait_seconds=int(self.args.ack_wait_seconds),
        )
        while time.monotonic() < deadline:
            ack = read_json(self.ack_path, {})
            if (
                ack.get("status") == "verified"
                and ack.get("archive_sha256") == ready.get("archive_sha256")
                and int(ack.get("archive_bytes", -1))
                == int(ready.get("archive_bytes", -2))
            ):
                self.event(
                    "download_ack_verified",
                    local_path=ack.get("local_path"),
                    sha256=ready.get("archive_sha256"),
                )
                return "verified_local_download"
            time.sleep(max(5, int(self.args.poll_seconds)))
        self.event(
            "download_ack_timeout",
            wait_seconds=int(self.args.ack_wait_seconds),
            archive=str(self.archive_path),
        )
        return "ack_timeout_archive_preserved_on_stopped_instance"

    def stop_instance(self, reason: str, ready: dict[str, Any]) -> None:
        if self.args.no_auto_stop or self.args.dry_run:
            self.write_state(
                status="completed_not_stopped",
                stop_reason=reason,
                ready=ready,
            )
            return
        container_id = os.environ.get("CONTAINER_ID")
        api_key = os.environ.get("CONTAINER_API_KEY")
        if not container_id or not api_key:
            self.write_state(
                status="stop_failed_missing_container_credentials",
                stop_reason=reason,
                ready=ready,
            )
            self.event(
                "auto_stop_unavailable",
                container_id_present=bool(container_id),
                api_key_present=bool(api_key),
            )
            return
        marker = self.report_root / "stop_intent.json"
        atomic_json(
            marker,
            {
                "status": "stop_intent_recorded",
                "container_id": container_id,
                "reason": reason,
                "archive_sha256": ready.get("archive_sha256"),
                "recorded_at": utc_now(),
            },
        )
        self.write_state(
            status="stop_requested",
            stop_reason=reason,
            ready=ready,
        )
        self.event(
            "vast_stop_requested",
            container_id=container_id,
            reason=reason,
        )
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
        self.event(
            "vast_stop_command_returned",
            returncode=result.returncode,
            stdout=result.stdout[-1000:],
            stderr=result.stderr[-1000:],
        )
        if result.returncode != 0:
            self.write_state(
                status="stop_command_failed",
                stop_returncode=result.returncode,
            )

    def run(self) -> None:
        terminal_status = "completed"
        error: BaseException | None = None
        try:
            self.write_state(status="running", current_stage="preflight")
            self.preflight()
            self.run_experiments()
            self.write_state(
                status="completed",
                current_stage=None,
                completed_at=utc_now(),
            )
            self.event(
                "pipeline_completed",
                runs=len(self.runs),
                elapsed_seconds=time.monotonic() - self.started_monotonic,
            )
        except BudgetExpired as exc:
            terminal_status = "budget_paused"
            error = exc
            self.write_state(
                status=terminal_status,
                error=str(exc),
                paused_at=utc_now(),
            )
            self.event("pipeline_budget_paused", error=str(exc))
        except BaseException as exc:
            terminal_status = "failed"
            error = exc
            self.write_state(
                status=terminal_status,
                error=f"{type(exc).__name__}: {exc}",
                failed_at=utc_now(),
                traceback=traceback.format_exc(),
            )
            self.event(
                "pipeline_failed",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        ready = self.create_archive(terminal_status)
        reason = self.wait_for_ack(ready)
        self.stop_instance(reason, ready)
        if error is not None:
            raise error


def main() -> None:
    args = parse_args()
    Pipeline(args).run()


if __name__ == "__main__":
    main()
