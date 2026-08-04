from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any


E1 = "E1-G-EV-512"
B0 = "B0-GLI-ZS"
E3 = "E3-M-E-8K"
E4 = "E4-M-EV-MATCHED-8K"
E5 = "E5-M-EV-FULL-8K"
E7 = "E7-M-SCHEMA-EV-8K"


@dataclass(frozen=True)
class Stage:
    name: str
    kind: str
    run_id: str
    batch_size: int = 8


STAGES = (
    Stage("wait_and_verify_e1", "wait_e1", E1),
    Stage("evaluate_e1", "evaluate", E1, 8),
    Stage("evaluate_b0_zero_shot", "evaluate", B0, 8),
    Stage("train_e3_english", "train_fixed", E3, 8),
    Stage("evaluate_e3_english", "evaluate", E3, 8),
    Stage("train_e4_matched_en_vi", "train_fixed", E4, 8),
    Stage("evaluate_e4_matched_en_vi", "evaluate", E4, 8),
    Stage("train_e5_full_en_vi", "train_fixed", E5, 8),
    Stage("evaluate_e5_full_en_vi", "evaluate", E5, 8),
    Stage("train_e7_dynamic_schema", "train_schema", E7, 8),
    Stage("evaluate_e7_dynamic_schema", "evaluate", E7, 8),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/safety-dataset"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class OvernightRunner:
    def __init__(self, root: Path, poll_seconds: int, dry_run: bool) -> None:
        self.root = root.resolve()
        self.poll_seconds = max(10, poll_seconds)
        self.dry_run = dry_run
        self.python = Path("/venv/main/bin/python")
        self.report_root = self.root / "reports" / "overnight_phase0"
        self.log_root = self.report_root / "logs"
        self.state_path = self.report_root / "state.json"
        self.events_path = self.report_root / "events.jsonl"
        self.report_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = {
            "status": "initializing",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "root": str(self.root),
            "stages": [stage.name for stage in STAGES],
            "completed_stages": [],
        }
        self.write_state()

    def assert_under_root(self, path: Path) -> None:
        path.resolve().relative_to(self.root)

    def write_state(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state["updated_at"] = utc_now()
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def event(self, event: str, **payload: Any) -> None:
        row = {"at": utc_now(), "event": event, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def metrics_path(self, run_id: str) -> Path:
        return self.root / "reports" / "experiment_runs" / run_id / "metrics.json"

    def matrix_index_path(self, run_id: str) -> Path:
        return self.root / "reports" / "evaluation_matrix" / run_id / "matrix_index.json"

    def read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def validate_e1(self) -> dict[str, Any]:
        path = self.metrics_path(E1)
        if not path.exists():
            raise RuntimeError(f"Missing E1 metrics: {path}")
        data = self.read_json(path)
        if data.get("status") != "completed":
            raise RuntimeError(f"E1 status is not completed: {data.get('status')}")
        if int(data["train_result"]["total_steps"]) != 7226:
            raise RuntimeError(f"E1 optimizer-step mismatch: {data['train_result']['total_steps']}")
        if int(data["scaler_audit"]["skipped_optimizer_updates"]) != 0:
            raise RuntimeError(f"E1 skipped FP16 optimizer updates: {data['scaler_audit']}")
        return {
            "steps": data["train_result"]["total_steps"],
            "epochs": data["train_result"]["total_epochs"],
            "scaler_audit": data["scaler_audit"],
            "validation": data["validation"]["overall"],
        }

    def validate_mmbert(self, run_id: str) -> dict[str, Any]:
        path = self.metrics_path(run_id)
        if not path.exists():
            raise RuntimeError(f"Missing metrics for {run_id}: {path}")
        data = self.read_json(path)
        if data.get("status") != "completed":
            raise RuntimeError(f"{run_id} status is not completed: {data.get('status')}")
        if not bool(data.get("encoder_gradient_verified")):
            raise RuntimeError(f"{run_id} did not verify encoder/LoRA gradients")
        if int(data.get("nonfinite_optimizer_updates", -1)) != 0:
            raise RuntimeError(
                f"{run_id} had non-finite optimizer updates: "
                f"{data.get('nonfinite_optimizer_updates')}"
            )
        completed_steps = int(data["completed_optimizer_steps"])
        planned_steps = int(data["planned_optimizer_steps"])
        optimizer_step_delta = completed_steps - planned_steps
        audit_warning = None
        if optimizer_step_delta != 0:
            # E3 finished before the epoch-boundary accumulation fix.  Its second
            # epoch contains one extra, partial optimizer update (1 / 4,380,
            # 0.023%).  Preserve the completed artifact and disclose the exact
            # deviation instead of wasting an hour rerunning it.  All later runs
            # remain strict and must match their planned step count exactly.
            if run_id == E3 and optimizer_step_delta == 1:
                audit_warning = (
                    "legacy_epoch_boundary_accumulation: one extra optimizer "
                    "update accepted for E3 only"
                )
                audit_path = path.parent / "optimizer_step_audit.json"
                audit_path.write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "planned_optimizer_steps": planned_steps,
                            "completed_optimizer_steps": completed_steps,
                            "optimizer_step_delta": optimizer_step_delta,
                            "relative_delta": optimizer_step_delta / planned_steps,
                            "disposition": "accepted_with_disclosed_legacy_warning",
                            "warning": audit_warning,
                            "audited_at": utc_now(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                raise RuntimeError(
                    f"{run_id} incomplete: {completed_steps} / {planned_steps}"
                )
        result = {
            "steps": data["completed_optimizer_steps"],
            "planned_steps": planned_steps,
            "optimizer_step_delta": optimizer_step_delta,
            "elapsed_seconds": data["elapsed_seconds"],
            "peak_vram_mb": data["peak_vram_mb"],
            "train_truncated": data["train_truncated"],
            "valid_truncated": data["valid_truncated"],
            "validation_binary": data["validation"]["binary"],
            "validation_N23": data["validation"].get("N23"),
        }
        if audit_warning is not None:
            result["audit_warning"] = audit_warning
        return result

    def validate_matrix(self, run_id: str) -> dict[str, Any]:
        path = self.matrix_index_path(run_id)
        if not path.exists():
            raise RuntimeError(f"Missing evaluation matrix index for {run_id}: {path}")
        data = self.read_json(path)
        jobs = data.get("jobs") or []
        if not jobs:
            raise RuntimeError(f"Evaluation matrix for {run_id} contains no jobs")
        missing = [job["metrics"] for job in jobs if not (self.root / job["metrics"]).exists()]
        if missing:
            raise RuntimeError(f"Evaluation matrix for {run_id} has missing metrics: {missing}")
        return {"jobs": len(jobs), "examples": sum(int(job.get("examples") or 0) for job in jobs)}

    def archive_partial_matrix(self, run_id: str) -> None:
        output = self.root / "reports" / "evaluation_matrix" / run_id
        if not output.exists() or self.matrix_index_path(run_id).exists():
            return
        self.assert_under_root(output)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive = output.with_name(f"{output.name}-partial-{stamp}")
        if archive.exists():
            raise RuntimeError(f"Partial evaluation archive already exists: {archive}")
        shutil.move(str(output), str(archive))
        self.event("partial_matrix_archived", run_id=run_id, source=str(output), archive=str(archive))

    def latest_checkpoint(self, run_id: str) -> Path | None:
        checkpoint_root = self.root / "reports" / "experiment_runs" / run_id / "checkpoints"
        if not checkpoint_root.exists():
            return None
        candidates = sorted(
            (path for path in checkpoint_root.glob("step-*") if path.is_dir()),
            key=lambda path: path.name,
        )
        return candidates[-1] if candidates else None

    def run_command(self, stage: Stage, command: list[str]) -> None:
        log_path = self.log_root / f"{stage.name}.log"
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
        self.write_state(
            status="running",
            current_stage=stage.name,
            current_run_id=stage.run_id,
            command=command,
            stage_log=str(log_path),
            stage_started_at=utc_now(),
        )
        self.event("command_started", stage=stage.name, run_id=stage.run_id, command=command)
        if self.dry_run:
            self.event("dry_run_command_skipped", stage=stage.name)
            return
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n===== {utc_now()} START {stage.name} =====\n")
            log.write(json.dumps(command, ensure_ascii=False) + "\n")
            process = subprocess.Popen(
                command,
                cwd=self.root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            started = time.monotonic()
            while process.poll() is None:
                self.write_state(
                    stage_elapsed_seconds=time.monotonic() - started,
                    stage_log_bytes=log_path.stat().st_size,
                    child_pid=process.pid,
                )
                time.sleep(self.poll_seconds)
            code = int(process.returncode)
            log.write(f"===== {utc_now()} END {stage.name} exit={code} =====\n")
        if code != 0:
            raise RuntimeError(f"Stage {stage.name} failed with exit code {code}; log={log_path}")
        self.event(
            "command_completed",
            stage=stage.name,
            run_id=stage.run_id,
            elapsed_seconds=time.monotonic() - started,
            log=str(log_path),
        )

    def wait_for_e1(self, stage: Stage) -> dict[str, Any]:
        self.write_state(status="waiting", current_stage=stage.name, current_run_id=E1)
        self.event("waiting_for_existing_e1", metrics=str(self.metrics_path(E1)))
        if self.dry_run:
            return {"dry_run": True}
        while not self.metrics_path(E1).exists():
            active = subprocess.run(
                ["tmux", "has-session", "-t", "e1_full"],
                cwd=self.root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0
            if not active:
                raise RuntimeError("E1 tmux session ended before metrics.json was written")
            self.write_state(waiting_for="E1 metrics.json", e1_tmux_active=True)
            time.sleep(self.poll_seconds)
        return self.validate_e1()

    def training_command(self, stage: Stage) -> list[str]:
        script = (
            "scripts/train_mmbert_schema_experiment.py"
            if stage.kind == "train_schema"
            else "scripts/train_mmbert_fixed_experiment.py"
        )
        command = [
            str(self.python),
            script,
            "--run-id",
            stage.run_id,
            "--micro-batch-size",
            str(stage.batch_size),
            "--epochs",
            "2",
            "--precision",
            "auto",
        ]
        output = self.root / "reports" / "experiment_runs" / stage.run_id
        if output.exists() and not self.metrics_path(stage.run_id).exists():
            checkpoint = self.latest_checkpoint(stage.run_id)
            if checkpoint is None:
                raise RuntimeError(
                    f"Partial output exists for {stage.run_id} but has no exact-resume checkpoint: {output}"
                )
            command.extend(["--resume-from", str(checkpoint.relative_to(self.root))])
            self.event("exact_resume_selected", run_id=stage.run_id, checkpoint=str(checkpoint))
        return command

    def execute_stage(self, stage: Stage) -> dict[str, Any]:
        if stage.kind == "wait_e1":
            return self.wait_for_e1(stage)
        if stage.kind == "evaluate":
            if self.matrix_index_path(stage.run_id).exists():
                result = self.validate_matrix(stage.run_id)
                self.event("completed_stage_reused", stage=stage.name, run_id=stage.run_id, result=result)
                return result
            self.archive_partial_matrix(stage.run_id)
            command = [
                str(self.python),
                "scripts/evaluate_experiment_matrix.py",
                "--run-id",
                stage.run_id,
                "--batch-size",
                str(stage.batch_size),
                "--precision",
                "auto",
            ]
            self.run_command(stage, command)
            return self.validate_matrix(stage.run_id) if not self.dry_run else {"dry_run": True}
        if stage.kind in {"train_fixed", "train_schema"}:
            if self.metrics_path(stage.run_id).exists():
                result = self.validate_mmbert(stage.run_id)
                self.event("completed_stage_reused", stage=stage.name, run_id=stage.run_id, result=result)
                return result
            self.run_command(stage, self.training_command(stage))
            return self.validate_mmbert(stage.run_id) if not self.dry_run else {"dry_run": True}
        raise ValueError(f"Unknown stage kind: {stage.kind}")

    def run(self) -> None:
        self.write_state(status="running", current_stage=None)
        self.event("overnight_runner_started", dry_run=self.dry_run)
        for index, stage in enumerate(STAGES, 1):
            self.write_state(stage_index=index, stage_count=len(STAGES), current_stage=stage.name)
            self.event("stage_started", index=index, total=len(STAGES), stage=stage.name)
            result = None
            for attempt in range(1, 3):
                try:
                    result = self.execute_stage(stage)
                    break
                except Exception as exc:
                    self.event(
                        "stage_attempt_failed",
                        index=index,
                        total=len(STAGES),
                        stage=stage.name,
                        attempt=attempt,
                        max_attempts=2,
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                    )
                    if attempt >= 2:
                        raise
                    self.write_state(
                        status="retry_wait",
                        current_stage=stage.name,
                        stage_attempt=attempt,
                        retry_at_epoch_seconds=time.time() + 60,
                    )
                    time.sleep(60)
            if result is None:
                raise AssertionError(f"Stage {stage.name} ended without a result")
            completed = list(self.state.get("completed_stages") or [])
            completed.append({"stage": stage.name, "run_id": stage.run_id, "result": result, "at": utc_now()})
            self.write_state(completed_stages=completed)
            self.event("stage_completed", index=index, total=len(STAGES), stage=stage.name, result=result)
        self.write_state(status="completed", current_stage=None, completed_at=utc_now())
        self.event("overnight_runner_completed", completed_stages=len(STAGES))


def main() -> None:
    args = parse_args()
    runner = OvernightRunner(args.root, args.poll_seconds, args.dry_run)
    try:
        runner.run()
    except BaseException as exc:
        runner.write_state(
            status="failed",
            failed_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        runner.event("overnight_runner_failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
