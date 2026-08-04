from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


ROOT = Path("/workspace/safety-dataset")
RUN_ID = "D1-NEMOTRON-GUARD-8B-V3"
MODEL = ROOT / "models" / "llama_3_1_nemotron_safety_guard_8b_v3"
VLLM_PYTHON = Path("/workspace/venvs/nemotron-vllm/bin/python")
PHASE0_STATE = ROOT / "reports" / "overnight_phase0" / "state.json"
INSTALL_STATE = ROOT / "reports" / "decoder_baseline" / "vllm_install_state.json"
STATE = ROOT / "reports" / "decoder_baseline" / "state.json"
EVENTS = ROOT / "reports" / "decoder_baseline" / "events.jsonl"
SMOKE_MANIFEST = ROOT / "data" / "decoder_baseline_full" / "smoke.jsonl"
FULL_MANIFEST = ROOT / "data" / "decoder_baseline_full" / "all_three.jsonl"
TEST_SEA_MANIFEST = ROOT / "data" / "decoder_baseline_full" / "test_and_sea.jsonl"
SMOKE_OUTPUT = ROOT / "reports" / "decoder_baseline" / "smoke_vllm"
FULL_OUTPUT = ROOT / "reports" / "decoder_baseline" / "full_combined"
VLLM_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "name": "high_throughput",
        "gpu_memory_utilization": 0.94,
        "max_num_seqs": 128,
        "max_num_batched_tokens": 16384,
        "request_chunk_size": 1024,
        "enforce_eager": False,
    },
    {
        "name": "balanced",
        "gpu_memory_utilization": 0.92,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 8192,
        "request_chunk_size": 512,
        "enforce_eager": False,
    },
    {
        "name": "balanced_eager",
        "gpu_memory_utilization": 0.92,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 8192,
        "request_chunk_size": 512,
        "enforce_eager": True,
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            count += chunk.count(b"\n")
    return count


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        anchor = datetime.fromisoformat(args.budget_anchor_at.replace("Z", "+00:00"))
        self.anchor_epoch = anchor.timestamp()
        usable_hours = (args.initial_budget_usd - args.reserve_usd) / args.hourly_rate
        self.deadline_epoch = self.anchor_epoch + usable_hours * 3600
        self.state: dict[str, Any] = {
            "status": "initializing",
            "run_id": RUN_ID,
            "started_at": utc_now(),
            "budget_anchor_at": anchor.isoformat(),
            "hourly_rate_usd": args.hourly_rate,
            "initial_budget_usd": args.initial_budget_usd,
            "reserve_usd": args.reserve_usd,
            "current_stage": None,
            "engine": "vllm",
            "precision": "float16",
            "quantization": "none",
            "official_sampling": {
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": 50,
                "max_new_tokens": 100,
            },
        }
        self.selected_profile: dict[str, Any] | None = None
        self.write_state()

    def elapsed_from_anchor(self) -> float:
        return max(0.0, time.time() - self.anchor_epoch)

    def usable_seconds_left(self) -> float:
        return max(0.0, self.deadline_epoch - time.time())

    def estimated_spend(self) -> float:
        return self.elapsed_from_anchor() / 3600 * self.args.hourly_rate

    def write_state(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state["updated_at"] = utc_now()
        self.state["elapsed_seconds"] = self.elapsed_from_anchor()
        self.state["estimated_spend_since_budget_anchor_usd"] = self.estimated_spend()
        self.state["usable_seconds_left"] = self.usable_seconds_left()
        atomic_json(STATE, self.state)

    def event(self, event: str, **fields: Any) -> None:
        EVENTS.parent.mkdir(parents=True, exist_ok=True)
        payload = {"at": utc_now(), "event": event, **fields}
        with EVENTS.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def wait_for_dependencies(self) -> None:
        while True:
            phase0 = read_json(PHASE0_STATE, {})
            install = read_json(INSTALL_STATE, {})
            phase_status = str(phase0.get("status") or "missing")
            install_status = str(install.get("status") or "missing")
            self.write_state(
                status="waiting_for_dependencies",
                current_stage="wait_for_E7_and_vLLM",
                phase0_status=phase_status,
                phase0_current_stage=phase0.get("current_stage"),
                vllm_install_status=install_status,
                vllm_install_stage=install.get("stage"),
            )
            if phase_status == "failed":
                raise RuntimeError(f"Phase-0 failed: {phase0.get('error')}")
            if install_status == "failed":
                raise RuntimeError(f"vLLM install failed: {install.get('error')}")
            if phase_status == "completed" and install_status == "completed":
                self.event("dependencies_ready")
                return
            if self.usable_seconds_left() < self.args.minimum_eval_seconds:
                raise RuntimeError("Budget reserve would be crossed before dependencies became ready")
            time.sleep(self.args.poll_seconds)

    def validate_inputs(self) -> None:
        required = [
            MODEL / "config.json",
            MODEL / "tokenizer.json",
            SMOKE_MANIFEST,
            FULL_MANIFEST,
            TEST_SEA_MANIFEST,
            VLLM_PYTHON,
        ]
        if any(not path.exists() for path in required):
            missing = [str(path) for path in required if not path.exists()]
            raise FileNotFoundError(f"Missing decoder inputs: {missing}")
        if len(list(MODEL.glob("model-*.safetensors"))) != 4:
            raise FileNotFoundError("Expected four full-precision model shards")

    def eval_command(
        self,
        manifest: Path,
        output: Path,
        *,
        resume: bool,
        profile: dict[str, Any],
    ) -> list[str]:
        command = [
            str(VLLM_PYTHON),
            "scripts/evaluate_nemotron_decoder_guard_vllm.py",
            "--model",
            str(MODEL),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--gpu-memory-utilization",
            str(profile["gpu_memory_utilization"]),
            "--max-num-seqs",
            str(profile["max_num_seqs"]),
            "--max-num-batched-tokens",
            str(profile["max_num_batched_tokens"]),
            "--request-chunk-size",
            str(profile["request_chunk_size"]),
            "--seed",
            "3407",
        ]
        if resume:
            command.append("--resume")
        if profile["enforce_eager"]:
            command.append("--enforce-eager")
        return command

    def run_eval(self, name: str, manifest: Path, output: Path) -> dict[str, Any]:
        existing_metrics = read_json(output / "metrics.json")
        if isinstance(existing_metrics, dict) and int(existing_metrics.get("examples") or 0) == line_count(manifest):
            self.event("evaluation_reused", stage=name, examples=existing_metrics.get("examples"))
            return existing_metrics
        output.mkdir(parents=True, exist_ok=True)
        resume = (output / "predictions.jsonl").exists()
        attempts = list(VLLM_PROFILES)
        if self.selected_profile is not None:
            attempts.sort(key=lambda item: item["name"] != self.selected_profile["name"])
        last_code = None
        command: list[str] = []
        for profile in attempts:
            self.write_state(
                status="running",
                current_stage=name,
                current_manifest=str(manifest.relative_to(ROOT)),
                current_output=str(output.relative_to(ROOT)),
                vllm_profile=profile,
            )
            command = self.eval_command(
                manifest,
                output,
                resume=resume,
                profile=profile,
            )
            self.event(
                "evaluation_started",
                stage=name,
                examples=line_count(manifest),
                resume=resume,
                vllm_profile=profile,
            )
            started = time.monotonic()
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    check=False,
                    timeout=max(60, int(self.usable_seconds_left())),
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"{name} reached the budget deadline") from exc
            wall_seconds = time.monotonic() - started
            last_code = result.returncode
            if result.returncode == 0:
                self.selected_profile = profile
                metrics = read_json(output / "metrics.json")
                if not isinstance(metrics, dict):
                    raise RuntimeError(f"{name} finished without metrics.json")
                metrics["orchestrator_wall_seconds"] = wall_seconds
                atomic_json(output / "metrics.json", metrics)
                self.event(
                    "evaluation_completed",
                    stage=name,
                    examples=metrics.get("examples"),
                    wall_seconds=wall_seconds,
                    runtime=metrics.get("runtime"),
                    format=metrics.get("format"),
                    vllm_profile=profile,
                )
                return metrics
            self.event(
                "evaluation_attempt_failed",
                stage=name,
                returncode=result.returncode,
                vllm_profile=profile,
                wall_seconds=wall_seconds,
            )
            resume = (output / "predictions.jsonl").exists()
        raise subprocess.CalledProcessError(int(last_code or 1), command)

    def choose_full_manifest(self, smoke_metrics: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        runtime = smoke_metrics.get("runtime") or {}
        rate = float(runtime.get("examples_per_second") or 0.0)
        load_seconds = float(runtime.get("model_load_seconds") or 0.0)
        if rate <= 0:
            raise RuntimeError("Smoke test did not produce a usable throughput estimate")
        smoke_examples = int(smoke_metrics.get("examples") or 0)
        options = []
        for name, manifest in (("all_three", FULL_MANIFEST), ("test_and_sea", TEST_SEA_MANIFEST)):
            examples = line_count(manifest)
            projected = load_seconds + max(0, examples - smoke_examples) / rate * 1.5 + 5 * 60
            options.append(
                {
                    "name": name,
                    "manifest": str(manifest.relative_to(ROOT)),
                    "examples": examples,
                    "projected_seconds_with_1_5x_guard": projected,
                }
            )
        available = self.usable_seconds_left()
        self.event("full_scope_projection", available_seconds=available, options=options)
        for option, manifest in zip(options, (FULL_MANIFEST, TEST_SEA_MANIFEST)):
            if float(option["projected_seconds_with_1_5x_guard"]) <= available:
                return manifest, {"selected": option, "options": options, "available_seconds": available}
        raise RuntimeError(
            "Neither all-three nor test+SEA full evaluation fits the guarded remaining budget"
        )

    def run(self) -> None:
        self.wait_for_dependencies()
        self.validate_inputs()
        smoke_metrics = self.run_eval("smoke_vllm", SMOKE_MANIFEST, SMOKE_OUTPUT)
        manifest, projection = self.choose_full_manifest(smoke_metrics)
        self.write_state(
            status="running",
            current_stage="prepare_full_vllm",
            full_scope=projection,
        )
        FULL_OUTPUT.mkdir(parents=True, exist_ok=True)
        full_predictions = FULL_OUTPUT / "predictions.jsonl"
        if not full_predictions.exists():
            shutil.copy2(SMOKE_OUTPUT / "predictions.jsonl", full_predictions)
        full_metrics = self.run_eval("full_vllm", manifest, FULL_OUTPUT)

        self.write_state(status="running", current_stage="split_three_benchmarks")
        split_command = [
            str(VLLM_PYTHON),
            "scripts/split_decoder_full_matrix.py",
            "--combined-dir",
            str(FULL_OUTPUT),
            "--run-id",
            RUN_ID,
        ]
        subprocess.run(split_command, cwd=ROOT, check=True)
        matrix_index = read_json(
            ROOT / "reports" / "evaluation_matrix" / RUN_ID / "matrix_index.json", {}
        )
        summary = {
            "run_id": RUN_ID,
            "model": "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3",
            "comparison_role": "zero-shot pretrained full-precision decoder baseline",
            "architecture": "Llama-3.1-8B causal decoder",
            "training_in_this_project": False,
            "engine": "vllm",
            "precision": "float16 on Turing",
            "quantization": None,
            "official_prompt": True,
            "official_sampling": self.state["official_sampling"],
            "smoke": smoke_metrics,
            "projection": projection,
            "full": full_metrics,
            "matrix_index": matrix_index,
            "budget": {
                "anchor_at": self.state["budget_anchor_at"],
                "initial_usd": self.args.initial_budget_usd,
                "reserve_usd": self.args.reserve_usd,
                "hourly_rate_usd": self.args.hourly_rate,
                "estimated_spend_since_anchor_usd": self.estimated_spend(),
            },
            "completed_at": utc_now(),
        }
        atomic_json(ROOT / "reports" / "decoder_baseline" / "summary.json", summary)
        self.write_state(
            status="completed",
            current_stage=None,
            completed_at=utc_now(),
            evaluation_jobs=len(matrix_index.get("jobs") or []),
            final_examples=int(full_metrics.get("examples") or 0),
        )
        self.event(
            "decoder_baseline_completed",
            jobs=self.state["evaluation_jobs"],
            examples=self.state["final_examples"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Budget-bounded full vLLM baseline after the E7 matrix completes."
    )
    parser.add_argument("--hourly-rate", type=float, default=0.159)
    parser.add_argument("--initial-budget-usd", type=float, default=2.28)
    parser.add_argument("--reserve-usd", type=float, default=0.50)
    parser.add_argument(
        "--budget-anchor-at",
        default="2026-07-22T08:19:19.099109+00:00",
        help="UTC time when the user reported the remaining budget; waiting time counts.",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--minimum-eval-seconds", type=int, default=20 * 60)
    args = parser.parse_args()
    if args.hourly_rate <= 0:
        raise ValueError("hourly-rate must be positive")
    if not 0 <= args.reserve_usd < args.initial_budget_usd:
        raise ValueError("reserve must be non-negative and less than the initial budget")
    return args


def main() -> None:
    runner = Runner(parse_args())
    try:
        runner.run()
    except BaseException as exc:
        runner.write_state(
            status="failed",
            failed_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        runner.event("decoder_baseline_failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
