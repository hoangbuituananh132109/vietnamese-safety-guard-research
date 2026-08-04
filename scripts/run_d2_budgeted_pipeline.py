from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "d2_nemotron"
TRAIN_PYTHON = Path("/workspace/venvs/nemotron-train/bin/python")
VLLM_PYTHON = Path("/workspace/venvs/nemotron-vllm/bin/python")
MODEL = ROOT / "models" / "llama_3_1_nemotron_safety_guard_8b_v3"
SOURCE = ROOT / "data" / "guard_full" / "train.jsonl"
BASELINE = ROOT / "reports" / "decoder_baseline" / "full_combined" / "predictions.jsonl"


def atomic_state(**values: object) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / "state.json"
    previous: dict[str, object] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    previous.update(values)
    if values.get("status") in {"running", "complete"}:
        for stale_key in ("failed_at", "error", "traceback"):
            previous.pop(stale_key, None)
    if values.get("status") == "running" and not str(values.get("stage", "")).endswith("_finished"):
        previous.pop("exit_code", None)
    previous["updated_at"] = time.time()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def event(stage: str, **values: object) -> None:
    value = {"stage": stage, "time": time.time(), **values}
    with (REPORT / "pipeline_events.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    atomic_state(status="running", stage=stage, **values)


def run(stage: str, command: list[str]) -> None:
    event(stage, command=command)
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, check=False)
    elapsed = time.monotonic() - started
    event(stage + "_finished", exit_code=result.returncode, elapsed_seconds=elapsed)
    if result.returncode != 0:
        raise RuntimeError(f"{stage} failed with exit code {result.returncode}")


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    atomic_state(status="running", stage="starting", started_at=started)
    main_manifest = REPORT / "manifests" / "paired_8192.jsonl"
    smoke_dir = REPORT / "smoke_train"
    smoke_eval = REPORT / "smoke_eval"
    main_dir = REPORT / "main_train"
    full_eval = REPORT / "full_eval"
    try:
        run(
            "build_manifest",
            [
                str(TRAIN_PYTHON),
                str(ROOT / "scripts" / "build_decoder_sft_pilot.py"),
                "--source", str(SOURCE),
                "--output", str(main_manifest),
                "--rows", "8192",
                "--seed", "3407",
            ],
        )
        run(
            "smoke_train",
            [
                str(TRAIN_PYTHON),
                str(ROOT / "scripts" / "train_nemotron_guard_qlora.py"),
                "--model", str(MODEL),
                "--source", str(SOURCE),
                "--manifest", str(main_manifest),
                "--output-dir", str(smoke_dir),
                "--limit", "256",
                "--max-seq-length", "1024",
                "--gradient-accumulation", "16",
                "--max-wall-seconds", "2400",
                "--save-every", "8",
            ],
        )
        run(
            "smoke_eval",
            [
                str(VLLM_PYTHON),
                str(ROOT / "scripts" / "evaluate_nemotron_decoder_guard_vllm.py"),
                "--model", str(MODEL),
                "--lora-adapter", str(smoke_dir / "final_adapter"),
                "--manifest", str(ROOT / "data" / "decoder_baseline_full" / "smoke.jsonl"),
                "--output-dir", str(smoke_eval),
                "--request-chunk-size", "400",
                "--max-num-seqs", "64",
                "--gpu-memory-utilization", "0.90",
            ],
        )
        run(
            "smoke_compare",
            [
                str(VLLM_PYTHON),
                str(ROOT / "scripts" / "compare_d2_predictions.py"),
                "--baseline", str(BASELINE),
                "--candidate", str(smoke_eval / "predictions.jsonl"),
                "--output-json", str(smoke_eval / "comparison.json"),
                "--output-md", str(smoke_eval / "comparison.md"),
            ],
        )
        run(
            "main_train",
            [
                str(TRAIN_PYTHON),
                str(ROOT / "scripts" / "train_nemotron_guard_qlora.py"),
                "--model", str(MODEL),
                "--source", str(SOURCE),
                "--manifest", str(main_manifest),
                "--output-dir", str(main_dir),
                "--max-seq-length", "1024",
                "--gradient-accumulation", "16",
                "--max-steps", "512",
                "--max-wall-seconds", "21600",
                "--save-every", "100",
            ],
        )
        run(
            "full_eval",
            [
                str(VLLM_PYTHON),
                str(ROOT / "scripts" / "evaluate_nemotron_decoder_guard_vllm.py"),
                "--model", str(MODEL),
                "--lora-adapter", str(main_dir / "final_adapter"),
                "--manifest", str(ROOT / "data" / "decoder_baseline_full" / "test_and_sea.jsonl"),
                "--output-dir", str(full_eval),
                "--request-chunk-size", "512",
                "--max-num-seqs", "64",
                "--gpu-memory-utilization", "0.90",
            ],
        )
        run(
            "full_compare",
            [
                str(VLLM_PYTHON),
                str(ROOT / "scripts" / "compare_d2_predictions.py"),
                "--baseline", str(BASELINE),
                "--candidate", str(full_eval / "predictions.jsonl"),
                "--output-json", str(full_eval / "comparison.json"),
                "--output-md", str(full_eval / "comparison.md"),
            ],
        )
        atomic_state(
            status="complete",
            stage="complete",
            completed_at=time.time(),
            elapsed_seconds=time.time() - started,
        )
    except BaseException as error:
        atomic_state(
            status="failed",
            stage="failed",
            failed_at=time.time(),
            elapsed_seconds=time.time() - started,
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
