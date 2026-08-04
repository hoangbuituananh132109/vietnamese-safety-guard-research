from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path("/workspace/safety-dataset")
RUN_ROOT = ROOT / "results/no_r_decoder_4080s"
PIPELINE_STATE = RUN_ROOT / "pipeline_state.json"

RUNS = (
    {
        "id": "Q2-QWEN3GUARD-4B-NR-EV",
        "short_id": "Q2",
        "name": "Qwen3Guard 4B · no-R EN+VI LoRA",
        "kind": "qwen",
        "model_kind": "decoder_guard_binary_lora",
        "languages": "EN+VI",
        "context": 2048,
        "train_examples": 101274,
        "category_tasks": False,
    },
    {
        "id": "D3-NEMOTRON-8B-NR-VI",
        "short_id": "D3",
        "name": "Nemotron Guard 8B · no-R Vietnamese LoRA",
        "kind": "nemotron",
        "model_kind": "decoder_guard_n23_lora",
        "languages": "VI",
        "context": 2048,
        "train_examples": 50637,
        "category_tasks": True,
    },
)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                values.append(value)
    return values


def finite(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def downsample(rows: list[dict[str, Any]], limit: int = 750) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    stride = math.ceil(len(rows) / limit)
    result = rows[::stride]
    if result[-1] is not rows[-1]:
        result.append(rows[-1])
    return result


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        return {"available": False, "error": result.stderr.strip()}
    fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
    try:
        return {
            "available": True,
            "name": fields[0],
            "utilization_percent": float(fields[1]),
            "memory_used_mb": float(fields[2]),
            "memory_total_mb": float(fields[3]),
            "temperature_c": float(fields[4]),
            "power_w": float(fields[5]),
        }
    except (IndexError, ValueError):
        return {"available": False, "error": result.stdout.strip()}


def supervisor_snapshot() -> list[dict[str, str]]:
    names = (
        "no-r-decoder-pipeline",
        "decoder-download-qwen",
        "decoder-download-nemotron",
        "decoder-eval-setup",
    )
    result = subprocess.run(
        ["supervisorctl", "status", *names],
        capture_output=True,
        text=True,
        check=False,
    )
    values: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2:
            values.append(
                {
                    "name": parts[0],
                    "status": parts[1].casefold(),
                    "detail": parts[2] if len(parts) > 2 else "",
                }
            )
    return values


def compact_binary(metrics: Any) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    wanted = (
        "examples",
        "accuracy",
        "macro_f1",
        "safe_precision",
        "safe_recall",
        "safe_f1",
        "safe_support",
        "unsafe_precision",
        "unsafe_recall",
        "unsafe_f1",
        "unsafe_support",
        "unsafe_auprc",
        "average_precision",
        "auroc",
        "roc_auc",
        "brier",
        "brier_score",
        "ece_10",
    )
    result = {key: finite(metrics.get(key)) for key in wanted if key in metrics}
    confusion = metrics.get("confusion_matrix")
    if isinstance(confusion, dict):
        result["confusion_matrix"] = {
            key: int(confusion.get(key, 0)) for key in ("tn", "fp", "fn", "tp")
        }
    return result


def compact_n23(metrics: Any) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    wanted = (
        "supervised_examples",
        "labels",
        "labels_with_support",
        "threshold",
        "exact_match_accuracy",
        "hamming_error_rate",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_f1_supported",
        "samples_f1",
        "micro_auprc",
        "macro_auprc_supported",
        "gold_positive_labels",
        "predicted_positive_labels",
    )
    result: dict[str, Any] = {
        key: finite(metrics.get(key)) for key in wanted if key in metrics
    }
    per_label = metrics.get("per_label")
    if isinstance(per_label, dict):
        result["per_label"] = {
            str(label): {
                key: finite(values.get(key))
                for key in (
                    "support",
                    "predicted_positive",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                    "auprc",
                )
                if isinstance(values, dict) and key in values
            }
            for label, values in per_label.items()
            if isinstance(values, dict)
        }
    language = metrics.get("slices", {}).get("language")
    if isinstance(language, dict):
        result["slices"] = {
            "language": {
                str(code): compact_n23(values)
                for code, values in language.items()
                if isinstance(values, dict)
            }
        }
    return result


def evaluation_job(kind: str) -> tuple[list[dict[str, Any]], float]:
    output = RUN_ROOT / f"eval/{kind}"
    metrics = read_json(output / "metrics.json", {})
    if isinstance(metrics, dict) and metrics:
        if kind == "qwen":
            report = metrics.get("binary_primary_conservative", {}).get("metrics", {})
            n23 = None
        else:
            report = metrics.get("binary", {})
            n23 = compact_n23(metrics.get("N23"))
        overall = compact_binary(report.get("overall"))
        slices = report.get("slices", {}) if isinstance(report, dict) else {}
        language = slices.get("language", {}) if isinstance(slices, dict) else {}
        job = {
            "id": f"{kind}-no-r-test-and-sea",
            "benchmark": "Nemotron test + SEA",
            "suite": "no_r_p_pr",
            "label_order": "canonical",
            "binary": overall,
            "language": {
                str(code): compact_binary(value)
                for code, value in language.items()
                if isinstance(value, dict)
            },
            "N23": n23,
            "paired_en_vi": None,
        }
        return [job], 1.0
    progress = read_jsonl(output / "progress.jsonl")
    fraction = finite(progress[-1].get("progress")) if progress else 0.0
    return [], float(fraction or 0.0)


def active_kind(current_stage: str | None) -> str | None:
    value = str(current_stage or "")
    if value.startswith("qwen") or value == "wait_download_qwen":
        return "qwen"
    if value.startswith("nemotron") or value == "wait_download_nemotron":
        return "nemotron"
    return None


def experiment(spec: dict[str, Any], pipeline: dict[str, Any]) -> dict[str, Any]:
    kind = str(spec["kind"])
    train_root = RUN_ROOT / f"train/{kind}"
    current_stage = str(pipeline.get("current_stage") or "")
    smoke_root = RUN_ROOT / f"smoke/{kind}"
    full_progress = read_json(train_root / "progress.json", {})
    smoke_active = current_stage == f"{kind}_trainer_smoke"
    progress = (
        read_json(smoke_root / "progress.json", {})
        if smoke_active and not (train_root / "completed.marker").exists()
        else full_progress
    )
    contract = read_json(train_root / "run_contract.json", {})
    history_root = smoke_root if smoke_active and not full_progress else train_root
    history_raw = read_jsonl(history_root / "train_log.jsonl")
    history = downsample(
        [
            {
                "step": int(item.get("global_step") or 0),
                "epoch": int(item.get("epoch_index") or 0) + 1,
                "loss": finite(item.get("loss")),
                "binary_loss": None,
                "category_loss": None,
                "throughput": None,
                "max_tokens": None,
            }
            for item in history_raw
            if int(item.get("global_step") or 0) > 0
        ]
    )
    current_kind = active_kind(pipeline.get("current_stage"))
    completed_marker = (train_root / "completed.marker").exists()
    failed = (train_root / "failure.json").exists() and not completed_marker
    if completed_marker:
        train_status = "completed"
    elif current_kind == kind and current_stage in {
        f"{kind}_stress_mb4",
        f"{kind}_stress_mb2",
        f"{kind}_trainer_smoke",
        f"{kind}_full_train",
    }:
        train_status = "running"
    elif failed:
        train_status = "failed"
    else:
        train_status = "pending"
    completed_steps = int(progress.get("global_step") or 0)
    planned_steps = int(progress.get("total_updates") or 0)
    fraction = finite(progress.get("progress_fraction"))
    if completed_marker:
        fraction = 1.0
    if fraction is None:
        fraction = completed_steps / max(1, planned_steps) if planned_steps else 0.0
    epoch_index = int(progress.get("epoch_index") or 0)
    current_epoch = min(1, epoch_index + int(completed_steps > 0))
    elapsed = finite(progress.get("elapsed_seconds_this_process"))
    eta = finite(progress.get("eta_seconds"))
    feature_audit = (
        contract.get("dataset", {}).get("feature_audit", {})
        if isinstance(contract, dict)
        else {}
    )
    jobs, eval_progress = evaluation_job(kind)
    evaluating = str(pipeline.get("current_stage") or "") == f"{kind}_eval"
    eval_done = int(bool(jobs))
    checkpoints = sorted(train_root.glob("checkpoint-*/complete.marker"))
    latest_checkpoint = (
        checkpoints[-1].parent.name if checkpoints else None
    )
    return {
        **spec,
        "train_status": train_status,
        "planned_steps": planned_steps,
        "completed_steps": completed_steps,
        "progress": min(max(float(fraction), 0.0), 1.0),
        "epochs": 1,
        "current_epoch": current_epoch,
        "epoch_progress": min(max(float(fraction), 0.0), 1.0),
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "projected_total_seconds": (
            float(elapsed) + float(eta)
            if elapsed is not None and eta is not None
            else None
        ),
        "peak_vram_mb": finite(progress.get("peak_vram_mb")),
        "valid_examples": 11736,
        "truncated_train": int(feature_audit.get("truncated") or 0),
        "truncated_valid": None,
        "nonfinite_updates": 0,
        "encoder_gradient_verified": None,
        "validation": None,
        "validation_N23": None,
        "validation_history": [],
        "history": history,
        "latest_loss": finite(progress.get("loss")),
        "actual_tokens": int(progress.get("actual_tokens") or 0),
        "padded_tokens": int(progress.get("padded_tokens") or 0),
        "padding_overhead_fraction": finite(
            progress.get("padding_overhead_fraction")
        ),
        "checkpoint_count": len(checkpoints),
        "latest_checkpoint": latest_checkpoint,
        "evaluation": {
            "done": eval_done,
            "total": 1,
            "progress": 1.0 if eval_done else min(max(eval_progress, 0.0), 1.0),
            "jobs": jobs,
            "evaluating": evaluating,
        },
    }


def stage_list(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    order = (
        ("download_qwen", "Qwen checkpoint"),
        ("qwen_stress_mb4", "Qwen VRAM stress"),
        ("qwen_trainer_smoke", "Qwen smoke"),
        ("qwen_full_train", "Qwen full train"),
        ("qwen_eval", "Qwen evaluation"),
        ("download_nemotron", "Nemotron checkpoint"),
        ("nemotron_stress_mb4", "Nemotron VRAM stress"),
        ("nemotron_stress_mb2", "Nemotron VRAM fallback"),
        ("nemotron_trainer_smoke", "Nemotron smoke"),
        ("nemotron_full_train", "Nemotron full train"),
        ("nemotron_eval", "Nemotron evaluation"),
    )
    stages = pipeline.get("stages", {}) if isinstance(pipeline, dict) else {}
    downloads = {
        "download_qwen": read_json(ROOT / "results/download_qwen.json", {}),
        "download_nemotron": read_json(ROOT / "results/download_nemotron.json", {}),
    }
    result: list[dict[str, Any]] = []
    mb2_status = str(stages.get("nemotron_stress_mb2", {}).get("status") or "")
    for key, label in order:
        source = downloads.get(key) or stages.get(key) or {}
        status = str(source.get("status") or "pending")
        if (
            key == "nemotron_stress_mb4"
            and status == "failed"
            and mb2_status in {"running", "completed"}
        ):
            status = "fallback"
            detail = "Không vừa VRAM · tự chuyển MB2"
        if status == "fallback":
            pass
        elif key.startswith("download_") and status == "completed":
            detail = (
                f"{float(source.get('bytes', 0)) / (1024**3):.1f} GiB · "
                f"{float(source.get('effective_mib_per_second', 0)):.1f} MiB/s"
            )
        elif source:
            detail = (
                f"attempt {source.get('attempt')}"
                if source.get("attempt") is not None
                else ""
            )
        else:
            detail = ""
        result.append(
            {
                "key": key,
                "label": label,
                "status": status,
                "detail": detail,
            }
        )
    return result


def main() -> None:
    pipeline = read_json(PIPELINE_STATE, {})
    if not isinstance(pipeline, dict):
        pipeline = {}
    experiments = [experiment(spec, pipeline) for spec in RUNS]
    kind = active_kind(pipeline.get("current_stage"))
    current_run_id = next(
        (item["id"] for item in experiments if item["kind"] == kind),
        None,
    )
    current = next(
        (item for item in experiments if item["id"] == current_run_id),
        None,
    )
    status = str(pipeline.get("status") or "pending")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": {
            "status": status,
            "started_at": pipeline.get("created_at"),
            "updated_at": pipeline.get("updated_at"),
            "current_stage": pipeline.get("current_stage"),
            "current_run_id": current_run_id,
            "stage_index": None,
            "stage_count": 11,
            "stage_elapsed_seconds": (
                current.get("elapsed_seconds") if current else None
            ),
            "child_pid": (
                pipeline.get("stages", {})
                .get(str(pipeline.get("current_stage")), {})
                .get("pid")
                if isinstance(pipeline.get("stages"), dict)
                else None
            ),
            "error": pipeline.get("failure"),
            "failed_at": pipeline.get("failed_at"),
            "completed_at": pipeline.get("completed_at"),
        },
        "gpu": gpu_snapshot(),
        "summary": {
            "experiments": len(experiments),
            "training_complete": sum(
                item["train_status"] == "completed" for item in experiments
            ),
            "evaluation_complete": sum(
                item["evaluation"]["done"] >= item["evaluation"]["total"]
                for item in experiments
            ),
            "current_run_id": current_run_id,
            "current_progress": current.get("progress") if current else None,
            "current_eta_seconds": current.get("eta_seconds") if current else None,
        },
        "experiments": experiments,
        "pipeline": {
            "contract": pipeline.get("contract"),
            "stages": stage_list(pipeline),
            "supervisor": supervisor_snapshot(),
            "eval_environment_ready": (ROOT / ".eval_setup_complete").exists(),
            "auto_stop": pipeline.get("auto_stop"),
        },
    }
    print(
        "DASHBOARD_SNAPSHOT="
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
