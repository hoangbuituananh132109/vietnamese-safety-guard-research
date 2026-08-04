from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable


ROOT = Path("/workspace/safety-dataset")
CONFIG_PATH = ROOT / os.environ.get("PHASE0_CONFIG_PATH", "configs/phase0_experiments.json")
STATE_PATH = ROOT / os.environ.get("PHASE0_STATE_PATH", "reports/overnight_phase0/state.json")
DECODER_STATE_PATH = ROOT / "reports" / "decoder_baseline" / "state.json"
D2_STATE_PATH = ROOT / "reports" / "d2_nemotron" / "state.json"
D2_REPORT_ROOT = ROOT / "reports" / "d2_nemotron"
COUNT_CACHE_PATH = ROOT / os.environ.get(
    "PHASE0_COUNT_CACHE_PATH",
    "reports/training_dashboard/manifest_counts.json",
)
EXPERIMENT_RUN_ROOT = ROOT / os.environ.get(
    "PHASE0_EXPERIMENT_RUN_ROOT",
    "reports/experiment_runs",
)
EVALUATION_MATRIX_ROOT = ROOT / os.environ.get(
    "PHASE0_EVALUATION_MATRIX_ROOT",
    "reports/evaluation_matrix",
)

RUN_ORDER = [
    "B0-GLI-ZS",
    "E1-G-EV-512",
    "E2-M-EV-GLI-COMPAT-8K",
    "E3-M-E-8K",
    "E4-M-EV-MATCHED-8K",
    "E5-M-EV-FULL-8K",
    "E7-M-SCHEMA-EV-8K",
    "D1-NEMOTRON-GUARD-8B-V3",
    "D2-NEMOTRON-GUARD-8B-VI-PILOT",
]

RUN_NAMES = {
    "B0-GLI-ZS": "GLiGuard zero-shot",
    "E1-G-EV-512": "GLiGuard EN+VI 512",
    "E2-M-EV-GLI-COMPAT-8K": "mmBERT EN+VI GLi-compatible",
    "E3-M-E-8K": "mmBERT English-only 8K",
    "E4-M-EV-MATCHED-8K": "mmBERT matched EN+VI 8K",
    "E5-M-EV-FULL-8K": "mmBERT full EN+VI 8K",
    "E7-M-SCHEMA-EV-8K": "mmBERT dynamic schema 8K",
    "D1-NEMOTRON-GUARD-8B-V3": "Nemotron Guard 8B · vLLM full FP16",
    "D2-NEMOTRON-GUARD-8B-VI-PILOT": "Nemotron Guard 8B · EN–VI QLoRA pilot",
}
if os.environ.get("PHASE0_RUN_ORDER"):
    RUN_ORDER = [
        item.strip()
        for item in os.environ["PHASE0_RUN_ORDER"].split(",")
        if item.strip()
    ]
if os.environ.get("PHASE0_RUN_NAMES_JSON"):
    RUN_NAMES.update(json.loads(os.environ["PHASE0_RUN_NAMES_JSON"]))


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
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
        "auroc",
        "roc_auc",
        "average_precision",
        "brier",
        "brier_score",
        "ece_10",
    )
    result = {key: finite_number(metrics.get(key)) for key in wanted if key in metrics}
    confusion = metrics.get("confusion_matrix")
    if isinstance(confusion, dict):
        result["confusion_matrix"] = {
            key: int(confusion.get(key, 0)) for key in ("tn", "fp", "fn", "tp")
        }
    return result


def compact_n23_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
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
        key: finite_number(metrics.get(key)) for key in wanted if key in metrics
    }
    per_label = metrics.get("per_label")
    if isinstance(per_label, dict):
        label_fields = ("support", "predicted_positive", "tp", "fp", "fn", "precision", "recall", "f1", "auprc")
        result["per_label"] = {
            str(label): {
                key: finite_number(values.get(key))
                for key in label_fields
                if isinstance(values, dict) and key in values
            }
            for label, values in per_label.items()
            if isinstance(values, dict)
        }
    slices = metrics.get("slices")
    language = slices.get("language") if isinstance(slices, dict) else None
    if isinstance(language, dict):
        result["slices"] = {
            "language": {
                str(code): compact_n23_metrics(values)
                for code, values in language.items()
                if isinstance(values, dict)
            }
        }
    return result


def confusion_from_predictions(predictions: Any) -> dict[str, int] | None:
    if not isinstance(predictions, list) or not predictions:
        return None
    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    seen = 0
    for item in predictions:
        if not isinstance(item, dict):
            continue
        target = item.get("target")
        prediction = item.get("prediction")
        if target not in (0, 1) or prediction not in (0, 1):
            continue
        seen += 1
        if target == 0 and prediction == 0:
            counts["tn"] += 1
        elif target == 0 and prediction == 1:
            counts["fp"] += 1
        elif target == 1 and prediction == 0:
            counts["fn"] += 1
        else:
            counts["tp"] += 1
    return counts if seen else None


def load_manifest_counts(config: dict[str, Any]) -> dict[str, int]:
    cache = read_json(COUNT_CACHE_PATH, {})
    if not isinstance(cache, dict):
        cache = {}
    changed = False
    for run in config.get("runs", []):
        manifest = run.get("train_manifest")
        if not isinstance(manifest, str) or manifest in cache:
            continue
        path = ROOT / manifest
        if not path.exists():
            continue
        count = 0
        with path.open("rb") as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                count += chunk.count(b"\n")
        cache[manifest] = count
        changed = True
    if changed:
        COUNT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        COUNT_CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return {str(key): int(value) for key, value in cache.items()}


def planned_steps(run: dict[str, Any], counts: dict[str, int]) -> tuple[int, int]:
    explicit = int(run.get("max_optimizer_steps") or 0)
    epochs = int(run.get("epochs") or 0)
    if explicit:
        return explicit, math.ceil(explicit / max(epochs, 1))
    manifest = run.get("train_manifest")
    examples = int(counts.get(str(manifest), 0))
    if not examples or not epochs:
        return 0, 0
    micro_batch = int(run.get("micro_batch_size") or 8)
    effective_batch = 32
    accumulation = max(1, math.ceil(effective_batch / micro_batch))
    batches = math.ceil(examples / micro_batch)
    per_epoch = math.ceil(batches / accumulation)
    return per_epoch * epochs, per_epoch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def downsample(rows: list[dict[str, Any]], limit: int = 700) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    stride = math.ceil(len(rows) / limit)
    sampled = rows[::stride]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled


def gliguard_live_history(stage_log: Path) -> list[dict[str, Any]]:
    try:
        raw = stage_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    by_step: dict[int, dict[str, Any]] = {}
    last_loss: float | None = None
    for segment in re.split(r"[\r\n]+", raw):
        if "Training:" not in segment:
            continue
        step_match = re.search(r"\|\s*(\d+)/(\d+)\s*\[", segment)
        if not step_match:
            continue
        step = int(step_match.group(1))
        loss_match = re.search(r"\bloss=([0-9.eE+-]+)", segment)
        throughput_match = re.search(r"\bsamples/s=([0-9.eE+-]+)", segment)
        epoch_match = re.search(r"\bepoch=([0-9.eE+-]+)", segment)
        if loss_match:
            last_loss = float(loss_match.group(1))
        by_step[step] = {
            "step": step,
            "epoch": float(epoch_match.group(1)) if epoch_match else None,
            "loss": last_loss,
            "binary_loss": last_loss,
            "category_loss": None,
            "throughput": float(throughput_match.group(1)) if throughput_match else None,
            "max_tokens": None,
        }
    return downsample([by_step[key] for key in sorted(by_step)])


def normalized_history(
    run_id: str,
    model_kind: str,
    metrics: dict[str, Any] | None,
    run_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if model_kind == "gliguard_schema" and isinstance(metrics, dict):
        history = metrics.get("train_result", {}).get("train_metrics_history", [])
        for item in history:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "step": int(item.get("step") or 0),
                    "epoch": finite_number(item.get("epoch")),
                    "loss": finite_number(item.get("loss")),
                    "binary_loss": finite_number(item.get("classification_loss")),
                    "category_loss": None,
                    "throughput": finite_number(item.get("throughput")),
                    "max_tokens": None,
                }
            )
    else:
        raw_rows = read_jsonl(run_dir / "train_log.jsonl")
        if not raw_rows and isinstance(metrics, dict):
            raw_rows = metrics.get("history_tail", [])
        by_step: dict[int, dict[str, Any]] = {}
        for item in raw_rows:
            step = int(item.get("optimizer_step") or item.get("step") or 0)
            if not step:
                continue
            by_step[step] = {
                "step": step,
                "epoch": finite_number(item.get("epoch")),
                "loss": finite_number(item.get("loss")),
                "binary_loss": finite_number(item.get("binary_loss")),
                "category_loss": finite_number(item.get("category_loss")),
                "throughput": finite_number(item.get("throughput")),
                "max_tokens": finite_number(item.get("max_tokens")),
            }
        rows = [by_step[key] for key in sorted(by_step)]
    return downsample(rows)


def evaluation_jobs(run_id: str, expected: int) -> tuple[list[dict[str, Any]], int]:
    matrix_root = EVALUATION_MATRIX_ROOT / run_id
    index = read_json(matrix_root / "matrix_index.json", {})
    indexed_jobs = index.get("jobs", []) if isinstance(index, dict) else []
    candidates: list[tuple[str, Path, dict[str, Any]]] = []
    if indexed_jobs:
        for job in indexed_jobs:
            metric_path = ROOT / str(job.get("metrics"))
            name = metric_path.parent.name
            candidates.append((name, metric_path, job))
    elif matrix_root.exists():
        for metric_path in sorted(matrix_root.glob("*/metrics.json")):
            candidates.append((metric_path.parent.name, metric_path, {}))

    jobs: list[dict[str, Any]] = []
    for name, metric_path, indexed in candidates:
        data = read_json(metric_path, {})
        if not isinstance(data, dict):
            continue
        binary = data.get("binary", {})
        overall = binary.get("overall") if isinstance(binary, dict) else None
        paired = binary.get("paired_en_vi") if isinstance(binary, dict) else None
        slices = binary.get("slices", {}) if isinstance(binary, dict) else {}
        language_slices = slices.get("language", {}) if isinstance(slices, dict) else {}
        n23 = data.get("N23") if isinstance(data.get("N23"), dict) else None
        parts = name.split("__")
        suite = indexed.get("suite") or (parts[0] if parts else "")
        benchmark = indexed.get("benchmark") or (parts[1] if len(parts) > 1 else name)
        label_order = indexed.get("label_order") or (parts[2] if len(parts) > 2 else "canonical")
        jobs.append(
            {
                "id": name,
                "suite": suite,
                "benchmark": benchmark,
                "label_order": label_order,
                "examples": int(data.get("examples") or indexed.get("examples") or 0),
                "truncated_examples": int(data.get("truncated_examples") or 0),
                "mean_loss": finite_number(data.get("mean_loss")),
                "binary": compact_metrics(overall),
                "language": {
                    str(language): compact_metrics(metrics)
                    for language, metrics in language_slices.items()
                    if isinstance(metrics, dict)
                }
                if isinstance(language_slices, dict)
                else {},
                "paired_en_vi": {
                    key: finite_number(paired.get(key))
                    for key in ("pairs", "same_decision_rate", "mean_probability_gap")
                    if isinstance(paired, dict) and key in paired
                },
                "N23": compact_n23_metrics(n23),
                "runtime": {
                    key: finite_number(data.get("runtime", {}).get(key))
                    for key in ("elapsed_seconds", "examples_per_second", "peak_vram_mb")
                    if isinstance(data.get("runtime"), dict) and key in data.get("runtime", {})
                },
                "format": {
                    key: data.get("format", {}).get(key)
                    for key in ("parsed", "parse_failures", "strict_json", "unknown_category_outputs")
                    if isinstance(data.get("format"), dict) and key in data.get("format", {})
                },
            }
        )
    return jobs, max(expected, len(jobs))


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "error": result.stderr.strip()}
    values = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    try:
        return {
            "available": True,
            "name": values[0],
            "utilization_percent": float(values[1]),
            "memory_used_mb": float(values[2]),
            "memory_total_mb": float(values[3]),
            "temperature_c": float(values[4]),
            "power_w": float(values[5]),
        }
    except (IndexError, ValueError):
        return {"available": False, "error": result.stdout.strip()}


def experiment_summary(
    run: dict[str, Any],
    state: dict[str, Any],
    counts: dict[str, int],
    reference_seconds_per_step: float | None,
) -> dict[str, Any]:
    run_id = str(run["id"])
    run_dir = EXPERIMENT_RUN_ROOT / run_id
    metrics = read_json(run_dir / "metrics.json")
    planned, per_epoch = planned_steps(run, counts)
    history = normalized_history(run_id, str(run.get("model_kind") or ""), metrics, run_dir)
    if (
        not history
        and run.get("model_kind") == "gliguard_schema"
        and state.get("current_run_id") == run_id
    ):
        stage_log = state.get("stage_log")
        if stage_log:
            history = gliguard_live_history(ROOT / str(stage_log))
    validation_history: list[dict[str, Any]] = []

    completed_steps = 0
    elapsed_seconds = None
    peak_vram_mb = None
    validation = None
    validation_n23 = None
    train_examples = counts.get(str(run.get("train_manifest")))
    valid_examples = None
    truncated_train = None
    truncated_valid = None
    nonfinite_updates = None
    encoder_gradient_verified = None
    train_status = "not_applicable" if not run.get("train") else "pending"

    if isinstance(metrics, dict):
        if run.get("model_kind") == "gliguard_schema":
            result = metrics.get("train_result", {})
            completed_steps = int(result.get("total_steps") or 0)
            elapsed_seconds = finite_number(result.get("total_time_seconds"))
            validation = compact_metrics(metrics.get("validation", {}).get("overall"))
            confusion = confusion_from_predictions(metrics.get("validation", {}).get("predictions"))
            if validation is not None and confusion is not None:
                validation["confusion_matrix"] = confusion
            nonfinite_updates = metrics.get("scaler_audit", {}).get("skipped_optimizer_updates")
        else:
            completed_steps = int(metrics.get("completed_optimizer_steps") or 0)
            planned = int(metrics.get("planned_optimizer_steps") or planned)
            elapsed_seconds = finite_number(metrics.get("elapsed_seconds"))
            peak_vram_mb = finite_number(metrics.get("peak_vram_mb"))
            train_examples = int(metrics.get("train_examples") or train_examples or 0)
            valid_examples = int(metrics.get("valid_examples") or 0)
            truncated_train = int(metrics.get("train_truncated") or 0)
            truncated_valid = int(metrics.get("valid_truncated") or 0)
            nonfinite_updates = int(metrics.get("nonfinite_optimizer_updates") or 0)
            encoder_gradient_verified = bool(metrics.get("encoder_gradient_verified"))
            validation_block = metrics.get("validation", {})
            validation = compact_metrics(validation_block.get("binary"))
            confusion = confusion_from_predictions(validation_block.get("predictions"))
            if validation is not None and confusion is not None:
                validation["confusion_matrix"] = confusion
            n23 = validation_block.get("N23")
            if isinstance(n23, dict):
                validation_n23 = {
                    key: finite_number(n23.get(key))
                    for key in ("supervised_examples", "micro_f1", "macro_f1")
                    if key in n23
                }
        train_status = "completed" if metrics.get("status") == "completed" else "failed"
    elif state.get("current_run_id") == run_id and str(state.get("current_stage", "")).startswith("train"):
        train_status = "running"
        completed_steps = int(history[-1]["step"] if history else 0)
        elapsed_seconds = finite_number(state.get("stage_elapsed_seconds"))

    raw_validation_history = (
        metrics.get("validation_history", [])
        if isinstance(metrics, dict) and isinstance(metrics.get("validation_history"), list)
        else read_jsonl(run_dir / "validation_log.jsonl")
    )
    for item in raw_validation_history:
        if not isinstance(item, dict):
            continue
        validation_history.append(
            {
                "epoch": int(item.get("epoch") or 0),
                "optimizer_step": int(item.get("optimizer_step") or item.get("step") or 0),
                "binary": compact_metrics(item.get("binary")),
                "N23": {
                    key: finite_number(item.get("N23", {}).get(key))
                    for key in ("supervised_examples", "micro_f1", "macro_f1")
                    if isinstance(item.get("N23"), dict) and key in item.get("N23", {})
                },
            }
        )
    if validation is None and validation_history:
        validation = validation_history[-1]["binary"]
        validation_n23 = validation_history[-1]["N23"] or None

    if run_id == "D1-NEMOTRON-GUARD-8B-V3":
        expected_eval = 3
    else:
        expected_eval = 12 if run.get("model_kind") == "mmbert_schema" else 6
    eval_jobs, eval_total = evaluation_jobs(run_id, expected_eval)
    eval_done = len(eval_jobs)
    if not run.get("train") and eval_done:
        train_status = "not_applicable"

    progress = completed_steps / planned if planned else (1.0 if train_status == "completed" else 0.0)
    progress = min(max(progress, 0.0), 1.0)
    epoch = min(int(math.ceil(completed_steps / per_epoch)), int(run.get("epochs") or 0)) if per_epoch else 0
    epoch_progress = (
        ((completed_steps - max(epoch - 1, 0) * per_epoch) / per_epoch) if per_epoch and epoch else 0.0
    )

    eta_seconds = None
    projected_total_seconds = None
    if train_status == "running" and planned and elapsed_seconds is not None:
        dynamic_rate = (float(elapsed_seconds) / completed_steps) if completed_steps >= 20 else None
        if dynamic_rate is not None and reference_seconds_per_step is not None:
            seconds_per_step = 0.75 * dynamic_rate + 0.25 * reference_seconds_per_step
        else:
            seconds_per_step = dynamic_rate or reference_seconds_per_step
        if seconds_per_step is not None:
            schema_factor = 1.15 if run.get("model_kind") == "mmbert_schema" else 1.0
            projected_total_seconds = seconds_per_step * planned * schema_factor
            eta_seconds = max(0.0, projected_total_seconds - float(elapsed_seconds))

    return {
        "id": run_id,
        "short_id": run_id.split("-")[0],
        "name": RUN_NAMES.get(run_id, run_id),
        "model_kind": run.get("model_kind"),
        "context": run.get("context"),
        "languages": (
            "EN+VI"
            if "EV" in run_id
            else ("EN" if run_id.startswith(("E3-M-E-", "E3NR-M-E-")) else "—")
        ),
        "category_tasks": bool(run.get("category_tasks")),
        "train_status": train_status,
        "planned_steps": planned,
        "completed_steps": completed_steps,
        "progress": progress,
        "epochs": int(run.get("epochs") or 0),
        "current_epoch": epoch,
        "epoch_progress": min(max(epoch_progress, 0.0), 1.0),
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
        "projected_total_seconds": projected_total_seconds,
        "peak_vram_mb": peak_vram_mb,
        "train_examples": train_examples,
        "valid_examples": valid_examples,
        "truncated_train": truncated_train,
        "truncated_valid": truncated_valid,
        "nonfinite_updates": nonfinite_updates,
        "encoder_gradient_verified": encoder_gradient_verified,
        "validation": validation,
        "validation_N23": validation_n23,
        "validation_history": validation_history,
        "history": history,
        "evaluation": {
            "done": eval_done,
            "total": eval_total,
            "progress": eval_done / eval_total if eval_total else 0.0,
            "jobs": eval_jobs,
        },
    }


def decoder_progress(summary: dict[str, Any], decoder_state: dict[str, Any]) -> dict[str, Any]:
    current_output = decoder_state.get("current_output")
    progress_rows: list[dict[str, Any]] = []
    output_path = ROOT / str(current_output) if isinstance(current_output, str) else None
    if output_path is not None:
        progress_rows = read_jsonl(output_path / "progress.jsonl")
    current_progress = progress_rows[-1] if progress_rows else {}
    processed = int(current_progress.get("processed") or 0)
    total = int(current_progress.get("total") or 0)
    fraction = float(current_progress.get("progress") or 0.0)
    evaluation = summary["evaluation"]
    active_job_has_metrics = bool(output_path and (output_path / "metrics.json").exists())
    partial = 0.0 if active_job_has_metrics else min(max(fraction, 0.0), 1.0)
    # D1 evaluates all three datasets in one combined vLLM pass and only splits
    # the metrics afterwards.  While that pass is active, its manifest fraction
    # is therefore the fraction of the whole D1 evaluation, not one third of it.
    combined_pass_active = bool(
        output_path
        and output_path.name == "full_combined"
        and not active_job_has_metrics
    )
    if combined_pass_active:
        evaluation["progress"] = partial
    else:
        evaluation["progress"] = min(
            1.0, (int(evaluation["done"]) + partial) / max(1, int(evaluation["total"]))
        )
    status = str(decoder_state.get("status") or "pending")
    summary.update(
        {
            "train_status": "not_applicable",
            "progress": evaluation["progress"],
            "planned_steps": total,
            "completed_steps": processed,
            "elapsed_seconds": finite_number(decoder_state.get("elapsed_seconds")),
            "eta_seconds": finite_number(current_progress.get("eta_seconds")),
            "peak_vram_mb": finite_number(current_progress.get("peak_vram_mb")),
            "decoder_status": status,
            "languages": "EN+VI",
        }
    )
    if status == "completed":
        summary["progress"] = 1.0
        evaluation["progress"] = 1.0
    return summary


def d2_progress(d2_state: dict[str, Any]) -> dict[str, Any]:
    run_id = "D2-NEMOTRON-GUARD-8B-VI-PILOT"
    main_progress = read_json(D2_REPORT_ROOT / "main_train" / "progress.json", {})
    feature_audit = read_json(D2_REPORT_ROOT / "main_train" / "feature_audit.json", {})
    main_history = read_jsonl(D2_REPORT_ROOT / "main_train" / "train_log.jsonl")
    history = downsample(
        [
            {
                "step": int(item.get("optimizer_step") or 0),
                "epoch": finite_number(item.get("epoch")),
                "loss": finite_number(item.get("loss")),
                "binary_loss": None,
                "category_loss": None,
                "throughput": finite_number(item.get("input_tokens_per_second")),
                "max_tokens": None,
            }
            for item in main_history
            if int(item.get("optimizer_step") or 0) > 0
        ]
    )
    completed_steps = int(main_progress.get("optimizer_step") or 0)
    planned_steps = int(main_progress.get("planned_optimizer_steps") or 512)
    stage = str(d2_state.get("stage") or "pending")
    status = str(d2_state.get("status") or "pending")
    final_adapter_exists = (D2_REPORT_ROOT / "main_train" / "final_adapter" / "adapter_model.safetensors").exists()
    if status == "failed":
        train_status = "failed"
    elif final_adapter_exists or stage in {"full_eval", "full_compare", "complete"} or status == "complete":
        train_status = "completed"
    elif status == "running":
        train_status = "running"
    else:
        train_status = "pending"

    eval_jobs, eval_total = evaluation_jobs(run_id, 2)
    eval_done = len(eval_jobs)
    eval_progress = eval_done / eval_total if eval_total else 0.0
    if stage in {"full_eval", "full_compare"} and status == "running":
        eval_rows = read_jsonl(D2_REPORT_ROOT / "full_eval" / "progress.jsonl")
        if eval_rows:
            eval_progress = float(eval_rows[-1].get("progress") or eval_progress)
    if status == "complete":
        eval_progress = 1.0

    progress = completed_steps / max(1, planned_steps)
    progress = min(max(progress, 0.0), 1.0)
    elapsed = finite_number(main_progress.get("elapsed_seconds"))
    eta = finite_number(main_progress.get("eta_seconds")) if train_status == "running" else None
    return {
        "id": run_id,
        "short_id": "D2",
        "name": RUN_NAMES[run_id],
        "model_kind": "decoder_guard_qlora_sft",
        "context": 1024,
        "languages": "EN+VI",
        "category_tasks": True,
        "train_status": train_status,
        "planned_steps": planned_steps,
        "completed_steps": completed_steps,
        "progress": progress,
        "epochs": 1,
        "current_epoch": 1 if completed_steps else 0,
        "epoch_progress": progress,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "projected_total_seconds": (float(elapsed) + float(eta)) if elapsed is not None and eta is not None else None,
        "peak_vram_mb": finite_number(main_progress.get("peak_vram_mb")),
        "train_examples": int(feature_audit.get("examples") or 8192),
        "valid_examples": None,
        "truncated_train": int(feature_audit.get("truncated") or 0),
        "truncated_valid": None,
        "nonfinite_updates": 0,
        "encoder_gradient_verified": None,
        "validation": None,
        "validation_N23": None,
        "validation_history": [],
        "history": history,
        "evaluation": {
            "done": eval_done,
            "total": eval_total,
            "progress": min(max(eval_progress, 0.0), 1.0),
            "jobs": eval_jobs,
        },
    }


def d2_active_state(d2_state: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    stage = str(d2_state.get("stage") or "pending")
    stage_map = {
        "build_manifest": "train_d2_manifest",
        "smoke_train": "train_d2_smoke",
        "smoke_eval": "evaluate_d2_smoke",
        "smoke_compare": "evaluate_d2_smoke_compare",
        "main_train": "train_d2_main",
        "full_eval": "evaluate_d2_full",
        "full_compare": "evaluate_d2_compare",
    }
    evaluating = stage in {"smoke_eval", "smoke_compare", "full_eval", "full_compare"}
    elapsed = summary["elapsed_seconds"]
    if evaluating:
        rows = read_jsonl(D2_REPORT_ROOT / ("full_eval" if stage.startswith("full") else "smoke_eval") / "progress.jsonl")
        if rows:
            elapsed = finite_number(rows[-1].get("elapsed_seconds"))
    return {
        "status": d2_state.get("status"),
        "started_at": d2_state.get("started_at"),
        "updated_at": d2_state.get("updated_at"),
        "current_stage": stage_map.get(stage, stage),
        "current_run_id": summary["id"],
        "stage_index": None,
        "stage_count": None,
        "stage_elapsed_seconds": elapsed,
        "child_pid": None,
        "error": d2_state.get("error"),
        "failed_at": d2_state.get("failed_at"),
        "completed_at": d2_state.get("completed_at"),
    }


def generic_phase0_pipeline(
    state: dict[str, Any],
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    current_stage = str(state.get("current_stage") or "")
    stages: list[dict[str, Any]] = [
        {
            "key": "preflight",
            "label": "Preflight no-R",
            "detail": "manifest, tokenizer, model và GPU gate",
            "status": (
                "running"
                if current_stage == "preflight"
                else ("completed" if state.get("started_at") else "pending")
            ),
        }
    ]
    for item in experiments:
        stages.append(
            {
                "key": f"train_{item['id']}",
                "label": str(item.get("short_id") or item["id"]),
                "detail": str(item.get("name") or item["id"]),
                "status": str(item.get("train_status") or "pending"),
            }
        )
    evaluation_done = sum(
        item.get("evaluation", {}).get("done", 0)
        >= item.get("evaluation", {}).get("total", 1)
        for item in experiments
    )
    stages.append(
        {
            "key": "evaluation",
            "label": "Evaluation matrix",
            "detail": f"{evaluation_done}/{len(experiments)} thí nghiệm đã đủ suite",
            "status": (
                "running"
                if current_stage.startswith("evaluate")
                else ("completed" if experiments and evaluation_done == len(experiments) else "pending")
            ),
        }
    )
    stages.append(
        {
            "key": "export",
            "label": "Export + ACK + stop",
            "detail": "đóng gói, tải về máy, xác minh SHA256 rồi stop instance",
            "status": (
                "running"
                if current_stage in {"package", "export", "wait_for_ack"}
                else (
                    "completed"
                    if str(state.get("status") or "") in {"completed", "complete", "stopped"}
                    else "pending"
                )
            ),
        }
    )
    return {
        "stages": stages,
        "eval_environment_ready": True,
        "auto_stop": {
            "enabled": bool(state.get("auto_stop")),
            "delay_seconds": float(state.get("max_wall_hours") or 0) * 3600,
        }
        if state.get("auto_stop")
        else None,
    }


def main() -> None:
    config = read_json(CONFIG_PATH, {})
    state = read_json(STATE_PATH, {})
    decoder_state = read_json(DECODER_STATE_PATH, {})
    d2_state = read_json(D2_STATE_PATH, {})
    if not isinstance(config, dict):
        raise RuntimeError(f"Cannot read {CONFIG_PATH}")
    extra_config_paths = [
        item.strip()
        for item in os.environ.get("PHASE0_EXTRA_CONFIG_PATHS", "").split(",")
        if item.strip()
    ]
    if extra_config_paths:
        merged_runs = list(config.get("runs", []))
        known_ids = {str(item.get("id")) for item in merged_runs}
        for relative_path in extra_config_paths:
            extra_config = read_json(ROOT / relative_path, {})
            for run in extra_config.get("runs", []) if isinstance(extra_config, dict) else []:
                run_id = str(run.get("id"))
                if run_id and run_id not in known_ids:
                    merged_runs.append(run)
                    known_ids.add(run_id)
        config = dict(config)
        config["runs"] = merged_runs
    if not isinstance(state, dict):
        state = {}
    counts = load_manifest_counts(config)
    runs_by_id = {str(item["id"]): item for item in config.get("runs", [])}
    runs_by_id["D1-NEMOTRON-GUARD-8B-V3"] = {
        "id": "D1-NEMOTRON-GUARD-8B-V3",
        "train": False,
        "model_kind": "decoder_guard_zero_shot",
        "context": 8192,
        "category_tasks": True,
    }
    runs_by_id["D2-NEMOTRON-GUARD-8B-VI-PILOT"] = {
        "id": "D2-NEMOTRON-GUARD-8B-VI-PILOT",
        "train": True,
        "model_kind": "decoder_guard_qlora_sft",
        "context": 1024,
        "category_tasks": True,
        "epochs": 1,
    }

    reference_run_id = os.environ.get("PHASE0_REFERENCE_RUN_ID", "E3-M-E-8K")
    e3_metrics = read_json(EXPERIMENT_RUN_ROOT / reference_run_id / "metrics.json", {})
    reference_seconds_per_step = None
    if isinstance(e3_metrics, dict):
        steps = int(e3_metrics.get("completed_optimizer_steps") or 0)
        elapsed = finite_number(e3_metrics.get("elapsed_seconds"))
        if steps and elapsed is not None:
            reference_seconds_per_step = float(elapsed) / steps

    experiments = [
        experiment_summary(runs_by_id[run_id], state, counts, reference_seconds_per_step)
        for run_id in RUN_ORDER
        if run_id in runs_by_id
    ]
    for index, item in enumerate(experiments):
        if item["id"] == "D1-NEMOTRON-GUARD-8B-V3":
            experiments[index] = decoder_progress(item, decoder_state)
            break
    d2_summary = d2_progress(d2_state)
    experiments = [d2_summary if item["id"] == d2_summary["id"] else item for item in experiments]

    active_state = dict(state)
    decoder_status = str(decoder_state.get("status") or "missing")
    phase0_finished = str(state.get("status")) in {"completed", "failed"}
    if phase0_finished and decoder_status not in {"missing", "completed"}:
        active_state.update(
            {
                "status": decoder_status,
                "current_run_id": "D1-NEMOTRON-GUARD-8B-V3",
                "current_stage": decoder_state.get("current_stage"),
                "stage_elapsed_seconds": decoder_state.get("elapsed_seconds"),
                "updated_at": decoder_state.get("updated_at"),
                "error": decoder_state.get("error"),
                "failed_at": decoder_state.get("failed_at"),
            }
        )
    elif phase0_finished and decoder_status == "completed":
        active_state.update(
            {
                "status": "completed",
                "current_run_id": None,
                "current_stage": None,
                "updated_at": decoder_state.get("updated_at"),
                "completed_at": decoder_state.get("completed_at"),
            }
        )

    d2_status = str(d2_state.get("status") or "missing")
    if d2_status in {"running", "failed"}:
        active_state = d2_active_state(d2_state, d2_summary)
    elif d2_status == "complete":
        active_state.update(
            {
                "status": "completed",
                "current_run_id": None,
                "current_stage": None,
                "updated_at": d2_state.get("updated_at"),
                "completed_at": d2_state.get("completed_at"),
            }
        )

    current_run_id = active_state.get("current_run_id")
    current = next((item for item in experiments if item["id"] == current_run_id), None)
    completed_train = sum(item["train_status"] in {"completed", "not_applicable"} for item in experiments)
    completed_eval = sum(item["evaluation"]["done"] >= item["evaluation"]["total"] for item in experiments)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": {
            key: active_state.get(key)
            for key in (
                "status",
                "started_at",
                "updated_at",
                "current_stage",
                "current_run_id",
                "stage_index",
                "stage_count",
                "stage_elapsed_seconds",
                "child_pid",
                "error",
                "failed_at",
                "completed_at",
            )
        },
        "gpu": gpu_snapshot(),
        "pipeline": generic_phase0_pipeline(active_state, experiments),
        "summary": {
            "experiments": len(experiments),
            "training_complete": completed_train,
            "evaluation_complete": completed_eval,
            "current_run_id": current_run_id,
            "current_progress": current.get("progress") if current else None,
            "current_eta_seconds": current.get("eta_seconds") if current else None,
        },
        "experiments": experiments,
    }
    print("DASHBOARD_SNAPSHOT=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
