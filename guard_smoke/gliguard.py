"""GLiGuard/GLiNER2 smoke training for one binary text-safety task.

The dataset contract is deliberately strict:

* every materialized P/R/PR text uses the same ``text safety classification`` task;
* P gets its gold target from ``prompt_label``;
* R and PR get their gold target from ``response_label``;
* view/scope remain reporting metadata, never task routing or separate heads.

GLiNER2's ``max_len`` is a word-token cap for the text before the schema is
prepended.  The helpers below therefore audit the final encoder subword length
instead of reporting ``max_len`` as if it were the model sequence length.
"""

from __future__ import annotations

import gc
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .constants import SAFETY_LABELS, TEXT_SAFETY_TASK
from .data import GuardExample, load_manifest
from .gliguard_single_label import install_phase0_single_label_ce
from guard_train.precision import cuda_supports_native_bf16
from guard_train.gliguard_precision import AuditedGradScaler


LABEL_TO_ID = {"safe": 0, "unsafe": 1}


def _binary_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "safe_precision": float(precision[0]),
        "safe_recall": float(recall[0]),
        "unsafe_precision": float(precision[1]),
        "unsafe_recall": float(recall[1]),
        "unsafe_f1": float(f1[1]),
    }


def _schema_for_example(model: Any, example: GuardExample) -> dict:
    """Build the same Phase-0 binary schema for every materialized text."""

    example.validate()
    return (
        model.create_schema()
        .classification(TEXT_SAFETY_TASK, list(SAFETY_LABELS))
        .build()
    )


def _parse_binary_result(payload: dict, task: str) -> tuple[int, float, str, float]:
    """Return prediction and an unsafe probability from GLiNER2 output."""

    if task not in payload:
        raise KeyError(f"Missing task {task!r} in GLiNER2 result: {payload!r}")
    value = payload[task]
    if isinstance(value, str):
        label = value
        confidence = 1.0
    elif isinstance(value, dict):
        label = str(value.get("label") or value.get("value") or "")
        confidence = float(value.get("confidence", value.get("score", 1.0)))
    else:
        raise TypeError(f"Unsupported GLiNER2 classification result: {value!r}")
    if label not in LABEL_TO_ID:
        raise ValueError(f"Unexpected label {label!r} for {task}: {value!r}")
    confidence = min(1.0, max(0.0, confidence))
    unsafe_probability = confidence if label == "unsafe" else 1.0 - confidence
    return LABEL_TO_ID[label], unsafe_probability, label, confidence


@torch.inference_mode()
def evaluate_gliguard(
    model: Any,
    examples: Iterable[GuardExample],
    batch_size: int = 1,
    max_words: int | None = 128,
) -> dict[str, Any]:
    """Evaluate all text through one task while preserving target provenance."""

    examples = list(examples)
    model.eval()
    rows: list[dict[str, Any]] = []

    tasks = {TEXT_SAFETY_TASK: list(SAFETY_LABELS)}
    results = model.batch_classify_text(
        [example.text for example in examples],
        tasks,
        batch_size=batch_size,
        include_confidence=True,
        max_len=max_words,
    )
    if len(results) != len(examples):
        raise RuntimeError(
            f"GLiNER2 returned {len(results)} results for {len(examples)} examples"
        )
    for example, result in zip(examples, results):
        prediction, unsafe_probability, label, confidence = _parse_binary_result(
            result, TEXT_SAFETY_TASK
        )
        rows.append(
            {
                "example_id": example.example_id,
                "record_uid": example.record_uid,
                "view": example.view,
                "scope": example.safety_scope,
                "language": example.language,
                "tag": example.tag,
                "target": LABEL_TO_ID[example.safety_label],
                "prediction": prediction,
                "predicted_label": label,
                "predicted_confidence": confidence,
                "unsafe_probability": unsafe_probability,
                "task": TEXT_SAFETY_TASK,
            }
        )

    rows.sort(key=lambda row: row["example_id"])
    metrics: dict[str, Any] = {
        "examples": len(rows),
        "max_words": max_words,
        "overall": _binary_metrics(
            [row["target"] for row in rows],
            [row["prediction"] for row in rows],
        ),
        "slices": {},
        "predictions": rows,
    }
    for dimension in ("scope", "view", "language", "tag"):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row[dimension]].append(row)
        metrics["slices"][dimension] = {
            name: {
                "examples": len(items),
                **_binary_metrics(
                    [item["target"] for item in items],
                    [item["prediction"] for item in items],
                ),
            }
            for name, items in sorted(grouped.items())
        }

    paired: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        paired[(row["record_uid"], row["view"])][row["language"]] = row
    complete_pairs = [pair for pair in paired.values() if set(pair) == {"en", "vi"}]
    metrics["paired_en_vi"] = {
        "pairs": len(complete_pairs),
        "same_decision_rate": (
            float(
                np.mean(
                    [pair["en"]["prediction"] == pair["vi"]["prediction"] for pair in complete_pairs]
                )
            )
            if complete_pairs
            else math.nan
        ),
        "mean_probability_gap": (
            float(
                np.mean(
                    [
                        abs(
                            pair["en"]["unsafe_probability"]
                            - pair["vi"]["unsafe_probability"]
                        )
                        for pair in complete_pairs
                    ]
                )
            )
            if complete_pairs
            else math.nan
        ),
    }
    return metrics


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
        "over_512": int((array > 512).sum()),
        "over_1024": int((array > 1024).sum()),
        "over_2048": int((array > 2048).sum()),
    }


def audit_gliner_lengths(
    model: Any,
    examples: Iterable[GuardExample],
    max_words: int | None,
) -> dict[str, Any]:
    """Measure final encoder lengths after the dynamic schema is prepended."""

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for example in examples:
        schema = _schema_for_example(model, example)
        try:
            batch = model.processor.collate_fn_inference(
                [(example.text, schema)], max_len=max_words
            )
            rows.append(
                {
                    "example_id": example.example_id,
                    "view": example.view,
                    "scope": example.safety_scope,
                    "language": example.language,
                    "category_scope": example.category_scope,
                    "text_word_count": int(batch.text_word_counts[0]),
                    "encoder_subwords": int(batch.original_lengths[0]),
                    "schema_count": int(batch.schema_counts[0]),
                }
            )
        except Exception as exc:  # audit must expose, never silently replace
            failures.append(
                {
                    "example_id": example.example_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    by_schema_count: dict[str, list[int]] = defaultdict(list)
    by_view: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_schema_count[str(row["schema_count"])].append(row["encoder_subwords"])
        by_view[row["view"]].append(row["encoder_subwords"])
    longest = sorted(rows, key=lambda row: row["encoder_subwords"], reverse=True)[:10]
    return {
        "max_words_argument": max_words,
        "max_words_semantics": "text word-token cap before schema; not total encoder length",
        "successful": len(rows),
        "failed": len(failures),
        "encoder_subwords": _distribution([row["encoder_subwords"] for row in rows]),
        "text_words_after_cap": _distribution([row["text_word_count"] for row in rows]),
        "by_schema_count": {
            key: _distribution(values) for key, values in sorted(by_schema_count.items())
        },
        "by_view": {key: _distribution(values) for key, values in sorted(by_view.items())},
        "longest": longest,
        "failures": failures,
    }


@torch.inference_mode()
def probe_context_forward(
    model: Any,
    word_counts: Iterable[int],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run real forward passes at increasing lengths and record failures/VRAM."""

    schema = (
        model.create_schema()
        .classification(TEXT_SAFETY_TASK, list(SAFETY_LABELS))
        .build()
    )
    rows: list[dict[str, Any]] = []
    for word_count in word_counts:
        text = "neutral " * word_count
        batch = model.processor.collate_fn_inference(
            [(text, schema)], max_len=word_count
        )
        actual_length = int(batch.original_lengths[0])
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            result = model.batch_classify_text(
                [text],
                {TEXT_SAFETY_TASK: list(SAFETY_LABELS)},
                batch_size=1,
                include_confidence=True,
                max_len=word_count,
            )[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            prediction, probability, label, confidence = _parse_binary_result(
                result, TEXT_SAFETY_TASK
            )
            rows.append(
                {
                    "requested_text_words": word_count,
                    "actual_encoder_subwords": actual_length,
                    "status": "success",
                    "elapsed_seconds": elapsed,
                    "peak_vram_mb": (
                        torch.cuda.max_memory_allocated(device) / (1024**2)
                        if device.type == "cuda"
                        else 0.0
                    ),
                    "prediction": prediction,
                    "predicted_label": label,
                    "confidence": confidence,
                    "unsafe_probability": probability,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "requested_text_words": word_count,
                    "actual_encoder_subwords": actual_length,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "peak_vram_mb": (
                        torch.cuda.max_memory_allocated(device) / (1024**2)
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    return rows


def _prediction_map(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        row["example_id"]: float(row["unsafe_probability"])
        for row in metrics["predictions"]
    }


def train_gliguard_lora_smoke(
    model_path: Path,
    train_gliner_jsonl: Path,
    valid_gliner_jsonl: Path,
    valid_manifest: Path,
    output_dir: Path,
    *,
    max_steps: int = 8,
    max_words: int = 128,
    max_train_samples: int = 24,
    max_eval_samples: int = 12,
    lora_r: int = 4,
    inference_batch_size: int = 1,
    run_context_probe: bool = True,
    precision: str = "auto",
) -> dict[str, Any]:
    """Run zero-shot, LoRA train/eval, checkpoint reload and length audits."""

    # Imported lazily so data-only unit tests do not require GLiNER2.
    from gliner2 import GLiNER2
    from gliner2.processor import SamplingConfig
    import gliner2.training.trainer as gliner_trainer_module
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
    from peft import PeftModel

    gliner_trainer_module.GradScaler = AuditedGradScaler

    output_dir.mkdir(parents=True, exist_ok=True)
    valid_examples = load_manifest(valid_manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    if precision not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported precision: {precision}")
    if precision == "auto":
        if device.type == "cuda" and cuda_supports_native_bf16(device):
            resolved_precision = "bf16"
        elif device.type == "cuda":
            resolved_precision = "fp16"
        else:
            resolved_precision = "fp32"
    else:
        resolved_precision = precision
    if resolved_precision == "bf16" and (
        device.type != "cuda" or not cuda_supports_native_bf16(device)
    ):
        raise RuntimeError("BF16 was requested but is not natively supported by this device")

    print(f"Loading GLiGuard for zero-shot evaluation: {model_path}", flush=True)
    inference_model = GLiNER2.from_pretrained(str(model_path)).to(device)
    zero_shot = evaluate_gliguard(
        inference_model,
        valid_examples,
        batch_size=inference_batch_size,
        max_words=max_words,
    )
    length_untruncated = audit_gliner_lengths(
        inference_model, valid_examples, max_words=None
    )
    length_train_cap = audit_gliner_lengths(
        inference_model, valid_examples, max_words=max_words
    )
    context_probe = (
        probe_context_forward(
            inference_model,
            word_counts=(128, 256, 384, 512, 768),
            device=device,
        )
        if run_context_probe
        else []
    )
    del inference_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Loading a fresh GLiGuard for LoRA training: {model_path}", flush=True)
    train_model = GLiNER2.from_pretrained(str(model_path))
    # GLiNER2's generic defaults can synthesize classification labels with
    # probability 0.5 and remove labels. Those augmentations are useful for
    # generic open-schema extraction but invalidate this locked two-class
    # single-label phase. Keep only order shuffling and preserve both labels.
    phase0_sampling = SamplingConfig(
        remove_json_structure_prob=0.0,
        shuffle_json_fields=False,
        remove_json_field_prob=0.0,
        remove_entities_prob=0.0,
        shuffle_entities=False,
        remove_entity_prob=0.0,
        synthetic_entity_label_prob=0.0,
        remove_relations_prob=0.0,
        swap_head_tail_prob=0.0,
        remove_classification_prob=0.0,
        shuffle_classification_labels=True,
        remove_classification_label_prob=0.0,
        synthetic_label_prob=0.0,
        include_true_label_prob=1.0,
        max_num_labels=2,
    )
    train_model.processor.sampling_config = phase0_sampling
    # The installed package defaults to independent BCE classification loss.
    # Phase 0 is mutually exclusive safe/unsafe, so follow the paper contract.
    install_phase0_single_label_ce(train_model)
    config = TrainingConfig(
        output_dir=str(output_dir),
        experiment_name="gliguard_text_safety_binary_smoke",
        num_epochs=2,
        max_steps=max_steps,
        batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=1,
        encoder_lr=1e-5,
        task_lr=1e-4,
        weight_decay=0.01,
        scheduler_type="linear",
        warmup_steps=0,
        fp16=resolved_precision == "fp16",
        bf16=resolved_precision == "bf16",
        eval_strategy="steps",
        eval_steps=max_steps,
        save_total_limit=1,
        save_best=False,
        logging_steps=1,
        report_to_wandb=False,
        num_workers=0,
        pin_memory=False,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        validate_data=True,
        max_len=max_words,
        use_lora=True,
        lora_r=lora_r,
        lora_alpha=float(2 * lora_r),
        lora_dropout=0.0,
        save_adapter_only=True,
        seed=3407,
    )
    trainer = GLiNER2Trainer(model=train_model, config=config)
    trainable_parameters = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    train_result = trainer.train(
        train_data=str(train_gliner_jsonl),
        eval_data=str(valid_gliner_jsonl),
    )
    scaler_audit = (
        trainer.scaler.audit()
        if isinstance(trainer.scaler, AuditedGradScaler)
        else {"enabled": False, "skipped_optimizer_updates": 0}
    )
    if int(scaler_audit["skipped_optimizer_updates"]) != 0:
        raise RuntimeError(f"GLiNER2 FP16 optimizer updates were skipped: {scaler_audit}")
    peak_train_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    after_train = evaluate_gliguard(
        trainer.model,
        valid_examples,
        batch_size=inference_batch_size,
        max_words=max_words,
    )
    final_adapter = output_dir / "final"
    expected_adapter_files = {
        "adapter_config.json": (final_adapter / "adapter_config.json").exists(),
        "adapter_model.safetensors": (final_adapter / "adapter_model.safetensors").exists(),
    }

    after_map = _prediction_map(after_train)
    del trainer, train_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Reloading PEFT adapter from: {final_adapter}", flush=True)
    fresh_base = GLiNER2.from_pretrained(str(model_path))
    reloaded_model = PeftModel.from_pretrained(fresh_base, str(final_adapter)).to(device)
    reloaded = evaluate_gliguard(
        reloaded_model,
        valid_examples,
        batch_size=inference_batch_size,
        max_words=max_words,
    )
    reload_map = _prediction_map(reloaded)
    common_ids = sorted(set(after_map).intersection(reload_map))
    max_reload_delta = max(
        (abs(after_map[key] - reload_map[key]) for key in common_ids),
        default=math.nan,
    )

    result: dict[str, Any] = {
        "contract": {
            "task": TEXT_SAFETY_TASK,
            "task_type": "single_label",
            "labels": list(SAFETY_LABELS),
            "P": "text <- prompt; target <- prompt_label",
            "R": "text <- response; target <- response_label",
            "PR": "text <- prompt + response; target <- response_label",
            "scope_or_view_model_feature": False,
            "separate_prompt_response_heads": False,
            "cross_target_inference": "forbidden",
            "activation": "softmax",
            "loss": "categorical_cross_entropy",
            "package_default_loss_replaced": "binary_cross_entropy_with_logits",
            "training_augmentation": {
                "shuffle_classification_labels": True,
                "remove_classification_prob": 0.0,
                "remove_classification_label_prob": 0.0,
                "synthetic_label_prob": 0.0,
                "include_true_label_prob": 1.0,
                "max_num_labels": 2,
            },
        },
        "device": str(device),
        "gpu": gpu_name,
        "precision": resolved_precision,
        "model_path": str(model_path),
        "max_words": max_words,
        "max_len_warning": (
            "GLiNER2 max_len caps text word tokens before schema; use audited "
            "encoder_subwords for real context length."
        ),
        "zero_shot": zero_shot,
        "length_audit_untruncated": length_untruncated,
        "length_audit_train_cap": length_train_cap,
        "context_forward_probe": context_probe,
        "training": {
            **train_result,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_percent": 100.0 * trainable_parameters / total_parameters,
            "peak_vram_mb": peak_train_vram_mb,
            "scaler_audit": scaler_audit,
            "adapter_files": expected_adapter_files,
        },
        "after_train": after_train,
        "reloaded": reloaded,
        "max_reload_probability_delta": max_reload_delta,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return result
