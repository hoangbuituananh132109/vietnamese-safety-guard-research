from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from guard_smoke.constants import SAFETY_TO_ID
from guard_smoke.data import GuardExample, load_manifest
from guard_smoke.fixed_head import FixedHeadGuard, GuardCollator, GuardDataset
from guard_smoke.mmbert_schema import (
    BinarySchemaCodec,
    MMBertBinarySchemaGuard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone no-truncation inference/evaluation for Phase-0 mmBERT guards."
    )
    parser.add_argument("--model-kind", choices=("fixed", "schema"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="fp16")
    parser.add_argument("--label-order", choices=("canonical", "reversed"), default="canonical")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow-tail",
        action="store_true",
        help="Evaluate eligible rows and write over-context rows instead of failing.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def length_bucket(length: int) -> str:
    if length <= 128:
        return "0001-0128"
    if length <= 256:
        return "0129-0256"
    if length <= 384:
        return "0257-0384"
    if length <= 512:
        return "0385-0512"
    if length <= 1024:
        return "0513-1024"
    if length <= 2048:
        return "1025-2048"
    if length <= 4096:
        return "2049-4096"
    if length <= 8192:
        return "4097-8192"
    return "8193+"


def maybe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def expected_calibration_error(
    targets: Sequence[int], unsafe_probabilities: Sequence[float], bins: int = 10
) -> float | None:
    if not targets:
        return None
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(unsafe_probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        count = int(mask.sum())
        if count:
            value += count / total * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(value)


def binary_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    targets = [int(row["target"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    probabilities = [float(row["unsafe_probability"]) for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    has_both = len(set(targets)) == 2
    positives = sum(targets)
    return {
        "examples": len(rows),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets, predictions, labels=[0, 1], average="macro", zero_division=0
            )
        ),
        "safe_precision": float(precision[0]),
        "safe_recall": float(recall[0]),
        "safe_f1": float(f1[0]),
        "safe_support": int(support[0]),
        "unsafe_precision": float(precision[1]),
        "unsafe_recall": float(recall[1]),
        "unsafe_f1": float(f1[1]),
        "unsafe_support": int(support[1]),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc_auc": float(roc_auc_score(targets, probabilities)) if has_both else None,
        "average_precision": (
            float(average_precision_score(targets, probabilities)) if positives else None
        ),
        "brier_score": float(np.mean((np.asarray(probabilities) - np.asarray(targets)) ** 2)),
        "ece_10": expected_calibration_error(targets, probabilities, bins=10),
    }


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"overall": binary_metrics(rows), "slices": {}}
    for dimension in ("scope", "view", "language", "tag", "length_bucket"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[dimension])].append(row)
        report["slices"][dimension] = {
            key: binary_metrics(value) for key, value in sorted(groups.items())
        }

    paired: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(row["record_uid"], row["view"])][row["language"]] = row
    complete = [value for value in paired.values() if set(value) == {"en", "vi"}]
    report["paired_en_vi"] = {
        "pairs": len(complete),
        "same_decision_rate": (
            float(np.mean([pair["en"]["prediction"] == pair["vi"]["prediction"] for pair in complete]))
            if complete
            else None
        ),
        "mean_probability_gap": (
            float(
                np.mean(
                    [
                        abs(
                            pair["en"]["unsafe_probability"]
                            - pair["vi"]["unsafe_probability"]
                        )
                        for pair in complete
                    ]
                )
            )
            if complete
            else None
        ),
    }
    return report


def resolve_base_model(args: argparse.Namespace) -> str:
    if args.base_model is not None:
        return str(args.base_model)
    if args.model_kind == "schema":
        contract = json.loads(
            (args.checkpoint / "schema_contract.json").read_text(encoding="utf-8")
        )
        value = contract.get("config", {}).get("model_path")
    elif args.checkpoint.is_dir():
        payload = torch.load(
            args.checkpoint / "fixed_head.pt", map_location="cpu", weights_only=True
        )
        value = payload.get("config", {}).get("model_path")
    else:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        value = payload.get("config", {}).get("model_path")
    if not value:
        raise ValueError("Cannot infer base model; pass --base-model explicitly")
    return str(value)


def schema_collate(
    codec: BinarySchemaCodec,
    label_order: tuple[str, str],
    examples: Sequence[GuardExample],
) -> dict[str, Any]:
    encoded = [codec.encode(example, label_order) for example in examples]
    maximum = max(len(item.input_ids) for item in encoded)
    input_ids = torch.full(
        (len(encoded), maximum),
        int(codec.tokenizer.pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, item in enumerate(encoded):
        length = len(item.input_ids)
        input_ids[row_index, :length] = torch.tensor(item.input_ids, dtype=torch.long)
        attention_mask[row_index, :length] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_positions": torch.tensor(
            [item.label_positions for item in encoded], dtype=torch.long
        ),
        "targets": torch.tensor([item.target_index for item in encoded], dtype=torch.long),
        "label_orders": [item.label_order for item in encoded],
        "sequence_lengths": torch.tensor(
            [len(item.input_ids) for item in encoded], dtype=torch.long
        ),
        "examples": list(examples),
    }


def main() -> None:
    args = parse_args()
    if args.max_length <= 0 or args.batch_size <= 0:
        raise ValueError("max-length and batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_model = resolve_base_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp != "none" and device.type == "cuda"
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    all_examples = load_manifest(args.manifest)
    if args.limit is not None:
        all_examples = all_examples[: args.limit]

    if args.model_kind == "fixed":
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            local_files_only=True,
            fix_mistral_regex=False,
        )
        eligible: list[GuardExample] = []
        tail: list[dict[str, Any]] = []
        for example in all_examples:
            length = len(tokenizer.encode(example.text, add_special_tokens=True))
            if length <= args.max_length:
                eligible.append(example)
            else:
                tail.append(
                    {
                        "example_id": example.example_id,
                        "serialized_tokens": length,
                        "max_length": args.max_length,
                    }
                )
        collator = GuardCollator(tokenizer, args.max_length)
        model = FixedHeadGuard(base_model, freeze_encoder=False)
        if args.checkpoint.is_dir():
            model.encoder = PeftModel.from_pretrained(
                model.encoder, args.checkpoint / "adapter", is_trainable=False
            )
            payload = torch.load(
                args.checkpoint / "fixed_head.pt", map_location="cpu", weights_only=True
            )
            model.safety_head.load_state_dict(payload["safety_head"])
        else:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["model_state"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint / "tokenizer",
            local_files_only=True,
            fix_mistral_regex=False,
        )
        codec = BinarySchemaCodec(tokenizer, args.max_length)
        eligible, tail = codec.partition_eligible(all_examples)
        order = ("safe", "unsafe") if args.label_order == "canonical" else ("unsafe", "safe")
        collator = lambda batch: schema_collate(codec, order, batch)
        model = MMBertBinarySchemaGuard.load_checkpoint(
            args.checkpoint,
            base_model,
            len(tokenizer),
            device,
        )

    if tail and not args.allow_tail:
        raise ValueError(
            f"Manifest contains {len(tail)} over-context rows; use the exact eligible "
            "manifest or pass --allow-tail to evaluate only the remainder."
        )
    tail_path = args.output_dir / "tail.jsonl"
    with tail_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in tail:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    loader = DataLoader(
        GuardDataset(eligible),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    model = model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    weighted_loss = 0.0
    started = time.perf_counter()

    with torch.inference_mode():
        for raw_batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in raw_batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                if args.model_kind == "fixed":
                    output = model(
                        input_ids=tensors["input_ids"],
                        attention_mask=tensors["attention_mask"],
                        safety_targets=tensors["safety_targets"],
                    )
                    probabilities = output["safety_logits"].float().softmax(dim=-1)
                    target_indices = tensors["safety_targets"]
                    label_orders = [("safe", "unsafe")] * len(raw_batch["examples"])
                    sequence_lengths = tensors["attention_mask"].sum(dim=1)
                else:
                    output = model(
                        input_ids=tensors["input_ids"],
                        attention_mask=tensors["attention_mask"],
                        label_positions=tensors["label_positions"],
                        targets=tensors["targets"],
                    )
                    probabilities = output["logits"].float().softmax(dim=-1)
                    target_indices = tensors["targets"]
                    label_orders = raw_batch["label_orders"]
                    sequence_lengths = tensors["sequence_lengths"]

            batch_size = len(raw_batch["examples"])
            weighted_loss += float(output["loss"].item()) * batch_size
            predicted_indices = probabilities.argmax(dim=-1)
            for index, example in enumerate(raw_batch["examples"]):
                order = tuple(label_orders[index])
                target_label = order[int(target_indices[index].item())]
                predicted_label = order[int(predicted_indices[index].item())]
                if target_label != example.safety_label:
                    raise AssertionError(
                        f"Target/order mismatch for {example.example_id}: "
                        f"{target_label} != {example.safety_label}"
                    )
                unsafe_index = order.index("unsafe")
                seq_length = int(sequence_lengths[index].item())
                rows.append(
                    {
                        "example_id": example.example_id,
                        "record_uid": example.record_uid,
                        "source_split": example.source_split,
                        "view": example.view,
                        "scope": example.safety_scope,
                        "language": example.language,
                        "tag": example.tag,
                        "target": SAFETY_TO_ID[target_label],
                        "target_label": target_label,
                        "prediction": SAFETY_TO_ID[predicted_label],
                        "predicted_label": predicted_label,
                        "unsafe_probability": float(probabilities[index, unsafe_index].item()),
                        "label_order": list(order),
                        "serialized_tokens": seq_length,
                        "length_bucket": length_bucket(seq_length),
                    }
                )

    elapsed = time.perf_counter() - started
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = build_report(rows)
    report.update(
        {
            "contract": {
                "task": "text safety classification",
                "task_type": "single_label",
                "labels": ["safe", "unsafe"],
                "activation": "softmax",
                "loss": "categorical_cross_entropy",
                "truncation": False,
                "view_or_scope_model_feature": False,
            },
            "run": {
                "model_kind": args.model_kind,
                "checkpoint": str(args.checkpoint),
                "base_model": base_model,
                "manifest": str(args.manifest),
                "manifest_sha256": sha256_file(args.manifest),
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "amp": args.amp,
                "label_order": args.label_order if args.model_kind == "schema" else None,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "input_examples": len(all_examples),
                "evaluated_examples": len(rows),
                "tail_examples": len(tail),
                "loss": weighted_loss / len(rows) if rows else None,
                "elapsed_seconds": elapsed,
                "examples_per_second": len(rows) / elapsed if elapsed else None,
                "peak_vram_mb": (
                    torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
                ),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            },
            "artifacts": {
                "predictions": str(predictions_path),
                "tail": str(tail_path),
            },
        }
    )
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "metrics": str(metrics_path),
                "predictions": str(predictions_path),
                "examples": len(rows),
                "tail": len(tail),
                "macro_f1": report["overall"].get("macro_f1"),
                "examples_per_second": report["run"]["examples_per_second"],
                "peak_vram_mb": report["run"]["peak_vram_mb"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
