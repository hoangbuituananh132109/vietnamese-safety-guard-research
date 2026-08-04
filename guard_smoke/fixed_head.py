"""One fixed binary text-safety head used as the schema-head control."""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from .constants import SAFETY_TO_ID, TEXT_SAFETY_TASK
from .data import GuardExample, load_manifest


@dataclass
class FixedHeadConfig:
    model_path: str
    max_length: int = 256
    batch_size: int = 4
    epochs: int = 2
    max_steps: int = 24
    learning_rate: float = 5e-4
    freeze_encoder: bool = True
    use_lora: bool = False
    lora_r: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    fp16: bool = True
    seed: int = 3407


class GuardDataset(Dataset):
    def __init__(self, examples: Sequence[GuardExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> GuardExample:
        return self.examples[index]


class GuardCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: Sequence[GuardExample]) -> dict:
        encoded = self.tokenizer(
            [example.text for example in examples],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                "Fixed-head batch exceeded max_length after exact tokenization; "
                "the caller must move it to the reported tail rather than truncate"
            )
        return {
            **encoded,
            "safety_targets": torch.tensor(
                [SAFETY_TO_ID[example.safety_label] for example in examples],
                dtype=torch.long,
            ),
            "examples": list(examples),
        }


class FixedHeadGuard(nn.Module):
    """One encoder and one shared ``Linear(hidden, 2)`` for all text views."""

    def __init__(self, model_path: str, freeze_encoder: bool = True) -> None:
        super().__init__()
        self.model_path = model_path
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden_size = int(self.encoder.config.hidden_size)
        dropout_probability = float(
            getattr(self.encoder.config, "classifier_dropout", None)
            or getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        )
        self.dropout = nn.Dropout(dropout_probability)
        self.safety_head = nn.Linear(hidden_size, 2)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                hidden = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
        else:
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        # Masked mean pooling is robust across ModernBERT/DeBERTa-style encoders.
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.dropout(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        safety_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled = self._encode(input_ids, attention_mask)
        safety_logits = self.safety_head(pooled)
        result = {"safety_logits": safety_logits}
        if safety_targets is not None:
            safety_loss = F.cross_entropy(safety_logits.float(), safety_targets)
            result.update(
                {
                    "loss": safety_loss,
                    "safety_loss": safety_loss,
                }
            )
        return result


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


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


@torch.no_grad()
def evaluate_fixed_guard(
    model: FixedHeadGuard,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> dict:
    model.eval()
    rows: list[dict] = []
    losses: list[float] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_fp16 and device.type == "cuda",
        ):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                safety_targets=batch["safety_targets"],
            )
        losses.append(float(output["loss"].item()))
        probabilities = output["safety_logits"].float().softmax(dim=-1)[:, 1]
        predictions = (probabilities >= 0.5).long()
        for index, example in enumerate(raw_batch["examples"]):
            rows.append(
                {
                    "example_id": example.example_id,
                    "record_uid": example.record_uid,
                    "view": example.view,
                    "scope": example.safety_scope,
                    "language": example.language,
                    "tag": example.tag,
                    "target": int(batch["safety_targets"][index].item()),
                    "prediction": int(predictions[index].item()),
                    "unsafe_probability": float(probabilities[index].item()),
                }
            )

    metrics: dict[str, object] = {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "examples": len(rows),
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
    complete_pairs = [value for value in paired.values() if set(value) == {"en", "vi"}]
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


def train_fixed_guard(
    train_manifest: Path,
    valid_manifest: Path,
    output_dir: Path,
    config: FixedHeadConfig,
) -> dict:
    _set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
        # mmBERT uses a Gemma-2 Metaspace pre-tokenizer. Transformers 4.57's
        # Mistral heuristic misidentifies this local tokenizer and its proposed
        # patch only works for a Sequence pre-tokenizer, not Metaspace.
        fix_mistral_regex=False,
    )
    if config.use_lora and config.freeze_encoder:
        raise ValueError("use_lora requires freeze_encoder=False so LoRA stays in autograd")
    model = FixedHeadGuard(config.model_path, config.freeze_encoder)
    if config.use_lora:
        model.encoder = get_peft_model(
            model.encoder,
            LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules="all-linear",
            ),
        )
    model = model.to(device)
    all_train_examples = load_manifest(train_manifest)
    all_valid_examples = load_manifest(valid_manifest)
    def partition(examples: Sequence[GuardExample]) -> tuple[list[GuardExample], list[dict]]:
        eligible: list[GuardExample] = []
        tail: list[dict] = []
        for example in examples:
            length = len(tokenizer.encode(example.text, add_special_tokens=True))
            if length <= config.max_length:
                eligible.append(example)
            else:
                tail.append(
                    {
                        "example_id": example.example_id,
                        "serialized_tokens": length,
                        "max_length": config.max_length,
                    }
                )
        return eligible, tail
    train_examples, train_tail = partition(all_train_examples)
    valid_examples, valid_tail = partition(all_valid_examples)
    collator = GuardCollator(tokenizer, config.max_length)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        GuardDataset(train_examples),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
    )
    valid_loader = DataLoader(
        GuardDataset(valid_examples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=config.fp16 and device.type == "cuda",
    )
    history: list[dict] = []
    global_step = 0
    started = time.perf_counter()
    model.train()
    stop = False
    for epoch in range(config.epochs):
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.fp16 and device.type == "cuda",
            ):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    safety_targets=batch["safety_targets"],
                )
            scaler.scale(output["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": float(output["loss"].item()),
                    "safety_loss": float(output["safety_loss"].item()),
                }
            )
            if config.max_steps > 0 and global_step >= config.max_steps:
                stop = True
                break
        if stop:
            break

    elapsed = time.perf_counter() - started
    valid_metrics = evaluate_fixed_guard(
        model,
        valid_loader,
        device,
        config.fp16,
    )
    if config.use_lora:
        checkpoint_path = output_dir / "checkpoint"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model.encoder.save_pretrained(
            checkpoint_path / "adapter",
            save_embedding_layers=False,
        )
        torch.save(
            {
                "safety_head": {
                    key: value.detach().cpu()
                    for key, value in model.safety_head.state_dict().items()
                },
                "config": asdict(config),
                "safety_labels": SAFETY_TO_ID,
            },
            checkpoint_path / "fixed_head.pt",
        )
        reloaded = FixedHeadGuard(config.model_path, freeze_encoder=False)
        reloaded.encoder = PeftModel.from_pretrained(
            reloaded.encoder,
            checkpoint_path / "adapter",
            is_trainable=False,
        )
        head_payload = torch.load(
            checkpoint_path / "fixed_head.pt",
            map_location="cpu",
            weights_only=True,
        )
        reloaded.safety_head.load_state_dict(head_payload["safety_head"])
        reloaded = reloaded.to(device)
    else:
        checkpoint_path = output_dir / "fixed_head_checkpoint.pt"
        checkpoint = {
            "config": asdict(config),
            "model_state": model.state_dict(),
            "safety_labels": SAFETY_TO_ID,
            "contract": "one shared binary head; view/scope are metadata only",
        }
        torch.save(checkpoint, checkpoint_path)
        reloaded = FixedHeadGuard(config.model_path, config.freeze_encoder).to(device)
        reloaded.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )["model_state"]
        )

    # Reconstruct and compare predictions to prove that the checkpoint is usable.
    reloaded_metrics = evaluate_fixed_guard(
        reloaded,
        valid_loader,
        device,
        config.fp16,
    )
    before = {
        row["example_id"]: row["unsafe_probability"]
        for row in valid_metrics["predictions"]
    }
    after = {
        row["example_id"]: row["unsafe_probability"]
        for row in reloaded_metrics["predictions"]
    }
    max_reload_delta = max(abs(before[key] - after[key]) for key in before)

    result = {
        "config": asdict(config),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "contract": {
            "task": TEXT_SAFETY_TASK,
            "labels": ["safe", "unsafe"],
            "separate_prompt_response_heads": False,
            "scope_or_view_model_feature": False,
            "encoder_training": "LoRA" if config.use_lora else (
                "frozen" if config.freeze_encoder else "full_unfreeze"
            ),
        },
        "train_examples_total": len(all_train_examples),
        "train_examples": len(train_examples),
        "train_tail": train_tail,
        "valid_examples_total": len(all_valid_examples),
        "valid_examples": len(valid_examples),
        "valid_tail": valid_tail,
        "global_steps": global_step,
        "elapsed_seconds": elapsed,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
        ),
        "checkpoint": str(checkpoint_path),
        "max_reload_probability_delta": max_reload_delta,
        "history": history,
        "valid": valid_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
