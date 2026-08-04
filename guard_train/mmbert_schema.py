from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from guard_smoke.constants import N23_CATEGORIES, SAFETY_LABELS, TEXT_SAFETY_TASK
from guard_smoke.data import GuardExample, load_manifest
from guard_smoke.mmbert_schema import (
    LABEL_MARKER,
    SCHEMA_SEPARATOR,
    SCHEMA_SPECIAL_TOKENS,
    TASK_MARKER,
)
from guard_train.mmbert_fixed import (
    ExampleDataset,
    LengthBucketBatchSampler,
    _binary_summary,
    _load_checkpoint,
    _precision,
    _save_checkpoint,
    _set_seed,
    _to_device,
)
from guard_train.truncation import truncate_text_exact


CATEGORY_TASK = "safety policy categories"


@dataclass
class SchemaTrainingConfig:
    model_path: str
    max_length: int = 8192
    micro_batch_size: int = 1
    effective_batch_size: int = 32
    epochs: int = 2
    max_optimizer_steps: int = 0
    encoder_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lora_r: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    gradient_checkpointing: bool = True
    mixed_precision: str = "auto"
    # Keep FP16 stable on Turing across rare long-schema batches.  The very
    # large growth interval prevents GradScaler from repeatedly returning to
    # a scale already observed to overflow on this target GPU.
    grad_scaler_init_scale: float = 512.0
    grad_scaler_growth_interval: int = 1_000_000
    enable_categories: bool = True
    category_loss_weight: float = 1.0
    negative_label_keep_probability: float = 0.5
    full_category_schema_probability: float = 0.25
    bucket_multiplier: int = 64
    num_workers: int = 0
    log_every: int = 10
    save_every: int = 250
    seed: int = 3407

    @property
    def gradient_accumulation_steps(self) -> int:
        if self.effective_batch_size % self.micro_batch_size != 0:
            raise ValueError("effective_batch_size must be divisible by micro_batch_size")
        return self.effective_batch_size // self.micro_batch_size


@dataclass(frozen=True)
class EncodedDynamicSchema:
    input_ids: tuple[int, ...]
    binary_positions: tuple[int, int]
    binary_target: int
    binary_order: tuple[str, str]
    category_positions: tuple[int, ...]
    category_targets: tuple[float, ...]
    category_order: tuple[str, ...]


class DynamicSchemaCodec:
    """GLi-style tasks whose label positions/count/order are decided per sample."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in SCHEMA_SPECIAL_TOKENS)
        if len(set(ids)) != 3 or any(value is None or value < 0 for value in ids):
            raise ValueError(f"Invalid schema marker IDs: {ids}")
        self.task_marker_id, self.label_marker_id, self.separator_id = ids
        self.marker_token_ids = ids
        self.task_ids = {
            TEXT_SAFETY_TASK: tuple(tokenizer.encode(TEXT_SAFETY_TASK, add_special_tokens=False)),
            CATEGORY_TASK: tuple(tokenizer.encode(CATEGORY_TASK, add_special_tokens=False)),
        }
        self.label_ids = {
            label: tuple(tokenizer.encode(label, add_special_tokens=False))
            for label in (*SAFETY_LABELS, *N23_CATEGORIES)
        }
        if any(not value for value in (*self.task_ids.values(), *self.label_ids.values())):
            raise ValueError("A task/label name tokenized to an empty sequence")

    def _prefix(
        self,
        binary_order: Sequence[str],
        category_order: Sequence[str],
    ) -> tuple[list[int], list[int], list[int]]:
        ids: list[int] = []
        binary_positions: list[int] = []
        category_positions: list[int] = []

        ids.extend((self.task_marker_id, *self.task_ids[TEXT_SAFETY_TASK]))
        for label in binary_order:
            binary_positions.append(len(ids))
            ids.append(self.label_marker_id)
            ids.extend(self.label_ids[label])

        if category_order:
            ids.extend((self.task_marker_id, *self.task_ids[CATEGORY_TASK]))
            for label in category_order:
                category_positions.append(len(ids))
                ids.append(self.label_marker_id)
                ids.extend(self.label_ids[label])
        ids.append(self.separator_id)
        return ids, binary_positions, category_positions

    def encode(
        self,
        example: GuardExample,
        binary_order: Sequence[str],
        category_order: Sequence[str],
        *,
        enforce_length: bool = True,
    ) -> EncodedDynamicSchema:
        binary = tuple(binary_order)
        categories = tuple(category_order)
        if len(binary) != 2 or set(binary) != set(SAFETY_LABELS):
            raise ValueError(f"Binary labels must be a safe/unsafe permutation: {binary}")
        if len(set(categories)) != len(categories) or not set(categories).issubset(N23_CATEGORIES):
            raise ValueError("Category schema contains duplicates or unknown labels")
        if categories and example.category_scope == "unavailable":
            raise ValueError("N23 task must be omitted when category supervision is unavailable")
        ids, binary_positions, category_positions = self._prefix(binary, categories)
        ids.extend(self.tokenizer.encode(example.text, add_special_tokens=False))
        if enforce_length and len(ids) > self.max_length:
            raise ValueError(
                f"Schema sequence exceeds max_length: {example.example_id} {len(ids)}>{self.max_length}"
            )
        positives = set(example.categories)
        return EncodedDynamicSchema(
            input_ids=tuple(ids),
            binary_positions=(binary_positions[0], binary_positions[1]),
            binary_target=binary.index(example.safety_label),
            binary_order=(binary[0], binary[1]),
            category_positions=tuple(category_positions),
            category_targets=tuple(float(label in positives) for label in categories),
            category_order=categories,
        )

    def full_length(self, example: GuardExample, text: str | None = None) -> int:
        category_order = (
            N23_CATEGORIES
            if example.category_scope != "unavailable"
            else ()
        )
        candidate = example if text is None else replace(example, text=text)
        return len(
            self.encode(
                candidate,
                SAFETY_LABELS,
                category_order,
                enforce_length=False,
            ).input_ids
        )


class DynamicSchemaCollator:
    def __init__(
        self,
        codec: DynamicSchemaCodec,
        config: SchemaTrainingConfig,
        *,
        training: bool,
        binary_order_override: Sequence[str] | None = None,
    ) -> None:
        self.codec = codec
        self.config = config
        self.training = training
        self.binary_order_override = (
            tuple(binary_order_override) if binary_order_override is not None else None
        )
        self.rng = random.Random(config.seed + (1 if training else 0))

    def _orders(self, example: GuardExample) -> tuple[list[str], list[str]]:
        binary = list(self.binary_order_override or SAFETY_LABELS)
        if self.binary_order_override is None and self.training and self.rng.random() < 0.5:
            binary.reverse()
        categories: list[str] = []
        if self.config.enable_categories and example.category_scope != "unavailable":
            if not self.training or self.rng.random() < self.config.full_category_schema_probability:
                categories = list(N23_CATEGORIES)
            else:
                positives = set(example.categories)
                categories = [
                    label
                    for label in N23_CATEGORIES
                    if label in positives
                    or self.rng.random() < self.config.negative_label_keep_probability
                ]
                if not categories:
                    categories = [self.rng.choice(N23_CATEGORIES)]
                if positives and not positives.issubset(categories):
                    raise AssertionError("Dynamic schema dropped a positive N23 label")
            if self.training:
                self.rng.shuffle(categories)
        return binary, categories

    def __call__(self, examples: Sequence[GuardExample]) -> dict[str, Any]:
        encoded: list[EncodedDynamicSchema] = []
        for example in examples:
            binary, categories = self._orders(example)
            encoded.append(self.codec.encode(example, binary, categories))
        maximum = max(len(item.input_ids) for item in encoded)
        input_ids = torch.full(
            (len(encoded), maximum),
            int(self.codec.tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        max_categories = max((len(item.category_positions) for item in encoded), default=0)
        category_positions = torch.full((len(encoded), max_categories), -1, dtype=torch.long)
        category_targets = torch.zeros((len(encoded), max_categories), dtype=torch.float32)
        category_mask = torch.zeros((len(encoded), max_categories), dtype=torch.bool)
        for row, item in enumerate(encoded):
            length = len(item.input_ids)
            input_ids[row, :length] = torch.tensor(item.input_ids)
            attention_mask[row, :length] = 1
            count = len(item.category_positions)
            if count:
                category_positions[row, :count] = torch.tensor(item.category_positions)
                category_targets[row, :count] = torch.tensor(item.category_targets)
                category_mask[row, :count] = True
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "binary_positions": torch.tensor([item.binary_positions for item in encoded]),
            "binary_targets": torch.tensor([item.binary_target for item in encoded]),
            "binary_orders": [item.binary_order for item in encoded],
            "category_positions": category_positions,
            "category_targets": category_targets,
            "category_mask": category_mask,
            "category_orders": [item.category_order for item in encoded],
            "examples": list(examples),
        }


class DynamicSchemaGuard(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, category_loss_weight: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, 1),
        )
        self.category_loss_weight = category_loss_weight

    def _score(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
        anchors = hidden[rows, positions.clamp_min(0)]
        return self.classifier(anchors).squeeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        binary_positions: torch.Tensor,
        binary_targets: torch.Tensor | None = None,
        category_positions: torch.Tensor | None = None,
        category_targets: torch.Tensor | None = None,
        category_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        binary_logits = self._score(hidden, binary_positions)
        output: dict[str, torch.Tensor] = {"binary_logits": binary_logits}
        if category_positions is not None and category_positions.shape[1] > 0:
            output["category_logits"] = self._score(hidden, category_positions)
        if binary_targets is not None:
            binary_loss = F.cross_entropy(binary_logits.float(), binary_targets)
            total = binary_loss
            output["binary_loss"] = binary_loss
            if (
                "category_logits" in output
                and category_targets is not None
                and category_mask is not None
                and bool(category_mask.any().item())
            ):
                category_loss = F.binary_cross_entropy_with_logits(
                    output["category_logits"][category_mask].float(),
                    category_targets[category_mask].float(),
                )
                total = total + self.category_loss_weight * category_loss
                output["category_loss"] = category_loss
            output["loss"] = total
        return output


def load_schema_tokenizer(model_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, fix_mistral_regex=False
    )
    tokenizer.add_special_tokens({"additional_special_tokens": list(SCHEMA_SPECIAL_TOKENS)})
    return tokenizer


def prepare_schema_examples(
    examples: Sequence[GuardExample], codec: DynamicSchemaCodec
) -> tuple[list[GuardExample], list[int], list[dict[str, Any]]]:
    prepared: list[GuardExample] = []
    lengths: list[int] = []
    audits: list[dict[str, Any]] = []
    for index, example in enumerate(examples, 1):
        original = codec.full_length(example)
        if original <= codec.max_length:
            prepared.append(example)
            lengths.append(original)
        else:
            fitted, audit = truncate_text_exact(
                example.text,
                example.view,
                lambda value: codec.full_length(example, value),
                codec.max_length,
            )
            prepared.append(replace(example, text=fitted))
            lengths.append(audit.final_tokens)
            audits.append({"example_id": example.example_id, **audit.to_dict()})
        if index % 10_000 == 0:
            print(f"schema_preprocess={index:,}/{len(examples):,}", flush=True)
    if any(length > codec.max_length for length in lengths):
        raise AssertionError("Schema preprocessing left an over-limit example")
    return prepared, lengths, audits


@torch.inference_mode()
def evaluate_schema(
    model: DynamicSchemaGuard,
    dataset: ExampleDataset,
    collator: DynamicSchemaCollator,
    device: torch.device,
    *,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    targets: list[int] = []
    predictions: list[int] = []
    unsafe_probabilities: list[float] = []
    rows: list[dict[str, Any]] = []
    n23_true: list[list[int]] = []
    n23_pred: list[list[int]] = []
    losses: list[float] = []
    for raw in tqdm(loader, desc="schema validation", leave=False):
        batch = _to_device(raw, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                binary_positions=batch["binary_positions"],
                binary_targets=batch["binary_targets"],
                category_positions=batch["category_positions"],
                category_targets=batch["category_targets"],
                category_mask=batch["category_mask"],
            )
        losses.append(float(output["loss"].item()))
        binary_probs = output["binary_logits"].float().softmax(-1)
        local_predictions = binary_probs.argmax(-1)
        for row_index, example in enumerate(raw["examples"]):
            order = raw["binary_orders"][row_index]
            predicted_label = order[int(local_predictions[row_index].item())]
            target_label = order[int(batch["binary_targets"][row_index].item())]
            semantic_target = SAFETY_LABELS.index(target_label)
            semantic_prediction = SAFETY_LABELS.index(predicted_label)
            targets.append(semantic_target)
            predictions.append(semantic_prediction)
            unsafe_probability = float(binary_probs[row_index, order.index("unsafe")].item())
            unsafe_probabilities.append(unsafe_probability)
            prediction_row = {
                "example_id": example.example_id,
                "record_uid": example.record_uid,
                "view": example.view,
                "scope": example.safety_scope,
                "language": example.language,
                "tag": example.tag,
                "target": semantic_target,
                "prediction": semantic_prediction,
                "unsafe_probability": unsafe_probability,
                "serialized_tokens": int(batch["attention_mask"][row_index].sum().item()),
                "binary_label_order": list(order),
                "category_scope": example.category_scope,
                "category_gold": list(example.categories),
            }
            rows.append(prediction_row)
            category_order = raw["category_orders"][row_index]
            if category_order:
                truth = [0] * len(N23_CATEGORIES)
                pred = [0] * len(N23_CATEGORIES)
                count = len(category_order)
                local_pred = (
                    output["category_logits"][row_index, :count].sigmoid() >= 0.5
                ).cpu().tolist()
                local_true = batch["category_targets"][row_index, :count].cpu().tolist()
                for label, expected, guessed in zip(category_order, local_true, local_pred):
                    position = N23_CATEGORIES.index(label)
                    truth[position] = int(expected)
                    pred[position] = int(guessed)
                n23_true.append(truth)
                n23_pred.append(pred)
                prediction_row["category_schema"] = list(category_order)
                prediction_row["category_predictions"] = [
                    label for label, guessed in zip(category_order, local_pred) if guessed
                ]
                prediction_row["category_probabilities"] = {
                    label: float(probability)
                    for label, probability in zip(
                        category_order,
                        output["category_logits"][row_index, :count].float().sigmoid().cpu().tolist(),
                    )
                }
    result: dict[str, Any] = {
        "examples": len(targets),
        "loss": float(np.mean(losses)),
        "binary": _binary_summary(targets, predictions, unsafe_probabilities),
        "predictions": rows,
    }
    if n23_true:
        result["N23"] = {
            "supervised_examples": len(n23_true),
            "micro_f1": float(f1_score(n23_true, n23_pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(n23_true, n23_pred, average="macro", zero_division=0)),
        }
    return result


def train_schema_multitask(
    train_manifest: Path,
    valid_manifest: Path,
    output_dir: Path,
    config: SchemaTrainingConfig,
    *,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    _set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = _precision(config, device)  # compatible config fields
    tokenizer = load_schema_tokenizer(config.model_path)
    codec = DynamicSchemaCodec(tokenizer, config.max_length)
    train_raw = load_manifest(train_manifest)
    valid_raw = load_manifest(valid_manifest)
    train_examples, train_lengths, train_audit = prepare_schema_examples(train_raw, codec)
    valid_examples, valid_lengths, valid_audit = prepare_schema_examples(valid_raw, codec)
    for name, audit in (("train", train_audit), ("valid", valid_audit)):
        with (output_dir / f"{name}_truncation_audit.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for row in audit:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    base = AutoModel.from_pretrained(config.model_path, local_files_only=True)
    base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    hidden_size = int(base.config.hidden_size)
    if config.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    encoder = get_peft_model(
        base,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules="all-linear",
            trainable_token_indices=list(codec.marker_token_ids),
        ),
    )
    model = DynamicSchemaGuard(encoder, hidden_size, config.category_loss_weight).to(device)
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("encoder.")
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("encoder.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": config.encoder_learning_rate},
            {"params": head_parameters, "lr": config.head_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    train_dataset = ExampleDataset(train_examples, train_lengths)
    valid_dataset = ExampleDataset(valid_examples, valid_lengths)
    updates_per_epoch = math.ceil(
        math.ceil(len(train_dataset) / config.micro_batch_size)
        / config.gradient_accumulation_steps
    )
    planned_steps = updates_per_epoch * config.epochs
    if config.max_optimizer_steps > 0:
        planned_steps = min(planned_steps, config.max_optimizer_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(round(planned_steps * config.warmup_ratio)), planned_steps
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
        init_scale=config.grad_scaler_init_scale,
        growth_interval=config.grad_scaler_growth_interval,
    )
    train_collator = DynamicSchemaCollator(codec, config, training=True)
    valid_collator = DynamicSchemaCollator(codec, config, training=False)
    state = {"epoch": 0, "batch_in_epoch": 0, "optimizer_step": 0, "micro_step": 0}
    if resume_from is not None:
        state = _load_checkpoint(resume_from, model, optimizer, scheduler, scaler)
        # Checkpoints are optimizer-boundary snapshots; gradients are not part of
        # them, so accumulation must always restart from an empty window.  This
        # also normalizes checkpoints written by the legacy lifetime counter.
        state["micro_step"] = 0
    state.setdefault("nonfinite_optimizer_updates", 0)
    state.setdefault("nonfinite_micro_batches", 0)
    optimizer.zero_grad(set_to_none=True)
    encoder_gradient_verified = False
    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    best_score: tuple[float, float] | None = None
    best_checkpoint: Path | None = None
    started = time.perf_counter()
    stop = False
    for epoch in range(int(state["epoch"]), config.epochs):
        sampler = LengthBucketBatchSampler(
            train_lengths,
            config.micro_batch_size,
            seed=config.seed,
            epoch=epoch,
            bucket_multiplier=config.bucket_multiplier,
            shuffle=True,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=train_collator,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        skip = int(state["batch_in_epoch"]) if epoch == int(state["epoch"]) else 0
        progress = tqdm(loader, desc=f"schema epoch {epoch + 1}/{config.epochs}")
        for batch_index, raw in enumerate(progress):
            if batch_index < skip:
                continue
            model.train()
            batch = _to_device(raw, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    binary_positions=batch["binary_positions"],
                    binary_targets=batch["binary_targets"],
                    category_positions=batch["category_positions"],
                    category_targets=batch["category_targets"],
                    category_mask=batch["category_mask"],
                )
                loss = output["loss"] / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if not encoder_gradient_verified:
                if not any(parameter.grad is not None for parameter in encoder_parameters):
                    raise RuntimeError(
                        "No trainable encoder/LoRA parameter received a gradient after backward"
                    )
                encoder_gradient_verified = True
            state["micro_step"] += 1
            state["batch_in_epoch"] = batch_index + 1
            update = (
                state["micro_step"] >= config.gradient_accumulation_steps
                or batch_index + 1 == len(loader)
            )
            if not update:
                continue
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < scale_before
            optimizer.zero_grad(set_to_none=True)
            accumulated_micro_batches = int(state["micro_step"])
            state["micro_step"] = 0
            if skipped:
                state["nonfinite_optimizer_updates"] += 1
                state["nonfinite_micro_batches"] += accumulated_micro_batches
                overflow_item = {
                    "epoch": epoch + 1,
                    "batch_index": batch_index,
                    "optimizer_step": state["optimizer_step"],
                    "scale_before": float(scale_before),
                    "scale_after": float(scaler.get_scale()),
                    "estimated_micro_batches_skipped": accumulated_micro_batches,
                }
                with (output_dir / "nonfinite_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(overflow_item, ensure_ascii=False) + "\n")
                progress.set_postfix(step=state["optimizer_step"], skipped="nonfinite")
                continue
            scheduler.step()
            state["optimizer_step"] += 1
            item = {
                "epoch": epoch + 1,
                "optimizer_step": state["optimizer_step"],
                "loss": float(output["loss"].item()),
                "binary_loss": float(output["binary_loss"].item()),
                "category_loss": (
                    float(output["category_loss"].item()) if "category_loss" in output else None
                ),
                "grad_norm": grad_norm,
                "max_tokens": int(batch["attention_mask"].sum(1).max().item()),
                "mean_schema_category_labels": float(batch["category_mask"].sum(1).float().mean().item()),
            }
            history.append(item)
            progress.set_postfix(step=state["optimizer_step"], loss=f"{item['loss']:.4f}")
            if state["optimizer_step"] % config.log_every == 0:
                with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            if config.save_every > 0 and state["optimizer_step"] % config.save_every == 0:
                checkpoint = output_dir / "checkpoints" / f"step-{state['optimizer_step']:07d}"
                _save_checkpoint(checkpoint, model, optimizer, scheduler, scaler, state)
            if config.max_optimizer_steps > 0 and state["optimizer_step"] >= config.max_optimizer_steps:
                stop = True
                break
        epoch_validation = evaluate_schema(
            model,
            valid_dataset,
            valid_collator,
            device,
            batch_size=config.micro_batch_size,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        score = (
            float(epoch_validation["binary"].get("unsafe_auprc") or -1.0),
            float(epoch_validation["binary"]["macro_f1"]),
        )
        validation_item = {
            "epoch": epoch + 1,
            "optimizer_step": state["optimizer_step"],
            "score": list(score),
            "binary": epoch_validation["binary"],
            "N23": epoch_validation.get("N23"),
        }
        validation_history.append(validation_item)
        with (output_dir / "validation_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(validation_item, ensure_ascii=False) + "\n")
        if best_score is None or score > best_score:
            best_score = score
            best_checkpoint = output_dir / "checkpoints" / "best"
            _save_checkpoint(best_checkpoint, model, optimizer, scheduler, scaler, state)
        if stop:
            break
        state["epoch"] = epoch + 1
        state["batch_in_epoch"] = 0

    if best_checkpoint is None:
        raise AssertionError("No validation checkpoint was selected")
    best_weights = torch.load(
        best_checkpoint / "trainable_model.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(best_weights, strict=False)
    validation = evaluate_schema(
        model,
        valid_dataset,
        valid_collator,
        device,
        batch_size=config.micro_batch_size,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(final_dir / "adapter", save_embedding_layers=False)
    torch.save(model.classifier.state_dict(), final_dir / "shared_label_mlp.pt")
    tokenizer.save_pretrained(final_dir / "tokenizer")
    (final_dir / "run_contract.json").write_text(
        json.dumps(
            {
                "model_kind": "mmbert_dynamic_schema_multitask",
                "config": asdict(config),
                "binary": "one dynamic task; softmax CE over two [L] anchors",
                "N23": "optional dynamic task; independent BCE over present [L] anchors",
                "unavailable_category_scope": "N23 task omitted, not encoded as 23 negatives",
                "schema_augmentation": "shuffle plus negative-label subsampling; all positives retained",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "completed",
        "config": asdict(config),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "autocast_dtype": str(amp_dtype),
        "encoder_gradient_verified": encoder_gradient_verified,
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "train_truncated": len(train_audit),
        "valid_truncated": len(valid_audit),
        "planned_optimizer_steps": planned_steps,
        "completed_optimizer_steps": state["optimizer_step"],
        "nonfinite_optimizer_updates": state["nonfinite_optimizer_updates"],
        "nonfinite_micro_batches": state["nonfinite_micro_batches"],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_mb": (
            float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
        ),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "validation": validation,
        "history_tail": history[-100:],
        "validation_history": validation_history,
        "selection_metric": "valid unsafe_auprc then macro_f1",
        "best_score": list(best_score) if best_score is not None else None,
        "best_checkpoint": str(best_checkpoint),
        "final": str(final_dir),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
