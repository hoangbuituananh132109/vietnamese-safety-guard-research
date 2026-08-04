from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from guard_smoke.constants import N23_CATEGORIES, N23_TO_ID, SAFETY_TO_ID
from guard_smoke.data import GuardExample, load_manifest
from guard_train.precision import cuda_supports_native_bf16
from guard_train.truncation import truncate_text_exact


@dataclass
class FixedTrainingConfig:
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
    use_lora: bool = True
    lora_r: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    gradient_checkpointing: bool = True
    mixed_precision: str = "auto"
    # Turing GPUs have FP16 tensor cores but no native BF16.  A conservative
    # static-ish scale avoids silently dropping long effective batches.
    grad_scaler_init_scale: float = 512.0
    grad_scaler_growth_interval: int = 1_000_000
    enable_categories: bool = True
    category_loss_weight: float = 1.0
    bucket_multiplier: int = 64
    num_workers: int = 0
    log_every: int = 10
    eval_every: int = 0
    save_every: int = 250
    seed: int = 3407

    @property
    def gradient_accumulation_steps(self) -> int:
        if self.effective_batch_size % self.micro_batch_size != 0:
            raise ValueError("effective_batch_size must be divisible by micro_batch_size")
        return self.effective_batch_size // self.micro_batch_size


class ExampleDataset(Dataset):
    def __init__(self, examples: Sequence[GuardExample], lengths: Sequence[int]) -> None:
        if len(examples) != len(lengths):
            raise ValueError("examples/lengths mismatch")
        self.examples = list(examples)
        self.lengths = list(lengths)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> GuardExample:
        return self.examples[index]


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Deterministic shuffled buckets; batches contain similar sequence lengths."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        seed: int,
        epoch: int,
        bucket_multiplier: int = 64,
        shuffle: bool = True,
    ) -> None:
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.bucket_size = max(batch_size, batch_size * bucket_multiplier)
        self.shuffle = shuffle

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=self.lengths.__getitem__)
            for offset in range(0, len(bucket), self.batch_size):
                batches.append(bucket[offset : offset + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


class FixedMultiTaskCollator:
    def __init__(self, tokenizer: Any, max_length: int, enable_categories: bool) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_categories = enable_categories

    def __call__(self, examples: Sequence[GuardExample]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [example.text for example in examples],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > self.max_length:
            raise ValueError("Preprocessed fixed-head text exceeded max_length")
        category_targets = torch.zeros(
            (len(examples), len(N23_CATEGORIES)), dtype=torch.float32
        )
        category_mask = torch.zeros(len(examples), dtype=torch.bool)
        if self.enable_categories:
            for row, example in enumerate(examples):
                available = example.category_scope != "unavailable"
                category_mask[row] = available
                if available:
                    for category in example.categories:
                        category_targets[row, N23_TO_ID[category]] = 1.0
        return {
            **encoded,
            "safety_targets": torch.tensor(
                [SAFETY_TO_ID[example.safety_label] for example in examples],
                dtype=torch.long,
            ),
            "category_targets": category_targets,
            "category_mask": category_mask,
            "examples": list(examples),
        }


class FixedMultiTaskGuard(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        *,
        enable_categories: bool,
        category_loss_weight: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.safety_head = nn.Linear(hidden_size, 2)
        self.category_head = (
            nn.Linear(hidden_size, len(N23_CATEGORIES)) if enable_categories else None
        )
        self.category_loss_weight = category_loss_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        safety_targets: torch.Tensor | None = None,
        category_targets: torch.Tensor | None = None,
        category_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        safety_logits = self.safety_head(pooled)
        output: dict[str, torch.Tensor] = {"safety_logits": safety_logits}
        if self.category_head is not None:
            output["category_logits"] = self.category_head(pooled)
        if safety_targets is not None:
            binary_loss = F.cross_entropy(safety_logits.float(), safety_targets)
            total = binary_loss
            output["binary_loss"] = binary_loss
            if (
                self.category_head is not None
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _precision(config: FixedTrainingConfig, device: torch.device) -> tuple[bool, torch.dtype]:
    value = config.mixed_precision
    if value == "auto":
        value = "bf16" if device.type == "cuda" and cuda_supports_native_bf16(device) else "fp16"
    if device.type != "cuda" or value == "fp32":
        return False, torch.float32
    if value == "bf16":
        if not cuda_supports_native_bf16(device):
            raise RuntimeError("BF16 requested on GPU without native BF16 support")
        return True, torch.bfloat16
    if value == "fp16":
        return True, torch.float16
    raise ValueError(f"Unsupported precision: {config.mixed_precision}")


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def prepare_fixed_examples(
    examples: Sequence[GuardExample], tokenizer: Any, max_length: int
) -> tuple[list[GuardExample], list[int], list[dict[str, Any]]]:
    """Retain every row and scope-aware truncate only exact tokenizer tails."""

    prepared: list[GuardExample] = []
    lengths: list[int] = []
    audits: list[dict[str, Any]] = []
    batch_size = 256
    raw_lengths: list[int] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        encoded = tokenizer(
            [example.text for example in chunk],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True,
        )
        raw_lengths.extend(int(value) for value in encoded["length"])
    for example, raw_length in zip(examples, raw_lengths):
        if raw_length <= max_length:
            prepared.append(example)
            lengths.append(raw_length)
            continue

        def length(value: str) -> int:
            return len(tokenizer.encode(value, add_special_tokens=True))

        fitted, audit = truncate_text_exact(
            example.text, example.view, length, max_length
        )
        prepared.append(replace(example, text=fitted))
        lengths.append(audit.final_tokens)
        audits.append({"example_id": example.example_id, **audit.to_dict()})
    if len(prepared) != len(examples) or any(value > max_length for value in lengths):
        raise AssertionError("Fixed-head preprocessing did not preserve/fix every row")
    return prepared, lengths, audits


def _binary_summary(
    targets: list[int],
    predictions: list[int],
    unsafe_probabilities: list[float] | None = None,
) -> dict[str, float | None]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=[0, 1], average=None, zero_division=0
    )
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=[0, 1], average="macro")),
        "safe_precision": float(precision[0]),
        "safe_recall": float(recall[0]),
        "unsafe_precision": float(precision[1]),
        "unsafe_recall": float(recall[1]),
        "unsafe_f1": float(f1[1]),
    }
    if unsafe_probabilities is not None:
        probabilities = np.asarray(unsafe_probabilities, dtype=np.float64)
        truth = np.asarray(targets, dtype=np.float64)
        result.update(
            {
                "unsafe_auprc": (
                    float(average_precision_score(targets, unsafe_probabilities))
                    if any(targets)
                    else None
                ),
                "auroc": (
                    float(roc_auc_score(targets, unsafe_probabilities))
                    if len(set(targets)) == 2
                    else None
                ),
                "brier": float(np.mean((probabilities - truth) ** 2)),
            }
        )
    return result


@torch.inference_mode()
def evaluate(
    model: FixedMultiTaskGuard,
    dataset: ExampleDataset,
    collator: FixedMultiTaskCollator,
    device: torch.device,
    *,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    targets: list[int] = []
    predictions: list[int] = []
    unsafe_probabilities: list[float] = []
    rows: list[dict[str, Any]] = []
    category_true: list[np.ndarray] = []
    category_pred: list[np.ndarray] = []
    losses: list[float] = []
    for raw in tqdm(loader, desc="validation", leave=False):
        batch = _to_device(raw, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                safety_targets=batch["safety_targets"],
                category_targets=batch["category_targets"],
                category_mask=batch["category_mask"],
            )
        losses.append(float(output["loss"].item()))
        probs = output["safety_logits"].float().softmax(-1)[:, 1]
        preds = (probs >= 0.5).long()
        targets.extend(batch["safety_targets"].cpu().tolist())
        predictions.extend(preds.cpu().tolist())
        unsafe_probabilities.extend(probs.cpu().tolist())
        category_probabilities = (
            output["category_logits"].float().sigmoid()
            if "category_logits" in output
            else None
        )
        if category_probabilities is not None and bool(batch["category_mask"].any().item()):
            mask = batch["category_mask"]
            category_true.extend(batch["category_targets"][mask].cpu().numpy())
            category_pred.extend((category_probabilities[mask] >= 0.5).cpu().numpy())
        for index, example in enumerate(raw["examples"]):
            row = {
                "example_id": example.example_id,
                "record_uid": example.record_uid,
                "view": example.view,
                "scope": example.safety_scope,
                "language": example.language,
                "tag": example.tag,
                "target": int(batch["safety_targets"][index].item()),
                "prediction": int(preds[index].item()),
                "unsafe_probability": float(probs[index].item()),
                "serialized_tokens": int(batch["attention_mask"][index].sum().item()),
                "category_scope": example.category_scope,
                "category_gold": list(example.categories),
            }
            if category_probabilities is not None and bool(batch["category_mask"][index].item()):
                row["category_probabilities"] = {
                    label: float(category_probabilities[index, position].item())
                    for position, label in enumerate(N23_CATEGORIES)
                }
                row["category_predictions"] = [
                    label
                    for position, label in enumerate(N23_CATEGORIES)
                    if float(category_probabilities[index, position].item()) >= 0.5
                ]
            rows.append(row)
    result: dict[str, Any] = {
        "examples": len(targets),
        "loss": float(np.mean(losses)),
        "binary": _binary_summary(targets, predictions, unsafe_probabilities),
        "predictions": rows,
    }
    if category_true:
        truth = np.asarray(category_true, dtype=np.int64)
        pred = np.asarray(category_pred, dtype=np.int64)
        result["N23"] = {
            "supervised_examples": int(truth.shape[0]),
            "micro_f1": float(f1_score(truth, pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        }
    return result


def _trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    state: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(_trainable_state(model), path / "trainable_model.pt")
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(scheduler.state_dict(), path / "scheduler.pt")
    torch.save(scaler.state_dict(), path / "scaler.pt")
    (path / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    payload = torch.load(path / "trainable_model.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(payload, strict=False)
    unexpected = [name for name in unexpected if name in payload]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint parameters: {unexpected}")
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True))
    scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True))
    scaler.load_state_dict(torch.load(path / "scaler.pt", map_location="cpu", weights_only=True))
    return json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))


def train_fixed_multitask(
    train_manifest: Path,
    valid_manifest: Path,
    output_dir: Path,
    config: FixedTrainingConfig,
    *,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    _set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = _precision(config, device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, local_files_only=True, fix_mistral_regex=False
    )
    train_raw = load_manifest(train_manifest)
    valid_raw = load_manifest(valid_manifest)
    train_examples, train_lengths, train_audit = prepare_fixed_examples(
        train_raw, tokenizer, config.max_length
    )
    valid_examples, valid_lengths, valid_audit = prepare_fixed_examples(
        valid_raw, tokenizer, config.max_length
    )
    for name, audit in (("train", train_audit), ("valid", valid_audit)):
        with (output_dir / f"{name}_truncation_audit.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for row in audit:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    base = AutoModel.from_pretrained(config.model_path, local_files_only=True)
    hidden_size = int(base.config.hidden_size)
    if config.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    if config.use_lora:
        encoder = get_peft_model(
            base,
            LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules="all-linear",
            ),
        )
    else:
        encoder = base
    model = FixedMultiTaskGuard(
        encoder,
        hidden_size,
        enable_categories=config.enable_categories,
        category_loss_weight=config.category_loss_weight,
    ).to(device)
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
    warmup_steps = int(round(planned_steps * config.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, planned_steps)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
        init_scale=config.grad_scaler_init_scale,
        growth_interval=config.grad_scaler_growth_interval,
    )
    collator = FixedMultiTaskCollator(
        tokenizer, config.max_length, config.enable_categories
    )
    state = {"epoch": 0, "batch_in_epoch": 0, "optimizer_step": 0, "micro_step": 0}
    if resume_from is not None:
        state = _load_checkpoint(resume_from, model, optimizer, scheduler, scaler)
        # Checkpoints are written only immediately after an optimizer update and
        # gradients themselves are not serialized.  Therefore every exact resume
        # starts at a fresh accumulation window, even for legacy checkpoints whose
        # micro_step stored the lifetime number of micro-batches.
        state["micro_step"] = 0
    state.setdefault("nonfinite_optimizer_updates", 0)
    state.setdefault("nonfinite_micro_batches", 0)
    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    best_score: tuple[float, float] | None = None
    best_checkpoint: Path | None = None
    optimizer.zero_grad(set_to_none=True)
    encoder_gradient_verified = False
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
            collate_fn=collator,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        skip_batches = int(state["batch_in_epoch"]) if epoch == int(state["epoch"]) else 0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config.epochs}", initial=0)
        for batch_index, raw in enumerate(progress):
            if batch_index < skip_batches:
                continue
            model.train()
            batch = _to_device(raw, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    safety_targets=batch["safety_targets"],
                    category_targets=batch["category_targets"],
                    category_mask=batch["category_mask"],
                )
                scaled_loss = output["loss"] / config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            if not encoder_gradient_verified:
                if not any(parameter.grad is not None for parameter in encoder_parameters):
                    raise RuntimeError(
                        "No trainable encoder/LoRA parameter received a gradient after backward"
                    )
                encoder_gradient_verified = True
            state["micro_step"] += 1
            state["batch_in_epoch"] = batch_index + 1
            should_update = (
                state["micro_step"] >= config.gradient_accumulation_steps
                or batch_index + 1 == len(loader)
            )
            if not should_update:
                continue
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_was_skipped = scaler.get_scale() < scale_before
            optimizer.zero_grad(set_to_none=True)
            accumulated_micro_batches = int(state["micro_step"])
            state["micro_step"] = 0
            if step_was_skipped:
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
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "max_tokens": int(batch["attention_mask"].sum(1).max().item()),
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
        epoch_validation = evaluate(
            model,
            valid_dataset,
            collator,
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
    validation = evaluate(
        model,
        valid_dataset,
        collator,
        device,
        batch_size=config.micro_batch_size,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if config.use_lora:
        model.encoder.save_pretrained(final_dir / "adapter")
    torch.save(
        {
            "safety_head": model.safety_head.state_dict(),
            "category_head": (
                model.category_head.state_dict() if model.category_head is not None else None
            ),
        },
        final_dir / "heads.pt",
    )
    tokenizer.save_pretrained(final_dir / "tokenizer")
    (final_dir / "run_contract.json").write_text(
        json.dumps(
            {
                "model_kind": "mmbert_fixed_multitask",
                "config": asdict(config),
                "binary": "softmax categorical CE on every instance",
                "N23": "23 independent sigmoid logits; BCE only when category_scope is available",
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
