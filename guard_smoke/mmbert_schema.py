"""Basic GLiGuard-style binary schema classifier on a local mmBERT encoder.

Phase 0 deliberately implements one task only::

    [P] text safety classification [L] safe [L] unsafe [SEP] <text>

Prompt/response/view are metadata used to materialize and audit examples; they
are not task names, routing features, or separate heads.  The two label anchors
share one scalar MLP and form one single-label softmax/CE problem.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from .constants import TEXT_SAFETY_TASK
from .data import GuardExample, load_manifest
from guard_train.precision import cuda_supports_native_bf16


TASK_MARKER = "[P]"
LABEL_MARKER = "[L]"
SCHEMA_SEPARATOR = "[SEP]"
SCHEMA_SPECIAL_TOKENS = (TASK_MARKER, LABEL_MARKER, SCHEMA_SEPARATOR)
TASK_NAME = TEXT_SAFETY_TASK
CANONICAL_LABELS = ("safe", "unsafe")
SEMANTIC_LABEL_TO_ID = {label: index for index, label in enumerate(CANONICAL_LABELS)}


@dataclass(frozen=True)
class MMBertSchemaConfig:
    model_path: str
    max_length: int = 256
    batch_size: int = 4
    epochs: int = 2
    max_steps: int = 24
    seed: int = 17
    lora_r: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    encoder_learning_rate: float = 2e-4
    head_learning_rate: float = 5e-4
    weight_decay: float = 0.01
    shuffle_labels: bool = True
    mixed_precision: bool = True
    gradient_checkpointing: bool = False


@dataclass(frozen=True)
class EncodedSchemaExample:
    input_ids: tuple[int, ...]
    label_positions: tuple[int, int]
    label_order: tuple[str, str]
    target_index: int


class BinarySchemaCodec:
    """Serialize the one Phase-0 task without truncation or scope routing."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        marker_ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in SCHEMA_SPECIAL_TOKENS)
        if any(token_id is None or token_id < 0 for token_id in marker_ids):
            raise ValueError(f"Unresolved schema markers: {dict(zip(SCHEMA_SPECIAL_TOKENS, marker_ids))}")
        if len(set(marker_ids)) != len(marker_ids):
            raise ValueError(f"Schema markers are not distinct: {marker_ids}")
        for token, token_id in zip(SCHEMA_SPECIAL_TOKENS, marker_ids):
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if encoded != [token_id]:
                raise ValueError(f"{token!r} must be one token, got {encoded}")
        self.task_marker_id, self.label_marker_id, self.separator_id = marker_ids
        self.marker_token_ids = marker_ids
        self.task_ids = tuple(tokenizer.encode(TASK_NAME, add_special_tokens=False))
        self.label_name_ids = {
            label: tuple(tokenizer.encode(label, add_special_tokens=False))
            for label in CANONICAL_LABELS
        }
        if not self.task_ids or any(not value for value in self.label_name_ids.values()):
            raise ValueError("Task and label names must tokenize to non-empty sequences")
        if tokenizer.pad_token_id is None:
            raise ValueError("mmBERT tokenizer must define pad_token_id")

    @property
    def canonical_schema_text(self) -> str:
        return (
            f"{TASK_MARKER} {TASK_NAME} "
            f"{LABEL_MARKER} safe {LABEL_MARKER} unsafe {SCHEMA_SEPARATOR}"
        )

    def encode(
        self,
        example: GuardExample,
        label_order: Sequence[str] = CANONICAL_LABELS,
        *,
        enforce_max_length: bool = True,
    ) -> EncodedSchemaExample:
        order = tuple(label_order)
        if len(order) != 2 or set(order) != set(CANONICAL_LABELS):
            raise ValueError(f"Phase-0 label order must be a permutation of {CANONICAL_LABELS}: {order}")
        if example.safety_label not in CANONICAL_LABELS:
            raise ValueError(f"Unsupported target {example.safety_label!r}")

        ids: list[int] = [self.task_marker_id, *self.task_ids]
        label_positions: list[int] = []
        for label in order:
            label_positions.append(len(ids))
            ids.append(self.label_marker_id)
            ids.extend(self.label_name_ids[label])
        ids.append(self.separator_id)
        schema_prefix_ids = tuple(ids)
        # The materialized text may be P, R, or PR, but no view/scope value is
        # passed separately and the schema task name is always identical.
        ids.extend(self.tokenizer.encode(example.text, add_special_tokens=False))

        if enforce_max_length and len(ids) > self.max_length:
            raise ValueError(
                f"Serialized sequence {example.example_id} has {len(ids)} tokens, "
                f"exceeding max_length={self.max_length}; Phase 0 does not truncate"
            )
        # Audit the generated schema prefix only. A safety/jailbreak text is
        # allowed to literally mention strings such as "[L]" without being
        # mistaken for an additional schema label anchor.
        if schema_prefix_ids.count(self.task_marker_id) != 1:
            raise AssertionError("Binary schema must contain exactly one [P]")
        if schema_prefix_ids.count(self.label_marker_id) != 2:
            raise AssertionError("Binary schema must contain exactly two [L] markers")
        if schema_prefix_ids.count(self.separator_id) != 1:
            raise AssertionError("Binary schema must contain exactly one [SEP]")
        if any(ids[position] != self.label_marker_id for position in label_positions):
            raise AssertionError("Recorded label positions do not point at [L]")

        return EncodedSchemaExample(
            input_ids=tuple(ids),
            label_positions=(label_positions[0], label_positions[1]),
            label_order=(order[0], order[1]),
            target_index=order.index(example.safety_label),
        )

    def length(self, example: GuardExample) -> int:
        return len(self.encode(example, enforce_max_length=False).input_ids)

    def partition_eligible(
        self, examples: Iterable[GuardExample]
    ) -> tuple[list[GuardExample], list[dict[str, Any]]]:
        eligible: list[GuardExample] = []
        tail: list[dict[str, Any]] = []
        for example in examples:
            length = self.length(example)
            if length <= self.max_length:
                eligible.append(example)
            else:
                tail.append(
                    {
                        "example_id": example.example_id,
                        "view": example.view,
                        "language": example.language,
                        "safety_label": example.safety_label,
                        "serialized_tokens": length,
                        "max_length": self.max_length,
                    }
                )
        return eligible, tail


class BinarySchemaCollator:
    """Pad binary-schema samples and retain per-sample semantic label order."""

    def __init__(
        self,
        codec: BinarySchemaCodec,
        *,
        shuffle_labels: bool,
        seed: int,
    ) -> None:
        self.codec = codec
        self.shuffle_labels = shuffle_labels
        self.rng = random.Random(seed)

    def __call__(self, examples: Sequence[GuardExample]) -> dict[str, Any]:
        encoded: list[EncodedSchemaExample] = []
        for example in examples:
            order = list(CANONICAL_LABELS)
            if self.shuffle_labels and self.rng.random() < 0.5:
                order.reverse()
            encoded.append(self.codec.encode(example, order))

        maximum = max(len(item.input_ids) for item in encoded)
        input_ids = torch.full(
            (len(encoded), maximum),
            int(self.codec.tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, item in enumerate(encoded):
            length = len(item.input_ids)
            input_ids[row, :length] = torch.tensor(item.input_ids, dtype=torch.long)
            attention_mask[row, :length] = 1

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


class MMBertBinarySchemaGuard(nn.Module):
    """mmBERT encoder plus one shared GLi-style scalar label scorer."""

    def __init__(self, encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        # Matches GLiNER2's classifier recipe: d -> 2d -> 1, ReLU, no dropout.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, 1),
        )

    @classmethod
    def create_for_training(
        cls,
        model_path: str,
        tokenizer_size: int,
        marker_token_ids: Sequence[int],
        *,
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        gradient_checkpointing: bool,
    ) -> "MMBertBinarySchemaGuard":
        base = AutoModel.from_pretrained(model_path, local_files_only=True)
        base.resize_token_embeddings(tokenizer_size, mean_resizing=False)
        hidden_size = int(base.config.hidden_size)
        if gradient_checkpointing:
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
        encoder = get_peft_model(
            base,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules="all-linear",
                # PEFT stores and trains only these rows, not the 256K-row table.
                trainable_token_indices=list(marker_token_ids),
            ),
        )
        return cls(encoder, hidden_size)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: Path,
        model_path: str,
        tokenizer_size: int,
        device: torch.device,
    ) -> "MMBertBinarySchemaGuard":
        base = AutoModel.from_pretrained(model_path, local_files_only=True)
        base.resize_token_embeddings(tokenizer_size, mean_resizing=False)
        hidden_size = int(base.config.hidden_size)
        encoder = PeftModel.from_pretrained(
            base,
            checkpoint_dir / "adapter",
            is_trainable=False,
        )
        model = cls(encoder, hidden_size)
        payload = torch.load(
            checkpoint_dir / "schema_head.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.classifier.load_state_dict(payload["classifier"])
        return model.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_positions: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        if label_positions.ndim != 2 or label_positions.shape[1] != 2:
            raise ValueError(
                "Phase-0 binary schema requires label_positions with shape [batch, 2]"
            )
        row_indices = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
        label_hidden = hidden[row_indices, label_positions]
        logits = self.classifier(label_hidden).squeeze(-1)
        result = {"logits": logits, "label_hidden": label_hidden}
        if targets is not None:
            # This is a single-label task: exactly one softmax class is correct.
            result["loss"] = F.cross_entropy(logits.float(), targets)
        return result


def load_mmbert_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        # Transformers 4.57 otherwise misidentifies mmBERT's Gemma-2 Metaspace
        # pre-tokenizer as a Mistral regex tokenizer.
        fix_mistral_regex=False,
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(SCHEMA_SPECIAL_TOKENS)}
    )
    return tokenizer


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _binary_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    if not targets:
        return {
            "accuracy": math.nan,
            "macro_f1": math.nan,
            "safe_precision": math.nan,
            "safe_recall": math.nan,
            "unsafe_precision": math.nan,
            "unsafe_recall": math.nan,
            "unsafe_f1": math.nan,
        }
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=[0, 1], average="macro")),
        "safe_precision": float(precision[0]),
        "safe_recall": float(recall[0]),
        "unsafe_precision": float(precision[1]),
        "unsafe_recall": float(recall[1]),
        "unsafe_f1": float(f1[1]),
    }


@torch.no_grad()
def evaluate_binary_schema_guard(
    model: MMBertBinarySchemaGuard,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled and device.type == "cuda",
        ):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                label_positions=batch["label_positions"],
                targets=batch["targets"],
            )
        losses.append(float(output["loss"].item()))
        probabilities = output["logits"].float().softmax(dim=-1)
        predicted_indices = probabilities.argmax(dim=-1)
        for index, example in enumerate(raw_batch["examples"]):
            order = tuple(raw_batch["label_orders"][index])
            predicted_label = order[int(predicted_indices[index].item())]
            target_label = order[int(batch["targets"][index].item())]
            if target_label != example.safety_label:
                raise AssertionError("Target index no longer aligns with semantic label order")
            unsafe_index = order.index("unsafe")
            rows.append(
                {
                    "example_id": example.example_id,
                    "record_uid": example.record_uid,
                    "view": example.view,
                    "scope": example.safety_scope,
                    "language": example.language,
                    "tag": example.tag,
                    "target": SEMANTIC_LABEL_TO_ID[target_label],
                    "target_label": target_label,
                    "prediction": SEMANTIC_LABEL_TO_ID[predicted_label],
                    "predicted_label": predicted_label,
                    "unsafe_probability": float(probabilities[index, unsafe_index].item()),
                    "label_order": list(order),
                    "serialized_tokens": int(raw_batch["sequence_lengths"][index].item()),
                }
            )

    metrics: dict[str, Any] = {
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
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
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

    paired: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(row["record_uid"], row["view"])][row["language"]] = row
    complete_pairs = [pair for pair in paired.values() if set(pair) == {"en", "vi"}]
    metrics["paired_en_vi"] = {
        "pairs": len(complete_pairs),
        "same_decision_rate": (
            float(np.mean([pair["en"]["prediction"] == pair["vi"]["prediction"] for pair in complete_pairs]))
            if complete_pairs
            else math.nan
        ),
        "mean_probability_gap": (
            float(
                np.mean(
                    [
                        abs(pair["en"]["unsafe_probability"] - pair["vi"]["unsafe_probability"])
                        for pair in complete_pairs
                    ]
                )
            )
            if complete_pairs
            else math.nan
        ),
    }
    return metrics


def run_schema_contract_checks(
    codec: BinarySchemaCodec,
    examples: Sequence[GuardExample],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("Contract checks require at least one example")
    example = examples[0]
    canonical = codec.encode(example, ("safe", "unsafe"), enforce_max_length=False)
    reversed_order = codec.encode(example, ("unsafe", "safe"), enforce_max_length=False)
    expected_canonical = ("safe", "unsafe").index(example.safety_label)
    expected_reversed = ("unsafe", "safe").index(example.safety_label)
    if canonical.target_index != expected_canonical:
        raise AssertionError("Canonical label target is misaligned")
    if reversed_order.target_index != expected_reversed:
        raise AssertionError("Reversed label target is misaligned")
    if canonical.target_index == reversed_order.target_index:
        raise AssertionError("Target index should change when the two labels are reversed")
    for item in (canonical, reversed_order):
        if [item.input_ids[position] for position in item.label_positions] != [
            codec.label_marker_id,
            codec.label_marker_id,
        ]:
            raise AssertionError("Label positions do not point at [L]")
    return {
        "passed": True,
        "task_name": TASK_NAME,
        "task_type": "single_label",
        "loss": "softmax_cross_entropy",
        "schema_fields": ["task_name", "labels", "text"],
        "metadata_not_routed_to_model": ["view", "safety_scope", "language", "tag"],
        "canonical_schema": codec.canonical_schema_text,
        "marker_token_ids": dict(zip(SCHEMA_SPECIAL_TOKENS, codec.marker_token_ids)),
        "canonical_target_index": canonical.target_index,
        "reversed_target_index": reversed_order.target_index,
        "canonical_length": len(canonical.input_ids),
    }


def _parameter_group(name: str) -> str:
    if "trainable_tokens_delta" in name:
        return "marker_embeddings"
    if ".lora_A." in name or ".lora_B." in name:
        return "lora"
    if name.startswith("classifier."):
        return "shared_classifier"
    return "unexpected"


def _gradient_norm(parameters: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def save_binary_schema_checkpoint(
    model: MMBertBinarySchemaGuard,
    tokenizer: PreTrainedTokenizerBase,
    checkpoint_dir: Path,
    config: MMBertSchemaConfig,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # ``auto`` treats any resized vocabulary as a reason to dump the complete
    # 256K-row embedding table (~394 MB).  ``trainable_token_indices`` already
    # stores the three replacement rows inside the PEFT adapter, so saving the
    # frozen full table is both redundant and contrary to the Phase-0 contract.
    model.encoder.save_pretrained(
        checkpoint_dir / "adapter",
        save_embedding_layers=False,
    )
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    torch.save(
        {
            "classifier": {
                key: value.detach().cpu()
                for key, value in model.classifier.state_dict().items()
            }
        },
        checkpoint_dir / "schema_head.pt",
    )
    (checkpoint_dir / "schema_contract.json").write_text(
        json.dumps(
            {
                "phase": 0,
                "task_name": TASK_NAME,
                "task_type": "single_label",
                "labels": list(CANONICAL_LABELS),
                "special_tokens": list(SCHEMA_SPECIAL_TOKENS),
                "activation": "softmax",
                "loss": "categorical_cross_entropy",
                "config": asdict(config),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def train_binary_schema_guard(
    train_manifest: Path,
    valid_manifest: Path,
    output_dir: Path,
    config: MMBertSchemaConfig,
) -> dict[str, Any]:
    _set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tokenizer = load_mmbert_tokenizer(config.model_path)
    codec = BinarySchemaCodec(tokenizer, config.max_length)
    all_train = load_manifest(train_manifest)
    all_valid = load_manifest(valid_manifest)
    train_examples, train_tail = codec.partition_eligible(all_train)
    valid_examples, valid_tail = codec.partition_eligible(all_valid)
    if not train_examples or not valid_examples:
        raise ValueError("No eligible train or validation examples after exact schema token audit")
    contract_checks = run_schema_contract_checks(codec, train_examples)

    model = MMBertBinarySchemaGuard.create_for_training(
        config.model_path,
        len(tokenizer),
        codec.marker_token_ids,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)

    trainable_named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    grouped_parameters: dict[str, list[torch.Tensor]] = defaultdict(list)
    grouped_names: dict[str, list[str]] = defaultdict(list)
    for name, parameter in trainable_named:
        group = _parameter_group(name)
        grouped_parameters[group].append(parameter)
        grouped_names[group].append(name)
    if grouped_names.get("unexpected"):
        raise AssertionError(f"Unexpected trainable parameters: {grouped_names['unexpected']}")
    for required in ("marker_embeddings", "lora", "shared_classifier"):
        if not grouped_parameters.get(required):
            raise AssertionError(f"Missing required trainable parameter group: {required}")

    marker_before = [parameter.detach().cpu().clone() for parameter in grouped_parameters["marker_embeddings"]]
    train_collator = BinarySchemaCollator(
        codec,
        shuffle_labels=config.shuffle_labels,
        seed=config.seed,
    )
    valid_collator = BinarySchemaCollator(codec, shuffle_labels=False, seed=config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_examples,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=train_collator,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_examples,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=valid_collator,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": grouped_parameters["lora"] + grouped_parameters["marker_embeddings"],
                "lr": config.encoder_learning_rate,
            },
            {
                "params": grouped_parameters["shared_classifier"],
                "lr": config.head_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.mixed_precision and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and cuda_supports_native_bf16(device)
        else torch.float16
        if amp_enabled
        else torch.float32
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    history: list[dict[str, Any]] = []
    first_step_gradient_norms: dict[str, float] | None = None
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
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    label_positions=batch["label_positions"],
                    targets=batch["targets"],
                )
            if not bool(torch.isfinite(output["loss"]).item()):
                raise FloatingPointError(f"Non-finite loss at step {global_step + 1}")
            scaler.scale(output["loss"]).backward()
            scaler.unscale_(optimizer)
            if first_step_gradient_norms is None:
                first_step_gradient_norms = {
                    name: _gradient_norm(grouped_parameters[name])
                    for name in ("marker_embeddings", "lora", "shared_classifier")
                }
                if any(
                    not math.isfinite(value) or value <= 0
                    for value in first_step_gradient_norms.values()
                ):
                    raise AssertionError(
                        f"Missing/non-finite Phase-0 gradients: {first_step_gradient_norms}"
                    )
            total_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in trainable_named], 1.0
                ).item()
            )
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": float(output["loss"].item()),
                    "grad_norm_before_clip": total_grad_norm,
                    "batch_size": len(raw_batch["examples"]),
                    "max_serialized_tokens": int(raw_batch["sequence_lengths"].max().item()),
                    "label_orders": [list(order) for order in raw_batch["label_orders"]],
                }
            )
            if config.max_steps > 0 and global_step >= config.max_steps:
                stop = True
                break
        if stop:
            break

    elapsed_seconds = time.perf_counter() - started
    marker_after = [parameter.detach().cpu() for parameter in grouped_parameters["marker_embeddings"]]
    marker_max_update = max(
        float((after - before).abs().max().item())
        for before, after in zip(marker_before, marker_after)
    )
    if marker_max_update <= 0:
        raise AssertionError("Trainable marker embeddings did not change")

    valid_metrics = evaluate_binary_schema_guard(
        model,
        valid_loader,
        device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    checkpoint_dir = output_dir / "checkpoint"
    save_binary_schema_checkpoint(model, tokenizer, checkpoint_dir, config)

    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir / "tokenizer",
        local_files_only=True,
        fix_mistral_regex=False,
    )
    reloaded_codec = BinarySchemaCodec(reloaded_tokenizer, config.max_length)
    reloaded_loader = DataLoader(
        valid_examples,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=BinarySchemaCollator(
            reloaded_codec,
            shuffle_labels=False,
            seed=config.seed,
        ),
        num_workers=0,
    )
    reloaded = MMBertBinarySchemaGuard.load_checkpoint(
        checkpoint_dir,
        config.model_path,
        len(reloaded_tokenizer),
        device,
    )
    reloaded_metrics = evaluate_binary_schema_guard(
        reloaded,
        reloaded_loader,
        device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    before_probabilities = {
        row["example_id"]: row["unsafe_probability"]
        for row in valid_metrics["predictions"]
    }
    after_probabilities = {
        row["example_id"]: row["unsafe_probability"]
        for row in reloaded_metrics["predictions"]
    }
    max_reload_probability_delta = max(
        abs(before_probabilities[key] - after_probabilities[key])
        for key in before_probabilities
    )
    if max_reload_probability_delta > 1e-5:
        raise AssertionError(
            f"Checkpoint reload parity failed: delta={max_reload_probability_delta}"
        )

    parameter_counts = {
        group: int(sum(parameter.numel() for parameter in parameters))
        for group, parameters in grouped_parameters.items()
    }
    result: dict[str, Any] = {
        "phase": 0,
        "status": "passed",
        "config": asdict(config),
        "contract": contract_checks,
        "architecture": {
            "task_name": TASK_NAME,
            "task_type": "single_label",
            "labels": list(CANONICAL_LABELS),
            "head": "shared two-layer MLP d->2d->1 applied to each [L]",
            "probabilities": "softmax over the two schema labels",
            "loss": "categorical cross-entropy",
            "separate_prompt_response_heads": False,
            "scope_or_view_model_feature": False,
            "future_multilabel_contract": "independent sigmoid + BCE per label",
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "autocast_dtype": str(amp_dtype),
        "train_examples_total": len(all_train),
        "train_examples_eligible": len(train_examples),
        "train_tail": train_tail,
        "valid_examples_total": len(all_valid),
        "valid_examples_eligible": len(valid_examples),
        "valid_tail": valid_tail,
        "global_steps": global_step,
        "elapsed_seconds": elapsed_seconds,
        "first_step_gradient_norms": first_step_gradient_norms,
        "marker_embedding_max_update": marker_max_update,
        "trainable_parameter_counts": parameter_counts,
        "trainable_parameters": int(sum(parameter.numel() for _, parameter in trainable_named)),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_names": grouped_names,
        "peak_vram_mb": (
            float(torch.cuda.max_memory_allocated() / 2**20)
            if device.type == "cuda"
            else 0.0
        ),
        "checkpoint": str(checkpoint_dir),
        "max_reload_probability_delta": max_reload_probability_delta,
        "history": history,
        "valid": valid_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
