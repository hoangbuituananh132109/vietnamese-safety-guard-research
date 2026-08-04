"""Strict Phase-0 single-label loss override for the local GLiNER2 package.

The installed GLiNER2 classification path uses independent BCE logits for every
classification task.  GLiGuard's paper specifies categorical cross-entropy for
single-label tasks.  Phase 0 has exactly one classification task and exactly one
true label, so this module installs a narrow instance-level override without
editing site-packages or changing span/structure training globally.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F


def _phase0_single_label_loss(
    self: Any,
    token_embeddings: torch.Tensor,
    embs_per_schema: list[list[torch.Tensor]],
    task_types: list[str],
    structure_labels: list[Any],
    device: torch.device,
    span_info: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    del span_info
    if len(embs_per_schema) != len(task_types) or len(task_types) != len(structure_labels):
        raise ValueError("GLiNER2 schema embeddings/task types/targets are misaligned")
    if any(task_type != "classifications" for task_type in task_types):
        raise ValueError(
            "Phase-0 CE override supports classification-only batches; "
            "span/entity/relation tasks must use the original GLiNER2 loss"
        )

    classification_loss = torch.zeros((), dtype=torch.float32, device=device)
    for index, schema_embeddings in enumerate(embs_per_schema):
        if not schema_embeddings:
            raise ValueError(f"Missing schema embeddings for classification task {index}")
        schema = torch.stack(schema_embeddings)
        if schema.shape[0] < 2:
            raise ValueError("Classification schema must contain [P] plus at least one [L]")
        label_embeddings = schema[1:]  # GLiNER2 orders schema anchors as [P], [L]...
        logits = self.classifier(label_embeddings).squeeze(-1).float()
        binary_targets = torch.as_tensor(
            structure_labels[index], dtype=torch.float32, device=device
        )
        if binary_targets.ndim != 1 or binary_targets.numel() != logits.numel():
            raise ValueError(
                f"Target/logit mismatch for task {index}: "
                f"targets={tuple(binary_targets.shape)}, logits={tuple(logits.shape)}"
            )
        true_positions = torch.nonzero(binary_targets > 0.5, as_tuple=False).flatten()
        if true_positions.numel() != 1:
            raise ValueError(
                "Phase-0 single-label task requires exactly one true label; "
                f"task {index} has {int(true_positions.numel())}"
            )
        target_index = true_positions[0].to(dtype=torch.long).view(1)
        classification_loss = classification_loss + F.cross_entropy(
            logits.view(1, -1), target_index, reduction="sum"
        )

    # Keep the same return contract expected by GLiNER2Trainer.
    zero = token_embeddings.sum().float() * 0.0
    return {
        "classification": classification_loss,
        "structure": zero,
        "count": zero,
    }


def install_phase0_single_label_ce(model: Any) -> Any:
    """Install the strict CE path on one GLiNER2 model instance and return it."""

    if getattr(model, "_phase0_single_label_ce_installed", False):
        return model
    if not hasattr(model, "_compute_sample_loss") or not hasattr(model, "classifier"):
        raise TypeError("Expected a GLiNER2-like model with classifier and _compute_sample_loss")
    model._phase0_original_compute_sample_loss = model._compute_sample_loss
    model._compute_sample_loss = MethodType(_phase0_single_label_loss, model)
    model._phase0_single_label_ce_installed = True
    model._phase0_loss_contract = {
        "task_type": "single_label",
        "activation": "softmax",
        "loss": "categorical_cross_entropy",
        "package_default_replaced": "binary_cross_entropy_with_logits",
    }
    return model
