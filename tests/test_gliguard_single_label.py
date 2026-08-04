from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from guard_smoke.gliguard_single_label import install_phase0_single_label_ce


class DummyGLi(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([[0.75, -0.25]]))
        self._compute_sample_loss = lambda *args, **kwargs: None


def test_phase0_override_matches_categorical_cross_entropy():
    model = install_phase0_single_label_ce(DummyGLi())
    schema = [
        torch.tensor([0.0, 0.0]),  # [P]
        torch.tensor([1.0, 0.0]),  # first [L]
        torch.tensor([0.0, 1.0]),  # second [L]
    ]
    losses = model._compute_sample_loss(
        token_embeddings=torch.tensor([[1.0, 2.0]]),
        embs_per_schema=[schema],
        task_types=["classifications"],
        structure_labels=[[0, 1]],
        device=torch.device("cpu"),
    )
    logits = model.classifier(torch.stack(schema)[1:]).squeeze(-1).view(1, -1)
    expected = F.cross_entropy(logits, torch.tensor([1]))
    assert torch.allclose(losses["classification"], expected)
    assert losses["structure"].item() == 0.0
    assert losses["count"].item() == 0.0
    assert model._phase0_loss_contract["loss"] == "categorical_cross_entropy"


def test_phase0_override_rejects_zero_or_multiple_true_labels():
    model = install_phase0_single_label_ce(DummyGLi())
    schema = [torch.zeros(2), torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    for invalid in ([0, 0], [1, 1]):
        with pytest.raises(ValueError, match="exactly one true label"):
            model._compute_sample_loss(
                token_embeddings=torch.ones(1, 2),
                embs_per_schema=[schema],
                task_types=["classifications"],
                structure_labels=[invalid],
                device=torch.device("cpu"),
            )


def test_phase0_override_rejects_non_classification_tasks():
    model = install_phase0_single_label_ce(DummyGLi())
    with pytest.raises(ValueError, match="classification-only"):
        model._compute_sample_loss(
            token_embeddings=torch.ones(1, 2),
            embs_per_schema=[[torch.zeros(2), torch.ones(2)]],
            task_types=["entities"],
            structure_labels=[[1, []]],
            device=torch.device("cpu"),
        )
