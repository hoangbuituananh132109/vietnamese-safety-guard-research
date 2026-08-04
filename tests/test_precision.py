from __future__ import annotations

import torch

from guard_train.precision import cuda_supports_native_bf16


def test_turing_does_not_select_bf16_even_if_torch_reports_supported(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (7, 5))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert cuda_supports_native_bf16(torch.device("cuda")) is False


def test_ampere_selects_bf16_when_torch_supports_it(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (8, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert cuda_supports_native_bf16(torch.device("cuda")) is True
