from __future__ import annotations

import torch


def cuda_supports_native_bf16(device: torch.device | int | None = None) -> bool:
    """Return true only when CUDA and the GPU have native BF16 tensor support.

    Recent PyTorch builds may report ``is_bf16_supported()`` on older GPUs via
    software fallbacks. Ampere (compute capability 8.x) is the first NVIDIA
    generation with native BF16, so the architecture check is required for
    selecting a performant automatic training precision.
    """

    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major >= 8 and torch.cuda.is_bf16_supported()
