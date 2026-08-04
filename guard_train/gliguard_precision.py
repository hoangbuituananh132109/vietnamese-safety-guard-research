from __future__ import annotations

from typing import Any

from torch.cuda.amp import GradScaler


class AuditedGradScaler(GradScaler):
    """GLiNER2-compatible FP16 scaler with conservative Turing defaults.

    The pinned GLiNER2 trainer advances its scheduler/global step even when
    GradScaler skips an optimizer update.  Counting scale reductions lets the
    experiment launcher reject such a run instead of silently accepting it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("init_scale", 512.0)
        kwargs.setdefault("growth_interval", 1_000_000)
        super().__init__(*args, **kwargs)
        self.skipped_optimizer_updates = 0

    def update(self, new_scale: Any = None) -> None:
        scale_before = float(self.get_scale())
        super().update(new_scale)
        if float(self.get_scale()) < scale_before:
            self.skipped_optimizer_updates += 1

    def audit(self) -> dict[str, float | int | bool]:
        return {
            "enabled": bool(self.is_enabled()),
            "initial_scale": 512.0,
            "growth_interval": 1_000_000,
            "final_scale": float(self.get_scale()),
            "skipped_optimizer_updates": int(self.skipped_optimizer_updates),
        }
