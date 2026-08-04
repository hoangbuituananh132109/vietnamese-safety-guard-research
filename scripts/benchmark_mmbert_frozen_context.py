"""Microbenchmark mmBERT frozen-encoder/head training across context lengths."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import torch

from guard_smoke.fixed_head import FixedHeadGuard
from guard_train.precision import cuda_supports_native_bf16


def parse_lengths(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_case(
    model: FixedHeadGuard,
    optimizer: torch.optim.Optimizer,
    *,
    length: int,
    batch_size: int,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    input_ids = torch.full(
        (batch_size, length), 5, dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids)
    targets = torch.arange(batch_size, device=device) % 2
    elapsed: list[float] = []
    status = "success"
    error = None
    try:
        for repeat in range(repeats + 1):
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda",
            ):
                result = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    safety_targets=targets,
                )
            result["loss"].backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            if repeat:
                elapsed.append(duration)
        peak = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc)
        peak = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "length": length,
        "batch_size": batch_size,
        "status": status,
        "mean_step_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
        "samples_per_second": (
            batch_size * len(elapsed) / sum(elapsed) if elapsed else None
        ),
        "peak_allocated_vram_mb": peak,
        "repeats_measured": len(elapsed),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/mmbert_small_base_smoke")
    )
    parser.add_argument("--lengths", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/mmbert_frozen_context_benchmark.json")
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and cuda_supports_native_bf16(device)
        else torch.float16
    )
    model = FixedHeadGuard(str(args.model_path), freeze_encoder=True).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=5e-4,
    )
    rows: list[dict[str, Any]] = []
    for length in parse_lengths(args.lengths):
        print(f"benchmark length={length} batch={args.batch_size}", flush=True)
        row = run_case(
            model,
            optimizer,
            length=length,
            batch_size=args.batch_size,
            repeats=args.repeats,
            device=device,
            dtype=dtype,
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if row["status"] == "oom":
            break
    report = {
        "model_path": str(args.model_path),
        "freeze_encoder": True,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_total_vram_mb": (
            torch.cuda.get_device_properties(0).total_memory / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "torch": torch.__version__,
        "autocast_dtype": str(dtype),
        "warning": (
            "Frozen encoder uses torch.no_grad(); this benchmark does not estimate "
            "LoRA, partial-unfreeze, or full-finetuning memory."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
