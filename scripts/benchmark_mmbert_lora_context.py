"""Microbenchmark mmBERT's one fixed binary head with encoder LoRA.

This is intentionally separate from ``benchmark_mmbert_frozen_context.py``:
the frozen benchmark executes the encoder under ``torch.no_grad()`` and is not
an estimate of E2/E3/E4/E5 once those experiments use LoRA.
"""

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
from peft import LoraConfig, get_peft_model

from guard_train.precision import cuda_supports_native_bf16

from guard_smoke.fixed_head import FixedHeadGuard


def parse_lengths(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_model(
    model_path: Path,
    *,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    gradient_checkpointing: bool,
) -> FixedHeadGuard:
    # freeze_encoder=False is essential: _encode() must not enter no_grad().
    model = FixedHeadGuard(str(model_path), freeze_encoder=False)
    if gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
        model.encoder.enable_input_require_grads()
    model.encoder = get_peft_model(
        model.encoder,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            # PEFT expands this to attention and MLP projections. GLiGuard's
            # current adapter also covers attention and feed-forward linears.
            target_modules="all-linear",
        ),
    )
    return model


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
    input_ids = torch.full((batch_size, length), 5, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    targets = torch.arange(batch_size, device=device) % 2
    elapsed: list[float] = []
    status = "success"
    error = None
    peak = 0.0
    lora_parameters_with_grad = 0
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
            current_lora_gradients = sum(
                1
                for name, parameter in model.named_parameters()
                if "lora_" in name and parameter.requires_grad and parameter.grad is not None
            )
            if current_lora_gradients == 0:
                raise RuntimeError(
                    "No LoRA parameter received a gradient; the benchmark would only measure the head"
                )
            lora_parameters_with_grad = max(
                lora_parameters_with_grad, current_lora_gradients
            )
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
        "lora_parameters_with_grad": lora_parameters_with_grad,
        "repeats_measured": len(elapsed),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/mmbert_small_base_smoke")
    )
    parser.add_argument("--lengths", default="512")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/mmbert_lora_context_benchmark.json")
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and cuda_supports_native_bf16(device)
        else torch.float16
    )
    load_started = time.perf_counter()
    model = build_model(
        args.model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
    load_seconds = time.perf_counter() - load_started
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)

    rows: list[dict[str, Any]] = []
    for length in parse_lengths(args.lengths):
        print(f"benchmark LoRA length={length} batch={args.batch_size}", flush=True)
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
        "recipe": "one fixed binary text-safety head plus encoder LoRA",
        "freeze_encoder": False,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": "all-linear",
        },
        "gradient_checkpointing": args.gradient_checkpointing,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "load_seconds": load_seconds,
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
            "Synthetic fixed-length benchmark. It measures the correct LoRA "
            "backward path but not tokenizer/schema preprocessing or evaluation."
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
