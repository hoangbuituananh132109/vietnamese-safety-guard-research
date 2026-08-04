"""Synthetic backward profiler for E7's dynamic-schema mmBERT path."""

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
from transformers import AutoModel

from guard_train.mmbert_schema import DynamicSchemaGuard, load_schema_tokenizer
from guard_train.precision import cuda_supports_native_bf16


def parse_lengths(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_case(
    model: DynamicSchemaGuard,
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
    binary_positions = torch.tensor([[2, 5]] * batch_size, device=device)
    binary_targets = torch.arange(batch_size, device=device) % 2
    category_positions = torch.tensor(
        [[8 + index for index in range(23)]] * batch_size, device=device
    )
    category_targets = torch.zeros((batch_size, 23), device=device)
    category_targets[:, :2] = 1
    category_mask = torch.ones((batch_size, 23), dtype=torch.bool, device=device)
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
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    binary_positions=binary_positions,
                    binary_targets=binary_targets,
                    category_positions=category_positions,
                    category_targets=category_targets,
                    category_mask=category_mask,
                )
            output["loss"].backward()
            current_lora_gradients = sum(
                1
                for name, parameter in model.named_parameters()
                if "lora_" in name and parameter.requires_grad and parameter.grad is not None
            )
            if current_lora_gradients == 0:
                raise RuntimeError(
                    "No LoRA parameter received a gradient; the benchmark would only measure the schema head"
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
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc)
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "length": length,
        "batch_size": batch_size,
        "status": status,
        "mean_step_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
        "samples_per_second": batch_size * len(elapsed) / sum(elapsed) if elapsed else None,
        "peak_allocated_vram_mb": peak,
        "lora_parameters_with_grad": lora_parameters_with_grad,
        "repeats_measured": len(elapsed),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/mmbert_small_base_smoke"))
    parser.add_argument("--lengths", default="512,1024,2048,4096,8192")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/mmbert_schema_context_benchmark.json")
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and cuda_supports_native_bf16(device)
        else torch.float16
    )
    tokenizer = load_schema_tokenizer(str(args.model_path))
    base = AutoModel.from_pretrained(str(args.model_path), local_files_only=True)
    base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    marker_ids = [tokenizer.convert_tokens_to_ids(token) for token in ("[P]", "[L]", "[SEP]")]
    encoder = get_peft_model(
        base,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules="all-linear",
            trainable_token_indices=marker_ids,
        ),
    )
    model = DynamicSchemaGuard(encoder, int(base.config.hidden_size), 1.0).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4
    )
    rows: list[dict[str, Any]] = []
    for length in parse_lengths(args.lengths):
        print(f"benchmark schema length={length} batch={args.batch_size}", flush=True)
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
        "recipe": "E7 shared [L] MLP, binary CE plus full 23-label BCE, encoder LoRA",
        "gradient_checkpointing": args.gradient_checkpointing,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_total_vram_mb": (
            torch.cuda.get_device_properties(0).total_memory / 2**20 if device.type == "cuda" else 0.0
        ),
        "autocast_dtype": str(dtype),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
