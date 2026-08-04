from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import time
import traceback
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from profile_decoder_lora_training import (
    batches,
    build_features,
    collate,
    prompt_labels,
    read_rows,
    stratified_quantile_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic BF16 LoRA trainer for the no-R decoder-guard pilots. "
            "Checkpoints are written only at completed optimizer-step boundaries."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-kind", choices=("qwen", "nemotron"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", choices=("all", "en", "vi"), default="all")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Stratified smoke subset; 0 means all filtered rows.",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=0,
        help="Stop after this many global optimizer updates; 0 means the full run.",
    )
    parser.add_argument(
        "--stress-longest-first",
        action="store_true",
        help="Place the longest microbatches first; intended for smoke/stress only.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def capture_rng() -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    return value


def restore_rng(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in value:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def checkpoint_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def complete_checkpoints(output_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in output_dir.glob("checkpoint-*")
            if path.is_dir() and (path / "complete.marker").exists()
        ),
        key=checkpoint_number,
    )


def save_checkpoint(
    *,
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: dict[str, Any],
    keep: int,
) -> Path:
    step = int(state["global_step"])
    final_path = output_dir / f"checkpoint-{step:08d}"
    temporary = output_dir / f".checkpoint-{step:08d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(temporary / "tokenizer")
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
    torch.save(capture_rng(), temporary / "rng.pt")
    atomic_json(temporary / "trainer_state.json", state)
    (temporary / "complete.marker").write_text("complete\n", encoding="utf-8")
    if final_path.exists():
        shutil.rmtree(final_path)
    os.replace(temporary, final_path)
    checkpoints = complete_checkpoints(output_dir)
    for obsolete in checkpoints[:-max(1, keep)]:
        shutil.rmtree(obsolete)
    return final_path


def load_checkpoint_state(checkpoint: Path) -> dict[str, Any]:
    return json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))


def load_model(
    *,
    model_path: Path,
    resume_checkpoint: Path | None,
) -> Any:
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    base.enable_input_require_grads()
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if resume_checkpoint is not None:
        model = PeftModel.from_pretrained(
            base,
            resume_checkpoint / "adapter",
            is_trainable=True,
        )
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    model.to("cuda")
    model.train()
    return model


def environment_report() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_vram_mb": torch.cuda.get_device_properties(0).total_memory
        / (1024 * 1024),
        "git_commit": git_commit(),
    }


def progress_payload(
    *,
    state: dict[str, Any],
    status: str,
    total_updates: int,
    started_monotonic: float,
    recent_loss: float | None,
    actual_tokens: int,
    padded_tokens: int,
) -> dict[str, Any]:
    elapsed = max(0.001, time.monotonic() - started_monotonic)
    completed_this_process = int(state["global_step"]) - int(
        state["process_initial_global_step"]
    )
    step_rate = completed_this_process / elapsed
    remaining = max(0, total_updates - int(state["global_step"]))
    return {
        "status": status,
        "updated_at_unix": time.time(),
        "model_kind": state["model_kind"],
        "global_step": int(state["global_step"]),
        "total_updates": total_updates,
        "progress_fraction": int(state["global_step"]) / max(1, total_updates),
        "epoch_index": int(state["epoch_index"]),
        "next_group_index": int(state["next_group_index"]),
        "loss": recent_loss,
        "elapsed_seconds_this_process": elapsed,
        "optimizer_steps_per_second": step_rate,
        "eta_seconds": remaining / step_rate if step_rate > 0 else None,
        "actual_tokens": actual_tokens,
        "padded_tokens": padded_tokens,
        "padding_overhead_fraction": (
            1 - actual_tokens / padded_tokens if padded_tokens else None
        ),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "reserved_vram_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")
    if args.microbatch <= 0 or args.effective_batch <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.effective_batch % args.microbatch:
        raise ValueError("effective-batch must be divisible by microbatch")
    accumulation = args.effective_batch // args.microbatch
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    event_path = output_dir / "events.jsonl"
    train_log_path = output_dir / "train_log.jsonl"
    started_monotonic = time.monotonic()

    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        rows = read_rows(args.manifest, args.language)
        selected = (
            stratified_quantile_sample(rows, args.sample_limit, args.seed)
            if args.sample_limit
            else list(rows)
        )
        counts = Counter(
            f"{row['language']}|{row['view']}|{row['safety_label']}"
            for row in selected
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        features, feature_audit = build_features(
            tokenizer,
            args.model_kind,
            selected,
            prompt_labels(rows),
            args.max_seq_length,
        )
        microbatches_per_epoch = math.ceil(len(features) / args.microbatch)
        groups_per_epoch = math.ceil(microbatches_per_epoch / accumulation)
        planned_updates = groups_per_epoch * args.epochs
        total_updates = (
            min(planned_updates, args.max_updates)
            if args.max_updates
            else planned_updates
        )
        warmup_steps = round(total_updates * args.warmup_ratio)

        checkpoints = complete_checkpoints(output_dir)
        resume_checkpoint = checkpoints[-1] if args.resume and checkpoints else None
        resumed = (
            load_checkpoint_state(resume_checkpoint)
            if resume_checkpoint is not None
            else None
        )
        state: dict[str, Any] = {
            "schema_version": 1,
            "model_kind": args.model_kind,
            "model": str(args.model.resolve()),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "language": args.language,
            "epoch_index": int(resumed["epoch_index"]) if resumed else 0,
            "next_group_index": int(resumed["next_group_index"]) if resumed else 0,
            "global_step": int(resumed["global_step"]) if resumed else 0,
            "process_initial_global_step": int(resumed["global_step"]) if resumed else 0,
            "total_updates": total_updates,
            "groups_per_epoch": groups_per_epoch,
            "config": {
                "precision": "bf16",
                "quantization": None,
                "max_seq_length": args.max_seq_length,
                "microbatch": args.microbatch,
                "effective_batch": args.effective_batch,
                "gradient_accumulation": accumulation,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
                "warmup_steps": warmup_steps,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "lora_r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "lora_targets": ["q_proj", "v_proj"],
                "seed": args.seed,
                "sample_limit": args.sample_limit,
                "max_updates": args.max_updates,
                "stress_longest_first": args.stress_longest_first,
            },
        }
        contract = {
            "status": "initializing",
            "created_at_unix": time.time(),
            "state": state,
            "environment": environment_report(),
            "dataset": {
                "rows_after_view_language_filter": len(rows),
                "selected_rows": len(selected),
                "counts": dict(sorted(counts.items())),
                "feature_audit": feature_audit,
            },
        }
        atomic_json(output_dir / "run_contract.json", contract)
        append_jsonl(
            event_path,
            {
                "event": "initializing",
                "time_unix": time.time(),
                "resume_checkpoint": str(resume_checkpoint)
                if resume_checkpoint
                else None,
                "global_step": state["global_step"],
            },
        )

        model = load_model(
            model_path=args.model,
            resume_checkpoint=resume_checkpoint,
        )
        trainable, total = model.get_nb_trainable_parameters()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )
        if resume_checkpoint is not None:
            optimizer.load_state_dict(
                torch.load(
                    resume_checkpoint / "optimizer.pt",
                    map_location="cuda",
                    weights_only=False,
                )
            )
            scheduler.load_state_dict(
                torch.load(
                    resume_checkpoint / "scheduler.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            )
            restore_rng(
                torch.load(
                    resume_checkpoint / "rng.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            )

        contract["status"] = "running"
        contract["model_profile"] = {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": trainable / total,
        }
        atomic_json(output_dir / "run_contract.json", contract)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        actual_tokens = 0
        padded_tokens = 0
        recent_loss: float | None = None

        while (
            int(state["epoch_index"]) < args.epochs
            and int(state["global_step"]) < total_updates
        ):
            epoch_index = int(state["epoch_index"])
            epoch_batches = batches(features, args.microbatch, args.seed + epoch_index)
            # batches() drops an incomplete microbatch. Keep that final subset so
            # the full manifest is represented once per epoch.
            remainder = len(features) % args.microbatch
            if remainder:
                epoch_batches.append(features[-remainder:])
            if args.stress_longest_first:
                epoch_batches.sort(
                    key=lambda batch: max(int(item["length"]) for item in batch),
                    reverse=True,
                )
            groups = [
                epoch_batches[index : index + accumulation]
                for index in range(0, len(epoch_batches), accumulation)
            ]
            group_index = int(state["next_group_index"])
            while (
                group_index < len(groups)
                and int(state["global_step"]) < total_updates
            ):
                group = groups[group_index]
                group_loss = 0.0
                group_actual = 0
                group_padded = 0
                group_samples = 0
                for batch in group:
                    input_ids, attention, targets, actual, padded = collate(
                        tokenizer,
                        batch,
                        torch.device("cuda"),
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(
                            input_ids=input_ids,
                            attention_mask=attention,
                            labels=targets,
                            use_cache=False,
                        ).loss
                        (loss / len(group)).backward()
                    group_loss += float(loss.detach())
                    group_actual += actual
                    group_padded += padded
                    group_samples += len(batch)
                grad_norm = float(
                    clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad),
                        args.max_grad_norm,
                    )
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()

                state["global_step"] = int(state["global_step"]) + 1
                group_index += 1
                state["next_group_index"] = group_index
                actual_tokens += group_actual
                padded_tokens += group_padded
                recent_loss = group_loss / len(group)
                log_value = {
                    "time_unix": time.time(),
                    "epoch_index": epoch_index,
                    "group_index": group_index - 1,
                    "global_step": int(state["global_step"]),
                    "total_updates": total_updates,
                    "loss": recent_loss,
                    "grad_norm": grad_norm,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "samples": group_samples,
                    "actual_tokens": group_actual,
                    "padded_tokens": group_padded,
                    "max_memory_allocated_mb": torch.cuda.max_memory_allocated()
                    / (1024 * 1024),
                }
                append_jsonl(train_log_path, log_value)
                progress = progress_payload(
                    state=state,
                    status="running",
                    total_updates=total_updates,
                    started_monotonic=started_monotonic,
                    recent_loss=recent_loss,
                    actual_tokens=actual_tokens,
                    padded_tokens=padded_tokens,
                )
                atomic_json(progress_path, progress)
                print(json.dumps(progress, ensure_ascii=False), flush=True)

                if (
                    int(state["global_step"]) % args.save_every == 0
                    or int(state["global_step"]) == total_updates
                ):
                    saved = save_checkpoint(
                        output_dir=output_dir,
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        state=state,
                        keep=args.keep_checkpoints,
                    )
                    append_jsonl(
                        event_path,
                        {
                            "event": "checkpoint_saved",
                            "time_unix": time.time(),
                            "path": str(saved),
                            "global_step": int(state["global_step"]),
                        },
                    )

            state["epoch_index"] = epoch_index + 1
            state["next_group_index"] = 0

        final_adapter = output_dir / "final_adapter"
        if final_adapter.exists():
            shutil.rmtree(final_adapter)
        model.save_pretrained(final_adapter, safe_serialization=True)
        tokenizer.save_pretrained(final_adapter)
        final_progress = progress_payload(
            state=state,
            status="completed",
            total_updates=total_updates,
            started_monotonic=started_monotonic,
            recent_loss=recent_loss,
            actual_tokens=actual_tokens,
            padded_tokens=padded_tokens,
        )
        atomic_json(progress_path, final_progress)
        contract["status"] = "completed"
        contract["completed_at_unix"] = time.time()
        contract["final_state"] = state
        contract["runtime"] = final_progress
        atomic_json(output_dir / "run_contract.json", contract)
        (output_dir / "completed.marker").write_text("completed\n", encoding="utf-8")
        append_jsonl(
            event_path,
            {
                "event": "completed",
                "time_unix": time.time(),
                "global_step": int(state["global_step"]),
            },
        )
    except BaseException as exc:
        failure = {
            "status": "failed",
            "time_unix": time.time(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(output_dir / "failure.json", failure)
        append_jsonl(output_dir / "events.jsonl", {"event": "failed", **failure})
        atomic_json(output_dir / "progress.json", failure)
        raise


if __name__ == "__main__":
    main()
