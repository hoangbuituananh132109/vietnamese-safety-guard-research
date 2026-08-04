from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guard_smoke.constants import N23_CATEGORIES
from scripts.evaluate_nemotron_decoder_guard import prepare_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budget-aware QLoRA SFT for D2 Nemotron Guard.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-wall-seconds", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def target_payload(
    row: dict[str, Any], prompt_labels: dict[tuple[str, str], str]
) -> dict[str, str]:
    view = str(row["view"])
    response_label = str(row["safety_label"])
    payload: dict[str, str] = {}
    if view == "P":
        payload["User Safety"] = response_label
    elif view == "R":
        payload["User Safety"] = "safe"
        payload["Response Safety"] = response_label
    elif view == "PR":
        key = (str(row["normalized_prompt_sha256"]), str(row["language"]))
        if key not in prompt_labels:
            raise KeyError(f"Missing joined prompt label for {key}")
        payload["User Safety"] = prompt_labels[key]
        payload["Response Safety"] = response_label
    else:
        raise ValueError(f"Unsupported view: {view}")

    categories = set(str(value) for value in row.get("categories") or ())
    ordered = [category for category in N23_CATEGORIES if category in categories]
    if ordered and str(row.get("category_scope")) in {"prompt", "interaction"}:
        payload["Safety Categories"] = ", ".join(ordered)
    return payload


def build_features(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    prompt_labels: dict[tuple[str, str], str],
    max_seq_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if not isinstance(eot_id, int) or eot_id < 0:
        eot_id = tokenizer.eos_token_id
    features: list[dict[str, Any]] = []
    truncated = 0
    input_tokens = 0
    supervised_tokens = 0
    for row in rows:
        target = json.dumps(target_payload(row, prompt_labels), ensure_ascii=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False) + [int(eot_id)]
        prompt_budget = max_seq_length - len(target_ids)
        if prompt_budget < 256:
            raise ValueError(f"Target leaves too little prompt budget: {len(target_ids)} tokens")
        prompt, _, was_truncated, _ = prepare_prompt(tokenizer, row, prompt_budget)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = prompt_ids + target_ids
        if len(input_ids) > max_seq_length:
            raise AssertionError(f"Feature length {len(input_ids)} exceeds {max_seq_length}")
        labels = [-100] * len(prompt_ids) + target_ids
        features.append(
            {
                "example_id": str(row["example_id"]),
                "input_ids": input_ids,
                "labels": labels,
                "length": len(input_ids),
                "supervised": len(target_ids),
            }
        )
        truncated += int(was_truncated)
        input_tokens += len(input_ids)
        supervised_tokens += len(target_ids)
    return features, {
        "examples": len(features),
        "input_tokens": input_tokens,
        "supervised_tokens": supervised_tokens,
        "truncated": truncated,
        "truncated_fraction": truncated / max(1, len(features)),
        "min_length": min(item["length"] for item in features),
        "max_length": max(item["length"] for item in features),
        "mean_length": input_tokens / max(1, len(features)),
    }


def save_adapter(model: Any, tokenizer: Any, directory: Path, state: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory / "tokenizer")
    atomic_json(directory / "trainer_state.json", state)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "events.jsonl"
    progress_path = args.output_dir / "progress.json"
    train_log_path = args.output_dir / "train_log.jsonl"
    tensorboard_dir = args.output_dir / "tensorboard"

    source_rows = read_jsonl(args.source)
    if any(str(row.get("source_split")) != "train" for row in source_rows):
        raise ValueError("Training source contains a non-train split")
    prompt_label_sets: dict[tuple[str, str], set[str]] = {}
    for row in source_rows:
        if str(row["view"]) != "P":
            continue
        key = (str(row["normalized_prompt_sha256"]), str(row["language"]))
        prompt_label_sets.setdefault(key, set()).add(str(row["safety_label"]))
    conflicting = {key: labels for key, labels in prompt_label_sets.items() if len(labels) != 1}
    if conflicting:
        raise ValueError(f"Conflicting prompt labels for {len(conflicting)} normalized hashes")
    prompt_labels = {key: next(iter(labels)) for key, labels in prompt_label_sets.items()}
    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if any(str(row.get("source_split")) != "train" for row in rows):
        raise ValueError("Training manifest contains a non-train split")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    features, feature_audit = build_features(
        tokenizer, rows, prompt_labels, args.max_seq_length
    )
    atomic_json(args.output_dir / "feature_audit.json", feature_audit)
    append_jsonl(events_path, {"event": "features_ready", **feature_audit, "time": time.time()})

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()
    trainable, total = model.get_nb_trainable_parameters()
    model_load_seconds = time.monotonic() - load_started
    append_jsonl(
        events_path,
        {
            "event": "model_ready",
            "time": time.time(),
            "model_load_seconds": model_load_seconds,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": trainable / total,
        },
    )

    micro_steps_per_epoch = len(features)
    planned_updates = math.ceil(micro_steps_per_epoch * args.epochs / args.gradient_accumulation)
    if args.max_steps > 0:
        planned_updates = min(planned_updates, args.max_steps)
    warmup_steps = max(1, round(planned_updates * args.warmup_ratio))
    optimizer = bnb.optim.PagedAdamW8bit(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=planned_updates
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=512.0, growth_interval=1000000)
    writer = SummaryWriter(str(tensorboard_dir))

    rng = random.Random(args.seed)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    update_step = 0
    micro_step = 0
    samples_seen = 0
    tokens_seen = 0
    supervised_seen = 0
    accumulated_loss = 0.0
    stopped_by_wall = False

    def current_state(status: str) -> dict[str, Any]:
        elapsed = time.monotonic() - started
        rate = tokens_seen / max(elapsed, 1e-9)
        remaining_updates = max(0, planned_updates - update_step)
        eta = elapsed / max(update_step, 1) * remaining_updates if update_step else None
        return {
            "status": status,
            "updated_at": time.time(),
            "optimizer_step": update_step,
            "planned_optimizer_steps": planned_updates,
            "micro_step": micro_step,
            "samples_seen": samples_seen,
            "input_tokens_seen": tokens_seen,
            "supervised_tokens_seen": supervised_seen,
            "elapsed_seconds": elapsed,
            "input_tokens_per_second": rate,
            "eta_seconds": eta,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            "stopped_by_wall": stopped_by_wall,
            "feature_audit": feature_audit,
        }

    try:
        for epoch in range(args.epochs):
            indices = list(range(len(features)))
            rng.shuffle(indices)
            for index in indices:
                if update_step >= planned_updates:
                    break
                item = features[index]
                input_ids = torch.tensor(item["input_ids"], dtype=torch.long, device="cuda").unsqueeze(0)
                labels = torch.tensor(item["labels"], dtype=torch.long, device="cuda").unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    ).loss
                    scaled_loss = loss / args.gradient_accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at micro step {micro_step}")
                scaler.scale(scaled_loss).backward()
                micro_step += 1
                samples_seen += 1
                tokens_seen += int(item["length"])
                supervised_seen += int(item["supervised"])
                accumulated_loss += float(loss.detach().cpu())

                should_update = micro_step % args.gradient_accumulation == 0
                is_last = samples_seen >= len(features) * args.epochs
                if should_update or is_last:
                    scaler.unscale_(optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    update_step += 1
                    mean_loss = accumulated_loss / (
                        args.gradient_accumulation if should_update else micro_step % args.gradient_accumulation
                    )
                    accumulated_loss = 0.0
                    state = current_state("training")
                    log = {
                        **state,
                        "epoch": epoch,
                        "loss": mean_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm,
                        "scale": scaler.get_scale(),
                    }
                    append_jsonl(train_log_path, log)
                    atomic_json(progress_path, log)
                    writer.add_scalar("train/loss", mean_loss, update_step)
                    writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], update_step)
                    writer.add_scalar("train/tokens_per_second", state["input_tokens_per_second"], update_step)
                    writer.add_scalar("train/grad_norm", grad_norm, update_step)
                    writer.flush()
                    print(json.dumps(log, ensure_ascii=False), flush=True)

                    if args.save_every > 0 and update_step % args.save_every == 0:
                        save_adapter(
                            model,
                            tokenizer,
                            args.output_dir / "checkpoints" / f"step-{update_step:06d}",
                            current_state("checkpoint"),
                        )
                    if args.max_wall_seconds > 0 and time.monotonic() - started >= args.max_wall_seconds:
                        stopped_by_wall = True
                        break
            if update_step >= planned_updates or stopped_by_wall:
                break
        final_state = current_state("completed")
        save_adapter(model, tokenizer, args.output_dir / "final_adapter", final_state)
        atomic_json(progress_path, final_state)
        append_jsonl(events_path, {"event": "completed", **final_state})
    except BaseException as error:
        emergency_state = current_state("failed")
        emergency_state["error"] = repr(error)
        atomic_json(progress_path, emergency_state)
        append_jsonl(events_path, {"event": "failed", **emergency_state})
        try:
            save_adapter(model, tokenizer, args.output_dir / "emergency_adapter", emergency_state)
        except BaseException as save_error:
            append_jsonl(events_path, {"event": "emergency_save_failed", "error": repr(save_error)})
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
