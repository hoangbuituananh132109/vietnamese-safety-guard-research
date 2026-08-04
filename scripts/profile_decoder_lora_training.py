from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


N23 = (
    ("S1", "Violence"),
    ("S2", "Sexual"),
    ("S3", "Criminal Planning/Confessions"),
    ("S4", "Guns and Illegal Weapons"),
    ("S5", "Controlled/Regulated Substances"),
    ("S6", "Suicide and Self Harm"),
    ("S7", "Sexual (minor)"),
    ("S8", "Hate/Identity Hate"),
    ("S9", "PII/Privacy"),
    ("S10", "Harassment"),
    ("S11", "Threat"),
    ("S12", "Profanity"),
    ("S13", "Needs Caution"),
    ("S14", "Other"),
    ("S15", "Manipulation"),
    ("S16", "Fraud/Deception"),
    ("S17", "Malware"),
    ("S18", "High Risk Gov Decision Making"),
    ("S19", "Political/Misinformation/Conspiracy"),
    ("S20", "Copyright/Trademark/Plagiarism"),
    ("S21", "Unauthorized Advice"),
    ("S22", "Illegal Activity"),
    ("S23", "Immoral/Unethical"),
)
N23_NAMES = tuple(name for _, name in N23)
TAXONOMY = "\n".join(f"{code}: {name}." for code, name in N23)
NEMOTRON_PROMPT = """Task: Check if there is unsafe content in the user or agent messages in the below conversation in accordance with our safety policy with the below categories. Provide a 'safe' or 'unsafe' rating for both the user message and the agent response. If there is unsafe content, please also provide a list of violated categories according to our safety policy below.

<BEGIN UNSAFE CONTENT CATEGORIES>
{taxonomy}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

user: {query}
{response_block}<END CONVERSATION>

Provide your safety assessment for the above conversation in the below output JSON format:
{{"User Safety": safe or unsafe rating for the user message, "Response Safety": safe or unsafe rating for the agent response. Omit if no agent response present. "Safety Categories": a comma-separated list of applicable safety categories from the provided taxonomy. Omit if all safe.}}

Do not include anything other than the output JSON in your response.
Output JSON: """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short, budget-bounded BF16 LoRA throughput profile for decoder guards."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-kind", choices=("nemotron", "qwen"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("all", "en", "vi"), default="all")
    parser.add_argument("--sample-limit", type=int, default=2048)
    parser.add_argument(
        "--projection-rows",
        type=int,
        default=None,
        help="Full-run row count used for ETA; defaults to rows in the manifest.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--optimizer-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--synthetic-weights",
        action="store_true",
        help=(
            "Instantiate the exact architecture in BF16 without checkpoint "
            "weights. Valid for compute/VRAM profiling only, never quality."
        ),
    )
    parser.add_argument(
        "--stress-longest",
        action="store_true",
        help="Use the longest available length buckets first for VRAM stress.",
    )
    return parser.parse_args()


def read_rows(path: Path, language: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("view")) not in {"P", "PR"}:
                continue
            if language != "all" and str(row.get("language")) != language:
                continue
            rows.append(row)
    if not rows:
        raise ValueError("No P/PR rows survived the requested filters")
    return rows


def stratified_quantile_sample(
    rows: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("language")),
            str(row.get("view")),
            str(row.get("safety_label")),
        )
        groups[key].append(row)

    allocation: dict[tuple[str, str, str], int] = {}
    remaining = limit
    ordered_keys = sorted(groups)
    for index, key in enumerate(ordered_keys):
        if index == len(ordered_keys) - 1:
            take = remaining
        else:
            take = max(1, round(limit * len(groups[key]) / len(rows)))
            take = min(take, remaining - (len(ordered_keys) - index - 1))
        allocation[key] = min(take, len(groups[key]))
        remaining -= allocation[key]

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for key in ordered_keys:
        values = groups[key]
        rng.shuffle(values)
        values.sort(key=lambda row: len(str(row.get("text") or "")))
        count = allocation[key]
        if count >= len(values):
            sampled.extend(values)
            continue
        # Evenly spaced order statistics preserve the entire length distribution,
        # including the rare long tail, without making the profile random-length.
        indices = {
            min(len(values) - 1, round((position + 0.5) * len(values) / count - 0.5))
            for position in range(count)
        }
        if len(indices) < count:
            for candidate in range(len(values)):
                indices.add(candidate)
                if len(indices) == count:
                    break
        sampled.extend(values[index] for index in sorted(indices))
    rng.shuffle(sampled)
    return sampled[:limit]


def strip_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :].lstrip() if text.startswith(prefix) else text


def conversation(row: dict[str, Any]) -> tuple[str, str | None]:
    text = str(row["text"])
    if str(row["view"]) == "P":
        return strip_prefix(text, "Prompt:"), None
    marker = "\nResponse:"
    if marker not in text:
        raise ValueError(f"PR row cannot be split: {row.get('example_id')}")
    prompt, response = text.split(marker, 1)
    return strip_prefix(prompt, "Prompt:"), response.lstrip()


def prompt_labels(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    values: dict[tuple[str, str], str] = {}
    for row in rows:
        if str(row["view"]) != "P":
            continue
        key = (str(row["normalized_prompt_sha256"]), str(row["language"]))
        label = str(row["safety_label"])
        previous = values.setdefault(key, label)
        if previous != label:
            raise ValueError(f"Conflicting prompt labels for {key}")
    return values


def render_nemotron(tokenizer: Any, query: str, response: str | None) -> str:
    instruction = NEMOTRON_PROMPT.format(
        taxonomy=TAXONOMY,
        query=query,
        response_block=f"response: agent: {response}\n" if response is not None else "",
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
        tokenize=False,
    )


def render_qwen(tokenizer: Any, query: str, response: str | None) -> str:
    messages = [{"role": "user", "content": query}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return tokenizer.apply_chat_template(messages, tokenize=False)


def nemotron_target(
    row: dict[str, Any],
    labels: dict[tuple[str, str], str],
    allow_missing_prompt_label: bool = False,
) -> str:
    view = str(row["view"])
    safety = str(row["safety_label"])
    if view == "P":
        payload: dict[str, str] = {"User Safety": safety}
    else:
        key = (str(row["normalized_prompt_sha256"]), str(row["language"]))
        prompt_label = labels.get(key)
        if prompt_label is None:
            if not allow_missing_prompt_label:
                raise KeyError(key)
            # Compute-only manifests may omit the paired P row. Safe and unsafe
            # are both one-token ratings here, so this preserves target FLOPs.
            prompt_label = "safe"
        payload = {
            "User Safety": prompt_label,
            "Response Safety": safety,
        }
    categories = set(str(value) for value in row.get("categories") or ())
    ordered = [name for name in N23_NAMES if name in categories]
    if ordered and str(row.get("category_scope")) in {"prompt", "interaction"}:
        payload["Safety Categories"] = ", ".join(ordered)
    return json.dumps(payload, ensure_ascii=False)


def target_terminator(tokenizer: Any, model_kind: str) -> int:
    if model_kind == "nemotron":
        value = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(value, int) and value >= 0:
            return value
    if isinstance(tokenizer.eos_token_id, int):
        return tokenizer.eos_token_id
    raise ValueError("Tokenizer has no usable target terminator")


def build_features(
    tokenizer: Any,
    model_kind: str,
    sampled: list[dict[str, Any]],
    labels: dict[tuple[str, str], str],
    max_length: int,
    allow_missing_prompt_labels: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terminator = target_terminator(tokenizer, model_kind)
    features: list[dict[str, Any]] = []
    untruncated_lengths: list[int] = []
    truncated = 0
    supervised = 0
    missing_prompt_labels = 0
    for row in sampled:
        query, response = conversation(row)
        if model_kind == "nemotron":
            prompt = render_nemotron(tokenizer, query, response)
            if str(row["view"]) == "PR":
                key = (
                    str(row["normalized_prompt_sha256"]),
                    str(row["language"]),
                )
                missing_prompt_labels += int(key not in labels)
            target = nemotron_target(
                row,
                labels,
                allow_missing_prompt_label=allow_missing_prompt_labels,
            )
        else:
            prompt = render_qwen(tokenizer, query, response)
            label = "Unsafe" if str(row["safety_label"]) == "unsafe" else "Safe"
            target = f"Safety: {label}"
        target_ids = tokenizer.encode(target, add_special_tokens=False) + [terminator]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        untruncated_lengths.append(len(prompt_ids) + len(target_ids))
        budget = max_length - len(target_ids)
        if budget < 64:
            raise ValueError(f"Target consumes too much of max length: {len(target_ids)}")
        if len(prompt_ids) > budget:
            # This is a compute profiler. Head+tail keeps the planned sequence
            # length distribution without claiming semantic truncation quality.
            head = budget // 2
            prompt_ids = prompt_ids[:head] + prompt_ids[-(budget - head) :]
            truncated += 1
        input_ids = prompt_ids + target_ids
        features.append(
            {
                "input_ids": input_ids,
                "labels": [-100] * len(prompt_ids) + target_ids,
                "length": len(input_ids),
            }
        )
        supervised += len(target_ids)
    features.sort(key=lambda item: int(item["length"]))
    lengths = sorted(int(item["length"]) for item in features)
    return features, {
        "sample_rows": len(features),
        "mean_tokens": sum(lengths) / len(lengths),
        "mean_untruncated_tokens": sum(untruncated_lengths) / len(untruncated_lengths),
        "min_tokens": lengths[0],
        "p50_tokens": lengths[round((len(lengths) - 1) * 0.50)],
        "p95_tokens": lengths[round((len(lengths) - 1) * 0.95)],
        "p99_tokens": lengths[round((len(lengths) - 1) * 0.99)],
        "max_tokens": lengths[-1],
        "truncated": truncated,
        "truncated_fraction": truncated / len(features),
        "mean_supervised_tokens": supervised / len(features),
        "missing_prompt_label_fallbacks": missing_prompt_labels,
    }


def batches(
    features: list[dict[str, Any]], microbatch: int, seed: int
) -> list[list[dict[str, Any]]]:
    # Shuffle batch order while keeping similarly sized examples together.
    values = [
        features[index : index + microbatch]
        for index in range(0, len(features), microbatch)
        if len(features[index : index + microbatch]) == microbatch
    ]
    random.Random(seed).shuffle(values)
    return values


def collate(
    tokenizer: Any, batch: list[dict[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
    input_ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    targets = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention = input_ids.ne(tokenizer.pad_token_id)
    actual = sum(len(item["input_ids"]) for item in batch)
    padded = int(input_ids.numel())
    return (
        input_ids.to(device),
        attention.to(device),
        targets.to(device),
        actual,
        padded,
    )


def main() -> None:
    args = parse_args()
    if args.effective_batch % args.microbatch:
        raise ValueError("effective-batch must be divisible by microbatch")
    accumulation = args.effective_batch // args.microbatch
    if args.optimizer_steps <= args.warmup_steps:
        raise ValueError("optimizer-steps must exceed warmup-steps")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = read_rows(args.manifest, args.language)
    counts = Counter(
        f"{row['language']}|{row['view']}|{row['safety_label']}" for row in rows
    )
    sampled = stratified_quantile_sample(rows, args.sample_limit, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    labels = prompt_labels(rows)
    features, feature_audit = build_features(
        tokenizer,
        args.model_kind,
        sampled,
        labels,
        args.max_seq_length,
        allow_missing_prompt_labels=args.synthetic_weights,
    )
    train_batches = batches(features, args.microbatch, args.seed)
    if args.stress_longest:
        train_batches.sort(
            key=lambda batch: max(int(item["length"]) for item in batch),
            reverse=True,
        )
    required_batches = args.optimizer_steps * accumulation
    if len(train_batches) < required_batches:
        repeats = math.ceil(required_batches / max(1, len(train_batches)))
        train_batches = (train_batches * repeats)[:required_batches]
    else:
        train_batches = train_batches[:required_batches]

    load_started = time.monotonic()
    if args.synthetic_weights:
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        config.use_cache = False
        with init_empty_weights(include_buffers=True):
            model = AutoModelForCausalLM.from_config(
                config,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        model.to_empty(device=torch.device("cuda"))
        model.tie_weights()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda")
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    trainable, total = model.get_nb_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    model_load_seconds = time.monotonic() - load_started
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    measured_started: float | None = None
    measured_actual_tokens = 0
    measured_padded_tokens = 0
    measured_samples = 0
    measured_optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    batch_index = 0
    for optimizer_step in range(args.optimizer_steps):
        for _ in range(accumulation):
            batch = train_batches[batch_index]
            batch_index += 1
            input_ids, attention, targets, actual, padded = collate(
                tokenizer, batch, torch.device("cuda")
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    labels=targets,
                    use_cache=False,
                ).loss
                (loss / accumulation).backward()
            if optimizer_step >= args.warmup_steps:
                measured_actual_tokens += actual
                measured_padded_tokens += padded
                measured_samples += len(batch)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if optimizer_step + 1 == args.warmup_steps:
            measured_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
        elif optimizer_step >= args.warmup_steps:
            measured_optimizer_steps += 1

    torch.cuda.synchronize()
    if measured_started is None:
        raise AssertionError("Measured timer was not initialized")
    measured_seconds = time.monotonic() - measured_started
    actual_rate = measured_actual_tokens / measured_seconds
    padded_rate = measured_padded_tokens / measured_seconds
    sample_rate = measured_samples / measured_seconds
    projection_rows = args.projection_rows or len(rows)
    estimated_tokens_per_epoch = feature_audit["mean_tokens"] * projection_rows
    epoch_seconds = estimated_tokens_per_epoch / actual_rate
    report = {
        "status": "completed",
        "model_kind": args.model_kind,
        "model": str(args.model),
        "manifest": str(args.manifest),
        "language": args.language,
        "config": {
            "precision": "bf16",
            "quantization": None,
            "max_seq_length": args.max_seq_length,
            "microbatch": args.microbatch,
            "effective_batch": args.effective_batch,
            "gradient_accumulation": accumulation,
            "optimizer_steps": args.optimizer_steps,
            "warmup_steps_excluded": args.warmup_steps,
            "learning_rate": args.learning_rate,
            "scheduler": "constant",
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_targets": ["q_proj", "v_proj"],
            "gradient_checkpointing": True,
            "attention": "torch_sdpa",
            "synthetic_weights": args.synthetic_weights,
            "profile_scope": (
                "compute_and_vram_only"
                if args.synthetic_weights
                else "checkpoint_training"
            ),
            "stress_longest": args.stress_longest,
        },
        "dataset": {
            "rows": len(rows),
            "projection_rows": projection_rows,
            "counts": dict(sorted(counts.items())),
            "sample_limit": len(sampled),
            "feature_audit": feature_audit,
            "estimated_tokens_per_epoch": estimated_tokens_per_epoch,
        },
        "model_profile": {
            "model_load_seconds": model_load_seconds,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": trainable / total,
        },
        "throughput": {
            "measured_seconds": measured_seconds,
            "measured_optimizer_steps": measured_optimizer_steps,
            "actual_tokens": measured_actual_tokens,
            "padded_tokens": measured_padded_tokens,
            "samples": measured_samples,
            "actual_tokens_per_second": actual_rate,
            "padded_tokens_per_second": padded_rate,
            "samples_per_second": sample_rate,
            "padding_overhead_fraction": 1
            - measured_actual_tokens / max(1, measured_padded_tokens),
            "seconds_per_optimizer_step": measured_seconds
            / measured_optimizer_steps,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            "reserved_vram_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
        },
        "projection": {
            "one_epoch_seconds": epoch_seconds,
            "one_epoch_hours": epoch_seconds / 3600,
            "five_epoch_hours": epoch_seconds * 5 / 3600,
            "cost_at_0_333_per_hour": {
                "one_epoch": epoch_seconds / 3600 * 0.333,
                "five_epochs": epoch_seconds * 5 / 3600 * 0.333,
            },
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
