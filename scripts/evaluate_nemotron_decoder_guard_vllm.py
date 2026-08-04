from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guard_train.binary_metrics import build_report, length_bucket
from guard_train.n23_metrics import build_n23_report
from scripts.evaluate_nemotron_decoder_guard import (
    MODEL_ID,
    MODEL_PATH,
    PreparedExample,
    completed_predictions,
    extract_json,
    normalize_categories,
    normalize_rating,
    prepare_prompt,
    read_jsonl,
    strip_probability_metrics,
    write_jsonl_line,
)


OFFICIAL_TEMPERATURE = 0.6
OFFICIAL_TOP_P = 0.9
# NVIDIA's file omits top-k, so the official Transformers `model.generate()`
# example inherits GenerationConfig's default top_k=50. vLLM defaults to -1,
# therefore it must be set explicitly to reproduce the published path.
OFFICIAL_TOP_K = 50
OFFICIAL_MAX_NEW_TOKENS = 100
OFFICIAL_EOS_TOKEN_IDS = (128001, 128008, 128009)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Nemotron Safety Guard 8B with vLLM continuous batching and APC."
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--lora-adapter",
        type=Path,
        help="Optional PEFT LoRA adapter to apply on top of --model during evaluation.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=8064)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=OFFICIAL_MAX_NEW_TOKENS)
    parser.add_argument("--request-chunk-size", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--quantization",
        choices=("none", "bitsandbytes", "awq", "gptq", "gguf"),
        default="none",
    )
    args = parser.parse_args()
    if args.max_input_tokens + args.max_new_tokens > args.max_model_len:
        raise ValueError("max-input-tokens + max-new-tokens must not exceed max-model-len")
    if args.request_chunk_size <= 0 or args.max_num_seqs <= 0:
        raise ValueError("chunk size and max-num-seqs must be positive")
    return args


def gpu_memory_used_mb() -> float | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(result.stdout.splitlines()[0].strip()) if result.returncode == 0 else None
    except (IndexError, ValueError):
        return None


def make_result(item: PreparedExample, output_text: str, generated_tokens: int) -> dict[str, Any]:
    payload, strict_json = extract_json(output_text)
    rating = normalize_rating(payload.get(item.rating_key) if payload else None)
    parsed_ok = rating is not None
    target = 1 if str(item.row["safety_label"]) == "unsafe" else 0
    # Invalid formatting is a strict end-to-end error, not a silently dropped row.
    prediction = 1 if rating == "unsafe" else 0 if rating == "safe" else 1 - target
    categories, unknown_categories = normalize_categories(
        payload.get("Safety Categories") if payload else None
    )
    result = {
        "example_id": item.row["example_id"],
        "record_uid": item.row.get("record_uid")
        or item.row.get("pair_uid")
        or str(item.row["example_id"]).rsplit(":", 1)[0],
        "source_split": item.row.get("source_split") or item.row.get("benchmark"),
        "decoder_benchmark": item.row.get("decoder_benchmark"),
        "view": item.row["view"],
        "scope": item.row["safety_scope"],
        "language": item.row["language"],
        "tag": item.row.get("tag") or item.row.get("subset") or "unknown",
        "target": target,
        "prediction": prediction,
        "unsafe_probability": float(prediction),
        "parsed_ok": parsed_ok,
        "strict_json": bool(strict_json and parsed_ok),
        "rating_key": item.rating_key,
        "rating": rating,
        "raw_output": output_text,
        "prompt_tokens": item.prompt_tokens,
        "serialized_tokens": item.prompt_tokens,
        "generated_tokens": int(generated_tokens),
        "truncated": item.truncated,
        "length_bucket": length_bucket(item.prompt_tokens),
        "category_scope": item.row.get("category_scope", "unavailable"),
        "category_gold": list(item.row.get("categories") or ()),
        "reported_categories": categories,
        "unknown_categories": unknown_categories,
    }
    if str(item.row.get("category_scope")) in {"prompt", "interaction"}:
        result["category_predictions"] = categories
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    progress_path = args.output_dir / "progress.jsonl"
    existing = completed_predictions(predictions_path) if args.resume else {}
    if predictions_path.exists() and not args.resume:
        predictions_path.unlink()
    if progress_path.exists() and not args.resume:
        progress_path.unlink()

    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prepared: list[PreparedExample] = []
    for index, row in enumerate(rows):
        if str(row["example_id"]) in existing:
            continue
        prompt, tokens, truncated, rating_key = prepare_prompt(
            tokenizer, row, args.max_input_tokens
        )
        prepared.append(PreparedExample(index, row, prompt, tokens, truncated, rating_key))
    # Similar lengths improve batching; the shared official taxonomy remains a common prefix.
    prepared.sort(key=lambda item: (item.prompt_tokens, item.source_index))

    load_started = time.monotonic()
    engine_kwargs: dict[str, Any] = {}
    if args.lora_adapter is not None:
        engine_kwargs.update(enable_lora=True, max_lora_rank=8, max_loras=1)
    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        tokenizer_mode="auto",
        dtype="half",
        quantization=None if args.quantization == "none" else args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
        trust_remote_code=False,
        disable_log_stats=False,
        **engine_kwargs,
    )
    lora_request = (
        LoRARequest("d2_nemotron_vi", 1, str(args.lora_adapter))
        if args.lora_adapter is not None
        else None
    )
    model_load_seconds = time.monotonic() - load_started
    sampling = SamplingParams(
        n=1,
        temperature=OFFICIAL_TEMPERATURE,
        top_p=OFFICIAL_TOP_P,
        top_k=OFFICIAL_TOP_K,
        seed=args.seed,
        max_tokens=args.max_new_tokens,
        stop_token_ids=list(OFFICIAL_EOS_TOKEN_IDS),
        ignore_eos=False,
        skip_special_tokens=True,
    )

    started = time.monotonic()
    peak_vram_mb = gpu_memory_used_mb() or 0.0
    prompt_tokens_total = 0
    generated_tokens_total = 0
    resumed_count = len(rows) - len(prepared)
    mode = "a" if existing else "w"
    with predictions_path.open(mode, encoding="utf-8", newline="\n") as prediction_handle, progress_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as progress_handle:
        for offset in range(0, len(prepared), args.request_chunk_size):
            chunk = prepared[offset : offset + args.request_chunk_size]
            chunk_started = time.monotonic()
            request_outputs = engine.generate(
                [item.prompt for item in chunk],
                sampling,
                use_tqdm=False,
                lora_request=lora_request,
            )
            if len(request_outputs) != len(chunk):
                raise AssertionError(
                    f"vLLM returned {len(request_outputs)} outputs for {len(chunk)} prompts"
                )
            for item, request_output in zip(chunk, request_outputs):
                completion = request_output.outputs[0]
                output_text = completion.text
                generated_tokens = len(completion.token_ids)
                result = make_result(item, output_text, generated_tokens)
                write_jsonl_line(prediction_handle, result)
                existing[str(item.row["example_id"])] = result
                prompt_tokens_total += item.prompt_tokens
                generated_tokens_total += generated_tokens

            current_vram = gpu_memory_used_mb()
            if current_vram is not None:
                peak_vram_mb = max(peak_vram_mb, current_vram)
            elapsed = time.monotonic() - started
            new_processed = len(existing) - resumed_count
            rate = new_processed / max(elapsed, 1e-9)
            progress = {
                "processed": len(existing),
                "total": len(rows),
                "progress": len(existing) / max(1, len(rows)),
                "elapsed_seconds": elapsed,
                "examples_per_second": rate,
                "prompt_tokens_per_second": prompt_tokens_total / max(elapsed, 1e-9),
                "generated_tokens_per_second": generated_tokens_total / max(elapsed, 1e-9),
                "eta_seconds": (len(rows) - len(existing)) / max(rate, 1e-9),
                "last_chunk_seconds": time.monotonic() - chunk_started,
                "peak_vram_mb": peak_vram_mb,
                "engine": "vllm",
                "prefix_caching": True,
            }
            write_jsonl_line(progress_handle, progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)

    predictions = [
        existing[str(row["example_id"])]
        for row in rows
        if str(row["example_id"]) in existing
    ]
    if len(predictions) != len(rows):
        raise AssertionError(f"Only {len(predictions)}/{len(rows)} predictions were produced")
    binary_report = build_report(predictions)
    strip_probability_metrics(binary_report)
    parsed_rows = [row for row in predictions if row["parsed_ok"]]
    conditional_report = build_report(parsed_rows) if parsed_rows else None
    if conditional_report is not None:
        strip_probability_metrics(conditional_report)
    n23_report = build_n23_report(predictions)
    elapsed = time.monotonic() - started
    report = {
        "model_kind": "decoder_guard_zero_shot",
        "model": MODEL_ID,
        "model_path": str(args.model),
        "manifest": str(args.manifest),
        "examples": len(predictions),
        "engine": {
            "name": "vllm",
            "dtype": "float16",
            "quantization": args.quantization,
            "automatic_prefix_caching": True,
            "chunked_prefill": True,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "request_chunk_size": args.request_chunk_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
        },
        "generation": {
            "source": "official generation_config.json",
            "do_sample": True,
            "temperature": OFFICIAL_TEMPERATURE,
            "top_p": OFFICIAL_TOP_P,
            "top_k": OFFICIAL_TOP_K,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "eos_token_ids": list(OFFICIAL_EOS_TOKEN_IDS),
        },
        "max_input_tokens": args.max_input_tokens,
        "truncated_examples": sum(bool(row["truncated"]) for row in predictions),
        "format": {
            "parsed": len(parsed_rows),
            "parse_failures": len(predictions) - len(parsed_rows),
            "strict_json": sum(bool(row["strict_json"]) for row in predictions),
            "unknown_category_outputs": sum(
                bool(row["unknown_categories"]) for row in predictions
            ),
            "strict_failure_policy": "A missing Safe/Unsafe rating is counted as an incorrect prediction.",
        },
        "binary": binary_report,
        "binary_conditional_on_parsed": conditional_report,
        "N23": n23_report,
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "elapsed_seconds": elapsed,
            "wall_seconds_including_load": model_load_seconds + elapsed,
            "examples_per_second": len(prepared) / max(elapsed, 1e-9),
            "prompt_tokens": prompt_tokens_total,
            "generated_tokens": generated_tokens_total,
            "prompt_tokens_per_second": prompt_tokens_total / max(elapsed, 1e-9),
            "generated_tokens_per_second": generated_tokens_total / max(elapsed, 1e-9),
            "peak_vram_mb": peak_vram_mb,
        },
    }
    temporary = args.output_dir / "metrics.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output_dir / "metrics.json")
    print(json.dumps(report["runtime"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
