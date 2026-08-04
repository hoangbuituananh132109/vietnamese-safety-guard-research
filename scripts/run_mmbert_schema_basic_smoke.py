"""Run the locked Phase-0 mmBERT GLi-style binary schema smoke test."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from guard_smoke.mmbert_schema import MMBertSchemaConfig, train_binary_schema_guard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/mmbert_small_base_smoke"),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("data/guard_smoke/train.jsonl"),
    )
    parser.add_argument(
        "--valid-manifest",
        type=Path,
        default=Path("data/guard_smoke/valid.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/guard_smoke/mmbert_schema_basic"),
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-4)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-label-shuffle", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MMBertSchemaConfig(
        model_path=str(args.model_path),
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        encoder_learning_rate=args.encoder_learning_rate,
        head_learning_rate=args.head_learning_rate,
        shuffle_labels=not args.no_label_shuffle,
        mixed_precision=not args.no_mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    result = train_binary_schema_guard(
        args.train_manifest,
        args.valid_manifest,
        args.output_dir,
        config,
    )
    concise = {
        "status": result["status"],
        "task": result["architecture"]["task_name"],
        "labels": result["architecture"]["labels"],
        "loss": result["architecture"]["loss"],
        "separate_prompt_response_heads": result["architecture"][
            "separate_prompt_response_heads"
        ],
        "device": result["device"],
        "gpu": result["gpu"],
        "eligible": {
            "train": result["train_examples_eligible"],
            "valid": result["valid_examples_eligible"],
        },
        "global_steps": result["global_steps"],
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_vram_mb": result["peak_vram_mb"],
        "trainable_parameter_counts": result["trainable_parameter_counts"],
        "first_step_gradient_norms": result["first_step_gradient_norms"],
        "marker_embedding_max_update": result["marker_embedding_max_update"],
        "reload_delta": result["max_reload_probability_delta"],
        "valid_overall": result["valid"]["overall"],
        "paired_en_vi": result["valid"]["paired_en_vi"],
        "metrics": str(args.output_dir / "metrics.json"),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
