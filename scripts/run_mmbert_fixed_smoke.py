from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guard_smoke.fixed_head import FixedHeadConfig, train_fixed_guard


DEFAULT_MMBERT_SMALL = Path("models/mmbert_small_base_smoke")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MMBERT_SMALL)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/guard_smoke/train.jsonl"))
    parser.add_argument("--valid-manifest", type=Path, default=Path("data/guard_smoke/valid.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/guard_smoke/mmbert_fixed"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FixedHeadConfig(
        model_path=str(args.model_path),
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        freeze_encoder=not (args.unfreeze_encoder or args.lora),
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        fp16=not args.no_fp16,
    )
    result = train_fixed_guard(
        args.train_manifest,
        args.valid_manifest,
        args.output_dir,
        config,
    )
    concise = {
        "device": result["device"],
        "gpu": result["gpu"],
        "global_steps": result["global_steps"],
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_vram_mb": result["peak_vram_mb"],
        "trainable_parameters": result["trainable_parameters"],
        "total_parameters": result["total_parameters"],
        "reload_delta": result["max_reload_probability_delta"],
        "valid_overall": result["valid"]["overall"],
        "paired_en_vi": result["valid"]["paired_en_vi"],
        "contract": result["contract"],
        "metrics": str(args.output_dir / "metrics.json"),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
