from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# GLiNER2 logs Unicode symbols. Force UTF-8 on Windows consoles so a progress
# message cannot abort a valid GPU run with UnicodeEncodeError.
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from guard_smoke.gliguard import train_gliguard_lora_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/fastino_gliguard_300m"),
    )
    parser.add_argument(
        "--train-gliner-jsonl",
        type=Path,
        default=Path("data/guard_smoke/train_gliner.jsonl"),
    )
    parser.add_argument(
        "--valid-gliner-jsonl",
        type=Path,
        default=Path("data/guard_smoke/valid_gliner.jsonl"),
    )
    parser.add_argument(
        "--valid-manifest",
        type=Path,
        default=Path("data/guard_smoke/valid.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/guard_smoke/gliguard_lora"),
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=24)
    parser.add_argument("--max-eval-samples", type=int, default=12)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument("--skip-context-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_gliguard_lora_smoke(
        model_path=args.model_path,
        train_gliner_jsonl=args.train_gliner_jsonl,
        valid_gliner_jsonl=args.valid_gliner_jsonl,
        valid_manifest=args.valid_manifest,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        max_words=args.max_words,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        lora_r=args.lora_r,
        inference_batch_size=args.inference_batch_size,
        run_context_probe=not args.skip_context_probe,
        precision=args.precision,
    )
    concise = {
        "device": result["device"],
        "gpu": result["gpu"],
        "precision": result["precision"],
        "zero_shot": result["zero_shot"]["overall"],
        "after_train": result["after_train"]["overall"],
        "steps": result["training"]["total_steps"],
        "train_seconds": result["training"]["total_time_seconds"],
        "peak_train_vram_mb": result["training"]["peak_vram_mb"],
        "trainable_parameters": result["training"]["trainable_parameters"],
        "context_forward_probe": result["context_forward_probe"],
        "reload_delta": result["max_reload_probability_delta"],
        "metrics": str(args.output_dir / "metrics.json"),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
