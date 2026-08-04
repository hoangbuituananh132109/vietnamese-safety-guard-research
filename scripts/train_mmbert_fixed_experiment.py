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

from guard_train.mmbert_fixed import FixedTrainingConfig, train_fixed_multitask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase0_experiments.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("reports/experiment_runs"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = json.loads(args.config.read_text(encoding="utf-8"))
    run = next((item for item in experiment["runs"] if item["id"] == args.run_id), None)
    if run is None:
        raise KeyError(f"Unknown run ID: {args.run_id}")
    if run["model_kind"] != "mmbert_fixed":
        raise ValueError(f"{args.run_id} is not an mmbert_fixed run")
    recipe = experiment["global_recipe"]
    config = FixedTrainingConfig(
        model_path=run["base_model"],
        max_length=int(run["context"]),
        micro_batch_size=int(args.micro_batch_size or run.get("micro_batch_size", 1)),
        effective_batch_size=int(recipe["effective_batch_target"]),
        epochs=int(args.epochs or run.get("epochs", 2)),
        max_optimizer_steps=int(
            args.max_optimizer_steps
            if args.max_optimizer_steps is not None
            else run.get("max_optimizer_steps", 0)
        ),
        encoder_learning_rate=float(run.get("encoder_learning_rate", 1e-5)),
        head_learning_rate=float(run.get("head_learning_rate", 1e-4)),
        use_lora=True,
        lora_r=int(recipe["lora"]["r"]),
        lora_alpha=float(recipe["lora"]["alpha"]),
        lora_dropout=float(recipe["lora"]["dropout"]),
        gradient_checkpointing=bool(run.get("gradient_checkpointing", True)),
        mixed_precision=str(args.precision or run.get("precision", "auto")),
        grad_scaler_init_scale=float(run.get("grad_scaler_init_scale", 512.0)),
        grad_scaler_growth_interval=int(run.get("grad_scaler_growth_interval", 1_000_000)),
        enable_categories=bool(run.get("category_tasks", False)),
        category_loss_weight=float(run.get("category_loss_weight", 1.0)),
        save_every=int(run.get("save_every", 250)),
        log_every=int(run.get("log_every", 10)),
        seed=int(recipe["seed"]),
    )
    output_dir = args.output_root / args.run_id
    result = train_fixed_multitask(
        Path(run["train_manifest"]),
        Path(run["valid_manifest"]),
        output_dir,
        config,
        resume_from=args.resume_from,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": args.run_id,
                "optimizer_steps": result["completed_optimizer_steps"],
                "validation": result["validation"]["binary"],
                "output": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
