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

import torch

from guard_smoke.data import load_manifest
from guard_smoke.gliguard import evaluate_gliguard
from guard_smoke.gliguard_single_label import install_phase0_single_label_ce
from guard_train.gliguard_precision import AuditedGradScaler
from guard_train.precision import cuda_supports_native_bf16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase0_experiments.json"))
    parser.add_argument("--run-id", default="E1-G-EV-512")
    parser.add_argument("--output-root", type=Path, default=Path("reports/experiment_runs"))
    parser.add_argument(
        "--warm-resume-adapter",
        type=Path,
        help=(
            "Reload adapter weights. The pinned GLiNER2 trainer does not restore optimizer/scheduler/global "
            "step, so this is explicitly a warm restart rather than exact resume."
        ),
    )
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-strategy", choices=("steps", "epoch"))
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--max-train-samples", type=int, default=-1)
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument(
        "--post-eval-limit",
        type=int,
        default=None,
        help="Limit the standalone post-train validation only for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from gliner2 import GLiNER2
    from gliner2.processor import SamplingConfig
    import gliner2.training.trainer as gliner_trainer_module
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    # The pinned trainer otherwise creates an FP16 scaler at 65,536 and then
    # advances scheduler/global_step even when the optimizer update is skipped.
    gliner_trainer_module.GradScaler = AuditedGradScaler

    experiment = json.loads(args.config.read_text(encoding="utf-8"))
    run = next((item for item in experiment["runs"] if item["id"] == args.run_id), None)
    if run is None or run["model_kind"] != "gliguard_schema":
        raise ValueError(f"Unknown/non-GLiGuard run: {args.run_id}")
    recipe = experiment["global_recipe"]
    micro_batch = int(args.micro_batch_size or run.get("micro_batch_size", 1))
    effective_batch = int(recipe["effective_batch_target"])
    if effective_batch % micro_batch:
        raise ValueError("effective batch must be divisible by micro batch")
    accumulation = effective_batch // micro_batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = args.precision
    if precision == "auto":
        precision = (
            "bf16"
            if device.type == "cuda" and cuda_supports_native_bf16(device)
            else "fp16"
            if device.type == "cuda"
            else "fp32"
        )
    if precision == "bf16" and (device.type != "cuda" or not cuda_supports_native_bf16(device)):
        raise RuntimeError("BF16 requested on device without native BF16 support")

    output_dir = args.output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model = GLiNER2.from_pretrained(str(run["base_model"]))
    model.processor.sampling_config = SamplingConfig(
        remove_json_structure_prob=0.0,
        shuffle_json_fields=False,
        remove_json_field_prob=0.0,
        remove_entities_prob=0.0,
        shuffle_entities=False,
        remove_entity_prob=0.0,
        synthetic_entity_label_prob=0.0,
        remove_relations_prob=0.0,
        swap_head_tail_prob=0.0,
        remove_classification_prob=0.0,
        shuffle_classification_labels=True,
        remove_classification_label_prob=0.0,
        synthetic_label_prob=0.0,
        include_true_label_prob=1.0,
        max_num_labels=2,
    )
    install_phase0_single_label_ce(model)
    config = TrainingConfig(
        output_dir=str(output_dir),
        experiment_name=args.run_id,
        num_epochs=int(args.epochs or run.get("epochs", 2)),
        max_steps=int(
            args.max_optimizer_steps
            if args.max_optimizer_steps is not None
            else run.get("max_optimizer_steps", -1)
        ),
        batch_size=micro_batch,
        eval_batch_size=int(run.get("eval_batch_size", micro_batch)),
        gradient_accumulation_steps=accumulation,
        encoder_lr=float(run.get("encoder_learning_rate", 1e-5)),
        # GLiNER2 freezes every non-LoRA parameter, and its LoRA optimizer uses
        # task_lr for all adapters.  Use the locked encoder/adapter LR so E1 is
        # comparable to E2 instead of accidentally training adapters at head LR.
        task_lr=float(run.get("encoder_learning_rate", 1e-5)),
        weight_decay=0.01,
        scheduler_type="linear",
        warmup_ratio=0.03,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        eval_strategy=str(args.eval_strategy or run.get("eval_strategy", "epoch")),
        eval_steps=int(run.get("eval_every", 250)),
        save_total_limit=3,
        save_best=True,
        metric_for_best="eval_loss",
        greater_is_better=False,
        logging_steps=int(run.get("log_every", 10)),
        report_to_wandb=False,
        num_workers=int(run.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        seed=int(recipe["seed"]),
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        validate_data=True,
        # Native-fit manifests already satisfy final schema+text <=512. This
        # word cap is only an additional guard; it is not used for eligibility.
        max_len=int(run["context"]),
        use_lora=True,
        lora_r=int(recipe["lora"]["r"]),
        lora_alpha=float(recipe["lora"]["alpha"]),
        lora_dropout=float(recipe["lora"]["dropout"]),
        save_adapter_only=True,
    )
    trainer = GLiNER2Trainer(model=model, config=config)
    resume_semantics = "fresh"
    if args.warm_resume_adapter is not None:
        trainer.load_checkpoint(str(args.warm_resume_adapter))
        install_phase0_single_label_ce(trainer.model)
        resume_semantics = "adapter_weights_only_warm_restart"
    train_result = trainer.train(
        train_data=str(run["train_manifest"]),
        eval_data=str(run["valid_manifest"]),
    )
    scaler_audit = (
        trainer.scaler.audit()
        if isinstance(trainer.scaler, AuditedGradScaler)
        else {"enabled": False, "skipped_optimizer_updates": 0}
    )
    (output_dir / "scaler_audit.json").write_text(
        json.dumps(scaler_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if int(scaler_audit["skipped_optimizer_updates"]) != 0:
        raise RuntimeError(
            "GLiNER2 silently skipped one or more FP16 optimizer updates; "
            f"audit={scaler_audit}"
        )
    valid_examples = load_manifest(Path(run["canonical_valid"]))
    if args.post_eval_limit is not None:
        valid_examples = valid_examples[: args.post_eval_limit]
    validation = evaluate_gliguard(
        trainer.model,
        valid_examples,
        batch_size=int(run.get("inference_batch_size", micro_batch)),
        max_words=int(run["context"]),
    )
    result = {
        "status": "completed",
        "run_id": args.run_id,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "precision": precision,
        "micro_batch_size": micro_batch,
        "effective_batch_size": effective_batch,
        "gradient_accumulation_steps": accumulation,
        "actual_lora_learning_rate": float(run.get("encoder_learning_rate", 1e-5)),
        "eval_strategy": str(args.eval_strategy or run.get("eval_strategy", "epoch")),
        "scaler_audit": scaler_audit,
        "resume_semantics": resume_semantics,
        "critical_resume_limit": (
            "Installed GLiNER2 restores model/adapter weights only; optimizer, scheduler and "
            "global step are not restored. Use tmux and stable storage for the primary run."
        ),
        "train_result": train_result,
        "validation": validation,
        "final_adapter": str(output_dir / "final"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": args.run_id,
                "validation": validation["overall"],
                "resume_semantics": resume_semantics,
                "output": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
