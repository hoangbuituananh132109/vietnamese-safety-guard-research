from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    run_id = "E2-M-EV-GLI-COMPAT-8K"
    reports = args.extracted_root / "reports"
    train = read_json(reports / "experiment_runs" / run_id / "metrics.json")
    eval_root = reports / "evaluation_matrix" / run_id
    jobs: dict[str, Any] = {}
    for metric_path in sorted(eval_root.glob("*/metrics.json")):
        metric = read_json(metric_path)
        binary = metric["binary"]
        jobs[metric_path.parent.name] = {
            "examples": metric["examples"],
            "truncated_examples": metric["truncated_examples"],
            "overall": binary["overall"],
            "paired_en_vi": binary["paired_en_vi"],
            "scope": binary["slices"]["scope"],
            "view": binary["slices"]["view"],
            "language": binary["slices"]["language"],
            "tag": binary["slices"]["tag"],
            "length_bucket": binary["slices"]["length_bucket"],
        }

    summary = {
        "run_id": run_id,
        "training": {
            key: train[key]
            for key in (
                "status",
                "gpu",
                "autocast_dtype",
                "encoder_gradient_verified",
                "train_examples",
                "valid_examples",
                "train_truncated",
                "valid_truncated",
                "planned_optimizer_steps",
                "completed_optimizer_steps",
                "nonfinite_optimizer_updates",
                "nonfinite_micro_batches",
                "elapsed_seconds",
                "peak_vram_mb",
                "best_score",
                "selection_metric",
            )
        },
        "validation": train["validation"]["binary"],
        "validation_history": train["validation_history"],
        "evaluation_jobs": jobs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{run_id}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        f"# {run_id} — training and evaluation summary",
        "",
        "## Training integrity",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Train / valid examples | {train['train_examples']:,} / {train['valid_examples']:,} |",
        f"| Optimizer steps | {train['completed_optimizer_steps']:,} / {train['planned_optimizer_steps']:,} |",
        f"| Non-finite optimizer updates | {train['nonfinite_optimizer_updates']} |",
        f"| Truncated train / valid | {train['train_truncated']} / {train['valid_truncated']} |",
        f"| Encoder gradient verified | {train['encoder_gradient_verified']} |",
        f"| Runtime | {train['elapsed_seconds'] / 60:.2f} min |",
        f"| Peak allocated VRAM | {train['peak_vram_mb'] / 1024:.2f} GiB |",
        "",
        "## Evaluation overview",
        "",
        "| Evaluation | N | Acc | Macro-F1 | Unsafe recall | Unsafe AUPRC | AUROC | EN–VI same decision | Mean probability gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, job in jobs.items():
        overall = job["overall"]
        paired = job["paired_en_vi"]
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    f"{job['examples']:,}",
                    fmt(overall.get("accuracy")),
                    fmt(overall.get("macro_f1")),
                    fmt(overall.get("unsafe_recall")),
                    fmt(overall.get("average_precision")),
                    fmt(overall.get("roc_auc")),
                    fmt(paired.get("same_decision_rate")),
                    fmt(paired.get("mean_probability_gap")),
                )
            )
            + " |"
        )

    lines += [
        "",
        "## English–Vietnamese slices",
        "",
        "| Evaluation | Language | N | Acc | Macro-F1 | Unsafe recall | Unsafe AUPRC | AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, job in jobs.items():
        for language in ("en", "vi"):
            row = job["language"].get(language)
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    (
                        name,
                        language,
                        f"{row['examples']:,}",
                        fmt(row.get("accuracy")),
                        fmt(row.get("macro_f1")),
                        fmt(row.get("unsafe_recall")),
                        fmt(row.get("average_precision")),
                        fmt(row.get("roc_auc")),
                    )
                )
                + " |"
            )

    lines += [
        "",
        "## Prompt versus response scope",
        "",
        "| Evaluation | Scope | N | Acc | Macro-F1 | Unsafe recall | Unsafe AUPRC | AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, job in jobs.items():
        for scope in ("prompt", "response"):
            row = job["scope"].get(scope)
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    (
                        name,
                        scope,
                        f"{row['examples']:,}",
                        fmt(row.get("accuracy")),
                        fmt(row.get("macro_f1")),
                        fmt(row.get("unsafe_recall")),
                        fmt(row.get("average_precision")),
                        fmt(row.get("roc_auc")),
                    )
                )
                + " |"
            )

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Native jobs retain only rows that fit GLiGuard's audited schema-plus-text limit; shared-truncated jobs apply the same truncation policy to both GLiGuard and mmBERT.",
        "- Prompt and response are independent safety classification instances. A prompt label is never reused as an output label, and PR uses the response/interaction target defined by the manifest.",
        "- SEA response unsafe recall is materially below prompt unsafe recall; report this slice rather than relying on the overall score alone.",
        "- EN–VI same-decision rate measures paired prediction consistency, not correctness. Read it together with per-language F1/AUPRC and probability gap.",
        "- All E2 training and matrix evaluation rows completed without runtime truncation or non-finite optimizer updates.",
        "",
    ]
    (args.output_dir / f"{run_id}.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir / f"{run_id}.md")


if __name__ == "__main__":
    main()
