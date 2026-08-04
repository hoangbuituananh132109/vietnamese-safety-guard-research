from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one completed B0/E1/E2/E3/E4/E5/E7 run on its locked matrix."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/phase0_experiments.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("reports/experiment_runs"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/evaluation_matrix"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--limit", type=int, help="Smoke only; cap every benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run = next((item for item in config["runs"] if item["id"] == args.run_id), None)
    if run is None:
        raise KeyError(args.run_id)
    benchmarks = config["benchmarks"]
    # Prefer the suites locked into the selected experiment config.  The
    # fallback keeps old configs reproducible, while allowing clean ablations
    # (for example the P/PR-only no-R matrix) to use new run IDs without
    # silently selecting the wrong benchmark family.
    configured_suites = run.get("eval")
    if configured_suites:
        suites = tuple(str(value) for value in configured_suites)
    elif run["id"] in {"B0-GLI-ZS", "E1-G-EV-512", "E2-M-EV-GLI-COMPAT-8K"}:
        suites = ("e1_e2_primary_native", "e1_e2_secondary_shared_truncated")
    else:
        suites = ("full_mmbert", "e1_e2_primary_native")

    jobs: list[dict] = []
    for suite in suites:
        for benchmark_name, manifest in benchmarks[suite].items():
            if not isinstance(manifest, str) or not manifest.endswith(".jsonl"):
                continue
            label_orders = (
                ("canonical", "reversed")
                if run["model_kind"] == "mmbert_schema"
                else ("canonical",)
            )
            for label_order in label_orders:
                jobs.append(
                    {
                        "suite": suite,
                        "benchmark": benchmark_name,
                        "manifest": manifest,
                        "label_order": label_order,
                    }
                )

    run_output = args.output_root / args.run_id
    run_output.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, job in enumerate(jobs, 1):
        name = f"{job['suite']}__{job['benchmark']}__{job['label_order']}"
        output = run_output / name
        print(f"[{index}/{len(jobs)}] {name}", flush=True)
        metric_path = output / "metrics.json"
        if metric_path.exists() and args.limit is None:
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            results.append(
                {
                    **job,
                    "output": str(output),
                    "metrics": str(metric_path),
                    "examples": metric.get("examples"),
                    "reused": True,
                }
            )
            print(f"[{index}/{len(jobs)}] reused {metric_path}", flush=True)
            continue
        if run["model_kind"] == "gliguard_schema":
            command = [
                sys.executable,
                "scripts/evaluate_gliguard_checkpoint.py",
                "--base-model",
                run["base_model"],
                "--manifest",
                job["manifest"],
                "--output-dir",
                str(output),
                "--batch-size",
                str(args.batch_size),
            ]
            if run["id"] != "B0-GLI-ZS":
                command.extend(["--adapter", str(args.run_root / args.run_id / "final")])
        else:
            command = [
                sys.executable,
                "scripts/evaluate_scalable_mmbert_checkpoint.py",
                "--model-kind",
                "schema" if run["model_kind"] == "mmbert_schema" else "fixed",
                "--checkpoint",
                str(args.run_root / args.run_id / "final"),
                "--base-model",
                run["base_model"],
                "--manifest",
                job["manifest"],
                "--output-dir",
                str(output),
                "--max-length",
                str(run["context"]),
                "--batch-size",
                str(args.batch_size),
                "--precision",
                args.precision,
                "--label-order",
                job["label_order"],
            ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        subprocess.run(command, check=True)
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        results.append(
            {
                **job,
                "output": str(output),
                "metrics": str(metric_path),
                "examples": metric.get("examples"),
            }
        )
    index_path = run_output / "matrix_index.json"
    index_path.write_text(
        json.dumps(
            {"run_id": args.run_id, "jobs": results, "smoke_limit": args.limit},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"matrix_index={index_path}", flush=True)


if __name__ == "__main__":
    main()
