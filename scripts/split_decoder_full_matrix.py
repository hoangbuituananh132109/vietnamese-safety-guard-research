from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guard_train.binary_metrics import build_report
from guard_train.n23_metrics import build_n23_report
from scripts.evaluate_nemotron_decoder_guard import read_jsonl, strip_probability_metrics

RUN_ID = "D1-NEMOTRON-GUARD-8B-V3"
EXPECTED = ("nemotron_valid", "nemotron_test", "sea_paired")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a single vLLM run into the 3 E3-E7 benchmark jobs.")
    parser.add_argument(
        "--combined-dir",
        type=Path,
        default=ROOT / "reports" / "decoder_baseline" / "full_combined",
    )
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()

    combined_metrics = read_json(args.combined_dir / "metrics.json")
    predictions = read_jsonl(args.combined_dir / "predictions.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get("decoder_benchmark") or "missing")].append(row)
    present = tuple(name for name in EXPECTED if grouped.get(name))
    if not present:
        raise AssertionError("No recognized decoder benchmark groups were found")

    matrix_root = ROOT / "reports" / "evaluation_matrix" / args.run_id
    jobs: list[dict[str, Any]] = []
    for benchmark in present:
        rows = grouped[benchmark]
        binary = build_report(rows)
        strip_probability_metrics(binary)
        parsed = [row for row in rows if row.get("parsed_ok")]
        conditional = build_report(parsed) if parsed else None
        if conditional is not None:
            strip_probability_metrics(conditional)
        output_dir = matrix_root / f"full_mmbert__{benchmark}__canonical"
        write_jsonl(output_dir / "predictions.jsonl", rows)
        metrics = {
            "model_kind": combined_metrics.get("model_kind"),
            "model": combined_metrics.get("model"),
            "engine": combined_metrics.get("engine"),
            "generation": combined_metrics.get("generation"),
            "benchmark": benchmark,
            "examples": len(rows),
            "truncated_examples": sum(bool(row.get("truncated")) for row in rows),
            "format": {
                "parsed": len(parsed),
                "parse_failures": len(rows) - len(parsed),
                "strict_json": sum(bool(row.get("strict_json")) for row in rows),
                "unknown_category_outputs": sum(
                    bool(row.get("unknown_categories")) for row in rows
                ),
                "strict_failure_policy": combined_metrics.get("format", {}).get(
                    "strict_failure_policy"
                ),
            },
            "binary": binary,
            "binary_conditional_on_parsed": conditional,
            "N23": build_n23_report(rows),
            "runtime": {
                **(combined_metrics.get("runtime") or {}),
                "shared_combined_run": True,
                "combined_examples": len(predictions),
            },
        }
        atomic_json(output_dir / "metrics.json", metrics)
        jobs.append(
            {
                "suite": "full_mmbert",
                "benchmark": benchmark,
                "label_order": "canonical",
                "examples": len(rows),
                "metrics": str((output_dir / "metrics.json").relative_to(ROOT)),
                "predictions": str((output_dir / "predictions.jsonl").relative_to(ROOT)),
            }
        )

    atomic_json(
        matrix_root / "matrix_index.json",
        {
            "run_id": args.run_id,
            "model_kind": "decoder_guard_zero_shot",
            "model": combined_metrics.get("model"),
            "engine": combined_metrics.get("engine"),
            "generation": combined_metrics.get("generation"),
            "jobs": jobs,
        },
    )
    print(json.dumps({"run_id": args.run_id, "jobs": jobs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
