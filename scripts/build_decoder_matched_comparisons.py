from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from guard_train.n23_metrics import build_n23_report
from scripts.evaluate_mmbert_checkpoint import build_report


ROOT = Path(__file__).resolve().parents[1]
ENCODER_RUNS = (
    "E3-M-E-8K",
    "E4-M-EV-MATCHED-8K",
    "E5-M-EV-FULL-8K",
    "E7-M-SCHEMA-EV-8K",
)
BENCHMARKS = ("nemotron_test", "sea_paired")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


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


def upsert_index_job(run_id: str, job: dict[str, Any]) -> None:
    path = ROOT / "reports" / "evaluation_matrix" / run_id / "matrix_index.json"
    index = read_json(path, {})
    if not isinstance(index, dict):
        index = {}
    jobs = [
        item
        for item in index.get("jobs", [])
        if not (
            item.get("suite") == job["suite"]
            and item.get("benchmark") == job["benchmark"]
            and item.get("label_order") == job["label_order"]
        )
    ]
    jobs.append(job)
    index.update({"run_id": run_id, "jobs": jobs})
    atomic_json(path, index)


def build_one(decoder_run_id: str, encoder_run_id: str, benchmark: str) -> dict[str, Any]:
    decoder_dir = (
        ROOT
        / "reports"
        / "evaluation_matrix"
        / decoder_run_id
        / f"decoder_matched__{benchmark}__canonical"
    )
    decoder_predictions = read_jsonl(decoder_dir / "predictions.jsonl")
    target_ids = {str(row["example_id"]) for row in decoder_predictions}
    source_dir = (
        ROOT
        / "reports"
        / "evaluation_matrix"
        / encoder_run_id
        / f"full_mmbert__{benchmark}__canonical"
    )
    source_predictions = read_jsonl(source_dir / "predictions.jsonl")
    selected = [row for row in source_predictions if str(row.get("example_id")) in target_ids]
    if len(selected) != len(target_ids):
        found = {str(row.get("example_id")) for row in selected}
        missing = sorted(target_ids - found)
        raise AssertionError(
            f"{encoder_run_id}/{benchmark}: matched {len(selected)}/{len(target_ids)}; "
            f"first missing IDs: {missing[:5]}"
        )
    output_dir = (
        ROOT
        / "reports"
        / "evaluation_matrix"
        / encoder_run_id
        / f"decoder_matched__{benchmark}__canonical"
    )
    write_jsonl(output_dir / "predictions.jsonl", selected)
    binary = build_report(selected)
    n23 = build_n23_report(selected)
    metrics = {
        "comparison_scope": "Exact example IDs evaluated by the decoder baseline",
        "source_predictions": str(source_dir.relative_to(ROOT)),
        "decoder_predictions": str(decoder_dir.relative_to(ROOT)),
        "examples": len(selected),
        "truncated_examples": sum(bool(row.get("truncated")) for row in selected),
        "binary": binary,
        "N23": n23,
    }
    atomic_json(output_dir / "metrics.json", metrics)
    job = {
        "suite": "decoder_matched",
        "benchmark": benchmark,
        "label_order": "canonical",
        "examples": len(selected),
        "metrics": str((output_dir / "metrics.json").relative_to(ROOT)),
        "predictions": str((output_dir / "predictions.jsonl").relative_to(ROOT)),
    }
    upsert_index_job(encoder_run_id, job)
    return {
        "encoder_run_id": encoder_run_id,
        "benchmark": benchmark,
        "examples": len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute encoder metrics on the exact IDs sampled by the decoder baseline."
    )
    parser.add_argument("--decoder-run-id", default="D1-NEMOTRON-GUARD-8B-V3")
    args = parser.parse_args()
    results = [
        build_one(args.decoder_run_id, encoder_run_id, benchmark)
        for encoder_run_id in ENCODER_RUNS
        for benchmark in BENCHMARKS
    ]
    output = (
        ROOT / "reports" / "decoder_baseline" / "matched_encoder_comparisons.json"
    )
    atomic_json(output, {"decoder_run_id": args.decoder_run_id, "jobs": results})
    print(json.dumps({"jobs": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
