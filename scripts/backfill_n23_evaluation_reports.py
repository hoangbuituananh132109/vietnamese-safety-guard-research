from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guard_train.n23_metrics import build_n23_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add detailed N23 metrics to completed evaluation jobs.")
    parser.add_argument("--evaluation-root", type=Path, default=Path("reports/evaluation_matrix"))
    parser.add_argument("--run-id", action="append", dest="run_ids")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def main() -> None:
    args = parse_args()
    roots = (
        [args.evaluation_root / run_id for run_id in args.run_ids]
        if args.run_ids
        else [path for path in args.evaluation_root.iterdir() if path.is_dir()]
    )
    updated = 0
    skipped = 0
    for run_root in roots:
        for metrics_path in sorted(run_root.glob("*/metrics.json")):
            predictions_path = metrics_path.parent / "predictions.jsonl"
            if not predictions_path.exists():
                skipped += 1
                continue
            report = build_n23_report(read_jsonl(predictions_path))
            if report is None:
                skipped += 1
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["N23"] = report
            temporary = metrics_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(metrics_path)
            updated += 1
            print(f"updated={metrics_path}", flush=True)
    print(f"summary updated={updated} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
