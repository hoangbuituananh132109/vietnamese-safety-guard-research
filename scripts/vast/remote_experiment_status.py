from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    return parser.parse_args()


def main() -> None:
    run_id = parse_args().run_id
    totals = {
        "E1-G-EV-512": 7226,
        "E2-M-EV-GLI-COMPAT-8K": 7226,
    }
    total = totals.get(run_id)
    log = Path("reports/run_logs") / f"{run_id}.log"
    metrics = Path("reports/experiment_runs") / run_id / "metrics.json"
    text = log.read_text(errors="replace") if log.exists() else ""
    if run_id == "E1-G-EV-512":
        found = [
            int(value)
            for value in re.findall(r"Training:\s+\d+%.*?\|\s+(\d+)/7226", text)
        ]
    else:
        train_log = Path("reports/experiment_runs") / run_id / "train_log.jsonl"
        found = []
        if train_log.exists():
            for line in train_log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    found.append(int(json.loads(line)["optimizer_step"]))
    step = max(found, default=0)
    print(f"run={run_id}")
    suffix = f"/{total} ({100 * step / total:.2f}%)" if total else ""
    print(f"step={step}{suffix}")
    print(f"log={log}")
    print(f"metrics_ready={metrics.exists()}")
    if not metrics.exists():
        return
    data = json.loads(metrics.read_text(encoding="utf-8"))
    if run_id == "E1-G-EV-512":
        print(f"optimizer_steps={data['train_result']['total_steps']}")
        print(f"scaler={json.dumps(data['scaler_audit'])}")
        print(f"validation={json.dumps(data['validation']['overall'])}")
    else:
        print(f"optimizer_steps={data['completed_optimizer_steps']}")
        print(f"nonfinite_updates={data.get('nonfinite_optimizer_updates')}")
        print(f"validation={json.dumps(data['validation']['binary'])}")


if __name__ == "__main__":
    main()
