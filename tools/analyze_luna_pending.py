from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.luna_overnight_runner import length_bucket, route_for


def load(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    root = ROOT / "data" / "luna_overnight"
    specs = [
        ("train", ROOT / "data/final/nemotron_train_en_vi_v10_final.jsonl", root / "nemotron_train_20260801"),
        ("valid", ROOT / "data/final/nemotron_valid_en_vi_v10_final.jsonl", root / "nemotron_valid_20260802"),
        ("test", ROOT / "data/final/nemotron_test_en_vi_v10_final.jsonl", root / "nemotron_test_20260802"),
    ]
    report: dict[str, dict] = {}
    for split, source_path, output_dir in specs:
        rows = load(source_path)
        terminal: set[str] = set()
        for name in ("passed.jsonl", "needs_audit.jsonl", "exhausted_normal.jsonl", "tail_escalation.jsonl"):
            path = output_dir / name
            if path.exists():
                terminal.update(str(row.get("record_uid")) for row in load(path) if row.get("record_uid"))
        pending = [row for row in rows if str(row.get("record_uid")) not in terminal]
        route_bucket = Counter(
            f"{str(row.get('length_bucket') or length_bucket(row))}/{route_for(row)}" for row in pending
        )
        tags = Counter(str(row.get("tag")) for row in pending)
        labels = Counter(str(row.get("prompt_label")) for row in pending)
        categories = Counter()
        for row in pending:
            for category in str(row.get("violated_categories") or "").split(","):
                if category.strip():
                    categories[category.strip()] += 1
        lengths = [len(row.get("prompt_en") or "") + len(row.get("response_en") or "") for row in pending]
        report[split] = {
            "source": len(rows), "terminal": len(terminal), "pending": len(pending),
            "route_bucket": dict(route_bucket), "tags": dict(tags), "prompt_labels": dict(labels),
            "top_categories": categories.most_common(12),
            "chars": {"min": min(lengths, default=0), "max": max(lengths, default=0)},
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
