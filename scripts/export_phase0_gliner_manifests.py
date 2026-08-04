from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guard_smoke.constants import TEXT_SAFETY_TASK
from guard_smoke.data import GuardExample, gliner_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export locked one-task GLiNER2 payloads from common Phase-0 manifests."
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/guard_phase0_common_512")
    )
    parser.add_argument("--splits", default="train,valid,test")
    return parser.parse_args()


def example_from_row(row: dict[str, Any]) -> GuardExample:
    names = {field.name for field in fields(GuardExample)}
    payload = {name: row[name] for name in names}
    payload["categories"] = tuple(payload.get("categories") or ())
    example = GuardExample(**payload)
    example.validate()
    return example


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gliner_payload(payload: dict[str, Any], example: GuardExample) -> None:
    if set(payload) != {"input", "output", "metadata"}:
        raise ValueError(f"Unexpected GLi payload keys for {example.example_id}: {payload.keys()}")
    if payload["input"] != example.text:
        raise ValueError(f"Text changed during GLi export: {example.example_id}")
    output = payload["output"]
    if set(output) != {"classifications"}:
        raise ValueError(f"Unexpected GLi output tasks for {example.example_id}: {output.keys()}")
    tasks = output["classifications"]
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError(f"Expected exactly one classification task: {example.example_id}")
    task = tasks[0]
    if task.get("task") != TEXT_SAFETY_TASK:
        raise ValueError(f"Wrong task for {example.example_id}: {task.get('task')}")
    if task.get("labels") != ["safe", "unsafe"]:
        raise ValueError(f"Wrong labels for {example.example_id}: {task.get('labels')}")
    if task.get("true_label") != [example.safety_label]:
        raise ValueError(
            f"Wrong true label for {example.example_id}: {task.get('true_label')}"
        )
    metadata = payload["metadata"]
    if metadata.get("example_id") != example.example_id:
        raise ValueError(f"Wrong metadata example_id for {example.example_id}")
    if metadata.get("record_uid") != example.record_uid:
        raise ValueError(f"Wrong metadata record_uid for {example.example_id}")


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "contract": {
            "task": TEXT_SAFETY_TASK,
            "task_type": "single_label",
            "labels": ["safe", "unsafe"],
            "classification_blocks_per_example": 1,
            "extra_task_blocks": 0,
        },
        "splits": {},
    }
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        source = args.manifest_dir / f"{split}.jsonl"
        output = args.manifest_dir / f"{split}_gliner.jsonl"
        count = 0
        with (
            source.open("r", encoding="utf-8") as input_handle,
            output.open("w", encoding="utf-8", newline="\n") as output_handle,
        ):
            for line in input_handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                example = example_from_row(row)
                payload = gliner_payload(example)
                validate_gliner_payload(payload, example)
                output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                count += 1
        report["splits"][split] = {
            "examples": count,
            "source": str(source),
            "source_sha256": sha256_file(source),
            "gliner": str(output),
            "gliner_sha256": sha256_file(output),
        }
        print(f"{split}: exported={count:,} -> {output}", flush=True)

    report_path = args.manifest_dir / "gliner_export_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
