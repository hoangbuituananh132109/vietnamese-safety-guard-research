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
from peft import PeftModel

from guard_smoke.gliguard import audit_gliner_lengths, evaluate_gliguard
from scripts.evaluate_mmbert_checkpoint import build_report, length_bucket, sha256_file
from scripts.evaluate_scalable_mmbert_checkpoint import load_examples, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path("models/fastino_gliguard_300m"))
    parser.add_argument("--adapter", type=Path, help="Optional E1 PEFT adapter directory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-total-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from gliner2 import GLiNER2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = GLiNER2.from_pretrained(str(args.base_model))
    model = (
        PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
        if args.adapter is not None
        else base
    ).to(device)
    examples = load_examples(args.manifest, args.limit)
    length_audit = audit_gliner_lengths(model, examples, max_words=None)
    if length_audit["failed"]:
        raise RuntimeError(f"GLi length audit failed: {length_audit['failures'][:3]}")
    over = [
        row
        for row in length_audit["longest"]
        if int(row["encoder_subwords"]) > args.max_total_tokens
    ]
    # `longest` contains the global top 10. If its maximum fits, all rows fit.
    if over:
        raise ValueError(
            "Manifest is not pre-fitted for GLiGuard; use the native/shared-truncated "
            f"artifact. First over-limit rows: {over[:3]}"
        )
    metrics = evaluate_gliguard(
        model,
        examples,
        batch_size=args.batch_size,
        max_words=args.max_total_tokens,
    )
    predictions = metrics.pop("predictions")
    lengths_by_id = {
        row["example_id"]: row
        for row in length_audit["longest"]
    }
    # The audit helper intentionally retains only the ten longest rows. Fill
    # exact per-row lengths once for prediction artifacts.
    for row, example in zip(predictions, sorted(examples, key=lambda item: item.example_id)):
        schema = model.create_schema().classification(
            "text safety classification", ["safe", "unsafe"]
        ).build()
        batch = model.processor.collate_fn_inference([(example.text, schema)], max_len=None)
        tokens = int(batch.original_lengths[0])
        row["serialized_tokens"] = tokens
        row["length_bucket"] = length_bucket(tokens)
        row["model_kind"] = "gliguard_schema"
        row["adapter"] = str(args.adapter) if args.adapter is not None else None
    report = {
        "model_kind": "gliguard_schema",
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "examples": len(predictions),
        "max_total_tokens": args.max_total_tokens,
        "binary": build_report(predictions),
        "length_audit": length_audit,
        "raw_evaluate_summary": metrics,
    }
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "examples": len(predictions),
                "binary": report["binary"]["overall"],
                "paired_en_vi": report["binary"]["paired_en_vi"],
                "output": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
