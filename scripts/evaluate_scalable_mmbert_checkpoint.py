from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

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
from transformers import AutoModel, AutoTokenizer

from guard_smoke.data import GuardExample
from guard_train.mmbert_fixed import (
    ExampleDataset,
    FixedMultiTaskCollator,
    FixedMultiTaskGuard,
    FixedTrainingConfig,
    _precision,
    evaluate,
    prepare_fixed_examples,
)
from guard_train.mmbert_schema import (
    DynamicSchemaCodec,
    DynamicSchemaCollator,
    DynamicSchemaGuard,
    SchemaTrainingConfig,
    evaluate_schema,
    prepare_schema_examples,
)
from guard_train.n23_metrics import build_n23_report
from scripts.evaluate_mmbert_checkpoint import build_report, length_bucket, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=("fixed", "schema"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="The run's final directory")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--label-order", choices=("canonical", "reversed"), default="canonical")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def to_example(row: dict[str, Any]) -> GuardExample:
    allowed = {field.name for field in fields(GuardExample)}
    if allowed.issubset(row):
        payload = {key: row[key] for key in allowed}
        payload["categories"] = tuple(payload.get("categories") or ())
        example = GuardExample(**payload)
        example.validate()
        return example
    # SEA paired/official benchmark shape.
    pair_uid = str(row.get("pair_uid") or row["example_id"].rsplit(":", 1)[0])
    example = GuardExample(
        example_id=str(row["example_id"]),
        record_uid=pair_uid,
        source_split=str(row.get("benchmark") or "benchmark"),
        source_id=str(row.get("source_path") or row["example_id"]),
        tag=str(row.get("subset") or "benchmark"),
        language=str(row["language"]),
        view=str(row["view"]),
        safety_scope=str(row["safety_scope"]),
        text=str(row["text"]),
        safety_label=str(row["safety_label"]),
        categories=(),
        category_scope="unavailable",
        normalized_prompt_sha256=_hash(str(row.get("prompt") or row["text"])),
    )
    example.validate()
    return example


def load_examples(path: Path, limit: int | None) -> list[GuardExample]:
    rows = iter_jsonl(path)
    examples: list[GuardExample] = []
    for row in rows:
        examples.append(to_example(row))
        if limit is not None and len(examples) >= limit:
            break
    return examples


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = json.loads((args.checkpoint / "run_contract.json").read_text(encoding="utf-8"))
    raw_examples = load_examples(args.manifest, args.limit)
    if args.model_kind == "fixed":
        saved = contract["config"]
        eval_config = FixedTrainingConfig(
            model_path=str(args.base_model),
            max_length=args.max_length,
            mixed_precision=args.precision,
            enable_categories=bool(saved["enable_categories"]),
        )
        amp_enabled, amp_dtype = _precision(eval_config, device)
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint / "tokenizer", local_files_only=True, fix_mistral_regex=False
        )
        examples, lengths, truncation = prepare_fixed_examples(
            raw_examples, tokenizer, args.max_length
        )
        base = AutoModel.from_pretrained(str(args.base_model), local_files_only=True)
        encoder = PeftModel.from_pretrained(base, args.checkpoint / "adapter", is_trainable=False)
        model = FixedMultiTaskGuard(
            encoder,
            int(base.config.hidden_size),
            enable_categories=eval_config.enable_categories,
            category_loss_weight=1.0,
        )
        heads = torch.load(args.checkpoint / "heads.pt", map_location="cpu", weights_only=True)
        model.safety_head.load_state_dict(heads["safety_head"])
        if model.category_head is not None:
            model.category_head.load_state_dict(heads["category_head"])
        model = model.to(device)
        metrics = evaluate(
            model,
            ExampleDataset(examples, lengths),
            FixedMultiTaskCollator(tokenizer, args.max_length, eval_config.enable_categories),
            device,
            batch_size=args.batch_size,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
    else:
        saved = contract["config"]
        eval_config = SchemaTrainingConfig(
            model_path=str(args.base_model),
            max_length=args.max_length,
            mixed_precision=args.precision,
            enable_categories=bool(saved["enable_categories"]),
        )
        amp_enabled, amp_dtype = _precision(eval_config, device)
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint / "tokenizer", local_files_only=True, fix_mistral_regex=False
        )
        codec = DynamicSchemaCodec(tokenizer, args.max_length)
        examples, lengths, truncation = prepare_schema_examples(raw_examples, codec)
        base = AutoModel.from_pretrained(str(args.base_model), local_files_only=True)
        base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        encoder = PeftModel.from_pretrained(base, args.checkpoint / "adapter", is_trainable=False)
        model = DynamicSchemaGuard(encoder, int(base.config.hidden_size), 1.0)
        model.classifier.load_state_dict(
            torch.load(
                args.checkpoint / "shared_label_mlp.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        model = model.to(device)
        order = ("safe", "unsafe") if args.label_order == "canonical" else ("unsafe", "safe")
        metrics = evaluate_schema(
            model,
            ExampleDataset(examples, lengths),
            DynamicSchemaCollator(
                codec,
                eval_config,
                training=False,
                binary_order_override=order,
            ),
            device,
            batch_size=args.batch_size,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

    predictions = metrics.pop("predictions")
    for row in predictions:
        row["length_bucket"] = length_bucket(int(row["serialized_tokens"]))
        row["model_kind"] = args.model_kind
        row["label_order_eval"] = args.label_order
    binary_report = build_report(predictions)
    n23_report = build_n23_report(predictions)
    report = {
        "model_kind": args.model_kind,
        "checkpoint": str(args.checkpoint),
        "base_model": str(args.base_model),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "examples": len(predictions),
        "max_length": args.max_length,
        "label_order": args.label_order,
        "truncated_examples": len(truncation),
        "binary": binary_report,
        "N23": n23_report or metrics.get("N23"),
        "mean_loss": metrics.get("loss"),
    }
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_jsonl(args.output_dir / "truncation_audit.jsonl", truncation)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "examples": len(predictions),
                "truncated": len(truncation),
                "binary": binary_report["overall"],
                "N23": report["N23"],
                "output": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
