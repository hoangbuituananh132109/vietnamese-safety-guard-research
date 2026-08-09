from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "final_luna_sol_pure_v1"
DEFAULT_OUTPUT = ROOT / "exports" / "hf_release" / "nemotron_safety_guard_vi_luna_sol"
SPLITS = {
    "train": "nemotron_train_en_vi_v10_final.jsonl",
    "validation": "nemotron_valid_en_vi_v10_final.jsonl",
    "test": "nemotron_test_en_vi_v10_final.jsonl",
}
PRIVATE_OPERATIONAL_FIELDS = {
    "translation_api_key_slot",
    "translation_batch_id",
    "translated_at",
    "usage_metadata",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_split(source: Path, target: Path) -> dict:
    count = 0
    labels = Counter()
    providers = Counter()
    fields = set()
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as src, target.open("w", encoding="utf-8", newline="\n") as dst:
        for line_number, line in enumerate(src, 1):
            row = json.loads(line)
            for key in PRIVATE_OPERATIONAL_FIELDS:
                row.pop(key, None)
            if row.get("language") != "vi":
                raise ValueError(f"{source}:{line_number}: expected language=vi")
            uid = row.get("record_uid")
            if not isinstance(uid, str) or not uid:
                raise ValueError(f"{source}:{line_number}: missing record_uid")
            fields.update(row)
            labels[str(row.get("prompt_label"))] += 1
            providers[str(row.get("translation_provider"))] += 1
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return {
        "records": count,
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
        "prompt_labels": dict(sorted(labels.items())),
        "translation_providers": dict(sorted(providers.items())),
        "fields": sorted(fields),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = {}
    for split, filename in SPLITS.items():
        results[split] = sanitize_split(args.source / filename, args.output / "data" / f"{split}.jsonl")

    card_source = ROOT / "release" / "datasets" / "nemotron_safety_guard_vi_luna_sol" / "README.md"
    (args.output / "README.md").write_text(card_source.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "prepared_at": str(date.today()),
        "source_dataset": "nvidia/Nemotron-Safety-Guard-Dataset-v3",
        "source_license": "CC-BY-4.0",
        "adaptation": "English-to-Vietnamese machine translation with Luna/Sol",
        "removed_private_operational_fields": sorted(PRIVATE_OPERATIONAL_FIELDS),
        "splits": results,
        "total_records": sum(item["records"] for item in results.values()),
    }
    (args.output / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "output": str(args.output), "records": manifest["total_records"]}, indent=2))


if __name__ == "__main__":
    main()
