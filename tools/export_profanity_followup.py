from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator.checkpoint import load_checkpoint
from translator.dashboard import DashboardData
from translator.full_run import paths
from translator.jsonl_io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the final profanity-register follow-up set.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report = json.loads((root / "reports" / "final_quality" / "translation_quality_summary.json").read_text(encoding="utf-8"))
    failures = {
        item["record_uid"]: item
        for item in report["validator"]["hard_failures"]
        if item.get("disposition") == "known_profanity_caveat"
    }
    source_index = DashboardData(root).source_index()
    translations = {}
    for split in ("train", "valid", "test"):
        for group_index in range(5):
            translations.update(load_checkpoint(paths(root, split, group_index)["checkpoint"]))
    rows = []
    for uid in sorted(failures):
        group, split, source = source_index[uid]
        translated = translations[uid]
        rows.append({
            "record_uid": uid,
            "split": split,
            "group": group,
            "tag": source.get("tag"),
            "prompt_label": source.get("prompt_label"),
            "response_label": source.get("response_label"),
            "violated_categories": source.get("violated_categories"),
            "length_bucket": source.get("length_bucket"),
            "source_chars": source.get("source_chars"),
            "prompt_en": source.get("prompt"),
            "response_en": source.get("response"),
            "prompt_vi": translated.get("prompt_vi"),
            "response_vi": translated.get("response_vi"),
            "translation_status": translated.get("translation_status"),
            "hard_quality_errors": failures[uid].get("errors"),
            "recommended_action": "Compare the English profanity with the Vietnamese register; strengthen only if meaning/aggression was softened.",
        })
    expected = report["summary"]["followup_recommended_records"]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} follow-up records, found {len(rows)}")
    output = root / "reports" / "final_quality" / "profanity_followup_64.jsonl"
    write_jsonl(output, rows)
    print(json.dumps({"records": len(rows), "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
