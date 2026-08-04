#!/usr/bin/env python3
from pathlib import Path

from translator.jsonl_io import read_jsonl, write_jsonl


def main() -> None:
    base_path = Path("data/translated/pilot_50_vi_machine.jsonl")
    repair_path = Path("data/translated/pilot_6_repair_v5_vi_machine.jsonl")
    output_path = Path("data/translated/pilot_50_vi_machine_best.jsonl")
    base = [r for _, r, _ in read_jsonl(base_path)]
    repaired = {r["record_uid"]: r for _, r, _ in read_jsonl(repair_path)}
    merged = []
    for row in base:
        if row["record_uid"] in repaired:
            replacement = repaired[row["record_uid"]]
            replacement["qa_repair_history"] = [{
                "previous_prompt_version": row.get("translation_prompt_version"),
                "replacement_prompt_version": replacement.get("translation_prompt_version"),
                "reason": "manual_qa_major_or_retranslate",
            }]
            merged.append(replacement)
        else:
            merged.append(row)
    write_jsonl(output_path, merged)
    print(f"merged={len(merged)} replaced={len(repaired)} output={output_path}")


if __name__ == "__main__": main()
