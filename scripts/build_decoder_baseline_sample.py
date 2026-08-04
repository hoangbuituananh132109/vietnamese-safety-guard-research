from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic paired EN/VI decoder smoke manifest.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-per-stratum", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def score(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    pairs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        record_uid = row.get("record_uid") or row.get("pair_uid") or str(row["example_id"]).rsplit(":", 1)[0]
        pairs[(str(record_uid), str(row["view"]))][str(row["language"])] = row

    strata: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key, languages in pairs.items():
        if set(languages) != {"en", "vi"}:
            continue
        english = languages["en"]
        vietnamese = languages["vi"]
        if english["safety_label"] != vietnamese["safety_label"]:
            raise AssertionError(f"Paired labels differ for {key}")
        strata[(str(english["view"]), str(english["safety_label"]))].append(key)

    selected: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"input": str(args.input), "strata": {}}
    for stratum, keys in sorted(strata.items()):
        ranked = sorted(keys, key=lambda key: score(args.seed, "|".join(key)))
        chosen = ranked[: args.pairs_per_stratum]
        for key in chosen:
            selected.extend((pairs[key]["en"], pairs[key]["vi"]))
        audit["strata"][f"{stratum[0]}:{stratum[1]}"] = {
            "available_pairs": len(keys),
            "selected_pairs": len(chosen),
        }

    selected.sort(key=lambda row: str(row["example_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit["examples"] = len(selected)
    audit["paired_examples"] = len(selected) // 2
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
