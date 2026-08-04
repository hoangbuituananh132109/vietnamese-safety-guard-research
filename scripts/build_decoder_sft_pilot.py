from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic paired EN/VI D2 train-only pilot manifest."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("P", "R", "PR"),
        default=("P", "R", "PR"),
        help="Views to stratify. Use '--views P PR' for the no-response-only contract.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.rows % 2:
        raise ValueError("--rows must be a positive even number")
    source_rows = read_jsonl(args.source)
    if any(str(row.get("source_split")) != "train" for row in source_rows):
        raise ValueError("D2 source must contain train-only rows")

    pairs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in source_rows:
        key = (str(row["record_uid"]), str(row["view"]))
        pairs[key][str(row["language"])] = row

    strata: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key, languages in pairs.items():
        if set(languages) != {"en", "vi"}:
            continue
        en = languages["en"]
        vi = languages["vi"]
        if str(en["safety_label"]) != str(vi["safety_label"]):
            raise AssertionError(f"EN/VI label mismatch for {key}")
        strata[(str(en["view"]), str(en["safety_label"]))].append(key)

    requested_views = tuple(dict.fromkeys(args.views))
    expected = [(view, label) for view in requested_views for label in ("safe", "unsafe")]
    missing = [key for key in expected if not strata[key]]
    if missing:
        raise ValueError(f"Missing sampling strata: {missing}")

    pair_target = args.rows // 2
    base, remainder = divmod(pair_target, len(expected))
    rng = random.Random(args.seed)
    selected_keys: list[tuple[str, str]] = []
    quota: dict[str, int] = {}
    for index, stratum in enumerate(expected):
        candidates = list(strata[stratum])
        rng.shuffle(candidates)
        take = base + (1 if index < remainder else 0)
        if take > len(candidates):
            raise ValueError(f"Not enough rows for {stratum}: need {take}, have {len(candidates)}")
        selected_keys.extend(candidates[:take])
        quota[f"{stratum[0]}:{stratum[1]}"] = take

    rng.shuffle(selected_keys)
    selected: list[dict[str, Any]] = []
    for key in selected_keys:
        selected.extend((pairs[key]["en"], pairs[key]["vi"]))

    if len(selected) != args.rows:
        raise AssertionError(f"Selected {len(selected)} rows, expected {args.rows}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(
        (str(row["language"]), str(row["view"]), str(row["safety_label"]))
        for row in selected
    )
    audit = {
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "source_rows": len(source_rows),
        "source_splits": sorted({str(row.get("source_split")) for row in source_rows}),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows": len(selected),
        "paired_keys": len(selected_keys),
        "seed": args.seed,
        "views": list(requested_views),
        "pair_quota": quota,
        "counts": {"|".join(key): value for key, value in sorted(counts.items())},
        "leakage_contract": "train-only source; no valid/test/SEA rows",
    }
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
