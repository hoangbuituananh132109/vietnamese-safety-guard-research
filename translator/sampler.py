from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from .batching import length_bucket, source_chars
from .jsonl_io import read_jsonl, write_jsonl


PRIORITY = {
    "Profanity": 4, "Hate/Identity Hate": 4, "Harassment": 4, "Threat": 4,
    "Sexual": 4, "Violence": 4, "Criminal Planning/Confessions": 4,
    "Suicide and Self Harm": 4,
}
SECONDARY = {
    "Sexual (minor)": 2, "PII/Privacy": 2, "Controlled/Regulated Substances": 2,
    "Guns and Illegal Weapons": 2, "Manipulation": 2, "Illegal Activity": 2,
}
CELL_TARGETS = {
    ("normal", "generic"): 18, ("normal", "jailbreaking"): 12,
    ("near_tail", "generic"): 6, ("near_tail", "jailbreaking"): 4,
    ("tail", "generic"): 5, ("tail", "jailbreaking"): 3,
    ("high_tail", "generic"): 1, ("high_tail", "jailbreaking"): 1,
}
PAIR_MIN = {"unsafe→safe": 15, "unsafe→unsafe": 10, "safe→safe": 5, "prompt_only": 10}


def categories(row: dict[str, Any]) -> set[str]:
    return {x.strip() for x in (row.get("violated_categories") or "").split(",") if x.strip()}


def pair_name(row: dict[str, Any]) -> str:
    if row.get("response") is None:
        return "prompt_only"
    return f"{row.get('prompt_label')}→{row.get('response_label')}"


def source_ids(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for _, row, _ in read_jsonl(path):
            source_id = str(row.get("source_id") or row.get("id") or "")
            if source_id:
                excluded.add(source_id)
    return excluded


def prepare(source: Path, excluded_source_ids: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exact_seen: set[str] = set()
    id_seen: set[str] = set()
    excluded_source_ids = excluded_source_ids or set()
    for line_no, row, physical in read_jsonl(source):
        prompt, response = row.get("prompt") or "", row.get("response")
        if prompt == "REDACTED" or not prompt:
            continue
        digest = hashlib.sha256((prompt.strip().casefold() + "\n" + (response or "").strip().casefold()).encode("utf-8")).hexdigest()
        source_id = str(row.get("id") or "")
        if source_id and source_id in excluded_source_ids:
            continue
        if digest in exact_seen or source_id in id_seen:
            continue
        exact_seen.add(digest); id_seen.add(source_id)
        item = dict(row)
        item.update({
            "record_uid": f"en-train-{line_no:08d}-{hashlib.sha256(physical.encode('utf-8')).hexdigest()[:12]}",
            "source_split": "train", "source_line_number": line_no,
            "source_line_sha256": hashlib.sha256(physical.encode("utf-8")).hexdigest(),
            "source_id": source_id, "source_chars": source_chars(row),
            "length_bucket": length_bucket(source_chars(row)),
            "is_above_p95": source_chars(row) > 2342,
            "oversized_single_record": source_chars(row) > 25_000,
        })
        if item["length_bucket"] != "oversized":
            rows.append(item)
    return rows


def select(
    rows: list[dict[str, Any]],
    seed: int,
    cell_targets: dict[tuple[str, str], int] | None = None,
    pair_min: dict[str, int] | None = None,
    category_targets: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    cell_targets = cell_targets or CELL_TARGETS
    pair_min = pair_min or PAIR_MIN
    category_targets = category_targets or {**PRIORITY, **SECONDARY}
    rng = random.Random(seed)
    rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    selected_uids: set[str] = set()
    cat_counts: collections.Counter[str] = collections.Counter()
    pair_counts: collections.Counter[str] = collections.Counter()
    combo_counts: collections.Counter[str] = collections.Counter()
    target_cats = category_targets

    for (bucket, tag), needed in cell_targets.items():
        pool = [r for r in rows if r["length_bucket"] == bucket and r.get("tag") == tag and r["record_uid"] not in selected_uids]
        for _ in range(needed):
            def score(r: dict[str, Any]) -> tuple[float, float, float]:
                cats = categories(r); pair = pair_name(r); combo = "+".join(sorted(cats)) or "<none>"
                cat_need = sum(max(target_cats[c] - cat_counts[c], 0) * (3 if c in PRIORITY else 2) for c in cats if c in target_cats)
                # Label-pair controls often have no violated category, so give
                # their remaining quota enough weight to compete with unsafe
                # multi-label examples during the greedy selection.
                pair_need = max(pair_min.get(pair, 0) - pair_counts[pair], 0) * 25
                complexity = 1.5 if len(cats) >= 3 and sum(v >= 3 for v in combo_counts.values()) < 10 else 0
                diversity = 2 if combo_counts[combo] == 0 else (-5 if combo_counts[combo] >= 3 else 0)
                return cat_need + pair_need + complexity + diversity, rng.random(), -r["source_chars"]
            available = [r for r in pool if r["record_uid"] not in selected_uids]
            if not available:
                raise RuntimeError(f"Not enough candidates for {bucket}/{tag}")
            choice = max(available, key=score)
            selected.append(choice); selected_uids.add(choice["record_uid"])
            cat_counts.update(categories(choice)); pair_counts[pair_name(choice)] += 1
            combo_counts["+".join(sorted(categories(choice))) or "<none>"] += 1

    deficits = {c: n - cat_counts[c] for c, n in target_cats.items() if cat_counts[c] < n}
    pair_deficits = {p: n - pair_counts[p] for p, n in pair_min.items() if pair_counts[p] < n}
    if deficits or pair_deficits:
        raise RuntimeError(f"Sampler quota deficit categories={deficits}, pairs={pair_deficits}")
    return sorted(selected, key=lambda r: (list(cell_targets).index((r["length_bucket"], r["tag"])), r["source_line_number"]))


def manifest(rows: list[dict[str, Any]], source: Path, seed: int) -> dict[str, Any]:
    cats = collections.Counter(); pairs = collections.Counter(); tags = collections.Counter(); buckets = collections.Counter()
    for row in rows:
        cats.update(categories(row)); pairs[pair_name(row)] += 1; tags[row["tag"]] += 1; buckets[row["length_bucket"]] += 1
    return {
        "source_file": str(source), "selection_seed": seed, "pilot_size": len(rows),
        "p90_source_chars": 1695, "p95_source_chars": 2342, "p99_source_chars": 3772,
        "record_uids": [r["record_uid"] for r in rows], "length_buckets": buckets,
        "tags": tags, "label_pairs": pairs, "atomic_categories": cats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/nemotron_safety_guard_v3/en/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/pilot/nemotron_en_train_pilot_50_diverse_v2.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/pilot/nemotron_en_train_pilot_50_manifest_v2.json"))
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--multiplier", type=int, default=1, help="Scale the 50-record quota template (2 creates 100 records)")
    parser.add_argument("--exclude-pilot", type=Path, action="append", default=[], help="Exclude source IDs already present in another pilot; repeatable")
    args = parser.parse_args()
    if args.multiplier < 1:
        parser.error("--multiplier must be at least 1")
    excluded = source_ids(args.exclude_pilot)
    cells = {key: value * args.multiplier for key, value in CELL_TARGETS.items()}
    pairs = {key: value * args.multiplier for key, value in PAIR_MIN.items()}
    cat_targets = {key: value * args.multiplier for key, value in {**PRIORITY, **SECONDARY}.items()}
    chosen = select(prepare(args.source, excluded), args.seed, cells, pairs, cat_targets)
    write_jsonl(args.output, chosen)
    args.manifest.write_text(json.dumps(manifest(chosen, args.source, args.seed), ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(f"created {args.output} rows={len(chosen)}")


if __name__ == "__main__":
    main()
