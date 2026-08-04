#!/usr/bin/env python3
"""Streaming audit for Nemotron Safety Guard v3 JSONL splits."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import re
from typing import Any


def clean_constant(_: str) -> None:
    return None


def atomic_categories(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def normalized_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip().casefold()


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def audit(root: pathlib.Path, sample_limit: int) -> dict[str, Any]:
    result: dict[str, Any] = {"splits": {}, "atomic_categories": {}, "combos": {}, "leakage": {}}
    ids_by_split: dict[str, set[str]] = {}
    texts_by_split: dict[str, set[str]] = {}
    global_atomic: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    global_combos: collections.Counter[str] = collections.Counter()
    samples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    for path in sorted(root.glob("*.jsonl")):
        split = "val" if path.stem == "valid" else path.stem
        counts = collections.Counter()
        label_pairs = collections.Counter()
        tags = collections.Counter()
        languages = collections.Counter()
        prompt_sources = collections.Counter()
        response_sources = collections.Counter()
        nulls = collections.Counter()
        lengths_prompt: list[int] = []
        lengths_response: list[int] = []
        ids, texts = set(), set()
        duplicate_ids = duplicate_texts = invalid = mojibake = redacted = 0

        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    row = json.loads(line, parse_constant=clean_constant)
                except (json.JSONDecodeError, ValueError):
                    invalid += 1
                    continue
                counts["rows"] += 1
                for key, value in row.items():
                    if value is None or value == "":
                        nulls[key] += 1
                pid = str(row.get("id") or "")
                if pid in ids:
                    duplicate_ids += 1
                ids.add(pid)
                prompt = row.get("prompt") if isinstance(row.get("prompt"), str) else ""
                response = row.get("response") if isinstance(row.get("response"), str) else ""
                joined = normalized_text(prompt) + "\n" + normalized_text(response)
                digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
                if digest in texts:
                    duplicate_texts += 1
                texts.add(digest)
                if "REDACTED" in prompt or "REDACTED" in response:
                    redacted += 1
                if any(mark in prompt or mark in response for mark in ("â€™", "â€œ", "â€", "Ã", "Â")):
                    mojibake += 1
                lengths_prompt.append(len(prompt))
                if response:
                    lengths_response.append(len(response))
                pl = str(row.get("prompt_label") if row.get("prompt_label") is not None else "<null>")
                rl_raw = row.get("response_label")
                rl = "<null>" if rl_raw is None else ("<empty>" if rl_raw == "" else str(rl_raw))
                label_pairs[f"{pl} → {rl}"] += 1
                tags[str(row.get("tag") or "<missing>")] += 1
                languages[str(row.get("language") or "<missing>")] += 1
                prompt_sources[str(row.get("prompt_label_source") or "<missing>")] += 1
                response_sources[str(row.get("response_label_source") or "<missing>")] += 1
                cats = atomic_categories(row.get("violated_categories"))
                combo = " + ".join(sorted(set(cats))) if cats else "<none>"
                global_combos[combo] += 1
                for cat in set(cats):
                    global_atomic[cat][split] += 1
                    if len(samples[cat]) < sample_limit:
                        samples[cat].append({
                            "split": split, "id": pid, "prompt_label": pl, "response_label": rl,
                            "prompt": prompt[:700], "response": response[:700],
                            "all_categories": row.get("violated_categories") or "",
                        })

        ids_by_split[split], texts_by_split[split] = ids, texts
        result["splits"][split] = {
            "file": str(path), "bytes": path.stat().st_size, "rows": counts["rows"], "invalid_json": invalid,
            "duplicate_ids_within": duplicate_ids, "duplicate_prompt_response_within": duplicate_texts,
            "redacted_rows": redacted, "suspected_mojibake_rows": mojibake,
            "label_pairs": dict(label_pairs.most_common()), "tags": dict(tags.most_common()),
            "languages": dict(languages.most_common()), "prompt_label_sources": dict(prompt_sources.most_common()),
            "response_label_sources": dict(response_sources.most_common()), "missing_or_empty": dict(nulls.most_common()),
            "prompt_chars": {"min": min(lengths_prompt, default=0), "p50": percentile(lengths_prompt, .5), "p95": percentile(lengths_prompt, .95), "max": max(lengths_prompt, default=0)},
            "response_chars": {"min": min(lengths_response, default=0), "p50": percentile(lengths_response, .5), "p95": percentile(lengths_response, .95), "max": max(lengths_response, default=0)},
        }

    split_names = sorted(ids_by_split)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            result["leakage"][f"{left}__{right}"] = {
                "shared_ids": len(ids_by_split[left] & ids_by_split[right]),
                "shared_exact_prompt_response": len(texts_by_split[left] & texts_by_split[right]),
            }
    for cat, by_split in sorted(global_atomic.items(), key=lambda item: (-sum(item[1].values()), item[0])):
        result["atomic_categories"][cat] = {"total": sum(by_split.values()), **dict(by_split), "samples": samples[cat]}
    result["combos"] = dict(global_combos.most_common())
    result["summary"] = {
        "rows": sum(v["rows"] for v in result["splits"].values()),
        "atomic_category_count": len(result["atomic_categories"]),
        "category_combination_count": len(result["combos"]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, default=pathlib.Path("data/raw/nemotron_safety_guard_v3/en"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("reports/nemotron_safety_audit.json"))
    parser.add_argument("--samples-per-category", type=int, default=5)
    args = parser.parse_args()
    data = audit(args.input, args.samples_per_category)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
