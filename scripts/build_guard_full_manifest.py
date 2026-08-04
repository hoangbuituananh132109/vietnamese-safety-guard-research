from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guard_smoke.data import gliner_payload, iter_full_examples


def _csv_set(value: str) -> set[str] | None:
    if value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", type=Path, default=Path("data/final"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/guard_full"))
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--languages", default="all", help="all, en, vi, or en,vi")
    parser.add_argument("--views", default="all", help="all or comma list from P,R,PR")
    parser.add_argument("--tags", default="all", help="all, generic, jailbreaking")
    parser.add_argument("--count-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    languages = _csv_set(args.languages)
    views = _csv_set(args.views)
    tags = _csv_set(args.tags)
    summary: dict[str, object] = {
        "filters": {
            "languages": sorted(languages) if languages else "all",
            "views": sorted(views) if views else "all",
            "tags": sorted(tags) if tags else "all",
        },
        "splits": {},
    }
    if not args.count_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        counters: Counter[str] = Counter()
        manifest_handle = None
        gliner_handle = None
        try:
            if not args.count_only:
                manifest_handle = (args.output_dir / f"{split}.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                )
                gliner_handle = (args.output_dir / f"{split}_gliner.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                )
            for example in iter_full_examples(
                args.final_dir,
                split,
                languages=languages,
                views=views,
                tags=tags,
            ):
                counters["examples"] += 1
                counters[f"view:{example.view}"] += 1
                counters[f"scope:{example.safety_scope}"] += 1
                counters[f"language:{example.language}"] += 1
                counters[f"label:{example.safety_label}"] += 1
                counters[f"tag:{example.tag}"] += 1
                counters[f"cell:{example.view}|{example.safety_label}|{example.tag}|{example.language}"] += 1
                if manifest_handle is not None and gliner_handle is not None:
                    payload = asdict(example)
                    payload["categories"] = list(example.categories)
                    manifest_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    gliner_handle.write(
                        json.dumps(gliner_payload(example), ensure_ascii=False) + "\n"
                    )
        finally:
            if manifest_handle is not None:
                manifest_handle.close()
            if gliner_handle is not None:
                gliner_handle.close()
        summary["splits"][split] = dict(sorted(counters.items()))
        print(f"{split}: {counters['examples']:,} examples", flush=True)

    summary_path = (
        Path("reports/guard_full_count_summary.json")
        if args.count_only
        else args.output_dir / "manifest_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
