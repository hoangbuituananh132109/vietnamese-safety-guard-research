from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FULL_SOURCES = (
    ("nemotron_valid", ROOT / "data" / "guard_full" / "valid.jsonl"),
    ("nemotron_test", ROOT / "data" / "guard_full" / "test.jsonl"),
    ("sea_paired", ROOT / "data" / "benchmarks" / "sea_safeguard" / "paired_en_vi.jsonl"),
)
SMOKE_SOURCES = (
    ("nemotron_test", ROOT / "data" / "decoder_baseline_smoke" / "nemotron_test.jsonl"),
    ("sea_paired", ROOT / "data" / "decoder_baseline_smoke" / "sea_paired.jsonl"),
)


def build(output: Path, sources: tuple[tuple[str, Path], ...]) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts: dict[str, int] = {}
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for benchmark, source in sources:
            count = 0
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    example_id = str(row["example_id"])
                    if example_id in seen:
                        raise AssertionError(f"Duplicate example_id across matrix: {example_id}")
                    seen.add(example_id)
                    row["decoder_benchmark"] = benchmark
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
            counts[benchmark] = count
    audit = {
        "output": str(output),
        "examples": len(seen),
        "benchmarks": counts,
        "sources": {name: str(path) for name, path in sources},
    }
    output.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine full or smoke decoder manifests while retaining benchmark identity."
    )
    parser.add_argument("--mode", choices=("full", "smoke", "both"), default="both")
    args = parser.parse_args()
    results: list[dict[str, Any]] = []
    if args.mode in {"full", "both"}:
        results.append(
            build(ROOT / "data" / "decoder_baseline_full" / "all_three.jsonl", FULL_SOURCES)
        )
        results.append(
            build(
                ROOT / "data" / "decoder_baseline_full" / "test_and_sea.jsonl",
                tuple(item for item in FULL_SOURCES if item[0] != "nemotron_valid"),
            )
        )
    if args.mode in {"smoke", "both"}:
        results.append(
            build(ROOT / "data" / "decoder_baseline_full" / "smoke.jsonl", SMOKE_SOURCES)
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
