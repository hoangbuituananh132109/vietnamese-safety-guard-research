from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_VIEWS = {"P", "PR"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_view(row: dict[str, Any]) -> str:
    return str(row.get("view") or (row.get("metadata") or {}).get("view") or "")


def row_language(row: dict[str, Any]) -> str:
    return str(row.get("language") or (row.get("metadata") or {}).get("language") or "")


def row_label(row: dict[str, Any]) -> str:
    if row.get("safety_label") is not None:
        return str(row["safety_label"])
    classifications = ((row.get("output") or {}).get("classifications") or [])
    values = classifications[0].get("true_label") if classifications else None
    return str(values[0]) if values else ""


def filter_manifest(source: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    source_rows = 0
    with source.open("r", encoding="utf-8") as source_handle, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as target_handle:
        for line in source_handle:
            if not line.strip():
                continue
            source_rows += 1
            row = json.loads(line)
            view = row_view(row)
            if view not in ALLOWED_VIEWS:
                counts[f"excluded_view:{view or 'missing'}"] += 1
                continue
            target_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["rows"] += 1
            counts[f"view:{view}"] += 1
            counts[f"language:{row_language(row)}"] += 1
            counts[f"label:{row_label(row)}"] += 1

    audit = {
        "contract": "P and PR only; response-only R is excluded",
        "source": str(source),
        "source_sha256": sha256(source),
        "source_rows": source_rows,
        "output": str(output),
        "output_sha256": sha256(output),
        "counts": dict(sorted(counts.items())),
    }
    output.with_suffix(output.suffix + ".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def combine_decoder(sources: list[tuple[str, Path]], output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for benchmark, source in sources:
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    view = row_view(row)
                    if view not in ALLOWED_VIEWS:
                        continue
                    example_id = str(row["example_id"])
                    if example_id in seen:
                        raise AssertionError(f"Duplicate decoder example_id: {example_id}")
                    seen.add(example_id)
                    row["decoder_benchmark"] = benchmark
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts["rows"] += 1
                    counts[f"benchmark:{benchmark}"] += 1
                    counts[f"view:{view}"] += 1
                    counts[f"language:{row_language(row)}"] += 1
    audit = {
        "contract": "P and PR only; response-only R is excluded",
        "output": str(output),
        "output_sha256": sha256(output),
        "counts": dict(sorted(counts.items())),
        "sources": {name: str(path) for name, path in sources},
    }
    output.with_suffix(output.suffix + ".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def audit_matched_e3_e4(e3_path: Path, e4_path: Path) -> dict[str, Any]:
    def keyed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
        values: dict[tuple[str, str], dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["record_uid"]), str(row["view"]))
                if key in values:
                    raise AssertionError(f"Duplicate semantic key in {path}: {key}")
                values[key] = row
        return values

    e3 = keyed(e3_path)
    e4 = keyed(e4_path)
    if set(e3) != set(e4):
        raise AssertionError(
            f"E3/E4 no-R semantic keys differ: E3-only={len(set(e3) - set(e4))}, "
            f"E4-only={len(set(e4) - set(e3))}"
        )
    label_mismatches = [
        key for key in e3 if str(e3[key]["safety_label"]) != str(e4[key]["safety_label"])
    ]
    if label_mismatches:
        raise AssertionError(f"E3/E4 label mismatches: {len(label_mismatches)}")
    counts = Counter(str(row["language"]) for row in e4.values())
    audit = {
        "contract": "E3 and E4 contain exactly the same (record_uid, view) semantic units; E4 selects one deterministic language per unit",
        "views": sorted({key[1] for key in e3}),
        "semantic_units": len(e3),
        "same_key_set": True,
        "label_mismatches": 0,
        "e3_languages": dict(sorted(Counter(str(row["language"]) for row in e3.values()).items())),
        "e4_languages": dict(sorted(counts.items())),
        "e3_sha256": sha256(e3_path),
        "e4_sha256": sha256(e4_path),
    }
    audit_path = e4_path.with_suffix(e4_path.suffix + ".matching_audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    jobs = [
        (
            ROOT / "data" / "guard_phase0_gliguard_native_512" / f"{split}{suffix}.jsonl",
            ROOT / "data" / "no_r" / "phase0_gliguard_native_512" / f"{split}{suffix}.jsonl",
        )
        for split in ("train", "valid", "test")
        for suffix in ("", "_gliner")
    ]
    jobs += [
        (
            ROOT / "data" / "guard_experiments_v2" / "e3_english_full" / "train.jsonl",
            ROOT / "data" / "no_r" / "e3_english_full" / "train.jsonl",
        ),
        (
            ROOT / "data" / "guard_experiments_v2" / "e4_ev_matched_full" / "train.jsonl",
            ROOT / "data" / "no_r" / "e4_ev_matched_full" / "train.jsonl",
        ),
    ]
    jobs += [
        (
            ROOT / "data" / "guard_full" / f"{split}.jsonl",
            ROOT / "data" / "no_r" / "full" / f"{split}.jsonl",
        )
        for split in ("train", "valid", "test")
    ]
    # E1/E2 have a second evaluation view where every example is truncated by
    # the same audited policy before it is sent to either encoder.  The original
    # manifests contain P/R/PR, so materialize explicit no-R copies rather than
    # accidentally evaluating a no-R checkpoint on response-only rows.
    jobs += [
        (
            ROOT
            / "data"
            / "eval_shared_gliguard_512"
            / f"nemotron_{split}_shared_truncated.jsonl",
            ROOT
            / "data"
            / "no_r"
            / "eval_shared_gliguard_512"
            / f"nemotron_{split}_shared_truncated.jsonl",
        )
        for split in ("valid", "test")
    ]
    audits = [filter_manifest(source, output) for source, output in jobs]
    audits.append(
        audit_matched_e3_e4(
            ROOT / "data" / "no_r" / "e3_english_full" / "train.jsonl",
            ROOT / "data" / "no_r" / "e4_ev_matched_full" / "train.jsonl",
        )
    )

    sea = ROOT / "data" / "benchmarks" / "sea_safeguard" / "paired_en_vi.jsonl"
    valid = ROOT / "data" / "no_r" / "full" / "valid.jsonl"
    test = ROOT / "data" / "no_r" / "full" / "test.jsonl"
    audits.append(
        combine_decoder(
            [("nemotron_valid", valid), ("nemotron_test", test), ("sea_paired", sea)],
            ROOT / "data" / "no_r" / "decoder" / "all_three.jsonl",
        )
    )
    audits.append(
        combine_decoder(
            [("nemotron_test", test), ("sea_paired", sea)],
            ROOT / "data" / "no_r" / "decoder" / "test_and_sea.jsonl",
        )
    )
    print(json.dumps(audits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
