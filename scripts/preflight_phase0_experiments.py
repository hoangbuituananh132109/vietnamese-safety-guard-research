from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase0_experiments.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/phase0_experiment_preflight.json")
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def scan_canonical(
    path: Path, allowed_views: set[str] | None = None
) -> dict[str, Any]:
    allowed_views = allowed_views or {"P", "R", "PR"}
    ids: set[str] = set()
    pair_keys: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for row in iter_jsonl(path):
        example_id = str(row["example_id"])
        if example_id in ids:
            raise ValueError(f"Duplicate example_id in {path}: {example_id}")
        ids.add(example_id)
        label = str(row["safety_label"])
        if label not in {"safe", "unsafe"}:
            raise ValueError(f"Invalid label in {path}: {label}")
        if row["view"] not in allowed_views:
            raise ValueError(f"Invalid view in {path}: {row['view']}")
        expected_scope = "prompt" if row["view"] == "P" else "response"
        if row["safety_scope"] != expected_scope:
            raise ValueError(f"Scope/view mismatch in {path}: {example_id}")
        counts["examples"] += 1
        counts[f"language:{row['language']}"] += 1
        counts[f"view:{row['view']}"] += 1
        counts[f"label:{label}"] += 1
        counts[f"tag:{row['tag']}"] += 1
        pair_keys.add((str(row["record_uid"]), str(row["view"])))
    return {"ids": ids, "pair_keys": pair_keys, "counts": dict(sorted(counts.items()))}


def scan_gliner(
    path: Path, allowed_views: set[str] | None = None
) -> dict[str, Any]:
    allowed_views = allowed_views or {"P", "R", "PR"}
    ids: set[str] = set()
    counts: Counter[str] = Counter()
    for row in iter_jsonl(path):
        if set(row) != {"input", "output", "metadata"}:
            raise ValueError(f"Unexpected GLi record keys in {path}: {row.keys()}")
        example_id = str(row["metadata"]["example_id"])
        if example_id in ids:
            raise ValueError(f"Duplicate GLi example_id in {path}: {example_id}")
        ids.add(example_id)
        output = row["output"]
        if set(output) != {"classifications"} or len(output["classifications"]) != 1:
            raise ValueError(f"GLi record must contain exactly one classification: {example_id}")
        task = output["classifications"][0]
        if task.get("task") != "text safety classification":
            raise ValueError(f"Wrong GLi task: {example_id}")
        if task.get("labels") != ["safe", "unsafe"]:
            raise ValueError(f"Wrong GLi labels: {example_id}")
        true_label = task.get("true_label")
        if not isinstance(true_label, list) or len(true_label) != 1:
            raise ValueError(f"GLi record is not single-label: {example_id}")
        view = str(row["metadata"].get("view") or "")
        if view not in allowed_views:
            raise ValueError(f"Invalid GLi view in {path}: {view!r}")
        counts["examples"] += 1
        counts[f"view:{view}"] += 1
        counts[f"label:{true_label[0]}"] += 1
    return {"ids": ids, "counts": dict(sorted(counts.items()))}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    allowed_views = set(
        str(value) for value in config.get("contract", {}).get(
            "allowed_views", ["P", "R", "PR"]
        )
    )
    failures: list[str] = []
    warnings: list[str] = []
    required_paths: set[Path] = set()
    for value in config["benchmarks"].values():
        if isinstance(value, dict):
            required_paths.update(
                Path(item)
                for item in value.values()
                if isinstance(item, str) and item.endswith(".jsonl")
            )
        elif isinstance(value, str) and value.endswith(".jsonl"):
            required_paths.add(Path(value))
    for run in config["runs"]:
        for key in (
            "base_model", "train_manifest", "valid_manifest", "canonical_valid", "test_manifest"
        ):
            if run.get(key):
                required_paths.add(Path(run[key]))
    missing = sorted(str(path) for path in required_paths if not path.exists())
    if missing:
        failures.extend(f"missing:{path}" for path in missing)

    scans: dict[str, Any] = {}
    canonical_paths = {
        Path(run[key])
        for run in config["runs"]
        for key in ("train_manifest", "valid_manifest", "test_manifest", "canonical_valid")
        if run.get(key) and not str(run[key]).endswith("_gliner.jsonl")
    }
    benchmark_paths = {
        Path("data/benchmarks/sea_safeguard/official_en_vi.jsonl"),
        Path("data/benchmarks/sea_safeguard/paired_en_vi.jsonl"),
    }
    canonical_paths -= benchmark_paths
    for path in sorted(canonical_paths):
        if path.exists():
            result = scan_canonical(path, allowed_views)
            scans[str(path)] = {"type": "canonical", "counts": result["counts"]}

    e1_run = next(
        (run for run in config["runs"] if run.get("model_kind") == "gliguard_schema"),
        None,
    )
    common_dir = (
        Path(e1_run["canonical_valid"]).parent
        if e1_run and e1_run.get("canonical_valid")
        else Path("data/guard_phase0_gliguard_native_512")
    )
    common_contract: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        canonical_path = common_dir / f"{split}.jsonl"
        gliner_path = common_dir / f"{split}_gliner.jsonl"
        if canonical_path.exists() and gliner_path.exists():
            canonical = scan_canonical(canonical_path, allowed_views)
            gliner = scan_gliner(gliner_path, allowed_views)
            equal = canonical["ids"] == gliner["ids"]
            if not equal:
                failures.append(
                    f"e1_e2_native_id_mismatch:{split}:canonical_only="
                    f"{len(canonical['ids'] - gliner['ids'])}:gliner_only="
                    f"{len(gliner['ids'] - canonical['ids'])}"
                )
            common_contract[split] = {
                "canonical_examples": len(canonical["ids"]),
                "gliner_examples": len(gliner["ids"]),
                "exact_same_ids": equal,
            }

    e3_run = next(
        (
            run
            for run in config["runs"]
            if str(run.get("legacy_id") or run.get("id", "")).startswith("E3")
        ),
        None,
    )
    e4_run = next(
        (
            run
            for run in config["runs"]
            if str(run.get("legacy_id") or run.get("id", "")).startswith("E4")
        ),
        None,
    )
    # The E3/E4 parity contract only applies to configs that actually define
    # both comparison runs.  Standalone ablations such as E6 must not fall back
    # to legacy hard-coded manifests that are outside their declared contract.
    e3_e4_report: dict[str, Any] = {}
    if e3_run and e4_run:
        e3_path = Path(e3_run["train_manifest"])
        e4_path = Path(e4_run["train_manifest"])
        e3 = scan_canonical(e3_path, allowed_views)
        e4 = scan_canonical(e4_path, allowed_views)
        if len(e4["pair_keys"]) != e4["counts"]["examples"]:
            failures.append("E4 contains duplicate semantic (record_uid, view) keys")
        e3_e4_gap = abs(e3["counts"]["examples"] - e4["counts"]["examples"])
        if e3_e4_gap != 0:
            failures.append(f"E3/E4 semantic budgets must be identical: {e3_e4_gap}")
        e3_e4_report = {
            "E3_examples": e3["counts"]["examples"],
            "E4_examples": e4["counts"]["examples"],
            "absolute_gap": e3_e4_gap,
            "E4_en": e4["counts"].get("language:en", 0),
            "E4_vi": e4["counts"].get("language:vi", 0),
            "unique_semantic_keys": len(e4["pair_keys"]),
        }

    sea_counts: dict[str, int] = {}
    for name, path in (
        ("official", Path("data/benchmarks/sea_safeguard/official_en_vi.jsonl")),
        ("paired_language_instances", Path("data/benchmarks/sea_safeguard/paired_en_vi.jsonl")),
    ):
        if path.exists():
            sea_counts[name] = sum(1 for _ in iter_jsonl(path))
    if sea_counts.get("official") != 3430:
        failures.append(f"Unexpected bundled SEA official count: {sea_counts.get('official')}")
    if sea_counts.get("paired_language_instances") != 3680:
        failures.append(
            f"Unexpected bundled SEA paired count: {sea_counts.get('paired_language_instances')}"
        )

    gate = config.get("full_gpu_gate")
    fixed_ready = bool(gate.get("fixed_scalable_trainer")) if gate else True
    schema_ready = bool(gate.get("schema_scalable_trainer")) if gate else True
    gli_ready = bool(gate.get("gliguard_scalable_launcher")) if gate else True
    scalable_ready = fixed_ready and schema_ready and gli_ready
    context_ready = bool(gate.get("target_32gb_context_profile")) if gate else True
    if not scalable_ready:
        warnings.append("paid_full_run_blocked: one or more scalable launchers are incomplete")
    if not context_ready:
        warnings.append("paid_8k_run_blocked: target 32GB GPU profile is not complete")
    if not gate or not gate.get("checkpoint_resume_gliguard_exact"):
        warnings.append(
            "E1 resume limitation: GLiNER2 warm-restores adapter weights but not optimizer/scheduler/global step"
        )

    report = {
        "status": "failed" if failures else "data_preflight_passed",
        "config": str(args.config),
        "runs": [run["id"] for run in config["runs"]],
        "missing_paths": missing,
        "e1_e2_native_contract": common_contract,
        "E3_E4": e3_e4_report,
        "sea_bundled_counts": sea_counts,
        "failures": failures,
        "warnings": warnings,
        "paid_full_run_authorized_by_preflight": bool(
            not failures and scalable_ready and context_ready
        ),
        "scans": scans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "status", "e1_e2_native_contract", "E3_E4", "sea_bundled_counts",
        "failures", "warnings", "paid_full_run_authorized_by_preflight"
    )}, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
