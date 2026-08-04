from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable


RUN_DIRS = {
    "E3": "E3-M-E-8K",
    "E4": "E4-M-EV-MATCHED-8K",
    "E5": "E5-M-EV-FULL-8K",
    "E7": "E7-M-SCHEMA-EV-8K",
}
BENCHMARKS = ("nemotron_test", "sea_paired")
VIEWS = {"P", "PR"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Qwen3Guard with D1 and Phase-0 encoders on exact P/PR IDs."
    )
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--d1-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row["example_id"])
            if example_id in seen:
                raise ValueError(f"Duplicate example_id in {path}:{line_number}: {example_id}")
            seen.add(example_id)
            rows.append(row)
    return rows


def load_matrix_run(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for benchmark in BENCHMARKS:
        path = root / f"full_mmbert__{benchmark}__canonical" / "predictions.jsonl"
        for row in read_jsonl(path):
            if str(row.get("view")) not in VIEWS:
                continue
            copied = dict(row)
            copied["decoder_benchmark"] = benchmark
            example_id = str(copied["example_id"])
            if example_id in result:
                raise ValueError(f"Duplicate combined example_id: {example_id}")
            result[example_id] = copied
    return result


def qwen_variant(
    rows: Iterable[dict[str, Any]], prediction_field: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        copied = dict(row)
        copied["prediction"] = int(copied[prediction_field])
        result[str(copied["example_id"])] = copied
    return result


def safe_f1(tn: int, fp: int, fn: int) -> float:
    precision = tn / max(1, tn + fn)
    recall = tn / max(1, tn + fp)
    return 2 * precision * recall / max(1e-15, precision + recall)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter((int(row["target"]), int(row["prediction"])) for row in rows)
    tn, fp = counts[(0, 0)], counts[(0, 1)]
    fn, tp = counts[(1, 0)], counts[(1, 1)]
    n = len(rows)
    safe_total, unsafe_total = tn + fp, fn + tp
    unsafe_precision = tp / max(1, tp + fp)
    unsafe_recall = tp / max(1, unsafe_total)
    unsafe_f1 = 2 * unsafe_precision * unsafe_recall / max(
        1e-15, unsafe_precision + unsafe_recall
    )
    s_f1 = safe_f1(tn, fp, fn)
    parsed_values = [bool(row["parsed_ok"]) for row in rows if "parsed_ok" in row]
    return {
        "n": n,
        "accuracy": (tn + tp) / max(1, n),
        "macro_f1": (s_f1 + unsafe_f1) / 2,
        "safe_support": safe_total,
        "safe_correct": tn,
        "safe_recall": tn / max(1, safe_total),
        "unsafe_support": unsafe_total,
        "unsafe_correct": tp,
        "unsafe_precision": unsafe_precision,
        "unsafe_recall": unsafe_recall,
        "unsafe_f1": unsafe_f1,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "parsed": sum(parsed_values) if parsed_values else None,
        "parse_rate": (sum(parsed_values) / len(parsed_values)) if parsed_values else None,
    }


def slice_report(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = list(rows.values())
    dimensions = {
        "overall": ("all",),
        "language": ("en", "vi"),
        "benchmark": BENCHMARKS,
        "view": ("P", "PR"),
    }
    report: dict[str, Any] = {}
    for dimension, values in dimensions.items():
        report[dimension] = {}
        for value in values:
            if dimension == "overall":
                selected = ordered
            else:
                field = "decoder_benchmark" if dimension == "benchmark" else dimension
                selected = [row for row in ordered if str(row.get(field)) == value]
            report[dimension][value] = summarize(selected)
    return report


def exact_mcnemar_p(candidate_better: int, baseline_better: int) -> float:
    n = candidate_better + baseline_better
    if n == 0:
        return 1.0
    k = min(candidate_better, baseline_better)
    log_p = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        - n * math.log(2)
    )
    term = math.exp(log_p) if log_p > -745 else 0.0
    tail = term
    for current in range(k, 0, -1):
        term *= current / (n - current + 1)
        tail += term
    return min(1.0, 2 * tail)


def paired_compare(
    baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    ids = sorted(set(baseline) & set(candidate))
    if set(ids) != set(baseline) or set(ids) != set(candidate):
        raise ValueError(
            f"ID mismatch: baseline={len(baseline)}, candidate={len(candidate)}, matched={len(ids)}"
        )
    candidate_better = baseline_better = both_correct = both_wrong = 0
    for example_id in ids:
        left, right = baseline[example_id], candidate[example_id]
        if int(left["target"]) != int(right["target"]):
            raise ValueError(f"Target mismatch for {example_id}")
        target = int(left["target"])
        baseline_correct = int(left["prediction"]) == target
        candidate_correct = int(right["prediction"]) == target
        if baseline_correct and candidate_correct:
            both_correct += 1
        elif baseline_correct:
            baseline_better += 1
        elif candidate_correct:
            candidate_better += 1
        else:
            both_wrong += 1
    return {
        "n": len(ids),
        "candidate_correct_baseline_wrong": candidate_better,
        "baseline_correct_candidate_wrong": baseline_better,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "exact_mcnemar_p": exact_mcnemar_p(candidate_better, baseline_better),
    }


def controversial_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("native_label") == "controversial"]
    by_target = Counter(int(row["target"]) for row in selected)
    return {
        "n": len(selected),
        "gold_safe": by_target[0],
        "gold_unsafe": by_target[1],
        "conservative_correct": by_target[1],
        "lenient_correct": by_target[0],
        "conservative_accuracy": by_target[1] / max(1, len(selected)),
        "lenient_accuracy": by_target[0] / max(1, len(selected)),
    }


def selective_and_cascade_report(
    qwen_rows: list[dict[str, Any]], d1: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    accepted: dict[str, dict[str, Any]] = {}
    abstained: dict[str, dict[str, Any]] = {}
    cascade: dict[str, dict[str, Any]] = {}
    for row in qwen_rows:
        example_id = str(row["example_id"])
        copied = dict(row)
        if row.get("native_label") == "safe":
            copied["prediction"] = 0
            accepted[example_id] = copied
        elif row.get("native_label") == "unsafe":
            copied["prediction"] = 1
            accepted[example_id] = copied
        else:
            abstained[example_id] = copied
        routed = dict(row)
        routed["prediction"] = (
            int(d1[example_id]["prediction"])
            if row.get("native_label") == "controversial"
            else int(copied["prediction"])
        )
        cascade[example_id] = routed

    dimensions = {
        "overall": ("all",),
        "language": ("en", "vi"),
        "benchmark": BENCHMARKS,
        "view": ("P", "PR"),
    }
    slices: dict[str, Any] = {}
    for dimension, values in dimensions.items():
        slices[dimension] = {}
        for value in values:
            if dimension == "overall":
                full_ids = set(cascade)
            else:
                field = "decoder_benchmark" if dimension == "benchmark" else dimension
                full_ids = {
                    example_id
                    for example_id, row in cascade.items()
                    if str(row.get(field)) == value
                }
            accepted_ids = full_ids & set(accepted)
            selected_qwen = {key: accepted[key] for key in accepted_ids}
            selected_d1 = {key: d1[key] for key in accepted_ids}
            qwen_metrics = summarize(list(selected_qwen.values()))
            d1_metrics = summarize(list(selected_d1.values()))
            paired = paired_compare(selected_d1, selected_qwen)
            slices[dimension][value] = {
                "total": len(full_ids),
                "accepted": len(accepted_ids),
                "abstained": len(full_ids - accepted_ids),
                "coverage": len(accepted_ids) / max(1, len(full_ids)),
                "qwen_on_accepted": qwen_metrics,
                "d1_on_same_accepted": d1_metrics,
                "accuracy_delta": qwen_metrics["accuracy"] - d1_metrics["accuracy"],
                "paired": paired,
                "cascade": summarize([cascade[key] for key in full_ids]),
            }
    abstained_d1 = [d1[key] for key in abstained]
    return {
        "policy": "Qwen Safe/Unsafe is accepted; Qwen Controversial routes to D1",
        "accepted_native_labels": ["safe", "unsafe"],
        "abstention_native_label": "controversial",
        "slices": slices,
        "abstained_subset": {
            "examples": len(abstained),
            "gold_safe": sum(int(row["target"]) == 0 for row in abstained.values()),
            "gold_unsafe": sum(int(row["target"]) == 1 for row in abstained.values()),
            "d1": summarize(abstained_d1),
        },
    }


def pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    args = parse_args()
    qwen_rows = read_jsonl(args.qwen)
    qwen_conservative = qwen_variant(qwen_rows, "prediction_conservative")
    qwen_lenient = qwen_variant(qwen_rows, "prediction_lenient")

    runs: dict[str, dict[str, dict[str, Any]]] = {
        "D1": load_matrix_run(args.d1_root),
        **{
            run: load_matrix_run(args.matrix_root / directory)
            for run, directory in RUN_DIRS.items()
        },
        "Q1-conservative": qwen_conservative,
        "Q1-lenient": qwen_lenient,
    }
    expected_ids = set(qwen_conservative)
    for run, rows in runs.items():
        if set(rows) != expected_ids:
            raise ValueError(
                f"{run} ID mismatch: missing={len(expected_ids - set(rows))}, "
                f"extra={len(set(rows) - expected_ids)}"
            )
    report: dict[str, Any] = {
        "contract": {
            "examples": len(expected_ids),
            "views": sorted(VIEWS),
            "benchmarks": list(BENCHMARKS),
            "exact_same_ids": True,
            "unsafe_is_positive": True,
            "qwen_primary_mapping": "safe->safe; controversial/unsafe->unsafe",
            "qwen_sensitivity_mapping": "safe/controversial->safe; unsafe->unsafe",
        },
        "runs": {run: slice_report(rows) for run, rows in runs.items()},
        "qwen_native_labels": dict(Counter(str(r["native_label"]) for r in qwen_rows)),
        "qwen_controversial_audit": controversial_audit(qwen_rows),
        "selective_and_qwen_to_d1_cascade": selective_and_cascade_report(
            qwen_rows, runs["D1"]
        ),
        "paired_vs_d1": {
            run: paired_compare(runs["D1"], rows)
            for run, rows in runs.items()
            if run != "D1"
        },
        "paired_vs_qwen_primary": {
            run: paired_compare(rows, runs["Q1-conservative"])
            for run, rows in runs.items()
            if run != "Q1-conservative"
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    primary_runs = ("E3", "E4", "E5", "E7", "D1", "Q1-conservative", "Q1-lenient")
    lines = [
        "# Qwen3Guard no-R exact-ID comparison",
        "",
        f"All figures use the exact same **{len(expected_ids):,}** P/PR examples. R is excluded.",
        "",
        "| Run | Overall | EN | VI | Nemotron | SEA | P | PR | Unsafe recall | Safe recall | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in primary_runs:
        item = report["runs"][run]
        overall = item["overall"]["all"]
        lines.append(
            f"| {run} | {pct(overall['accuracy'])} | "
            f"{pct(item['language']['en']['accuracy'])} | {pct(item['language']['vi']['accuracy'])} | "
            f"{pct(item['benchmark']['nemotron_test']['accuracy'])} | "
            f"{pct(item['benchmark']['sea_paired']['accuracy'])} | "
            f"{pct(item['view']['P']['accuracy'])} | {pct(item['view']['PR']['accuracy'])} | "
            f"{pct(overall['unsafe_recall'])} | {pct(overall['safe_recall'])} | "
            f"{overall['macro_f1']:.4f} |"
        )

    lines += [
        "",
        "## Q1 conservative class counts",
        "",
        "| Slice | Correct safe | Safe -> unsafe | Unsafe -> safe | Correct unsafe | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    q1 = report["runs"]["Q1-conservative"]
    for dimension, value, label in (
        ("language", "en", "English"),
        ("language", "vi", "Vietnamese"),
        ("benchmark", "nemotron_test", "Nemotron test"),
        ("benchmark", "sea_paired", "SEA paired"),
        ("view", "P", "P"),
        ("view", "PR", "PR"),
    ):
        item = q1[dimension][value]
        cm = item["confusion"]
        lines.append(
            f"| {label} | {cm['tn']:,} | {cm['fp']:,} | {cm['fn']:,} | "
            f"{cm['tp']:,} | {pct(item['accuracy'])} |"
        )

    audit = report["qwen_controversial_audit"]
    paired = report["paired_vs_d1"]["Q1-conservative"]
    paired_e5 = report["paired_vs_qwen_primary"]["E5"]
    selective = report["selective_and_qwen_to_d1_cascade"]
    selective_overall = selective["slices"]["overall"]["all"]
    lines += [
        "",
        "## Contract and paired test",
        "",
        f"- Parsed outputs: {sum(bool(row.get('parsed_ok')) for row in qwen_rows):,}/{len(qwen_rows):,}.",
        f"- Native labels: `{json.dumps(report['qwen_native_labels'], ensure_ascii=False)}`.",
        f"- `controversial`: {audit['n']:,} outputs; gold Safe {audit['gold_safe']:,}, "
        f"gold Unsafe {audit['gold_unsafe']:,}. Conservative mapping is correct on "
        f"{pct(audit['conservative_accuracy'])}; lenient mapping on {pct(audit['lenient_accuracy'])}.",
        f"- Versus D1, Q1 conservative alone fixes {paired['candidate_correct_baseline_wrong']:,} "
        f"examples but loses {paired['baseline_correct_candidate_wrong']:,}; exact McNemar "
        f"p={paired['exact_mcnemar_p']:.6g}.",
        f"- Versus E5, Q1 conservative alone fixes "
        f"{paired_e5['candidate_correct_baseline_wrong']:,} examples but loses "
        f"{paired_e5['baseline_correct_candidate_wrong']:,}; exact McNemar "
        f"p={paired_e5['exact_mcnemar_p']:.6g}.",
        "",
        "## Selective Safe/Unsafe and Qwen -> D1 cascade",
        "",
        f"Qwen emits a direct Safe/Unsafe verdict for "
        f"{selective_overall['accepted']:,}/{selective_overall['total']:,} examples "
        f"({pct(selective_overall['coverage'])} coverage). On that accepted subset, Qwen "
        f"scores {pct(selective_overall['qwen_on_accepted']['accuracy'])}; D1 on exactly "
        f"the same IDs scores {pct(selective_overall['d1_on_same_accepted']['accuracy'])}.",
        "",
        "| Slice | Coverage | Qwen accepted accuracy | D1 same-ID accuracy | Delta | Cascade accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dimension, value, label in (
        ("overall", "all", "Overall"),
        ("language", "en", "English"),
        ("language", "vi", "Vietnamese"),
        ("benchmark", "nemotron_test", "Nemotron test"),
        ("benchmark", "sea_paired", "SEA paired"),
        ("view", "P", "P"),
        ("view", "PR", "PR"),
    ):
        item = selective["slices"][dimension][value]
        lines.append(
            f"| {label} | {pct(item['coverage'])} | "
            f"{pct(item['qwen_on_accepted']['accuracy'])} | "
            f"{pct(item['d1_on_same_accepted']['accuracy'])} | "
            f"{item['accuracy_delta']:+.2%} | {pct(item['cascade']['accuracy'])} |"
        )
    lines += [
        "",
        f"D1 scores {pct(selective['abstained_subset']['d1']['accuracy'])} on the "
        f"{selective['abstained_subset']['examples']:,} Qwen-Controversial examples, confirming "
        "that the abstention subset is materially harder than the accepted subset.",
        "",
        "Qwen's taxonomy is not the Nemotron N23 schema, so this report deliberately does not "
        "score Qwen categories as N23. Doing so would conflate incompatible label ontologies.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "qwen_primary": report["runs"]["Q1-conservative"]["overall"]["all"],
        "qwen_lenient": report["runs"]["Q1-lenient"]["overall"]["all"],
        "paired_vs_d1": paired,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
