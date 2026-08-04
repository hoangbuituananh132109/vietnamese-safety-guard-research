from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare D1 and D2 on exactly matched examples.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("P", "R", "PR"),
        help="Optional view filter, for example '--views P PR' for a no-R comparison.",
    )
    return parser.parse_args()


def read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter((int(row["target"]), int(row["prediction"])) for row in rows)
    tn, fp = counts[(0, 0)], counts[(0, 1)]
    fn, tp = counts[(1, 0)], counts[(1, 1)]
    total = len(rows)
    safe_total, unsafe_total = tn + fp, tp + fn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, unsafe_total)
    return {
        "n": total,
        "accuracy": (tn + tp) / max(1, total),
        "safe_correct": tn,
        "safe_total": safe_total,
        "safe_recall": tn / max(1, safe_total),
        "unsafe_correct": tp,
        "unsafe_total": unsafe_total,
        "unsafe_recall": recall,
        "unsafe_precision": precision,
        "unsafe_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "parsed_ok": sum(bool(row.get("parsed_ok")) for row in rows),
        "parsed_rate": sum(bool(row.get("parsed_ok")) for row in rows) / max(1, total),
        "strict_json": sum(bool(row.get("strict_json")) for row in rows),
        "strict_json_rate": sum(bool(row.get("strict_json")) for row in rows) / max(1, total),
    }


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def main() -> None:
    args = parse_args()
    allowed_views = set(args.views) if args.views else None
    baseline_all = {
        str(row["example_id"]): row
        for row in read(args.baseline)
        if allowed_views is None or str(row.get("view")) in allowed_views
    }
    candidate_all = {
        str(row["example_id"]): row
        for row in read(args.candidate)
        if allowed_views is None or str(row.get("view")) in allowed_views
    }
    ids = sorted(set(baseline_all) & set(candidate_all))
    if not ids:
        raise ValueError("No matched D1/D2 predictions")
    baseline = [baseline_all[key] for key in ids]
    candidate = [candidate_all[key] for key in ids]
    if any(int(a["target"]) != int(b["target"]) for a, b in zip(baseline, candidate)):
        raise AssertionError("Target mismatch between D1 and D2")

    dimensions = {
        "overall": ["all"],
        "language": sorted({str(row["language"]) for row in candidate}),
        "benchmark": sorted({str(row.get("decoder_benchmark")) for row in candidate}),
        "view": sorted({str(row["view"]) for row in candidate}),
    }
    report: dict[str, Any] = {
        "matched_examples": len(ids),
        "view_filter": sorted(allowed_views) if allowed_views is not None else None,
        "slices": {},
    }
    for dimension, values in dimensions.items():
        report["slices"][dimension] = {}
        for value in values:
            if dimension == "overall":
                base_rows, candidate_rows = baseline, candidate
            else:
                field = "decoder_benchmark" if dimension == "benchmark" else dimension
                selected = [
                    index for index, row in enumerate(candidate) if str(row.get(field)) == value
                ]
                base_rows = [baseline[index] for index in selected]
                candidate_rows = [candidate[index] for index in selected]
            base_metrics = metrics(base_rows)
            candidate_metrics = metrics(candidate_rows)
            report["slices"][dimension][value] = {
                "d1": base_metrics,
                "d2": candidate_metrics,
                "delta": {
                    key: candidate_metrics[key] - base_metrics[key]
                    for key in (
                        "accuracy",
                        "safe_recall",
                        "unsafe_recall",
                        "unsafe_precision",
                        "unsafe_f1",
                        "parsed_rate",
                        "strict_json_rate",
                    )
                },
            }

    b = c = both_correct = both_wrong = 0
    for base_row, candidate_row in zip(baseline, candidate):
        target = int(base_row["target"])
        base_correct = int(base_row["prediction"]) == target
        candidate_correct = int(candidate_row["prediction"]) == target
        if base_correct and candidate_correct:
            both_correct += 1
        elif base_correct:
            c += 1
        elif candidate_correct:
            b += 1
        else:
            both_wrong += 1
    report["paired_test"] = {
        "d2_correct_d1_wrong": b,
        "d1_correct_d2_wrong": c,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "exact_mcnemar_p": exact_mcnemar_p(b, c),
    }
    vi = report["slices"]["language"].get("vi", {})
    en = report["slices"]["language"].get("en", {})
    report["headline"] = {
        "vi_accuracy_delta": vi.get("delta", {}).get("accuracy"),
        "vi_unsafe_recall_delta": vi.get("delta", {}).get("unsafe_recall"),
        "en_accuracy_delta": en.get("delta", {}).get("accuracy"),
        "candidate_improves_vi_accuracy": (vi.get("delta", {}).get("accuracy") or 0) > 0,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# D2 versus D1 matched evaluation",
        "",
        f"Matched examples: **{len(ids):,}**",
        "",
        "| Slice | N | D1 accuracy | D2 accuracy | Delta accuracy | D1 unsafe recall | D2 unsafe recall | Delta unsafe recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    slices = [("overall", "all")]
    slices += [("language", value) for value in dimensions["language"]]
    slices += [("benchmark", value) for value in dimensions["benchmark"]]
    slices += [("view", value) for value in dimensions["view"]]
    for dimension, value in slices:
        item = report["slices"][dimension][value]
        lines.append(
            f"| {dimension}:{value} | {item['d2']['n']:,} | "
            f"{item['d1']['accuracy']:.2%} | {item['d2']['accuracy']:.2%} | "
            f"{item['delta']['accuracy']:+.2%} | {item['d1']['unsafe_recall']:.2%} | "
            f"{item['d2']['unsafe_recall']:.2%} | {item['delta']['unsafe_recall']:+.2%} |"
        )
    lines += [
        "",
        "## Paired correctness",
        "",
        f"- D2 correct / D1 wrong: {b:,}",
        f"- D1 correct / D2 wrong: {c:,}",
        f"- Exact McNemar p: {report['paired_test']['exact_mcnemar_p']:.6g}",
        "",
        (
            "A positive Vietnamese delta is preliminary evidence for the translated-data pipeline; "
            "it is not by itself sufficient to claim a final Vietnamese benchmark contribution."
            if (report["headline"]["vi_accuracy_delta"] or 0) > 0
            else "This comparison does not show a Vietnamese accuracy gain on the selected views."
        ),
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["headline"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
