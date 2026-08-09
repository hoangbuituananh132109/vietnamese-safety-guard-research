from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "clean_experiments"
FIGURES = ROOT / "reports" / "figures"

NO_R = (
    ROOT
    / "reports/analysis_20260724/no_r/reports/no_r_phase0/evaluation_matrix"
)
E6 = (
    ROOT
    / "reports/analysis_20260724/e6/reports/e6_no_r_binary_ablation/evaluation_matrix"
    / "E6NR-M-EV-FULL-BIN-8K"
)
Q1_PATH = ROOT / "reports/qwen3guard/Q1-QWEN3GUARD-GEN-4B-ZS-NR/metrics.json"
Q2_PATH = (
    ROOT
    / "reports/vast_download/d3_nemotron_no_r_4080s_20260724/extracted/results"
    / "no_r_decoder_4080s/eval/qwen/metrics.json"
)
D3_PATH = (
    ROOT
    / "reports/vast_download/d3_nemotron_no_r_4080s_20260724/extracted/results"
    / "no_r_decoder_4080s/eval/nemotron/metrics.json"
)
D3_PREDICTIONS = D3_PATH.with_name("predictions.jsonl")
D12_PATH = ROOT / "reports/research_archive/D2_VS_D1_NO_R_20260723.json"
D1_PREDICTIONS = (
    ROOT
    / "reports/vast_download/d1_nemotron_guard_8b_v3_20260722"
    / "decoder_baseline/full_combined/predictions.jsonl"
)

BLUE = "#3569c8"
GOLD = "#c6922e"
ORANGE = "#d66b32"
INK = "#172033"
MUTED = "#667085"
GRID = "#d9deea"
PALE = "#eef3fb"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def binary_overall(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["binary"]["overall"]


def macro_f1(confusion: dict[str, int]) -> float:
    tn, fp = confusion["tn"], confusion["fp"]
    fn, tp = confusion["fn"], confusion["tp"]
    safe_den = 2 * tn + fp + fn
    unsafe_den = 2 * tp + fp + fn
    safe_f1 = 2 * tn / safe_den if safe_den else 0.0
    unsafe_f1 = 2 * tp / unsafe_den if unsafe_den else 0.0
    return (safe_f1 + unsafe_f1) / 2


def metric_row(run: str, benchmark: str, path: Path, contract: str) -> dict[str, Any]:
    payload = load(path)
    overall = binary_overall(payload)
    languages = payload["binary"]["slices"]["language"]
    paired = payload["binary"].get("paired_en_vi") or {}
    return {
        "run": run,
        "benchmark": benchmark,
        "contract": contract,
        "examples": payload["examples"],
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "safe_recall": overall["safe_recall"],
        "unsafe_recall": overall["unsafe_recall"],
        "en_accuracy": languages["en"]["accuracy"],
        "vi_accuracy": languages["vi"]["accuracy"],
        "vi_minus_en": languages["vi"]["accuracy"] - languages["en"]["accuracy"],
        "en_vi_agreement": paired.get("same_decision_rate"),
        "truncated_examples": payload.get("truncated_examples"),
    }


def encoder_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    primary_runs = {
        "E1NR": "E1NR-G-EV-512",
        "E2NR": "E2NR-M-EV-GLI-COMPAT-8K",
    }
    full_runs = {
        "E3NR": "E3NR-M-E-8K",
        "E4NR": "E4NR-M-EV-MATCHED-8K",
        "E5NR": "E5NR-M-EV-FULL-8K",
        "E7NR": "E7NR-M-SCHEMA-EV-8K",
    }
    for short, run in primary_runs.items():
        for bench, suffix in (
            ("Nemotron", "e1_e2_primary_native__nemotron_test__canonical"),
            ("SEA", "e1_e2_primary_native__sea_paired__canonical"),
        ):
            rows.append(
                metric_row(short, bench, NO_R / run / suffix / "metrics.json", "primary-native")
            )
    for short, run in full_runs.items():
        for bench, suffix in (
            ("Nemotron", "full_mmbert__nemotron_test__canonical"),
            ("SEA", "full_mmbert__sea_paired__canonical"),
        ):
            rows.append(metric_row(short, bench, NO_R / run / suffix / "metrics.json", "full-8K"))
    for bench, suffix in (
        ("Nemotron", "full_mmbert__nemotron_test__canonical"),
        ("SEA", "full_mmbert__sea_paired__canonical"),
    ):
        rows.append(metric_row("E6NR", bench, E6 / suffix / "metrics.json", "full-8K"))
    order = {name: index for index, name in enumerate(("E1NR", "E2NR", "E3NR", "E4NR", "E5NR", "E6NR", "E7NR"))}
    return sorted(rows, key=lambda row: (order[row["run"]], row["benchmark"]))


def qwen_row(name: str, payload: dict[str, Any], branch: str) -> dict[str, Any]:
    metrics = payload[branch]["metrics"]
    overall = metrics["overall"]
    languages = metrics["slices"]["language"]
    benchmarks = metrics["slices"]["benchmark"]
    views = metrics["slices"]["view"]
    paired = metrics.get("paired_en_vi") or {}
    return {
        "run": name,
        "examples": overall["examples"],
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "safe_recall": overall["safe_recall"],
        "unsafe_recall": overall["unsafe_recall"],
        "en_accuracy": languages["en"]["accuracy"],
        "vi_accuracy": languages["vi"]["accuracy"],
        "nemotron_accuracy": benchmarks["nemotron_test"]["accuracy"],
        "sea_accuracy": benchmarks["sea_paired"]["accuracy"],
        "p_accuracy": views["P"]["accuracy"],
        "pr_accuracy": views["PR"]["accuracy"],
        "en_vi_agreement": paired.get("same_decision_rate"),
    }


def nemotron_row(name: str, source: dict[str, Any]) -> dict[str, Any]:
    overall = source["slices"]["overall"]["all"][name.lower()]
    languages = source["slices"]["language"]
    benchmarks = source["slices"]["benchmark"]
    views = source["slices"]["view"]
    return {
        "run": name,
        "examples": overall["n"],
        "accuracy": overall["accuracy"],
        "macro_f1": macro_f1(overall["confusion"]),
        "safe_recall": overall["safe_recall"],
        "unsafe_recall": overall["unsafe_recall"],
        "en_accuracy": languages["en"][name.lower()]["accuracy"],
        "vi_accuracy": languages["vi"][name.lower()]["accuracy"],
        "nemotron_accuracy": benchmarks["nemotron_test"][name.lower()]["accuracy"],
        "sea_accuracy": benchmarks["sea_paired"][name.lower()]["accuracy"],
        "p_accuracy": views["P"][name.lower()]["accuracy"],
        "pr_accuracy": views["PR"][name.lower()]["accuracy"],
        "en_vi_agreement": None,
    }


def prediction_slice_accuracy(path: Path, field: str, value: str) -> float:
    correct = total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get(field)) != value:
                continue
            total += 1
            correct += int(row.get("prediction") == row.get("target"))
    if not total:
        raise ValueError(f"No rows for {field}={value} in {path}")
    return correct / total


def d3_row(payload: dict[str, Any]) -> dict[str, Any]:
    overall = payload["binary"]["overall"]
    slices = payload["binary"]["slices"]
    return {
        "run": "D3",
        "examples": overall["examples"],
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "safe_recall": overall["safe_recall"],
        "unsafe_recall": overall["unsafe_recall"],
        "en_accuracy": slices["language"]["en"]["accuracy"],
        "vi_accuracy": slices["language"]["vi"]["accuracy"],
        "nemotron_accuracy": prediction_slice_accuracy(D3_PREDICTIONS, "decoder_benchmark", "nemotron_test"),
        "sea_accuracy": prediction_slice_accuracy(D3_PREDICTIONS, "decoder_benchmark", "sea_paired"),
        "p_accuracy": slices["view"]["P"]["accuracy"],
        "pr_accuracy": slices["view"]["PR"]["accuracy"],
        "en_vi_agreement": payload["binary"]["paired_en_vi"]["same_decision_rate"],
    }


def prediction_agreement(path: Path) -> float:
    groups: dict[tuple[str, str, str], dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("view") not in {"P", "PR"}:
                continue
            if row.get("decoder_benchmark") not in {"nemotron_test", "sea_paired"}:
                continue
            key = (str(row["record_uid"]), str(row["view"]), str(row["decoder_benchmark"]))
            groups.setdefault(key, {})[str(row["language"])] = int(row["prediction"])
    pairs = [value for value in groups.values() if {"en", "vi"} <= value.keys()]
    return sum(value["en"] == value["vi"] for value in pairs) / len(pairs)


def decoder_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    q1 = load(Q1_PATH)
    q2 = load(Q2_PATH)
    d3 = load(D3_PATH)
    d12 = load(D12_PATH)
    rows = [
        qwen_row("Q1", q1, "binary_primary_conservative"),
        qwen_row("Q2", q2, "binary_primary_conservative"),
        nemotron_row("D1", d12),
        nemotron_row("D2", d12),
        d3_row(d3),
    ]
    rows[2]["en_vi_agreement"] = prediction_agreement(D1_PREDICTIONS)
    qwen = {
        "native_labels_q1": q1["native_labels"],
        "native_labels_q2": q2["native_labels"],
        "q1_conservative": qwen_row("Q1 conservative", q1, "binary_primary_conservative"),
        "q1_lenient": qwen_row("Q1 lenient", q1, "binary_sensitivity_lenient"),
        "q2": qwen_row("Q2 binary LoRA", q2, "binary_primary_conservative"),
    }
    return rows, qwen


def n23_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    e5_test = load(NO_R / "E5NR-M-EV-FULL-8K/full_mmbert__nemotron_test__canonical/metrics.json")
    e5_valid = load(NO_R / "E5NR-M-EV-FULL-8K/full_mmbert__nemotron_valid__canonical/metrics.json")
    d3 = load(D3_PATH)["N23"]
    split_rows = []
    for split, path in (
        ("train", ROOT / "data/no_r/full/train.jsonl"),
        ("valid", ROOT / "data/no_r/full/valid.jsonl"),
        ("test", ROOT / "data/no_r/full/test.jsonl"),
    ):
        total = supervised = positives = duplicates = 0
        support: dict[str, int] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                total += 1
                if row.get("category_scope") not in {"prompt", "interaction"}:
                    continue
                supervised += 1
                labels = list(row.get("categories") or [])
                unique = set(labels)
                duplicates += len(labels) - len(unique)
                positives += len(unique)
                for label in unique:
                    support[label] = support.get(label, 0) + 1
        split_rows.append(
            {
                "split": split,
                "total_examples": total,
                "n23_supervised_examples": supervised,
                "gold_positive_labels": positives,
                "duplicate_labels_removed_by_multihot": duplicates,
            }
        )
    per_label = [
        {"label": label, **values}
        for label, values in e5_test["N23"]["per_label"].items()
    ]
    per_label.sort(key=lambda row: (-row["f1"], -row["support"], row["label"]))
    aggregate = [
        {
            "model": "E5NR",
            "exact_match": e5_test["N23"]["exact_match_accuracy"],
            "hamming_error": e5_test["N23"]["hamming_error_rate"],
            "micro_f1": e5_test["N23"]["micro_f1"],
            "macro_f1": e5_test["N23"]["macro_f1"],
            "supervised_examples": e5_test["N23"]["supervised_examples"],
        },
        {"model": "D1", "exact_match": 0.5231, "hamming_error": 0.04017, "micro_f1": 0.5783, "macro_f1": 0.4469, "supervised_examples": 5768},
        {"model": "D2", "exact_match": 0.5173, "hamming_error": 0.03978, "micro_f1": 0.5820, "macro_f1": 0.4749, "supervised_examples": 5768},
        {
            "model": "D3",
            "exact_match": d3["exact_match_accuracy"],
            "hamming_error": d3["hamming_error_rate"],
            "micro_f1": d3["micro_f1"],
            "macro_f1": d3["macro_f1"],
            "supervised_examples": d3["supervised_examples"],
        },
    ]
    # Preserve valid metrics in the summary for split-level audit.
    split_rows[1].update(
        evaluation_micro_f1=e5_valid["N23"]["micro_f1"],
        evaluation_macro_f1=e5_valid["N23"]["macro_f1"],
        evaluation_exact_match=e5_valid["N23"]["exact_match_accuracy"],
    )
    split_rows[2].update(
        evaluation_micro_f1=e5_test["N23"]["micro_f1"],
        evaluation_macro_f1=e5_test["N23"]["macro_f1"],
        evaluation_exact_match=e5_test["N23"]["exact_match_accuracy"],
    )
    return split_rows, per_label, aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#172033}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#667085}.label{font-size:14px}.small{font-size:12px;fill:#667085}.value{font-size:12px;font-weight:600}.axis{stroke:#aeb7c7;stroke-width:1}.grid{stroke:#e5e8ef;stroke-width:1}</style>',
        f'<text class="title" x="48" y="40">{html.escape(title)}</text>',
        f'<text class="sub" x="48" y="65">{html.escape(subtitle)}</text>',
    ]


def grouped_bars(path: Path, title: str, subtitle: str, rows: list[dict[str, Any]], names: list[str]) -> None:
    width = 1100
    height = 120 + len(names) * 58
    left, right, top = 150, 55, 92
    plot_width = width - left - right
    values = {(row["run"], row["benchmark"]): row["accuracy"] * 100 for row in rows}
    svg = svg_header(width, height, title, subtitle)
    for tick in range(0, 101, 20):
        x = left + plot_width * tick / 100
        svg.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-35}"/>')
        svg.append(f'<text class="small" text-anchor="middle" x="{x:.1f}" y="{height-15}">{tick}%</text>')
    for index, name in enumerate(names):
        y = top + index * 58
        svg.append(f'<text class="label" text-anchor="end" x="{left-12}" y="{y+24}">{name}</text>')
        for offset, (bench, color) in enumerate((("Nemotron", BLUE), ("SEA", GOLD))):
            value = values[(name, bench)]
            by = y + 4 + offset * 22
            bar_width = plot_width * value / 100
            svg.append(f'<rect x="{left}" y="{by}" width="{bar_width:.1f}" height="16" rx="3" fill="{color}"/>')
            svg.append(f'<text class="value" x="{left+bar_width+7:.1f}" y="{by+12}">{value:.2f}%</text>')
    svg.append(f'<rect x="{width-245}" y="27" width="13" height="13" fill="{BLUE}"/><text class="small" x="{width-225}" y="38">Nemotron</text>')
    svg.append(f'<rect x="{width-135}" y="27" width="13" height="13" fill="{GOLD}"/><text class="small" x="{width-115}" y="38">SEA</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def heatmap(path: Path, title: str, subtitle: str, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    width = 1180
    cell_w, cell_h = 145, 58
    left, top = 165, 118
    height = top + len(rows) * cell_h + 45
    svg = svg_header(width, height, title, subtitle)
    for col, (_, label) in enumerate(columns):
        x = left + col * cell_w + cell_w / 2
        svg.append(f'<text class="small" text-anchor="middle" x="{x}" y="{top-18}">{html.escape(label)}</text>')
    for r_index, row in enumerate(rows):
        y = top + r_index * cell_h
        svg.append(f'<text class="label" text-anchor="end" x="{left-15}" y="{y+35}">{row["run"]}</text>')
        for c_index, (field, _) in enumerate(columns):
            x = left + c_index * cell_w
            value = row.get(field)
            if value is None:
                color, text = "#f2f4f7", "n/a"
            else:
                pct = value * 100
                ratio = max(0.0, min(1.0, (pct - 45) / 50))
                base = (238, 243, 251)
                target = (53, 105, 200)
                rgb = tuple(round(base[i] + ratio * (target[i] - base[i])) for i in range(3))
                color = "#%02x%02x%02x" % rgb
                text = f"{pct:.2f}%"
            svg.append(f'<rect x="{x+3}" y="{y+3}" width="{cell_w-6}" height="{cell_h-6}" rx="5" fill="{color}" stroke="#ffffff"/>')
            text_color = "#ffffff" if value is not None and value >= 0.75 else INK
            svg.append(f'<text class="value" text-anchor="middle" x="{x+cell_w/2}" y="{y+35}" fill="{text_color}" style="fill:{text_color}">{text}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def qwen_chart(path: Path, qwen: dict[str, Any]) -> None:
    rows = [qwen["q1_conservative"], qwen["q1_lenient"], qwen["q2"]]
    fields = [("accuracy", "Accuracy"), ("safe_recall", "Safe recall"), ("unsafe_recall", "Unsafe recall")]
    width, height = 1080, 430
    left, top, plot_h = 105, 110, 230
    group_w = 270
    colors = [GOLD, "#e1c78e", BLUE]
    svg = svg_header(width, height, "Qwen before and after LoRA", "Same 11,736 P/PR examples; Q1 mapping changes the Safe/Unsafe trade-off")
    for tick in range(0, 101, 20):
        y = top + plot_h * (1 - tick / 100)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-45}" y2="{y:.1f}"/>')
        svg.append(f'<text class="small" text-anchor="end" x="{left-10}" y="{y+4:.1f}">{tick}%</text>')
    for f_index, (field, label) in enumerate(fields):
        gx = left + 80 + f_index * group_w
        svg.append(f'<text class="label" text-anchor="middle" x="{gx+70}" y="{top+plot_h+30}">{label}</text>')
        for r_index, row in enumerate(rows):
            value = row[field] * 100
            x = gx + r_index * 48
            y = top + plot_h * (1 - value / 100)
            h = top + plot_h - y
            svg.append(f'<rect x="{x}" y="{y:.1f}" width="34" height="{h:.1f}" rx="3" fill="{colors[r_index]}"/>')
            svg.append(f'<text class="small" text-anchor="middle" x="{x+17}" y="{y-6:.1f}">{value:.1f}</text>')
    labels = ["Q1 conservative", "Q1 lenient", "Q2 LoRA binary"]
    for i, label in enumerate(labels):
        x = 210 + i * 250
        svg.append(f'<rect x="{x}" y="385" width="13" height="13" fill="{colors[i]}"/><text class="small" x="{x+20}" y="396">{label}</text>')
    q1 = qwen["native_labels_q1"]
    svg.append(f'<text class="small" x="48" y="86">Q1 native outputs: Safe {q1["safe"]:,} · Controversial {q1["controversial"]:,} · Unsafe {q1["unsafe"]:,}; Q2 emits only Safe/Unsafe.</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def n23_label_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    width, row_h = 1280, 34
    height = 120 + len(rows) * row_h
    left, right, top = 330, 125, 90
    plot_width = width - left - right
    svg = svg_header(width, height, "E5NR N23 F1 by category", "Nemotron test, 5,768 supervised examples; labels sorted by F1, support shown at right")
    for tick in (0, 0.2, 0.4, 0.6, 0.8):
        x = left + plot_width * tick / 0.8
        svg.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-30}"/>')
        svg.append(f'<text class="small" text-anchor="middle" x="{x:.1f}" y="{height-12}">{tick:.1f}</text>')
    for index, row in enumerate(rows):
        y = top + index * row_h
        label = html.escape(row["label"])
        f1 = row["f1"]
        bar_w = plot_width * f1 / 0.8
        color = BLUE if f1 >= 0.2 else GOLD if f1 > 0 else "#c9ced8"
        svg.append(f'<text class="small" text-anchor="end" x="{left-12}" y="{y+20}">{label}</text>')
        svg.append(f'<rect x="{left}" y="{y+6}" width="{bar_w:.1f}" height="17" rx="3" fill="{color}"/>')
        svg.append(f'<text class="value" x="{left+bar_w+7:.1f}" y="{y+19}">{f1:.3f}</text>')
        svg.append(f'<text class="small" x="{width-right+22}" y="{y+19}">n={row["support"]}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def n23_models_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    transformed = []
    for row in rows:
        transformed.append(
            {
                "run": row["model"],
                "Exact match": row["exact_match"],
                "Micro-F1": row["micro_f1"],
                "Macro-F1": row["macro_f1"],
            }
        )
    heatmap(
        path,
        "N23 aggregate comparison",
        "Nemotron test, exact same 5,768 supervised P/PR examples",
        transformed,
        [("Exact match", "Exact match"), ("Micro-F1", "Micro-F1"), ("Macro-F1", "Macro-F1")],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    encoders = encoder_rows()
    decoders, qwen = decoder_rows()
    n23_splits, n23_labels, n23_models = n23_data()

    write_csv(OUT / "encoder_metrics.csv", encoders)
    write_csv(OUT / "decoder_metrics.csv", decoders)
    write_csv(OUT / "n23_split_support.csv", n23_splits)
    write_csv(OUT / "n23_e5_per_label.csv", n23_labels)
    write_csv(OUT / "n23_model_comparison.csv", n23_models)

    summary = {
        "schema_version": 1,
        "generated_from": "immutable local clean P/PR metric artifacts",
        "encoders": encoders,
        "decoders": decoders,
        "qwen_before_after": qwen,
        "n23_splits": n23_splits,
        "n23_e5_per_label": n23_labels,
        "n23_models": n23_models,
        "data_quality_notes": [
            "E1/E2 primary-native and E3-E7 full-8K populations are charted separately.",
            "N23 support is multi-hot support; repeated labels inside one row are deduplicated.",
            "Six duplicate test labels explain the raw 6,972 versus evaluated 6,966 positive-label difference.",
            "D2 is retained as a historical comparator because its pilot training included R.",
        ],
    }
    (OUT / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    grouped_bars(
        FIGURES / "encoder_primary_e1_e2.svg",
        "GLiGuard versus mmBERT on the primary-native contract",
        "E1/E2 only; 6,676 Nemotron and 3,146 SEA examples",
        [row for row in encoders if row["run"] in {"E1NR", "E2NR"}],
        ["E1NR", "E2NR"],
    )
    grouped_bars(
        FIGURES / "encoder_full_suite.svg",
        "Clean P/PR encoder accuracy",
        "Full-8K contract; 8,056 Nemotron and 3,680 SEA examples",
        [row for row in encoders if row["run"] not in {"E1NR", "E2NR"}],
        ["E3NR", "E4NR", "E5NR", "E6NR", "E7NR"],
    )
    nem_rows = [row for row in encoders if row["benchmark"] == "Nemotron" and row["run"] not in {"E1NR", "E2NR"}]
    sea_by_run = {row["run"]: row for row in encoders if row["benchmark"] == "SEA"}
    language_rows = [
        {
            "run": row["run"],
            "nem_en": row["en_accuracy"],
            "nem_vi": row["vi_accuracy"],
            "sea_en": sea_by_run[row["run"]]["en_accuracy"],
            "sea_vi": sea_by_run[row["run"]]["vi_accuracy"],
            "nem_agreement": row["en_vi_agreement"],
            "sea_agreement": sea_by_run[row["run"]]["en_vi_agreement"],
        }
        for row in nem_rows
    ]
    heatmap(
        FIGURES / "encoder_language_and_agreement.svg",
        "Encoder accuracy by language and EN–VI agreement",
        "Full-8K P/PR contract; agreement means same binary decision for paired EN/VI inputs",
        language_rows,
        [("nem_en", "Nem EN"), ("nem_vi", "Nem VI"), ("sea_en", "SEA EN"), ("sea_vi", "SEA VI"), ("nem_agreement", "Nem agreement"), ("sea_agreement", "SEA agreement")],
    )
    heatmap(
        FIGURES / "decoder_clean_comparison.svg",
        "Decoder performance on the exact common P/PR set",
        "11,736 examples; D2 is a mixed-view historical training comparator",
        decoders,
        [("accuracy", "Accuracy"), ("en_accuracy", "EN accuracy"), ("vi_accuracy", "VI accuracy"), ("safe_recall", "Safe recall"), ("unsafe_recall", "Unsafe recall"), ("en_vi_agreement", "EN–VI agree")],
    )
    qwen_chart(FIGURES / "qwen_before_after.svg", qwen)
    n23_label_chart(FIGURES / "n23_e5_per_label.svg", n23_labels)
    n23_models_chart(FIGURES / "n23_model_comparison.svg", n23_models)
    print(json.dumps({"status": "ok", "output": str(OUT), "figures": 7}, indent=2))


if __name__ == "__main__":
    main()
