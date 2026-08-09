from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/luna_sol_dataset_v1/missing_luna_sol_candidates.jsonl"
PROMPT_PATH = ROOT / "configs/sol_web_fallback_repair_prompt.md"
BASE_WEB = ROOT / "web/sol_fallback_queue"
WEB_OUT = ROOT / "web/sol_missing_25"
DATA_OUT = ROOT / "data/sol_missing_25"
MAX_SOURCE_CHARS = 45_000
MAX_ITEMS = 4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def payload_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def batch_page(batch: dict[str, Any]) -> str:
    title = html.escape(batch["batch_id"])
    payload = payload_script(batch)
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · 25 mẫu cuối</title><link rel="stylesheet" href="app.css"><link rel="stylesheet" href="sidebar.css"></head>
<body><div class="app-shell"><aside class="queue-sidebar"><a href="index.html" class="sidebar-home">Tổng quan</a><nav id="queueNav"></nav></aside><div class="app-content"><header class="top"><a href="index.html" class="back">← Hàng đợi</a><div><p class="eyebrow">SOL WEB · 25 MẪU CUỐI</p>
<h1>{title}</h1><p class="sub" id="summary"></p></div></header>
<main class="workbench">
  <section class="panel actions"><h2>1. Gửi sang Sol</h2><p>Copy toàn bộ prompt sang Sol Web. Có thể yêu cầu Sol tạo đúng file JSON mang tên được nêu trong prompt.</p>
    <div class="buttons"><button class="primary" id="copyFull">Copy full prompt</button><button id="downloadPrompt">Tải prompt TXT</button>
    <button id="downloadSource">Tải source JSONL</button><button id="togglePreview">Xem trước 4.000 ký tự</button></div>
    <pre id="preview" class="hidden"></pre><p id="copyStatus" class="status"></p>
  </section>
  <section class="panel"><h2>2. Nhận kết quả</h2><p>Dán JSON/JSONL hoặc chọn file JSON do Sol tạo. Validate sẽ tự lưu batch vào dự án khi bộ nhận local đang chạy.</p>
    <input id="resultFile" type="file" accept=".json,.jsonl,.txt,application/json"><textarea id="result" spellcheck="false" placeholder="Dán kết quả Sol tại đây..."></textarea>
    <div class="buttons"><button class="primary" id="validate">Kiểm tra cấu trúc</button><button id="downloadAll">Tải item đã parse</button><button id="clearDraft">Xóa bản nháp</button></div>
    <div id="validation" class="validation neutral">Chưa kiểm tra</div><pre id="details"></pre>
  </section>
  <section class="panel wide"><h2>UID trong batch</h2><div id="records" class="records"></div></section>
</main></div></div>
<script id="batch-data" type="application/json">{payload}</script><script src="queue-data.js"></script><script src="queue-nav.js"></script><script src="batch.js"></script></body></html>"""


def pack(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rows.sort(key=lambda row: (
        {"high_tail": 0, "oversized": 1}.get(str(row.get("length_bucket")), 2),
        len(row.get("prompt_en") or "") + len(row.get("response_en") or ""),
    ))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    chars = 0
    for row in rows:
        row_chars = len(row.get("prompt_en") or "") + len(row.get("response_en") or "")
        if current and (len(current) >= MAX_ITEMS or chars + row_chars > MAX_SOURCE_CHARS):
            batches.append(current)
            current, chars = [], 0
        current.append(row)
        chars += row_chars
        # Oversized records and any record above the ceiling always stand alone.
        if row.get("length_bucket") == "oversized" or row_chars >= MAX_SOURCE_CHARS:
            batches.append(current)
            current, chars = [], 0
    if current:
        batches.append(current)
    return batches


def main() -> None:
    raw_rows = read_jsonl(SOURCE)
    rows = [
        {
            "seq": int(row["original_seq"]),
            "record_uid": str(row["record_uid"]),
            "source_split": row["source_split"],
            "length_bucket": row.get("length_bucket"),
            "route": "sol_missing_infrastructure",
            "prompt_en": row.get("prompt_en") or "",
            "response_en": row.get("response_en"),
            "previous_prompt_vi": None,
            "previous_response_vi": None,
            "validator_errors": ["No Luna/Sol candidate after infrastructure timeout; translate from source."],
            "validator_warnings": [],
        }
        for row in raw_rows
    ]
    batches = pack(rows)
    WEB_OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    for filename in ("app.css", "sidebar.css", "queue-nav.js"):
        shutil.copy2(BASE_WEB / filename, WEB_OUT / filename)
    batch_js = (BASE_WEB / "batch.js").read_text(encoding="utf-8").replace(
        "http://127.0.0.1:8791/save-batch", "http://127.0.0.1:8792/save-batch"
    )
    (WEB_OUT / "batch.js").write_text(batch_js, encoding="utf-8")
    export_js = (BASE_WEB / "queue-export.js").read_text(encoding="utf-8")
    export_js = export_js.replace("http://127.0.0.1:8791/save-results", "http://127.0.0.1:8792/save-results")
    export_js = export_js.replace("sol-fallback-all-results.json", "sol-missing-25-all-results.json")
    (WEB_OUT / "queue-export.js").write_text(export_js, encoding="utf-8")

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    manifest_batches: list[dict[str, Any]] = []
    for index, batch_rows in enumerate(batches, 1):
        batch_id = f"nemotron-sol-missing-25-{index:03d}"
        result_filename = f"{batch_id}-result.json"
        source_jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in batch_rows)
        prompt = prompt_template.replace("{{BATCH_ID}}", batch_id).replace(
            "{{RESULT_FILENAME}}", result_filename
        ).replace("{{SOURCE_RECORDS}}", source_jsonl)
        batch = {
            "batch_id": batch_id,
            "result_filename": result_filename,
            "records": batch_rows,
            "source_jsonl": source_jsonl,
            "prompt": prompt,
            "source_chars": sum(len(row["prompt_en"]) + len(row.get("response_en") or "") for row in batch_rows),
            "prompt_chars": len(prompt),
        }
        page = f"batch-{index:03d}.html"
        (WEB_OUT / page).write_text(batch_page(batch), encoding="utf-8")
        (DATA_OUT / f"batch-{index:03d}.jsonl").write_text(source_jsonl + "\n", encoding="utf-8")
        (DATA_OUT / f"batch-{index:03d}-prompt.txt").write_text(prompt, encoding="utf-8")
        manifest_batches.append({
            "batch_id": batch_id,
            "page": page,
            "records": len(batch_rows),
            "source_chars": batch["source_chars"],
            "prompt_chars": batch["prompt_chars"],
            "splits": sorted({str(row["source_split"]) for row in batch_rows}),
            "buckets": sorted({str(row.get("length_bucket")) for row in batch_rows}),
        })

    manifest = {
        "records": len(rows),
        "batches": manifest_batches,
        "batch_policy": {"max_items": MAX_ITEMS, "max_source_chars": MAX_SOURCE_CHARS, "oversized_stands_alone": True},
        "result_receiver": "python tools/sol_missing_25_receiver.py",
    }
    (DATA_OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    queue = {"records": len(rows), "batches": [{"batch_id": b["batch_id"], "page": b["page"], "records": b["records"]} for b in manifest_batches]}
    (WEB_OUT / "queue-data.js").write_text("window.SOL_QUEUE=" + json.dumps(queue, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")

    cards = "".join(
        f'<a class="batch-card" data-batch-id="{b["batch_id"]}" href="{b["page"]}"><span>Batch {i:03d}</span><strong>{b["records"]} mẫu</strong><small>{b["source_chars"]:,} ký tự nguồn · {b["prompt_chars"]:,} ký tự prompt</small></a>'
        for i, b in enumerate(manifest_batches, 1)
    )
    index = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sol Web · 25 mẫu cuối</title><link rel="stylesheet" href="app.css"><link rel="stylesheet" href="sidebar.css"></head><body><div class="app-shell"><aside class="queue-sidebar"><a href="index.html" class="sidebar-home">Tổng quan</a><nav id="queueNav"></nav></aside><div class="app-content"><header class="hero"><p class="eyebrow">NEMOTRON · HOÀN THIỆN 100%</p>
<h1>25 mẫu Luna còn thiếu</h1><p class="sub">Đây là 25 mẫu high-tail/oversized không có candidate vì timeout hạ tầng. Hoàn thành hàng đợi này sẽ thay thế toàn bộ Gemini fallback và tạo corpus Luna/Sol thuần đủ 45.416 UID.</p>
<div class="metrics"><div><strong>{len(rows)}</strong><span>mẫu còn lại</span></div><div><strong>{len(batches)}</strong><span>batch Sol</span></div><div><strong>45.000</strong><span>trần ký tự nguồn/batch</span></div></div></header>
<main class="queue"><section class="notice"><strong>Cách làm:</strong> mở từng batch ở sidebar, copy full prompt sang Sol Web, tải hoặc dán JSON trả về, rồi bấm “Kiểm tra cấu trúc”. Trang tự lưu từng batch khi receiver chạy. Khi hoàn tất, bấm nút dưới đây để gom toàn bộ kết quả.<div class="buttons"><button class="primary" id="exportQueue">Lưu tất cả vào dự án</button></div><p id="exportQueueStatus" class="status">Đích lưu: data/sol_missing_25/sol_results.</p></section><div class="batch-grid">{cards}</div></main></div></div><script src="queue-data.js"></script><script src="queue-nav.js"></script><script src="queue-export.js"></script></body></html>"""
    (WEB_OUT / "index.html").write_text(index, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
