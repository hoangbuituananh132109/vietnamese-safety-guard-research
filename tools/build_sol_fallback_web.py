from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.luna_overnight_runner import source_fields, validate_item
from translator.models import TranslationInputItem


SOURCE = ROOT / "data/final/nemotron_valid_en_vi_v10_final.jsonl"
RUN = ROOT / "data/luna_overnight/nemotron_valid_20260802"
OUT = ROOT / "web/sol_fallback_queue"
DATA_OUT = ROOT / "data/sol_fallback_queue"
PROMPT_PATH = ROOT / "configs/sol_web_fallback_repair_prompt.md"
MAX_ITEMS = 15
MAX_SOURCE_CHARS = 18_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def payload_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def batch_page(batch: dict[str, Any]) -> str:
    title = html.escape(batch["batch_id"])
    payload = payload_script(batch)
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Sol fallback</title><link rel="stylesheet" href="app.css"><link rel="stylesheet" href="sidebar.css"></head>
<body><div class="app-shell"><aside class="queue-sidebar"><a href="index.html" class="sidebar-home">Tổng quan</a><nav id="queueNav"></nav></aside><div class="app-content"><header class="top"><a href="index.html" class="back">← Hàng đợi</a><div><p class="eyebrow">SOL WEB FALLBACK</p>
<h1>{title}</h1><p class="sub" id="summary"></p></div></header>
<main class="workbench">
  <section class="panel actions"><h2>1. Gửi sang Sol</h2><p>Copy toàn bộ prompt hoặc tải file TXT. Trang chỉ dựng chuỗi đầy đủ khi bạn bấm, nên batch dài không làm lag lúc mở.</p>
    <div class="buttons"><button class="primary" id="copyFull">Copy full prompt</button><button id="downloadPrompt">Tải prompt TXT</button>
    <button id="downloadSource">Tải source JSONL</button><button id="togglePreview">Xem trước 4.000 ký tự</button></div>
    <pre id="preview" class="hidden"></pre><p id="copyStatus" class="status"></p>
  </section>
  <section class="panel"><h2>2. Nhận kết quả</h2><p>Dán JSON/JSONL hoặc chọn file JSON Sol tạo. Bản nháp được lưu cục bộ theo batch.</p>
    <input id="resultFile" type="file" accept=".json,.jsonl,.txt,application/json"><textarea id="result" spellcheck="false" placeholder="Dán kết quả Sol tại đây..."></textarea>
    <div class="buttons"><button class="primary" id="validate">Kiểm tra cấu trúc</button><button id="downloadAll">Tải tất cả item đã parse</button><button id="clearDraft">Xóa bản nháp</button></div>
    <div id="validation" class="validation neutral">Chưa kiểm tra</div><pre id="details"></pre>
  </section>
  <section class="panel wide"><h2>UID và lỗi cần sửa</h2><div id="records" class="records"></div></section>
</main></div></div>
<script id="batch-data" type="application/json">{payload}</script><script src="queue-data.js"></script><script src="queue-nav.js"></script><script src="batch.js"></script></body></html>"""


def main() -> None:
    source_rows = read_jsonl(SOURCE)
    source_by_uid: dict[str, tuple[int, dict[str, Any]]] = {
        str(row["record_uid"]): (seq, row) for seq, row in enumerate(source_rows, 1)
    }
    rejected: dict[str, dict[str, Any]] = {}
    for filename in ("exhausted_normal.jsonl", "tail_escalation.jsonl"):
        for row in read_jsonl(RUN / filename):
            rejected[str(row["record_uid"])] = row

    hard_records: list[dict[str, Any]] = []
    recovered: list[str] = []
    for uid, terminal in rejected.items():
        seq, source = source_by_uid[uid]
        candidate = terminal.get("candidate")
        if not isinstance(candidate, dict):
            continue
        expected = TranslationInputItem(
            seq=seq, record_uid=uid, prompt=source_fields(source)[0], response=source_fields(source)[1]
        )
        hard_errors, warnings = validate_item(expected, candidate, "sol-fallback-selection")
        if not hard_errors:
            recovered.append(uid)
            continue
        prompt_en, response_en = source_fields(source)
        hard_records.append({
            "seq": seq,
            "record_uid": uid,
            "source_split": source.get("source_split"),
            "length_bucket": terminal.get("length_bucket"),
            "route": terminal.get("route"),
            "prompt_en": prompt_en,
            "response_en": response_en,
            "previous_prompt_vi": candidate.get("prompt_vi"),
            "previous_response_vi": candidate.get("response_vi"),
            "validator_errors": hard_errors,
            "validator_warnings": warnings,
        })

    hard_records.sort(key=lambda row: (row["length_bucket"] != "normal", row["seq"]))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for row in hard_records:
        row_chars = len(row["prompt_en"]) + len(row["response_en"] or "")
        if current and (len(current) >= MAX_ITEMS or current_chars + row_chars > MAX_SOURCE_CHARS):
            batches.append(current)
            current, current_chars = [], 0
        current.append(row)
        current_chars += row_chars
    if current:
        batches.append(current)

    OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    manifest_batches = []
    for index, rows in enumerate(batches, 1):
        batch_id = f"nemotron-valid-sol-fallback-{index:02d}"
        result_filename = f"{batch_id}-result.json"
        source_jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        prompt = (
            prompt_template.replace("{{BATCH_ID}}", batch_id)
            .replace("{{RESULT_FILENAME}}", result_filename)
            .replace("{{SOURCE_RECORDS}}", source_jsonl)
        )
        batch = {
            "batch_id": batch_id,
            "result_filename": result_filename,
            "records": rows,
            "source_jsonl": source_jsonl,
            "prompt": prompt,
            "source_chars": sum(len(row["prompt_en"]) + len(row["response_en"] or "") for row in rows),
            "prompt_chars": len(prompt),
        }
        page_name = f"batch-{index:02d}.html"
        (OUT / page_name).write_text(batch_page(batch), encoding="utf-8")
        (DATA_OUT / f"batch-{index:02d}.jsonl").write_text(source_jsonl + "\n", encoding="utf-8")
        manifest_batches.append({
            "batch_id": batch_id, "page": page_name, "records": len(rows),
            "source_chars": batch["source_chars"], "prompt_chars": batch["prompt_chars"],
            "buckets": sorted({str(row["length_bucket"]) for row in rows}),
        })

    manifest = {
        "source_split": "valid",
        "selected_hard_records": len(hard_records),
        "recovered_by_current_validator": recovered,
        "batch_policy": {"max_items": MAX_ITEMS, "max_source_chars": MAX_SOURCE_CHARS},
        "batches": manifest_batches,
    }
    (DATA_OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = "".join(
        f'<a class="batch-card" href="{b["page"]}"><span>Batch {i:02d}</span><strong>{b["records"]} mẫu</strong>'
        f'<small>{b["source_chars"]:,} ký tự nguồn · {b["prompt_chars"]:,} ký tự prompt</small></a>'
        for i, b in enumerate(manifest_batches, 1)
    )
    index = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sol fallback queue</title><link rel="stylesheet" href="app.css"></head><body><header class="hero"><p class="eyebrow">NEMOTRON · VALID</p>
<h1>Hàng đợi Sol web</h1><p class="sub">Chỉ gồm {len(hard_records)} mẫu có candidate Luna nhưng vẫn sai chất lượng sau khi chạy lại validator hiện tại. Mỗi batch là một trang riêng để không tải toàn bộ văn bản dài cùng lúc.</p>
<div class="metrics"><div><strong>{len(hard_records)}</strong><span>mẫu Sol</span></div><div><strong>{len(batches)}</strong><span>batch</span></div><div><strong>{len(recovered)}</strong><span>mẫu đã cứu khỏi false positive</span></div></div></header>
<main class="queue"><section class="notice"><strong>Không đưa sang Sol trong đợt này:</strong> các mẫu train/test chưa có candidate hoàn chỉnh vẫn phải chạy Luna hoặc revalidate trước; thiếu candidate không phải bằng chứng Sol là cần thiết.</section>
<div class="batch-grid">{cards}</div></main></body></html>"""
    (OUT / "index.html").write_text(index, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
