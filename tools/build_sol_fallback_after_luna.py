from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_sol_fallback_web import batch_page


QUEUE = ROOT / "data/luna_remaining_20260803/queue.jsonl"
RUN = ROOT / "data/luna_remaining_20260803/run_light_retry2"
RECOVERY = ROOT / "data/luna_remaining_20260803/recovery"
CONTINUATION = ROOT / "data/luna_remaining_20260803/run_light_retry2_continuation"
BASELINE = ROOT / "data/sol_fallback_queue"
DATA_OUT = ROOT / "data/sol_fallback_after_luna"
WEB_OUT = ROOT / "web/sol_fallback_queue"
PROMPT_PATH = ROOT / "configs/sol_web_fallback_repair_prompt.md"
MAX_SOURCE_CHARS = 50_000


def item_limit(row_chars: int) -> int:
    """Use larger batches for short records and smaller batches for long records."""
    if row_chars <= 1_562:
        return 32
    if row_chars <= 3_125:
        return 16
    if row_chars <= 6_250:
        return 8
    return 4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # JSON string values may legally contain U+0085. str.splitlines() treats that character
    # as a line boundary and corrupts the record, whereas JSONL is delimited only by LF here.
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="strict").split("\n"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A watcher can observe the final append while it is still being written. The next
            # refresh will read the complete line, so never fail or reshuffle existing batches.
            continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repack",
        action="store_true",
        help="Ignore prior batch boundaries and rebuild all current records with the latest policy.",
    )
    args = parser.parse_args()
    records: dict[str, dict[str, Any]] = {}

    # Preserve the 16 valid records that were already proven hard before this run.
    for path in sorted(BASELINE.glob("batch-*.jsonl")):
        for row in read_jsonl(path):
            records[str(row["record_uid"])] = row

    queue_rows = read_jsonl(QUEUE)
    queue_by_uid = {str(row["record_uid"]): (seq, row) for seq, row in enumerate(queue_rows, 1)}
    accepted_uids = {
        str(row["record_uid"])
        for directory in (RUN, CONTINUATION)
        for filename in ("passed.jsonl", "needs_audit.jsonl")
        for row in read_jsonl(directory / filename)
    }
    failure_rows = read_jsonl(RECOVERY / "pre_web_failures.jsonl")
    for filename in ("exhausted_normal.jsonl", "tail_escalation.jsonl"):
        failure_rows.extend(read_jsonl(CONTINUATION / filename))
    for failure in failure_rows:
            uid = str(failure["record_uid"])
            if uid not in queue_by_uid or uid in accepted_uids:
                continue
            _, source = queue_by_uid[uid]
            candidate = failure.get("candidate") if isinstance(failure.get("candidate"), dict) else {}
            records[uid] = {
                "seq": int(source["original_seq"]),
                "record_uid": uid,
                "source_split": source["source_split"],
                "length_bucket": failure.get("length_bucket") or source.get("length_bucket"),
                "route": failure.get("route"),
                "prompt_en": source.get("prompt_en") or "",
                "response_en": source.get("response_en"),
                "previous_prompt_vi": candidate.get("prompt_vi"),
                "previous_response_vi": candidate.get("response_vi"),
                "validator_errors": failure.get("hard_errors") or ["missing candidate after two Luna Light attempts"],
                "validator_warnings": failure.get("heuristic_warnings") or [],
                "luna_attempts": failure.get("attempts"),
                "luna_batch_id": failure.get("batch_id"),
            }

    # Incremental policy: never reshuffle a batch already exposed to the user. Reuse its
    # record order and boundaries, then pack only newly failed UIDs into batches appended
    # at the end. This allows Sol web work to start while long Luna requests are still live.
    existing_batches: list[list[dict[str, Any]]] = []
    existing_uids: set[str] = set()
    if not args.repack and (DATA_OUT / "manifest.json").exists():
        for path in sorted(DATA_OUT.glob("batch-*.jsonl")):
            previous_rows = read_jsonl(path)
            kept = [records[str(row["record_uid"])] for row in previous_rows if str(row["record_uid"]) in records]
            if kept:
                existing_batches.append(kept)
                existing_uids.update(str(row["record_uid"]) for row in kept)

    new_rows = sorted(
        (row for uid, row in records.items() if uid not in existing_uids),
        key=lambda row: (
            str(row.get("source_split")),
            item_limit(len(row.get("prompt_en") or "") + len(row.get("response_en") or "")),
            {"normal": 0, "near_tail": 1, "tail": 2, "high_tail": 3, "oversized": 4}.get(str(row.get("length_bucket")), 9),
            str(row.get("route")), int(row.get("seq") or 0),
        ),
    )
    appended_batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    chars = 0
    current_limit: int | None = None
    for row in new_rows:
        row_chars = len(row.get("prompt_en") or "") + len(row.get("response_en") or "")
        row_limit = item_limit(row_chars)
        # Keep the 32/16/8/4 tiers homogeneous so a late long record cannot make a
        # mostly-short batch unexpectedly unwieldy. The 50k character ceiling is hard.
        if current and (
            row_limit != current_limit
            or len(current) >= int(current_limit or row_limit)
            or chars + row_chars > MAX_SOURCE_CHARS
        ):
            appended_batches.append(current)
            current, chars, current_limit = [], 0, None
        if not current:
            current_limit = row_limit
        current.append(row)
        chars += row_chars
    if current:
        appended_batches.append(current)
    batches = existing_batches + appended_batches
    ordered = [row for batch in batches for row in batch]

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    WEB_OUT.mkdir(parents=True, exist_ok=True)
    if args.repack:
        # These are generated artifacts. Remove only obsolete numbered batch files;
        # shared CSS/JS and unrelated files remain untouched.
        for directory, pattern in ((DATA_OUT, "batch-*.jsonl"), (WEB_OUT, "batch-*.html")):
            for path in directory.glob(pattern):
                path.unlink()
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    manifest_batches: list[dict[str, Any]] = []
    for index, rows in enumerate(batches, 1):
        batch_id = f"nemotron-sol-fallback-after-luna-{index:03d}"
        result_filename = f"{batch_id}-result.json"
        source_jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        prompt = (
            prompt_template.replace("{{BATCH_ID}}", batch_id)
            .replace("{{RESULT_FILENAME}}", result_filename)
            .replace("{{SOURCE_RECORDS}}", source_jsonl)
        )
        batch = {
            "batch_id": batch_id, "result_filename": result_filename, "records": rows,
            "source_jsonl": source_jsonl, "prompt": prompt,
            "source_chars": sum(len(row.get("prompt_en") or "") + len(row.get("response_en") or "") for row in rows),
            "prompt_chars": len(prompt),
        }
        page = f"batch-{index:03d}.html"
        (WEB_OUT / page).write_text(batch_page(batch), encoding="utf-8")
        (DATA_OUT / f"batch-{index:03d}.jsonl").write_text(source_jsonl + "\n", encoding="utf-8")
        manifest_batches.append({
            "batch_id": batch_id, "page": page, "records": len(rows),
            "source_chars": batch["source_chars"], "prompt_chars": batch["prompt_chars"],
            "splits": sorted({str(row.get("source_split")) for row in rows}),
            "buckets": sorted({str(row.get("length_bucket")) for row in rows}),
        })

    manifest = {
        "records": len(ordered), "baseline_records": 16,
        "new_luna_failures": max(0, len(ordered) - 16),
        "existing_batches_preserved": len(existing_batches),
        "records_appended_this_build": len(new_rows),
        "batches_appended_this_build": len(appended_batches),
        "batch_policy": {
            "item_tiers": [32, 16, 8, 4],
            "tier_source_char_thresholds": [1562, 3125, 6250],
            "max_source_chars": MAX_SOURCE_CHARS,
        },
        "batches": manifest_batches,
    }
    (DATA_OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    queue_data = {
        "records": len(ordered),
        "batches": [
            {"batch_id": b["batch_id"], "page": b["page"], "records": b["records"]}
            for b in manifest_batches
        ],
    }
    (WEB_OUT / "queue-data.js").write_text(
        "window.SOL_QUEUE=" + json.dumps(queue_data, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    cards = "".join(
        f'<a class="batch-card" data-batch-id="{b["batch_id"]}" href="{b["page"]}"><span>Batch {i:03d}</span><strong>{b["records"]} mẫu</strong>'
        f'<small>{b["source_chars"]:,} ký tự nguồn · {b["prompt_chars"]:,} ký tự prompt</small></a>'
        for i, b in enumerate(manifest_batches, 1)
    )
    index = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sol fallback after Luna</title><link rel="stylesheet" href="app.css"><link rel="stylesheet" href="sidebar.css"></head><body><div class="app-shell"><aside class="queue-sidebar"><a href="index.html" class="sidebar-home">Tổng quan</a><nav id="queueNav"></nav></aside><div class="app-content"><header class="hero"><p class="eyebrow">NEMOTRON · LUNA → SOL</p>
<h1>Hàng đợi Sol web</h1><p class="sub">Gồm 16 mẫu khó đã biết và {manifest["new_luna_failures"]} mẫu mới không vượt qua tối đa hai lần Luna Light. Mỗi batch là một trang riêng và bị giới hạn theo tổng ký tự.</p>
<div class="metrics"><div><strong>{len(ordered)}</strong><span>mẫu Sol</span></div><div><strong>{len(batches)}</strong><span>batch</span></div><div><strong>{manifest["new_luna_failures"]}</strong><span>thất bại Luna mới</span></div></div></header>
<main class="queue"><section class="notice"><strong>Quy tắc:</strong> mọi candidate Luna đạt đã được lưu riêng; trang này chỉ chứa UID còn lỗi sau hai lần. Kết quả Sol từng item vẫn được tải kể cả batch chưa hoàn chỉnh.<div class="buttons"><button class="primary" id="exportQueue">Lưu tất cả vào dự án</button></div><p id="exportQueueStatus" class="status">Gom mọi bản nháp và tự lưu vào data/sol_fallback_after_luna/sol_results.</p></section><div class="batch-grid">{cards}</div></main></div></div><script src="queue-data.js"></script><script src="queue-nav.js"></script><script src="queue-export.js"></script></body></html>"""
    (WEB_OUT / "index.html").write_text(index, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
