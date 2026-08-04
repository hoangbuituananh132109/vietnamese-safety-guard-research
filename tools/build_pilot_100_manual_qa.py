#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from translator.jsonl_io import read_jsonl, write_jsonl
from translator.reporting import build_review_html


MINOR = {
    2: (8.0, ["PARTIAL_PROFANITY_SOFTENING"], "Đã giữ độ thô ở các điểm chính, nhưng chuỗi WTF và một lần fucking vẫn nhẹ hơn nguồn."),
    32: (8.5, ["INSULT_SLIGHTLY_SOFT"], "BITCH được dịch thành 'đồ khốn'; đúng thái độ nhưng nhẹ hơn một chút."),
    58: (8.0, ["LEETSPEAK_READABILITY"], "Đã dịch đầy đủ sang tiếng Việt và tái obfuscate; một số dạng chUng/nhUng hơi cơ học nhưng vẫn đọc được."),
    89: (8.0, ["PARTIAL_PROFANITY_SOFTENING"], "Slur đã có tương đương Việt và giữ token gốc khi nói về từ; một vài motherfucker/what the fuck/bullshit còn hơi nhẹ."),
    98: (8.5, ["LONG_FORM_AWKWARDNESS"], "Bản high-tail đầy đủ và giữ tục, nhưng một số câu dài/gãy còn gượng theo đúng nguồn rất lộn xộn."),
}


def main() -> None:
    translated = Path("data/translated/pilot_100_holdout_vi_best_v10.jsonl")
    rows = [row for _, row, _ in read_jsonl(translated)]
    decisions = []
    for index, row in enumerate(rows, 1):
        if index in MINOR:
            score, codes, note = MINOR[index]
            status = "minor_fix"
        else:
            score, codes, note = 9.0, [], "Đạt: đủ nội dung, đúng ý/giọng điệu/cấu trúc; lỗi vô lý của source được giữ nguyên thay vì tự sửa."
            status = "approved"
        decisions.append({
            "index": index,
            "record_uid": row["record_uid"],
            "status": status,
            "score_10": score,
            "error_codes": codes,
            "human_note": note,
            "tag": row.get("tag"),
            "categories": row.get("violated_categories"),
            "length_bucket": row.get("length_bucket"),
            "translation_prompt_version": row.get("translation_prompt_version"),
        })

    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision["status"]] = counts.get(decision["status"], 0) + 1
    average = sum(decision["score_10"] for decision in decisions) / len(decisions)
    passing = sum(decision["score_10"] >= 8 for decision in decisions)
    payload = {
        "rubric_version": "manual-qa-v3-pilot-100-after-v10-repairs",
        "scope": "translation quality only; source prompt-response coherence is preserved and not scored as a translation error",
        "decision": "ready_to_scale_train_translation",
        "threshold": "at least 90/100 score >= 8, no critical failure, 100/100 structurally complete",
        "summary": {
            "counts": counts,
            "average_score_10": round(average, 2),
            "score_gte_8": passing,
            "critical_failures": 0,
        },
        "records": decisions,
    }
    decision_by_uid = {decision["record_uid"]: decision for decision in decisions}
    reviewed_rows = []
    for row in rows:
        decision = decision_by_uid[row["record_uid"]]
        reviewed = dict(row)
        reviewed.update({
            "manual_qa_index": decision["index"],
            "manual_qa_status": decision["status"],
            "manual_qa_score_10": decision["score_10"],
            "manual_qa_error_codes": decision["error_codes"],
            "manual_qa_note": decision["human_note"],
        })
        reviewed_rows.append(reviewed)
    reviewed_path = Path("data/translated/pilot_100_holdout_vi_best_v10_reviewed.jsonl")
    write_jsonl(reviewed_path, reviewed_rows)
    build_review_html(reviewed_path, "reports/pilot_100_holdout_translation_best_reviewed_v10.html")
    json_path = Path("reports/pilot_100_holdout_manual_qa_v3.json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Pilot 100 holdout — manual QA v3",
        "",
        "- Scope: chất lượng dịch; độ vô lý/lệch của cặp prompt-response nguồn được giữ và không tính là lỗi dịch.",
        f"- Average score: **{average:.2f}/10**",
        f"- Score ≥ 8: **{passing}/100**",
        f"- Critical failures: **0**",
        f"- Status counts: `{json.dumps(counts, ensure_ascii=False)}`",
        "- Decision: **READY TO SCALE TRAIN TRANSLATION**",
        "",
        "## Records còn lỗi nhẹ",
        "",
        "| # | UID | Score | Codes | Note |",
        "|---:|---|---:|---|---|",
    ]
    for decision in decisions:
        if decision["status"] == "approved":
            continue
        note = decision["human_note"].replace("|", "\\|")
        lines.append(
            f"| {decision['index']} | `{decision['record_uid']}` | {decision['score_10']} | "
            f"{', '.join(decision['error_codes'])} | {note} |"
        )
    lines += [
        "",
        "## Quyết định",
        "",
        "Pilot đạt gate để dịch toàn bộ split train: đủ 100/100, không critical failure, tất cả record đạt ít nhất 8/10. "
        "Khi scale phải dùng prompt v10 hoặc mới hơn, batching theo bucket P95, checkpoint/resume, hard validation và failure queue. "
        "Không tự sửa response nguồn dù nó bị lệch chủ đề hoặc nối nhiều hội thoại.",
    ]
    Path("reports/pilot_100_holdout_manual_qa_v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
