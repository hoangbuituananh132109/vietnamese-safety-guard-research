#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from translator.jsonl_io import read_jsonl


# One-based index: (status, score/10, error codes, concise human note).
FLAGS = {
    7: ("minor_fix", 7.5, ["ACTIONABILITY_SOFTENED"], "'easier to get my hands on' thành 'dễ tiếp cận', làm nhẹ sắc thái săn/tóm mục tiêu."),
    10: ("minor_fix", 7.5, ["TERM_BROADENED"], "'schizophrenic' bị làm rộng thành 'người bị tâm thần' thay vì tâm thần phân liệt."),
    13: ("minor_fix", 8.0, ["UNNATURAL_VI"], "'bodily harm' thành cụm gượng 'thương tích cơ thể'."),
    18: ("minor_fix", 7.5, ["PROFANITY_SOFTENED"], "'shithead'/'damn annoying' được dịch nhẹ hơn cường độ nguồn."),
    24: ("minor_fix", 8.0, ["UNNATURAL_VI"], "'tools for a suicide attempt' thành 'công cụ cho một vụ tự tử', đúng ý nhưng không tự nhiên."),
    41: ("minor_fix", 8.0, ["UNNATURAL_VI"], "Một số cụm gượng như 'thoát tội bất cứ điều gì'; ý chính vẫn được giữ."),
    42: ("major_fix", 6.5, ["PROFANITY_SOFTENED", "HARD_VALIDATION_FAILED"], "'bullshit' và 'fuckin' vẫn bị làm nhẹ sau repair; v5 đã từ chối output và đưa record vào failure queue."),
    45: ("minor_fix", 7.5, ["SEXUAL_EXPLICITNESS_LOST"], "'graphic sexual content' thành 'nội dung tình dục rõ ràng', chưa giữ độ trần trụi."),
    46: ("minor_fix", 8.0, ["TERM_DRIFT"], "Một số thuật ngữ heroin chưa tự nhiên: 'black tar heroin', 'rush'."),
    50: ("minor_fix", 7.5, ["OVER_OBFUSCATED", "LENGTH_EXPANSION"], "V5 đã dịch leetspeak sang tiếng Việt, nhưng tái obfuscation bằng dấu gạch chéo quá dày và làm độ dài tăng mạnh."),
}


def main() -> None:
    translated = Path("data/translated/pilot_50_vi_machine_best.jsonl")
    rows = [r for _, r, _ in read_jsonl(translated)]
    decisions = []
    for index, row in enumerate(rows, 1):
        status, score, codes, note = FLAGS.get(index, ("approved", 9.0, [], "Bản dịch giữ đủ ý, thái độ và cấu trúc; không thấy lỗi đáng sửa khi đọc thủ công."))
        if index in (2, 5, 10):
            note += " Empty response đã được sửa về chuỗi rỗng theo schema sau QA."
        decisions.append({
            "index": index, "record_uid": row["record_uid"], "status": status, "score_10": score,
            "error_codes": codes, "human_note": note, "tag": row.get("tag"),
            "categories": row.get("violated_categories"), "length_bucket": row.get("length_bucket"),
        })
    out = Path("reports/pilot_50_manual_qa_v2.json")
    out.write_text(json.dumps({"rubric_version":"manual-qa-v2-after-v5-repair","records":decisions}, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {}
    for d in decisions: counts[d["status"]] = counts.get(d["status"], 0) + 1
    avg = sum(d["score_10"] for d in decisions) / len(decisions)
    passing = sum(d["score_10"] >= 8 for d in decisions)
    lines = ["# Pilot 50 — manual QA", "", f"- Average score: **{avg:.2f}/10**", f"- Score ≥ 8: **{passing}/50**", f"- Status counts: `{json.dumps(counts, ensure_ascii=False)}`", "", "## Records requiring attention", "", "| # | UID | Status | Score | Codes | Note |", "|---:|---|---|---:|---|---|"]
    for d in decisions:
        if d["status"] != "approved":
            note = d["human_note"].replace("|", "\\|")
            lines.append(f"| {d['index']} | `{d['record_uid']}` | {d['status']} | {d['score_10']} | {', '.join(d['error_codes'])} | {note} |")
    lines += ["", "## Decision", "", "Sau repair v5, pilot vẫn chưa đạt ngưỡng scale tự động: còn một hard failure về cường độ tục và chín minor-fix. Nên giữ hard validation, xử lý failure queue, rồi mới tạo holdout 50."]
    Path("reports/pilot_50_manual_qa_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"counts":counts,"average":round(avg,2),"score_gte_8":passing}, ensure_ascii=False))


if __name__ == "__main__": main()
