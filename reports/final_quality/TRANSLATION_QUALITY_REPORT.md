# Nemotron Safety Guard V3 EN→VI — báo cáo chất lượng cuối

Tạo lúc `2026-07-20T12:26:27.553037+00:00` từ source prepared và 15 checkpoint mới nhất.

## Kết luận

- Đã có bản dịch cho **45,416/45,416 record (100.0000%)**.
- **45,416 record (100.0000%)** sẵn sàng theo các gate hiện tại.
- **0 record (0.0000%)** đầy đủ nhưng nên kiểm tra thêm độ nặng của từ thô tục.
- Hàng đợi khó đã xử lý **478/478**; còn lại **0**, failure queue **0**.
- Kiểm tra toàn vẹn quan trọng: **PASS**.

## Toàn vẹn dữ liệu

| Kiểm tra | Số lỗi |
| --- | --- |
| source_unique_records | 45416 |
| source_duplicate_uids | 0 |
| checkpoint_unique_records | 45416 |
| checkpoint_physical_rows | 45416 |
| checkpoint_duplicate_uids_across_files | 0 |
| wrong_checkpoint_location | 0 |
| missing_translations | 0 |
| extra_translations | 0 |
| structural_validator_failures | 0 |
| source_hash_mismatches | 0 |
| translation_hash_mismatches | 0 |
| input_char_count_mismatches | 0 |
| output_char_count_mismatches | 0 |

## Chất lượng theo trạng thái

| Trạng thái | Record | Tỷ lệ | Tier | Khuyến nghị |
| --- | --- | --- | --- | --- |
| machine_translated | 41,245 | 90.8160% | A_strict_machine_pass | ready |
| provisional_clean | 3,526 | 7.7638% | B_salvaged_clean | ready |
| provisional_format_repaired | 43 | 0.0947% | B_salvaged_format_repaired | ready_with_metadata |
| terra_revised | 336 | 0.7398% | A_revised | ready |
| luna_revised | 113 | 0.2488% | A_revised | ready |
| gemini_revised | 29 | 0.0639% | A_revised | ready |
| codex_quality_repaired | 26 | 0.0572% | A_human_repaired | ready |
| codex_quality_approved | 34 | 0.0749% | A_human_approved | ready |
| codex_profanity_repaired | 34 | 0.0749% | A_human_repaired | ready |
| codex_profanity_approved | 30 | 0.0661% | A_human_approved | ready |

Các tier là đánh giá vận hành của pipeline, không phải điểm BLEU/COMET. Toàn bộ 64 mẫu từng mang trạng thái `provisional_mild_profanity` đã được duyệt song ngữ thủ công: mẫu thực sự bị làm nhẹ đã được sửa, còn cảnh báo giả đã được phê duyệt kèm audit.

## Theo split

| Split | Tổng | machine_translated | provisional_clean | provisional_format_repaired | provisional_mild_profanity | terra_revised | luna_revised | gemini_revised | codex_quality_repaired | codex_quality_approved | codex_profanity_repaired | codex_profanity_approved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 40,007 | 36,765 | 2,741 | 38 | 0 | 277 | 71 | 9 | 22 | 28 | 27 | 29 |
| test | 2,964 | 2,541 | 346 | 4 | 0 | 41 | 22 | 0 | 4 | 3 | 3 | 0 |
| valid | 2,445 | 1,939 | 439 | 1 | 0 | 18 | 20 | 20 | 0 | 3 | 4 | 1 |

## Validator cuối

- Pass trực tiếp: **45,330**.
- Human override có audit (cảnh báo giả hoặc sửa có kiểm chứng): **86**.
- Caveat thô tục đã biết: **0**.
- Lỗi hard-validator không được giải thích: **0**.

## Theo nhãn safety nguyên tử

Một record có thể thuộc nhiều nhãn, vì vậy tổng các dòng lớn hơn tổng dataset.

| Nhãn | Tổng | % dataset | Machine | Salvaged clean | Format repair | Caveat profanity | Revised | % revised |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Criminal Planning/Confessions | 11,100 | 24.44% | 10,175 | 813 | 4 | 0 | 108 | 0.97% |
| Needs Caution | 5,194 | 11.44% | 4,693 | 401 | 4 | 0 | 96 | 1.85% |
| Hate/Identity Hate | 4,194 | 9.23% | 3,782 | 340 | 3 | 0 | 69 | 1.65% |
| Violence | 4,169 | 9.18% | 3,802 | 311 | 0 | 0 | 56 | 1.34% |
| Harassment | 3,623 | 7.98% | 3,316 | 264 | 2 | 0 | 41 | 1.13% |
| Controlled/Regulated Substances | 3,333 | 7.34% | 2,957 | 313 | 4 | 0 | 59 | 1.77% |
| PII/Privacy | 2,780 | 6.12% | 2,476 | 257 | 2 | 0 | 45 | 1.62% |
| Profanity | 2,639 | 5.81% | 2,310 | 211 | 0 | 0 | 118 | 4.47% |
| Immoral/Unethical | 2,203 | 4.85% | 1,789 | 337 | 9 | 0 | 68 | 3.09% |
| Sexual | 2,177 | 4.79% | 2,004 | 142 | 3 | 0 | 28 | 1.29% |
| Illegal Activity | 2,132 | 4.69% | 1,715 | 353 | 6 | 0 | 58 | 2.72% |
| Guns and Illegal Weapons | 2,024 | 4.46% | 1,818 | 177 | 2 | 0 | 27 | 1.33% |
| Suicide and Self Harm | 2,013 | 4.43% | 1,740 | 224 | 1 | 0 | 48 | 2.38% |
| Unauthorized Advice | 1,402 | 3.09% | 1,181 | 179 | 5 | 0 | 37 | 2.64% |
| Manipulation | 1,363 | 3.00% | 1,127 | 199 | 3 | 0 | 34 | 2.49% |
| Sexual (minor) | 1,120 | 2.47% | 917 | 162 | 3 | 0 | 38 | 3.39% |
| Political/Misinformation/Conspiracy | 1,108 | 2.44% | 946 | 131 | 1 | 0 | 30 | 2.71% |
| Fraud/Deception | 790 | 1.74% | 710 | 66 | 1 | 0 | 13 | 1.65% |
| Threat | 659 | 1.45% | 541 | 96 | 2 | 0 | 20 | 3.03% |
| Other | 351 | 0.77% | 316 | 27 | 0 | 0 | 8 | 2.28% |
| Malware | 227 | 0.50% | 203 | 21 | 2 | 0 | 1 | 0.44% |
| Copyright/Trademark/Plagiarism | 119 | 0.26% | 107 | 8 | 0 | 0 | 4 | 3.36% |
| High Risk Gov Decision Making | 119 | 0.26% | 108 | 10 | 0 | 0 | 1 | 0.84% |

## Cách dùng file final

Mỗi dòng chứa `prompt_en`, `response_en`, `prompt_vi`, `response_vi`, nhãn safety gốc, metadata dịch, tier chất lượng và cờ `needs_followup_review`. Hiện có 0 mẫu còn mang khuyến nghị duyệt tiếp.
Danh sách 64 mẫu đầu vào của vòng duyệt cuối được giữ tại `reports/final_quality/profanity_followup_64.jsonl`; quyết định và hash trước/sau nằm tại `data/revision_handoff/human_review_decisions/final_profanity_manual_review.jsonl`.

## Diễn giải giới hạn

Báo cáo xác nhận độ phủ, cấu trúc, hash, tính nhất quán, các gate heuristic và toàn bộ revision queue. Nó không chứng minh từng câu đạt mức tương đương hoàn hảo như một đánh giá song ngữ độc lập trên 45.416 mẫu. Các cảnh báo heuristic có thể là báo động giả khi nguồn chứa code, ASCII art, SQL/JSON, URL, tên riêng hoặc văn bản đa ngôn ngữ.
