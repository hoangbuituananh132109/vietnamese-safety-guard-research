# Phân tích accuracy theo ngôn ngữ và so sánh Phase 0

Thời điểm snapshot: 22/07/2026. Tất cả chênh lệch trong báo cáo được tính trên **đúng cùng suite, benchmark và label order**. `pp` là điểm phần trăm, không phải phần trăm tương đối.

## Kết luận ngắn

- **E4 (mmBERT matched EN+VI)** cải thiện phía tiếng Việt so với **E3 (English-only)**, đồng thời làm khoảng cách EN–VI nhỏ hơn và tăng độ nhất quán giữa hai bản ngữ. Mức tăng không lớn nhưng xuất hiện cả trên Nemotron test lẫn SEA paired.
- E4 không thắng E3 trên mọi tiêu chí: **Vietnamese unsafe recall giảm**, trong khi safe recall và unsafe precision tăng. Nói cách khác, E4 bớt xu hướng báo Unsafe quá mức, nhưng đổi lại để lọt nhiều mẫu Unsafe hơn tại threshold hiện tại.
- **E2 (mmBERT GLi-compatible)** cân bằng EN–VI tốt hơn E1 rất rõ. E1 có unsafe recall tiếng Việt rất cao nhưng accuracy/Macro-F1 thấp, cho thấy E1 thiên về dự đoán Unsafe trên tiếng Việt chứ không phải hiểu tiếng Việt tốt hơn.
- **E5 (full EN+VI)** là fixed-head encoder tốt nhất. **E7** đã hoàn tất kỹ thuật nhưng dynamic-schema recipe hiện tại thất bại (56,13% Nemotron, 54,54% SEA và N23 không dự đoán positive ở threshold 0,5). **D1 Nemotron Guard 8B** là quality leader (86,86% Nemotron, 83,91% SEA) nhưng chậm hơn encoder khoảng một bậc độ lớn. Xem báo cáo đã chốt trong `reports/research_archive/PROJECT_EXPERIMENT_SYNTHESIS_20260722.md`.

## E3 vs E4 — full mmBERT

### Nemotron test (5.341 EN + 5.341 VI)

| Run | Overall accuracy | EN accuracy | VI accuracy | VI − EN | EN Macro-F1 | VI Macro-F1 | EN–VI consistency |
|---|---:|---:|---:|---:|---:|---:|---:|
| E3 English-only | 75,58% | **77,98%** | 73,17% | −4,81 pp | **0,7795** | 0,7267 | 82,91% |
| E4 matched EN+VI | **76,13%** | 77,46% | **74,80%** | **−2,66 pp** | 0,7744 | **0,7477** | **86,59%** |

E4 so với E3:

- Overall accuracy: **+0,55 pp**.
- English accuracy: **−0,52 pp**.
- Vietnamese accuracy: **+1,63 pp**.
- Vietnamese Macro-F1: **+2,10 pp**.
- Khoảng cách VI–EN thu hẹp **2,15 pp**.
- Paired EN–VI consistency: **+3,69 pp**.

Trade-off lớp Unsafe trên tiếng Việt:

| Run | VI safe recall | VI unsafe precision | VI unsafe recall |
|---|---:|---:|---:|
| E3 | 62,83% | 71,09% | **82,51%** |
| E4 | **75,10%** | **76,83%** | 74,53% |

E4 giảm false positive Unsafe và cân bằng hai lớp tốt hơn, nhưng unsafe recall giảm **7,98 pp**. Vì đây là guardrail, threshold sau huấn luyện cần được tune trên validation theo chi phí false-negative mong muốn; không nên chọn model chỉ bằng accuracy.

### SEA paired (1.840 EN + 1.840 VI)

| Run | Overall accuracy | EN accuracy | VI accuracy | VI − EN | EN Macro-F1 | VI Macro-F1 | EN–VI consistency |
|---|---:|---:|---:|---:|---:|---:|---:|
| E3 English-only | 71,82% | 73,80% | 69,84% | −3,97 pp | 0,7361 | 0,6958 | 80,71% |
| E4 matched EN+VI | **72,58%** | **74,08%** | **71,09%** | **−2,99 pp** | **0,7390** | **0,7053** | **85,49%** |

E4 tiếp tục tăng overall **0,76 pp**, EN **0,27 pp**, VI **1,25 pp**, và consistency **4,78 pp**. Kết quả trên benchmark ngoài Nemotron củng cố rằng lợi ích song ngữ không chỉ là học thuộc phân phối test Nemotron.

## E5 — full EN+VI

### Binary Safe/Unsafe · Nemotron test

| Run | Overall accuracy | EN accuracy | VI accuracy | EN Macro-F1 | VI Macro-F1 | EN unsafe recall | VI unsafe recall | EN–VI consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | 75,58% | 77,98% | 73,17% | 0,7795 | 0,7267 | 77,63% | **82,51%** | 82,91% |
| E4 | 76,13% | 77,46% | 74,80% | 0,7744 | 0,7477 | 76,45% | 74,53% | 86,59% |
| E5 | **78,58%** | **79,97%** | **77,20%** | **0,7995** | **0,7718** | **78,87%** | 75,99% | **87,23%** |

E5 hơn E4 **2,45 pp overall**, **2,51 pp English** và **2,40 pp Vietnamese**. E5 cũng hơn E3 **3,00 pp overall** và **4,03 pp Vietnamese**. Vietnamese AUPRC của E5 đạt **0,8644**, cao hơn E4 `0,8292` và E3 `0,8156`.

Số đúng/sai tuyệt đối của E5:

| Ngôn ngữ | Gold Safe | Safe đúng | Safe sai → Unsafe | Gold Unsafe | Unsafe đúng | Unsafe sai → Safe | Tổng đúng | Tổng sai |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| English | 2.534 | 2.057 (81,18%) | 477 (18,82%) | 2.807 | 2.214 (78,87%) | 593 (21,13%) | 4.271 | 1.070 |
| Tiếng Việt | 2.534 | 1.990 (78,53%) | 544 (21,47%) | 2.807 | 2.133 (75,99%) | 674 (24,01%) | 4.123 | 1.218 |

So với E4, E5 sửa đúng thêm **66 Safe + 68 Unsafe ở English**, và **87 Safe + 41 Unsafe ở Vietnamese**. Do đó mức tăng của E5 không đến từ việc đổi lỗi từ lớp này sang lớp kia; cả hai lớp đều tăng số dự đoán đúng.

Trên SEA paired, E5 cũng đứng đầu: overall `74,13%`, English `75,22%`, Vietnamese `73,04%`, Macro-F1 `0,7380`, AUPRC `0,7957`. Consistency `85,11%` thấp hơn E4 rất nhẹ `0,38 pp`, nhưng accuracy/F1/AUPRC đều cao hơn.

### N23 · Nemotron test

N23 chỉ được tính trên 5.768 mẫu có category supervision, gồm 2.884 English và 2.884 Vietnamese. Đây là bài toán multi-label độc lập, threshold `0,5` cho từng nhãn.

| Run | Micro precision | Micro recall | Micro-F1 | Macro-F1 | Exact match | EN Micro-F1 | VI Micro-F1 | EN Macro-F1 | VI Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | **0,7485** | 0,1410 | 0,2373 | 0,0826 | 42,75% | 0,2703 | 0,2022 | 0,0966 | 0,0673 |
| E4 | 0,7350 | 0,1410 | 0,2366 | 0,0702 | 42,22% | 0,2389 | 0,2343 | 0,0731 | 0,0673 |
| E5 | 0,7216 | **0,2702** | **0,3931** | **0,2042** | **46,41%** | **0,4084** | **0,3775** | **0,2219** | **0,1849** |

E5 tăng N23 Micro-F1 khoảng **15,59 pp so với E3** và **15,66 pp so với E4**. Macro-F1 tăng lần lượt **12,16 pp** và **13,39 pp**. Tuy nhiên N23 vẫn thiên về precision: có **6.966 gold-positive labels** nhưng chỉ dự đoán **2.608 positive labels**, nên micro recall mới `27,02%`.

Các nhãn E5 làm tương đối tốt nhất gồm Criminal Planning/Confessions (`F1 0,6599`), Controlled/Regulated Substances (`0,5978`), Guns and Illegal Weapons (`0,5729`) và PII/Privacy (`0,5272`). Các nhãn Political/Misinformation/Conspiracy, Fraud/Deception, Threat, Other, Malware, Copyright/Trademark/Plagiarism và High Risk Gov Decision Making chưa có dự đoán positive nào ở threshold `0,5`; cần tune threshold theo nhãn hoặc xử lý mất cân bằng lớp.

## E1 vs E2 — GLi-compatible

### Primary native · Nemotron test (4.238 EN + 4.238 VI)

| Run | Overall accuracy | EN accuracy | VI accuracy | VI − EN | EN Macro-F1 | VI Macro-F1 | EN–VI consistency |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 GLiGuard 512 | 68,70% | **79,16%** | 58,24% | −20,93 pp | **0,7854** | 0,4941 | 69,07% |
| E2 mmBERT GLi-compatible | **77,18%** | 78,15% | **76,22%** | **−1,93 pp** | 0,7812 | **0,7617** | **86,60%** |

E2 tăng overall **8,48 pp** và Vietnamese accuracy **17,98 pp**; English giảm nhẹ **1,01 pp**. Khoảng cách ngôn ngữ thu hẹp khoảng **19,00 pp**.

E1 có VI unsafe recall 95,11%, nhưng VI accuracy chỉ 58,24% và Macro-F1 0,4941. Đây là dấu hiệu lệch về lớp Unsafe. E2 giảm VI unsafe recall xuống 76,44% nhưng phân loại cân bằng hơn nhiều.

### Primary native · SEA paired (1.573 EN + 1.573 VI)

| Run | Overall accuracy | EN accuracy | VI accuracy | VI − EN | EN Macro-F1 | VI Macro-F1 | EN–VI consistency |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 GLiGuard 512 | 65,00% | **78,58%** | 51,43% | −27,15 pp | **0,7819** | 0,4039 | 65,73% |
| E2 mmBERT GLi-compatible | **72,98%** | 74,00% | **71,96%** | **−2,03 pp** | 0,7398 | **0,7177** | **84,49%** |

E2 tăng overall **7,98 pp** và Vietnamese accuracy **20,53 pp**, trong khi English giảm **4,58 pp**. Kết quả `secondary_shared_truncated` lặp lại cùng xu hướng: E2 hơn E1 khoảng **7,99 pp** overall trên Nemotron test và **7,88 pp** trên SEA paired.

## Cách đọc kết quả cho quyết định mô hình

1. Dùng **Macro-F1 + unsafe recall + AUPRC**, không chỉ accuracy. Accuracy có thể tăng vì model bớt báo Unsafe, trong khi false-negative cũng tăng.
2. Dùng **VI − EN gap** và **paired consistency** để đo mức cân bằng ngôn ngữ. Consistency cao không tự động đồng nghĩa đúng; nó phải đi cùng accuracy/F1 của từng ngôn ngữ.
3. Với E3/E4, bằng chứng hiện tại ủng hộ dữ liệu bilingual vì VI accuracy, VI Macro-F1 và consistency đều tăng trên hai benchmark. Tuy nhiên cần tune threshold hoặc cost-sensitive objective nếu ưu tiên bắt Unsafe.
4. E5 là ứng viên fixed-head tốt nhất. E7 canonical/reversed đã hoàn tất và cho thấy label-order agreement yếu; không nên chạy lại cùng recipe. Chẩn đoán đã được sửa theo paper gốc trong `reports/research_archive/GLIGUARD_E7_PROVENANCE_AUDIT_20260722.md`.
