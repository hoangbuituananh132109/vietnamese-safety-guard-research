# Detailed Experiment Evidence — Nemotron Safety EN–VI

> Đây là phụ lục bằng chứng cho CV/portfolio, không phải bản thay thế các raw metrics và predictions. Nó gom các kết quả phân tích chi tiết đã có trong các báo cáo cũ của dự án, đồng thời giữ rõ evaluation contract để không trộn các con số không cùng tập đánh giá.

## 1. Phạm vi và cách đọc số liệu

Dự án có nhiều lớp đánh giá. Vì vậy, hai con số khác nhau của cùng một run không nhất thiết mâu thuẫn:

1. **Phase 0 full-R headline:** các run E3/E4/E5/E7 được đánh giá trên full 8K suite có đủ các view `P`, `PR` và `R` khi contract cho phép. Đây là nguồn của các headline trong báo cáo CV chính: E3 **75,58%/71,82%**, E4 **76,13%/72,58%**, E5 **78,58%/74,13%**, E7 **56,13%/54,54%** trên Nemotron/SEA.
2. **Follow-up no-R:** loại response-only `R`, giữ `P` và `PR`, đồng thời train trên 101.274 thay vì 140.136 instances. SEA vẫn giữ cùng 3.680 mẫu nên phù hợp hơn để xem tác động của contract; Nemotron test thay population từ 10.682 xuống 8.056 nên chỉ được diễn giải mô tả, không gọi là paired causal ablation.
3. **Common decoder no-R:** Q1/Q2/D1/D2/D3 được so trên exact common set **11.736** mẫu P/PR, gồm 8.056 Nemotron test và 3.680 SEA paired; 5.868 EN và 5.868 VI. Qwen chỉ được chấm binary Safe/Unsafe vì taxonomy native không tương đương N23.

Không được lấy một cột full-R của E5 so trực tiếp với một cột no-R của Q2 mà không ghi evaluation contract.

## 2. Translation and data-readiness evidence

| Hạng mục | Kết quả đã khóa |
|---|---:|
| Source records | 45.416 |
| EN→VI coverage | 45.416/45.416, 100% |
| Train/valid/test records | 40.007 / 2.445 / 2.964 |
| Atomic safety categories | 23 |
| Category combinations | 1.669 |
| Missing/extra/duplicate UID | 0 |
| Structural validator failures | 0 |
| Source/translation hash mismatches | 0 |
| Difficult/revision queue | 478/478 completed |
| Direct pass + audited override | 45.330 + 86 |
| Unexplained hard-validator errors | 0 |

### Phân bố trạng thái xử lý

| Trạng thái | Record | Tỷ lệ |
|---|---:|---:|
| Strict machine pass | 41.245 | 90,8160% |
| Salvaged clean | 3.526 | 7,7638% |
| Deterministic format repair | 43 | 0,0947% |
| Terra revised | 336 | 0,7398% |
| Luna revised | 113 | 0,2488% |
| Gemini revised | 29 | 0,0639% |
| Codex quality repaired/approved | 60 | 0,1321% |
| Codex profanity repaired/approved | 64 | 0,1407% |

Luna xuất hiện ở đây chỉ như một trạng thái revision trong lịch sử QA/dịch. Theo scope CV hiện tại, các bản dịch Luna/Sol mới hoàn thành chưa được dùng làm nguồn của kết quả train/evaluation; kết quả mô hình được báo cáo ở đây dùng corpus Gemini đã hoàn tất.

### Một số nhóm safety có nhiều revision hơn

Do một record có thể có nhiều atomic labels, tổng các dòng không cộng thành 45.416.

| Nhãn | Tổng record | Đã revised | Tỷ lệ revised |
|---|---:|---:|---:|
| Criminal Planning/Confessions | 11.100 | 108 | 0,97% |
| Needs Caution | 5.194 | 96 | 1,85% |
| Hate/Identity Hate | 4.194 | 69 | 1,65% |
| Violence | 4.169 | 56 | 1,34% |
| Harassment | 3.623 | 41 | 1,13% |
| Controlled/Regulated Substances | 3.333 | 59 | 1,77% |
| PII/Privacy | 2.780 | 45 | 1,62% |
| Profanity | 2.639 | 118 | 4,47% |
| Immoral/Unethical | 2.203 | 68 | 3,09% |
| Illegal Activity | 2.132 | 58 | 2,72% |

Đây là evidence về coverage, provenance, validator và review completion; không phải BLEU/COMET hoặc bằng chứng human-equivalent cho mọi câu.

## 3. Encoder — full-R headline results

| Run | Training condition | Nemotron | SEA | N23 Micro-F1 | N23 Macro-F1 |
|---|---|---:|---:|---:|---:|
| E3 | English-only, mmBERT fixed-head | 75,58% | 71,82% | 0,2373 | 0,0826 |
| E4 | Matched EN–VI, mmBERT fixed-head | 76,13% | 72,58% | 0,2366 | 0,0702 |
| **E5** | **Full EN+VI, mmBERT fixed-head** | **78,58%** | **74,13%** | **0,3931** | **0,2042** |
| E7 | Full EN+VI, dynamic schema | 56,13% | 54,54% | 0,0000 | 0,0000 |

Diễn giải full-R:

- E3 → E4: bổ sung dữ liệu VI theo matched budget cải thiện nhẹ binary và giúp đánh giá VI tốt hơn trong các slice language.
- E4 → E5: dùng toàn bộ EN+VI là điều kiện fixed-head mạnh nhất; tăng **+2,45 pp Nemotron** và **+1,55 pp SEA** so với E4.
- E7 hoàn tất về mặt chạy GPU nhưng negative về chất lượng; không được mô tả là thành công chỉ vì pipeline không crash.

## 4. Encoder — R/no-R follow-up chi tiết

### 4.1 Binary theo benchmark

| Run | Nemotron có R | Nemotron no-R | Δ mô tả | SEA có R | SEA no-R | Δ |
|---|---:|---:|---:|---:|---:|---:|
| E1 GLiGuard | 68,70 | 69,07 | +0,37* | 65,00 | 64,88 | −0,12 |
| E2 mmBERT GLi-compatible | 77,18 | 76,89 | −0,29* | 72,98 | 72,25 | −0,73 |
| E3 English-only | 75,58 | 73,80 | −1,78* | 71,82 | 71,11 | −0,71 |
| E4 matched EN+VI | 76,13 | 74,76 | −1,37* | 72,58 | 71,47 | −1,11 |
| **E5 full EN+VI** | **78,58** | **78,26** | −0,32* | **74,13** | **73,34** | −0,79 |
| E7 dynamic schema | 56,13 | 56,62 | +0,49* | 54,54 | 48,34 | **−6,20** |

`*` Nemotron test thay đổi thành phần khi no-R, nên các Δ Nemotron là descriptive only. SEA dùng cùng 3.680 mẫu và là đối chứng đáng tin hơn.

### 4.2 Xếp hạng no-R theo nhiều metric

| Run | Benchmark | Accuracy | Macro-F1 | EN | VI | VI−EN | Unsafe recall | AUPRC | EN–VI agreement |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | Nemotron | 73,80 | 73,16 | 76,66 | 70,93 | −5,73 | **83,06** | 83,11 | 79,92 |
| E4 | Nemotron | 74,76 | 74,67 | 75,55 | 73,98 | −1,56 | 75,40 | 83,15 | 86,72 |
| **E5** | **Nemotron** | **78,26** | **78,16** | **79,74** | **76,79** | −2,95 | 79,43 | **86,90** | 86,92 |
| E6 | Nemotron | 78,19 | 78,07 | 79,62 | 76,76 | −2,86 | 79,57 | 86,80 | **87,02** |
| E7 | Nemotron | 56,62 | 56,04 | 56,63 | 56,60 | **−0,03** | 42,09 | 64,41 | 79,47 |
| E3 | SEA | 71,11 | 71,04 | 73,32 | 68,91 | −4,40 | 70,53 | 75,58 | 82,12 |
| E4 | SEA | 71,47 | 71,24 | 72,61 | 70,33 | −2,28 | 66,71 | 75,54 | 85,76 |
| **E5** | **SEA** | **73,34** | **73,00** | **74,51** | 72,17 | −2,34 | 66,30 | 78,33 | 85,71 |
| E6 | SEA | 73,40 | 73,11 | 74,46 | **72,34** | −2,12 | 67,29 | **78,82** | **86,03** |
| E7 | SEA | 48,34 | 44,36 | 49,02 | 47,66 | −1,36 | **80,16** | 48,13 | 74,18 |

### 4.3 E3 → E4 → E5: tác động của dữ liệu Việt và quy mô

- E3 → E4: Nemotron VI tăng **+3,05 pp** trong no-R; SEA VI tăng **+1,41 pp**; language gap giảm rõ.
- E4 → E5: Nemotron tăng **+3,50 pp** và SEA tăng **+1,88 pp** trong no-R; lợi ích xuất hiện ở cả EN và VI.
- E3 có unsafe recall cao hơn nhưng accuracy/Macro-F1 thấp hơn, nghĩa là model thiên về cảnh báo Unsafe và không đồng nghĩa với việc hiểu safety tốt hơn.

### 4.4 P versus PR — vấn đề deployment

| Run | Benchmark | View | N | Accuracy | Macro-F1 | Safe recall | Unsafe recall | AUPRC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E5 | Nemotron | P | 5.430 | 78,21 | 77,83 | 73,83 | 81,67 | 87,85 |
| E5 | Nemotron | PR | 2.626 | 78,37 | 78,30 | 82,44 | 74,15 | 85,06 |
| E5 | SEA | P | 2.260 | 73,85 | 73,72 | 70,39 | 76,98 | 82,08 |
| **E5** | **SEA** | **PR** | **1.420** | **72,54** | **67,26** | **90,70** | **42,75** | **69,07** |
| E6 | SEA | P | 2.260 | 74,07 | 73,89 | 69,18 | 78,50 | 82,92 |
| E6 | SEA | PR | 1.420 | 72,32 | 67,03 | 90,48 | **42,57** | 68,61 |

Điểm quan trọng: SEA PR có accuracy khoảng 72% nhưng unsafe recall chỉ khoảng 43%; guard sau output không thể đánh giá chỉ bằng accuracy tổng. Cần threshold/calibration riêng cho P và PR, đồng thời báo cáo false-negative/unsafe recall.

### 4.5 E5 versus E6 — ablation N23 auxiliary loss

E5 và E6 no-R dùng cùng 101.274 train instances, 6.330 steps, seed 3407, effective batch 32, LoRA rank 4, max length 8.192 và cùng evaluation IDs. Khác biệt chính là E5 có binary + N23 supervision, còn E6 binary-only.

| Benchmark | E5 accuracy | E6 accuracy | E6−E5 | E5 Macro-F1 | E6 Macro-F1 | E5 AUPRC | E6 AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nemotron | 78,265% | 78,190% | −0,075 pp | 78,156% | 78,075% | 86,895% | 86,797% |
| SEA | 73,342% | 73,397% | +0,054 pp | 73,002% | 73,108% | 78,326% | 78,817% |

McNemar exact:

- Nemotron: E5-only correct 127, E6-only correct 121, `p=0,751`.
- SEA: E5-only correct 50, E6-only correct 52, `p=0,921`.

Kết luận: E5 và E6 gần như hòa về binary; E5 vẫn có giá trị vì giữ được N23, còn E6 chỉ đơn giản hơn về output.

### 4.6 N23 theo nhãn và calibration

| Run | Micro-F1 có R | Micro-F1 no-R | Macro-F1 có R | Macro-F1 no-R | Exact có R | Exact no-R |
|---|---:|---:|---:|---:|---:|---:|
| E3 | 0,2373 | 0,1816 | 0,0826 | 0,0462 | 42,75% | 41,92% |
| E4 | 0,2366 | 0,1768 | 0,0702 | 0,0327 | 42,22% | 41,19% |
| **E5** | **0,3931** | **0,3442** | **0,2042** | **0,1576** | **46,41%** | **44,89%** |
| E7 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | — | 37,86% |

Trong no-R E5, model dự đoán 2.186 positive trên 6.966 gold-positive; precision micro **0,7205** nhưng recall chỉ **0,2261** ở threshold 0,5. Các nhãn mạnh hơn gồm Criminal Planning/Confessions (F1 0,646), Controlled/Regulated Substances (0,539), Guns and Illegal Weapons (0,504), PII/Privacy (0,450) và Sexual (0,372). Nhiều nhãn hiếm không có positive prediction ở threshold 0,5 nhưng vẫn có AUPRC trên baseline tần suất, cho thấy vấn đề gồm mất cân bằng và calibration chứ không thể kết luận head không học.

### 4.7 E7 dynamic-schema diagnosis

| Chỉ số | E7 có R | E7 no-R |
|---|---:|---:|
| Nemotron accuracy | 56,13% | 56,62%* |
| SEA accuracy | 54,54% | 48,34% |
| N23 Micro-F1 @0,5 | 0 | 0 |
| Nemotron order disagreement | 22,00% | 16,31% |
| SEA order disagreement | 33,53% | 19,92% |

E7 no-R có order stability tốt hơn nhưng semantic safety mapping yếu: Nemotron unsafe recall **42,09%**, SEA unsafe recall **80,16%**, SEA AUPRC **48,13%**, ECE **16,51%**. Một validation threshold 0,12 có thể khôi phục một phần N23 ranking signal (test Micro-F1 0,218; Macro-F1 0,074), nhưng vẫn thấp hơn E5 và không biến E7 thành một kết quả thành công. Đây là negative/diagnostic result về recipe và calibration, không phải lỗi pipeline.

## 5. Decoder — Qwen, Nemotron và phân tích theo slice

### 5.1 Common no-R leaderboard

| System | Overall accuracy | Macro-F1 | Safe recall | Unsafe recall |
|---|---:|---:|---:|---:|
| **Q2 Qwen EN–VI LoRA** | **87,8323%** | **87,8230%** | 87,7946% | 87,8678% |
| D1 Nemotron v3 zero-shot | 86,9547% | 86,9539% | **90,5206%** | 83,6033% |
| D2 Nemotron pilot LoRA | 86,8098% | 86,8097% | 89,8347% | 83,9669% |
| Q1 Qwen zero-shot | 85,1483% | 84,9440% | 75,8530% | **93,8843%** |
| D3 Nemotron full VI LoRA | 86,52% | 86,51% | **92,12%** | 81,26% |

Q2 so với Q1: 694 mẫu Q2 đúng/Q1 sai, 379 mẫu Q1 đúng/Q2 sai, net +315, McNemar exact `p=4,65e-22`.

Q2 so với D1: 601 mẫu Q2 đúng/D1 sai, 498 mẫu D1 đúng/Q2 sai, net +103, McNemar exact `p=0,002078`. Q2 hơn D1 **+0,88 pp overall**, nhưng D1 vẫn có Safe recall cao hơn.

### 5.2 Q1 → Q2 theo language, benchmark, view

| Slice | N | Q1 | Q2 | Q2−Q1 |
|---|---:|---:|---:|---:|
| Overall | 11.736 | 85,15% | **87,83%** | +2,68 pp |
| English | 5.868 | 86,06% | **88,45%** | +2,39 pp |
| Vietnamese | 5.868 | 84,24% | **87,22%** | +2,98 pp |
| Nemotron test | 8.056 | 85,86% | **88,49%** | +2,63 pp |
| SEA paired | 3.680 | 83,59% | **86,39%** | +2,80 pp |
| Prompt-only P | 7.690 | 84,92% | **87,13%** | +2,21 pp |
| Prompt+response PR | 4.046 | 85,59% | **89,17%** | +3,58 pp |

| Benchmark | Lang | View | N | Q1 | Q2 | Q2−D1 |
|---|---|---|---:|---:|---:|---:|
| Nemotron | EN | P | 2.715 | 85,97% | **88,77%** | +0,63 pp |
| Nemotron | EN | PR | 1.313 | 88,65% | **90,18%** | −0,53 pp |
| Nemotron | VI | P | 2.715 | 83,72% | **86,92%** | +0,15 pp |
| Nemotron | VI | PR | 1.313 | 87,28% | 89,49% | −0,15 pp |
| SEA | EN | P | 1.130 | **86,28%** | 85,66% | +2,65 pp |
| SEA | EN | PR | 710 | 81,27% | **88,45%** | +1,27 pp |
| SEA | VI | P | 1.130 | 83,89% | **85,13%** | +3,10 pp |
| SEA | VI | PR | 710 | 81,13% | **87,46%** | +2,39 pp |

Q2 không thắng mọi ô: D1 vẫn nhỉnh hơn trên Nemotron PR ở cả hai ngôn ngữ; Q1 vẫn tốt nhất trên SEA EN prompt-only. Lợi ích rõ nhất của Q2 là SEA VI và SEA PR.

### 5.3 Qwen confusion matrix

| System | Safe đúng | Safe→Unsafe | Unsafe→Safe | Unsafe đúng |
|---|---:|---:|---:|---:|
| Q1 | 4.313 | 1.373 | **370** | **5.680** |
| **Q2** | **4.992** | **694** | 734 | 5.316 |
| D1 | **5.147** | **539** | 992 | 5.058 |
| D2 | 5.108 | 578 | 970 | 5.080 |

Q1 bắt Unsafe mạnh nhất nhưng false-block Safe nhiều. Q2 có predicted unsafe prevalence **51,21%**, gần gold prevalence **51,55%** nhất trong bốn system; đây là lý do Q2 có Macro-F1 cân bằng hơn.

### 5.4 Topic slices Q1/Q2/D1/D2

| Topic | N | Q1 | Q2 | D1 | D2 |
|---|---:|---:|---:|---:|---:|
| Cultural content generation | 440 | 82,05% | **87,05%** | 82,05% | 84,55% |
| Cultural in-the-wild | 840 | **92,98%** | 88,45% | 82,74% | 80,71% |
| General | 2.400 | 80,58% | **85,54%** | 84,67% | 84,50% |
| Generic | 5.456 | 83,25% | **85,91%** | 85,65% | 85,28% |
| Jailbreaking | 2.600 | 91,35% | 93,92% | 94,00% | **94,50%** |

Q2 mạnh hơn Q1 ở jailbreaking nhưng D1/D2 vẫn nhỉnh hơn nhẹ; Q1 đặc biệt mạnh ở Cultural in-the-wild. Không nên tóm tắt Q2 là tốt nhất ở mọi topic.

### 5.5 N23 chỉ áp dụng cho D1/D2/D3

Q1/Q2 native category output không được ép vào N23. Trên 5.768 Nemotron P/PR có N23 supervision:

| System | Exact match | Hamming error | Micro-F1 | Macro-F1 |
|---|---:|---:|---:|---:|
| D1 | **52,31%** | 4,017% | 57,83% | 44,69% |
| D2 | 51,73% | **3,978%** | **58,20%** | **47,49%** |
| D3 | 50,85% | **3,920%** | 56,63% | **48,51%** |

D2 tăng N23 Macro-F1, đặc biệt ở VI, dù binary accuracy không tăng. D3 có Macro-F1 tổng cao hơn nhưng gain tập trung ở PR; D3 P chỉ có Micro-F1 49,83% và Macro-F1 39,84%. Đây là lý do binary và N23 phải được báo cáo riêng.

## 6. Archived reports and machine-readable evidence

Các file cũ không bị mất; chúng là nguồn chi tiết của phụ lục này:

- `reports/research_archive/PROJECT_FINAL_SYNTHESIS_ENCODERS_QWEN_NEMOTRON_PUBLISHED_20260724.md` — tổng kết cuối encoder, Qwen, Nemotron D1/D2/D3, common leaderboard, N23 và claim boundaries.
- `reports/research_archive/Q2_VS_Q1_D1_D2_NO_R_20260724.md` — Qwen Q1/Q2, D1/D2: confusion matrix, language/benchmark/view intersections, topic slices, paired McNemar và N23 D1/D2.
- `reports/research_archive/Q2_VS_Q1_D2_NO_R_20260724.md` — Q2 đối chiếu thêm D3 Nemotron full-VI LoRA, binary slices, paired correctness và N23 theo view.
- `reports/research_archive/D3_NEMOTRON_NO_R_FINAL_ANALYSIS_20260724.md` — D3 training/evaluation contract, binary trade-off và N23 D1/D2/D3.
- `reports/analysis_20260724/R_VS_NO_R_E6_E7_COMPREHENSIVE_ANALYSIS_20260724.md` — full-R/no-R, E3/E4/E5/E6/E7, per-view P/PR, N23 labels, AUPRC/ECE, McNemar và deployment caveats.
- `reports/research_archive/PROJECT_EXPERIMENT_SYNTHESIS_20260722.md` — contract, định nghĩa metric, experiment matrix, error counts và runtime trade-offs.
- `reports/research_archive/QWEN3GUARD_Q1_ANALYSIS_20260723.md` — Qwen base zero-shot và policy mapping Controversial.
- `reports/phase0/E7_SCHEMA_ANALYSIS.md` — E7 order invariance, calibration, threshold và N23 collapse.
- `reports/final_quality/TRANSLATION_QUALITY_REPORT.md` — coverage, integrity, validator, revision queue, split và atomic safety label.

Machine-readable evidence nằm trong:

- `reports/analysis_20260724/no_r/reports/no_r_phase0/experiment_runs/*/metrics.json`;
- `reports/analysis_20260724/no_r/reports/no_r_phase0/evaluation_matrix/*/metrics.json`;
- `reports/vast_download/d3_nemotron_no_r_4080s_20260724/extracted/results/no_r_decoder_4080s/eval/qwen/metrics.json`;
- `reports/final_quality/translation_quality_summary.json` và các bảng CSV theo category/status.

## 7. Claim boundaries cho CV

- Có thể nói dự án xây được evaluation harness chi tiết theo benchmark, language, view, topic, binary/N23, confusion, paired consistency và statistical comparison.
- Có thể nói Q2 là binary leader trên exact common no-R set và tăng rõ so với Q1; không nói Q2 thắng mọi topic hoặc mọi view.
- Có thể nói E5 là fixed-head encoder tốt nhất trong ma trận chính; E7 là negative/diagnostic result; E6 là binary-only ablation.
- Không được trộn full-R headline với no-R follow-up mà không ghi population.
- Không được dùng Qwen native taxonomy để tuyên bố N23.
- Không được dùng accuracy tổng để che SEA PR unsafe recall thấp.
- Không được gọi audit pipeline là human-perfect translation; audit chứng minh coverage, fidelity gates, provenance và review completion.
