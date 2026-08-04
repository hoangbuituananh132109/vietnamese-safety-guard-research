# Tổng kết dự án Nemotron Safety EN–VI: encoder, Qwen, Llama Nemotron và benchmark công bố

Ngày chốt: 24/07/2026.

## 1. Kết luận điều hành

Dự án đã hoàn thành ba đóng góp thực nghiệm có thể bảo vệ được:

1. **Tạo được corpus Nemotron Safety EN–VI hoàn chỉnh:** 45.416/45.416 record có bản dịch, giữ nguyên split, UID, nhãn Safe/Unsafe, N23 và metadata; toàn bộ critical integrity checks đều pass.
2. **Chứng minh dữ liệu Việt có thể được mô hình học và tổng quát ra benchmark ngoài miền:** rõ nhất ở Qwen3Guard-Gen-4B Q2, nơi fine-tuning EN–VI nâng accuracy từ 85,15% lên 87,83% trên đúng 11.736 mẫu no-R, đồng thời đạt 86,39% trên SEA paired.
3. **Xây được một evaluation harness thống nhất cho encoder và decoder:** cùng test IDs, EN/VI, P/PR, Safe/Unsafe, N23, topic, paired consistency, parse failures và kiểm định McNemar.

Kết luận mô hình:

- **Q2 là binary quality leader** trên common no-R set: 87,83% accuracy, 87,82% Macro-F1.
- **D1 Llama Nemotron v3 zero-shot vẫn là structured-output baseline mạnh nhất:** 86,95% binary và N23 Micro/Macro-F1 57,83%/44,69%.
- **D2/D3 không cải thiện binary Nemotron v3**, nhưng có các cải thiện N23 cục bộ.
- **E5/E6 là encoder thực dụng:** chỉ khoảng 76,7% trên common set nhưng inference nhanh hơn decoder nhiều; E5 giữ N23, E6 binary-only.
- **E7 dynamic schema thất bại với recipe rẻ hiện tại**, không phải lỗi thực thi.
- Bỏ R không giúp fixed-head và không cứu E7; no-R chủ yếu giúp contract khớp deployment P/PR hơn.

## 2. Tài sản dữ liệu hiện có

### Corpus dịch

- Source records: 45.416.
- Train: 40.007 record.
- Validation: 2.445 record.
- Test: 2.964 record.
- Coverage: 100%.
- Ready theo gate hiện tại: 100%.
- Missing/extra/duplicate UID: 0.
- Structural validator failure: 0.
- Source/translation hash mismatch: 0.
- Atomic safety labels: 23.
- Category combinations: 1.669.
- Hard review queue đã xử lý: 478/478.

Phân bố chất lượng:

| Trạng thái | Record | Tỷ lệ |
|---|---:|---:|
| Strict machine pass | 41.245 | 90,816% |
| Salvaged clean | 3.526 | 7,764% |
| Deterministic format repair | 43 | 0,095% |
| Terra revised | 336 | 0,740% |
| Luna revised | 113 | 0,249% |
| Gemini revised | 29 | 0,064% |
| Codex quality repaired/approved | 60 | 0,132% |
| Codex profanity repaired/approved | 64 | 0,141% |

Đây là chất lượng vận hành và auditability, không phải đánh giá BLEU/COMET hoặc human-equivalence trên toàn bộ 45.416 record.

### Record và training instance không phải một

Một record có thể sinh các view:

- `P`: prompt-only;
- `PR`: prompt + response;
- `R`: response-only.

Full có R tạo 140.136 train instance. Full no-R P/PR tạo 101.274 train instance.

## 3. Bản đồ thí nghiệm

### Encoder

| ID | Model | Dữ liệu | Output |
|---|---|---|---|
| E1 | GLiGuard trained | GLi-compatible EN–VI, 512 | Dynamic binary |
| E2 | mmBERT fixed-head | Cùng common subset với E1 | Binary |
| E3 | mmBERT fixed-head | English-only | Binary + N23 |
| E4 | mmBERT fixed-head | Matched EN–VI | Binary + N23 |
| E5 | mmBERT fixed-head | Full EN+VI | Binary + N23 |
| E6 | mmBERT fixed-head | Full no-R EN+VI | Binary-only |
| E7 | mmBERT `[L]` schema scorer | Full EN+VI | Dynamic binary + N23 |

### Decoder

| ID | Model | Local training | Output |
|---|---|---|---|
| Q1 | Qwen3Guard-Gen-4B | Zero-shot | Safe/Controversial/Unsafe |
| Q2 | Qwen3Guard-Gen-4B LoRA | 101.274 no-R EN–VI, context 2.048 | Binary Safe/Unsafe |
| D1 | Llama-3.1-Nemotron-Safety-Guard-8B-v3 | Zero-shot released model | Official JSON + N23 |
| D2 | Nemotron v3 pilot LoRA | 8.192 balanced EN–VI P/R/PR, context 1.024 | Official JSON + N23 |
| D3 | Nemotron v3 full VI LoRA | 50.637 VI no-R P/PR, context 2.048, 1 epoch | Official JSON + N23 |

## 4. Leaderboard trên exact common no-R EN–VI set

Population:

- 11.736 exact IDs;
- 5.868 EN + 5.868 VI;
- 8.056 Nemotron test + 3.680 SEA paired;
- 7.690 P + 4.046 PR;
- không có R;
- zero target mismatch.

| System | Parameters/architecture | Accuracy | Macro-F1 | Safe recall | Unsafe recall |
|---|---|---:|---:|---:|---:|
| **Q2 Qwen EN–VI LoRA** | Decoder 4B | **87,83%** | **87,82%** | 87,79% | 87,87% |
| D1 Nemotron v3 zero-shot | Decoder 8B | 86,95% | 86,95% | 90,52% | 83,60% |
| D2 Nemotron pilot | Decoder 8B | 86,81% | 86,81% | 89,83% | 83,97% |
| D3 Nemotron full VI | Decoder 8B | 86,52% | 86,51% | **92,12%** | 81,26% |
| Q1 Qwen zero-shot | Decoder 4B | 85,15% | 84,94% | 75,85% | **93,88%** |
| E5 no-R | mmBERT encoder | 76,72% | 76,72% | 77,82% | 75,69% |
| E6 no-R binary-only | mmBERT encoder | 76,69% | 76,68% | 77,35% | 76,07% |

E7 đạt khoảng 54% trên combined common set và không cạnh tranh.

### Ý nghĩa

- Q1 bắt Unsafe rất mạnh nhưng false-block Safe cao.
- D1/D2/D3 thiên Safe hơn và bỏ lọt nhiều Unsafe hơn Q2.
- Q2 cân bằng hai class tốt nhất.
- E5/E6 thấp hơn decoder khoảng 8–11 điểm, nhưng encoder trực tiếp phát logits, không phải autoregressively sinh JSON.

## 5. Qwen: đóng góp rõ nhất của dữ liệu dịch

### Q1 → Q2

| Slice | Q1 zero-shot | Q2 EN–VI LoRA | Gain |
|---|---:|---:|---:|
| Overall | 85,15% | **87,83%** | +2,68 pp |
| English | 86,06% | **88,45%** | +2,39 pp |
| Vietnamese | 84,24% | **87,22%** | +2,98 pp |
| Nemotron | 85,86% | **88,49%** | +2,63 pp |
| SEA | 83,59% | **86,39%** | +2,80 pp |
| P | 84,92% | **87,13%** | +2,21 pp |
| PR | 85,59% | **89,17%** | +3,58 pp |

Q2 vs Q1:

- Q2 đúng/Q1 sai: 694;
- Q1 đúng/Q2 sai: 379;
- net +315;
- McNemar exact `p = 4,65e-22`.

Đây là bằng chứng rất mạnh rằng dataset no-R EN–VI có supervision hữu ích đối với Qwen.

### Q2 so với D1

- Overall: Q2 hơn 0,88 pp.
- EN: +0,84 pp.
- VI: +0,92 pp.
- Nemotron: +0,15 pp.
- SEA: +2,47 pp.
- SEA VI: +2,83 pp.
- McNemar exact overall: `p = 0,002078`.

Lợi thế lớn nhất của Q2 nằm ở benchmark ngoài miền SEA, không phải chỉ Nemotron in-family.

Giới hạn:

- Q2 chỉ output binary.
- Qwen taxonomy native không phải Nemotron N23.
- Không được ánh xạ category tùy ý rồi gọi đó là N23.

## 6. Llama Nemotron D1/D2/D3

### Binary

| System | Overall | EN | VI | Nemotron | SEA | P | PR |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 zero-shot | **86,95%** | **87,61%** | **86,30%** | **88,34%** | **83,91%** | **86,01%** | **88,75%** |
| D2 pilot | 86,81% | 87,41% | 86,21% | 88,26% | 83,64% | 85,85% | 88,63% |
| D3 full VI | 86,52% | 87,00% | 86,04% | 87,88% | 83,53% | 85,47% | 88,51% |

D2 vs D1:

- binary accuracy −0,14 pp;
- McNemar `p = 0,431`;
- không khác biệt có ý nghĩa.

D3 vs D1:

- binary −0,43 pp;
- D3 đúng/D1 sai 156, D1 đúng/D3 sai 207;
- McNemar `p = 0,00859`;
- D3 kém D1 có ý nghĩa.

D3 VI-only làm model thiên Safe hơn:

- safe recall tăng lên 92,12%;
- unsafe recall giảm xuống 81,26%.

Vì vậy loss gần 0 không chứng minh decision boundary tốt hơn.

### N23

| System | Exact match | Hamming error | Micro-F1 | Macro-F1 |
|---|---:|---:|---:|---:|
| E5 có R | 46,41% | 4,38% | 39,31% | 20,42% |
| E5 no-R | 44,89% | 4,52% | 34,42% | 15,76% |
| D1 | **52,31%** | 4,017% | 57,83% | 44,69% |
| D2 | 51,73% | 3,978% | **58,20%** | 47,49% |
| D3 | 50,85% | **3,920%** | 56,63% | **48,51%** |

Diễn giải:

- D2 tăng Macro-F1, nhất là VI.
- D3 có Macro-F1 tổng cao nhất, nhưng gain tập trung ở PR.
- D3 PR: Micro-F1 62,20%, Macro-F1 52,78%.
- D3 P: Micro-F1 chỉ 49,83%, Macro-F1 39,84%.
- VI-only training không tạo VI N23 gain thống nhất; D3 VI còn kém D2.

### Vì sao D3 không cải thiện như Q2?

D1 đã được NVIDIA train trên 386.661 mẫu, 9 ngôn ngữ, 5 epoch bằng LoRA rank 8/alpha 32. Khả năng multilingual của nó đã mạnh và gần bão hòa trên VI zero-shot.

D3 chỉ:

- 50.637 VI instance;
- 1 epoch;
- một ngôn ngữ;
- context 2.048;
- distribution 55,04% Safe;
- vừa học binary vừa học structured N23.

Nó không phải full reproduction/retraining của paper. Qwen Q1 còn nhiều chỗ để adaptation binary phát huy, trong khi D1 đã có safety policy và taxonomy rất mạnh.

## 7. Encoder: kết luận chính

### Có R

| Run | Nemotron | SEA | N23 Micro-F1 | N23 Macro-F1 |
|---|---:|---:|---:|---:|
| E3 English-only | 75,58% | 71,82% | 23,73% | 8,26% |
| E4 matched EN–VI | 76,13% | 72,58% | 23,66% | 7,02% |
| **E5 full EN+VI** | **78,58%** | **74,13%** | **39,31%** | **20,42%** |
| E7 dynamic | 56,13% | 54,54% | 0% | 0% |

### No-R

- E5 và E6 hòa binary: McNemar `p=0,751` Nemotron, `p=0,921` SEA.
- N23 auxiliary không làm E5 binary kém đi.
- E6 chỉ nhanh hơn E5 khoảng 2%; encoder chi phối compute.
- Bỏ R làm E3/E4/E5 giảm nhẹ trên SEA và làm N23 giảm.
- E7 no-R có schema-order stability tốt hơn nhưng SEA giảm 6,20 pp và N23 vẫn collapse.

Kết luận:

- E5 là encoder nghiên cứu chính.
- E6 là binary-only deployment checkpoint.
- E7 là negative result có giá trị, chưa phải bác bỏ kiến trúc schema-conditioned.

## 8. So với benchmark NVIDIA công bố

### Metric phải so đúng

NVIDIA công bố **harmful-F1**, tức F1 của class harmful/unsafe; không phải accuracy và cũng không phải Macro-F1.

Model card v3 công bố:

| Benchmark | Harmful-F1 công bố |
|---|---:|
| Nemotron-Safety-Guard-Dataset-v3 | 85,32 |
| PolyGuardPrompts | 76,07 |
| RTP-LX | 91,49 |
| MultiJail | 95,36 |
| XSafety | 66,97 |
| Aya Red-teaming | 96,79 |

Trong paper, `85,32` là trung bình của:

- CultureGuard prompt harmful-F1: 85,15;
- CultureGuard response harmful-F1: 85,48.

Paper đánh giá:

- 17.676 standard CultureGuard test samples;
- 8.883 CultureGuard-JB samples;
- trung bình 9 ngôn ngữ train: EN, AR, DE, ES, FR, HI, JA, TH, ZH.

### Local reproduction trên Nemotron EN–VI

Local no-R P/PR Nemotron test có 8.056 instance:

- 4.028 EN;
- 4.028 VI;
- cùng semantic pairs;
- không có R.

| Model | Local Nemotron accuracy | Local unsafe/harmful-F1 |
|---|---:|---:|
| E5 no-R | 78,26% | 79,69% |
| E6 no-R | 78,19% | 79,67% |
| Q1 zero-shot | 85,86% | 87,81% |
| D3 Nemotron VI LoRA | 87,88% | 88,16% |
| D1 Nemotron v3 zero-shot | 88,34% | 88,83% |
| **Q2 Qwen EN–VI LoRA** | **88,49%** | **89,30%** |

D1 chi tiết theo task:

| Task | Local harmful-F1 | Published v3 harmful-F1 | Local−published |
|---|---:|---:|---:|
| Prompt/P | 88,36% | 85,15% | +3,21 pp |
| Response/PR | 89,88% | 85,48% | +4,40 pp |
| Simple mean P/PR | 89,12% | 85,32% | +3,80 pp |

Đây **không phải** bằng chứng local model “thắng NVIDIA” vì:

1. local chỉ có EN+VI, paper trung bình 9 ngôn ngữ;
2. VI local là bản dịch paired từ source EN;
3. paper test composition và cultural adaptation khác;
4. local instance construction P/PR không hoàn toàn trùng paper;
5. local report còn có accuracy, trong khi paper headline là harmful-F1;
6. paper có cả external benchmarks mà ta chưa chạy.

Diễn giải hợp lý:

- evaluator/prompt D1 local hoạt động đúng và cho điểm cùng vùng, thậm chí cao hơn trên selected EN–VI subset;
- D1 zero-shot tiếng Việt mạnh đáng ngạc nhiên;
- Q2 vượt D1 nhẹ trên local Nemotron và rõ hơn trên SEA;
- cần chạy đúng public suite trước khi so headline leaderboard.

### Vì sao bản có R cho D1 thấp hơn?

D1 full có R:

- overall accuracy 86,86%;
- overall unsafe-F1 86,81%.

Theo view:

- P unsafe-F1 88,36%;
- PR unsafe-F1 89,88%;
- R-only unsafe-F1 chỉ 79,23%.

Official prompt format của NVIDIA đánh giá prompt hoặc prompt+response; response safety được đánh giá với prompt context. R-only là extension riêng của dự án và kéo điểm tổng xuống. Vì vậy no-R P/PR phù hợp hơn khi đối chiếu paper.

## 9. Khoảng trống so với suite công bố

Ta chưa chạy E5/E6/Q1/Q2/D1/D3 trên:

- PolyGuardPrompts;
- RTP-LX;
- MultiJail;
- XSafety;
- Aya Red-teaming.

Do đó không được điền các cột này bằng SEA hoặc đoán từ Nemotron.

SEA paired là benchmark bổ sung độc lập:

| Model | SEA accuracy | SEA unsafe-F1 |
|---|---:|---:|
| E5 no-R | 73,34% | 69,97% |
| Q1 | 83,59% | 83,94% |
| D3 | 83,53% | 80,90% |
| D1 | 83,91% | 81,75% |
| **Q2** | **86,39%** | **85,24%** |

SEA là bằng chứng quan trọng hơn cho Vietnamese/generalization vì không thuộc Nemotron training family.

## 10. Khoảng trống tiếng Việt trong công bố NVIDIA

Tiếng Việt:

- không nằm trong 9 ngôn ngữ train của CultureGuard;
- không xuất hiện trong danh sách 17/20 ngôn ngữ zero-shot được paper liệt kê;
- không có cột VI trong các bảng per-language công bố.

Do đó, phần EN–VI của dự án bổ sung một evaluation axis mà paper/model card chưa báo cáo:

- D1 VI zero-shot: 86,30% combined Nemotron+SEA; 87,71% riêng Nemotron, 83,21% SEA.
- Q1 VI zero-shot: 84,24%.
- Q2 VI after EN–VI LoRA: 87,22%.
- Q2 SEA VI: 86,03%.

Không nên gọi đây là benchmark Việt Nam hoàn chỉnh vì VI test chủ yếu dịch paired và SEA vẫn có phạm vi hạn chế. Nhưng đây là bằng chứng thực nghiệm có kiểm soát về Vietnamese transfer.

## 11. Có thể tuyên bố gì về dự án?

### Có thể tuyên bố

1. Hoàn thành pipeline dịch/audit 45.416 record safety EN→VI với 23 nhãn.
2. Tạo được train/validation/test paired, không làm rò test vào train.
3. Bilingual supervision cải thiện mmBERT so với English-only.
4. Full EN+VI tốt hơn matched subset.
5. Qwen3Guard-4B học tốt dataset dịch và vượt zero-shot Qwen/D1 trên common binary evaluation.
6. D1 Nemotron v3 đã zero-shot tiếng Việt tốt dù VI không nằm trong công bố train/eval.
7. Nemotron adaptation nhỏ/full-VI không tăng binary, nhưng thay đổi N23 theo view.
8. Binary, N23, P và PR phải được báo cáo riêng.

### Chưa thể tuyên bố

1. Dataset đã đạt human-equivalent translation quality trên toàn bộ 45.416 record.
2. Q2 vượt Nemotron v3 trên toàn bộ official multilingual suite.
3. Q2 có N23 tốt hơn D1.
4. SEA đại diện đầy đủ cho safety Việt Nam.
5. E7 chứng minh dynamic schema không khả thi.
6. Bỏ R luôn tốt hơn; kết quả hiện tại cho thấy ngược lại ở training quality.

## 12. Việc tiếp theo có giá trị nhất

1. **Chạy official benchmark suite** cho D1, Q1, Q2 và E5/E6 bằng đúng harmful-F1:
   PolyGuardPrompts, RTP-LX, MultiJail, XSafety, Aya.
2. **Tạo Vietnamese benchmark độc lập:** native-authored, native-reviewed, không chỉ dịch; có P và PR, safe/unsafe, cultural harms và jailbreak riêng.
3. **Q2 replicate causality:** English-only vs matched EN–VI vs full EN–VI cùng recipe/steps.
4. **Q2 structured taxonomy:** không ép taxonomy Qwen sang N23; train explicit Nemotron JSON/N23 head hoặc schema.
5. **Nemotron D4:** EN+VI, balance P/PR và Safe/Unsafe theo từng view; giữ replay EN để tránh lệch.
6. **Threshold/calibration riêng P và PR** cho encoder; đặc biệt SEA PR unsafe recall còn thấp.
7. **E7a dynamic binary-only** và E7b faithful GLiGuard recipe nếu còn mục tiêu schema động.

## 13. Nguồn

### Local

- `reports/final_quality/TRANSLATION_QUALITY_REPORT.md`
- `reports/analysis_20260724/R_VS_NO_R_E6_E7_COMPREHENSIVE_ANALYSIS_20260724.md`
- `reports/research_archive/Q2_VS_Q1_D1_D2_NO_R_20260724.md`
- `reports/research_archive/D3_NEMOTRON_NO_R_FINAL_ANALYSIS_20260724.md`
- `reports/research_archive/PROJECT_EXPERIMENT_SYNTHESIS_20260722.md`
- `reports/phase0/E7_SCHEMA_ANALYSIS.md`

### Official

- NVIDIA model card:
  https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3
- CultureGuard paper:
  https://arxiv.org/abs/2508.01710
- CultureGuard HTML tables:
  https://arxiv.org/html/2508.01710v4
