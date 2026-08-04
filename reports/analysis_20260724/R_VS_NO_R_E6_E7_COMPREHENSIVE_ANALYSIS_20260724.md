# Đánh giá toàn diện dữ liệu có R, no-R, E6 và E7

Ngày chốt số liệu: 24/07/2026.

## Kết luận điều hành

1. **Bỏ view `R` không cải thiện kết quả tổng thể.** Trên SEA paired — phép so sánh sạch nhất vì tập đánh giá vẫn gồm đúng 3.680 mẫu — mọi fixed-head encoder đều giảm nhẹ: E3 `−0,71 pp`, E4 `−1,11 pp`, E5 `−0,79 pp`. N23 cũng giảm rõ ở cả E3/E4/E5.
2. **E5 full EN+VI vẫn là fixed-head đa nhiệm tốt nhất.** Sau khi bỏ R, E5 đạt `78,26%` Nemotron, `73,34%` SEA và N23 Micro-F1 `0,3442`. Nó đứng đầu E3/E4/E5 trên cả binary lẫn N23.
3. **E6 binary-only gần như hòa tuyệt đối với E5 no-R ở bài toán Safe/Unsafe.** E6 đạt `78,19%` Nemotron và `73,40%` SEA. Chênh lệch E6−E5 chỉ `−0,07 pp` và `+0,05 pp`; kiểm định McNemar ghép cặp cho `p=0,751` và `p=0,921`. Không có bằng chứng E6 tốt hơn hoặc kém hơn E5 về binary.
4. **E6 không tiết kiệm compute đáng kể.** E6 mất `59,68 phút`, E5 mất `60,93 phút`; VRAM gần như bằng nhau (`~9.821 MiB`). Encoder chi phối chi phí, head N23 rất nhỏ. E6 chỉ đơn giản hơn về đầu ra, nhưng đánh mất hoàn toàn khả năng N23.
5. **E7 no-R không phải một cải tiến thực chất.** Accuracy Nemotron tăng mô tả `+0,49 pp`, nhưng đây là test set đã đổi thành phần. Trên cùng SEA paired, E7 giảm mạnh `−6,20 pp`, từ `54,54%` xuống `48,34%`; N23 vẫn dự đoán 0 positive ở threshold 0,5.
6. **E7 no-R có một tiến bộ hẹp:** độ bất biến khi đảo thứ tự schema tốt hơn. Tỷ lệ quyết định thay đổi khi đảo Safe/Unsafe giảm từ `22,00%` xuống `16,31%` trên Nemotron và từ `33,53%` xuống `19,92%` trên SEA. Tuy nhiên, một mô hình có thể nhất quán hơn nhưng vẫn nhất quán theo quyết định sai; accuracy/AUPRC cho thấy đúng là trường hợp này.
7. **Điểm yếu deployment đáng chú ý nhất là SEA `PR`.** E5/E6 no-R chỉ có unsafe recall khoảng `42,6–42,8%` trên prompt+response, dù accuracy khoảng `72,3–72,5%`. Nếu dùng guard sau output, cần tune threshold riêng cho PR; không nên dùng thẳng threshold 0,5 chung với P.

## 1. “Có R” và “no-R” nghĩa là gì?

- `P`: chỉ prompt, dùng cho guard trước khi model trả lời.
- `PR`: prompt + response, dùng cho guard sau khi model sinh output và cần ngữ cảnh prompt.
- `R`: chỉ response.
- `no-R`: loại riêng view `R`, vẫn giữ cả `P` và `PR`.

Việc bỏ R đồng thời thay đổi hai yếu tố:

1. loại một kiểu biểu diễn đầu vào;
2. giảm lượng dữ liệu huấn luyện.

Ở full EN+VI, train giảm từ `140.136` instance xuống `101.274`, tức giảm `38.862` instance hay khoảng `27,73%`. Vì vậy, thí nghiệm hiện tại đo hiệu ứng tổng hợp của “không có R + ít lượt supervision hơn”, chưa phải causal ablation chỉ riêng nội dung R.

## 2. Điều kiện so sánh và giới hạn diễn giải

### Nemotron binary

- Bản có R: `10.682` mẫu full test (`5.341 EN + 5.341 VI`).
- Bản no-R: `8.056` mẫu full test (`4.028 EN + 4.028 VI`).

Vì tập binary Nemotron thay đổi khi loại R, chênh lệch accuracy R/no-R ở đây chỉ mang tính **mô tả**, không phải so sánh ghép cặp tuyệt đối.

### SEA paired

- Cả hai giai đoạn dùng `3.680` mẫu (`1.840 EN + 1.840 VI`) trong full suite.
- SEA không chứa view R trong phép đánh giá này, nên đây là đối chứng R/no-R đáng tin hơn: cùng benchmark ngoài miền, chỉ training recipe/data view thay đổi.

### N23

- Cả hai giai đoạn báo cáo trên `5.768` mẫu được supervision (`2.884 EN + 2.884 VI`).
- Tổng gold-positive đều là `6.966`, support từng nhãn trùng nhau.

Vì vậy, N23 R/no-R có mức so sánh rất cao, dù không có prediction cũ để chạy lại kiểm định ghép cặp.

## 3. R so với no-R: Safe/Unsafe

Đơn vị accuracy là phần trăm. `Δ` = no-R trừ có-R, tính bằng điểm phần trăm.

| Run | Nemotron có R | Nemotron no-R | Δ | SEA có R | SEA no-R | Δ |
|---|---:|---:|---:|---:|---:|---:|
| E1 GLiGuard | 68,70 | 69,07 | +0,37* | 65,00 | 64,88 | −0,12 |
| E2 mmBERT GLi-compatible | 77,18 | 76,89 | −0,29* | 72,98 | 72,25 | −0,73 |
| E3 English-only | 75,58 | 73,80 | −1,78* | 71,82 | 71,11 | −0,71 |
| E4 matched EN+VI | 76,13 | 74,76 | −1,37* | 72,58 | 71,47 | −1,11 |
| E5 full EN+VI | **78,58** | **78,26** | −0,32* | **74,13** | **73,34** | −0,79 |
| E7 dynamic schema | 56,13 | 56,62 | +0,49* | 54,54 | 48,34 | **−6,20** |

\* Nemotron test đã thay thành phần/kích thước; không diễn giải như chênh lệch causal.

### Theo ngôn ngữ cho E3/E4/E5

| Run | Benchmark | EN có R | EN no-R | Δ EN | VI có R | VI no-R | Δ VI |
|---|---|---:|---:|---:|---:|---:|---:|
| E3 | Nemotron | 77,98 | 76,66 | −1,32* | 73,17 | 70,93 | −2,24* |
| E3 | SEA | 73,80 | 73,32 | −0,48 | 69,84 | 68,91 | −0,93 |
| E4 | Nemotron | 77,46 | 75,55 | −1,91* | 74,80 | 73,98 | −0,82* |
| E4 | SEA | 74,08 | 72,61 | −1,47 | 71,09 | 70,33 | −0,76 |
| E5 | Nemotron | 79,97 | 79,74 | −0,23* | 77,20 | 76,79 | −0,41* |
| E5 | SEA | 75,22 | 74,51 | −0,71 | 73,04 | 72,17 | −0,87 |

E5 chịu ảnh hưởng ít nhất khi bỏ R. Điều này hợp lý với giả thuyết full bilingual P/PR đã cho đủ đa dạng để binary head ít phụ thuộc hơn vào supervision R. Tuy nhiên, SEA vẫn giảm ở cả EN lẫn VI, nên không có bằng chứng R gây nhiễu.

## 4. Xếp hạng các encoder no-R trên cùng full suite

| Run | Benchmark | Accuracy | Macro-F1 | EN | VI | VI−EN | Unsafe recall | AUPRC | EN–VI agreement |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | Nemotron | 73,80 | 73,16 | 76,66 | 70,93 | −5,73 | **83,06** | 83,11 | 79,92 |
| E4 | Nemotron | 74,76 | 74,67 | 75,55 | 73,98 | −1,56 | 75,40 | 83,15 | 86,72 |
| E5 | Nemotron | **78,26** | **78,16** | **79,74** | **76,79** | −2,95 | 79,43 | **86,90** | 86,92 |
| E6 | Nemotron | 78,19 | 78,07 | 79,62 | 76,76 | −2,86 | 79,57 | 86,80 | **87,02** |
| E7 | Nemotron | 56,62 | 56,04 | 56,63 | 56,60 | **−0,03** | 42,09 | 64,41 | 79,47 |
| E3 | SEA | 71,11 | 71,04 | 73,32 | 68,91 | −4,40 | 70,53 | 75,58 | 82,12 |
| E4 | SEA | 71,47 | 71,24 | 72,61 | 70,33 | −2,28 | 66,71 | 75,54 | 85,76 |
| E5 | SEA | 73,34 | 73,00 | **74,51** | 72,17 | −2,34 | 66,30 | 78,33 | 85,71 |
| E6 | SEA | **73,40** | **73,11** | 74,46 | **72,34** | **−2,12** | 67,29 | **78,82** | **86,03** |
| E7 | SEA | 48,34 | 44,36 | 49,02 | 47,66 | −1,36 | **80,16** | 48,13 | 74,18 |

### Ý nghĩa của chuỗi E3 → E4 → E5

- **E3 → E4:** thêm tiếng Việt theo matched budget làm Nemotron VI tăng `+3,05 pp`, thu hẹp gap ngôn ngữ khoảng `4,17 pp`; trên SEA, VI tăng `+1,41 pp`. Đây vẫn là bằng chứng rõ rằng dữ liệu Việt giúp encoder cân bằng song ngữ.
- **E4 → E5:** tăng từ matched subset lên full EN+VI làm Nemotron tăng `+3,50 pp` và SEA tăng `+1,88 pp`. Lợi ích đến ở cả EN và VI, cho thấy quy mô dữ liệu quan trọng hơn việc chỉ cân bằng số lượng.
- **E3 có unsafe recall cao nhất** nhưng accuracy/Macro-F1 và gap ngôn ngữ kém hơn. Nó thiên về cảnh báo Unsafe nhiều hơn, không nhất thiết hiểu safety tốt hơn. Threshold cần được tune theo chi phí false negative.

## 5. N23: có R tốt hơn no-R

Threshold của từng nhãn là `0,5`.

| Run | Micro-F1 có R | Micro-F1 no-R | Δ | Macro-F1 có R | Macro-F1 no-R | Δ | Exact có R | Exact no-R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | 0,2373 | 0,1816 | −0,0557 | 0,0826 | 0,0462 | −0,0364 | 42,75% | 41,92% |
| E4 | 0,2366 | 0,1768 | −0,0598 | 0,0702 | 0,0327 | −0,0375 | 42,22% | 41,19% |
| E5 | **0,3931** | **0,3442** | −0,0489 | **0,2042** | **0,1576** | −0,0466 | **46,41%** | **44,89%** |
| E7 | 0,0000 | 0,0000 | 0 | 0,0000 | 0,0000 | 0 | — | 37,86% |

No-R E5 chỉ dự đoán `2.186` positive trong khi có `6.966` gold-positive. Micro precision vẫn cao `0,7205`, nhưng recall chỉ `0,2261`; mô hình vẫn quá bảo thủ ở threshold 0,5.

Các nhãn no-R E5 mạnh nhất:

| Nhãn | Support | Precision | Recall | F1 | AUPRC |
|---|---:|---:|---:|---:|---:|
| Criminal Planning/Confessions | 1.420 | 0,723 | 0,583 | 0,646 | 0,740 |
| Controlled/Regulated Substances | 514 | 0,821 | 0,401 | 0,539 | 0,667 |
| Guns and Illegal Weapons | 240 | 0,769 | 0,375 | 0,504 | 0,601 |
| PII/Privacy | 376 | 0,792 | 0,314 | 0,450 | 0,592 |
| Sexual | 224 | 0,764 | 0,246 | 0,372 | 0,475 |

Các nhãn `Political/Misinformation/Conspiracy`, `Fraud/Deception`, `Threat`, `Other`, `Malware`, `Copyright/Trademark/Plagiarism` và `High Risk Gov Decision Making` không có positive prediction nào ở threshold 0,5. Nhiều nhãn vẫn có AUPRC lớn hơn baseline tần suất, nên đây vừa là vấn đề dữ liệu mất cân bằng vừa là vấn đề calibration/threshold, không nên kết luận head hoàn toàn không học.

## 6. E6 so với E5 no-R: ablation sạch của N23 auxiliary loss

Hai run có:

- cùng `101.274` train instance và `6.390` validation instance;
- cùng 2 epoch, seed `3407`, effective batch `32`;
- cùng LoRA rank `4`, alpha `8`;
- cùng encoder/head learning rate `1e-5/1e-4`;
- cùng max length `8.192`;
- cùng 8.056 Nemotron test ID và 3.680 SEA ID.

Khác biệt chính:

- E5: binary + N23 category supervision;
- E6: binary-only, không có N23 head/loss.

### Điểm tổng

| Benchmark | E5 accuracy | E6 accuracy | E6−E5 | E5 Macro-F1 | E6 Macro-F1 | E5 AUPRC | E6 AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nemotron | 78,265% | 78,190% | −0,075 pp | 78,156% | 78,075% | 86,895% | 86,797% |
| SEA | 73,342% | 73,397% | +0,054 pp | 73,002% | 73,108% | 78,326% | 78,817% |

### Kiểm định theo đúng từng mẫu

| Benchmark | Chỉ E5 đúng | Chỉ E6 đúng | Discordant | McNemar exact p |
|---|---:|---:|---:|---:|
| Nemotron | 127 | 121 | 248 | 0,751 |
| SEA | 50 | 52 | 102 | 0,921 |

Không thể bác bỏ giả thuyết hai model có cùng error rate. Nói ngắn gọn: **E5 và E6 hòa về Safe/Unsafe**.

### Số đúng/sai theo ngôn ngữ

#### Nemotron

| Run | Lang | Gold Safe | Safe đúng | Safe→Unsafe | Gold Unsafe | Unsafe đúng | Unsafe→Safe | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E5 | EN | 1.865 | 1.456 | 409 | 2.163 | 1.756 | 407 | 79,74% |
| E6 | EN | 1.865 | 1.454 | 411 | 2.163 | 1.753 | 410 | 79,62% |
| E5 | VI | 1.865 | 1.413 | 452 | 2.163 | 1.680 | 483 | 76,79% |
| E6 | VI | 1.865 | 1.403 | 462 | 2.163 | 1.689 | 474 | 76,76% |

#### SEA

| Run | Lang | Gold Safe | Safe đúng | Safe→Unsafe | Gold Unsafe | Unsafe đúng | Unsafe→Safe | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E5 | EN | 978 | 767 | 211 | 862 | 604 | 258 | 74,51% |
| E6 | EN | 978 | 761 | 217 | 862 | 609 | 253 | 74,46% |
| E5 | VI | 978 | 789 | 189 | 862 | 539 | 323 | 72,17% |
| E6 | VI | 978 | 780 | 198 | 862 | 551 | 311 | 72,34% |

E6 trên SEA đổi nhẹ từ safe recall sang unsafe recall, đặc biệt ở VI: bắt thêm 12 Unsafe nhưng báo nhầm thêm 9 Safe. Đây là thay đổi trade-off rất nhỏ, không phải bước nhảy chất lượng.

### Chi phí train

| Run | Trainable params | Steps | Thời gian | Peak VRAM |
|---|---:|---:|---:|---:|
| E5 no-R | 584.089 | 6.330 | 60,93 phút | 9.821 MiB |
| E6 no-R | 575.234 | 6.330 | 59,68 phút | 9.821 MiB |

E6 nhanh hơn khoảng `1,25 phút` hay `2,05%`; số trainable parameter giảm khoảng `1,52%`. Không đủ để coi đây là lợi thế compute lớn.

### Quyết định E5 hay E6

- Chỉ cần Safe/Unsafe, muốn checkpoint đơn giản nhất: **E6 hợp lý**.
- Cần binary và còn muốn khai thác/tune N23: **E5 tốt hơn**, vì binary không bị tổn hại đáng kể khi học N23.
- Không nên nói “N23 làm nhiễu binary”; dữ liệu hiện tại không ủng hộ nhận định đó.
- E6 là **fixed-head binary-only**, không phải dynamic-schema binary-only. Vì vậy E6 chưa trả lời câu hỏi “N23 có làm E7 dynamic schema sập hay không”.

## 7. P so với PR trong no-R

| Run | Benchmark | View | N | Accuracy | Macro-F1 | Safe recall | Unsafe recall | AUPRC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E5 | Nemotron | P | 5.430 | 78,21 | 77,83 | 73,83 | 81,67 | 87,85 |
| E5 | Nemotron | PR | 2.626 | 78,37 | 78,30 | 82,44 | 74,15 | 85,06 |
| E6 | Nemotron | P | 5.430 | 78,29 | 77,86 | 73,12 | 82,36 | 87,74 |
| E6 | Nemotron | PR | 2.626 | 77,99 | 77,90 | 82,81 | 72,98 | 84,92 |
| E5 | SEA | P | 2.260 | 73,85 | 73,72 | 70,39 | 76,98 | 82,08 |
| E5 | SEA | PR | 1.420 | 72,54 | 67,26 | 90,70 | **42,75** | 69,07 |
| E6 | SEA | P | 2.260 | 74,07 | 73,89 | 69,18 | 78,50 | 82,92 |
| E6 | SEA | PR | 1.420 | 72,32 | 67,03 | 90,48 | **42,57** | 68,61 |

SEA PR là điểm yếu thật sự. Accuracy che mất việc model nghiêng mạnh về Safe: safe recall hơn 90% nhưng unsafe recall chỉ khoảng 43%. Với guard sau output, cần:

1. tune threshold riêng cho `PR`;
2. báo cáo PR unsafe recall/FNR như chỉ số chính;
3. nếu có thể, calibration theo benchmark validation hoặc tập Việt độc lập;
4. không dùng một threshold chung chỉ vì overall accuracy nhìn ổn.

## 8. E7 no-R có cải tiến không?

### So sánh trực tiếp

| Chỉ số | E7 có R | E7 no-R | Nhận xét |
|---|---:|---:|---|
| Nemotron accuracy | 56,13% | 56,62% | +0,49 pp, nhưng test set đổi |
| SEA accuracy | 54,54% | 48,34% | **−6,20 pp trên cùng benchmark** |
| N23 Micro-F1 @0,5 | 0 | 0 | Không cải thiện |
| Nemotron order disagreement | 22,00% | 16,31% | Tốt hơn 5,69 pp |
| SEA order disagreement | 33,53% | 19,92% | Tốt hơn 13,61 pp |

### Chẩn đoán no-R E7

| Benchmark/order | Accuracy | Macro-F1 | AUPRC | ECE | EN | VI |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron canonical | 56,62 | 56,04 | 64,41 | 8,16 | 56,63 | 56,60 |
| Nemotron reversed | 57,91 | 57,83 | 65,83 | 4,80 | 57,92 | 57,89 |
| SEA canonical | 48,34 | 44,36 | 48,13 | 16,51 | 49,02 | 47,66 |
| SEA reversed | 49,08 | 42,31 | 51,48 | 17,09 | 49,73 | 48,42 |

Trên Nemotron canonical, E7 no-R có:

- safe recall `73,46%`;
- unsafe recall `42,09%`.

Trên SEA, hướng lệch đảo ngược:

- safe recall chỉ `20,30%`;
- unsafe recall `80,16%`.

Đây là dấu hiệu decision boundary không ổn định khi đổi miền dữ liệu. SEA AUPRC `0,4813` gần mức không hữu ích; không thể sửa chỉ bằng threshold chung.

### Kết luận cho E7

E7 no-R **học được bất biến thứ tự schema tốt hơn**, nhưng **không học được ánh xạ semantics→safety đủ tốt**. Bỏ R có thể làm giảm một shortcut vị trí/kiểu view, song đồng thời giảm 27,7% supervision và làm generalization SEA tệ hơn.

E7 vẫn:

- kém fixed-head hơn 21–25 pp;
- N23 collapse ở threshold 0,5;
- calibration kém;
- nhạy với miền Nemotron/SEA;
- chưa phải reproduction trung thành của recipe GLiGuard 20 epoch/full encoder.

Do đó, kết luận đúng là: **bỏ R không cứu E7**.

## 9. Các kết luận nghiên cứu có thể dùng để báo cáo

### Kết luận được dữ liệu ủng hộ

1. Dữ liệu song ngữ giúp mmBERT thu hẹp gap EN–VI và cải thiện Vietnamese performance so với English-only.
2. Full EN+VI cho kết quả tốt hơn matched EN+VI trên cả Nemotron và SEA.
3. Dữ liệu no-R vẫn học được và tổng quát sang SEA, nhưng không tốt hơn phiên bản có R.
4. Auxiliary N23 không gây suy giảm đáng kể cho binary fixed-head.
5. Dynamic-schema E7 với recipe rẻ hiện tại chưa cạnh tranh; vấn đề không chỉ là threshold.
6. Guard `P` và guard `PR` có error profile khác nhau; PR cần calibration riêng.

### Kết luận chưa được phép nói

1. Không thể nói R “gây nhiễu” hoặc bỏ R “tốt hơn”.
2. Không thể tách hiệu ứng nội dung R khỏi hiệu ứng giảm 27,7% số instance.
3. Không thể dùng E6 để kết luận N23 làm hỏng E7; E6 là fixed-head.
4. Không thể kết luận kiến trúc schema-conditioned không khả thi nói chung; mới chỉ có một recipe 2 epoch rank-4 LoRA dưới huấn luyện.
5. Không thể dùng accuracy tổng để khẳng định guard sau output an toàn, vì SEA PR unsafe recall chỉ khoảng 43%.

## 10. Thí nghiệm tiếp theo đáng tiền nhất

Theo thứ tự ưu tiên:

1. **R causal control:** train E5 no-R nhưng upsample P/PR hoặc tăng số step từ `6.330` lên khoảng `8.760`, sao cho tổng optimizer exposure bằng bản có R. Nếu vẫn kém, R có thông tin hữu ích; nếu phục hồi, phần lớn suy giảm đến từ ít dữ liệu/step.
2. **Tune threshold theo scope và ngôn ngữ:** ít nhất có threshold riêng P và PR, chọn bằng validation với ràng buộc unsafe recall. Đây có khả năng mang lại lợi ích deployment lớn hơn train lại E6.
3. **Tune threshold theo nhãn N23:** dùng validation, báo cáo raw threshold 0,5 song song với tuned threshold. Không giấu collapse bằng threshold, nhưng tận dụng được ranking signal hiện có.
4. **E7a dynamic-schema binary-only:** giữ đúng kiến trúc E7 nhưng bỏ N23; đây mới là ablation đo N23 interference trong schema learner.
5. **E7b faithful recipe:** nhiều epoch hơn, full encoder/top-layer unfreeze hoặc LoRA lớn hơn, effective batch và learning rate gần paper, shuffle/dropout/task removal đúng recipe, audit permutation sau mỗi epoch.
6. **Benchmark Việt độc lập:** SEA chỉ là một benchmark và PR hiện có domain shift lớn; cần tập Việt do người Việt viết/kiểm định trước khi tuyên bố deployment.

## 11. Nguồn số liệu cục bộ

- No-R evaluation:
  `reports/analysis_20260724/no_r/reports/no_r_phase0/evaluation_matrix`
- No-R training metrics:
  `reports/analysis_20260724/no_r/reports/no_r_phase0/experiment_runs`
- E6 evaluation:
  `reports/analysis_20260724/e6/reports/e6_no_r_binary_ablation/evaluation_matrix/E6NR-M-EV-FULL-BIN-8K`
- E6 training metrics:
  `reports/analysis_20260724/e6/reports/e6_no_r_binary_ablation/experiment_runs/E6NR-M-EV-FULL-BIN-8K/metrics.json`
- Báo cáo có R theo ngôn ngữ:
  `reports/training_dashboard/language_comparison_2026-07-22.md`
- Phân tích E7 có R:
  `reports/phase0/E7_SCHEMA_ANALYSIS.md`
- Snapshot toàn dự án có R:
  `reports/research_archive/EXPERIMENT_STATUS_20260722.json`

Các archive gốc đã tải và xác minh:

- `reports/vast_download/phase0_no_r_20260724/PHASE0_NO_R_20260724.tar.zst`
  - SHA-256: `f0c0c68a51021c09fb229bda5ef03cad51db3fcf02356a8b0431dd8d54cdf53a`
- `reports/vast_download/e6_no_r_binary_20260724/E6_NO_R_BINARY_ABLATION_20260724.tar.zst`
  - SHA-256: `7c73f20f3dcf5db288ae7d653a7db06dbbed5d5d38ae2f94f5152008019bd0f1`
