# Kế hoạch thực thi đã khóa — Nemotron Safety EN–VI, GLiGuard và mmBERT

**Bản hợp đồng:** v4, ngày 21/07/2026  
**Nguồn máy đọc:** `configs/phase0_experiments.json`  
**Mục tiêu gần nhất:** profile trên GPU 32 GB, sau đó chạy B0 và E1–E7 theo thứ tự đã khóa.

## 1. Guard đang học cái gì

Mỗi instance chỉ là một chuỗi text cần phân loại. `P/R/PR` là metadata và quy tắc lấy
gold target, không phải ba task hay ba head:

| View | Chuỗi đưa vào guard | Gold binary |
|---|---|---|
| `P` | prompt | `prompt_label` |
| `R` | response | `response_label` |
| `PR` | prompt + response | `response_label` |

Không suy nhãn prompt từ response hoặc ngược lại.

Hai loại supervision:

1. **Binary safety** có đúng một nhãn `safe/unsafe`, áp dụng cho mọi instance, dùng
   softmax và categorical cross-entropy.
2. **N23 policy categories** có thể có không, một hoặc nhiều nhãn, dùng 23 sigmoid và
   binary cross-entropy độc lập. Loss N23 chỉ được tính khi `category_scope` là
   `prompt` hoặc `interaction`. Khi scope là `unavailable`, phải mask toàn bộ loss N23;
   tuyệt đối không biến hàng đó thành 23 nhãn âm.

“23 head” trong trao đổi được hiện thực là **một đầu ra đa nhãn gồm 23 logit**, không
phải 23 softmax head. E7 còn dùng một shared scalar MLP cho mọi `[L]`, nên kích thước
đầu ra thực tế phụ thuộc schema.

## 2. Các run chính

| ID | Mô hình | Train | Task | Dữ liệu |
|---|---|---:|---|---|
| `B0-GLI-ZS` | GLiGuard gốc | không | binary | baseline zero-shot |
| `E1-G-EV-512` | GLiGuard + LoRA | có | binary | các cặp EN–VI GLi-native |
| `E2-M-EV-GLI-COMPAT-8K` | mmBERT fixed-head + LoRA | có | binary | đúng cùng ID/text với E1; mmBERT được dùng đến 8K |
| `E3-M-E-8K` | mmBERT fixed-head + LoRA | có | binary + N23 masked | toàn bộ English |
| `E4-M-EV-MATCHED-8K` | mmBERT fixed-head + LoRA | có | binary + N23 masked | một ngôn ngữ/hash cho mỗi semantic pair |
| `E5-M-EV-FULL-8K` | mmBERT fixed-head + LoRA | có | binary + N23 masked | full English + full Vietnamese |
| `E7-M-SCHEMA-EV-8K` | mmBERT GLi-style + LoRA | có | binary + dynamic N23 | full English + full Vietnamese |

Không còn E6 trong ma trận chính. E1/E2 trả lời câu hỏi GLiGuard so với mmBERT trên
miền dữ liệu mà GLiGuard thật sự xử lý được. E3/E4/E5 đo ảnh hưởng của exposure tiếng
Việt với cùng backbone/recipe. E7 kiểm tra có thể đưa cơ chế schema động của GLiGuard
sang mmBERT 8K hay không.

E3 và E4 có đúng cùng ngân sách 70.068 mẫu train:

- E3: 70.068 English;
- E4: 70.068 semantic unit, chọn ổn định một ngôn ngữ theo hash `record_uid`, hiện có
  35.248 EN và 34.820 VI;
- E5/E7: đủ 140.136 instance EN+VI.

## 3. Phép so E1/E2 và cách tính 512

Nút thắt là GLiGuard. Eligibility E1/E2 được tính như sau:

```text
GLiGuard: toàn sequence sau schema binary <= 512
mmBERT:   toàn sequence sau schema <= 8192
cặp:      cả EN và VI của cùng (record_uid, view) đều đạt
truncate: không dùng trong tập train/primary native
```

Phép tính GLiGuard **đã gồm** schema/prefix `text safety classification`, hai label
`safe/unsafe`, các marker và separator; không phải chỉ token text.

Kết quả native đã khóa:

| Split | Cặp EN–VI | Instance | Ghi chú |
|---|---:|---:|---|
| train | 57.804 | 115.608 | bỏ đúng một cặp R bệnh lý ~52K token mmBERT |
| valid | 3.333 | 6.666 | cùng ID giữa canonical và GLi payload |
| test | 4.238 | 8.476 | cùng ID giữa canonical và GLi payload |

mmBERT không bị ép xuống 512. E2 được encode bằng tokenizer mmBERT và có quyền dùng
tối đa 8K, nhưng nhận đúng chuỗi và đúng ID của E1.

## 4. Hai tầng đánh giá công bằng cho E1/E2

### Primary: native-fit

Chỉ dùng complete pair mà cả hai ngôn ngữ vừa GLiGuard 512 và mmBERT 8K, không
truncate:

| Benchmark | Cặp EN–VI | Instance |
|---|---:|---:|
| Nemotron valid | 3.333 | 6.666 |
| Nemotron test | 4.238 | 8.476 |
| SEA paired bundled | 1.573 | 3.146 |

### Secondary: full shared-truncated

Giữ mọi complete pair. Mỗi chuỗi được cắt một lần theo giới hạn GLiGuard, rồi chính
chuỗi sau cắt đó được gửi nguyên vẹn cho cả GLiGuard và mmBERT:

| Benchmark | Cặp EN–VI | Instance |
|---|---:|---:|
| Nemotron valid | 4.382 | 8.764 |
| Nemotron test | 5.341 | 10.682 |
| SEA paired bundled | 1.840 | 3.680 |

Audit lưu hash text trước/sau, số ký tự, token GLiGuard trước/sau và token mmBERT
trước/sau. Với `PR`, bộ cắt ưu tiên prompt head và response head+tail vì gold target là
response safety. Các file nằm trong `data/eval_shared_gliguard_512`.

## 5. E3/E4/E5/E7 dùng toàn bộ sức mạnh 8K

Không loại 36 mẫu fixed-head hoặc các tail schema chỉ vì vượt 8K. Trainer giữ đủ mọi
instance và chỉ cắt các tail bằng đúng tokenizer:

- `P`: head+tail của prompt;
- `R`: head+tail của response;
- `PR`: prompt head, response head+tail;
- mọi hàng bị cắt có audit JSONL riêng;
- validation/test/SEA cũng áp dụng cùng chính sách và báo metric native/truncated riêng.

`generic` và `jailbreaking` đều được train. Jailbreak không bị loại chỉ vì tag; nếu dài
thì được xử lý bằng cùng policy 8K.

## 6. Kiến trúc E7 schema động

Một ví dụ có N23 supervision được serialize về mặt logic như sau:

```text
[P] text safety classification
    [L] safe [L] unsafe
[P] safety policy categories
    [L] Criminal Planning/Confessions ... [L] Threat ...
[SEP] text
```

Một ví dụ có `category_scope=unavailable` chỉ chứa task binary; task N23 được bỏ khỏi
schema. Encoder chạy một forward bidirectional. Final hidden state ở từng `[L]` đi qua
cùng MLP `d -> 2d -> 1` để sinh một scalar.

- Binary: gom đúng hai scalar, softmax + CE.
- N23: các scalar hiện có dùng sigmoid + BCE.
- Train luôn giữ mọi positive category.
- Thứ tự label được shuffle.
- Với xác suất cấu hình, chỉ lấy một subset negative label hoặc full 23 labels. Điều này
  dạy mô hình thêm/bớt/hoán vị schema mà không xóa true label.
- Không tuyên bố typo/label hoàn toàn chưa từng học như `SAEF/UNSEFA` sẽ được hiểu đúng;
  dynamic output size không tự động bảo đảm zero-shot semantics.

Ba token `[P]`, `[L]`, `[SEP]` được thêm vào tokenizer. Chỉ ba embedding row đó, LoRA
encoder và shared MLP là trainable.

## 7. Metric và artifact bắt buộc

Binary:

- unsafe AUPRC, macro-F1, unsafe precision/recall/F1;
- AUROC, confusion matrix, Brier, ECE;
- slice theo language, view, tag và native/truncated;
- paired EN–VI decision consistency và probability gap.

N23:

- micro/macro-F1 trên đúng các hàng có category supervision;
- per-label precision/recall/F1 và support;
- không đưa hàng `unavailable` vào mẫu số.

Mọi run phải lưu prediction-level JSONL, resolved config, hash model/tokenizer/manifest,
training curve, checkpoint, truncation audit và metric. Nemotron valid dùng để chọn
checkpoint/threshold. Nemotron test và SEA chỉ dùng đánh giá cuối, không tune.

## 8. Trạng thái code trước khi thuê GPU

Đã xong:

- full manifests và E1/E2 native manifests;
- E3/E4 full-budget manifests;
- shared-truncated Nemotron/SEA manifests;
- fixed-head scalable trainer: LoRA, binary+masked N23, gradient accumulation,
  length buckets, exact checkpoint/resume, tqdm;
- dynamic-schema scalable trainer: task add/remove, label permutation/subset,
  binary CE + N23 BCE, marker embeddings, checkpoint/resume, tqdm;
- GLiGuard full launcher với single-label CE override;
- local one-step smoke cho fixed multi-task và dynamic-schema multi-task;
- matrix evaluator tự chạy đúng native/shared/full suite của từng run và lưu index;
- preflight v2 pass, không có missing path hay ID mismatch.

Gate còn lại:

1. profile thật trên GPU 32 GB tại 512/1K/2K/4K/8K để chốt microbatch;
2. chạy một mini end-to-end E1/E2/E7 trên máy thuê, gồm save/reload/eval;
3. GLiNER2 1.3.2 đang khóa trong môi trường chỉ warm-restore adapter, không restore
   optimizer/scheduler/global step.
   E1 phải chạy trong `tmux` và ổ đĩa ổn định; đây là hạn chế còn mở, không được gọi là
   exact resume.

Vì vậy hiện tại **đã đủ để thuê GPU làm profile/smoke**, nhưng chưa được bấm chạy toàn bộ
ma trận trước khi profile 32 GB qua.

## 9. Thứ tự chạy trên máy thuê

1. kiểm tra `nvidia-smi`, dung lượng disk và hash dữ liệu/model;
2. chạy preflight;
3. chạy profiler 512 → 1K → 2K → 4K → 8K, dừng riêng bucket bị OOM;
4. B0 zero-shot;
5. mini E1, E2, E7 khoảng 20–100 optimizer step và reload/eval;
6. E1 và E2 full;
7. E3, E4, E5 với cùng optimizer/effective batch/epoch;
8. E7 full;
9. inference Nemotron valid/test và SEA, sau đó khóa bảng kết quả.

Không dùng điểm smoke một step để kết luận mô hình nào tốt hơn.
