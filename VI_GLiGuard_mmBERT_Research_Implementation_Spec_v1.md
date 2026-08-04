# ĐẶC TẢ NGHIÊN CỨU VÀ TRIỂN KHAI

## Vietnamese GLiGuard-inspired Guardrail với mmBERT và Nemotron Safety Guard Dataset v3

**Phiên bản:** 1.0  
**Ngày:** 20/07/2026  
**Mục đích:** Tài liệu nghiên cứu kiêm đặc tả kỹ thuật để Codex triển khai thí nghiệm một cách có kiểm soát, tái lập được và không làm nhiễm benchmark.

---

## 0. Tóm tắt điều hành

Dự án hiện có một corpus safety tiếng Anh của NVIDIA Nemotron Safety Guard Dataset v3 và bản dịch tiếng Việt đang gần hoàn tất. Mục tiêu trước mắt không phải xây ngay một GLiGuard tiếng Việt hoàn chỉnh, mà là trả lời một câu hỏi khoa học rõ ràng:

> Khi giữ nguyên backbone và quy trình huấn luyện, việc bổ sung dữ liệu tiếng Việt có cải thiện khả năng moderation tiếng Việt hay không, cải thiện bao nhiêu, và có làm giảm năng lực tiếng Anh hay không?

Để trả lời câu hỏi này, giai đoạn đầu dùng **mmBERT** làm encoder đa ngôn ngữ với các đầu phân loại cố định. Sau khi có baseline đáng tin cậy, dự án mới tiến tới kiến trúc **schema-conditioned** lấy cảm hứng từ GLiGuard.

Lý do không fine-tune thẳng checkpoint GLiGuard hiện có:

1. Checkpoint GLiGuard được phát hành chủ yếu cho tiếng Anh.
2. Nó kế thừa `fastino/gliner2-base-v1`, mà config chính thức dùng `microsoft/deberta-v3-base`, context 512 token.
3. Dữ liệu của dự án có tiếng Việt và nhiều response/jailbreak dài.
4. Dùng thẳng GLiGuard sẽ làm lẫn nhiều biến cùng lúc: backbone, tokenizer, schema, taxonomy, dữ liệu và ngôn ngữ.
5. mmBERT là encoder đa ngôn ngữ hiện đại, 307M tham số, context 8.192 token, phù hợp để đo riêng tác động của dữ liệu tiếng Việt.

Kết quả giai đoạn đầu phải gồm:

- Model English-only.
- Model Vietnamese-only.
- Model bilingual English + Vietnamese.
- Đánh giá trên Nemotron EN/VI paired test.
- Đánh giá độc lập trên MultiJail EN/VI, RTP-LX VI, XSTest EN/VI và PolyGuardPrompts EN/VI.
- Báo cáo prompt safety, response safety, jailbreak, category, độ nhất quán EN-VI, false-positive và throughput.

---

# PHẦN I — BỐI CẢNH VÀ PHẠM VI DỰ ÁN

## 1. Dự án hiện tại là gì?

Dự án nhằm xây một guard model nhỏ, nhanh và có thể chạy nội bộ để kiểm tra:

- Prompt của người dùng có safe hay unsafe.
- Response của LLM có safe hay unsafe.
- Nội dung thuộc nhóm rủi ro nào.
- Prompt có dấu hiệu jailbreak hay không.
- Sau này có thể mở rộng sang refusal, PII và taxonomy đặc thù Việt Nam.

Guard model này là một thành phần trong hệ thống guardrail, không phải toàn bộ hệ thống.

### 1.1. Phân biệt guard model và guardrail orchestration

**Guard model** là model phân loại nội dung.

Ví dụ:

```text
Prompt -> safe / unsafe
Prompt + Response -> response safe / unsafe
```

**Guardrail orchestration** là hệ thống quyết định phải làm gì với kết quả:

```text
Guard model
    -> cho phép
    -> chặn
    -> yêu cầu model trả lời lại
    -> chuyển kiểm duyệt con người
    -> ghi log
    -> áp policy theo sản phẩm
```

NeMo Guardrails thuộc lớp orchestration. Nemotron Safety Guard, GLiGuard và model mmBERT của dự án thuộc lớp guard model.

---

## 2. Trạng thái dữ liệu hiện tại

### 2.1. Nemotron Safety Guard Dataset v3 — phần tiếng Anh

Các split chính thức đã tải:

| Split | Số dòng |
|---|---:|
| Train | 40.007 |
| Validation | 2.445 |
| Test | 2.964 |
| **Tổng** | **45.416** |

Đường dẫn nguồn đã biết:

```text
data/raw/nemotron_safety_guard_v3/en/train.jsonl
data/raw/nemotron_safety_guard_v3/en/valid.jsonl
data/raw/nemotron_safety_guard_v3/en/test.jsonl
```

### 2.2. Các trường dữ liệu

```text
id
language
prompt
prompt_label
prompt_label_source
reconstruction_id_if_redacted
response
response_label
response_label_source
tag
violated_categories
```

### 2.3. Các đặc điểm đã phát hiện

- `language = "en"` cho toàn bộ phần tiếng Anh.
- `id` của NVIDIA không duy nhất theo từng dòng.
- Train có 40.007 dòng nhưng chỉ 35.007 source ID khác nhau.
- Pipeline đã tạo hoặc phải tạo `record_uid` duy nhất cho từng dòng.
- Có prompt `REDACTED`; không được coi chữ `REDACTED` là nội dung thật.
- `response = null` và `response_label = ""` không phải lớp thứ ba.
- `tag` gồm `generic` và `jailbreaking`.
- `violated_categories` là multi-label, không phải một class chuỗi duy nhất.
- Bản dịch tiếng Việt hiện được báo cáo đã hoàn thành khoảng 43.000/45.416 dòng. Codex phải đếm lại chính xác trạng thái thành công, thất bại và pending thay vì tin vào số gần đúng.

### 2.4. Quy tắc split

Phải giữ nguyên train/valid/test của NVIDIA.

Không được chia lại ngẫu nhiên vì:

- Mất khả năng so sánh với benchmark gốc.
- Có nguy cơ biến thể cùng nguồn lọt sang cả train và test.
- Phá cặp EN-VI theo `record_uid`.

---

# PHẦN II — GLiGUARD LÀ GÌ?

## 3. Làm rõ tên gọi

Tài liệu này tập trung vào paper:

> **GLiGuard: Schema-Conditioned Classification for LLM Safeguard**  
> Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis, 2026.

Không nhầm với paper khác có tên gần giống:

> **GLiNER Guard: Unified Encoder Family for Production LLM Safety and Privacy**

Hai hướng có liên quan về encoder safety nhưng không phải cùng một paper.

---

## 4. Ý tưởng cốt lõi của GLiGuard

Các guard model kiểu LlamaGuard/Nemotron thường là decoder autoregressive:

```text
Prompt + policy
    -> model sinh từng token
    -> JSON / safe / unsafe / category
```

GLiGuard coi moderation là bài toán phân loại:

```text
Schema task + label + text
    -> một lần encoder forward
    -> nhiều kết quả phân loại song song
```

Checkpoint công khai khoảng 0,3B tham số, nhỏ hơn nhiều model guard decoder 7B-27B.

GLiGuard hỗ trợ các nhóm nhiệm vụ:

- Prompt safety.
- Response safety.
- Response refusal.
- Harm category multi-label.
- Jailbreak strategy multi-label.

---

## 5. Kiến trúc GLiGuard chi tiết

### 5.1. Schema được đưa vào input

Mỗi task gồm:

- Tên task.
- Danh sách nhãn.
- Loại single-label hoặc multi-label.

Dạng tuyến tính hóa khái niệm:

```text
[P] prompt safety classification
[L] safe
[L] unsafe

[P] harm category classification
[L] violence
[L] non-violent crime
[L] sexual content
...

[SEP]

Nội dung cần phân loại
```

Trong đó:

- `[P]` bắt đầu một task.
- `[L]` đánh dấu một candidate label.
- `[SEP]` tách schema và text.

### 5.2. Encoder hai chiều

Toàn bộ schema và text đi qua một bidirectional encoder:

```text
H = Encoder(schema + text)
```

Vì attention hai chiều:

- Label token nhìn thấy toàn bộ text.
- Text token nhìn thấy label và task.
- Label representation thay đổi theo từng input.

### 5.3. Lấy embedding tại token `[L]`

Hidden state tại mỗi `[L]` được dùng làm representation cho label tương ứng.

```text
e_label = H[position_of_[L]]
```

Đây không phải embedding tĩnh. Nó đã được contextualize bởi:

- Task name.
- Các label khác.
- Toàn bộ nội dung cần kiểm duyệt.

### 5.4. Shared classification MLP

Mỗi label embedding đi qua cùng một MLP:

```text
Linear(d, 2d)
ReLU
Linear(2d, 1)
```

Kết quả là một logit cho từng label.

### 5.5. Hàm kích hoạt

**Single-label**

```text
softmax
```

Ví dụ:

```text
safe / unsafe
refusal / compliance
```

**Multi-label**

```text
sigmoid từng nhãn
```

Ví dụ một mẫu có thể đồng thời là:

```text
violence
threat
weapons
```

### 5.6. Loss

- Cross entropy cho single-label.
- Binary cross entropy cho multi-label.
- Tổng loss là tổng đóng góp các task có nhãn.
- Paper còn dùng entropy regularization để giảm dự đoán quá tự tin.

### 5.7. Decision rule

GLiGuard không chỉ dùng binary safety head.

Ví dụ prompt có thể bị override thành unsafe nếu:

- Harm category khác benign.
- Jailbreak strategy khác benign.

Response có thể được override về safe nếu model phát hiện refusal.

Đây là rule của GLiGuard gốc. Dự án Việt Nam không được sao chép rule một cách máy móc nếu dữ liệu chưa có nhãn tương ứng.

---

## 6. GLiGuard được train bằng gì?

Theo model card công khai:

- Core safety và refusal từ WildGuardTrain.
- Harm-category và jailbreak-strategy labels được bổ sung bằng automatic annotation.
- Có thêm dữ liệu synthetic cho category và jailbreak strategy.
- Checkpoint được phát hành như unified moderation classifier, không phải generative model.

### 6.1. Các task/nhãn công khai của checkpoint

Prompt side:

```text
prompt_safety
prompt_toxicity
jailbreak_detection
```

Response side:

```text
response_safety
response_toxicity
response_refusal
```

Model card liệt kê 14 harm category cộng `benign`, và 11 jailbreak strategy cộng `benign`.

### 6.2. Backbone thực tế của checkpoint công khai

Model tree của GLiGuard:

```text
fastino/gliguard-LLMGuardrails-300M
    <- fastino/gliner2-base-v1
```

Config chính thức của `gliner2-base-v1`:

```text
model_name = microsoft/deberta-v3-base
max_position_embeddings = 512
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
```

Điều này rất quan trọng: GLiGuard paper mô tả kiến trúc tổng quát có thể dùng DeBERTa hoặc ModernBERT, nhưng checkpoint công khai cụ thể bắt nguồn từ GLiNER2 base dùng DeBERTa-v3-base.

---

# PHẦN III — mmBERT VÀ LÝ DO CHỌN

## 7. mmBERT là gì?

Model đề xuất:

```text
jhu-clsp/mmBERT-base
```

Đây là encoder đa ngôn ngữ xây trên ModernBERT.

Thông số chính:

| Thuộc tính | mmBERT-base |
|---|---:|
| Tổng tham số | 307M |
| Non-embedding parameters | 110M |
| Layers | 22 |
| Hidden size | 768 |
| Attention heads | 12 |
| Intermediate size | 1152 |
| Vocabulary | 256.000 |
| Context tối đa | 8.192 token |
| Số ngôn ngữ pretrain | hơn 1.800 |
| Dữ liệu pretrain | hơn 3 nghìn tỷ token |
| License | MIT |

Kiến trúc có:

- Bidirectional attention.
- RoPE.
- Global/local attention.
- Flash Attention 2.
- Unpadding.
- Tokenizer Gemma 2 đa ngôn ngữ.

---

## 8. Vì sao chọn mmBERT trước GLiGuard/GLiNER2?

### 8.1. Cần đo tác động dữ liệu tiếng Việt

Câu hỏi đầu tiên là:

```text
Cùng backbone
+ cùng task head
+ cùng số step
thì thêm dữ liệu tiếng Việt tác động ra sao?
```

Fixed-head classifier là thiết kế dễ kiểm soát nhất.

Nếu dùng schema GLiGuard ngay, kết quả bị ảnh hưởng đồng thời bởi:

- Schema wording.
- Label description.
- Token đặc biệt.
- Decision rules.
- Threshold multi-label.
- Task composition.
- Backbone.
- Dữ liệu.
- Ngôn ngữ.

### 8.2. Checkpoint GLiGuard công khai thiên tiếng Anh

Hugging Face card của checkpoint ghi language là English. Nó chưa phải Vietnamese safety checkpoint đã được xác nhận.

### 8.3. Backbone GLiGuard công khai dùng DeBERTa-v3-base

DeBERTa-v3-base checkpoint đó:

- Chủ yếu tiếng Anh.
- Context 512.
- Tokenizer không tối ưu cho tiếng Việt bằng một model massively multilingual.

### 8.4. Dữ liệu dự án có sequence dài

Local profiling cho thấy:

- Prompt train dài tối đa khoảng 18.504 ký tự.
- Response train dài tối đa khoảng 65.493 ký tự.
- P95 response hơn 2.000 ký tự.
- Nhiều jailbreak dài.

Không có nghĩa phải train 8.192 token ngay, nhưng context 8K tạo không gian cho ablation 512/1024/2048/4096.

### 8.5. mmBERT được pretrain cho multilingual transfer

mmBERT được train trên hơn 1.800 ngôn ngữ. Đây là lựa chọn hợp lý để:

- English-only fine-tune rồi test zero-shot tiếng Việt.
- Vietnamese-only fine-tune.
- Bilingual fine-tune.
- So sánh cross-lingual transfer.

### 8.6. Đây không phải tuyên bố mmBERT tốt hơn GLiGuard

Mục tiêu giai đoạn 1:

```text
mmBERT = backbone thử nghiệm ngôn ngữ
GLiGuard = kiến trúc đích tham khảo
```

Sau khi baseline ổn, có thể:

1. Gắn schema-conditioned head kiểu GLiGuard lên mmBERT.
2. Fine-tune `fastino/gliner2-multi-v1`.
3. So sánh fixed-head với schema-conditioned.
4. Đánh giá checkpoint GLiGuard gốc zero-shot EN và VI.

---

## 9. Vì sao chưa chọn `gliner2-multi-v1` làm model chính?

`fastino/gliner2-multi-v1` là lựa chọn đáng thử trong ablation sau này, nhưng chưa phải baseline đầu tiên vì:

- Nó là model schema-based information extraction/classification, không phải safety checkpoint.
- Khả năng tiếng Việt chưa có benchmark safety công khai.
- Fine-tuning bằng GLiNER2 library thêm một tầng biến số.
- Context và batching cần kiểm tra thực nghiệm.
- Dữ liệu hiện tại không có đủ nhãn refusal và 11 jailbreak strategy như GLiGuard.

Nó phải xuất hiện trong Phase 2 hoặc Phase 3, không bị bỏ qua hoàn toàn.

---

# PHẦN IV — KIẾN TRÚC MODEL GIAI ĐOẠN 1

## 10. Model v0: MMBERTFixedGuard

### 10.1. Nguyên tắc

Một encoder dùng chung:

```text
mmBERT encoder
```

Các đầu phân loại cố định:

```text
prompt safety head
response safety head
jailbreak binary head
conversation category head (optional, sau audit)
```

### 10.2. Input format

Prompt task:

```text
Prompt: {prompt}
```

Response task:

```text
Prompt: {prompt}
Response: {response}
```

Không dùng response khi `response = null`.

### 10.3. Pooling

Khuyến nghị mặc định:

```text
masked mean pooling
```

Không mean trên padding token.

Phải cho phép ablation:

```text
mean pooling
CLS pooling
```

### 10.4. Prompt safety head

```text
Linear(hidden_size, 2)
```

Nhãn:

```text
safe
unsafe
```

### 10.5. Response safety head

```text
Linear(hidden_size, 2)
```

Chỉ tính loss khi:

```text
response != null
and normalized_response_label in {"safe", "unsafe"}
```

Chuỗi rỗng không phải class.

### 10.6. Jailbreak head

```text
Linear(hidden_size, 2)
```

Nhãn:

```text
generic -> 0
jailbreaking -> 1
```

Đây là jailbreak binary, không phải 11 jailbreak strategy của GLiGuard.

Không được đặt tên output là `jailbreak_strategy`.

### 10.7. Category head

Chưa bật ở milestone đầu.

Lý do:

- Dataset chỉ có một `violated_categories`.
- Không có `prompt_categories` và `response_categories` tách biệt.
- Có thể là nhãn ở mức conversation.
- `Needs Caution` không được dùng để tự suy ra binary unsafe.
- `Other` cần giữ như category legacy/observed.
- `Safe` thường là vector rỗng.

Sau audit, có thể triển khai:

```text
conversation_categories
```

trên input prompt + response.

Không được huấn luyện riêng `response_toxicity` bằng cùng field nếu chưa chứng minh field đó thuộc response.

### 10.8. Refusal head

Không triển khai từ Nemotron v3 ở milestone đầu vì dataset không có refusal label chính thức.

Refusal head chỉ được thêm khi có dữ liệu như:

- WildGuardTrain.
- PolyGuardMix.
- Nguồn khác có refusal labels.

Không dùng PolyGuardPrompts benchmark để train.

---

## 11. Multi-task loss

Dạng tổng quát:

```text
L = w_prompt * L_prompt
  + w_response * mask_response * L_response
  + w_jailbreak * L_jailbreak
  + w_category * mask_category * L_category
```

Mặc định milestone đầu:

```text
w_prompt = 1.0
w_response = 1.0
w_jailbreak = 0.5
w_category = 0.0
```

Codex phải đưa các weight vào config.

Không hard-code.

---

## 12. Threshold

### 12.1. Binary classification

Phải báo hai kết quả:

1. Threshold mặc định 0,5.
2. Threshold chọn trên validation.

Threshold tối ưu không được chọn trên test.

### 12.2. Mục tiêu threshold

Có ít nhất ba chế độ:

```text
best_macro_f1
max_recall_subject_to_fpr
min_expected_cost
```

Ví dụ:

```text
unsafe recall >= 0.95
và chọn FPR thấp nhất
```

---

# PHẦN V — DỮ LIỆU VÀ KIỂM SOÁT RÒ RỈ

## 13. Data contract nội bộ

Mỗi record paired cần tối thiểu:

```json
{
  "record_uid": "unique-internal-id",
  "source_id": "nvidia-id",
  "split": "train",
  "prompt_en": "...",
  "prompt_vi": "...",
  "response_en": "...",
  "response_vi": "...",
  "prompt_label": "unsafe",
  "response_label": "safe",
  "tag": "jailbreaking",
  "violated_categories_raw": "...",
  "translation_status": "success"
}
```

### 13.1. Không ghi đè dữ liệu gốc

Giữ nguyên:

```text
data/raw/
```

Tất cả file chuẩn hóa vào:

```text
data/processed/
```

### 13.2. Khóa duy nhất

Dùng:

```text
record_uid
```

Không dùng NVIDIA `id` làm khóa duy nhất.

### 13.3. REDACTED

Mặc định:

```text
exclude_from_text_training = true
```

Chỉ đưa lại khi:

- Khôi phục đúng nguồn.
- Có hash và provenance.
- Không làm thay đổi split.

### 13.4. Dòng dịch lỗi

Không âm thầm fallback về English trong tập Vietnamese.

Phải phân trạng thái:

```text
success
failed
missing
blocked
quality_review
```

Vietnamese experiment chỉ lấy `success`, trừ khi có rule khác được ghi rõ.

### 13.5. Pair preservation

Model E, V và EV phải dùng cùng tập `record_uid` hợp lệ khi so sánh chính.

Ví dụ nếu 300 mẫu dịch lỗi:

```text
E-paired
V-paired
EV-paired
```

đều dùng tập giao nhau.

Ngoài ra có thể báo E-full riêng.

---

## 14. Chuẩn hóa label

### 14.1. Prompt label

```text
safe
unsafe
```

### 14.2. Response label

```text
safe
unsafe
missing
```

`missing` chỉ là mask, không phải class.

### 14.3. Category

Giữ hai trường:

```text
violated_categories_raw
violated_categories_list
```

Tách bằng comma, trim whitespace, loại duplicate trong cùng record.

Không sửa raw.

Danh sách observed từ local audit gồm:

```text
Criminal Planning/Confessions
Needs Caution
Hate/Identity Hate
Violence
Harassment
Controlled/Regulated Substances
PII/Privacy
Profanity
Immoral/Unethical
Sexual
Illegal Activity
Guns and Illegal Weapons
Suicide and Self Harm
Unauthorized Advice
Manipulation
Sexual (minor)
Political/Misinformation/Conspiracy
Fraud/Deception
Threat
Other
Malware
Copyright/Trademark/Plagiarism
High Risk Gov Decision Making
```

Không suy binary safety từ danh sách này.

---

## 15. Audit duplicate và contamination

Codex phải tạo ba báo cáo:

### 15.1. Within-split duplicate

- Exact normalized prompt.
- Exact prompt+response.
- Duplicate source ID.
- Duplicate record hash.

### 15.2. Cross-split duplicate

So train với valid/test:

- Exact.
- Near duplicate n-gram/MinHash.
- Semantic duplicate bằng embedding.

Không tự động xóa official test. Chỉ đánh dấu.

### 15.3. Benchmark contamination

So external benchmark với Nemotron train:

```text
all_score
clean_only_score
contaminated_count
```

---

# PHẦN VI — THÍ NGHIỆM BẮT BUỘC

## 16. Thứ tự triển khai

### Milestone A — Data audit

Không train trước khi hoàn thành:

- Đếm exact EN/VI paired rows.
- Đếm translation success/failure.
- Kiểm tra label distribution.
- Kiểm tra length tokenized bằng mmBERT tokenizer.
- Kiểm tra duplicate.
- Xuất HTML/JSON report.

### Milestone B — Prompt-only binary baseline

Đây là baseline ưu tiên số 1.

Train:

```text
English-only
Vietnamese-only
Bilingual
```

Không response/category/jailbreak trong lần đầu.

Mục tiêu:

- Chứng minh train/eval pipeline đúng.
- So sánh cross-lingual.
- Kiểm tra threshold/FPR.

### Milestone C — Response safety

Thêm response head với masked loss.

### Milestone D — Jailbreak binary

Thêm head theo `tag`.

### Milestone E — Multi-task fixed heads

Prompt + response + jailbreak.

### Milestone F — Category head

Chỉ sau category provenance audit.

### Milestone G — GLiGuard-inspired schema model

Thay fixed heads bằng schema-conditioned labels trên mmBERT hoặc fine-tune GLiNER2 multi.

---

## 17. Ma trận model

### 17.1. Baseline zero-shot

| ID | Model | Train |
|---|---|---|
| Z0 | mmBERT pretrained + random head | Không dùng để báo accuracy chính, chỉ smoke |
| Z1 | GLiGuard checkpoint gốc | Không train, evaluate EN/VI |
| Z2 | Nemotron 8B v3 nếu có hạ tầng | Evaluate EN/VI, reference |

### 17.2. Controlled language experiments

| ID | Backbone | Train language | Số step |
|---|---|---|---|
| E1 | mmBERT | EN | N |
| V1 | mmBERT | VI | N |
| EV1 | mmBERT | 50% EN + 50% VI | N |
| EV2 | mmBERT | EN + VI full | Khoảng 2N hoặc cùng epoch |

**EV1** là so sánh công bằng chính.  
**EV2** là model dùng toàn bộ dữ liệu, không phải controlled comparison.

### 17.3. Size ablation

| ID | Backbone |
|---|---|
| S-EV | mmBERT-small |
| B-EV | mmBERT-base |

### 17.4. Context ablation

```text
256
512
1024
2048
```

Không bắt đầu bằng 8192.

---

## 18. Sampling bilingual

### 18.1. Paired batch

Trong EV1:

```text
50% English
50% Vietnamese
```

Không để một epoch English rồi một epoch Vietnamese.

Khuyến nghị batch mixer:

```text
sample pair
randomly choose EN or VI for that step
```

hoặc mỗi batch có cặp EN-VI cân bằng.

### 18.2. Translation-pair leakage

EN và VI của cùng record phải ở cùng split.

Không được:

```text
EN train
VI test
```

---

# PHẦN VII — BENCHMARK ĐÁNH GIÁ

## 19. Track 0 — Nemotron paired test

Nguồn:

```text
Nemotron en/test
Vietnamese translation của cùng record_uid
```

Đây là paired diagnostic mạnh, nhưng không phải benchmark hoàn toàn độc lập vì cùng family với train.

Đo:

- Prompt safety.
- Response safety.
- Jailbreak binary.
- Category nếu head được bật.
- EN-VI consistency.

---

## 20. Track 1 — MultiJail

Dataset có:

- 315 English unsafe prompts.
- 315 Vietnamese human-annotated/transcreated prompts.
- Tổng 10 ngôn ngữ, mỗi split 315.

Ưu tiên cao vì không cần machine-translate sang Việt.

Đo:

```text
unsafe recall
false negative rate
EN-VI pair consistency
```

Không đo được safe FPR vì gần như toàn unsafe.

Không được báo accuracy đơn độc vì model always-unsafe có thể trông rất tốt.

---

## 21. Track 2 — RTP-LX Vietnamese

RTP-LX là corpus human-transcreated và human-annotated ở 28 ngôn ngữ, có tiếng Việt.

Ưu tiên cao vì:

- Không phụ thuộc Gemini translation của dự án.
- Có nội dung toxicity tinh tế.
- Có bias, identity attack, microaggression và context-dependent harm.
- Hữu ích để kiểm tra model có chỉ học keyword không.

Codex phải tải đúng dataset chính thức, xác định schema và số mẫu VI bằng code, không hard-code số lượng.

---

## 22. Track 3 — XSTest EN/VI

XSTest gồm:

- 250 safe prompts.
- 200 unsafe contrast prompts.
- Tổng 450.

Dùng để đo:

- Over-refusal.
- Safe false-positive.
- Khả năng hiểu từ nhạy cảm trong ngữ cảnh an toàn.

Cần dịch English sang Vietnamese bằng pipeline benchmark riêng.

Không dùng cùng prompt dịch/train pipeline một cách mù quáng. Phải review ít nhất toàn bộ 250 safe prompts hoặc một tỷ lệ rất cao.

---

## 23. Track 4 — PolyGuardPrompts EN/VI

PolyGuardPrompts:

- 29.325 samples tổng.
- 17 ngôn ngữ.
- Test-only benchmark.
- Có prompt, response, prompt harm, response harm, refusal và adversarial.
- Chưa có Vietnamese trong danh sách ngôn ngữ công bố.

Quy trình:

1. Tải dataset.
2. Lọc `language == "en"`.
3. Đếm exact.
4. Dịch EN subset sang VI.
5. Giữ label gốc.
6. Human-review phân tầng.
7. Không train bằng PolyGuardPrompts.

Đây là external benchmark gần nhất với đầy đủ moderation lifecycle.

---

## 24. Track 5 — XSafety, optional

XSafety có khoảng 28.000 prompt đa ngôn ngữ và được NVIDIA dùng trong evaluation Nemotron v3.

Có thể:

- Lọc English.
- Lấy stratified subset.
- Dịch sang VI.
- Dùng như unsafe coverage track.

Không ưu tiên hơn XSTest vì thiếu safe negatives.

---

## 25. Thứ tự benchmark ưu tiên

```text
P0 Nemotron EN/VI paired
P0 MultiJail EN/VI
P0 RTP-LX VI
P0 XSTest EN/VI
P1 PolyGuardPrompts EN/VI
P2 XSafety VI translated subset
```

---

# PHẦN VIII — METRIC

## 26. Prompt safety

Báo đầy đủ:

```text
accuracy
balanced accuracy
macro F1
unsafe precision
unsafe recall
unsafe F1
safe false-positive rate
unsafe false-negative rate
confusion matrix
```

Metric ưu tiên:

```text
unsafe recall
safe false-positive rate
macro F1
```

---

## 27. Response safety

Chỉ trên rows có response label hợp lệ:

```text
macro F1
unsafe recall
safe FPR
harmful compliance miss rate
```

---

## 28. Jailbreak

```text
jailbreak recall
jailbreak precision
jailbreak F1
generic false-positive rate
```

---

## 29. Category multi-label

Khi được triển khai:

```text
micro F1
macro F1
per-label F1
sample F1
exact match
Hamming loss
mAP
```

Không chỉ báo exact match.

---

## 30. Cross-lingual metrics

### 30.1. Pair consistency

```text
same_binary_decision_rate
```

### 30.2. Probability gap

```text
mean(abs(p_unsafe_en - p_unsafe_vi))
```

### 30.3. Vietnamese gain

```text
score(EV on VI) - score(E on VI)
```

### 30.4. English retention

```text
score(EV on EN) - score(E on EN)
```

### 30.5. Pair flip matrix

```text
EN safe -> VI unsafe
EN unsafe -> VI safe
```

---

## 31. Calibration

Bắt buộc:

```text
Brier score
ECE
reliability bins
risk-coverage curve
```

Threshold deployment không được chọn chỉ theo accuracy.

---

## 32. Confidence interval

Dùng bootstrap theo record/pair:

```text
95% confidence interval
```

Với EN-VI, bootstrap theo `record_uid`, không tách hai ngôn ngữ thành hai sample độc lập.

---

## 33. Hiệu năng

Báo:

```text
parameters
trainable parameters
VRAM peak
examples/second
tokens/second
p50 latency
p95 latency
batch size
max length
hardware
precision
```

---

# PHẦN IX — HARDWARE VÀ CẤU HÌNH HUẤN LUYỆN

## 34. Không giả định GPU

Codex phải detect hardware và hỗ trợ profile.

### 34.1. RTX 3050 4GB

Không full fine-tune mmBERT-base ở context dài.

Ưu tiên:

```text
mmBERT-small
max_length 256 hoặc 512
freeze encoder + train head
hoặc unfreeze top 2-4 layers
gradient accumulation
fp16
gradient checkpointing nếu hỗ trợ
```

LoRA cho ModernBERT chỉ bật sau smoke test tương thích PEFT.

### 34.2. GPU 16-24GB

```text
mmBERT-base
max_length 512-1024
LoRA hoặc partial fine-tune
batch dynamic
fp16/bf16
```

### 34.3. B200/H100/A100

```text
mmBERT-base full fine-tune
1024-2048 trước
long-context ablation
Flash Attention 2
bf16
```

Không dùng 8192 mặc định vì chi phí attention và padding.

---

## 35. Config khởi đầu

Prompt-only baseline:

```yaml
model_name: jhu-clsp/mmBERT-base
max_length: 512
pooling: mean
learning_rate: 2.0e-5
head_learning_rate: 1.0e-4
epochs: 3
warmup_ratio: 0.06
weight_decay: 0.01
dropout: 0.1
gradient_clip_norm: 1.0
metric_for_best_model: macro_f1
early_stopping_patience: 2
seed: 20260720
```

Đây là điểm bắt đầu, không phải cấu hình đảm bảo tối ưu.

---

# PHẦN X — CẤU TRÚC CODE ĐỀ XUẤT

## 36. Cây thư mục

```text
project/
├── configs/
│   ├── data/
│   ├── model/
│   ├── experiment/
│   └── hardware/
├── data/
│   ├── raw/
│   ├── translated/
│   ├── processed/
│   ├── benchmarks/
│   └── manifests/
├── src/
│   └── viguard/
│       ├── data/
│       │   ├── jsonl_io.py
│       │   ├── nemotron_adapter.py
│       │   ├── bilingual_pairing.py
│       │   ├── labels.py
│       │   ├── dedup.py
│       │   └── collators.py
│       ├── models/
│       │   ├── mmbert_fixed_guard.py
│       │   ├── heads.py
│       │   ├── pooling.py
│       │   └── schema_guard.py
│       ├── training/
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   ├── samplers.py
│       │   └── thresholds.py
│       ├── evaluation/
│       │   ├── metrics_binary.py
│       │   ├── metrics_multilabel.py
│       │   ├── paired_metrics.py
│       │   ├── calibration.py
│       │   └── bootstrap.py
│       ├── benchmarks/
│       │   ├── multijail.py
│       │   ├── rtplx.py
│       │   ├── xstest.py
│       │   ├── polyguard.py
│       │   └── xsafety.py
│       ├── reports/
│       │   ├── json_report.py
│       │   └── html_report.py
│       └── cli.py
├── scripts/
│   ├── audit_data.ps1
│   ├── prepare_pairs.ps1
│   ├── run_prompt_baselines.ps1
│   ├── run_multitask.ps1
│   └── run_external_benchmarks.ps1
├── tests/
│   ├── test_jsonl.py
│   ├── test_labels.py
│   ├── test_pairing.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_resume.py
│   └── test_benchmarks.py
├── outputs/
└── README.md
```

---

# PHẦN XI — NHIỆM VỤ CỤ THỂ CHO CODEX

## 37. Nguyên tắc làm việc

Codex phải:

1. Đọc toàn bộ tài liệu này.
2. Kiểm tra cây thư mục thật.
3. Không giả định tên file dịch.
4. Tự tìm manifest/checkpoint/output hiện có.
5. Không xóa hoặc sửa `data/raw`.
6. Không gọi API dịch.
7. Không train model lớn trước smoke test.
8. Chạy unit test sau mỗi milestone.
9. Lưu mọi config, seed, git diff và environment.
10. Không dùng test để chọn hyperparameter.

---

## 38. Milestone 1 — Inspect và audit

Deliverables:

```text
reports/data_inventory.json
reports/data_inventory.html
data/manifests/nemotron_en_vi_pairs.jsonl
```

Audit phải gồm:

- Exact file paths.
- Row counts.
- Valid JSON.
- Unique `record_uid`.
- EN-VI match rate.
- Translation status.
- Split distribution.
- Prompt/response label distribution.
- Tag distribution.
- Category distribution.
- Token-length percentiles bằng mmBERT tokenizer.
- REDACTED.
- Duplicate.
- Cross-split contamination.
- Samples lỗi.

Acceptance:

```text
pytest passes
pair manifest deterministic
rerun gives same hashes
```

---

## 39. Milestone 2 — Prompt safety baseline

Triển khai:

```text
MMBERTFixedGuard
prompt head only
```

Chạy smoke:

```text
32 train rows
16 valid rows
CPU
1 epoch
```

Sau đó small run:

```text
500 train rows
200 valid rows
```

Sau đó full:

```text
E1
V1
EV1
```

Outputs mỗi run:

```text
config.yaml
environment.json
train_log.jsonl
best_checkpoint/
valid_predictions.jsonl
test_predictions.jsonl
metrics.json
report.html
```

---

## 40. Milestone 3 — Response head

Yêu cầu:

- Mask missing response labels.
- Unit test chứng minh missing không đóng góp loss.
- Báo count denominator.
- Không tính response score trên prompt-only rows.

---

## 41. Milestone 4 — Jailbreak binary

Yêu cầu:

- `tag == jailbreaking` là positive.
- Báo performance generic và jailbreaking.
- Báo theo language.
- Kiểm tra model có dùng độ dài làm shortcut.

Ablation:

```text
evaluate after length-matched sampling
```

---

## 42. Milestone 5 — External benchmark

Tải và chuẩn hóa:

```text
MultiJail
RTP-LX
XSTest
PolyGuardPrompts
```

Không train bằng các bộ benchmark này.

Mỗi adapter phải:

- Cache raw.
- Ghi source URL/license/version.
- Hash file.
- Chuẩn hóa schema.
- Giữ raw labels.
- Có unit test.

---

## 43. Milestone 6 — Category audit và head

Trước khi code category head, tạo report:

```text
prompt_label x response_label x categories
prompt-only vs prompt-response
safe rows with non-empty categories
unsafe rows with empty categories
Needs Caution behavior
Other behavior
```

Sau đó chọn một trong hai:

```text
conversation category head
```

hoặc:

```text
restricted prompt category head
```

Quyết định phải ghi trong ADR:

```text
docs/adr/ADR-001-category-target.md
```

---

## 44. Milestone 7 — Schema-conditioned model

Chỉ bắt đầu sau khi fixed-head baseline hoàn tất.

Hai nhánh:

### Nhánh A — mmBERT schema head

Thêm special tokens:

```text
[P]
[L]
[SEP_SCHEMA]
```

Input:

```text
schema prefix + text
```

Lấy hidden state tại label anchors.

Dùng shared MLP.

### Nhánh B — GLiNER2 multi

Fine-tune:

```text
fastino/gliner2-multi-v1
```

với schema safety.

So sánh:

```text
fixed-head mmBERT
schema mmBERT
GLiNER2 multi
GLiGuard original zero-shot
```

---

# PHẦN XII — BẢNG KẾT QUẢ CUỐI

## 45. Bảng model chính

Do tài liệu Word không phù hợp với một bảng có quá nhiều cột, báo cáo cuối phải tách thành các bảng hẹp dưới đây.

### 45.1. Danh mục model

| Model | Dữ liệu train | Vai trò |
|---|---|---|
| GLiGuard zero-shot | Không train | Baseline kiến trúc gốc |
| E1 mmBERT | English | Baseline tiếng Anh |
| V1 mmBERT | Vietnamese | Baseline tiếng Việt |
| EV1 mmBERT | EN+VI, cùng số step | So sánh song ngữ có kiểm soát |
| EV2 mmBERT | EN+VI đầy đủ | Model khai thác toàn bộ dữ liệu |
| Schema mmBERT | EN+VI | Kiến trúc schema-conditioned |
| GLiNER2 multi | EN+VI | Ablation cùng họ GLiNER2 |

### 45.2. Kết quả Nemotron paired

| Model | Nemotron EN | Nemotron VI | Pair consistency |
|---|---:|---:|---:|
| GLiGuard zero-shot | | | |
| E1 mmBERT | | | |
| V1 mmBERT | | | |
| EV1 mmBERT | | | |
| EV2 mmBERT | | | |
| Schema mmBERT | | | |
| GLiNER2 multi | | | |

### 45.3. Benchmark unsafe và toxicity

| Model | MultiJail EN | MultiJail VI | RTP-LX VI |
|---|---:|---:|---:|
| GLiGuard zero-shot | | | |
| E1 mmBERT | | | |
| V1 mmBERT | | | |
| EV1 mmBERT | | | |
| EV2 mmBERT | | | |
| Schema mmBERT | | | |
| GLiNER2 multi | | | |

### 45.4. Benchmark over-refusal và moderation đầy đủ

| Model | XSTest VI | PolyGuard VI | Ghi chú |
|---|---:|---:|---|
| GLiGuard zero-shot | | | |
| E1 mmBERT | | | |
| V1 mmBERT | | | |
| EV1 mmBERT | | | |
| EV2 mmBERT | | | |
| Schema mmBERT | | | |
| GLiNER2 multi | | | |

---

## 46. Bảng cross-lingual

### 46.1. Độ nhất quán và mức cải thiện

| Model | Pair consistency | Mean probability gap | VI gain |
|---|---:|---:|---:|
| E1 | | | |
| V1 | | | |
| EV1 | | | |
| EV2 | | | |

### 46.2. Giữ năng lực tiếng Anh và hướng flip

| Model | EN retention | EN safe -> VI unsafe | EN unsafe -> VI safe |
|---|---:|---:|---:|
| E1 | | | |
| V1 | | | |
| EV1 | | | |
| EV2 | | | |

---

## 47. Bảng vận hành

### 47.1. Kích thước model

| Model | Params | Trainable params | Max length |
|---|---:|---:|---:|
| GLiGuard | | | |
| mmBERT-small | | | |
| mmBERT-base | | | |

### 47.2. Tài nguyên và thông lượng

| Model | Batch size | VRAM peak | Samples/second |
|---|---:|---:|---:|
| GLiGuard | | | |
| mmBERT-small | | | |
| mmBERT-base | | | |

### 47.3. Độ trễ

| Model | p50 latency | p95 latency | Hardware |
|---|---:|---:|---|
| GLiGuard | | | |
| mmBERT-small | | | |
| mmBERT-base | | | |

---

# PHẦN XIII — DEFINITION OF DONE

## 48. Data

- [ ] Không sửa raw.
- [ ] EN-VI pair manifest deterministic.
- [ ] Split giữ nguyên.
- [ ] REDACTED được loại hoặc reconstruct có provenance.
- [ ] Translation failures không bị lẫn vào VI.
- [ ] Duplicate/contamination report tồn tại.

## 49. Model

- [ ] Prompt baseline chạy được.
- [ ] Response masked loss đúng.
- [ ] Jailbreak binary đúng nghĩa.
- [ ] Category không triển khai khi chưa audit.
- [ ] Checkpoint resume được.
- [ ] Seed/config/environment được lưu.

## 50. Evaluation

- [ ] Không chọn threshold trên test.
- [ ] Có macro F1, recall, FPR, confusion.
- [ ] Có paired EN-VI metrics.
- [ ] Có bootstrap CI.
- [ ] Có calibration.
- [ ] Có clean-only external benchmark score.
- [ ] Có throughput/latency.

## 51. Reproducibility

- [ ] Một PowerShell script chạy lại từng experiment.
- [ ] Output không bị ghi đè âm thầm.
- [ ] Hash dataset/config/checkpoint.
- [ ] Pytest pass.
- [ ] README có lệnh Windows PowerShell.

---

# PHẦN XIV — NHỮNG ĐIỀU KHÔNG ĐƯỢC LÀM

1. Không chia lại official split.
2. Không dùng NVIDIA `id` làm unique key.
3. Không coi `response_label = ""` là class.
4. Không suy `prompt_label` từ category.
5. Không coi `Needs Caution` mặc định là unsafe.
6. Không dùng `tag` như 11 jailbreak strategy.
7. Không tạo refusal label bằng heuristic rồi gọi là gold.
8. Không train bằng external benchmark.
9. Không dùng test để chọn threshold/hyperparameter.
10. Không chỉ báo accuracy.
11. Không so EV 80K với E 40K rồi kết luận hoàn toàn do bilingual.
12. Không gọi fixed-head mmBERT là GLiGuard.
13. Không tuyên bố benchmark dịch là benchmark Việt Nam bản địa hoàn chỉnh.
14. Không khẳng định translation quality nếu chưa review.
15. Không full fine-tune mmBERT-base trên 4GB VRAM bằng cấu hình không khả thi.

---

# PHẦN XV — KẾT LUẬN KHOA HỌC MONG MUỐN

Kết luận tốt phải có dạng:

> Với cùng backbone mmBERT và cùng ngân sách optimizer step, bổ sung dữ liệu tiếng Việt làm tăng macro-F1/unsafe recall trên các benchmark tiếng Việt và tăng độ nhất quán EN-VI, trong khi điểm tiếng Anh không giảm đáng kể. Kết quả vẫn tồn tại trên các benchmark độc lập, đặc biệt MultiJail VI, RTP-LX VI và XSTest VI.

Không được chỉ kết luận:

> Accuracy tăng sau train tiếng Việt.

Sau baseline này, dự án mới có cơ sở để xây:

- Schema-conditioned Vietnamese guard.
- Taxonomy Việt Nam.
- Refusal head.
- PII head.
- Policy routing.
- Guardrail cho Viettel Super 120B.

---

# PHẦN XVI — TÀI LIỆU VÀ CODE THAM KHẢO

## 52. GLiGuard

- Paper: https://arxiv.org/abs/2605.07982
- Code: https://github.com/fastino-ai/GLiGuard
- Model: https://huggingface.co/fastino/gliguard-LLMGuardrails-300M

## 53. GLiNER2

- Paper: https://arxiv.org/abs/2507.18546
- ACL Anthology: https://aclanthology.org/2025.emnlp-demos.10/
- Code: https://github.com/fastino-ai/GLiNER2
- English base: https://huggingface.co/fastino/gliner2-base-v1
- Multilingual: https://huggingface.co/fastino/gliner2-multi-v1

## 54. mmBERT

- Paper: https://arxiv.org/abs/2509.06888
- Code: https://github.com/jhu-clsp/mmBERT
- Model: https://huggingface.co/jhu-clsp/mmBERT-base
- Small model: https://huggingface.co/jhu-clsp/mmBERT-small
- Technical blog: https://huggingface.co/blog/mmbert

## 55. Nemotron Safety Guard

- Model card: https://build.nvidia.com/nvidia/llama-3_1-nemotron-safety-guard-8b-v3/modelcard
- Dataset: https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3
- CultureGuard paper: https://arxiv.org/abs/2508.01710
- NeMo Guardrails: https://github.com/NVIDIA/NeMo-Guardrails

## 56. Benchmarks

- MultiJail: https://huggingface.co/datasets/walledai/MultiJail
- MultiJail paper: https://arxiv.org/abs/2310.06474
- RTP-LX: https://huggingface.co/datasets/adewynter/RTP-LX
- RTP-LX paper: https://arxiv.org/abs/2404.14397
- XSTest code/data: https://github.com/paul-rottger/xstest
- XSTest paper: https://arxiv.org/abs/2308.01263
- PolyGuardPrompts: https://huggingface.co/datasets/ToxicityPrompts/PolyGuardPrompts
- PolyGuard paper: https://arxiv.org/abs/2504.04377
- XSafety: https://huggingface.co/datasets/ToxicityPrompts/XSafety

---

# PHỤ LỤC A — PROMPT NGẮN GỬI CODEX

```text
Read the attached specification completely.

Work in the current project directory. First inspect the real files,
translation outputs, manifests, checkpoints, configs, and existing code.
Do not assume translated filenames and do not redownload or retranslate data.

Implement only Milestone 1 and Milestone 2 first:
1. deterministic EN-VI data audit and pair manifest;
2. prompt-safety-only mmBERT baseline;
3. dry-run and unit tests;
4. tiny CPU smoke run;
5. exact Windows PowerShell commands for E1, V1, and EV1.

Do not:
- touch data/raw;
- call external translation APIs;
- process test for hyperparameter selection;
- add category or refusal heads yet;
- start schema-conditioned GLiGuard implementation yet;
- launch a full expensive training run without showing the smoke-test report.

At the end, provide:
- changed file tree;
- pytest output;
- audit summary;
- smoke-run metrics;
- commands for the first controlled full experiment;
- any unresolved assumptions or blockers.
```
