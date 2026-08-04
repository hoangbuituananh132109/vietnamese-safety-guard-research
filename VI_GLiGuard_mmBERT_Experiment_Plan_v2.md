# Kế hoạch thí nghiệm v2: mmBERT fixed-head, schema-conditioned guard và SEA-SafeguardBench

**Ngày chốt thiết kế:** 20/07/2026  
**Trạng thái dữ liệu:** 45.416 bản ghi Nemotron EN–VI đã sẵn sàng  
**Mục tiêu của v2:** biến các đề xuất mới thành một chuỗi thí nghiệm có đối chứng, chạy được theo từng tầng, không đánh đồng ảnh hưởng của ngôn ngữ, kiến trúc, backbone và độ dài ngữ cảnh.

---

## 1. Các quyết định chính

1. **Giữ cả `generic` và `jailbreaking` trong train/valid/test nếu instance vừa context.**
   `tag` được giữ làm lát cắt báo cáo, không dùng làm filter. Mẫu chỉ chuyển sang
   tail/quarantine vì giới hạn token thực của model, dữ liệu rỗng/REDACTED hoặc
   vi phạm target contract; không loại chỉ vì mang tag jailbreak.

2. **Giữ cả ba chế độ đầu vào nhưng coi chúng là ba bài toán riêng:**
   - `P`: prompt-only, dự đoán `prompt_label`;
   - `R`: response-only, dự đoán `response_label`;
   - `PR`: prompt + response, vẫn dự đoán `response_label`.

3. **Không gọi random head của mmBERT là baseline “trước huấn luyện”.**
   mmBERT gốc không có head safety nên logits trước huấn luyện không mang nghĩa khoa học. So sánh trước/sau hợp lệ sẽ là:
   - frozen encoder + head đã train: linear/feature probe;
   - cùng head nhưng có adaptation backbone bằng partial unfreeze hoặc LoRA;
   - random/majority chỉ là sanity check.

4. **So sánh English-only với bilingual bằng hai ngân sách khác nhau:**
   - `EV-matched`: EN+VI nhưng tổng số sample/optimizer step bằng English-only;
   - `EV-full`: dùng cả hai bản EN và VI, do đó ngân sách dữ liệu lớn gấp đôi.
   Chỉ `E` so với `EV-matched` mới trả lời tương đối sạch câu hỏi “tiếng Việt có giúp không”. `EV-full` trả lời câu hỏi thực dụng “dùng toàn bộ dữ liệu thì tốt đến đâu”.

5. **GLiGuard 512 không có 512 token dành riêng cho văn bản.**
   Schema, task name, label name, label description, special token và văn bản đều dùng chung giới hạn sequence. Schema N23 có 23 nhãn sẽ để lại ít token văn bản hơn schema G14 có 14 nhãn.

6. **Không dùng P99 theo ký tự để cắt dữ liệu.**
   P95/P99 phải tính lại sau khi serialize đúng từng task và tokenize bằng đúng tokenizer của từng model. P99 được dùng để tạo bucket và chọn profile chạy; không dùng làm lý do xóa vĩnh viễn các mẫu dài.

7. **Cần hai kiểu so sánh độ dài độc lập:**
   - `common-capacity`: hai model nhận cùng một ngân sách token văn bản và cùng quy tắc truncation;
   - `native-capacity`: GLi dùng tối đa native 512 total sequence, mmBERT được dùng 1.024/2.048 và về sau có thể 8.192.

8. **N23 đã có sẵn; không cần GPT gán lại 23 category.**
   LLM annotation chỉ cần cân nhắc về sau nếu muốn refusal label, 11 jailbreak strategy, hoặc muốn tách category riêng cho prompt và response.

9. **SEA-SafeguardBench là external benchmark cuối, không dùng để train, chọn threshold hoặc prompt-tune.**
   Trong báo cáo, ITW Việt Nam và CG Việt Nam là bằng chứng văn hóa quan trọng hơn General; General có nguồn Aegis/JailbreakBench/WildGuard nên phải audit contamination với Nemotron.

---

## 2. Audit lịch sử trên generic-only (không còn là filter train)

Nguồn là các file cuối:

```text
data/final/nemotron_train_en_vi_v10_final.jsonl
data/final/nemotron_valid_en_vi_v10_final.jsonl
data/final/nemotron_test_en_vi_v10_final.jsonl
```

Điều kiện dưới đây chỉ giải thích provenance của các bảng audit generic-only đã
tính trước đó. Kế hoạch thực thi v3 dùng cả hai tag:

```text
tag == "generic"
prompt_en not in {"", "REDACTED"}
prompt_vi not empty
translation_disposition == "ready"
```

### 2.1. Prompt-only usable

| Split | Generic raw | Loại rỗng/REDACTED | Prompt usable | Safe | Unsafe |
|---|---:|---:|---:|---:|---:|
| Train | 30.007 | 912 | 29.095 | 11.889 | 17.206 |
| Valid | 1.445 | 44 | 1.401 | 553 | 848 |
| Test | 1.964 | 36 | 1.928 | 889 | 1.039 |

Mỗi record usable có một bản EN và một bản VI, nhưng hai ngôn ngữ phải giữ chung `record_uid` để split, sampling và bootstrap theo cặp.

### 2.2. Response usable

Chỉ giữ record có:

```text
response_en != ""
response_vi != ""
response_label in {"safe", "unsafe"}
```

| Split | Response text + label usable | Safe | Unsafe | Có label nhưng thiếu response, phải bỏ |
|---|---:|---:|---:|---:|
| Train | 14.431 | 10.890 | 3.541 | 803 |
| Valid | 686 | 441 | 245 | 35 |
| Test | 813 | 419 | 394 | 39 |

`R` và `PR` dùng cùng danh sách record; chỉ khác serialization đầu vào. Không được đưa 803/35/39 hàng thiếu response vào loss.

### 2.3. Dấu hiệu độ dài của lát cắt generic-only

Các số dưới đây là **ký tự**, chỉ dùng để hình dung trước khi có tokenizer audit:

| Mode / language / train | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|
| Prompt EN | 70 | 1.458 | 3.694 | 18.504 |
| Prompt VI | 78 | 1.539 | 3.929 | 18.053 |
| Response EN | 446 | 841 | 955 | 3.695 |
| Response VI | 470 | 897 | 1.034 | 3.951 |
| Prompt+Response EN | 529 | 923 | 1.054 | 3.764 |
| Prompt+Response VI | 564 | 991 | 1.145 | 4.021 |

Nhận xét ban đầu:

- Lát cắt generic-only làm phần response ngắn đi rất nhiều; đây chính là lý do
  không được dùng nó để đại diện cho coverage của full train nữa.
- Tail dài đáng kể vẫn tồn tại ở prompt generic.
- `R` và `PR` có khả năng vừa 512 tốt hơn `P`, nhưng chỉ tokenizer audit mới kết luận được.
- mmBERT 2K có thể bao phủ gần hết generic, nhưng đây mới là giả thuyết cần đo theo token, không phải kết luận từ số ký tự.

### 2.4. Duplicates và leakage

Phải báo hai view trên valid/test:

- `official`: giữ nguyên NVIDIA split để so sánh với nguồn;
- `clean`: loại exact normalized prompt xuất hiện trong train.

Không reshuffle official split. Mọi so sánh EN/VI phải dùng đúng cùng tập `record_uid`.

---

## 3. Ba task đầu vào phải tách rõ

### 3.1. Task P — prompt safety

Input fixed-head:

```text
Prompt: {prompt}
```

Target:

```text
prompt_label: safe | unsafe
```

Đây là task đầu tiên và là task duy nhất dùng để quyết định model/language recipe trước khi mở rộng.

### 3.2. Task R — response safety không có prompt

Input:

```text
Response: {response}
```

Target:

```text
response_label: safe | unsafe
```

Task này đo xem bản thân output có đủ tín hiệu để moderation hay không.

### 3.3. Task PR — contextual response safety

Input:

```text
Prompt: {prompt}
Response: {response}
```

Target vẫn là:

```text
response_label: safe | unsafe
```

So sánh `R` với `PR` trả lời trực tiếp giá trị của context prompt. Không dùng `prompt_label` làm target của chuỗi PR, vì khi triển khai thực tế prompt phải được kiểm tra trước khi response tồn tại.

### 3.4. Unified model về sau

Sau khi ba task độc lập đã ổn, model schema có thể học ba task name khác nhau:

```text
prompt_safety
response_safety
contextual_response_safety
```

Không trộn âm thầm hai format `R` và `PR` dưới cùng một task name trong thí nghiệm đầu, vì điều đó làm kết quả khó giải thích.

---

## 4. “Trước và sau huấn luyện” của mmBERT phải đo như thế nào

### 4.1. Những baseline có nghĩa

| ID | Encoder | Head | Ý nghĩa |
|---|---|---|---|
| `B-majority` | không dùng | không dùng | sanity theo class prior |
| `B-random` | pretrained, frozen | random, không train | chỉ kiểm tra pipeline; không dùng kết luận |
| `M-probe` | pretrained, frozen | train | chất lượng feature trước safety adaptation |
| `M-adapt` | LoRA hoặc unfreeze top layers | train | chất lượng sau safety adaptation |

So sánh chính:

```text
M-probe vs M-adapt
```

Random head có thể đạt tình cờ 50% nhưng không đại diện cho khả năng zero-shot của mmBERT.

### 4.2. Chế độ phù hợp RTX 3050 Laptop 4 GB

Thứ tự thử:

1. mmBERT-small frozen encoder + train head;
2. mmBERT-base frozen encoder + train head;
3. mmBERT-small LoRA/partial unfreeze;
4. mmBERT-base LoRA nếu smoke test VRAM đạt;
5. full fine-tune chỉ khi có GPU lớn hơn.

Mọi so sánh kiến trúc phải dùng cùng backbone size và cùng trainable regime. Không so fixed-head frozen với schema-LoRA rồi kết luận do schema.

---

## 5. English-only và bilingual

### 5.1. Các recipe ngôn ngữ

| Recipe | Dữ liệu mỗi semantic record | Số language instance tương đối | Mục đích |
|---|---|---:|---|
| `E` | chỉ EN | 1x | English baseline và zero-shot VI transfer |
| `V` | chỉ VI | 1x | optional nhưng hữu ích để đo native-only |
| `EV-matched` | chọn EN hoặc VI theo cặp/schedule, tổng vẫn 1x | 1x | đối chứng công bằng với E |
| `EV-full` | cả EN và VI | 2x | model thực dụng dùng toàn bộ data |

### 5.2. Sampling của EV-matched

Hai cách chấp nhận được:

1. mỗi epoch chọn ngẫu nhiên đúng một ngôn ngữ cho mỗi `record_uid`, cân bằng 50/50;
2. tạo sampler luân phiên EN/VI nhưng dừng khi tổng số example bằng E.

Phải giữ:

```text
same optimizer updates
same effective batch size
same warmup ratio
same model selection metric
same seed family
```

### 5.3. Đánh giá

Mỗi checkpoint được đánh giá trên:

```text
Nemotron valid EN
Nemotron valid VI
Nemotron test EN
Nemotron test VI
paired EN-VI consistency
official view
clean no-train-overlap view
```

Threshold được chọn trên valid theo recipe đã định trước rồi freeze. Không chọn lại bằng test hoặc SEA benchmark.

---

## 6. Thiết kế so sánh độ dài công bằng

### 6.1. Token audit bắt buộc

Phải tokenize sáu serialization độc lập:

```text
P-EN, P-VI
R-EN, R-VI
PR-EN, PR-VI
```

Với mỗi backbone/tokenizer:

```text
mmBERT tokenizer
GLiGuard/GLiNER2 tokenizer
```

Với schema model phải báo riêng:

```text
schema_tokens
text_tokens
special_tokens
total_tokens
available_text_budget = max_sequence - schema_tokens - special_tokens
```

### 6.2. Track C — common-capacity

Mục tiêu: so kiến trúc mà không để context quyết định thay.

1. Serialize schema thật.
2. Đo text budget còn lại của cấu hình schema dài nhất dùng trong so sánh.
3. Chọn `B_common` bằng budget nhỏ nhất giữa hai model.
4. Hai model nhận cùng tối đa `B_common` text token.
5. Dùng cùng quy tắc head-tail truncation.

Khởi điểm dự kiến để smoke test:

```text
B_common = 384 text tokens
```

Nhưng số cuối phải lấy từ tokenizer audit, không hard-code theo phỏng đoán.

### 6.3. Track F — fit-common, không truncation

Một record thuộc tập này khi serialization của nó vừa cả hai model mà không cắt:

```text
fits_mmbert == true
fits_gli_schema == true
```

Đây là tập quan trọng nhất để trả lời: nếu không bị bất lợi vì context, GLi schema hay fixed head tốt hơn?

### 6.4. Track N — native-capacity

| Model | Profile ban đầu |
|---|---:|
| GLi/GLiNER2 schema | 512 total sequence |
| mmBERT fixed/schema | 1.024 text token |
| mmBERT long | 2.048 text token |
| mmBERT 8K | để sau, không phải mặc định |

Track này trả lời câu hỏi sản phẩm, không dùng để cô lập thuần kiến trúc.

### 6.5. Bucket độ dài báo cáo

Không chỉ báo một điểm tổng:

```text
fits-common-no-truncation
common < length <= 512-text-equivalent
513..1024
1025..2048
>2048
top 1% longest
```

Ranh giới được tính bằng token, không dùng ký tự.

### 6.6. Truncation và long-input

Primary fair run:

```text
head-tail truncation giống nhau cho cả hai model
```

Product ablation về sau:

- prompt: giữ phần đầu và ưu tiên phần cuối chứa yêu cầu thực;
- PR: bảo đảm response có ngân sách tối thiểu, prompt dùng phần còn lại;
- chunking: score từng chunk rồi aggregate bằng max-risk/noisy-OR;
- chunking phải báo riêng, không trộn vào kết quả single-pass.

---

## 7. GLiGuard thực sự hoạt động như thế nào

Theo paper GLiGuard:

1. Mỗi task bắt đầu bằng token `[P]` và phần mô tả task tự nhiên.
2. Mỗi candidate label được đặt sau token `[L]`.
3. `[SEP]` tách schema khỏi text.
4. Schema + text chạy chung qua một bidirectional encoder.
5. Hidden state tại mỗi `[L]` là contextualized representation của label.
6. Cùng một MLP hai tầng chấm scalar logit cho mọi label.
7. Single-label dùng softmax/CrossEntropy; multi-label dùng sigmoid/BCE.

Đây không phải một cross-attention module riêng. Sự tương tác schema–text xuất hiện nhờ standard full bidirectional self-attention.

Nguồn:

- Paper: <https://arxiv.org/html/2605.07982>
- Model/code usage: <https://github.com/fastino-ai/GLiGuard>
- Checkpoint: <https://huggingface.co/fastino/gliguard-LLMGuardrails-300M>

### 7.1. “Schema linh động” có giới hạn gì?

GLiGuard cho phép ghép **các task/label block mà model đã được học hỗ trợ** theo nhiều tổ hợp. Không nên diễn giải thành “đưa tên nhãn hoàn toàn mới bất kỳ là model luôn hiểu chính xác”.

Với N23, ta phải fine-tune để model nhìn thấy:

- tên nhãn;
- mô tả nhãn;
- positive và hard negative của nhãn;
- nhiều thứ tự label;
- nhiều schema subset.

### 7.2. Schema augmentation từ paper

Paper công bố:

```text
label shuffle: mỗi step
label dropout: p = 0.15
task removal: p = 0.05
```

Triển khai của dự án phải thêm invariant an toàn:

- không bao giờ drop ground-truth label của single-label task;
- không drop positive labels của multi-label task;
- chỉ drop negative candidates;
- giữ ít nhất một negative competitor cho single-label;
- task removal không được xóa toàn bộ task của một sample;
- có xác suất dùng full N23 schema để tránh train/inference mismatch.

Paper không mô tả đủ corner case này trong public training code, nên cần unit test thay vì sao chép mù.

### 7.3. Schema sampling đề xuất cho N23

Mỗi category sample:

```text
all positive labels
+ K sampled negative labels
+ optional Benign
```

Khởi điểm:

```yaml
negative_label_dropout_probability: 0.15
full_schema_probability: 0.25
label_shuffle: true
task_removal_probability: 0.05
minimum_negative_labels: 5
```

Các giá trị này phải được ablate; không coi là tối ưu mặc định.

---

## 8. 23 nhãn Nemotron có đưa vào GLi schema được không?

### 8.1. Trả lời ngắn

**Về kiến trúc: có.** Multi-label schema có thể chứa 23 candidate label.  
**Về checkpoint GLiGuard phát hành: không thể giả định nó đã hiểu đủ N23.** Checkpoint được train trên 14 harm category khác với taxonomy Nemotron.

### 8.2. N23 và G14 không one-to-one

Nemotron có các nhãn không có đối ứng sạch trong G14, ví dụ:

```text
Needs Caution
Profanity
Harassment
Threat
Controlled/Regulated Substances
Malware
High Risk Gov Decision Making
```

Ngược lại, GLiGuard tách:

```text
pii_exposure
privacy_violation
```

trong khi Nemotron dùng một nhãn `PII/Privacy`.

Vì vậy cần hai track taxonomy:

| Track | Taxonomy | Vai trò |
|---|---|---|
| `N23-native` | 23 nhãn Nemotron + optional Benign | train/eval chính trên dữ liệu của ta |
| `G14-compat` | map có provenance sang 14 GLi labels + Benign | kiểm tra checkpoint/compatibility, không ghi đè N23 |

Mapping G14 phải là file versioned, cho phép many-to-many và đánh dấu `ambiguous`/`no_exact_match`. Không biến mapping suy diễn thành gold label.

### 8.3. `Benign` không được suy trực tiếp từ `prompt_label == safe`

Đây là khác biệt rất quan trọng với GLiGuard.

Trong phần generic usable của local data, có 2.617 record prompt được gắn `safe`, không có response safety hợp lệ, nhưng vẫn có category. Trong đó 2.448 record có `Needs Caution`; ngoài ra còn có một số `Unauthorized Advice`, `PII/Privacy`, `Harassment`...

Do đó:

```text
Benign := violated_categories rỗng
```

không phải:

```text
Benign := prompt_label == safe
```

Binary safety và category phải là hai target độc lập. Một prompt có thể `safe` ở binary task nhưng vẫn positive cho `Needs Caution`.

### 8.4. Category là target nào?

Local file chỉ có một field `violated_categories`, không tách:

```text
prompt_categories
response_categories
```

Trong generic, response unsafe chỉ xuất hiện cùng prompt unsafe; vì vậy không thể học sạch response category chỉ từ field hiện có.

Quyết định Phase 1–2:

- binary P/R/PR: train bình thường;
- N23 category: ưu tiên prompt/conversation category, ghi rõ provenance;
- chưa gọi N23 head là `response_toxicity`;
- muốn response category thật sự thì phải lấy nguồn có nhãn response category hoặc annotation riêng.

---

## 9. Có cần GPT gắn nhãn như GLiGuard không?

### 9.1. GLiGuard đã dùng GPT cho gì?

GLiGuard dùng WildGuardTrain có sẵn:

```text
prompt safety
response safety
refusal
```

Sau đó GPT-4.1 bổ sung weak labels:

```text
14 harm categories cho unsafe prompt/response
11 jailbreak strategies cho unsafe prompt
```

Safe sample được gán Benign mặc định trong pipeline của họ.

### 9.2. Nemotron của ta đã có gì?

| Task | Có sẵn? | Có cần GPT ngay? |
|---|---|---|
| Prompt safe/unsafe | Có | Không |
| Response safe/unsafe | Có ở subset | Không |
| N23 multi-label | Có | Không |
| Generic/jailbreaking tag | Có | Không cho binary origin tag |
| Refusal/compliance | Không | Có thể cần nguồn khác/annotation sau |
| 11 jailbreak strategies | Không | Chỉ cần nếu thật sự muốn task strategy |
| Prompt vs response category riêng | Không | Cần nguồn khác hoặc annotation có kiểm soát |

### 9.3. Kết luận annotation

Phase hiện tại **không gọi GPT để gắn lại category**. Việc đó vừa tốn chi phí vừa thay gold/observed taxonomy bằng weak supervision không cần thiết.

Nếu mở Phase jailbreak sau:

- dùng `tag` cho binary `jailbreaking` vs `generic`;
- không gán mọi unsafe generic vào một jailbreak strategy;
- nếu cần 11 strategy, annotate riêng tập jailbreaking và một tập hard-negative generic;
- lưu annotator model, prompt version, confidence và human audit sample.

---

## 10. Ma trận thí nghiệm tối thiểu

### 10.1. Phase A — Prompt safety, giữ toàn bộ source tag đủ điều kiện

Đây là phase ưu tiên số 1.

| ID | Model | Train language | Capacity | Mục đích |
|---|---|---|---|---|
| A00 | Majority/random sanity | none | common | kiểm pipeline |
| A01 | mmBERT fixed, frozen encoder | E | common | pretrained feature probe |
| A02 | mmBERT fixed, frozen encoder | EV-matched | common | bilingual effect khi encoder frozen |
| A03 | mmBERT fixed, adapted backbone | E | common | English safety adaptation |
| A04 | mmBERT fixed, adapted backbone | EV-matched | common | bilingual effect chính |
| A05 | mmBERT fixed, adapted backbone | EV-full | common | full-data product candidate |
| A06 | released GLiGuard | none | native 512 | zero-shot reference, G14/safety |

`A03 vs A04` là phép so sánh ngôn ngữ chính.  
`A01 vs A03` và `A02 vs A04` là trước/sau adaptation.

### 10.2. Phase B — Fixed head vs schema

| ID | Backbone | Head | Train | Capacity | Ý nghĩa |
|---|---|---|---|---|---|
| B01 | mmBERT | fixed | EV-matched | common | anchor |
| B02 | mmBERT | schema shared-MLP | EV-matched | common | kiến trúc-controlled |
| B03 | GLiNER2/GLiGuard-family | schema | EV-matched | common | product/reference comparison |
| B04 | mmBERT | fixed | EV-full | native 1K/2K | native-capacity |
| B05 | GLiNER2/GLiGuard-family | schema | EV-full | native 512 | native-capacity |

Phép so sánh kiến trúc sạch nhất là `B01 vs B02`, vì cùng backbone.  
`B01 vs B03` bị confound bởi backbone, tokenizer và pretraining nhưng vẫn hữu ích về mặt sản phẩm.

### 10.3. Phase C — Response modes

Chỉ dùng recipe thắng/ổn định từ Phase A–B:

| ID | Mode | Train language | Model branch |
|---|---|---|---|
| C01 | R | E | best fixed |
| C02 | R | EV-matched | best fixed |
| C03 | PR | E | best fixed |
| C04 | PR | EV-matched | best fixed |
| C05 | R | EV-matched | best schema |
| C06 | PR | EV-matched | best schema |

So sánh:

```text
C01 vs C02: tác động bilingual cho response-only
C03 vs C04: tác động bilingual cho contextual response
C02 vs C04: giá trị của prompt context ở fixed head
C05 vs C06: giá trị của prompt context ở schema model
```

### 10.4. Phase D — N23 category

| ID | Model | Taxonomy | Schema mode |
|---|---|---|---|
| D01 | mmBERT fixed multi-label head | N23 | fixed 23 logits |
| D02 | mmBERT schema shared-MLP | N23 | stochastic subset + full schema |
| D03 | GLi family schema | N23 | stochastic subset + full schema |
| D04 | released GLiGuard | G14 | zero-shot compatibility only |

Không chạy Phase D trước khi xong audit category attribution và label frequency theo split.

### 10.5. Phase J — Task chiến lược jailbreak riêng, để sau

Dữ liệu có tag jailbreak vẫn tham gia binary safety Phase A–C. Phase J chỉ là
task bổ sung để phân loại jailbreak/strategy và long-context riêng; hoãn Phase J
không có nghĩa bỏ các mẫu jailbreak khỏi binary guard.

```text
J01 mmBERT binary tag classifier
J02 schema binary jailbreak classifier
J03 long-context 512/1K/2K/8K length ablation
J04 optional 11-strategy annotation experiment
```

---

## 11. SEA-SafeguardBench làm benchmark cuối

> **Cập nhật audit 21/07/2026:** phần thực thi đã được rút gọn và chốt tại
> `GUARD_EXPERIMENT_EXECUTION_PLAN_LOCKED_V3.md`. Thông tin SEA dưới đây thay thế
> trạng thái “chưa tìm thấy dataset” trước đó.

Nguồn chính thức đã xác minh:

- ACL Findings paper: <https://aclanthology.org/2026.findings-acl.194/>
- Paper HTML/data description: <https://arxiv.org/html/2512.05501>

Paper công bố 21.640 prompt/response sample trên English + 7 ngôn ngữ Đông Nam Á, gồm General, In-the-Wild Cultural và Content Generation Cultural. Mỗi instance SEA được ghép với English để đánh giá cross-lingual.

### 11.1. Release thực tế trong SEA-HELM

Archive cục bộ `SEA-HELM-main.zip` có SHA-256:

```text
f9abd0b6f1f51d3ced8121716fee413f6f794aba1bcc9ff8bd650e29c29ff25b
```

Dữ liệu được bundle trực tiếp dưới:

```text
external/SEA-HELM-main/seahelm_tasks/safety/safeguard/
```

Loader gọi `datasets.load_dataset("json", split="train", data_files=filepath)`
trên file JSONL local. Chữ `train` chỉ là tên split kỹ thuật của JSON loader;
các file này được dùng để evaluation. Không có Hugging Face dataset ID, không
cần token gated và không cần tải dataset riêng.

Task group `sea_safeguard` đăng ký:

```text
safeguard_general_prompt
safeguard_general_response
safeguard_cultural_content_generation_prompt
safeguard_cultural_content_generation_response
safeguard_cultural_in_the_wild
```

Response task serialize cả `{prompt_text}` và `{response_text}`, rồi chấm
`response_label`; vì vậy đây là `PR`, không phải `R-only`.

### 11.2. Số lượng đã audit

| Subset/view | EN official | VI official |
|---|---:|---:|
| General P | 600 | 600 |
| General PR | 600 | 600 |
| Cultural Content P | 195 | 110 |
| Cultural Content PR | 195 | 110 |
| Cultural In-the-Wild P | 0 | 420 |

Tổng official EN–VI là 3.430 task instance: EN 1.590, VI 1.840. Toàn bộ file
bundled của tám ngôn ngữ có 8.644 dòng JSON và tạo 14.278 registered task
instance. Con số cục bộ này không bằng 21.640 trong mô tả paper; không được gọi
archive hiện tại là full paper release nếu chưa có bằng chứng bổ sung.

Manifest paired diagnostic có 1.840 pair unit / 3.680 language instance:

- General: ghép theo row index; 600/600 khớp prompt label, response label và topic,
  nhưng không có explicit pair ID;
- Cultural Content VI: mỗi row có `metadata.en_prompt` và
  `metadata.en_response`; các counterpart này không nằm trong standalone EN file;
- Cultural In-the-Wild VI: mỗi row có `local_prompt` và `en_prompt` trực tiếp;
  không có standalone EN task.

### 11.3. Ba subset và cách dùng

#### General

- 600 prompt-response instance mỗi language;
- nguồn 200 JailbreakBench + 200 Aegis2 + 200 WildGuardMix;
- Google NMT làm bản đầu, annotator song ngữ sửa tự nhiên và có thể tăng mức thô/harassment theo ngữ cảnh;
- có EN và VI tương ứng.

Vai trò: benchmark general/cross-lingual hữu ích, nhưng **không hoàn toàn độc lập** với Nemotron/Aegis. Phải báo overlap flag.

#### In-the-Wild Cultural

- prompt-only;
- native speaker viết cả English và ngôn ngữ bản địa;
- cân bằng safe/harmful;
- File bundled có 420 VI row, mỗi row có explicit English counterpart.

Vai trò: **external primary cho cultural Vietnamese prompt safety**.

#### Content Generation Cultural

- prompt-response;
- theo từng quốc gia, prompt/response English được professional translator dịch;
- file bundled chỉ có binary `Safe/Harmful` (`An toàn/Có hại` ở VI);
- VI có 110 prompt-response row, mỗi row chứa explicit English counterpart.

Vai trò: external primary cho cultural prompt và response safety. Archive hiện tại
chỉ có binary labels; không thêm một lớp `Sensitive` không tồn tại trong nguồn.

### 11.4. Nhãn và metric

Không áp dụng policy `Sensitive` cho archive này vì audit không tìm thấy nhãn
Sensitive trong bất kỳ file bundled nào. Nếu release khác xuất hiện sau này,
phải version dataset và định nghĩa mapping riêng, không thay đổi ngầm benchmark
đã chạy.

Paper dùng AUPRC làm metric chính. Dự án báo:

```text
AUPRC harmful
macro F1
harmful recall
safe FPR
Brier/ECE
EN-VI paired decision consistency
EN-VI probability gap
bootstrap 95% CI theo pair_uid
```

### 11.5. Chống contamination

Trước khi score:

```text
exact normalized overlap
near-duplicate MinHash/embedding candidates
source subset flag: Aegis | JailbreakBench | WildGuard | ITW | CG
```

Báo hai view:

- SEA full theo paper;
- SEA decontaminated đối với General.

ITW và CG không được dùng để chỉnh threshold dù rất nhỏ và hấp dẫn để tuning.

### 11.6. Artifact đã tạo

```text
reports/sea_safeguard_bench_audit.json
reports/sea_safeguard_bench_audit.md
data/benchmarks/sea_safeguard/official_en_vi.jsonl
data/benchmarks/sea_safeguard/paired_en_vi.jsonl
data/benchmarks/sea_safeguard/summary.json
```

`official_en_vi.jsonl` chỉ chứa task đã đăng ký trong SEA-HELM cho EN/VI.
`paired_en_vi.jsonl` bổ sung English counterpart nhúng để đo cross-lingual;
field `official_registered=false` ngăn nhầm counterpart suy ra với official EN
task.

---

## 12. Metric và báo cáo bắt buộc

### 12.1. Binary safety

```text
AUPRC
AUROC (secondary)
macro F1
unsafe precision/recall/F1
safe FPR
unsafe FNR
confusion matrix
```

### 12.2. Paired EN–VI

```text
same_decision_rate
mean_abs_probability_gap
EN safe -> VI unsafe
EN unsafe -> VI safe
EV gain on VI
EN retention after bilingual train
```

Bootstrap theo `record_uid`/`pair_uid`, không coi EN và VI là hai observation độc lập.

### 12.3. N23 multi-label

```text
micro/macro AUPRC
micro/macro F1
per-label precision/recall/F1/AUPRC
sample F1
Hamming loss
exact match (secondary only)
schema subset consistency
```

### 12.4. Length robustness

Cho từng bucket:

```text
sample count
class ratio
truncation rate
tokens retained ratio
AUPRC
macro F1
unsafe recall
safe FPR
```

### 12.5. Hiệu năng

```text
parameters
trainable parameters
peak VRAM
training tokens/sec
inference samples/sec
p50/p95 latency
schema token overhead
text token budget
```

---

## 13. Acceptance gates để tránh chạy tốn công vô ích

### Gate 0 — data

- deterministic manifest;
- no raw mutation;
- correct P/R/PR denominators;
- paired EN/VI hashes;
- official và clean view;
- token audit hoàn tất.

### Gate 1 — smoke

- 32 train / 16 valid;
- loss giảm;
- save/load prediction giống nhau trong tolerance;
- missing response không đóng góp loss;
- 4 GB không OOM ở profile đã chọn.

### Gate 2 — prompt baseline

- A01–A04 hoàn tất tối thiểu 3 seed nếu chi phí cho phép;
- threshold chỉ chọn trên valid;
- không có mismatch record giữa EN/VI;
- báo CI và length buckets.

### Gate 3 — schema

- unit test positive label không bị dropout;
- unit test task removal vẫn còn ít nhất một task;
- full schema và partial schema đều chạy;
- log schema token overhead;
- B01/B02 dùng cùng backbone, training budget và text budget.

### Gate 4 — external

- SEA official source + license + hash;
- overlap audit;
- threshold frozen;
- report General/ITW/CG riêng;
- Sensitive protocol đúng paper.

---

## 14. Thứ tự triển khai khuyến nghị

```text
1. Tạo pair/task manifest đầy đủ P/R/PR, giữ generic + jailbreaking
2. Token audit mmBERT và GLi tokenizer
3. Majority + mmBERT frozen-head smoke
4. A01..A04 prompt-only common-capacity
5. A05 EV-full nếu A04 có lợi
6. Released GLiGuard zero-shot A06
7. B01/B02 same-backbone architecture comparison
8. B03 GLi-family fine-tune
9. C01..C06 response-only vs prompt+response
10. Audit và train D01..D03 N23
11. Tải/normalize SEA official release, contamination audit
12. Freeze model/threshold và chạy SEA EN–VI cuối
13. Mở Phase J strategy/long-context riêng sau khi năm run binary hoàn tất
```

Lý do: vẫn train binary safety trên jailbreak ngay từ đầu, nhưng chưa nên chi
nhiều thời gian gắn nhãn strategy hoặc thiết kế chunk dài trước khi baseline
bilingual hoàn tất.

---

## 15. Kết luận thiết kế

Thiết kế này trả lời riêng bốn câu hỏi thay vì trộn chúng:

1. **Dữ liệu Việt có giúp không?** — `E` vs `EV-matched` trên cùng mmBERT fixed head.
2. **Safety adaptation của encoder có giúp không?** — `M-probe` vs `M-adapt`.
3. **Schema-conditioned có tốt hơn fixed head không?** — mmBERT fixed vs mmBERT schema trên cùng backbone và common-capacity.
4. **Model nào tốt hơn khi triển khai thật?** — mmBERT native 1K/2K vs GLi-family native 512, có latency/VRAM/length buckets.

SEA-SafeguardBench EN–VI sau đó kiểm tra liệu kết luận có tồn tại trên dữ liệu
người bản địa viết/hiệu đính và nội dung văn hóa Việt Nam hay không. Trong
Nemotron, generic/jailbreaking được báo thành các slice riêng nhưng đều tham gia
train nếu vừa giới hạn model; common-table GLi/mmBERT dùng đúng cùng eligible ID.
