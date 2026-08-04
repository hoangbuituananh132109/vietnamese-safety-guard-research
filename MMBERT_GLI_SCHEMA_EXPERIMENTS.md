# mmBERT schema-conditioned theo GLiGuard — đặc tả E7

**Khóa ngày:** 21/07/2026  
**Implementation:** `guard_train/mmbert_schema.py`

## 1. Một text classifier, hai loại task

`P/R/PR` không đi vào schema. Chúng chỉ quyết định chuỗi text và gold binary trước khi
model chạy. E7 có thể chứa hai task:

```text
binary safety: single_label, safe/unsafe, softmax + CE
N23 categories: multi_label, 0..23 positives, sigmoid + BCE
```

N23 chỉ tồn tại khi `category_scope` là `prompt` hoặc `interaction`. Nếu scope
`unavailable`, task N23 bị bỏ khỏi sequence và loss; không coi là 23 negatives.

## 2. Serializer và `[L]`

```text
[P] text safety classification [L] safe [L] unsafe
[P] safety policy categories [L] Violence [L] Threat ...
[SEP] <text>
```

`[P]`, `[L]`, `[SEP]` là ba special token. Serializer lưu chính xác vị trí từng `[L]`.
Sau một forward bidirectional:

```text
H = mmBERT(schema + text)
h_j = H[position([L]_j)]
z_j = SharedMLP(h_j)       # d -> 2d -> 1, ReLU
```

MLP nhận **final contextual hidden state ở `[L]`**, không nhận static embedding của ký
hiệu `[L]`. Hai `[L]` có cùng token ID đầu vào nhưng final state khác nhau vì vị trí,
tên label phía sau, các task/label còn lại và text đều tham gia self-attention.

Không pool thủ công từ `[L]` đến marker kế tiếp. Tên label vẫn ảnh hưởng `[L]` thông qua
bidirectional attention.

## 3. Loss không phụ thuộc việc task nằm chung schema

Hai scalar binary được lấy theo thứ tự schema thực tế:

```text
p_binary = softmax([z_safe, z_unsafe])
loss_binary = CrossEntropy(z_binary, remapped_target)
```

Các scalar N23 độc lập:

```text
p_j = sigmoid(z_j)
loss_N23 = BCEWithLogits(z_present_labels, y_present_labels)
```

Tổng loss:

```text
loss = loss_binary + category_loss_weight * loss_N23
```

Nếu N23 task không có mặt, chỉ còn binary loss. Cùng một shared MLP không có nghĩa mọi
task phải dùng softmax.

## 4. Schema động được train thế nào

Binary luôn giữ cả `safe` và `unsafe`, nhưng hoán vị thứ tự; gold index được remap theo
thứ tự sau shuffle.

Với N23:

- mọi positive label luôn được giữ;
- negative label được subsample theo xác suất;
- một phần sample vẫn dùng full 23-label schema;
- thứ tự category được shuffle;
- row không có category supervision bỏ cả task N23.

Như vậy model quan sát số `[P]`/`[L]` biến đổi và học label composition. Không bao giờ
xóa true label rồi vẫn tính nó như negative.

Ba mức năng lực phải phân biệt:

1. Hoán vị label đã biết: được train và phải đánh giá.
2. Thêm/bớt task hoặc label đã biết: được train qua composition/subsampling.
3. Label hoàn toàn chưa học, alias hay typo: kiến trúc vẫn sinh logit nhưng semantics
   chưa được bảo đảm; cần held-out-label protocol riêng.

## 5. Trainable parameters

- LoRA `r=4`, `alpha=8`, dropout 0 trên all-linear của encoder;
- đúng ba embedding row mới qua `trainable_token_indices`;
- shared scalar MLP;
- toàn bộ embedding table và base weights còn lại frozen bởi PEFT.

Checkpoint lưu adapter, tokenizer có marker, shared MLP, resolved config và schema
contract.

## 6. Độ dài và truncation

Eligibility/truncation tính trên toàn sequence sau full schema thích hợp của row.
E7 giữ đủ train/valid/test. Nếu vượt 8K:

- P/R giữ head+tail;
- PR giữ prompt head và response head+tail;
- audit lưu ID, original/final token và strategy.

Preprocessing dùng full 23-label schema cho các row có category supervision. Vì vậy mọi
schema subset lúc train đều không dài hơn sequence đã kiểm tra.

## 7. Gate đánh giá

- binary metrics trên mọi instance;
- N23 metrics chỉ trên row có supervision;
- canonical và reversed binary order;
- full N23 schema và các schema subset;
- paired EN–VI;
- native và truncated slices;
- save/reload parity;
- prediction-level artifact gồm label order/schema labels.

Smoke một optimizer step đã chạy qua cả binary và N23 trên RTX 3050. Điểm smoke không
có giá trị so sánh chất lượng; mục đích chỉ là chứng minh tensor shape, loss, gradient,
save và inference chạy được.
