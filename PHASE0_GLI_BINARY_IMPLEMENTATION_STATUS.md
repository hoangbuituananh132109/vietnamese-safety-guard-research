# Trạng thái triển khai guard EN–VI

**Cập nhật:** 21/07/2026

## Đã qua

- Materialize đầy đủ: train 140.136, valid 8.764, test 10.682 instance.
- E1/E2 paired native: 115.608 train, 6.666 valid, 8.476 test; canonical và GLi
  payload có exact same IDs.
- E3/E4 có ngân sách train bằng nhau: 70.068/70.068.
- E5/E7 dùng đủ 140.136 train instance.
- SEA bundled paired: 1.840 cặp; native E1/E2 1.573 cặp; full shared-truncated giữ đủ
  1.840 cặp.
- GLiGuard single-label classification đã override về softmax/categorical CE.
- Fixed multi-task trainer đã smoke: binary CE + N23 BCE masked.
- Dynamic-schema trainer đã smoke: binary permutation, optional N23 task, negative label
  subset, shared `[L]` MLP.
- Standalone checkpoint reload/inference đã chạy cho fixed và schema, gồm reversed order,
  prediction JSONL và calibration metrics.
- E1 full launcher đã chạy one-step bằng đúng manifest/sampling/effective batch và lưu
  adapter `final`.
- `scripts/preflight_phase0_experiments.py` hiện pass, zero failure.

## Gate chưa qua

- Chưa profile context 8K trên GPU thuê 32 GB.
- Chưa chạy mini end-to-end 20–100 step trên chính máy thuê.
- GLiGuard launcher chỉ warm-resume adapter do GLiNER2 pinned không lưu/khôi phục optimizer,
  scheduler và global step; E1 cần tmux + disk ổn định.
- Chưa có kết quả nghiên cứu full-run. Mọi F1 từ smoke một step chỉ là sanity check.

## Entry points

```text
scripts/preflight_phase0_experiments.py
scripts/train_gliguard_experiment.py
scripts/train_mmbert_fixed_experiment.py
scripts/train_mmbert_schema_experiment.py
scripts/evaluate_scalable_mmbert_checkpoint.py
scripts/evaluate_gliguard_checkpoint.py
scripts/evaluate_experiment_matrix.py
scripts/benchmark_mmbert_lora_context.py
notebooks/phase0_vast_runner.ipynb
```

## Quyết định hiện tại

Đủ điều kiện thuê GPU để **profile và mini smoke**. Chưa tự động chạy toàn bộ ma trận
trước khi profile 32 GB xác nhận microbatch ở 512/1K/2K/4K/8K.
