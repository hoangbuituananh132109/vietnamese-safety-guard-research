# Training and evaluation reproduction map

This document identifies the code paths that produced the reported results. It
also prevents a historical data-contract mistake from being mistaken for the
final experiment.

## Which experiment is final?

The early local matrix included response-only (`R`) examples and was useful for
diagnosis. It is retained as legacy evidence, but it is not the clean final
training contract.

The final rerun is **P/PR-only**:

- `P`: prompt only;
- `PR`: prompt plus response;
- `R`: response only, explicitly excluded from training, validation, and test.

This is encoded, rather than merely described, in
`configs/phase0_no_r_experiments.json` and
`configs/e6_no_r_binary_ablation.json` through `allowed_views: [P, PR]` and
`forbidden_views: [R]`. Decoder training records the same invariant in
`scripts/run_no_r_decoder_pipeline.py`.

Do not describe the no-R results as an `only-response` experiment. Do not mix
legacy full-R checkpoints with final `*NR-*` checkpoints in a matched table.

## Execution architecture

The Vast runs were driven by Python CLI programs under Supervisor. The notebook
`notebooks/phase0_vast_runner.ipynb` is an exploratory convenience and is not
the source of truth for the completed runs.

| Layer | Entrypoint | Role |
|---|---|---|
| Locked contract | `configs/phase0_no_r_experiments.json` | E1NR-E5NR and E7NR recipes and manifests |
| E6 ablation | `configs/e6_no_r_binary_ablation.json` | Binary-only counterpart of E5NR |
| Vast encoder coordinator | `scripts/vast/run_no_r_phase0_pipeline.py` | Train, evaluate, checkpoint, package, and hand off |
| GLiGuard trainer | `scripts/train_gliguard_experiment.py` | E1NR schema-conditioned encoder training |
| Fixed mmBERT trainer | `scripts/train_mmbert_fixed_experiment.py` | E2NR-E6NR fixed-head training |
| Schema mmBERT trainer | `scripts/train_mmbert_schema_experiment.py` | E7NR dynamic-schema training |
| Encoder evaluator | `scripts/evaluate_experiment_matrix.py` | Locked evaluation suites and matrix output |
| Decoder coordinator | `scripts/run_no_r_decoder_pipeline.py` | Q2 and D3 stress test, train, evaluate, and preserve |
| Generic decoder LoRA | `scripts/train_decoder_guard_lora.py` | Q2 Qwen and decoder pilot LoRA training |
| Nemotron QLoRA utility | `scripts/train_nemotron_guard_qlora.py` | Separate quantized/pilot route, not the final full-precision D3 coordinator |
| Qwen evaluator | `scripts/evaluate_qwen3guard_gen_vllm.py` | P/PR-only Qwen evaluation |
| Nemotron evaluator | `scripts/evaluate_nemotron_decoder_guard_vllm.py` | Strict JSON binary and N23 evaluation |

## Reproduce the final encoder matrix

After obtaining the upstream models and constructing the authorized local
manifests, run from the repository root on the Vast image:

```bash
/venv/main/bin/python scripts/vast/run_no_r_phase0_pipeline.py \
  --root /workspace/safety-dataset \
  --config configs/phase0_no_r_experiments.json \
  --report-root reports/no_r_phase0 \
  --archive-stem PHASE0_NO_R_REPRODUCE \
  --no-auto-stop
```

Run E6 with the same coordinator and its separate config/report root:

```bash
/venv/main/bin/python scripts/vast/run_no_r_phase0_pipeline.py \
  --root /workspace/safety-dataset \
  --config configs/e6_no_r_binary_ablation.json \
  --report-root reports/e6_no_r_binary_ablation \
  --archive-stem E6_NO_R_REPRODUCE \
  --no-auto-stop
```

The coordinator validates completion markers, finite optimizer updates,
verified encoder gradients, expected evaluation jobs, and referenced metrics
before packaging the result.

## Reproduce Q2 and D3

`scripts/run_no_r_decoder_pipeline.py` is a fixed `/workspace/safety-dataset`
coordinator for the original dual-RTX-4080S run. Its recorded contract is:

- P/PR only;
- Q2: 101,274 EN+VI training rows;
- D3: 50,637 Vietnamese training rows;
- one epoch, 2,048 context, BF16, no quantization;
- effective batch 32;
- LoRA rank 8, alpha 32, dropout 0.05 on `q_proj` and `v_proj`.

The immutable D3 handoff report is
`reports/research_archive/D3_NEMOTRON_NO_R_FINAL_ANALYSIS_20260724.md`.
It records 1,583/1,583 optimizer steps and 11,736/11,736 matched P/PR evaluation
examples. Q2 is the strongest balanced binary result in that matched decoder
comparison; D3 is retained because its P and PR N23 behavior differs materially.

## New Luna/Sol corpus

The 45,416-record Luna/Sol corpus is a second translation condition only. No
reported model above was trained on it. Reproduce its construction and audit
with:

```powershell
python tools/build_luna_sol_training_dataset.py
python tools/audit_luna_sol_training_readiness.py
pytest -q tests/test_luna_overnight_runner.py
```

Training it must follow the frozen matched protocol in
`docs/LUNA_SOL_TRAINING_EVALUATION_PROTOCOL.md`: same UID/split manifests,
seeds, P/PR materialization, model configs, thresholds, and SEA evaluation as
the Gemini condition. Until then, only translation-pipeline and data-readiness
claims are supported.

## Evidence hierarchy

1. Locked JSON experiment contract.
2. Python coordinator and trainer/evaluator command recorded in state/log data.
3. Completion marker and immutable artifact inventory/checksum.
4. Per-example predictions kept privately for paired tests.
5. Sanitized aggregate report committed publicly.

If filenames and prose disagree, the immutable contract, UID set, and recorded
command take precedence. Any future result should state whether it uses legacy
full-R or final P/PR-only data in its table title.
