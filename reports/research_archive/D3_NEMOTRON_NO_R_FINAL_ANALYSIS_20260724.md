# D3 Nemotron no-R final run and matched evaluation

Generated: 2026-07-24

## Completion and preservation

- Pipeline terminal status: completed.
- Nemotron training: 1,583 / 1,583 optimizer steps.
- Evaluation: 11,736 / 11,736 no-R P/PR examples.
- Local archive:
  `reports/vast_download/d3_nemotron_no_r_4080s_20260724/D3_NEMOTRON_NO_R_4080S_20260724.tar.zst`
- Archive bytes: 453,504,850.
- Archive SHA-256:
  `00257a83c7521ba055c88e8176688e705d1d9c2feb3d9c08f2a98b6fc52075da`.
- Archive transfer verified on the first attempt.
- Verified ACK uploaded to the Vast instance.
- SSH became unreachable after the ACK, consistent with the requested Vast stop.
- The instance was not destroyed or recycled.

The archive extracted successfully. All 246 immutable files in the internal
inventory matched. The only changing file was the intentionally live guardian
log, which appended archive-progress events after its per-file hash had been
computed; the archive-level SHA-256 protects that file.

## D3 training contract

- Base: `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3`.
- Revision: `8fdc246ba3d56db9c469d534233b9f582d3afafa`.
- Training rows: 50,637 Vietnamese no-R P/PR instances.
- P safe: 14,480.
- P unsafe: 16,726.
- PR safe: 13,390.
- PR unsafe: 6,041.
- Combined safe: 27,870 (55.04%).
- Combined unsafe: 22,767 (44.96%).
- Context: 2,048.
- Truncated train examples: 171 / 50,637 (0.338%).
- BF16, no quantization.
- Microbatch 2, effective batch 32, accumulation 16.
- One epoch.
- Learning rate: 1e-5.
- LoRA rank 8, alpha 32, dropout 0.05.
- LoRA targets: q_proj and v_proj.
- Trainable parameters: 3,407,872 / 8,033,669,120 (0.04242%).
- Final reported loss: 0.05614.
- Actual serialized train tokens: 24,352,850.
- Padding overhead: 0.00284%.
- Peak train VRAM: 22,501 MB.
- Resumed-process train time: 14,109 seconds.

## D3 evaluation runtime

- Strict JSON parsed: 11,733 / 11,736 (99.974%).
- Parse failures: 3.
- Unknown category outputs: 3.
- Evaluation input truncation: 0.
- vLLM inference time excluding load: 571.25 seconds.
- Load plus inference: 641.28 seconds.
- Throughput: 20.54 examples/second.
- Peak evaluation VRAM: 31,787 MB.

## Matched binary comparison

All five systems are evaluated on the exact same 11,736 IDs with zero target
mismatches. Q1 uses the conservative `controversial -> unsafe` mapping.

| System | Accuracy | Macro-F1 | Safe recall | Unsafe recall |
|---|---:|---:|---:|---:|
| Q1 — Qwen zero-shot | 85.15% | 84.94% | 75.85% | **93.88%** |
| **Q2 — Qwen EN–VI LoRA** | **87.83%** | **87.82%** | 87.79% | 87.87% |
| D1 — Nemotron zero-shot | 86.95% | 86.95% | 90.52% | 83.60% |
| D2 — Nemotron pilot LoRA | 86.81% | 86.81% | 89.83% | 83.97% |
| D3 — Nemotron full VI no-R LoRA | 86.52% | 86.51% | **92.12%** | 81.26% |

D3 confusion matrix:

- Safe correct: 5,238.
- Safe falsely blocked: 448.
- Unsafe missed: 1,134.
- Unsafe detected: 4,916.

D3 became more safe-biased than D1 and D2. It reduced safe false alarms but
missed substantially more unsafe examples.

## Binary slices

| Slice | N | Q1 | **Q2** | D1 | D2 | D3 |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 11,736 | 85.15% | **87.83%** | 86.95% | 86.81% | 86.52% |
| English | 5,868 | 86.06% | **88.45%** | 87.61% | 87.41% | 87.00% |
| Vietnamese | 5,868 | 84.24% | **87.22%** | 86.30% | 86.21% | 86.04% |
| Nemotron test | 8,056 | 85.86% | **88.49%** | 88.34% | 88.26% | 87.88% |
| SEA paired | 3,680 | 83.59% | **86.39%** | 83.91% | 83.64% | 83.53% |
| Prompt-only | 7,690 | 84.92% | **87.13%** | 86.01% | 85.85% | 85.47% |
| Prompt + response | 4,046 | 85.59% | **89.17%** | 88.75% | 88.63% | 88.51% |

D3 Vietnamese binary accuracy is 0.26 point below D1 and 0.17 point below
D2. Training on Vietnamese alone did not improve the Vietnamese binary score.

## Paired correctness tests

The first model in each row is D3.

| Pair | D3 correct / other wrong | Other correct / D3 wrong | D3 net | Exact McNemar p | Decision agreement |
|---|---:|---:|---:|---:|---:|
| D3 vs Q1 | 1,017 | 856 | +161 | 0.000216 | 84.04% |
| D3 vs Q2 | 488 | 642 | -154 | 0.00000516 | 90.37% |
| D3 vs D1 | 156 | 207 | -51 | 0.00859 | 96.91% |
| D3 vs D2 | 177 | 211 | -34 | 0.09374 | 96.69% |

D3 is significantly better than Q1 and significantly worse than Q2 and D1.
Its binary difference from D2 is not statistically significant at 0.05.

## Topic slices

| Topic | Q1 | **Q2** | D1 | D2 | D3 |
|---|---:|---:|---:|---:|---:|
| Cultural content generation | 82.05% | **87.05%** | 82.05% | 84.55% | 84.32% |
| Cultural in-the-wild | **92.98%** | 88.45% | 82.74% | 80.71% | 81.43% |
| General | 80.58% | **85.54%** | 84.67% | 84.50% | 84.13% |
| Generic | 83.25% | **85.91%** | 85.65% | 85.28% | 85.26% |
| Jailbreaking | 91.35% | 93.92% | 94.00% | **94.50%** | 93.38% |

## N23 matched comparison

D1, D2, and D3 are compared on the exact same 5,768 Nemotron-test P/PR
examples with available N23 supervision.

| System | Exact match | Hamming error | Micro precision | Micro recall | Micro-F1 | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| D1 | **52.31%** | 4.017% | 64.43% | 52.45% | 57.83% | 44.69% |
| D2 | 51.73% | 3.978% | 64.91% | **52.74%** | **58.20%** | 47.49% |
| D3 | 50.85% | **3.920%** | **67.58%** | 48.74% | 56.63% | **48.51%** |

D3 has the best N23 macro-F1 and hamming error, but it underpredicts positive
labels:

- Gold positive labels: 6,966.
- D2 predicted positives: 5,660.
- D3 predicted positives: 5,024.

This raises precision and some rare-label macro performance while lowering
recall and micro-F1.

## N23 by language

| Language | System | Exact match | Hamming error | Micro-F1 | Macro-F1 |
|---|---|---:|---:|---:|---:|
| English | D1 | **52.74%** | 3.914% | 58.81% | 45.87% |
| English | D2 | 52.15% | 3.870% | **58.93%** | 47.42% |
| English | D3 | 51.80% | **3.770%** | 58.35% | **50.82%** |
| Vietnamese | D1 | **51.87%** | 4.120% | 56.86% | 43.48% |
| Vietnamese | D2 | 51.32% | 4.087% | **57.47%** | **47.71%** |
| Vietnamese | D3 | 49.90% | **4.069%** | 54.90% | 45.77% |

Despite Vietnamese-only fine-tuning, D3's largest macro-F1 gain is on English.
Against D2, D3 loses 1.94 Vietnamese macro-F1 points and 2.57 Vietnamese
micro-F1 points. This does not support a Vietnamese N23 improvement over the D2
pilot.

## N23 by evaluation view

| View | System | Exact match | Hamming error | Micro-F1 | Macro-F1 |
|---|---|---:|---:|---:|---:|
| P | D1 | **55.51%** | **3.447%** | **58.93%** | 42.15% |
| P | D2 | 53.98% | 3.551% | 56.29% | **43.80%** |
| P | D3 | 50.45% | 3.747% | 49.83% | 39.84% |
| PR | D1 | 48.48% | 4.699% | 56.82% | 44.34% |
| PR | D2 | 49.05% | 4.490% | 59.86% | 47.74% |
| PR | D3 | **51.33%** | **4.126%** | **62.20%** | **52.78%** |

D3's N23 behavior splits sharply by view:

- Prompt-only N23 degrades substantially.
- Prompt-plus-response N23 improves strongly and is the best of D1/D2/D3.

Therefore the headline D3 macro-F1 gain is not uniform. It is a PR interaction
gain combined with a P regression.

## Interpretation

1. D3 completed technically and is fully reproducible.
2. D3 does not beat Q2 for binary guarding.
3. D3 does not beat D1 or D2 on binary accuracy.
4. D3's binary regression comes from a stronger safe bias: safe recall rises to
   92.12%, while unsafe recall falls to 81.26%.
5. The training distribution is 55.04% safe, whereas the evaluation population
   is 51.55% unsafe. That mismatch is a plausible contributor to the safe bias.
6. Vietnamese-only training did not improve Vietnamese binary or Vietnamese N23
   relative to D2.
7. D3 improves N23 PR substantially, indicating that the full translated data is
   useful for fine-grained prompt-response interaction classification.
8. D3 degrades prompt-only N23, so P and PR should be balanced or trained with
   view-aware sampling/losses in the next Nemotron run.
9. A low binary loss does not imply a better decision boundary. D3's final loss
   is low while its safe/unsafe calibration is shifted.
10. The next inexpensive analysis should evaluate threshold/calibration options
    only if score/logit information is available. The current generative
    evaluator records hard labels, so retraining with class/view-balanced sampling
    is more defensible than post-hoc threshold claims.

## Recommended next experiment

Keep Q2 as the current balanced binary leader. For Nemotron:

- retain no-R and context 2,048;
- train EN+VI rather than VI-only, or use matched EN/VI semantic pairs;
- balance safe/unsafe within P and PR separately;
- balance P versus PR contribution;
- report two objectives separately:
  - binary safe/unsafe;
  - N23 taxonomy;
- consider loss weighting so binary unsafe recall is not sacrificed to the
  structured N23 objective;
- compare the resulting model against D1/D2/D3 on this immutable evaluation
  intersection.

