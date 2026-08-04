# E7 dynamic-schema analysis

Run: `E7-M-SCHEMA-EV-8K`

Status: the run is technically complete, but the current dynamic-schema recipe
does not establish a competitive GLi-style mmBERT guard.

## Execution integrity

- 140,136 training instances, identical to E5.
- 8,760 / 8,760 optimizer steps completed over two epochs.
- No non-finite optimizer update.
- Encoder/LoRA gradients were verified.
- Peak allocated VRAM: 9,828.5 MiB.
- Trainable parameters: 872,065, versus 584,089 for E5.
- Runtime: 8,480.7 seconds, versus 7,538.5 seconds for E5.

The failure is therefore a model/optimization result, not an interrupted run,
OOM, NaN, missing gradient, or incomplete evaluation.

## Same-subset binary comparison

The `e1_e2_primary_native` suite contains identical GLi-compatible rows.

| Run | Nemotron test | EN | VI | SEA | SEA EN | SEA VI |
|---|---:|---:|---:|---:|---:|---:|
| B0 GLiGuard zero-shot | 66.80% | 78.76% | 54.84% | 68.88% | 79.91% | 57.85% |
| E1 trained GLiGuard | 68.70% | 79.16% | 58.24% | 65.00% | 78.58% | 51.43% |
| E2 mmBERT fixed head | 77.18% | 78.15% | 76.22% | 72.98% | 74.00% | 71.96% |
| E5 mmBERT fixed head, full EN+VI | 77.84% | 79.21% | 76.47% | 73.84% | 74.63% | 73.05% |
| **E7 mmBERT dynamic schema** | **56.24%** | **56.39%** | **56.09%** | **53.81%** | **56.71%** | **50.92%** |

E7 does not merely lose to a trained GLiGuard. It loses to the zero-shot GLi
checkpoint by 10.56 percentage points on Nemotron test and 15.07 points on SEA.
Its Vietnamese Nemotron score is slightly above zero-shot GLi, but almost the
entire English safety capability is missing.

The fixed-head results prove that the mmBERT encoder and Vietnamese data are not
the primary limitation. The regression is introduced by the current schema
conversion and its joint loss.

## Full 8K binary comparison

| Run | Nemotron test | EN | VI | SEA | SEA EN | SEA VI |
|---|---:|---:|---:|---:|---:|---:|
| E3 EN-only fixed head | 75.58% | 77.98% | 73.17% | 71.82% | 73.80% | 69.84% |
| E4 matched EN/VI fixed head | 76.13% | 77.46% | 74.80% | 72.58% | 74.08% | 71.09% |
| E5 full EN+VI fixed head | **78.58%** | **79.97%** | **77.20%** | **74.13%** | **75.22%** | **73.04%** |
| **E7 dynamic schema** | **56.13%** | **56.82%** | **55.44%** | **54.54%** | **57.61%** | **51.47%** |

E7's test AUROC is 0.605 and its SEA AUROC is 0.564. This is weak ranking, not
only a bad 0.5 decision threshold.

On the full Nemotron test set the class-level counts are:

| Language | Correct safe | Safe called unsafe | Unsafe called safe | Correct unsafe | Accuracy |
|---|---:|---:|---:|---:|---:|
| EN | 1,917 | 617 | 1,689 | 1,118 | 56.82% |
| VI | 1,729 | 805 | 1,575 | 1,232 | 55.44% |

E7 is biased toward predicting safe on Nemotron: safe recall is 71.94%, while
unsafe recall is only 41.86%. SEA is closer to balanced but nearly random:
54.54% overall, 57.61% EN, and 51.47% VI.

Choosing the best global unsafe threshold on Nemotron validation gives 0.45.
Applying it without looking at test labels changes full Nemotron test accuracy
from 56.13% to only 57.61%, while SEA falls to 53.02%. Threshold calibration
cannot recover the gap.

## Label-order contract

The model was trained with `safe/unsafe` reversed in 50% of training batches.
Nevertheless, changing only label order at inference changes the semantic
prediction on:

- 22.00% of full Nemotron test examples;
- 33.53% of full SEA examples.

Canonical-versus-reversed agreement is only 78.00% on Nemotron test and 66.47%
on SEA. Aggregate accuracy happens to be similar under the two orders, but the
individual decisions are not invariant. E7 has not reliably learned that label
meaning, rather than position, determines the result.

## N23 collapse

At threshold 0.5 E7 predicts no positive category:

- test gold positives: 6,966;
- test predicted positives: 0;
- micro-F1: 0;
- macro-F1: 0.

This is not a serialization bug. On full Nemotron test, category probabilities
have median 0.053, p95 0.133, p99 0.223, and maximum 0.456. Every value is truly
below 0.5.

A validation-selected global threshold of 0.12 produces test micro-F1 0.218 and
macro-F1 0.074. E5 at the normal 0.5 threshold reaches micro-F1 0.393 and
macro-F1 0.204. E7 contains some category ranking signal, but it is both poorly
calibrated and substantially weaker.

The dynamic sampler presents roughly 15 category labels on a supervised row but
usually only about one is positive. BCE therefore sees approximately 7–8%
positives. This creates a plausible low-probability shortcut for the current
undertrained recipe. It is not evidence that the shared scalar MLP is inherently
wrong: the GLiGuard paper uses the same shared two-layer scalar scorer and reports
successful multi-label results. E5 additionally benefits from 23 label-specific
logits and biases, whereas E7 must infer each label's prior from its contextualized
schema embedding. The observed collapse is therefore attributed to the complete
training recipe and supervision geometry, not to the shared MLP alone.

## Training dynamics

E5:

- mean total loss, first 20 logged points: 1.930;
- mean total loss, last 20: 0.500;
- binary loss: 1.208 to 0.369;
- category loss: 0.732 to 0.132.

E7:

- mean total loss, first 20 logged points: 0.938;
- mean total loss, last 20: 0.804;
- binary loss: 0.703 to 0.610;
- category loss: 0.336 to 0.271.

E7 learns slowly and plateaus high. More parameters did not compensate for a
harder representation problem.

## Corrected GLiGuard provenance and why E7 is weaker

The earlier version of this report claimed that GLiGuard's encoder had already
been pretrained on schema-conditioned tasks. That claim is not supported by the
paper and has been withdrawn.

The GLiGuard paper explicitly says that the released experiment initializes the
encoder from `microsoft/deberta-v3-base`, adds `[P]`, `[L]`, and `[SEP]` to the
tokenizer, resizes the embedding table, and trains for 20 epochs. It extracts each
contextualized `[L]` hidden state and applies one shared two-layer MLP
`Linear(d, 2d) -> ReLU -> Linear(2d, 1)`. Single-label tasks use softmax CE and
multi-label tasks use independent sigmoid/BCE. This is the same core scoring idea
implemented by E7; taking the `[L]` anchor is not an implementation mistake.

There is a provenance inconsistency that must be kept explicit. The paper names
raw `microsoft/deberta-v3-base` as the initialization checkpoint, while the current
Hugging Face model-card metadata lists `fastino/gliner2-base-v1` as the base model.
Both public configs name DeBERTa-v3-base as their underlying transformer. The
repository does not publish the GLiGuard training script needed to resolve whether
the released weights were initialized from raw DeBERTa or from the GLiNER2 task
checkpoint. Consequently this report does not use prior schema pretraining as an
explanation for E7.

The evidence instead points to a large recipe mismatch:

1. GLiGuard trains the encoder for 20 epochs; E7 trains rank-4 LoRA for 2 epochs.
2. GLiGuard uses encoder/head learning rates `2e-5`/`5e-5`, effective batch 8,
   label shuffling every step, label dropout 0.15, and task removal 0.05. E7 uses
   effective batch 32, reverses only the binary order, never drops a positive N23
   label, and has no whole-task removal.
3. The released GLiGuard config enables entropy regularization with weight 0.1;
   E7 has none.
4. GLiGuard gives harm and jailbreak tasks an explicit `Benign` label and falls
   back to the highest-probability label when no multi-label probability crosses
   0.5. E7 has neither mechanism and reports an empty N23 set at 0.5.
5. GLiGuard jointly learns prompt safety, response safety, refusal, harm category,
   and jailbreak strategy supervision from WildGuardTrain plus GPT-4.1 weak labels.
   Its published binary benchmark verdict also applies hard auxiliary-task decision
   rules. E7 learns Nemotron binary plus N23 only and evaluates its binary task
   directly, so the headline scores are not architecturally identical quantities.
6. GLiGuard uses DeBERTa-v3-base at a 512-token limit; E7 uses multilingual mmBERT,
   supports up to 8K tokens, and solves a substantially different bilingual task.

E7 therefore tests whether a very cheap two-epoch rank-4 LoRA adaptation is enough;
it is not a faithful reproduction of the GLiGuard optimization recipe. The fixed
head remains easier because it only learns stable output dimensions over a pooled
representation, whereas E7 must simultaneously learn schema routing, label-order
invariance, binary safety, and imbalanced open-label N23 scoring.

## Concrete next ablations

The next run should first reproduce the published recipe closely enough to
separate an architecture result from an undertraining result:

1. **E7a binary-only diagnostic.** Remove N23 and train only the two shuffled
   binary anchors. This isolates schema learning from negative-heavy multi-label
   interference.
2. **E7b faithful-schema recipe.** Use full encoder fine-tuning if memory permits
   (otherwise unfreeze the top layers or use a substantially larger LoRA rank),
   train longer, use the paper's two learning rates and effective batch 8, shuffle
   labels every step, add 0.15 label dropout, 0.05 task removal, and the released
   entropy weight 0.1.
3. **Faithful N23 semantics.** Add an explicit `Benign` candidate, implement the
   paper's highest-probability fallback, and compare both raw N23 metrics and the
   composed safety verdict. Keep direct binary scores separate so decision rules
   do not hide a weak binary classifier.
4. **Permutation audit.** Track canonical/reversed agreement as a first-class
   validation metric. A consistency loss is a proposed extension, not part of the
   published GLiGuard recipe, and should be tested only after the reproduction.
5. **Balanced-N23 ablation.** Positive weighting, focal/asymmetric BCE, or fewer
   negatives per positive may help this dataset, but each is an explicit departure
   from the paper and must be reported as such.
6. **Task-conditioned scorer** such as `MLP([h_L, h_P, h_L*h_P])` and generic
   schema pretraining are research extensions. They should not be presented as
   what GLiGuard did, nor used to diagnose E7 before the faithful ablation.

The correct conclusion is not that schema-conditioned mmBERT is impossible. It
is that a direct two-epoch rank-4 LoRA transfer of the GLi interface onto a
general mmBERT encoder is insufficient, and combining binary plus N23 in the
first full experiment obscured the binary schema-learning signal. No conclusion
about the value of prior GLiNER2 schema pretraining is supported by E7 or by the
paper's stated initialization.

## Primary-source audit trail

- Paper source: `research/papers/gliguard_2605.07982/src/colm2026_conference.tex`,
  especially lines 271-417, 466-500, and 795-930.
- Official GLiGuard repository: `research/sources/GLiGuard`.
- Official GLiNER2 repository: `research/sources/GLiNER2`.
- Saved Hugging Face API/config snapshots:
  `research/sources/huggingface/gliguard_*.json` and
  `research/sources/huggingface/gliner2_*.json`.
