# Luna pilot slice analysis and downstream experiment

## What the 200-record pilot contains

- 200 selected records; Luna completed 172 and 28 high-tail records remain missing.
- Prompt safety label: 110 unsafe, 90 safe.
- Response labels: 62 unsafe, 31 safe, 104 null and 3 empty/unset.
- Source tag: 93 jailbreaking and 107 generic.
- Length: 63 normal, 23 near-tail, 65 tail and 49 high-tail.

The pilot is intentionally enriched for hard cases and must not be used to estimate population prevalence without reweighting.

## Leetspeak prevalence

The refined detector excludes URLs, e-mails, hashes, H1-H6 headings and ISO-like date/version tokens. It requires multiple mixed letter-digit tokens whose decoded forms resemble common English prose.

- Pilot: 42/200 (21.0%).
- Full 45,416-record final population: 155/45,416 (0.34%).
- Full population unsafe: 134/26,881 (0.50%).
- Full population safe: 21/18,535 (0.11%).

Pilot leetspeak by length:

| Length | Leet | Total | Share |
|---|---:|---:|---:|
| normal | 23 | 63 | 36.5% |
| near-tail | 4 | 23 | 17.4% |
| tail | 2 | 65 | 3.1% |
| high-tail | 13 | 49 | 26.5% |

Full-population leetspeak by length:

| Length | Leet | Total | Share |
|---|---:|---:|---:|
| normal | 106 | 40,677 | 0.26% |
| near-tail | 16 | 2,368 | 0.68% |
| tail | 9 | 1,920 | 0.47% |
| high-tail | 24 | 446 | 5.38% |
| oversized | 0 | 5 | 0% |

Leetspeak is therefore rare globally but concentrated in unsafe and high-tail material. A dedicated routing queue is more efficient than forcing every batch through aggressive leetspeak rules.

## What Luna handled well in the completed pilot

The labels below combine automatic routing with the 22 manually audited validator disagreements. `automatic_clean` is not equivalent to a full human semantic pass.

- 147/172 outputs were automatically clean.
- 8/172 additional outputs had validator flags manually confirmed as false positives.
- 6/172 had confirmed Luna errors.
- 10/172 still need manual review.
- 1/172 was semantically better than Gemini on English leetspeak but failed to restore Vietnamese obfuscation.

By length:

| Length | Completed | Auto-clean | Confirmed validator FP | Confirmed Luna error | Review | Partial |
|---|---:|---:|---:|---:|---:|---:|
| normal | 63/63 | 50 | 4 | 2 | 6 | 1 |
| near-tail | 23/23 | 22 | 1 | 0 | 0 | 0 |
| tail | 65/65 | 60 | 2 | 2 | 1 | 0 |
| high-tail | 21/49 | 15 | 1 | 2 | 3 | 0 |

The strongest result is near-tail and tail prose: all 23 near-tail records completed with no confirmed model error, and 62/65 tail records were either clean or confirmed evaluator false positives. The weakest slice is high-tail: only 21/49 completed, with two confirmed errors and three unresolved reviews. Because 28 missing records are all high-tail, it is invalid to report a single 172-record success rate as if it represented all lengths equally.

## Is Luna weak at leetspeak?

Not broadly, based on current evidence:

- 42 pilot records contain clear prose leetspeak.
- Luna completed 32 of them; 10 missing leet records are high-tail failures/non-completions.
- Among the 32 completed: 27 automatic-clean, 1 confirmed evaluator false positive, 2 confirmed Luna errors, 1 unresolved review, and 1 partial advantage over Gemini but imperfect re-obfuscation.
- Thus 28/32 completed leet records have no confirmed Luna problem so far; this is not yet a blinded semantic pass rate.

The two confirmed leet errors are different:

1. A normal unsafe hate/unauthorized-advice record was decoded into plain Vietnamese but lost required obfuscation.
2. A high-tail unsafe guns record was summarized from 4,114 source characters to 244 target characters.

The second is primarily a completeness/length failure, not evidence that Luna cannot understand leetspeak. The remaining concern is exact style preservation: Luna may decode the meaning correctly but return plain Vietnamese rather than lightly obfuscated Vietnamese.

## Safety slices

| Prompt label | Selected | Completed | Auto-clean | Confirmed FP | Confirmed error | Review | Partial | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| safe | 90 | 83 | 73 | 4 | 1 | 5 | 0 | 7 |
| unsafe | 110 | 89 | 74 | 4 | 5 | 5 | 1 | 21 |

Unsafe material has more missing high-tail records and five of six confirmed Luna errors. This means the current pilot cannot yet claim equal translation reliability for safe and unsafe slices.

The most common overlapping hazard categories in the pilot are: Needs Caution (34), Immoral/Unethical (26), Criminal Planning/Confessions (23), Unauthorized Advice (21), Illegal Activity (20), Profanity (19), PII/Privacy (18), Political/Misinformation/Conspiracy (17), Controlled/Regulated Substances (14), Hate/Identity Hate (14), and Violence (14). Category counts overlap because a record may have multiple labels.

Leetspeak is especially concentrated in Criminal Planning (10), Illegal Activity (10), Immoral/Unethical (10), Hate/Identity Hate (9), Controlled Substances (8), Guns (6), Political/Misinformation (6), PII/Privacy (5), Harassment (5), and Manipulation (5).

The six confirmed Luna errors cover:

- normal + unsafe + leetspeak: Unauthorized Advice / Hate;
- high-tail + unsafe: PII/Privacy / Other truncation;
- high-tail + unsafe + leetspeak: Guns / Needs Caution truncation;
- tail + safe: JSON schema-key change;
- tail + unsafe: Suicide/Self-Harm JSON-key change;
- normal + unsafe: Profanity wordplay left untranslated.

This pattern suggests routing by content shape in addition to length.

## Recommended orthogonal routing labels

Do not replace length buckets; add independent content-shape labels:

- `plain_prose`
- `prose_leetspeak`
- `structured_json`
- `executable_code_or_sql`
- `template_placeholders`
- `opaque_identifiers_or_credentials`
- `repetitive_or_generated_noise`
- `multilingual_source`
- `source_already_truncated`
- `high_profanity_or_slur`

Suggested routing:

- normal plain prose: Luna batch 30;
- near-tail plain prose: Luna batch 20;
- leetspeak: dedicated batch 10-15 with mandatory decode/translate/re-obfuscate validation;
- JSON/schema: batch 10-15 with key-preservation validation;
- code/SQL/opaque data: dedicated prompt that distinguishes executable tokens from human-readable literals;
- tail prose: batch 8, one Luna attempt before fallback if blocking failure;
- high-tail/repetitive: one record per turn, then chunk or web/Sol fallback;
- oversized: chunk before model invocation.

Because the full population has only about 155 detected leetspeak records, the leet queue can be exhaustively audited without materially affecting total cost.

## How to establish a translation-pipeline contribution

Freeze two dataset versions with identical source UIDs and splits:

- Version G: existing Gemini/manual pipeline.
- Version L: Luna pipeline with frozen prompt, routing, validator and fallback policy.

Train the same downstream model twice using identical architecture, tokenizer, hyperparameters, number of steps, seeds and compute:

- Model G trained on Version G train.
- Model L trained on Version L train.

Use at least three seeds if compute permits. Evaluate a 2×2 matrix:

| Trained model | Gemini test | Luna test |
|---|---:|---:|
| Model G | G→G | G→L |
| Model L | L→G | L→L |

Then evaluate both models on the same untouched external Vietnamese SEA-HELM/SEA Bench tasks and, if available, the original English test.

Interpretation:

- Model L improves on both translated tests and SEA Bench: evidence the Luna pipeline improved training data.
- Both models score higher only on Luna test: Luna test may simply be easier; not evidence of better training.
- Model L improves only on Luna test but not Gemini test/SEA Bench: likely translation-style alignment.
- Translation blind review improves but downstream scores do not: still a translation-quality contribution, but not a downstream modeling improvement.
- SEA Bench improves while in-domain tests stay similar: evidence of better cross-benchmark generalization.

Report translation quality and downstream utility as separate claims. The pipeline contribution should include the prompt, routing rules, partial-save behavior, retry accounting, validator calibration, failure taxonomy, fallback policy and cost—not merely the Luna model name.
