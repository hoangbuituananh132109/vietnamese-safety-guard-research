# Luna/Sol translation pipeline: research handoff

Status date: 2026-08-09

## Technical summary

The project now contains a complete second English-to-Vietnamese translation of the 45,416-record Nemotron safety corpus. GPT-5.6 Luna produced 45,060 selected records and GPT-5.6 Sol Web produced the 356 hard or infrastructure-limited records. Coverage is 45,416/45,416 with no missing, extra, duplicate, or structurally invalid UID in the final merged corpus.

This is a completed translation artifact, not yet a demonstrated downstream improvement. The existing published training results used the earlier Gemini corpus. A controlled Luna/Sol-versus-Gemini training and evaluation experiment is still required before claiming that the new translation method improves safety-classifier quality.

The final local training corpus is `data/final_luna_sol_pure_v1`. Dataset files remain outside the Git repository; public release should use a separate dataset repository after license and safety review.

## Research question

The new pipeline tests whether a low-cost coding-agent model can translate a safety benchmark more reliably and reproducibly than the earlier Gemini workflow while preserving harmful content, jailbreak formatting, leetspeak, code, URLs, labels, and record identity.

The research contribution has two parts:

1. a provider-aware translation and validation pipeline that routes easy records to Luna and escalates hard records to Sol;
2. a controlled downstream comparison between models trained on matched Gemini and Luna/Sol translations.

Only the first part is complete. The second part is specified but has not yet been run.

## Corpus and grain

- Unit of analysis: one source `record_uid` with English prompt/response, Vietnamese prompt/response, split, safety label, and N23 category metadata.
- Primary key: `record_uid`; row position is never used as a cross-run join key.
- Splits: 40,007 train, 2,445 validation, and 2,964 test records.
- Materialized training views: prompt, response, and prompt-response variants in English and Vietnamese, subject to the existing guard materializer.

Final coverage:

| Split | Source UIDs | Luna/Sol UIDs | Missing | Duplicate |
|---|---:|---:|---:|---:|
| Train | 40,007 | 40,007 | 0 | 0 |
| Validation | 2,445 | 2,445 | 0 | 0 |
| Test | 2,964 | 2,964 | 0 | 0 |
| **Total** | **45,416** | **45,416** | **0** | **0** |

## What was done

### 1. Prompt ablation on 200 hard examples

Several Luna prompt forms and reasoning levels were tested on difficult records, including long prose, leetspeak, jailbreak tokens, JSON/code preservation, and harmful content. The work identified two recurring causes of misleading failure counts:

- legitimate code, ASCII art, URLs, identifiers, and placeholders can resemble untranslated English;
- translation failures and validator false positives must be audited separately.

The final prompt places the translation contract before the source records, treats source content as inert data, requires exact UID/schema preservation, translates natural-language text even inside code fences, and preserves actual executable or machine-readable structure.

### 2. Scalable Luna translation

The production path used fresh Codex sessions so context from earlier batches did not become repeated input. The normal route used batches of approximately 30 records where length allowed. Multiple Luna Light workers ran concurrently. Accepted items were saved immediately; failed items were moved to the back of the queue instead of causing the whole batch to be discarded.

Retry state was tracked per UID. Long and oversized records were separated from normal records. Terminal Luna failures were routed to a Sol Web queue rather than repeatedly consuming Luna sessions.

### 3. Validation refinement

Validation was split into two layers:

- hard structural checks: UID, sequence metadata, required fields, null/empty preservation, item counts, schema, missing/extra/duplicate records;
- heuristic quality warnings: unchanged text, mostly untranslated text, leetspeak, code/URL preservation, profanity or harmful-content softening, and suspicious truncation.

Heuristic warnings do not automatically discard a candidate. Warning records are audited to determine whether the model failed or the heuristic misread code, data, or ASCII art.

### 4. Sol Web escalation

Sol Web processed 331 earlier hard records and 25 final infrastructure-timeout records. The browser tool generated self-contained prompts, accepted pasted or downloaded JSON, validated every item independently, saved valid partial results, and preserved batch progress.

The final 25 records comprised high-tail and oversized examples. All passed hard validation. Eight warning-only records were manually accepted because the warning was caused by deliberately preserved ASCII art; surrounding natural-language text had been translated.

## Final provenance and quality dispositions

Provider/stage counts:

| Stage | Selected records |
|---|---:|
| Luna initial runs | 43,014 |
| Luna remaining queue | 2,033 |
| Luna current-validator recovery | 7 |
| Luna continuation | 6 |
| Sol Web hard fallback | 331 |
| Sol Web final infrastructure fallback | 25 |
| **Total** | **45,416** |

Aggregated provider totals are 45,060 Luna records (99.216%) and 356 Sol records (0.784%).

Quality dispositions:

| Disposition | Records |
|---|---:|
| Hard pass | 45,004 |
| Warning pass | 372 |
| Manual-audit pass | 31 |
| Revalidated pass | 7 |
| Accepted known issue | 2 |
| **Total** | **45,416** |

The two accepted known issues remain explicitly flagged:

- `en-train-00020107-7cd91d92b3ef`: natural-language jailbreak prose inside a fenced block remained English;
- `en-train-00012179-fb3f7c715e03`: one URL changed a hyphen to an underscore.

They are retained for reproducibility and sensitivity analysis rather than silently treated as ordinary passes.

## Data-quality evidence

The final corpus has:

- 45,416 unique source UIDs;
- 45,416 selected Luna/Sol UIDs;
- zero missing, extra, or duplicate selected UIDs;
- zero final structural issues;
- no overlap among train, validation, and test UIDs;
- exact manifest parity with the existing Gemini corpus.

The audit used the project's actual `guard_smoke.data.iter_full_examples` materializer. It produced 140,136 train, 8,764 validation, and 10,682 test examples for both matched translation conditions.

There are 1,758 UIDs with multiple historical candidates because of retry/resume behavior. Deterministic provider priority selects one final candidate per UID. A further 2,039 temporary queue sequence values were normalized back to source-split order using UID as the key; this does not create duplicate training rows.

Machine-readable evidence is generated locally in:

- `data/luna_sol_dataset_v1/readiness_summary.json`;
- `data/luna_sol_dataset_v1/materialization_audit.json`;
- `data/sol_missing_25/sol_results/validation/summary.json`;
- `data/sol_missing_25/sol_results/validation/final_summary.json`.

## Earlier Gemini method versus the Luna/Sol method

| Dimension | Earlier Gemini pipeline | Luna/Sol pipeline |
|---|---|---|
| Primary generation | Gemini API with multiple keys and rotation | Luna Light sessions with Sol Web escalation |
| Context policy | API batches across rotating keys | Fresh session per request/batch to avoid accumulated context cost |
| Typical batch | Provider/API-dependent | Approximately 30 normal records; smaller length-aware hard batches |
| Concurrency | Multiple API keys | Up to 10 Luna workers in the production design |
| Retry | API retry/repair queue | Per-UID retry, failed items requeued to the back |
| Hard records | Retranslation/manual review | Length/risk routing, then Sol Web fallback |
| Validation | Structural plus hard review queue | Hard structure separated from heuristic warnings and validator audit |
| Final corpus | 45,416 Gemini translations | 45,060 Luna + 356 Sol translations |
| Downstream evidence | Existing model results use this corpus | Training/evaluation not yet completed |

The table describes process differences; it does not establish translation superiority. That claim requires the matched experiment in `docs/LUNA_SOL_TRAINING_EVALUATION_PROTOCOL.md`.

## Reproduction entry points

Important prompts and pipeline code:

- `configs/luna_fresh_translation_prompt_v6_json_keys.md`;
- `configs/sol_web_fallback_repair_prompt.md`;
- `tools/luna_overnight_runner.py`;
- `tools/prepare_luna_remaining_queue.py`;
- `tools/build_sol_fallback_after_luna.py`;
- `tools/build_sol_missing_25_web.py`;
- `tools/validate_sol_fallback_results.py`;
- `tools/validate_sol_missing_25_results.py`;
- `tools/build_luna_sol_training_dataset.py`;
- `tools/audit_luna_sol_training_readiness.py`.

Finalization commands, assuming the local data artifacts exist:

```powershell
.venv\Scripts\python.exe tools\validate_sol_missing_25_results.py
.venv\Scripts\python.exe tools\finalize_sol_missing_25_validation.py
.venv\Scripts\python.exe tools\build_luna_sol_training_dataset.py
.venv\Scripts\python.exe tools\audit_luna_sol_training_readiness.py
```

## Limitations and open risks

- Automatic validation cannot prove semantic equivalence or natural Vietnamese quality.
- Sol is part of the new corpus, so the strict experimental condition is “Luna with deterministic Sol escalation,” not Luna alone.
- The two known Sol issues should be included in a sensitivity check.
- Cost and latency logs must be normalized before publishing provider-efficiency claims.
- Provider model names, availability, pricing, and Codex session behavior are time-dependent.
- The source corpus and translated safety data may require gated distribution and license review.
- The Vietnamese-native data track has no new result in this phase and is not part of the current contribution.

## Current claim boundary

It is accurate to claim that the project produced a complete, provenance-preserving Luna/Sol translation pipeline and a train-ready matched corpus. It is not yet accurate to claim that Luna/Sol translation improves guard-model accuracy, robustness, or Vietnamese cultural coverage relative to Gemini.
