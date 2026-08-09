# Luna/Sol versus Gemini: training and evaluation protocol

Status: specified, not yet executed.

## Decision this experiment supports

The experiment tests whether the Luna/Sol translation pipeline changes downstream safety-classifier quality relative to the earlier Gemini translation while keeping source records, model training, and evaluation contracts fixed.

## Frozen data conditions

Two translation conditions must contain the same 45,416 source UIDs:

- `G`: original Gemini Vietnamese translations;
- `L`: final Luna/Sol Vietnamese translations.

Both conditions use 40,007 train, 2,445 validation, and 2,964 test records. English source fields, safety labels, N23 labels, split membership, and materialized view policy must be identical.

The local prepared paths are:

- Luna/Sol: `data/final_luna_sol_pure_v1`;
- Gemini matched control: `data/final_gemini_paired_v1`.

Before training, record SHA-256 checksums and the readiness summaries for all six JSONL files.

## Primary training comparison

Train the same guard architecture twice:

| Run | Vietnamese training text | All other settings |
|---|---|---|
| `TRAIN-G` | Gemini | Frozen |
| `TRAIN-L` | Luna/Sol | Frozen |

Freeze:

- base checkpoint and revision;
- tokenizer and maximum context;
- prompt/response/combined view materialization;
- EN/VI sampling ratio;
- random seed and shuffle policy;
- optimizer, scheduler, learning rate, warmup, weight decay;
- batch size, gradient accumulation, precision, and clipping;
- LoRA modules/rank/alpha/dropout when applicable;
- number of optimizer updates or observed training tokens;
- checkpoint selection rule;
- threshold selection and calibration procedure;
- evaluation implementation and metric definitions.

Use at least three seeds if compute permits. A single-seed result is exploratory and must be labeled as such.

## Evaluation matrix

Evaluate both trained models on four conceptually different test conditions:

| Test condition | Purpose |
|---|---|
| Nemotron English source test | Detect whether Vietnamese training changes source-language performance |
| Nemotron Gemini Vietnamese test | In-translation-domain evaluation for `TRAIN-G` |
| Nemotron Luna/Sol Vietnamese test | In-translation-domain evaluation for `TRAIN-L` |
| SEA Bench / SEA-HELM paired EN–VI | External-domain generalization |

The core 2×2 translated-test matrix is:

| Model | Gemini test | Luna/Sol test |
|---|---:|---:|
| `TRAIN-G` | required | required |
| `TRAIN-L` | required | required |

This matrix distinguishes a real training improvement from test-style affinity. For example, if each model wins only on the test translated by its own provider, the result is translation-style matching rather than general superiority.

## Metrics

Primary binary metrics:

- accuracy;
- macro-F1;
- safe recall and unsafe recall;
- AUROC and AUPRC when scores are available;
- expected calibration error or Brier score when probabilities are comparable.

Structured-label metrics:

- N23 micro-F1 and macro-F1;
- per-category precision, recall, and F1;
- label cardinality and positive-prediction rate;
- exact-match or subset accuracy only as a secondary strict metric.

Paired uncertainty:

- McNemar test for paired binary decisions;
- paired bootstrap confidence intervals for metric differences;
- seed-level mean, standard deviation, and individual results;
- multiple-comparison correction for large per-category claim sets.

## Required slices

Report each primary metric by:

- language: English versus Vietnamese;
- dataset: Nemotron versus SEA;
- view: prompt, response, prompt-response;
- source label: safe versus unsafe;
- length bucket: normal, near-tail, tail, high-tail, oversized;
- translation provider in the Luna/Sol condition: Luna versus Sol fallback;
- warning status: hard pass, warning/manual pass, known issue;
- leetspeak/obfuscation signal;
- code, JSON, URL, placeholder, and ASCII-art signal;
- N23 harm category and frequent category combinations.

The Sol and oversized slices are small; report counts and confidence intervals rather than ranking models on unstable percentages.

## Translation-quality analysis

Because there is no human Vietnamese reference for every record, automatic MT scores alone are insufficient. Use a layered analysis:

1. deterministic preservation checks for UID, null/empty fields, code, URLs, JSON keys, placeholders, and delimiters;
2. language and truncation heuristics, reported as warnings rather than automatic truth;
3. a stratified human audit sampled by length, safe/unsafe label, category, leetspeak, provider, and warning state;
4. blinded pairwise preference between Gemini and Luna/Sol translations;
5. downstream model utility as a separate outcome, not a substitute for translation-quality review.

If an LLM judge is used, calibrate it against manually reviewed passes and failures, report judge disagreement, and do not treat judge output as ground truth.

## Baseline translation methods

The first defensible comparison is the existing Gemini corpus because it is aligned by UID and already used by the project. Additional baselines may be added later:

- a conventional machine-translation service such as Bing/Microsoft Translator;
- a reproducible open model with documented Vietnamese support;
- a simple zero-shot LLM translation prompt without routing or validation repair.

Do not add a baseline merely to make the new method look stronger. Record model/version, date, decoding settings, batch policy, cost, latency, failures, and license constraints.

## Robustness checks

- Repeat the headline comparison after excluding the two accepted known Sol issues.
- Repeat after excluding all Sol-routed records to estimate the Luna-only subset effect.
- Evaluate warning/manual-audit records separately.
- Check whether conclusions change after removing oversized ASCII-dominated examples.
- Compare fixed-threshold and validation-calibrated results.
- Verify that no translated test record or semantic duplicate enters training.

## Training readiness gate

Training may start only when:

- all expected UIDs are present once per split;
- train/validation/test intersections are empty;
- both translation conditions materialize to identical example counts and labels;
- checksums and environment/config snapshots are saved;
- output directories and run IDs clearly identify translation condition and seed;
- no API key, source data, private log, or model cache is included in the Git commit.

The current Luna/Sol data passes the UID, structure, split-overlap, and materialization gates. GPU training has not been launched for this comparison.

## Interpretation rules

- A difference on translated Nemotron alone is not evidence of Vietnamese cultural coverage.
- SEA Bench improvement is evidence of external-domain transfer, not proof of native-Vietnamese adequacy.
- A single-seed difference without paired uncertainty is exploratory.
- Provider-style affinity must be separated from training benefit using the 2×2 translated-test matrix.
- Native Vietnamese data generation remains a separate future track and currently has no new result.
