# Translation prompting and baseline research

## Current machine-translation practice to compare against

The WMT25 General MT findings show that modern MT research is mixed rather than dominated by one
method. Most external submissions used generative LLMs, but encoder-decoder systems and hybrid
systems remained present. Common components included continued pretraining, supervised finetuning,
preference optimization, back-translation, quality estimation, Minimum Bayes Risk decoding, LLM
automatic post-editing and prompting. Eight submissions used prompting without model training.

WMT25's organizer LLM baseline used one unified zero-shot instruction-following script, temperature
zero, and full-document translation. When a model failed to translate the whole document while
preserving paragraph breaks, the collector fell back to paragraph-level translation. Collection code
was released by WMT. This is the most defensible famous/reproducible LLM baseline for this project.

Primary source: https://aclanthology.org/2025.wmt-1.22/

Recommended comparison ladder:

1. Existing Gemini 3.1 Flash Lite/manual pipeline (the project baseline).
2. GPT-5.6 Luna pipeline (the proposed method).
3. WMT-style zero-shot document translation at deterministic settings with paragraph/record fallback.
4. NLLB-200 3.3B as an open local encoder-decoder baseline. WMT25 includes it among organizer/open
   baselines; it provides a reproducible non-chat translation condition.
5. Microsoft Translator API or Google Cloud Translation API as a commercial MT baseline. This is
   preferable to automating consumer Bing/Google Translate pages because the API version, request,
   response and cost can be recorded. A UI batch can be an informal demo, not the primary scientific
   baseline.
6. Optional translation-specialist open LLM (Tower/EuroLLM) only after Vietnamese support and model
   license are verified.

Report proprietary/API systems as an unconstrained track and downloadable local models as a
constrained/reproducible track, mirroring the distinction made by WMT25.

## What translation-prompt research supports

### Lean, explicit constraints

OpenAI's GPT-5.6 guidance recommends lean prompts, stating each instruction once, and retaining
examples/style rules only when they encode a requirement or repair a measured gap. It recommends
specifying the goal, context, hard constraints, success criteria and output format, then measuring
quality, completeness, tokens, latency and cost on representative tasks.

Source: https://developers.openai.com/api/docs/guides/latest-model

Implication: the detailed P2 prompt is a valid baseline, but repeated warnings should be removed. P3
tests a lean contract plus final checklist rather than simply adding more text.

### Few-shot examples

Prompting studies consistently find that example quality matters. Vilar et al. found example quality
to be the most important selection factor for PaLM translation. Zhang, Haddow and Birch found that
the number and quality of examples matter and that suboptimal examples can degrade translation.

Sources:

- https://aclanthology.org/2023.acl-long.859/
- https://arxiv.org/abs/2301.07069

Implication: do not add random demonstrations. Use a small content-shape library with human-approved
examples for leetspeak, JSON-key preservation, executable code and truncation/completeness. Retrieval
must be frozen for reproducibility.

### Multi-aspect prompting and selection

MAPS first induces translation-related knowledge (keywords, topic and relevant demonstrations), then
uses quality estimation to select helpful information/candidates. Its paper reports reductions in
hallucination, ambiguity, mistranslation, awkward style, untranslated text and omission.

Source: https://arxiv.org/abs/2305.04118

Implication: MAPS is a useful quality-first ablation on hard records, but it requires multiple model
calls and selection. It should not be silently folded into the main low-cost Luna baseline.

### Instruction position

“Instruction Position Matters in Sequence Generation with Large Language Models” reports that moving
the task instruction after the input can reduce instruction forgetting on long sequence-generation
tasks and improved zero-shot translation in the models studied.

Source: https://arxiv.org/abs/2308.12097

Implication: an instruction-after-input condition is worth testing on long records. An instruction
sandwich changes both position and repetition, so a later ablation must isolate these factors.

### Markup and structured text

Research on translation with markup shows that tag transfer is a separate challenge from translation
quality. LLMs can preserve markup, but domain-specific NMT may still translate better. XML/Markdown
are therefore useful delimiters, not evidence by themselves of better translation.

Source: https://aclanthology.org/2023.mtsummit-research.13/

Implication: no primary evidence found supports a universal claim that Markdown is better than XML or
JSON for translation. Format must be treated as an empirical factor. JSON schema improves parsing;
XML tags can separate inert data from task instructions; neither guarantees semantic completeness.

### Document fallback

WMT25 explicitly observed that current LLMs may fail to translate a whole document or preserve
paragraph breaks. Its collector translates the document first, then falls back to paragraph-level
translation after failure.

Source: https://aclanthology.org/2025.wmt-1.22/

Implication: the project should implement batch → item → field/paragraph fallback. Repeating the same
oversized batch is not the recommended recovery method.

## P3 quick experiment result

Prompt: `configs/luna_fresh_translation_prompt_v3_sandwich.md`

Regression records: the six manually confirmed P2 failures (leet style, two truncation/completeness
cases, two JSON-key cases and one profanity-wordplay case).

Configuration: one Luna ephemeral turn, one six-item batch, under 50k source characters, low reasoning.

Result:

- elapsed: 758.685 seconds before controlled termination;
- final response: missing;
- saved items: 0/6;
- usage telemetry: absent;
- event log: thread started and turn started only;
- all descendant Codex processes were explicitly terminated and verified stopped.

Conclusion: P3 in a mixed six-error batch fails the operational completion criterion. This does not
isolate prompt format because the batch contains two long/repetitive records already known to cause
output instability. The next valid experiment must stratify by content shape.

## Next prompt ablation (pre-registered recommendation)

Use three mini-slices, each with failures plus matched controls that P2 translated correctly:

1. Leetspeak: two known failures/partial failures + four clean leet controls.
2. Structured JSON: two key-change failures + four clean JSON controls.
3. Completeness: two truncation failures + four long clean controls, one record per turn.

Compare:

- P2 detailed instruction-first baseline;
- P3 lean XML sandwich;
- P4 P3 plus one human-approved content-shape demonstration;
- P5 lean instruction-after-input without duplicated opening instruction.

Freeze the 18 records before generating new outputs. Use the same model, reasoning effort and output
schema. For each condition report completion, blocking validator errors, manually confirmed errors,
latency, input/output tokens and credit. The primary endpoint is number of manually confirmed errors,
not raw validator flags.

Only after selecting a prompt on this development regression set should it be run once on a separate
holdout with the same content-shape proportions.
