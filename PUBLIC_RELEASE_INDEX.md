# Public release index

This commit adds the reproducible Luna/Sol translation contribution and the
release metadata needed to keep it separate from the earlier trained models.

## Read in this order

1. `README.md` — complete project motivation and reported Gemini-era results.
2. `docs/TRAINING_EVALUATION_REPRODUCTION.md` — exact Python/Vast entrypoints
   and the critical legacy full-R versus final P/PR-only boundary.
3. `docs/LUNA_SOL_TRANSLATION_RESEARCH_HANDOFF.md` — Luna/Sol methods, prompts,
   routing, validators, and final 45,416-record audit.
4. `docs/LUNA_SOL_TRAINING_EVALUATION_PROTOCOL.md` — frozen future comparison
   against the Gemini translation and SEA external test.
5. `docs/DATA_MODEL_RELEASE_AND_LICENSE.md` — what can be published, under
   which terms, and what must remain upstream/private.

## Release candidates

- Dataset card:
  `release/datasets/nemotron_safety_guard_vi_luna_sol/README.md`
- Q2 Qwen LoRA card: `release/model_cards/q2_qwen3guard_vi_en_lora.md`
- D3 Nemotron LoRA card: `release/model_cards/d3_nemotron_vi_lora.md`
- Machine-readable summary: `release/RELEASE_MANIFEST.json`

The Luna/Sol corpus has not trained a downstream model yet. Q2 and D3 were
trained on the earlier Gemini-derived P/PR condition. Their cards are included
to preserve the already completed model evidence without implying that they
validate the new translation.

## Publication boundaries

GitHub contains code, prompts, validators, tests, contracts, aggregate evidence,
and cards. Raw safety text, translated records, provider sessions, credentials,
base weights, adapters, prediction dumps, and runtime logs remain excluded.
Adapters are packaged locally under ignored `exports/hf_release/` for later
Hugging Face upload after authentication and license review.
