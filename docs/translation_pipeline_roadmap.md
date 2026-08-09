# Reproducible translation pipeline roadmap

## Research objective

Build a reproducible, provider-neutral safety-benchmark translation pipeline, then measure both
translation quality and downstream model utility against established machine-translation and LLM
baselines.

## Reproducibility requirements

- A user already logged into Codex can select the Codex CLI backend without supplying an API key.
- API users can select OpenAI or Gemini; optional auto-detection probes only documented model-list
  endpoints and never persists a raw key by default.
- Local backends support OpenAI-compatible servers plus explicit adapters for Ollama/llama.cpp when
  their discovery endpoints are available.
- Source and target languages are configurable with BCP-47-style language identifiers; English to
  Vietnamese remains the benchmark default, not a hard-coded assumption.
- Prompts are visible, editable, versioned, hashed, exportable and restorable.
- Provider, exact model slug, endpoint type, prompt hash, generation settings, retry policy,
  validator version, package versions, input hashes and random seed are written to a run manifest.
- Keys are read from environment variables, an OS keyring or one-process input; they are redacted
  from logs and artifacts.
- Provider/model auto-detection always has a manual override. Prefix guessing alone is not treated
  as proof because gateways and OpenAI-compatible local servers use arbitrary key formats.

## Provider discovery design

1. Codex backend: verify `codex login status`, then list/use configured Codex model slugs.
2. OpenAI backend: call the documented model-list endpoint and filter for models supporting the
   requested response mode.
3. Gemini backend: call the documented models-list endpoint and filter by generation capability.
4. OpenAI-compatible local backend: probe `/v1/models` at an explicitly allowed base URL.
5. Ollama backend: probe its documented local tags endpoint.
6. If more than one probe succeeds, show the candidates and require a selection rather than choosing
   silently.

## Baselines to reproduce

- Existing Gemini 3.1 Flash Lite/manual pipeline.
- GPT-5.6 Luna pipeline.
- WMT-style zero-shot document translation at deterministic settings, with paragraph fallback only
  after a document-level failure.
- NLLB-200 3.3B as an open, local encoder-decoder baseline when hardware permits.
- At least one commercial MT API baseline, preferably Microsoft Translator or Google Cloud
  Translation. Browser automation of consumer Bing/Google Translate is not the reproducible default.
- Optional local translation-specialist LLM such as Tower/EuroLLM when Vietnamese support and license
  are verified.
- Optional few-shot and MAPS-inspired variants as prompt-method ablations, not silently mixed into the
  main Luna condition.

## Prompt-method ablations

- P2: current detailed instruction-first prompt.
- P3: lean XML-delimited instruction sandwich with a final mandatory checklist.
- P4: P3 plus two high-quality, content-shape-matched demonstrations (leet and structured JSON).
- P5: instruction-after-input only, to isolate instruction position from delimiters/checklist.
- P6: MAPS-inspired multi-aspect analysis/candidate selection on hard slices only.

Change one factor at a time wherever possible. Freeze the error regression set before looking at new
outputs; retain a separate holdout so prompt tuning does not overfit the six known failures.

## Evaluation roadmap

- Translation: structural validity, completeness, terminology, preserved keys/tokens, leetspeak/style,
  blind pairwise human/judge preference, chrF/COMET-family metrics where justified, cost and latency.
- Downstream: Gemini-trained versus Luna-trained models in a 2×2 translated-test matrix, at least three
  seeds where feasible, plus untouched SEA-HELM/SEA Bench and original-English evaluation.
- Report safe/unsafe, hazard category, length and content-shape slices separately.
- Report proprietary/API results as an unconstrained track and local/open models as a reproducible
  constrained track.
