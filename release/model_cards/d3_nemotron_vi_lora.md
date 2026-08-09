---
base_model: nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3
library_name: peft
license: other
license_name: nvidia-open-model-license-with-llama3.1-terms
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
tags: [lora, safety, vietnamese, content-moderation]
---

# D3 Nemotron Vietnamese P/PR LoRA

Adapter-only release candidate for the project's final D3 run. Use and
redistribution are governed by the NVIDIA Open Model License and the additional
Llama 3.1 Community License terms identified by the base model card. This card
does not relicense the adapter solely under Apache-2.0.

## Training contract

- Final clean contract: P and PR only; response-only R excluded.
- Base revision: `8fdc246ba3d56db9c469d534233b9f582d3afafa`.
- 50,637 Vietnamese rows; 1,583/1,583 optimizer steps.
- One epoch; context 2,048; BF16; no quantization.
- Microbatch 2, effective batch 32, gradient accumulation 16.
- Learning rate `1e-5`.
- LoRA rank 8, alpha 32, dropout 0.05; targets `q_proj`, `v_proj`.
- Training data: Gemini-derived Vietnamese condition, not the new Luna/Sol
  condition.

## Matched local result

D3 completed 11,736/11,736 P/PR evaluations. Binary accuracy was 86.52%, safe
recall 92.12%, and unsafe recall 81.26%. Its N23 behavior was view-dependent:
prompt-only regressed while prompt-plus-response improved. See
`reports/research_archive/D3_NEMOTRON_NO_R_FINAL_ANALYSIS_20260724.md` for the
full matched analysis and limitations.

## Loading

Use the PEFT loading pattern from the base model's supported Transformers
version. Users must obtain the NVIDIA base model separately and accept all
applicable upstream terms.

This repository contains only the PEFT adapter and public adapter config; base weights and tokenizer files are not redistributed. Users must accept the upstream NVIDIA and Llama terms before loading the base model.

Adapter SHA-256: 38a328fffe43b7b455038f9a9a2f75288bf4b145203bc9e4d8f00fde226f1bac.
