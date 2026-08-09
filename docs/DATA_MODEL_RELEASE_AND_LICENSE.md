# Data and model release policy

This document is a practical publication assessment, not legal advice. It
separates original project code, the translated dataset, upstream base models,
and project adapter weights because they do not share one automatic license.

## Recommended release layout

| Artifact | Location | Proposed terms | Current decision |
|---|---|---|---|
| Original code/config/docs | GitHub | Apache-2.0 | Public in this repository |
| Aggregate reports/checksums | GitHub | Apache-2.0 documentation, source notices retained | Public |
| Vietnamese translated records | Separate Hugging Face dataset | CC BY 4.0 with NVIDIA attribution and change notice | Prepare card; publish gated first |
| Q2 Qwen adapter | Separate Hugging Face model | Apache-2.0, subject to Qwen base terms and dataset notices | Adapter-only candidate |
| D3 Nemotron adapter | Separate Hugging Face model | NVIDIA Open Model License plus applicable Llama 3.1 terms | Adapter-only candidate |
| Base weights/tokenizers | Upstream repositories | Upstream terms | Never re-upload from this project |
| Raw predictions/logs/review payloads | Controlled storage | N/A | Private by default |

## Vietnamese dataset conclusion

The source dataset card for
[`nvidia/Nemotron-Safety-Guard-Dataset-v3`](https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3)
declares CC BY 4.0. Creative Commons states that CC BY 4.0 permits sharing and
adaptation, including commercial use, provided appropriate credit, a license
link, and an indication of changes are supplied, without implying endorsement.

Translation is an adaptation. The Vietnamese release can therefore be made as
a CC BY 4.0 derivative if it:

1. credits NVIDIA and links the exact upstream dataset and CC BY 4.0;
2. states that English text was machine-translated by the Luna/Sol pipeline;
3. identifies the project author and date of the adaptation;
4. preserves upstream notices and does not claim NVIDIA endorsement;
5. does not add access terms that contradict CC BY 4.0.

The release should be described as a **Vietnamese machine-translated derivative
or 13th-language extension candidate**, not as an official NVIDIA language and
not as equivalent to CultureGuard cultural adaptation. The current contribution
has full UID coverage and audited translation structure, but no Luna/Sol-trained
downstream model result yet.

Gating is recommended initially because the corpus contains harmful, sexual,
violent, self-harm, jailbreak, and illegal-activity text. Gating is a safety and
access workflow; it must not impose legal restrictions that conflict with the
CC BY 4.0 permissions.

## Adapter conclusion

Only adapter deltas produced by this project should be uploaded. Do not include
downloaded base model shards or unchanged tokenizer files.

- Q2 derives from `Qwen/Qwen3Guard-Gen-4B`, whose model card declares
  Apache-2.0. An adapter-only repository can use Apache-2.0 while retaining the
  base-model citation and stating that users must obtain the base model
  separately.
- D3 derives from `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3`. Its model card
  specifies the NVIDIA Open Model License and additional Llama 3.1 Community
  License terms. The D3 repository must carry those terms and cannot be
  presented as governed only by this GitHub repository's Apache-2.0 license.
- Encoder `final/heads.pt` files are classifier heads, not LoRA adapters. They
  should get separate model cards and loading code before publication. Smoke
  checkpoints are not release candidates.

Training on a CC BY dataset does not justify silently discarding dataset
attribution in a model card. The adapter cards should cite both the base model
and source dataset even where the legal status of learned weights is
jurisdiction-dependent.

## Required dataset release files

- `README.md` dataset card;
- `LICENSE` or explicit `license: cc-by-4.0` metadata;
- `ATTRIBUTION.md` with upstream URL, citation, changes, and adaptation author;
- `manifest.json` with split counts, schema version, and SHA-256 hashes;
- JSONL/Parquet records keyed by original UID;
- quality report and known-issue summary;
- content warning and intended-use/limitations section.

## Required adapter release files

- `adapter_model.safetensors`;
- `adapter_config.json` with a public `base_model_name_or_path` rather than the
  original local `/workspace/...` path;
- model card with final P/PR-only training contract and matched evaluation;
- license/terms and upstream citations;
- hashes and minimal PEFT loading example.

The local Hugging Face CLI is installed but currently has no authenticated
account. The cards and manifests can be committed now; creating public Hub
repositories and uploading weights/data must wait for `hf auth login`.

## Sources checked

- NVIDIA dataset card: <https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3>
- CC BY 4.0 deed: <https://creativecommons.org/licenses/by/4.0/>
- Qwen3Guard model card: <https://huggingface.co/Qwen/Qwen3Guard-Gen-4B>
- NVIDIA Nemotron model card: <https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3>
- mmBERT-small model card: <https://huggingface.co/jhu-clsp/mmBERT-small>
- Fastino GLiGuard model card: <https://huggingface.co/fastino/gliguard-LLMGuardrails-300M>
